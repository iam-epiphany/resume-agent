from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

from backend.app.core.config import (
    ANSWER_GENERATION_API_KEY,
    ANSWER_GENERATION_BASE_URL,
    ANSWER_GENERATION_ENABLED,
    ANSWER_GENERATION_INCLUDE_THINKING,
    ANSWER_GENERATION_MAX_TOKENS,
    ANSWER_GENERATION_MODEL,
    ANSWER_GENERATION_PROVIDER,
    ANSWER_GENERATION_RESPONSE_FORMAT,
    ANSWER_GENERATION_STREAM,
    ANSWER_GENERATION_TIMEOUT_SECONDS,
    GROUNDING_VERIFY_ENABLED,
    HEDGE_PREFIX,
)
from backend.app.schemas.qa import RetrievalResult
from backend.app.services.prompt_builder import RAGPromptBuilder
from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
    open_chat_completion,
)
from backend.app.services.performance_metrics import measure
from backend.app.services.intent_router_service import INTENT_GREETING, INTENT_OFF_TOPIC
from backend.app.services.grounding_verification_service import verify_hard_facts

CancellationChecker = Callable[[], None]
PreviewReporter = Callable[[str], None]

AnswerMode = Literal["answered", "hedged", "redirected", "failed"]
EvidenceSufficiency = Literal["sufficient", "partial", "insufficient"]


@dataclass
class GeneratedAnswer:
    answer: str | None
    answer_mode: AnswerMode = "answered"
    evidence_sufficiency: EvidenceSufficiency | None = None
    hedge_note: str | None = None
    generation_status: str = "completed"
    degraded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


POLITE_REDIRECT_GREETING = (
    "你好！我是张三的简历问答助手，可以围绕他的教育背景、项目经历、专业技能、"
    "荣誉奖项和求职意向提问，比如「介绍一下你的项目经历」或「你的技术栈是什么」？"
)

POLITE_REDIRECT_OFF_TOPIC = (
    "抱歉，这个问题不在我的简历知识范围内。我是基于张三的简历和项目文档回答问题的助手，"
    "建议换个关于他个人经历的问题，比如他的教育背景、项目细节或求职意向，我很乐意详细解答。"
)

FALLBACK_NO_CONTEXT = (
    "抱歉，这个问题在我现有的简历知识库中还没有相关记录，暂时无法给出有价值的回答。"
    "你可以换个角度问我，比如教育背景、项目经历、专业技能、荣誉奖项或求职意向方面的问题。"
)


def generate_answer(
    question: str,
    context_chunks: list[RetrievalResult],
    *,
    intent: str,
    llm_prompt: str | None = None,
    cancellation_checker: CancellationChecker | None = None,
    preview_reporter: PreviewReporter | None = None,
    no_evidence: bool = False,
    known_entities: list[str] | None = None,
) -> GeneratedAnswer:
    """单次 LLM 生成 + 答案自评 + 置信度分级。

    - greeting / off_topic → 礼貌转移（零 LLM）
    - 其余 → 单次 LLM 调用输出 {answer, evidence_sufficiency, reason}
    - 自评 partial/insufficient → hedged 模式，后端强制"根据现有知识库推测"前缀（唯一一次）
    - no_evidence（兜底链第 3 级）：无检索证据，限制为 persona 软信息，硬事实必须说明未收录
    - LLM 异常 → 摘录兜底（hedged）；无上下文且 LLM 异常 → 礼貌兜底文案
    - known_entities：知识库已知的简历领域实体（学校/项目/奖项/证书/技术栈等），
      用于 grounding 校验——答案中出现但证据缺失的已知实体被标记为未核实
    """
    if intent == INTENT_GREETING:
        return _redirected(POLITE_REDIRECT_GREETING)
    if intent == INTENT_OFF_TOPIC:
        return _redirected(POLITE_REDIRECT_OFF_TOPIC)

    generated: GeneratedAnswer | None = None
    if ANSWER_GENERATION_ENABLED and ANSWER_GENERATION_API_KEY:
        try:
            generated = _call_llm(
                question,
                context_chunks,
                llm_prompt=llm_prompt,
                cancellation_checker=cancellation_checker,
                preview_reporter=preview_reporter,
                no_evidence=no_evidence,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("LLM 回答生成失败，降级: %s: %s", type(exc).__name__, exc)
            generated = None
    if generated is not None and (generated.answer or "").strip():
        return _apply_confidence_mode(generated, context_chunks, known_entities=known_entities)

    fallback = _extractive_fallback(context_chunks)
    if fallback is not None and (fallback.answer or "").strip():
        return fallback
    return _redirected(FALLBACK_NO_CONTEXT, mode="failed")


def _redirected(text: str, mode: AnswerMode = "redirected") -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=text,
        answer_mode=mode,
        evidence_sufficiency=None,
        generation_status="skipped",
    )


def _apply_confidence_mode(
    generated: GeneratedAnswer,
    context_chunks: list[RetrievalResult],
    *,
    known_entities: list[str] | None = None,
) -> GeneratedAnswer:
    """按自评结果分级；partial/insufficient 时由系统统一加推测前缀（且只加一次）。

    旧实现同时让 LLM 自写前缀 + 后端 prepend，出现过"根据现有知识库推测"双前缀
    （LLM 中途插入 + 后端开头追加）。现在前缀完全由系统管理：先剥掉 LLM 在
    开头自写的（含旧 prompt 历史行为的残留），再精确 prepend 一次。

    Grounding 硬事实校验：LLM 自评 sufficient 时，仍用确定性校验器核对答案中的
    数字/日期/书名号专名是否在检索证据中——缺失则降级 hedged（防"自己生成、
    自己判断"的高估盲区）。校验结果写入 extra 供 qa_logs/调试观测。
    """
    sufficiency = generated.evidence_sufficiency or "partial"
    generated.evidence_sufficiency = (
        sufficiency if sufficiency in {"sufficient", "partial", "insufficient"} else "partial"
    )
    if generated.evidence_sufficiency == "sufficient" and GROUNDING_VERIFY_ENABLED:
        evidence_texts = [
            part
            for chunk in context_chunks
            for part in (chunk.text, chunk.section_title or "")
            if part
        ]
        grounding = verify_hard_facts(generated.answer, evidence_texts, known_entities=known_entities)
        generated.extra["grounding_verification"] = grounding.to_dict()
        if not grounding.verified:
            generated.answer_mode = "hedged"
            generated.evidence_sufficiency = "partial"
            missing_parts = []
            if grounding.missing_numbers:
                missing_parts.append(f"数字 {len(grounding.missing_numbers)}")
            if grounding.missing_dates:
                missing_parts.append(f"日期/年份 {len(grounding.missing_dates)}")
            if grounding.missing_names:
                missing_parts.append(f"专名 {len(grounding.missing_names)}")
            if grounding.missing_entities:
                missing_parts.append(f"实体 {len(grounding.missing_entities)}")
            generated.hedge_note = (
                f"硬事实未在检索证据中核实：{'、'.join(missing_parts) or '无'}"
            )
            answer = (generated.answer or "").strip()
            while answer.startswith(HEDGE_PREFIX):
                answer = answer[len(HEDGE_PREFIX):].lstrip("，,。;； ")
            answer = answer.strip()
            generated.answer = f"{HEDGE_PREFIX}，{answer}" if answer else HEDGE_PREFIX
            return generated
    if generated.evidence_sufficiency == "sufficient":
        generated.answer_mode = "answered"
        return generated
    generated.answer_mode = "hedged"
    answer = (generated.answer or "").strip()
    while answer.startswith(HEDGE_PREFIX):
        answer = answer[len(HEDGE_PREFIX):].lstrip("，,。;； ")
    answer = answer.strip()
    generated.answer = f"{HEDGE_PREFIX}，{answer}" if answer else HEDGE_PREFIX
    return generated


def _call_llm(
    question: str,
    context_chunks: list[RetrievalResult],
    *,
    llm_prompt: str | None = None,
    cancellation_checker: CancellationChecker | None = None,
    preview_reporter: PreviewReporter | None = None,
    no_evidence: bool = False,
) -> GeneratedAnswer:
    messages = RAGPromptBuilder().build_generation_messages(
        question,
        context_chunks,
        llm_prompt=llm_prompt,
        no_evidence=no_evidence,
    )
    config = ChatCompletionConfig(
        provider=ANSWER_GENERATION_PROVIDER,
        api_key=ANSWER_GENERATION_API_KEY,
        base_url=ANSWER_GENERATION_BASE_URL,
        model=ANSWER_GENERATION_MODEL,
        timeout_seconds=ANSWER_GENERATION_TIMEOUT_SECONDS,
        include_thinking=ANSWER_GENERATION_INCLUDE_THINKING,
        response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
    )
    stream_attempts = [ANSWER_GENERATION_STREAM]
    if ANSWER_GENERATION_STREAM:
        stream_attempts.append(False)
    expanded_max_tokens = max(ANSWER_GENERATION_MAX_TOKENS * 2, 1800)
    if expanded_max_tokens > ANSWER_GENERATION_MAX_TOKENS:
        stream_attempts.append(False)
    expanded_config = ChatCompletionConfig(
        provider=ANSWER_GENERATION_PROVIDER,
        api_key=ANSWER_GENERATION_API_KEY,
        base_url=ANSWER_GENERATION_BASE_URL,
        model=ANSWER_GENERATION_MODEL,
        timeout_seconds=max(ANSWER_GENERATION_TIMEOUT_SECONDS * 2, 45.0),
        include_thinking=ANSWER_GENERATION_INCLUDE_THINKING,
        response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
    )
    last_error: Exception | None = None
    last_recovered: GeneratedAnswer | None = None
    for index, stream in enumerate(stream_attempts):
        content = ""
        attempt_config = (
            expanded_config
            if index == len(stream_attempts) - 1 and expanded_max_tokens > ANSWER_GENERATION_MAX_TOKENS
            else config
        )
        attempt_max_tokens = (
            expanded_max_tokens if attempt_config is expanded_config else ANSWER_GENERATION_MAX_TOKENS
        )
        try:
            content = _request_llm_content(
                attempt_config,
                messages,
                stream=stream,
                max_tokens=attempt_max_tokens,
                cancellation_checker=cancellation_checker,
                preview_reporter=preview_reporter,
            )
            return _parse_generated_content(content)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            recovered = _recover_generated_from_partial_content(content)
            if recovered is not None:
                last_recovered = recovered
                if index == len(stream_attempts) - 1:
                    return recovered
            last_error = exc
            continue
        except (OSError, TimeoutError, urllib.error.URLError, ChatCompletionError) as exc:
            last_error = OSError("answer generation unavailable")
            if stream and len(stream_attempts) > 1:
                continue
            raise OSError("answer generation unavailable") from exc
    if last_recovered is not None:
        return last_recovered
    if last_error is not None:
        raise last_error
    raise OSError("answer generation unavailable")


def _request_llm_content(
    config: ChatCompletionConfig,
    messages: list[dict[str, Any]],
    *,
    stream: bool,
    max_tokens: int,
    cancellation_checker: CancellationChecker | None,
    preview_reporter: PreviewReporter | None,
) -> str:
    with measure("answer_generation.external_api"):
        with open_chat_completion(
            config,
            messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
            stream=stream,
            opener=urllib.request.urlopen,
        ) as response:
            return _read_streaming_llm_content(
                response,
                cancellation_checker=cancellation_checker,
                preview_reporter=preview_reporter,
            )


def _parse_generated_content(content: str) -> GeneratedAnswer:
    """解析 {answer, evidence_sufficiency, reason} JSON。"""
    parsed = json.loads(_extract_json(content))
    answer = str(parsed.get("answer") or "").strip() or None
    sufficiency_raw = str(parsed.get("evidence_sufficiency") or "").strip().lower()
    sufficiency: EvidenceSufficiency | None = (
        sufficiency_raw if sufficiency_raw in {"sufficient", "partial", "insufficient"} else None
    )
    return GeneratedAnswer(
        answer=answer,
        evidence_sufficiency=sufficiency,
        hedge_note=str(parsed.get("reason") or "").strip()[:300] or None,
        generation_status="completed",
    )


def _recover_generated_from_partial_content(content: str) -> GeneratedAnswer | None:
    """流式内容被截断时，尽力恢复 answer 文本（自评按 partial 保守处理）。"""
    if not content:
        return None
    answer = _extract_answer_from_partial(content)
    if not answer:
        return None
    return GeneratedAnswer(
        answer=answer,
        evidence_sufficiency="partial",
        hedge_note="流式内容截断，按推测处理",
        generation_status="completed",
        degraded=True,
    )


def _extract_answer_from_partial(content: str) -> str | None:
    match = re.search(r'"answer"\s*:\s*"', content)
    if not match:
        return None
    start = match.end()
    # 解析 JSON 字符串（处理转义），到未转义的结束引号为止
    out: list[str] = []
    index = start
    while index < len(content):
        char = content[index]
        if char == "\\":
            if index + 1 < len(content):
                out.append(content[index : index + 2])
                index += 2
                continue
            break
        if char == '"':
            break
        out.append(char)
        index += 1
    return "".join(out).strip() or None


class _AnswerStreamExtractor:
    """流式 JSON 中提取 answer 字段值：预览只透出 answer 内容，隐藏 JSON 外壳。

    LLM 输出格式为 {"answer": "...", "evidence_sufficiency": "...", ...}，
    流式片段可能在任何位置截断；本类维护状态机，跨片段识别 "answer" 键
    并在字符串值内累积内容（处理转义），字符串结束即停止。
    """

    _KEY = '"answer"'

    def __init__(self) -> None:
        self._pending = ""
        self._started = False
        self._ended = False
        self._answer = ""
        self._escape = False

    def feed(self, piece: str) -> str:
        """输入新片段，返回本次新增的 answer 可见文本（已解码常见转义）。"""
        if self._ended or not piece:
            return ""
        if not self._started:
            self._pending += piece
            if not self._try_start():
                return ""
            piece = self._pending
            self._pending = ""
        out: list[str] = []
        index = 0
        while index < len(piece):
            char = piece[index]
            if self._escape:
                decoded = _decode_json_escape(char)
                out.append(decoded if decoded is not None else char)
                self._escape = False
                index += 1
                continue
            if char == "\\":
                self._escape = True
                index += 1
                continue
            if char == '"':
                self._ended = True
                break
            out.append(char)
            index += 1
        if out:
            self._answer += "".join(out)
        return "".join(out)

    def _try_start(self) -> bool:
        index = self._pending.find(self._KEY)
        while index != -1:
            after = self._pending[index + len(self._KEY):]
            match = re.match(r'\s*:\s*"', after)
            if match:
                self._started = True
                self._pending = self._pending[index + len(self._KEY) + match.end():]
                return True
            index = self._pending.find(self._KEY, index + 1)
        # 未命中完整模式：只保留可能跨片段的前缀（防 pending 无限增长）
        keep = len(self._KEY) + 24
        if len(self._pending) > keep:
            self._pending = self._pending[-keep:]
        return False

    @property
    def answer(self) -> str:
        return self._answer


def _decode_json_escape(char: str) -> str | None:
    return {
        "n": "\n",
        "t": "\t",
        "r": "\r",
        '"': '"',
        "\\": "\\",
        "/": "/",
    }.get(char)


def _read_streaming_llm_content(
    response: Any,
    *,
    cancellation_checker: CancellationChecker | None,
    preview_reporter: PreviewReporter | None = None,
) -> str:
    try:
        iterator = iter(response)
    except TypeError:
        # 非流式重试路径：读取前检查取消，避免用户取消后仍跑完整次生成
        if cancellation_checker is not None:
            cancellation_checker()
        body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])

    content = ""
    plain_response = bytearray()
    saw_sse = False
    last_preview_len = 0
    extractor = _AnswerStreamExtractor()

    def maybe_preview() -> None:
        nonlocal last_preview_len
        if preview_reporter is None:
            return
        # 预览只透出 answer 字段内容（隐藏 {"answer": 等 JSON 外壳）
        text = extractor.answer
        if len(text) - last_preview_len >= 24 or (text and last_preview_len == 0):
            preview_reporter(text)
            last_preview_len = len(text)

    for raw_line in iterator:
        if cancellation_checker is not None:
            cancellation_checker()
        line_bytes = raw_line.encode("utf-8") if isinstance(raw_line, str) else bytes(raw_line)
        line = line_bytes.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            plain_response.extend(line_bytes)
            continue
        saw_sse = True
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if not isinstance(piece, str) or not piece:
            continue
        content += piece
        extractor.feed(piece)
        maybe_preview()

    if cancellation_checker is not None:
        cancellation_checker()
    if preview_reporter is not None and content.strip():
        # 最终预览：优先用解析后的 answer，失败则用提取器累积内容
        final_answer = extractor.answer
        try:
            parsed = _parse_generated_content(content)
            if parsed.answer:
                final_answer = parsed.answer
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
        if final_answer:
            preview_reporter(final_answer)
    if saw_sse:
        if not content.strip():
            raise ValueError("answer generation returned an empty stream")
        return content
    if plain_response:
        body = json.loads(plain_response.decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])
    raise ValueError("answer generation returned no content")


def _extractive_fallback(context_chunks: list[RetrievalResult]) -> GeneratedAnswer | None:
    """简化摘录兜底：取 top chunk 原文片段，hedged 标注。"""
    if not context_chunks:
        return None
    chunk = context_chunks[0]
    excerpt = _relevant_extractive_excerpt(chunk.text, "")
    if not excerpt:
        return None
    return GeneratedAnswer(
        answer=f"{HEDGE_PREFIX}，{excerpt}",
        answer_mode="hedged",
        evidence_sufficiency="partial",
        hedge_note="LLM 生成失败，直接摘录知识库原文",
        generation_status="degraded",
        degraded=True,
    )


def _relevant_extractive_excerpt(text: str, question: str, *, max_chars: int = 420) -> str:
    """取 chunk 头部最相关的句子（简化版：长度截断 + 句界切分）。"""
    source = str(text or "").strip()
    if len(source) <= max_chars:
        return source
    segments = [
        item.strip()
        for item in re.split(r"(?<=[。！？；])\s*|\n{2,}", source)
        if item.strip()
    ]
    if not segments:
        return source[:max_chars].rstrip() + "…"
    picked = ""
    for segment in segments:
        if len(picked) + len(segment) > max_chars:
            break
        picked += segment
    return picked.rstrip() if picked else source[:max_chars].rstrip() + "…"


def _extract_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, re.S)
    if not match:
        raise ValueError("missing JSON response")
    return match.group(0)
