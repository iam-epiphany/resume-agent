# -*- coding: utf-8 -*-
"""面试官意图路由：纯 LLM 3 类意图分类 + 追问补全（一次调用合并完成）。

设计动机（认知中间层，参考主流开源 RAG 项目 Router 思路）：
用一次小 LLM 调用完成「意图分类 + 指代消解」双职责，替代此前关键词表与
规则预检的硬编码。任何关键词表都无法穷尽面试官的表达方式，且易误伤
（如"你好，介绍下你的项目"曾被整题转移为寒暄）。分类失败保守回退 resume_qa，
意图层绝不当掉任何检索机会。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from backend.app.core.config import (
    INTENT_FAST_PATH_ENABLED,
    INTENT_ROUTER_API_KEY,
    INTENT_ROUTER_BASE_URL,
    INTENT_ROUTER_ENABLED,
    INTENT_ROUTER_MAX_TOKENS,
    INTENT_ROUTER_MODEL,
    INTENT_ROUTER_PROVIDER,
    INTENT_ROUTER_RESPONSE_FORMAT,
    INTENT_ROUTER_TIMEOUT_SECONDS,
)
from backend.app.services.context_dependency import (
    RESUME_ANCHOR_TERMS,
    is_greeting_phrase,
    needs_context_resolution,
)
from backend.app.services.llm_client import ChatCompletionConfig, ChatCompletionError, chat_completion_content

# 3 类意图（精简：一切可检索的问题归 resume_qa，只留寒暄/无关两个旁路）
INTENT_RESUME_QA = "resume_qa"      # 简历问答（默认，含自我介绍/项目/技术/经历/意向等全部可检索问题）
INTENT_GREETING = "greeting"        # 寒暄类
INTENT_OFF_TOPIC = "off_topic"      # 完全无关类

ALL_INTENTS = (
    INTENT_RESUME_QA,
    INTENT_GREETING,
    INTENT_OFF_TOPIC,
)


@dataclass(frozen=True)
class RetrievalStrategy:
    """意图 → 检索策略映射（其余字段已随词表删除，只剩礼貌转移开关）。"""

    polite_redirect: bool = False   # 跳过检索，直接礼貌转移


@dataclass(frozen=True)
class IntentResult:
    intent: str
    classifier: str          # "llm" | "fallback"
    confidence: float
    reason: str | None = None
    rewritten_question: str = ""    # 追问补全后的独立完整问题（无需补全时与原问题一致）
    needs_context: bool = False     # 是否参考了上一轮对话并据此补全
    strategy: RetrievalStrategy = field(default_factory=RetrievalStrategy)


_STRATEGIES: dict[str, RetrievalStrategy] = {
    INTENT_RESUME_QA: RetrievalStrategy(),
    INTENT_GREETING: RetrievalStrategy(polite_redirect=True),
    INTENT_OFF_TOPIC: RetrievalStrategy(polite_redirect=True),
}

_CLASSIFY_PROMPT = """你是简历问答系统的意图分类与追问理解模块。用户（面试官）的问题都围绕简历主人公（张三，计算机方向求职者），任何关于他本人、简历或计算机领域的问题都属于可检索问题。
把问题分类到以下 3 类之一：
- resume_qa: 与简历主人公/简历内容/计算机技术相关的任何问题（自我介绍、项目细节、技术原理、教育经历、证书荣誉、求职意向、个人特质等全部归入此类）
- greeting: 纯寒暄问候（如"你好""谢谢""在吗"；寒暄之后跟着实质问题则归 resume_qa）
- off_topic: 与简历和计算机完全无关的话题（天气、烹饪、娱乐等）
规则：
1. 无法确定类别时一律归 resume_qa
2. 寒暄与实质问题混合的提问一律归 resume_qa
3. 若提供了上一轮对话且当前问题明显省略了话题（如"那怎么解决的？"指代上一轮提到的事物），必须把 rewritten_question 补全为不依赖上下文的独立完整问题；若当前问题是新话题或无需补全，rewritten_question 原样返回当前问题
只输出 JSON：{"intent": "<类别>", "confidence": 0.0-1.0, "reason": "<一句话理由>", "rewritten_question": "<补全后的问题>", "needs_context": true/false}"""


def _llm_config() -> ChatCompletionConfig:
    return ChatCompletionConfig(
        provider=INTENT_ROUTER_PROVIDER,
        api_key=INTENT_ROUTER_API_KEY,
        base_url=INTENT_ROUTER_BASE_URL,
        model=INTENT_ROUTER_MODEL,
        timeout_seconds=INTENT_ROUTER_TIMEOUT_SECONDS,
        response_format=INTENT_ROUTER_RESPONSE_FORMAT,
    )


def _fallback_result(question: str) -> IntentResult:
    return IntentResult(
        intent=INTENT_RESUME_QA,
        classifier="fallback",
        confidence=0.3,
        reason="意图分类失败，保守回退简历问答",
        rewritten_question=question,
        strategy=_STRATEGIES[INTENT_RESUME_QA],
    )


# 简历领域锚点（自足问题快速归类 resume_qa；无锚点的自足问题交给 LLM 判 off_topic）
# 定义见 context_dependency.RESUME_ANCHOR_TERMS（单一事实源）


def _fast_path_result(question: str) -> IntentResult | None:
    """自足问题（不依赖上一轮上下文）的确定性旁路。

    仅当问题判定为"自足"（needs_context_resolution=False）时才可能走 fast path：
    - 整句问候短语 → greeting（零 LLM 礼貌转移）
    - 含简历领域锚点 → resume_qa（直接进检索）
    - 无锚点（如"今天天气怎么样""帮我写一首诗"）→ 返回 None 交 LLM 判 off_topic
    注意：短词（"爱好/籍贯/薪资/项目"等 ≤4 字）会被 needs_context_resolution
    判为需要上下文（可能指代前文对象），不会在此被当作 greeting——它们交给 LLM。
    """
    stripped = question.strip()
    if not stripped:
        return _fallback_result(stripped)
    if is_greeting_phrase(stripped):
        return IntentResult(
            intent=INTENT_GREETING,
            classifier="fast_path",
            confidence=0.9,
            reason="fast path：整句问候短语",
            rewritten_question=stripped,
            strategy=_STRATEGIES[INTENT_GREETING],
        )
    if any(term in stripped for term in RESUME_ANCHOR_TERMS):
        return IntentResult(
            intent=INTENT_RESUME_QA,
            classifier="fast_path",
            confidence=0.7,
            reason="fast path：自足问题命中简历领域锚点",
            rewritten_question=stripped,
            strategy=_STRATEGIES[INTENT_RESUME_QA],
        )
    return None


def classify_and_resolve(question: str, previous_turn: dict | None = None) -> IntentResult:
    """意图分类 + 追问补全。

    完整链路：一次 LLM 调用完成「3 类意图分类 + 指代消解」；
    fast path（默认开启）：**无论是否有上一轮对话**，只要当前问题是自足的
    （needs_context_resolution=False，不依赖前文），就用确定性规则旁路：
    整句问候 → greeting；其余 → resume_qa。这样用户连续提多个独立问题时，
    第二问起仍能跳过 LLM。只有真依赖上下文（追问/指代/省略）时才走 LLM 消解。
    分类失败/坏 JSON → 保守回退 resume_qa 原问，意图层绝不当掉任何检索机会。
    """
    if not INTENT_ROUTER_ENABLED:
        return _fallback_result(question)
    if INTENT_FAST_PATH_ENABLED and not needs_context_resolution(question):
        fast_result = _fast_path_result(question)
        if fast_result is not None:
            return fast_result
    if previous_turn:
        user_content = (
            f"上一轮对话：\nQ: {previous_turn.get('question') or ''}\n"
            f"A: {(previous_turn.get('answer_excerpt') or '')[:120]}\n\n"
            f"当前问题：{question}"
        )
    else:
        user_content = f"当前问题：{question}"
    messages = [
        {"role": "system", "content": _CLASSIFY_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        content = chat_completion_content(
            _llm_config(), messages, temperature=0, max_tokens=INTENT_ROUTER_MAX_TOKENS
        )
    except ChatCompletionError:
        return _fallback_result(question)
    try:
        payload = json.loads(_extract_json_object(content))
    except (json.JSONDecodeError, ValueError):
        return _fallback_result(question)
    intent = str(payload.get("intent") or "").strip().lower()
    if intent not in ALL_INTENTS:
        # LLM 输出非法意图：回退 resume_qa 并按失败处理（保守）
        return _fallback_result(question)
    confidence = float(payload.get("confidence") or 0.5)
    rewritten = str(payload.get("rewritten_question") or "").strip() or question
    return IntentResult(
        intent=intent,
        classifier="llm",
        confidence=max(0.0, min(1.0, confidence)),
        reason=str(payload.get("reason") or "")[:200] or None,
        rewritten_question=rewritten,
        needs_context=bool(payload.get("needs_context")),
        strategy=_STRATEGIES[intent],
    )


def _extract_json_object(content: str) -> str:
    """从 LLM 输出中截取首个 JSON 对象（容错：输出可能夹带前后缀）。"""
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object")
    return content[start : end + 1]
