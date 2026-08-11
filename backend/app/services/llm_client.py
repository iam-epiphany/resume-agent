from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
import urllib.error
import urllib.request
from typing import Any, Callable


UrlOpen = Callable[..., Any]

# 请求级 LLM 调用计数：每次真正向模型 API 发起请求（urlopen）前 +1。
# 用 ContextVar 保证请求间隔离（并发/异步下互不干扰），任何模块（intent/planner/
# rewrite/generation）通过 llm_client 调用都会被统计。调用方在请求结束时读取
# 并写入响应（QAResponse.llm_call_count），供评测与观测——统计真实 API 调用，
# 而非根据 pipeline 状态推测。
_LLM_CALL_COUNT: ContextVar[int] = ContextVar("resumemind_llm_call_count", default=0)


def llm_call_count() -> int:
    """当前请求已发起的真实 LLM API 调用次数。"""
    return _LLM_CALL_COUNT.get()


def reset_llm_call_count() -> None:
    """请求开始时清零计数（qa_task 在单问处理前调用）。"""
    _LLM_CALL_COUNT.set(0)


def _record_llm_call() -> None:
    _LLM_CALL_COUNT.set(_LLM_CALL_COUNT.get() + 1)


@dataclass(frozen=True)
class ChatCompletionConfig:
    provider: str
    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: float
    include_thinking: bool = False
    response_format: str = "json_object"


class ChatCompletionError(RuntimeError):
    pass


def chat_completion_content(
    config: ChatCompletionConfig,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
    response_format: str | None = None,
    opener: UrlOpen | None = None,
) -> str:
    if not config.api_key:
        raise ChatCompletionError("LLM API key is not configured")
    payload = build_chat_payload(
        config,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        stream=False,
    )
    request = build_chat_request(config, payload)
    _record_llm_call()  # 真正发起 API 请求前计数（无论成败，超时/失败也算一次真实调用）
    try:
        with (opener or urllib.request.urlopen)(request, timeout=config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ChatCompletionError("LLM chat completion request failed") from exc
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatCompletionError("LLM chat completion response is not parseable") from exc


def open_chat_completion(
    config: ChatCompletionConfig,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
    response_format: str | None = None,
    stream: bool = True,
    opener: UrlOpen | None = None,
) -> Any:
    if not config.api_key:
        raise ChatCompletionError("LLM API key is not configured")
    payload = build_chat_payload(
        config,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        stream=stream,
    )
    request = build_chat_request(config, payload)
    _record_llm_call()  # 真正发起 API 请求前计数
    try:
        return (opener or urllib.request.urlopen)(request, timeout=config.timeout_seconds)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise ChatCompletionError("LLM chat completion request failed") from exc


def build_chat_payload(
    config: ChatCompletionConfig,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0,
    max_tokens: int | None = None,
    response_format: str | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": temperature,
        "messages": messages,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stream:
        payload["stream"] = True
    selected_response_format = config.response_format if response_format is None else response_format
    if selected_response_format and selected_response_format.lower() not in {"off", "none", "disabled"}:
        payload["response_format"] = {"type": selected_response_format}
    # DeepSeek 推理模型（如 deepseek-v4-flash）默认思考：流式响应先输出
    # reasoning_content（此时 content 为 null），思考结束后才输出 content——
    # 思考较长时会导致流式解析拿不到 content、被误判为空响应/超时。
    # 对 DeepSeek 服务显式设置 thinking：include_thinking=False（默认）→ disabled（不思考），True → enabled。
    # 其他 OpenAI 兼容服务不认识 thinking 参数，保持不发送。
    is_deepseek = (
        str(config.provider).strip().lower() == "deepseek"
        or "deepseek.com" in str(config.base_url or "").lower()
    )
    if is_deepseek:
        payload["thinking"] = {"type": "enabled" if config.include_thinking else "disabled"}
    return payload


def build_chat_request(config: ChatCompletionConfig, payload: dict[str, Any]) -> urllib.request.Request:
    return urllib.request.Request(
        _chat_completions_endpoint(config.base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _chat_completions_endpoint(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ChatCompletionError("LLM base URL is not configured")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"
