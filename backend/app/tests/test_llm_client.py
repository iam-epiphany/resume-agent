import json

from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
    build_chat_payload,
    build_chat_request,
    chat_completion_content,
)


def test_openai_compatible_payload_excludes_provider_specific_thinking() -> None:
    payload = build_chat_payload(
        ChatCompletionConfig(
            provider="openai_compatible",
            api_key="key",
            base_url="https://api.example.com/v1",
            model="model-a",
            timeout_seconds=10,
            include_thinking=False,
            response_format="json_object",
        ),
        [{"role": "user", "content": "return json"}],
        max_tokens=128,
    )

    assert payload["model"] == "model-a"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 128
    # 非 DeepSeek 服务不认识 thinking 参数，保持不发送
    assert "thinking" not in payload


def test_deepseek_payload_disables_thinking_by_default() -> None:
    payload = build_chat_payload(
        ChatCompletionConfig(
            provider="openai_compatible",
            api_key="key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            timeout_seconds=10,
            include_thinking=False,
            response_format="json_object",
        ),
        [{"role": "user", "content": "return json"}],
    )

    # DeepSeek 推理模型默认思考会导致流式先输出 reasoning_content（content 为 null），
    # 思考长时被误判为空响应；include_thinking=False 必须显式禁用。
    assert payload["thinking"] == {"type": "disabled"}


def test_deepseek_payload_can_opt_into_thinking_flag() -> None:
    payload = build_chat_payload(
        ChatCompletionConfig(
            provider="deepseek",
            api_key="key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            timeout_seconds=10,
            include_thinking=True,
            response_format="json_object",
        ),
        [{"role": "user", "content": "return json"}],
    )

    assert payload["thinking"] == {"type": "enabled"}


def test_chat_request_accepts_root_or_full_chat_completion_url() -> None:
    root_request = build_chat_request(
        ChatCompletionConfig("openai_compatible", "key", "https://api.example.com/v1", "m", 10),
        {"model": "m", "messages": []},
    )
    full_request = build_chat_request(
        ChatCompletionConfig("openai_compatible", "key", "https://api.example.com/v1/chat/completions", "m", 10),
        {"model": "m", "messages": []},
    )

    assert root_request.full_url == "https://api.example.com/v1/chat/completions"
    assert full_request.full_url == root_request.full_url
    assert json.loads(root_request.data.decode("utf-8"))["model"] == "m"


def test_missing_api_key_fails_before_http_request() -> None:
    called = False

    def opener(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("opener must not be called")

    try:
        chat_completion_content(
            ChatCompletionConfig("openai_compatible", None, "https://api.example.com/v1", "m", 10),
            [{"role": "user", "content": "hi"}],
            opener=opener,
        )
    except ChatCompletionError as exc:
        assert "API key" in str(exc)
    else:
        raise AssertionError("expected ChatCompletionError")
    assert called is False


# ---------------------------------------------------------------------------
# 请求级 LLM 调用计数（2026-08-08 二轮）：llm_client 真实 API 请求统计
# ---------------------------------------------------------------------------

from backend.app.services.llm_client import (
    llm_call_count,
    open_chat_completion,
    reset_llm_call_count,
)


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")


def _config() -> ChatCompletionConfig:
    return ChatCompletionConfig(
        provider="openai_compatible",
        api_key="key",
        base_url="https://api.example.com",
        model="model-a",
        timeout_seconds=5,
    )


def test_llm_call_count_increments_on_success(monkeypatch) -> None:
    reset_llm_call_count()
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: _FakeResponse())

    chat_completion_content(_config(), [{"role": "user", "content": "hi"}])

    assert llm_call_count() == 1


def test_llm_call_count_increments_on_timeout(monkeypatch) -> None:
    """超时/失败也算一次真实 API 调用（请求已发出）。"""
    reset_llm_call_count()

    def failing_urlopen(*_a, **_k):
        raise TimeoutError("timeout")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)

    try:
        chat_completion_content(_config(), [{"role": "user", "content": "hi"}])
    except ChatCompletionError:
        pass

    assert llm_call_count() == 1


def test_llm_call_count_isolated_between_requests() -> None:
    reset_llm_call_count()
    assert llm_call_count() == 0
    reset_llm_call_count()
    assert llm_call_count() == 0


def test_open_chat_completion_increments_count(monkeypatch) -> None:
    reset_llm_call_count()
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: _FakeResponse())

    open_chat_completion(_config(), [{"role": "user", "content": "hi"}], stream=False)

    assert llm_call_count() == 1
