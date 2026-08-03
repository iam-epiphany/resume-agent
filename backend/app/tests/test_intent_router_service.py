# -*- coding: utf-8 -*-
"""意图路由服务测试：纯 LLM 3 类分类 / 混合问题归 resume_qa / 失败回退 / 合并追问补全。"""

import pytest

from backend.app.services import intent_router_service
from backend.app.services.intent_router_service import (
    ALL_INTENTS,
    INTENT_GREETING,
    INTENT_OFF_TOPIC,
    INTENT_RESUME_QA,
    _STRATEGIES,
    classify_and_resolve,
)
from backend.app.services.llm_client import ChatCompletionError


def _llm_mock(monkeypatch, content: str) -> None:
    monkeypatch.setattr(
        intent_router_service,
        "chat_completion_content",
        lambda config, messages, **kwargs: content,
    )


def _payload(intent: str, **overrides) -> dict:
    base = {
        "intent": intent,
        "confidence": 0.85,
        "reason": "测试",
        "rewritten_question": "",
        "needs_context": False,
    }
    base.update(overrides)
    return base


def _llm_mock_json(monkeypatch, payload: dict) -> None:
    import json

    _llm_mock(monkeypatch, json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 3 类分类（classifier="llm"）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "intent",
    [INTENT_RESUME_QA, INTENT_GREETING, INTENT_OFF_TOPIC],
)
def test_llm_classifies_each_of_three_intents(monkeypatch, intent) -> None:
    _llm_mock_json(monkeypatch, _payload(intent, reason=f"{intent} 理由"))

    result = classify_and_resolve("某个面试问题")

    assert result.intent == intent
    assert result.classifier == "llm"
    assert result.confidence == 0.85
    assert result.reason == f"{intent} 理由"


def test_llm_classifies_resume_qa_for_project_question(monkeypatch) -> None:
    _llm_mock_json(monkeypatch, _payload(INTENT_RESUME_QA))

    result = classify_and_resolve("秒杀项目怎么解决超卖问题的？")

    assert result.intent == INTENT_RESUME_QA


def test_llm_classifies_greeting(monkeypatch) -> None:
    _llm_mock_json(monkeypatch, _payload(INTENT_GREETING))

    result = classify_and_resolve("你好")

    assert result.intent == INTENT_GREETING
    assert result.strategy.polite_redirect is True


def test_llm_classifies_off_topic(monkeypatch) -> None:
    _llm_mock_json(monkeypatch, _payload(INTENT_OFF_TOPIC))

    result = classify_and_resolve("你会做红烧肉吗")

    assert result.intent == INTENT_OFF_TOPIC
    assert result.strategy.polite_redirect is True


def test_strategy_mapping_only_polite_redirect_remains() -> None:
    assert _STRATEGIES[INTENT_RESUME_QA].polite_redirect is False
    assert _STRATEGIES[INTENT_GREETING].polite_redirect is True
    assert _STRATEGIES[INTENT_OFF_TOPIC].polite_redirect is True


# ---------------------------------------------------------------------------
# LLM 返回异常/未知 → 保守回退 resume_qa（classifier="fallback"）
# ---------------------------------------------------------------------------

def test_llm_returns_unknown_intent_falls_back_to_resume_qa(monkeypatch) -> None:
    _llm_mock_json(monkeypatch, _payload("unknown_intent"))

    result = classify_and_resolve("一个意图未知的问题")

    assert result.intent == INTENT_RESUME_QA
    assert result.classifier == "fallback"
    assert result.confidence == 0.3
    assert result.rewritten_question == "一个意图未知的问题"


def test_llm_failure_falls_back_to_resume_qa_with_original_question(monkeypatch) -> None:
    def failing_llm(config, messages, **kwargs):
        raise ChatCompletionError("LLM API key is not configured")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", failing_llm)

    result = classify_and_resolve("这个问题的意图完全无法判断")

    assert result.intent == INTENT_RESUME_QA
    assert result.classifier == "fallback"
    assert result.confidence == 0.3
    assert result.rewritten_question == "这个问题的意图完全无法判断"
    assert result.needs_context is False


def test_llm_invalid_json_falls_back_to_resume_qa(monkeypatch) -> None:
    _llm_mock(monkeypatch, "这不是一个 JSON")

    result = classify_and_resolve("规则无法判断的问题")

    assert result.intent == INTENT_RESUME_QA
    assert result.classifier == "fallback"


def test_router_disabled_falls_back_to_resume_qa(monkeypatch) -> None:
    monkeypatch.setattr(intent_router_service, "INTENT_ROUTER_ENABLED", False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("路由关闭时不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", fail_if_called)

    result = classify_and_resolve("你好，介绍一下你的项目")

    assert result.intent == INTENT_RESUME_QA
    assert result.classifier == "fallback"
    assert result.rewritten_question == "你好，介绍一下你的项目"


# ---------------------------------------------------------------------------
# 合并追问补全（带上一轮 / 无上一轮 / 新话题）
# ---------------------------------------------------------------------------

def test_resolve_completes_followup_with_previous_turn(monkeypatch) -> None:
    previous = {
        "question": "介绍一下你的秒杀项目",
        "answer_excerpt": "秒杀项目使用 Redis 预扣库存。",
    }
    _llm_mock_json(
        monkeypatch,
        _payload(
            INTENT_RESUME_QA,
            rewritten_question="秒杀项目中如何解决超卖问题？",
            needs_context=True,
        ),
    )

    result = classify_and_resolve("那怎么解决超卖的？", previous)

    assert result.intent == INTENT_RESUME_QA
    assert result.rewritten_question == "秒杀项目中如何解决超卖问题？"
    assert result.needs_context is True


def test_resolve_keeps_original_when_new_topic(monkeypatch) -> None:
    previous = {
        "question": "介绍一下你的秒杀项目",
        "answer_excerpt": "秒杀项目使用 Redis 预扣库存。",
    }
    _llm_mock_json(
        monkeypatch,
        _payload(
            INTENT_RESUME_QA,
            rewritten_question="你的技术栈是什么？",
            needs_context=False,
        ),
    )

    result = classify_and_resolve("你的技术栈是什么？", previous)

    assert result.needs_context is False
    assert result.rewritten_question == "你的技术栈是什么？"


def test_resolve_without_previous_turn_passes_question(monkeypatch) -> None:
    _llm_mock_json(
        monkeypatch,
        _payload(INTENT_RESUME_QA, rewritten_question="你的技术栈是什么？", needs_context=False),
    )

    result = classify_and_resolve("你的技术栈是什么？", None)

    assert result.rewritten_question == "你的技术栈是什么？"
    assert result.needs_context is False


def test_resolve_uses_original_question_when_rewrite_empty(monkeypatch) -> None:
    _llm_mock_json(
        monkeypatch,
        _payload(INTENT_RESUME_QA, rewritten_question="", needs_context=False),
    )

    result = classify_and_resolve("介绍一下你的项目")

    assert result.rewritten_question == "介绍一下你的项目"


def test_resolve_failure_keeps_original_question(monkeypatch) -> None:
    def failing_llm(config, messages, **kwargs):
        raise ChatCompletionError("LLM API key is not configured")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", failing_llm)
    previous = {"question": "秒杀项目怎么设计的？", "answer_excerpt": "使用 Redis 预扣库存。"}

    result = classify_and_resolve("那怎么解决超卖的？", previous)

    assert result.rewritten_question == "那怎么解决超卖的？"
    assert result.needs_context is False
    assert result.intent == INTENT_RESUME_QA


# ---------------------------------------------------------------------------
# 混合/边界：寒暄+实质问题归 resume_qa（由 LLM 判定；验证 prompt 语义被遵守）
# ---------------------------------------------------------------------------

def test_llm_json_escaped_output_is_parsed(monkeypatch) -> None:
    _llm_mock(
        monkeypatch,
        '前置说明 {"intent": "resume_qa", "confidence": 0.9, '
        '"reason": "项目问题", "rewritten_question": "介绍一下你的秒杀项目", '
        '"needs_context": false} 后置说明',
    )

    result = classify_and_resolve("介绍一下你的秒杀项目")

    assert result.intent == INTENT_RESUME_QA
    assert result.rewritten_question == "介绍一下你的秒杀项目"


def test_all_intents_are_exactly_three() -> None:
    assert set(ALL_INTENTS) == {INTENT_RESUME_QA, INTENT_GREETING, INTENT_OFF_TOPIC}
