# -*- coding: utf-8 -*-
"""意图路由服务测试：LLM 7 类分类 / 混合问题 / 失败回退 / 合并追问补全。"""

import pytest

from backend.app.services import intent_router_service
from backend.app.services.intent_router_service import (
    ALL_INTENTS,
    INTENT_GREETING,
    INTENT_HR_BEHAVIOR,
    INTENT_OFF_TOPIC,
    INTENT_PROJECT_DEEPDIVE,
    INTENT_RESUME_FACT,
    INTENT_RESUME_QA,
    INTENT_TECH_GENERAL,
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
    monkeypatch.setattr(intent_router_service, "INTENT_FAST_PATH_ENABLED", False)
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


def test_strategy_mapping_policies() -> None:
    assert _STRATEGIES[INTENT_RESUME_QA].polite_redirect is False
    assert _STRATEGIES[INTENT_RESUME_QA].evidence_policy == "default"
    assert _STRATEGIES[INTENT_RESUME_FACT].evidence_policy == "fact_strict"
    assert _STRATEGIES[INTENT_PROJECT_DEEPDIVE].evidence_policy == "project_grounded"
    assert _STRATEGIES[INTENT_TECH_GENERAL].evidence_policy == "tech_split"
    assert _STRATEGIES[INTENT_HR_BEHAVIOR].evidence_policy == "persona_soft"
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


def test_all_intents_are_seven() -> None:
    assert set(ALL_INTENTS) == {
        INTENT_RESUME_QA,
        INTENT_RESUME_FACT,
        INTENT_PROJECT_DEEPDIVE,
        INTENT_TECH_GENERAL,
        INTENT_HR_BEHAVIOR,
        INTENT_GREETING,
        INTENT_OFF_TOPIC,
    }


# ---------------------------------------------------------------------------
# fast path：普通独立问题跳过 LLM（INTENT_FAST_PATH_ENABLED 默认开启）
# ---------------------------------------------------------------------------

def test_fast_path_greeting_phrase_skips_llm(monkeypatch) -> None:
    called = {"llm": False}

    def assert_not_called(config, messages, **kwargs):
        called["llm"] = True
        raise AssertionError("fast path 不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)

    result = classify_and_resolve("你好")

    assert result.intent == INTENT_GREETING
    assert result.classifier == "fast_path"
    assert called["llm"] is False


def test_fast_path_resume_anchor_skips_llm(monkeypatch) -> None:
    def assert_not_called(config, messages, **kwargs):
        raise AssertionError("fast path 不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)

    result = classify_and_resolve("介绍一下你的项目经历")

    assert result.intent == INTENT_RESUME_QA
    assert result.classifier == "fast_path"


def test_fast_path_mixed_greeting_with_substance_returns_resume_qa(monkeypatch) -> None:
    """"你好，介绍下你的项目" 不是整句问候、但含简历锚点——fast path 直接归 resume_qa（不误伤）。"""
    def assert_not_called(config, messages, **kwargs):
        raise AssertionError("含锚点的混合问题不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)

    result = classify_and_resolve("你好，介绍下你的项目")

    assert result.intent == INTENT_RESUME_QA
    assert result.classifier == "fast_path"


def test_fast_path_disabled_falls_back_to_llm(monkeypatch) -> None:
    monkeypatch.setattr(intent_router_service, "INTENT_FAST_PATH_ENABLED", False)
    _llm_mock_json(monkeypatch, _payload(INTENT_RESUME_QA, reason="禁用 fast path 走 LLM"))

    result = classify_and_resolve("你好")

    assert result.intent == INTENT_RESUME_QA
    assert result.classifier == "llm"


def test_fast_path_turn_followup_uses_llm_for_disambiguation(monkeypatch) -> None:
    """有上一轮对话时 fast path 不生效，走 LLM 追问补全。"""
    _llm_mock_json(monkeypatch, _payload(INTENT_RESUME_QA, rewritten_question="那 Redis 预扣是怎么防超卖的？", needs_context=True))
    previous = {"question": "秒杀项目怎么设计的？", "answer_excerpt": "使用 Redis 预扣库存。"}

    result = classify_and_resolve("那怎么解决超卖的？", previous)

    assert result.classifier == "llm"
    assert result.needs_context is True
    assert "Redis 预扣" in result.rewritten_question


def test_fast_path_unknown_question_goes_to_llm(monkeypatch) -> None:
    """无锚点、非问候的问题 fast path 不拦截，交 LLM 判定（如 off_topic）。"""
    _llm_mock_json(monkeypatch, _payload(INTENT_OFF_TOPIC))

    result = classify_and_resolve("今天天气怎么样")

    assert result.intent == INTENT_OFF_TOPIC
    assert result.classifier == "llm"


# ---------------------------------------------------------------------------
# fast path 修复（2026-08-08 二轮）：有历史但问题独立仍走 fast path；短词不判 greeting
# ---------------------------------------------------------------------------

def test_fast_path_independent_question_skips_llm_even_with_history(monkeypatch) -> None:
    """Q1 项目 → Q2 HashMap（独立问题）：有上一轮但当前问题自足 → 仍走 fast path。"""
    def assert_not_called(config, messages, **kwargs):
        raise AssertionError("独立问题即使有历史也不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)
    previous = {"question": "你有哪些项目？", "answer_excerpt": "秒杀、外卖、REV。"}

    result = classify_and_resolve("HashMap 为什么线程不安全", previous)

    assert result.classifier == "fast_path"
    assert result.intent == INTENT_TECH_GENERAL


def test_fast_path_independent_fact_question(monkeypatch) -> None:
    def assert_not_called(config, messages, **kwargs):
        raise AssertionError("自足硬事实问题不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)

    result = classify_and_resolve("你得过什么奖学金？")

    assert result.classifier == "fast_path"
    assert result.intent == INTENT_RESUME_FACT


def test_fast_path_independent_hr_question(monkeypatch) -> None:
    def assert_not_called(config, messages, **kwargs):
        raise AssertionError("自足 HR/行为问题不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)

    result = classify_and_resolve("你最大的缺点是什么？")

    assert result.classifier == "fast_path"
    assert result.intent == INTENT_HR_BEHAVIOR


def test_fast_path_independent_project_deepdive(monkeypatch) -> None:
    def assert_not_called(config, messages, **kwargs):
        raise AssertionError("项目深挖问题不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)

    result = classify_and_resolve("秒杀项目怎么解决超卖问题的？")

    assert result.classifier == "fast_path"
    assert result.intent == INTENT_PROJECT_DEEPDIVE


def test_fast_path_project_overview_stays_resume_qa(monkeypatch) -> None:
    """“介绍一下你的项目经历”是概述而非深挖——仍归 resume_qa。"""
    def assert_not_called(config, messages, **kwargs):
        raise AssertionError("含锚点的混合问题不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)

    result = classify_and_resolve("介绍一下你的项目经历")

    assert result.classifier == "fast_path"
    assert result.intent == INTENT_RESUME_QA


def test_fast_path_independent_greeting_skips_llm_even_with_history(monkeypatch) -> None:
    """有上一轮但当前是问候 → 仍走 fast path greeting。"""
    def assert_not_called(config, messages, **kwargs):
        raise AssertionError("问候即使有历史也不应调用 LLM")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", assert_not_called)
    previous = {"question": "介绍一下你的项目", "answer_excerpt": "秒杀项目。"}

    result = classify_and_resolve("谢谢", previous)

    assert result.classifier == "fast_path"
    assert result.intent == INTENT_GREETING


def test_fast_path_dependent_followup_still_uses_llm(monkeypatch) -> None:
    """真依赖上下文的追问（指代）即使有历史也走 LLM 消解。"""
    _llm_mock_json(monkeypatch, _payload(INTENT_RESUME_QA, rewritten_question="第二个项目用了什么技术", needs_context=True))
    previous = {"question": "你有哪些项目？", "answer_excerpt": "秒杀、外卖、REV。"}

    result = classify_and_resolve("第二个项目用了什么技术？", previous)

    assert result.classifier == "llm"
    assert result.needs_context is True


def test_short_word_not_classified_as_greeting(monkeypatch) -> None:
    """"爱好/籍贯/薪资/项目"等短词不应判为 greeting（交给 LLM）。"""
    for word in ("爱好", "籍贯", "薪资", "项目"):
        _llm_mock_json(monkeypatch, _payload(INTENT_RESUME_QA, reason=f"短词 {word} 走 LLM"))

        result = classify_and_resolve(word)

        assert result.classifier == "llm", f"{word} 不应被判为 fast_path"
        assert result.intent == INTENT_RESUME_QA
