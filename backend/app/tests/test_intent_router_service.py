# -*- coding: utf-8 -*-
"""意图路由服务测试：规则命中 / LLM 兜底 / 失败回退 / 策略映射。"""

import pytest

from backend.app.services import intent_router_service
from backend.app.services.intent_router_service import (
    INTENT_GREETING,
    INTENT_HR_QUALITY,
    INTENT_OFF_TOPIC,
    INTENT_PROJECT_DEEP_DIVE,
    INTENT_RESUME_DETAIL,
    INTENT_SELF_INTRO,
    INTENT_TECH_QUIZ,
    _STRATEGIES,
    classify_intent,
)
from backend.app.services.llm_client import ChatCompletionError


def _llm_mock(monkeypatch, content: str) -> None:
    monkeypatch.setattr(
        intent_router_service,
        "chat_completion_content",
        lambda config, messages, **kwargs: content,
    )


# ---------------------------------------------------------------------------
# 规则命中（零 LLM，classifier="rule"）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("你好，能介绍一下自己吗？", INTENT_GREETING),
        ("您好，请问在吗", INTENT_GREETING),
        ("谢谢！", INTENT_GREETING),
        ("早上好", INTENT_GREETING),
    ],
)
def test_rule_hits_greeting(question, expected) -> None:
    result = classify_intent(question)

    assert result.intent == expected
    assert result.classifier == "rule"
    assert result.confidence == 0.9
    assert result.reason is not None and "命中规则词" in result.reason


@pytest.mark.parametrize(
    "question",
    [
        "今天天气怎么样",
        "帮我写一首诗",
        "现在几点了",
        "彩票怎么买",
        "讲个笑话",
    ],
)
def test_rule_hits_off_topic(question) -> None:
    result = classify_intent(question)

    assert result.intent == INTENT_OFF_TOPIC
    assert result.classifier == "rule"
    assert result.confidence == 0.9


@pytest.mark.parametrize(
    "question",
    [
        "请做个自我介绍",
        "介绍一下你自己",
        "你是谁",
        "简单介绍下你的情况",
        "自我介绍一下吧",
    ],
)
def test_rule_hits_self_intro(question) -> None:
    result = classify_intent(question)

    assert result.intent == INTENT_SELF_INTRO
    assert result.classifier == "rule"


@pytest.mark.parametrize(
    "question",
    [
        "你的技术栈是什么",
        "你在哪个学校上学",
        "你的邮箱是什么",
        "什么时候毕业",
        "获过哪些奖项",
        "你的绩点是多少",
    ],
)
def test_rule_hits_resume_detail(question) -> None:
    result = classify_intent(question)

    assert result.intent == INTENT_RESUME_DETAIL
    assert result.classifier == "rule"


def test_rule_priority_greeting_wins_over_off_topic() -> None:
    result = classify_intent("你好，今天天气怎么样")

    assert result.intent == INTENT_GREETING
    assert result.classifier == "rule"


# ---------------------------------------------------------------------------
# LLM 兜底（classifier="llm"）
# ---------------------------------------------------------------------------

def test_llm_path_returns_classified_intent(monkeypatch) -> None:
    _llm_mock(
        monkeypatch,
        '{"intent": "tech_quiz", "confidence": 0.82, "reason": "Java 原理类问题"}',
    )

    result = classify_intent("Java的HashMap扩容机制是怎么工作的？")

    assert result.intent == INTENT_TECH_QUIZ
    assert result.classifier == "llm"
    assert result.confidence == 0.82
    assert result.reason == "Java 原理类问题"


def test_llm_path_project_deep_dive(monkeypatch) -> None:
    _llm_mock(
        monkeypatch,
        '{"intent": "project_deep_dive", "confidence": 0.9, "reason": "追问项目细节"}',
    )

    result = classify_intent("秒杀项目怎么解决超卖问题的？")

    assert result.intent == INTENT_PROJECT_DEEP_DIVE
    assert result.classifier == "llm"
    assert result.strategy.rewrite is True


def test_llm_path_hr_quality(monkeypatch) -> None:
    _llm_mock(
        monkeypatch,
        '{"intent": "hr_quality", "confidence": 0.7, "reason": "职业规划"}',
    )

    result = classify_intent("你的职业规划是什么？")

    assert result.intent == INTENT_HR_QUALITY
    assert result.classifier == "llm"


def test_llm_returns_unknown_intent_falls_back(monkeypatch) -> None:
    _llm_mock(monkeypatch, '{"intent": "unknown_intent", "confidence": 0.9}')

    result = classify_intent("一个规则未命中的问题")

    assert result.intent == INTENT_RESUME_DETAIL
    assert result.classifier == "fallback"


# ---------------------------------------------------------------------------
# LLM 失败 → 兜底 resume_detail（classifier="fallback"）
# ---------------------------------------------------------------------------

def test_llm_failure_falls_back_to_resume_detail(monkeypatch) -> None:
    def failing_llm(config, messages, **kwargs):
        raise ChatCompletionError("LLM API key is not configured")

    monkeypatch.setattr(intent_router_service, "chat_completion_content", failing_llm)

    result = classify_intent("这个问题的意图完全无法用规则判断")

    assert result.intent == INTENT_RESUME_DETAIL
    assert result.classifier == "fallback"
    assert result.confidence == 0.3
    assert result.reason == "意图分类失败，回退简历细节类"


def test_llm_invalid_json_falls_back_to_resume_detail(monkeypatch) -> None:
    _llm_mock(monkeypatch, "这不是一个 JSON")

    result = classify_intent("规则无法判断的问题")

    assert result.intent == INTENT_RESUME_DETAIL
    assert result.classifier == "fallback"


# ---------------------------------------------------------------------------
# 策略映射
# ---------------------------------------------------------------------------

def test_strategy_mapping_polite_redirect_for_greeting_and_off_topic() -> None:
    greeting_strategy = _STRATEGIES[INTENT_GREETING]
    off_topic_strategy = _STRATEGIES[INTENT_OFF_TOPIC]

    assert greeting_strategy.polite_redirect is True
    assert greeting_strategy.direct_generation_allowed is False
    assert off_topic_strategy.polite_redirect is True
    assert off_topic_strategy.direct_generation_allowed is False


def test_strategy_mapping_rewrite_for_deep_dive_and_tech_quiz() -> None:
    assert _STRATEGIES[INTENT_PROJECT_DEEP_DIVE].rewrite is True
    assert _STRATEGIES[INTENT_TECH_QUIZ].rewrite is True
    assert _STRATEGIES[INTENT_RESUME_DETAIL].rewrite is False


def test_strategy_anchored_for_self_intro_and_hr_quality() -> None:
    assert _STRATEGIES[INTENT_SELF_INTRO].anchor_documents == ("自我介绍", "简历文字版")
    assert "求职动机与职业规划" in _STRATEGIES[INTENT_HR_QUALITY].anchor_documents


def test_classify_result_carries_retrieval_strategy() -> None:
    result = classify_intent("请做个自我介绍")

    assert result.strategy.anchor_documents == ("自我介绍", "简历文字版")
    assert result.strategy.polite_redirect is False
