# -*- coding: utf-8 -*-
"""答案模式分级测试：sufficiency → answered/hedged、意图转移 → redirected、兜底 → failed。

矩阵：
- sufficient → answered（无推测前缀）
- partial / insufficient / 缺失 / 非法值 → hedged（后端强制“根据现有知识库推测”前缀；
  LLM 自带前缀时不重复添加）
- greeting / off_topic → redirected（礼貌模板文案，零 LLM）
- 空上下文 + LLM 关闭/异常 → failed（FALLBACK_NO_CONTEXT）
"""

import pytest

from backend.app.services import answer_generation_service
from backend.app.services.answer_generation_service import (
    FALLBACK_NO_CONTEXT,
    HEDGE_PREFIX,
    POLITE_REDIRECT_GREETING,
    POLITE_REDIRECT_OFF_TOPIC,
    GeneratedAnswer,
    generate_answer,
)
from backend.app.services.intent_router_service import (
    INTENT_GREETING,
    INTENT_OFF_TOPIC,
    INTENT_RESUME_QA,
)


def _chunk(text: str = "证书有效期至2025年12月31日。") -> list:
    from backend.app.schemas.qa import RetrievalResult

    return [
        RetrievalResult(
            chunk_id="C1",
            rank=1,
            score=0.9,
            source_doc="技能专长.md",
            section_title="测试",
            section_path=["测试"],
            text=text,
            citation_label="[1]",
            metadata={},
        )
    ]


def _mock_llm(monkeypatch, *, answer: str, evidence_sufficiency=None) -> None:
    def fake_call(*args, **kwargs):
        return GeneratedAnswer(
            answer=answer,
            evidence_sufficiency=evidence_sufficiency,
            generation_status="completed",
        )

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_call)


# ---------------------------------------------------------------------------
# 映射矩阵：证据充分度 → 回答模式
# ---------------------------------------------------------------------------

def test_sufficient_maps_to_answered_without_prefix(monkeypatch) -> None:
    _mock_llm(monkeypatch, answer="证书有效期至2025年12月31日。", evidence_sufficiency="sufficient")

    result = generate_answer("证书有效期是什么？", _chunk(), intent=INTENT_RESUME_QA)

    assert result.answer_mode == "answered"
    assert result.evidence_sufficiency == "sufficient"
    assert not result.answer.startswith(HEDGE_PREFIX)


@pytest.mark.parametrize("sufficiency", ["partial", "insufficient"])
def test_partial_and_insufficient_map_to_hedged_with_forced_prefix(monkeypatch, sufficiency) -> None:
    _mock_llm(monkeypatch, answer="推测性回答内容。", evidence_sufficiency=sufficiency)

    result = generate_answer("某个没有直接依据的问题", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.evidence_sufficiency == sufficiency
    assert result.answer.startswith(f"{HEDGE_PREFIX}，")
    assert "推测性回答内容。" in result.answer


def test_llm_provided_prefix_is_kept_without_duplication(monkeypatch) -> None:
    _mock_llm(
        monkeypatch,
        answer=f"{HEDGE_PREFIX}，简历中未明确提及薪资预期。",
        evidence_sufficiency="insufficient",
    )

    result = generate_answer("期望薪资是多少？", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.answer.count(HEDGE_PREFIX) == 1
    assert result.answer == f"{HEDGE_PREFIX}，简历中未明确提及薪资预期。"


def test_llm_provided_prefix_without_comma_is_not_duplicated(monkeypatch) -> None:
    _mock_llm(
        monkeypatch,
        answer=f"{HEDGE_PREFIX}薪资方面没有直接记录。",
        evidence_sufficiency="partial",
    )

    result = generate_answer("薪资如何？", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.answer.count(HEDGE_PREFIX) == 1


def test_missing_sufficiency_maps_to_hedged_with_forced_prefix(monkeypatch) -> None:
    _mock_llm(monkeypatch, answer="没有给出自评的回答。", evidence_sufficiency=None)

    result = generate_answer("未自评的问题", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.evidence_sufficiency == "partial"
    assert result.answer.startswith(f"{HEDGE_PREFIX}，")


def test_invalid_sufficiency_maps_to_hedged(monkeypatch) -> None:
    _mock_llm(monkeypatch, answer="非法自评值的回答。", evidence_sufficiency="unknown")

    result = generate_answer("非法自评的问题", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.evidence_sufficiency == "partial"


# ---------------------------------------------------------------------------
# 意图转移 → redirected
# ---------------------------------------------------------------------------

def test_greeting_maps_to_redirected_with_template_text(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("greeting 不应触发 LLM")

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fail_if_called)

    result = generate_answer("你好", [], intent=INTENT_GREETING)

    assert result.answer_mode == "redirected"
    assert result.answer == POLITE_REDIRECT_GREETING.format(persona_name="简历主人公")
    assert result.evidence_sufficiency is None
    assert result.generation_status == "skipped"


def test_off_topic_maps_to_redirected_with_template_text(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("off_topic 不应触发 LLM")

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fail_if_called)

    result = generate_answer("今天天气怎么样", [], intent=INTENT_OFF_TOPIC)

    assert result.answer_mode == "redirected"
    assert result.answer == POLITE_REDIRECT_OFF_TOPIC.format(persona_name="简历主人公")
    assert result.evidence_sufficiency is None
    assert result.generation_status == "skipped"


# ---------------------------------------------------------------------------
# 空上下文 + LLM 不可用 → failed
# ---------------------------------------------------------------------------

def test_empty_chunks_with_llm_disabled_maps_to_failed(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer("知识库中没有的问题", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "failed"
    assert result.answer == FALLBACK_NO_CONTEXT
    assert result.generation_status == "skipped"
    assert result.evidence_sufficiency is None


def test_empty_chunks_with_llm_exception_maps_to_failed(monkeypatch) -> None:
    def failing_call(*args, **kwargs):
        raise OSError("answer generation unavailable")

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", failing_call)

    result = generate_answer("知识库中没有的问题", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "failed"
    assert result.answer == FALLBACK_NO_CONTEXT
    assert result.generation_status == "skipped"


def test_empty_answer_from_llm_with_chunks_maps_to_hedged_extract(monkeypatch) -> None:
    _mock_llm(monkeypatch, answer="")

    result = generate_answer("证书有效期是什么？", _chunk(), intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.degraded is True
    assert "2025年12月31日" in result.answer
