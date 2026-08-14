# -*- coding: utf-8 -*-
"""速度分级链路单测（2026-08-14）：跳重排分级 + 单问硬时间预算。

覆盖：
- _skip_rerank_reason 各信号（开关/单候选/分差/关键词锚定）
- _fusion_ordered_reranked 合成分数区间（top1=0.95，关键词块=0.1）
- _rerank_or_skip 分流（时间预算不足跳过、真实重排调用）
- TimeBudget 剩余/可负担/过期
- _budget_synthetic_plan 零 LLM 单方面计划
- generate_answer force_extractive 摘录兜底
- 多角度融合路由：默认检索钩子走融合；retrieve_citations 被替换时保持串行
"""

from __future__ import annotations

import pytest

from backend.app.services import rag_service
from backend.app.services.time_budget import TimeBudget
from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.vector_store_service import VectorSearchResult
from backend.app.services.answer_generation_service import generate_answer
from backend.app.schemas.qa import RetrievalResult
from backend.app.services.query_planner_service import QueryAspect, QueryPlan, QuerySearchQuery


def _candidate(chunk_id: str, metadata: dict | None = None) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="DOC-TEST-0001",
        filename="测试材料.md",
        section_title=None,
        page_number=None,
        text=f"测试片段 {chunk_id}",
        embedding_text=None,
        token_count=4,
        score=0.8,
        chunk_type="paragraph",
        section_path=None,
        section_number=None,
        parent_section_number=None,
        previous_chunk_id=None,
        next_chunk_id=None,
        metadata=metadata,
    )


# ---------------------------------------------------------------- 跳重排判定


def test_skip_rerank_disabled_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", False)
    fusion = {"a": 0.1, "b": 0.01}
    assert rag_service._skip_rerank_reason([_candidate("a"), _candidate("b")], fusion) == ""


def test_skip_rerank_single_candidate(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    assert (
        rag_service._skip_rerank_reason([_candidate("a")], {"a": 0.05})
        == "single_candidate"
    )


def test_skip_rerank_top1_below_floor_stays(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_TOP_FUSION_MIN", 0.10)
    # top1 0.05 < 0.10：即使分差巨大也不跳过（信号不够强）
    fusion = {"a": 0.05, "b": 0.005}
    assert rag_service._skip_rerank_reason([_candidate("a"), _candidate("b")], fusion) == ""


def test_skip_rerank_fusion_margin(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_TOP_FUSION_MIN", 0.01)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_MARGIN_RATIO", 0.30)
    # (0.10-0.04)/0.10 = 0.60 ≥ 0.30 → 跳过
    fusion = {"a": 0.10, "b": 0.04}
    assert (
        rag_service._skip_rerank_reason([_candidate("a"), _candidate("b")], fusion)
        == "fusion_margin"
    )


def test_skip_rerank_margin_too_small_keeps_rerank(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_TOP_FUSION_MIN", 0.01)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_MARGIN_RATIO", 0.30)
    # (0.10-0.09)/0.10 = 0.10 < 0.30 且无关键词锚定 → 真实重排
    fusion = {"a": 0.10, "b": 0.09}
    assert rag_service._skip_rerank_reason([_candidate("a"), _candidate("b")], fusion) == ""


def test_skip_rerank_keyword_anchor(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_TOP_FUSION_MIN", 0.01)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_MARGIN_RATIO", 0.90)
    # 分差不足，但 top1 有 2 次关键词精确命中 → 锚定跳过
    fusion = {"a": 0.10, "b": 0.09}
    candidates = [
        _candidate("a", {"recall_path": "keyword", "keyword_hits": 2}),
        _candidate("b"),
    ]
    assert rag_service._skip_rerank_reason(candidates, fusion) == "keyword_anchor"


def test_fusion_ordered_reranked_scores() -> None:
    candidates = [
        _candidate("top"),
        _candidate("mid"),
        _candidate("kw", {"recall_path": "keyword", "keyword_hits": 3}),
    ]
    fusion = {"top": 0.10, "mid": 0.06, "kw": 0.0}
    ordered = rag_service._fusion_ordered_reranked(candidates, fusion)
    scores = {item.candidate.chunk_id: item.rerank_score for item in ordered}
    # 词面锚定块排最前且给高置信分（0.55+0.15×3 封顶 0.95）
    assert ordered[0].candidate.chunk_id == "kw"
    assert scores["kw"] == pytest.approx(0.95)
    assert scores["top"] == pytest.approx(0.95)
    assert scores["mid"] == pytest.approx(0.05 + 0.90 * 0.6)


# ---------------------------------------------------------------- 重排/跳过分流


def test_rerank_or_skip_time_budget(monkeypatch) -> None:
    """预算不足 → 跳过真实重排（rerank_candidates 不得被调用）。"""
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_TOP_FUSION_MIN", 0.10)
    monkeypatch.setattr(rag_service, "RERANK_BUDGET_SECONDS", 3.0)

    called: list[str] = []

    def fake_rerank(**kwargs):
        called.append("rerank")
        raise AssertionError("预算不足时不应调用真实重排")

    monkeypatch.setattr(rag_service, "rerank_candidates", fake_rerank)
    budget = TimeBudget(1.0)  # 剩余 1s < 3s
    fusion = {"a": 0.05, "b": 0.04}
    reranked, reason = rag_service._rerank_or_skip(
        rerank_question="测试",
        rerank_input=[_candidate("a"), _candidate("b")],
        fusion_scores=fusion,
        budget=budget,
        rerank_limit=6,
    )
    assert reason == "time_budget"
    assert called == []
    assert reranked[0].candidate.chunk_id == "a"


def test_rerank_or_skip_real_rerank(monkeypatch) -> None:
    """无跳过信号且预算充足 → 调用真实重排并原样返回。"""
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    monkeypatch.setattr(rag_service, "SKIP_RERANK_TOP_FUSION_MIN", 0.10)
    monkeypatch.setattr(rag_service, "RERANK_BUDGET_SECONDS", 3.0)

    candidates = [_candidate("a"), _candidate("b")]
    monkeypatch.setattr(
        rag_service,
        "rerank_candidates",
        lambda question, candidates, limit: [
            RerankedChunk(candidate=candidates[1], rerank_score=0.7)
        ],
    )
    reranked, reason = rag_service._rerank_or_skip(
        rerank_question="测试",
        rerank_input=candidates,
        fusion_scores={"a": 0.05, "b": 0.04},
        budget=TimeBudget(60.0),
        rerank_limit=6,
    )
    assert reason == ""
    assert reranked[0].rerank_score == 0.7


# ---------------------------------------------------------------- 时间预算


def test_time_budget_lifecycle() -> None:
    clock = {"now": 0.0}
    budget = TimeBudget(10.0, now_fn=lambda: clock["now"])
    assert budget.remaining() == 10.0
    assert budget.can_afford(3.0)
    assert not budget.expired()
    clock["now"] = 7.5
    assert budget.remaining() == pytest.approx(2.5)
    assert not budget.can_afford(3.0)
    clock["now"] = 10.0
    assert budget.expired()
    assert not budget.can_afford(0.1)


def test_budget_synthetic_plan() -> None:
    plan = rag_service._budget_synthetic_plan("你参与过哪些项目？")
    assert plan.planner == "budget"
    assert len(plan.aspects) == 1
    assert plan.aspects[0].aspect_id == "budget"
    assert plan.aspects[0].search_queries[0].query == "你参与过哪些项目？"


# ---------------------------------------------------------------- 生成兜底


def test_generate_answer_force_extractive(monkeypatch) -> None:
    """force_extractive=True → 不调 LLM，直接摘录知识库原文（hedged）。"""
    monkeypatch.setattr(
        "backend.app.services.answer_generation_service.ANSWER_GENERATION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "backend.app.services.answer_generation_service.ANSWER_GENERATION_API_KEY",
        "test-key",
    )
    monkeypatch.setattr(
        "backend.app.services.answer_generation_service._call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应调用 LLM")),
    )
    chunk = RetrievalResult(
        chunk_id="c1",
        rank=1,
        score=0.8,
        source_doc="项目介绍_AI开放平台.md",
        section_title="Token 发放链路",
        section_path=["Token 发放链路"],
        text="Redis Lua 在单次执行中完成库存校验、限购、防重、预扣与 Stream 投递。",
        citation_label="[1]",
        metadata={},
    )
    generated = generate_answer(
        "AI 开放平台怎么防超卖？",
        [chunk],
        intent="resume_qa",
        force_extractive=True,
    )
    assert generated.answer_mode == "hedged"
    assert "Redis Lua" in (generated.answer or "")
    assert generated.generation_status == "degraded"


def test_generate_answer_timeout_override(monkeypatch) -> None:
    """timeout_override 收紧 LLM 超时并透传给 _call_llm。"""
    monkeypatch.setattr(
        "backend.app.services.answer_generation_service.ANSWER_GENERATION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "backend.app.services.answer_generation_service.ANSWER_GENERATION_API_KEY",
        "test-key",
    )
    seen: dict[str, object] = {}

    def fake_call_llm(question, context_chunks, **kwargs):
        seen["timeout_override"] = kwargs.get("timeout_override")
        raise OSError("模拟 LLM 不可用，走摘录兜底")

    monkeypatch.setattr(
        "backend.app.services.answer_generation_service._call_llm",
        fake_call_llm,
    )
    generate_answer("测试", [], intent="resume_qa", timeout_override=3.5)
    assert seen["timeout_override"] == 3.5


# ---------------------------------------------------------------- 融合路由


def _plan(two_aspects: bool = True) -> QueryPlan:
    aspect_1 = QueryAspect(
        aspect_id="a1",
        question="你参与过哪些项目？",
        search_queries=(QuerySearchQuery("项目经历", "semantic_question"),),
        evidence_need="项目",
        keywords=(),
    )
    aspects = (aspect_1,) if not two_aspects else (
        aspect_1,
        QueryAspect(
            aspect_id="a2",
            question="分别用了哪些技术栈？",
            search_queries=(QuerySearchQuery("技术栈", "semantic_question"),),
            evidence_need="技能",
            keywords=(),
        ),
    )
    return QueryPlan(original_question="你参与过哪些项目？", aspects=aspects, planner="test")


def test_retrieve_aspects_routes_multi_aspect_to_fused(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    calls: list[dict] = []

    def fake_fused(db, query_plan, progress_reporter=None, budget=None, persona_id=None):
        calls.append({"fused": True, "aspects": len(query_plan.aspects)})
        return []

    monkeypatch.setattr(rag_service, "_retrieve_aspects_fused", fake_fused)
    rag_service._retrieve_aspects(None, _plan(two_aspects=True))  # type: ignore[arg-type]
    assert calls == [{"fused": True, "aspects": 2}]


def test_retrieve_aspects_legacy_hook_stays_serial(monkeypatch) -> None:
    monkeypatch.setattr(rag_service, "SKIP_RERANK_ENABLED", True)
    monkeypatch.setattr(rag_service, "retrieve_citations", lambda question: [])

    per_aspect_calls: list[str] = []

    def fake_retrieve_aspect_matches(db, aspect, progress_reporter=None, document_chunk_cache=None, *, enumerative=False, budget=None, persona_id=None):
        per_aspect_calls.append(aspect.aspect_id)
        return [], []

    monkeypatch.setattr(rag_service, "_retrieve_aspect_matches", fake_retrieve_aspect_matches)
    monkeypatch.setattr(rag_service, "_validate_context_chunks", lambda db, chunks: (chunks, {}))
    monkeypatch.setattr(rag_service, "_to_retrieval_results", lambda matches: [])
    rag_service._retrieve_aspects(None, _plan(two_aspects=True))  # type: ignore[arg-type]
    assert per_aspect_calls == ["a1", "a2"]
