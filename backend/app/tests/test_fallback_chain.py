"""Tests for the retrieval fallback chain implemented in rag_service.answer_question (P3).

Chain levels (QAResponse.retrieval_fallback_level):
- 0: standard retrieval produced strong evidence
- 1: evidence weak -> relaxed re-selection (build_context_package(relaxed=True),
  filling up to RELAXED_MIN_PROMPT_CHUNKS by rerank score, bypassing gates)
- 2: still weak after relaxed -> LLM query rewrite (rewrite_search_queries ->
  build_context_package(rewritten_queries=..., relaxed=True))
- 3: no evidence -> direct generation with an empty context package (hedged)

Config gates: FALLBACK_LOWER_THRESHOLD_ENABLED / FALLBACK_REWRITE_RETRY_ENABLED /
FALLBACK_DIRECT_GENERATION_ENABLED. 面试部署默认关闭后二者，本文件会在需要时显式开启。

The retrieval and LLM layers are mocked; no network calls are made.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import HEDGE_PREFIX
from backend.app.core.database import Base
from backend.app.models.document import Document, DocumentChunk
from backend.app.schemas.qa import RetrievalResult
from backend.app.services.answer_generation_service import GeneratedAnswer
from backend.app.services.query_planner_service import QueryAspect, QueryPlan, QuerySearchQuery
from backend.app.services.rag_service import AspectRetrieval, answer_question

DOCUMENT_ID = "DOC-FALLBACK-0001"
CHUNK_PREFIX = f"{DOCUMENT_ID}-CHUNK-"
QUESTION = "请介绍你参与项目的技能亮点"
REWRITTEN_QUERIES = ["技术栈 项目 技能", "项目亮点 深度"]


def _chunk(index: int, score: float) -> RetrievalResult:
    """Craft a RetrievalResult; _prompt_score reads metadata['rerank_score'] first."""
    return RetrievalResult(
        chunk_id=f"{CHUNK_PREFIX}{index:04d}",
        rank=index,
        score=score,
        source_doc="技能专长.md",
        section_title="项目技能",
        section_path=["项目技能"],
        text=f"项目技能片段 {index}：候选人在项目中负责核心技能实现。",
        citation_label=f"[{index}]",
        metadata={"document_id": DOCUMENT_ID, "rerank_score": score},
    )


def _aspect_retrieval(aspect: QueryAspect, chunks: list[RetrievalResult]) -> AspectRetrieval:
    return AspectRetrieval(
        aspect=aspect,
        candidates=chunks,
        diagnostics=[],
        citation_validation={
            "checked_chunks": len(chunks),
            "valid_chunks": len(chunks),
            "invalid_chunks": 0,
            "invalid_chunk_ids": [],
        },
        selected_chunk_ids=[],
        retrieval_covered=bool(chunks),
    )


def _plan(question: str) -> QueryPlan:
    aspect = QueryAspect(
        aspect_id="aspect_1",
        question=question,
        search_queries=(QuerySearchQuery(question, "semantic_question"),),
        evidence_need="相关材料依据",
        keywords=("技能", "项目"),
    )
    return QueryPlan(original_question=question, aspects=(aspect,), planner="rule")


class _FakeRetrieval:
    """Scriptable _retrieve_aspects stand-in; records every aspect it is asked for."""

    def __init__(self, standard: list[RetrievalResult], rewritten: list[RetrievalResult] | None = None):
        self.standard = standard
        self.rewritten = rewritten if rewritten is not None else standard
        self.seen_aspects: list[tuple[str, list[str]]] = []

    def __call__(self, db, query_plan: QueryPlan, progress_reporter=None, budget=None) -> list[AspectRetrieval]:
        aspect = query_plan.aspects[0]
        self.seen_aspects.append(
            (aspect.aspect_id, [search_query.query for search_query in aspect.search_queries])
        )
        chunks = self.rewritten if aspect.aspect_id == "rewritten" else self.standard
        return [_aspect_retrieval(aspect, chunks)]


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'fallback_chain.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def _seed_chunks(db, count: int = 16) -> None:
    """Seed an indexed document + chunks so _validate_context_chunks accepts crafted ids."""
    db.add(
        Document(
            document_id=DOCUMENT_ID,
            filename="技能专长.md",
            file_type="md",
            size=128,
            storage_path="技能专长.md",
            status="indexed",
            chunk_count=count,
        )
    )
    for index in range(1, count + 1):
        db.add(
            DocumentChunk(
                chunk_id=f"{CHUNK_PREFIX}{index:04d}",
                document_id=DOCUMENT_ID,
                text=f"项目技能片段 {index}：候选人在项目中负责核心技能实现。",
                index_status="indexed",
                source_file="技能专长.md",
                token_count=12,
            )
        )
    db.commit()


@pytest.fixture()
def llm_calls(monkeypatch):
    """Turn on the LLM generation path and record every _call_llm invocation."""
    calls: list[dict] = []

    def fake_call(question, context_chunks, **kwargs):
        calls.append({"question": question, "context_chunks": list(context_chunks)})
        return GeneratedAnswer(
            answer="可能参与了项目核心开发。",
            evidence_sufficiency="partial",
            hedge_note="测试：证据不充分，按推测回答",
            generation_status="completed",
        )

    monkeypatch.setattr(
        "backend.app.services.answer_generation_service.ANSWER_GENERATION_ENABLED", True
    )
    monkeypatch.setattr(
        "backend.app.services.answer_generation_service.ANSWER_GENERATION_API_KEY", "test-key"
    )
    monkeypatch.setattr("backend.app.services.answer_generation_service._call_llm", fake_call)
    return calls


@pytest.fixture()
def rewritten_calls(monkeypatch):
    """Disable real intent LLM routing / planning / rewrite, returning fixed rewritten queries."""
    calls: list[str] = []

    def fake_rewrite(question, intent=None, *, catalog=None, cancellation_checker=None):
        calls.append(question)
        return list(REWRITTEN_QUERIES)

    monkeypatch.setattr("backend.app.services.intent_router_service.INTENT_ROUTER_ENABLED", False)
    monkeypatch.setattr(
        "backend.app.services.rag_service.plan_query",
        lambda question, *, catalog=None, memory_context=None, cancellation_checker=None: _plan(question),
    )
    monkeypatch.setattr("backend.app.services.rag_service.rewrite_search_queries", fake_rewrite)
    return calls


@pytest.fixture()
def install_retrieval(monkeypatch):
    """Install a scriptable _retrieve_aspects; returns the _FakeRetrieval spy."""

    def _install(standard, rewritten=None) -> _FakeRetrieval:
        retrieval = _FakeRetrieval(standard, rewritten)
        monkeypatch.setattr("backend.app.services.rag_service._retrieve_aspects", retrieval)
        return retrieval

    return _install


def test_strong_evidence_stays_at_level_0(
    db_session, rewritten_calls, install_retrieval, llm_calls
) -> None:
    _seed_chunks(db_session)
    strong_chunks = [_chunk(index, 0.6 - 0.02 * (index - 1)) for index in range(1, 7)]
    install_retrieval(standard=strong_chunks)

    response = answer_question(db_session, QUESTION, include_debug=True)

    assert response.retrieval_fallback_level == 0
    assert len(response.context_package.context_chunks) == 6
    assert all(chunk.score >= 0.5 for chunk in response.context_package.context_chunks)
    assert rewritten_calls == []  # no relaxed / rewrite stage entered


def test_weak_evidence_triggers_relaxed_reselection_level_1(
    db_session, rewritten_calls, install_retrieval, llm_calls
) -> None:
    """Low-rerank chunks clear the core pass via the lexical escape but grade weak
    (single chunk), so the relaxed pass fills the prompt to 6 by rerank score."""
    _seed_chunks(db_session)
    weak_chunks = [_chunk(index, 0.06) for index in range(1, 7)]
    install_retrieval(standard=weak_chunks)

    response = answer_question(db_session, QUESTION, include_debug=True)

    assert response.retrieval_fallback_level == 1
    chunks = response.context_package.context_chunks
    assert len(chunks) == 6
    assert any(
        chunk.metadata.get("prompt_selection_reason") == "relaxed_fallback" for chunk in chunks
    )
    assert rewritten_calls == []  # relaxed package now grades strong; rewrite not needed


def test_still_weak_after_relaxed_triggers_rewrite_level_2(
    monkeypatch, db_session, rewritten_calls, install_retrieval, llm_calls
) -> None:
    """Only 2 candidates: relaxed fills to 2, still weak -> rewrite + re-retrieve."""
    _seed_chunks(db_session)
    monkeypatch.setattr("backend.app.services.rag_service.FALLBACK_REWRITE_RETRY_ENABLED", True)
    weak_chunks = [_chunk(index, 0.06) for index in range(1, 3)]
    strong_rewritten = [_chunk(index, 0.6) for index in range(9, 15)]
    retrieval = install_retrieval(standard=weak_chunks, rewritten=strong_rewritten)

    response = answer_question(db_session, QUESTION, include_debug=True)

    assert response.retrieval_fallback_level == 2
    assert rewritten_calls == [QUESTION]
    assert ("aspect_1", [QUESTION]) in retrieval.seen_aspects
    # The rewrite stage must have re-retrieved with the rewritten queries.
    assert ("rewritten", REWRITTEN_QUERIES) in retrieval.seen_aspects
    assert len(response.context_package.context_chunks) == 6


def test_no_evidence_uses_direct_generation_level_3(
    monkeypatch, db_session, rewritten_calls, install_retrieval, llm_calls
) -> None:
    _seed_chunks(db_session)
    monkeypatch.setattr("backend.app.services.rag_service.FALLBACK_DIRECT_GENERATION_ENABLED", True)
    install_retrieval(standard=[])

    response = answer_question(db_session, QUESTION, include_debug=True)

    assert response.retrieval_fallback_level == 3
    assert response.context_package.context_chunks == []
    assert response.answer_mode == "hedged"
    assert response.answer.startswith(f"{HEDGE_PREFIX}，")
    assert llm_calls and llm_calls[-1]["context_chunks"] == []


def test_lower_threshold_gate_disabled_skips_relaxed_level(
    monkeypatch, db_session, rewritten_calls, install_retrieval, llm_calls
) -> None:
    monkeypatch.setattr(
        "backend.app.services.rag_service.FALLBACK_LOWER_THRESHOLD_ENABLED", False
    )
    _seed_chunks(db_session)
    install_retrieval(standard=[_chunk(index, 0.06) for index in range(1, 3)])

    response = answer_question(db_session, QUESTION, include_debug=True)

    assert response.retrieval_fallback_level == 0
    assert len(response.context_package.context_chunks) == 1  # standard core chunk only
    assert rewritten_calls == []


def test_rewrite_retry_gate_disabled_stops_at_level_1(
    monkeypatch, db_session, rewritten_calls, install_retrieval, llm_calls
) -> None:
    monkeypatch.setattr(
        "backend.app.services.rag_service.FALLBACK_REWRITE_RETRY_ENABLED", False
    )
    _seed_chunks(db_session)
    install_retrieval(standard=[_chunk(index, 0.06) for index in range(1, 3)])

    response = answer_question(db_session, QUESTION, include_debug=True)

    assert response.retrieval_fallback_level == 1
    assert len(response.context_package.context_chunks) == 2  # relaxed filled to pool size
    assert rewritten_calls == []


def test_direct_generation_gate_disabled_keeps_level_0_with_empty_context(
    monkeypatch, db_session, rewritten_calls, install_retrieval, llm_calls
) -> None:
    monkeypatch.setattr(
        "backend.app.services.rag_service.FALLBACK_DIRECT_GENERATION_ENABLED", False
    )
    _seed_chunks(db_session)
    install_retrieval(standard=[])

    response = answer_question(db_session, QUESTION, include_debug=True)

    assert response.retrieval_fallback_level == 0
    assert response.context_package.context_chunks == []
    assert llm_calls and llm_calls[-1]["context_chunks"] == []  # LLM still invoked
    assert response.answer_mode == "hedged"
    assert response.answer.startswith(f"{HEDGE_PREFIX}，")
