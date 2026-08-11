from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core import config
from backend.app.core.database import Base
from backend.app.models.document import Document, DocumentChunk
from backend.app.schemas.qa import RetrievalResult
from backend.app.services.answer_generation_service import GeneratedAnswer, generate_answer
from backend.app.services.intent_router_service import INTENT_RESUME_QA
from backend.app.services.retrieval_service import filter_candidates_by_metadata
from backend.app.services.query_planner_service import QueryAspect, QueryPlan, QuerySearchQuery
from backend.app.services.rag_service import (
    AspectRetrieval,
    _chunk_matches_query_aspect,
    _select_prompt_chunks,
)
from backend.app.services.vector_store_service import VectorSearchResult, _retrieval_filter


def _document(document_id: str, title: str, authority: str) -> Document:
    return Document(
        document_id=document_id,
        filename=f"{title}.pdf",
        filename_norm=f"{title}.pdf".casefold(),
        file_type="pdf",
        size=10,
        storage_path=f"/tmp/{document_id}.pdf",
        title=title,
        issuing_authority=authority,
        status="indexed",
        index_version=config.INDEX_VERSION,
    )


def test_prompt_selection_reuses_one_chunk_for_two_explicit_aspects() -> None:
    aspect = QueryAspect(
        aspect_id="condition_2",
        question="请说明与“除非项目需求变化，否则架构设计不可变更”相关的说明。",
        search_queries=(QuerySearchQuery("架构设计不可变更", "keyword_anchor"),),
        evidence_need="变更例外",
        keywords=("架构设计不可变更",),
    )

    # 同一 chunk 同时是两个 aspect 的候选：选择后通过覆盖同步复用于第二个 aspect
    chunk = RetrievalResult(
        chunk_id="C1",
        rank=1,
        source_doc="材料.pdf",
        text="不得以时间紧张为由变更架构；除非项目需求变化，否则架构设计不可变更。",
        citation_label="[1]",
        score=0.8,
        metadata={"rerank_score": 0.9, "document_id": "DOC-1"},
    )
    aspect_a = QueryAspect(
        aspect_id="condition_1",
        question="请说明与“不得以时间紧张为由变更架构”相关的说明。",
        search_queries=(QuerySearchQuery("架构设计不可变更", "keyword_anchor"),),
        evidence_need="变更原则",
        keywords=("架构设计",),
    )
    retrievals = [
        AspectRetrieval(
            aspect=aspect,
            candidates=[chunk],
            diagnostics=[],
            citation_validation={"valid_chunks": 1},
            selected_chunk_ids=[],
            retrieval_covered=True,
        )
        for aspect in (aspect_a, aspect)
    ]
    plan = QueryPlan(
        original_question="架构变更的原则与例外是什么？",
        aspects=(aspect_a, aspect),
        planner="test",
    )
    selected, summary = _select_prompt_chunks(plan.original_question, plan, retrievals)

    assert [item.chunk_id for item in selected] == ["C1"]  # 共享块只选一次
    assert summary["covered_aspects"] == ["condition_1", "condition_2"]
    assert summary["final_prompt_chunks"] == 1


@dataclass
class _MatchValue:
    value: object


@dataclass
class _MatchAny:
    any: list[str]


@dataclass
class _FieldCondition:
    key: str
    match: object


@dataclass
class _Filter:
    must: list[object]


class _Models:
    MatchValue = _MatchValue
    MatchAny = _MatchAny
    FieldCondition = _FieldCondition
    Filter = _Filter


def test_qdrant_filter_applies_same_scope_to_prefetch() -> None:
    compiled = _retrieval_filter(
        _Models,
        {"document_ids": ["DOC-1", "DOC-2"]},
    )
    conditions = {item.key: item.match for item in compiled.must}
    assert conditions["document_id"].any == ["DOC-1", "DOC-2"]
    assert "index_version" in conditions


def test_active_document_metadata_repairs_legacy_vector_payload(monkeypatch) -> None:
    from backend.app.services import retrieval_service as service

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(_document("DOC-PDF", "证书核实说明", "软考办"))
        db.commit()
    monkeypatch.setattr(service, "SessionLocal", session_factory)
    candidate = VectorSearchResult(
        chunk_id="CHUNK-1",
        document_id="DOC-PDF",
        filename="证书核实说明.pdf",
        section_title=None,
        page_number=1,
        text="核实时限",
        embedding_text="核实时限",
        token_count=4,
        score=0.9,
        metadata={},
    )

    active = service.filter_active_candidates([candidate])
    filtered = filter_candidates_by_metadata(active, {"file_type": "pdf"})

    assert filtered == [candidate]
    assert candidate.metadata["source_format"] == "pdf"


def test_document_snapshot_cache_reuses_and_invalidates_chunk_snapshot() -> None:
    from backend.app.services import rag_service
    from backend.app.models.document import DocumentChunk

    class _Result:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class _DB:
        def __init__(self, rows):
            self.rows = rows
            self.calls = 0

        def scalars(self, _statement):
            self.calls += 1
            return _Result(self.rows)

    chunk = DocumentChunk(
        id=901,
        chunk_id="CACHE-1",
        document_id="DOC-CACHE",
        text="缓存测试",
        embedding_text="缓存测试",
        index_status="indexed",
        source_file="cache.pdf",
    )
    rag_service.clear_document_snapshot_cache()
    db = _DB([chunk])

    first = rag_service._chunks_by_document(db, {"DOC-CACHE"}, document_chunk_cache={})
    second = rag_service._chunks_by_document(db, {"DOC-CACHE"}, document_chunk_cache={})
    assert db.calls == 1
    assert first["DOC-CACHE"][0].chunk_id == second["DOC-CACHE"][0].chunk_id

    rag_service.clear_document_snapshot_cache({"DOC-CACHE"})
    rag_service._chunks_by_document(db, {"DOC-CACHE"}, document_chunk_cache={})
    assert db.calls == 2


def test_mixed_answer_returns_llm_answer_unchanged(monkeypatch) -> None:
    from backend.app.services import answer_generation_service as service

    chunks = [
        RetrievalResult(
            chunk_id="TABLE-1",
            rank=1,
            source_doc="report.xlsx",
            text="A1=10",
            citation_label="[1]",
            metadata={
                "dynamic_table_evidence": True,
                "value": 10,
                "coordinate": "A1",
                "aspect_id": "table",
                "evidence_role": "direct_evidence",
            },
        ),
        RetrievalResult(
            chunk_id="MATERIAL-1",
            rank=2,
            source_doc="material.pdf",
            text="该成绩应当如实说明。",
            citation_label="[2]",
            metadata={"aspect_id": "material", "evidence_role": "direct_evidence"},
        ),
    ]

    def fake_call(*_args, **_kwargs):
        return GeneratedAnswer(
            answer="该成绩应当如实说明。",
            evidence_sufficiency="sufficient",
            generation_status="completed",
        )

    monkeypatch.setattr(service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(service, "ANSWER_GENERATION_API_KEY", "test")
    monkeypatch.setattr(service, "_call_llm", fake_call)
    result = generate_answer(
        "结合材料解释表格值",
        chunks,
        intent=INTENT_RESUME_QA,
    )
    assert result.answer_mode == "answered"
    assert result.evidence_sufficiency == "sufficient"
    assert result.generation_status == "completed"
    assert result.answer == "该成绩应当如实说明。"
    assert not result.answer.startswith("根据现有知识库推测")


def test_prompt_core_prefers_explicit_condition_match_over_generic_higher_score() -> None:
    aspect = QueryAspect(
        aspect_id="audit_requirement",
        question="请说明与“简历材料应每年对项目介绍开展更新维护”相关的说明。",
        search_queries=(QuerySearchQuery("更新维护", "keyword_anchor", "test"),),
        evidence_need="更新维护频率和留档要求",
        keywords=("简历材料", "更新维护", "留档备查"),
    )
    generic = RetrievalResult(
        chunk_id="GENERIC",
        rank=1,
        score=0.99,
        source_doc="material.pdf",
        text="简历材料应建立更新维护机制并完善审核流程。",
        citation_label="[1]",
        metadata={},
    )
    exact = RetrievalResult(
        chunk_id="EXACT",
        rank=2,
        score=0.8,
        source_doc="material.pdf",
        text="简历材料应每年对项目介绍开展更新维护，更新维护结果需留档备查。",
        citation_label="[2]",
        metadata={},
    )

    selected, _summary = _select_prompt_chunks(
        aspect.question,
        QueryPlan(original_question=aspect.question, aspects=(aspect,), planner="test"),
        [
            AspectRetrieval(
                aspect=aspect,
                candidates=[generic, exact],
                diagnostics=[],
                citation_validation={"valid_chunks": 2},
                selected_chunk_ids=[],
                retrieval_covered=True,
            )
        ],
    )

    # 引号锚点命中的块（EXACT）优先于高分泛化块（GENERIC）：EXACT 必须是 aspect 的核心块
    # （final 排序按分数/章节重排，GENERIC 会作为补位块排在前面，因此断言核心归属而非顺序）
    core_chunks = [
        chunk
        for chunk in selected
        if chunk.metadata.get("prompt_selection_reason") == "core"
    ]
    assert core_chunks and core_chunks[0].chunk_id == "EXACT"
    assert {chunk.chunk_id for chunk in selected} == {"GENERIC", "EXACT"}
