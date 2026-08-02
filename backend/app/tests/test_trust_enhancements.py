from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core import config
from backend.app.core.database import Base
from backend.app.models.document import Document, DocumentChunk
from backend.app.schemas.qa import RetrievalResult
from backend.app.services.answer_generation_service import GeneratedAnswer, generate_answer
from backend.app.services.document_identity_query_service import answer_document_identity_question
from backend.app.services.intent_router_service import INTENT_RESUME_DETAIL
from backend.app.services.retrieval_service import filter_candidates_by_metadata
from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery
from backend.app.services.rag_service import (
    _best_non_duplicate_candidate,
    _best_shared_candidate,
    _chunk_matches_query_aspect,
    _normalize_exact_support_text,
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
        version_status="current",
        status="indexed",
        index_version=config.INDEX_VERSION,
    )


def test_prompt_selection_reuses_one_chunk_for_two_explicit_aspects() -> None:
    shared = RetrievalResult(
        chunk_id="C1",
        rank=1,
        source_doc="材料.pdf",
        text="不得以时间紧张为由变更架构；除非项目需求变化，否则架构设计不可变更。",
        citation_label="[1]",
        score=0.8,
        metadata={"rerank_score": 0.9},
    )
    aspect = QueryAspect(
        aspect_id="condition_2",
        question="请说明与“除非项目需求变化，否则架构设计不可变更”相关的说明。",
        search_queries=(QuerySearchQuery("架构设计不可变更", "keyword_anchor"),),
        evidence_need="变更例外",
        keywords=("架构设计不可变更",),
    )

    reused = _best_shared_candidate([shared], [shared], aspect)

    assert reused is shared


def test_exact_support_normalization_unifies_common_project_aliases() -> None:
    left = _normalize_exact_support_text(
        "高并发电商秒杀平台 使用 Redis 预扣库存"
    )
    right = _normalize_exact_support_text(
        "秒杀平台 使用 Redis 预扣库存"
    )

    assert left == right


def test_prompt_aspect_gate_preserves_bounded_lexical_support() -> None:
    aspect = QueryAspect(
        aspect_id="material_basis",
        question="说明项目综合评分",
        search_queries=(QuerySearchQuery("项目综合评分", "semantic_question"),),
        evidence_need="项目综合评分依据",
        keywords=("并不逐字出现的长问题",),
    )
    chunk = RetrievalResult(
        chunk_id="DOC-TEST-CHUNK-0001",
        rank=1,
        score=0.9,
        source_doc="test.docx",
        section_title=None,
        section_path=[],
        text="原文直接支持该项目的综合评分。",
        citation_label="[1]",
        metadata={"evidence_role": "bounded_lexical_support"},
    )

    assert _chunk_matches_query_aspect(chunk, aspect)


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
        {"document_ids": ["DOC-1", "DOC-2"], "article_number": "第十条"},
    )
    conditions = {item.key: item.match for item in compiled.must}
    assert conditions["document_id"].any == ["DOC-1", "DOC-2"]
    assert conditions["article_number"].value == "第十条"
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
        intent=INTENT_RESUME_DETAIL,
    )
    assert result.answer_mode == "answered"
    assert result.evidence_sufficiency == "sufficient"
    assert result.generation_status == "completed"
    assert result.answer == "该成绩应当如实说明。"
    assert not result.answer.startswith("根据现有知识库推测")


def test_document_identity_question_returns_explicit_unknown_without_validity_inference() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        document = _document("DOC-IDENTITY-1", "技能掌握情况表", "")
        document.filename = "附件1：技能掌握情况表.doc"
        document.filename_norm = document.filename.casefold()
        document.file_sha256 = "a" * 64
        document.version_status = "unknown"
        document.identity_review_status = "unreviewed"
        db.add(document)
        db.commit()
        response = answer_document_identity_question(
            db,
            "根据知识库中的材料信息，能否确认《附件1：技能掌握情况表》的真实性？",
        )
    assert response is not None
    assert response.answer_mode == "answered"
    assert response.generation_status == "deterministic"
    assert "已收录在知识库材料中" in str(response.answer)


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

    selected = _best_non_duplicate_candidate([generic, exact], [], aspect)

    assert selected is exact
