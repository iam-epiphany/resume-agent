# -*- coding: utf-8 -*-
"""公开出处测试（2026-08-14）：匿名视角剥离内部上下文，只保留安全 citations。"""

from __future__ import annotations

from backend.app.api.qa import _build_public_citations, _strip_context_package
from backend.app.schemas.qa import (
    LLMContextPackage,
    QAResponse,
    QATaskStatusResponse,
    RetrievalResult,
)


def _chunk(source_doc: str, text: str, fact_status: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="CHUNK-1",
        rank=1,
        score=0.85,
        source_doc=source_doc,
        section_title="测试章节",
        section_path=["测试章节"],
        text=text,
        citation_label="[1]",
        metadata={"fact_status": fact_status} if fact_status else {},
    )


def _package() -> LLMContextPackage:
    return LLMContextPackage(
        query="测试",
        mode="rag_context",
        instruction="内部指令（不得外泄）",
        retrieval_summary={"internal": True},
        context_chunks=[_chunk("项目介绍_AI开放平台.md", "Redis Lua 原子预扣库存。", "confirmed")],
        llm_prompt="内部提示词（不得外泄）",
    )


def test_build_public_citations_keeps_only_safe_fields() -> None:
    citations = _build_public_citations(_package())
    assert len(citations) == 1
    citation = citations[0]
    assert citation.source_doc == "项目介绍_AI开放平台.md"
    assert citation.section_title == "测试章节"
    assert citation.excerpt == "Redis Lua 原子预扣库存。"
    assert citation.score == 0.85
    assert citation.fact_status == "confirmed"
    # 内部字段不存在于公开模型
    assert not hasattr(citation, "llm_prompt")
    assert not hasattr(citation, "chunk_id")


def test_build_public_citations_excerpt_capped_at_80_chars() -> None:
    package = _package()
    package.context_chunks = [_chunk("长文.md", "很" * 200)]
    citations = _build_public_citations(package)
    assert len(citations[0].excerpt) == 80


def test_strip_context_package_builds_citations_and_clears_internal() -> None:
    response = QAResponse(answer="回答", context_package=_package())
    stripped = _strip_context_package(response)
    assert isinstance(stripped, QAResponse)
    assert stripped.context_package is None
    assert stripped.citations is not None
    assert stripped.citations[0].source_doc == "项目介绍_AI开放平台.md"


def test_strip_context_package_preserves_admin_full_view() -> None:
    """管理员不经剥离——citations 保持 None、context_package 完整。"""
    response = QAResponse(answer="回答", context_package=_package())
    assert response.context_package is not None
    assert response.citations is None


def test_strip_task_status_response() -> None:
    task = QATaskStatusResponse(
        task_id="t1",
        question="问题",
        status="completed",
        answer=QAResponse(answer="回答", context_package=_package()),
        created_at="2026-08-14T00:00:00Z",
        updated_at="2026-08-14T00:00:01Z",
    )
    stripped = _strip_context_package(task)
    assert isinstance(stripped, QATaskStatusResponse)
    assert stripped.answer is not None
    assert stripped.answer.context_package is None
    assert stripped.answer.citations is not None
