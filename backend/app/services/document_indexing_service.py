from sqlalchemy import select
import json
from sqlalchemy.orm import Session
from typing import Callable

from backend.app.core.config import INDEX_VERSION
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.chunk_service import ChunkDraft, build_contextual_embedding_text, count_tokens
from backend.app.services.embedding_service import EmbeddingServiceError, embed_texts
from backend.app.services.vector_store_service import (
    VectorStoreError,
    count_document_vectors,
    delete_document_vectors,
    upsert_chunk_embeddings,
)
from backend.app.services.performance_metrics import measure
from backend.app.services.rerank_service import invalidate_rerank_score_cache
from backend.app.services.qa_cache_service import clear as clear_qa_answer_cache
from backend.app.services.document_metadata_service import retrieval_metadata_snapshot


class DocumentIndexingError(RuntimeError):
    pass


def index_document(
    db: Session,
    document: Document,
    *,
    stage_reporter: Callable[[str, int | None, int | None], None] | None = None,
) -> None:
    if document.status in {"deleting", "delete_failed"}:
        raise DocumentIndexingError("文档正在删除或删除失败待重试，不能构建向量索引")

    chunks = db.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document.document_id).order_by(DocumentChunk.id.asc())
    ).all()
    if not chunks:
        raise DocumentIndexingError("文档没有可索引的 chunk")

    with measure("index.embedding_text_build"):
        drafts = _to_chunk_drafts(chunks, document)
    chunk_ids = [draft.chunk_id for draft in drafts]
    document.status = "indexing"
    document.index_version = INDEX_VERSION
    document.index_error = None
    for chunk, draft in zip(chunks, drafts, strict=True):
        chunk.embedding_text = draft.embedding_text
        chunk.token_count = draft.token_count
        chunk.chunk_metadata = json.dumps(draft.metadata or {}, ensure_ascii=False)
        chunk.index_status = "indexing"
        chunk.index_version = INDEX_VERSION
    with measure("index.sqlite_state_persist"):
        db.commit()
    try:
        if stage_reporter is not None:
            stage_reporter("embedding", 0, len(drafts))
        with measure("index.document_embedding"):
            embeddings = embed_texts([chunk.embedding_text for chunk in drafts])
        if stage_reporter is not None:
            stage_reporter("embedding", len(drafts), len(drafts))
            stage_reporter("vector_upsert", 0, len(drafts))
        with measure("index.vector_upsert"):
            delete_document_vectors(document.document_id)
            upsert_chunk_embeddings(chunks=drafts, embeddings=embeddings, filename=document.filename)
        if stage_reporter is not None:
            stage_reporter("vector_upsert", len(drafts), len(drafts))
            stage_reporter("verifying", 0, len(drafts))
        with measure("index.vector_verify"):
            vector_count = count_document_vectors(document.document_id)
        if vector_count < len(drafts):
            raise VectorStoreError(f"Qdrant 向量数量不完整：expected={len(drafts)}, actual={vector_count}")
        if stage_reporter is not None:
            stage_reporter("verifying", vector_count, len(drafts))
    except (EmbeddingServiceError, VectorStoreError) as exc:
        _cleanup_partial_vectors(document.document_id, chunk_ids)
        failed_chunks = _load_chunks_by_ids(db, document.document_id, chunk_ids)
        _mark_index_failed(db, document, failed_chunks, str(exc))
        raise DocumentIndexingError(str(exc)) from exc

    try:
        chunks = _load_chunks_by_ids(db, document.document_id, chunk_ids)
        if len(chunks) != len(chunk_ids):
            raise DocumentIndexingError(
                f"SQLite chunk 数量在向量写入后发生变化：expected={len(chunk_ids)}, actual={len(chunks)}"
            )
        document.status = "indexed"
        document.index_version = INDEX_VERSION
        document.index_error = None
        for chunk in chunks:
            chunk.index_status = "indexed"
            chunk.index_version = INDEX_VERSION
        with measure("index.sqlite_finalize"):
            db.commit()
        from backend.app.services.rag_service import clear_document_snapshot_cache

        clear_document_snapshot_cache({document.document_id})
        invalidate_rerank_score_cache()
        # 知识库内容变更 → 问答答案缓存整体失效（答案基于旧知识库生成）
        clear_qa_answer_cache()
    except Exception as exc:
        db.rollback()
        _cleanup_partial_vectors(document.document_id, chunk_ids)
        raise DocumentIndexingError(f"SQLite 索引状态提交失败：{exc}") from exc


def _to_chunk_drafts(chunks: list[DocumentChunk], document: Document) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    document_metadata = retrieval_metadata_snapshot(document)
    title_by_section_number: dict[str, str] = {}
    for chunk in chunks:
        section_number = _section_number(chunk.section_title)
        if section_number and chunk.section_title:
            title_by_section_number[section_number] = chunk.section_title

    for previous, chunk, next_chunk in zip([None, *chunks[:-1]], chunks, [*chunks[1:], None], strict=True):
        section_number = _section_number(chunk.section_title)
        parent_section_number = _parent_section_number(section_number)
        section_path = []
        if parent_section_number and parent_section_number in title_by_section_number:
            section_path.append(title_by_section_number[parent_section_number])
        if chunk.section_title:
            section_path.append(chunk.section_title)
        chunk_type = _chunk_type(chunk.text)
        chunk_metadata = {**document_metadata, **_chunk_metadata(chunk)}
        token_count = chunk.token_count or count_tokens(chunk.text)
        embedding_text = build_contextual_embedding_text(
            text=chunk.text,
            source_file=chunk.source_file or document.filename,
            source_format=document.file_type,
            section_title=chunk.section_title,
            section_path=section_path,
            section_number=section_number,
            parent_section_number=parent_section_number,
            page_number=chunk.page_number,
            chunk_type=chunk_type,
            chunk_metadata=chunk_metadata,
        )
        drafts.append(
            ChunkDraft(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                embedding_text=embedding_text,
                token_count=token_count,
                title=chunk.title,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                chunk_type=chunk_type,
                section_path=section_path,
                section_number=section_number,
                parent_section_number=parent_section_number,
                previous_chunk_id=previous.chunk_id if previous else None,
                next_chunk_id=next_chunk.chunk_id if next_chunk else None,
                metadata=chunk_metadata,
            )
        )
    return drafts


def _chunk_type(text: str) -> str:
    stripped = text.lstrip()
    # 表格场景已移除，仅保留 markdown 表格（"\n|" 表格行）作为段落类型提示
    return "table" if "\n|" in stripped else "paragraph"


def _chunk_metadata(chunk: DocumentChunk) -> dict:
    if not chunk.chunk_metadata:
        return {}
    try:
        value = json.loads(chunk.chunk_metadata)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _load_chunks_by_ids(db: Session, document_id: str, chunk_ids: list[str]) -> list[DocumentChunk]:
    if not chunk_ids:
        return []
    chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id, DocumentChunk.chunk_id.in_(chunk_ids))
        .order_by(DocumentChunk.id.asc())
    ).all()
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def _section_number(section_title: str | None) -> str | None:
    if not section_title:
        return None
    import re

    match = re.match(r"^\s*(\d+(?:\.\d+)*)[\.、\s]", section_title)
    return match.group(1) if match else None


def _parent_section_number(section_number: str | None) -> str | None:
    if not section_number or "." not in section_number:
        return None
    return section_number.rsplit(".", maxsplit=1)[0]


def _mark_index_failed(db: Session, document: Document, chunks: list[DocumentChunk], error: str) -> None:
    document.status = "index_failed"
    document.index_error = error[:1000]
    document.index_version = INDEX_VERSION
    for chunk in chunks:
        chunk.index_status = "index_failed"
        chunk.index_version = INDEX_VERSION
    db.commit()


def _cleanup_partial_vectors(document_id: str, chunk_ids: list[str]) -> None:
    try:
        delete_document_vectors(document_id, chunk_ids=chunk_ids)
    except VectorStoreError:
        pass
