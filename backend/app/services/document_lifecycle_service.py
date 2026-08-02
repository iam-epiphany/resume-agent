from pathlib import Path, PureWindowsPath
import hashlib
import json
import unicodedata

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.app.core.config import DOCUMENT_DIR, INDEX_VERSION, SUPPORTED_DOCUMENT_EXTENSIONS
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.chunk_service import build_chunks_from_parsed
from backend.app.services.document_parser import DocumentParseError, parse_document
from backend.app.services.vector_store_service import (
    VectorStoreError,
    delete_document_vectors,
    document_vector_chunk_ids,
)
from backend.app.services.rerank_service import invalidate_rerank_score_cache


class DocumentDeletionError(RuntimeError):
    pass


def original_file_exists(document: Document) -> bool:
    return resolve_original_path(document).exists()


def resolve_original_path(document: Document) -> Path:
    storage_path = Path(document.storage_path)
    if storage_path.exists():
        return storage_path
    fallback = DOCUMENT_DIR / _storage_basename(document.storage_path)
    return fallback


def _storage_basename(storage_path: str) -> str:
    """Return the file name even when a Windows path is read inside Linux."""

    if "\\" in storage_path or ":" in storage_path:
        return PureWindowsPath(storage_path).name
    return Path(storage_path).name


def delete_document_record(db: Session, document: Document) -> None:
    """Delete Qdrant vectors first, then original file and SQLite rows.

    Qdrant is not transactional with SQLite. If vector cleanup fails, keep the
    document visible as delete_failed so the user can retry instead of leaving
    hidden orphan vectors behind.
    """

    chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)).all()
    document.status = "deleting"
    document.lifecycle_stage = "deleting_vectors"
    document.index_error = None
    for chunk in chunks:
        chunk.index_status = "deleting"
    db.commit()

    try:
        delete_document_vectors(document.document_id, chunk_ids=[chunk.chunk_id for chunk in chunks])
    except VectorStoreError as exc:
        _mark_delete_failed(db, document, chunks, "deleting_vectors", str(exc))
        raise DocumentDeletionError(str(exc)) from exc

    document.lifecycle_stage = "deleting_file"
    db.commit()
    try:
        storage_path = resolve_original_path(document)
        if storage_path.exists():
            storage_path.unlink()
    except OSError as exc:
        _mark_delete_failed(db, document, chunks, "deleting_file", str(exc))
        raise DocumentDeletionError(f"原文件删除失败：{exc}") from exc

    document.lifecycle_stage = "deleting_metadata"
    db.commit()
    try:
        db.delete(document)
        db.commit()
        from backend.app.services.rag_service import clear_document_snapshot_cache

        clear_document_snapshot_cache({document.document_id})
        invalidate_rerank_score_cache()
    except Exception as exc:
        db.rollback()
        current = db.scalar(
            select(Document).where(Document.document_id == document.document_id)
        )
        if current is not None:
            current_chunks = db.scalars(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == current.document_id
                )
            ).all()
            _mark_delete_failed(
                db,
                current,
                current_chunks,
                "deleting_metadata",
                str(exc),
            )
        raise DocumentDeletionError(f"元数据删除失败：{exc}") from exc


def remove_documents_with_missing_originals(db: Session) -> list[tuple[str, str]]:
    removed: list[tuple[str, str]] = []
    documents = db.scalars(select(Document).order_by(Document.uploaded_at.asc())).all()
    for document in documents:
        if original_file_exists(document):
            continue
        document_id = document.document_id
        document.status = "source_missing"
        document.index_error = "原始文件缺失，已暂停该文档参与知识库问答；请恢复文件或显式删除文档。"
        for chunk in db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)).all():
            chunk.index_status = "source_missing"
        removed.append((document_id, "marked_source_missing"))
    if removed:
        db.commit()
    return removed


def restore_documents_with_available_originals(db: Session) -> list[tuple[str, str]]:
    restored: list[tuple[str, str]] = []
    documents = db.scalars(select(Document).where(Document.status == "source_missing")).all()
    for document in documents:
        if not original_file_exists(document):
            continue
        chunks = db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)
        ).all()
        try:
            vector_chunk_ids = document_vector_chunk_ids(document.document_id)
        except VectorStoreError as exc:
            document.index_error = f"原文件已恢复，但向量完整性核对失败：{exc}"[:1000]
            continue
        expected_chunk_ids = {chunk.chunk_id for chunk in chunks}
        if vector_chunk_ids != expected_chunk_ids or document.index_version != INDEX_VERSION:
            document.status = "index_failed"
            document.index_error = (
                "原文件已恢复，但当前版本向量不完整，请重建索引："
                f"expected={len(expected_chunk_ids)}, actual={len(vector_chunk_ids)}"
            )
            for chunk in chunks:
                chunk.index_status = "index_failed"
            restored.append((document.document_id, "requires_reindex"))
            continue
        document.status = "indexed"
        document.lifecycle_stage = None
        document.index_error = None
        db.execute(
            update(DocumentChunk)
            .where(DocumentChunk.document_id == document.document_id)
            .where(DocumentChunk.index_status == "source_missing")
            .values(index_status="indexed")
        )
        restored.append((document.document_id, "restored_indexed"))
    if restored:
        db.commit()
    return restored


def recover_documents_from_originals(db: Session) -> list[tuple[str, str]]:
    recovered: list[tuple[str, str]] = []
    if not DOCUMENT_DIR.exists():
        return recovered

    existing_ids = set(db.scalars(select(Document.document_id)).all())
    for path in sorted(DOCUMENT_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue
        document_id = path.stem
        if not document_id.startswith("DOC-") or document_id in existing_ids:
            continue
        try:
            parsed = parse_document(path)
            chunks = build_chunks_from_parsed(document_id=document_id, parsed=parsed, source_file=path.name)
        except DocumentParseError as exc:
            recovered.append((document_id, f"parse_failed: {exc}"))
            continue
        if not chunks:
            recovered.append((document_id, "empty"))
            continue

        vector_count = 0
        vector_error: str | None = None
        try:
            vector_count = count_document_vectors(document_id)
        except VectorStoreError as exc:
            vector_error = str(exc)
        indexed = vector_count >= len(chunks)
        document = Document(
            document_id=document_id,
            filename=path.name,
            filename_norm=_normalize_filename(path.name),
            content_type=None,
            file_type=path.suffix.lower().lstrip("."),
            size=path.stat().st_size,
            file_sha256=_sha256_file(path),
            storage_path=str(path),
            status="indexed" if indexed else "uploaded",
            index_version=INDEX_VERSION,
            index_error=None
            if indexed
            else f"从原始文件恢复 SQLite 记录，向量索引需重建。{vector_error or ''}".strip(),
            chunk_count=len(chunks),
        )
        db.add(document)
        chunk_models: list[DocumentChunk] = []
        for chunk in chunks:
            chunk_model = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    text=chunk.text,
                    embedding_text=chunk.embedding_text,
                    chunk_metadata=json.dumps(chunk.metadata or {}, ensure_ascii=False),
                    token_count=chunk.token_count,
                    index_status="indexed" if indexed else "uploaded",
                    index_version=INDEX_VERSION,
                    title=chunk.title,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number,
                    source_file=path.name,
                )
            chunk_models.append(chunk_model)
            db.add(chunk_model)
        db.flush()
        existing_ids.add(document_id)
        recovered.append((document_id, "indexed" if indexed else "uploaded"))
    if recovered:
        db.commit()
    return recovered


def recover_interrupted_deletions() -> None:
    """Retry lifecycle operations left in progress by a process interruption."""

    from backend.app.core.database import SessionLocal

    with SessionLocal() as db:
        document_ids = db.scalars(
            select(Document.document_id).where(Document.status == "deleting")
        ).all()
    for document_id in document_ids:
        with SessionLocal() as db:
            document = db.scalar(
                select(Document).where(Document.document_id == document_id)
            )
            if document is None:
                continue
            try:
                delete_document_record(db, document)
            except DocumentDeletionError:
                continue


def _mark_delete_failed(
    db: Session,
    document: Document,
    chunks: list[DocumentChunk],
    stage: str,
    error: str,
) -> None:
    document.status = "delete_failed"
    document.lifecycle_stage = stage
    stage_labels = {
        "deleting_vectors": "Qdrant 向量删除失败",
        "deleting_file": "原文件删除失败",
        "deleting_metadata": "SQLite 元数据删除失败",
    }
    document.index_error = f"{stage_labels.get(stage, '文档删除失败')}：{error}"[:1000]
    for chunk in chunks:
        chunk.index_status = "delete_failed"
    db.commit()


def _normalize_filename(filename: str) -> str:
    if "\\" in filename or ":" in filename:
        basename = PureWindowsPath(filename).name
    else:
        basename = Path(filename).name
    return unicodedata.normalize("NFC", basename).strip().casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
