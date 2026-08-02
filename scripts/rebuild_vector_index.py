"""Re-parse originals, replace SQLite chunks, then rebuild Qdrant vectors."""

from pathlib import Path

from sqlalchemy import select

from backend.app.core.config import INDEX_VERSION
from backend.app.core.database import SessionLocal, init_db
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.chunk_service import build_chunks_from_parsed
from backend.app.services.document_indexing_service import DocumentIndexingError, index_document
from backend.app.services.document_parser import DocumentParseError, parse_document
from backend.app.services.vector_store_service import VectorStoreError, delete_document_vectors


def main() -> None:
    init_db()
    with SessionLocal() as db:
        documents = db.scalars(select(Document).order_by(Document.id.asc())).all()
        if not documents:
            print("No documents found; vector index rebuild skipped.")
            return

        for document in documents:
            try:
                refresh_document_chunks(db, document)
                index_document(db, document)
            except (DocumentIndexingError, DocumentParseError, VectorStoreError, FileNotFoundError) as exc:
                document.status = "index_failed"
                document.index_error = str(exc)[:1000]
                db.commit()
                print(f"Rebuild failed for {document.document_id}: {exc}")
                continue
            print(f"Indexed {document.chunk_count} chunks for {document.document_id}.")

        print("Vector index rebuild completed.")


def refresh_document_chunks(db, document: Document) -> None:
    source_path = Path(document.storage_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Original file missing: {source_path}")

    parsed = parse_document(source_path)
    drafts = build_chunks_from_parsed(
        document_id=document.document_id,
        parsed=parsed,
        source_file=document.filename,
    )
    if not drafts:
        raise DocumentParseError("文档没有可入库的文本片段")

    old_chunks = db.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document.document_id).order_by(DocumentChunk.id.asc())
    ).all()
    old_chunk_ids = [chunk.chunk_id for chunk in old_chunks]
    if old_chunk_ids:
        delete_document_vectors(document.document_id, old_chunk_ids)

    for chunk in old_chunks:
        db.delete(chunk)
    # 先 flush 删除，避免新 chunk 与旧 chunk 同批 flush 时撞唯一约束
    db.flush()

    for draft in drafts:
        db.add(
            DocumentChunk(
                chunk_id=draft.chunk_id,
                document_id=document.document_id,
                text=draft.text,
                embedding_text=draft.embedding_text,
                token_count=draft.token_count,
                index_status="uploaded",
                index_version=INDEX_VERSION,
                title=draft.title,
                section_title=draft.section_title,
                page_number=draft.page_number,
                source_file=document.filename,
            )
        )

    document.chunk_count = len(drafts)
    document.status = "uploaded"
    document.index_version = INDEX_VERSION
    document.index_error = None
    db.commit()


if __name__ == "__main__":
    main()
