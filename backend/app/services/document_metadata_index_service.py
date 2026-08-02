from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import Document, DocumentChunk
from backend.app.services.document_metadata_service import retrieval_metadata_snapshot
from backend.app.services.vector_store_service import refresh_document_metadata_payload


def refresh_document_metadata_indexes(
    db: Session,
    document: Document,
    *,
    refresh_qdrant: bool = True,
) -> dict[str, Any]:
    """Synchronize metadata into chunks and vector payloads without embedding work."""

    metadata = retrieval_metadata_snapshot(document)
    chunks = list(
        db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)
        ).all()
    )
    for chunk in chunks:
        chunk_metadata = _json_object(chunk.chunk_metadata)
        chunk_metadata.update(metadata)
        chunk.chunk_metadata = json.dumps(chunk_metadata, ensure_ascii=False)
    db.flush()
    qdrant_refreshed = refresh_qdrant and bool(chunks)
    if qdrant_refreshed:
        refresh_document_metadata_payload(document.document_id, metadata)
    from backend.app.services.rag_service import clear_document_snapshot_cache

    clear_document_snapshot_cache({document.document_id})
    return {
        "document_id": document.document_id,
        "chunk_count": len(chunks),
        "qdrant_refreshed": qdrant_refreshed,
    }


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}
