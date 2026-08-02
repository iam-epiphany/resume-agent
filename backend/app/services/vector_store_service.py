from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from backend.app.core.config import (
    EMBEDDING_DIMENSION,
    INDEX_VERSION,
    QDRANT_COLLECTION,
    QDRANT_AUTO_CREATE_COLLECTION,
    QDRANT_DENSE_VECTOR_NAME,
    QDRANT_UPSERT_BATCH_SIZE,
    QDRANT_URL,
)
from backend.app.services.chunk_service import ChunkDraft
from backend.app.services.embedding_service import TextEmbedding
from backend.app.services.performance_metrics import measure


class VectorStoreError(RuntimeError):
    pass


_COLLECTION_READY_LOCK = RLock()
_COLLECTION_READY_KEY: tuple[str, str] | None = None


@dataclass
class VectorSearchResult:
    chunk_id: str
    document_id: str
    filename: str
    section_title: str | None
    page_number: int | None
    text: str
    embedding_text: str
    token_count: int
    score: float
    chunk_type: str = "paragraph"
    section_path: list[str] | None = None
    section_number: str | None = None
    parent_section_number: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: dict[str, Any] | None = None


def ensure_vector_collection(*, force: bool = False) -> None:
    global _COLLECTION_READY_KEY
    readiness_key = (QDRANT_URL, QDRANT_COLLECTION)
    if not force and _COLLECTION_READY_KEY == readiness_key:
        return
    with _COLLECTION_READY_LOCK:
        if not force and _COLLECTION_READY_KEY == readiness_key:
            return
        client, models = _qdrant()
        try:
            with measure("qdrant.readiness"):
                if not client.collection_exists(QDRANT_COLLECTION):
                    if not QDRANT_AUTO_CREATE_COLLECTION:
                        raise VectorStoreError(
                            f"Qdrant collection/alias 尚未发布：{QDRANT_COLLECTION}"
                        )
                    client.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config={
                            QDRANT_DENSE_VECTOR_NAME: models.VectorParams(
                                size=EMBEDDING_DIMENSION,
                                distance=models.Distance.COSINE,
                            )
                        },
                    )
                _ensure_payload_indexes(client, models)
            _COLLECTION_READY_KEY = readiness_key
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Qdrant collection 初始化失败") from exc


def reset_vector_collection_readiness() -> None:
    global _COLLECTION_READY_KEY
    with _COLLECTION_READY_LOCK:
        _COLLECTION_READY_KEY = None


def upsert_chunk_embeddings(
    *,
    chunks: list[ChunkDraft],
    embeddings: list[TextEmbedding],
    filename: str,
) -> None:
    if len(chunks) != len(embeddings):
        raise VectorStoreError("chunk 数量与 embedding 数量不一致")

    ensure_vector_collection()
    client, models = _qdrant()
    points = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_metadata = chunk.metadata or {}
        points.append(
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    QDRANT_DENSE_VECTOR_NAME: embedding.dense,
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": filename,
                    "source_file": filename,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "embedding_text": chunk.embedding_text,
                    "token_count": chunk.token_count,
                    "chunk_type": chunk.chunk_type,
                    "section_path": chunk.section_path or ([chunk.section_title] if chunk.section_title else []),
                    "section_number": chunk.section_number,
                    "parent_section_number": chunk.parent_section_number,
                    "previous_chunk_id": chunk.previous_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                    "chunk_metadata": chunk_metadata,
                    "file_type": chunk_metadata.get("source_format"),
                    "source_title": chunk_metadata.get("source_title"),
                    "external_doc_id": chunk_metadata.get("external_doc_id"),
                    "issuing_authority": chunk_metadata.get("issuing_authority"),
                    "publication_date": chunk_metadata.get("publication_date"),
                    "effective_date": chunk_metadata.get("effective_date"),
                    "expiration_date": chunk_metadata.get("expiration_date"),
                    "document_number": chunk_metadata.get("document_number"),
                    "material_topic": chunk_metadata.get("material_topic"),
                    "business_domain": chunk_metadata.get("business_domain"),
                    "source_url": chunk_metadata.get("source_url"),
                    "attachment_url": chunk_metadata.get("attachment_url"),
                    "version_status": chunk_metadata.get("version_status") or "unknown",
                    "article_number": chunk_metadata.get("article_number") or chunk.section_number,
                    "table_id": chunk_metadata.get("table_id"),
                    "table_title": chunk_metadata.get("table_title"),
                    "sheet_name": chunk_metadata.get("sheet_name"),
                    "period": chunk_metadata.get("period"),
                    "year": _period_value(chunk_metadata, "year"),
                    "month": _period_value(chunk_metadata, "month"),
                    "quarter": _period_value(chunk_metadata, "quarter"),
                    "unit": chunk_metadata.get("unit"),
                    "row_label": chunk_metadata.get("row_label"),
                    "table_headers": chunk_metadata.get("table_headers") or chunk_metadata.get("headers"),
                    "row_index": chunk_metadata.get("row_index"),
                    "row_cells": chunk_metadata.get("row_cells"),
                    "raw_table_preview": chunk_metadata.get("raw_table_preview"),
                    "index_version": INDEX_VERSION,
                },
            )
        )

    try:
        with measure("qdrant.upsert"):
            for start in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points[start : start + QDRANT_UPSERT_BATCH_SIZE],
                    wait=True,
                )
    except Exception as exc:
        raise VectorStoreError("Qdrant chunk 向量写入失败") from exc


def count_document_vectors(document_id: str) -> int:
    ensure_vector_collection()
    client, models = _qdrant()
    try:
        response = client.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=_document_filter(document_id, models),
            exact=True,
        )
    except Exception as exc:
        raise VectorStoreError(f"Qdrant 文档向量计数失败：{exc}") from exc
    return int(getattr(response, "count", 0) or 0)


def document_vector_chunk_ids(document_id: str) -> set[str]:
    """Return current-index chunk IDs for exact cross-store verification."""

    ensure_vector_collection()
    client, models = _qdrant()
    chunk_ids: set[str] = set()
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=_document_filter(document_id, models),
                limit=256,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            chunk_ids.update(
                str(point.payload.get("chunk_id"))
                for point in points
                if point.payload and point.payload.get("chunk_id")
            )
            if offset is None:
                break
    except Exception as exc:
        raise VectorStoreError(f"Qdrant 文档向量明细核对失败：{exc}") from exc
    return chunk_ids


def get_vector_chunk_by_chunk_id(chunk_id: str) -> VectorSearchResult | None:
    """Return one current Qdrant payload by logical chunk_id."""

    if not chunk_id:
        return None
    ensure_vector_collection()
    client, _models = _qdrant()
    try:
        points = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[_point_id(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise VectorStoreError(f"读取 Qdrant chunk payload 失败：{chunk_id}") from exc
    if not points:
        return None
    return _to_search_result(points[0])


def delete_document_vectors(document_id: str, chunk_ids: list[str] | None = None) -> None:
    ensure_vector_collection()
    client, models = _qdrant()
    try:
        if chunk_ids:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.PointIdsList(points=[_point_id(chunk_id) for chunk_id in chunk_ids]),
                wait=True,
            )
        else:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
    except Exception as exc:
        raise VectorStoreError(f"Qdrant 文档向量删除失败：{exc}") from exc


def hybrid_search(
    query_embedding: TextEmbedding,
    *,
    limit: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[VectorSearchResult]:
    ensure_vector_collection()
    client, models = _qdrant()
    try:
        with measure("qdrant.hybrid_search"):
            response = client.query_points(
                collection_name=QDRANT_COLLECTION,
                **_hybrid_query_kwargs(query_embedding, limit, models, metadata_filter),
            )
    except Exception as exc:
        raise VectorStoreError("Qdrant hybrid 检索失败") from exc

    points = getattr(response, "points", response)
    return [_to_search_result(point) for point in points]


def hybrid_search_batch(
    query_embeddings: list[TextEmbedding],
    *,
    limit: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[list[VectorSearchResult]]:
    if not query_embeddings:
        return []
    ensure_vector_collection()
    client, models = _qdrant()
    if not hasattr(client, "query_batch_points"):
        return [
            hybrid_search(embedding, limit=limit, metadata_filter=metadata_filter)
            for embedding in query_embeddings
        ]
    requests = [
        models.QueryRequest(**_hybrid_query_kwargs(embedding, limit, models, metadata_filter))
        for embedding in query_embeddings
    ]
    try:
        with measure("qdrant.hybrid_search_batch"):
            responses = client.query_batch_points(
                collection_name=QDRANT_COLLECTION,
                requests=requests,
            )
    except Exception as exc:
        raise VectorStoreError("Qdrant batch hybrid 检索失败") from exc
    return [
        [_to_search_result(point) for point in getattr(response, "points", response)]
        for response in responses
    ]


def _hybrid_query_kwargs(
    query_embedding: TextEmbedding,
    limit: int,
    models: Any,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compiled_filter = _retrieval_filter(models, metadata_filter)
    return {
        "prefetch": [
            models.Prefetch(
                query=query_embedding.dense,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=limit,
                filter=compiled_filter,
            ),
        ],
        "query": models.FusionQuery(fusion=models.Fusion.RRF),
        "limit": limit,
        "with_payload": True,
    }


def refresh_document_metadata_payload(document_id: str, metadata: dict[str, Any]) -> None:
    """Refresh common document metadata without recomputing embeddings."""

    ensure_vector_collection()
    client, models = _qdrant()
    allowed = {
        "source_title",
        "source_filename",
        "file_sha256",
        "external_doc_id",
        "issuing_authority",
        "publication_date",
        "effective_date",
        "expiration_date",
        "document_number",
        "material_topic",
        "business_domain",
        "source_url",
        "attachment_url",
        "source_type",
        "version_label",
        "version_status",
        "supersedes_document_id",
    }
    payload = {key: metadata.get(key) for key in allowed}
    payload["version_status"] = payload.get("version_status") or "unknown"
    try:
        client.set_payload(
            collection_name=QDRANT_COLLECTION,
            payload=payload,
            points=models.FilterSelector(filter=_document_filter(document_id, models)),
            wait=True,
        )
    except Exception as exc:
        raise VectorStoreError(f"Qdrant 文档元数据刷新失败：{exc}") from exc


def _qdrant() -> tuple[Any, Any]:
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise VectorStoreError("缺少 qdrant-client 依赖，无法连接向量数据库") from exc
    return _qdrant_client(), models


@lru_cache(maxsize=1)
def _qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise VectorStoreError("缺少 qdrant-client 依赖，无法连接向量数据库") from exc
    # Metadata-only updates on very large spreadsheet documents can touch
    # thousands of points.  The client's short default timeout can expire even
    # though Qdrant completes the operation, so use a bounded long timeout.
    return QdrantClient(url=QDRANT_URL, timeout=60)


def _ensure_payload_indexes(client: Any, models: Any) -> None:
    indexes = {
        "document_id": models.PayloadSchemaType.KEYWORD,
        "index_version": models.PayloadSchemaType.KEYWORD,
        "source_file": models.PayloadSchemaType.KEYWORD,
        "chunk_type": models.PayloadSchemaType.KEYWORD,
        "sheet_name": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
        "month": models.PayloadSchemaType.INTEGER,
        "quarter": models.PayloadSchemaType.INTEGER,
        "external_doc_id": models.PayloadSchemaType.KEYWORD,
        "issuing_authority": models.PayloadSchemaType.KEYWORD,
        "publication_date": models.PayloadSchemaType.KEYWORD,
        "effective_date": models.PayloadSchemaType.KEYWORD,
        "document_number": models.PayloadSchemaType.KEYWORD,
        "material_topic": models.PayloadSchemaType.KEYWORD,
        "business_domain": models.PayloadSchemaType.KEYWORD,
        "version_status": models.PayloadSchemaType.KEYWORD,
        "article_number": models.PayloadSchemaType.KEYWORD,
    }
    try:
        collection_info = client.get_collection(QDRANT_COLLECTION)
        existing_indexes = set(
            (getattr(collection_info, "payload_schema", {}) or {}).keys()
        )
    except Exception:
        existing_indexes = set()

    for field_name, field_schema in indexes.items():
        if field_name in existing_indexes:
            continue
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
                timeout=60,
            )
        except Exception as exc:
            if "exist" in str(exc).lower():
                continue
            raise


def _point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"resumemind:{chunk_id}"))


def _index_version_filter(models: Any) -> Any:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="index_version",
                match=models.MatchValue(value=INDEX_VERSION),
            )
        ]
    )


def _retrieval_filter(models: Any, metadata_filter: dict[str, Any] | None) -> Any:
    conditions = list(_index_version_filter(models).must or [])
    filters = metadata_filter or {}
    document_ids = [str(value) for value in filters.get("document_ids") or [] if value]
    if "document_ids" in filters:
        conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=document_ids),
            )
        )
    for key in ("article_number", "version_status", "year", "month", "quarter"):
        value = filters.get(key)
        if value in (None, "", []):
            continue
        conditions.append(
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
        )
    return models.Filter(must=conditions)


def _document_filter(document_id: str, models: Any) -> Any:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            ),
            models.FieldCondition(
                key="index_version",
                match=models.MatchValue(value=INDEX_VERSION),
            ),
        ]
    )


def _to_search_result(point: Any) -> VectorSearchResult:
    payload = point.payload or {}
    metadata = _payload_dict(payload.get("chunk_metadata"))
    for key in (
        "source_title",
        "external_doc_id",
        "issuing_authority",
        "publication_date",
        "effective_date",
        "expiration_date",
        "document_number",
        "material_topic",
        "business_domain",
        "source_url",
        "attachment_url",
        "version_status",
        "article_number",
        "file_type",
        "year",
        "month",
        "quarter",
    ):
        if payload.get(key) not in (None, "", []):
            metadata[key] = payload[key]
    return VectorSearchResult(
        chunk_id=str(payload.get("chunk_id", "")),
        document_id=str(payload.get("document_id", "")),
        filename=str(payload.get("filename", "")),
        section_title=payload.get("section_title"),
        page_number=payload.get("page_number"),
        text=str(payload.get("text", "")),
        embedding_text=str(payload.get("embedding_text", "")),
        token_count=int(payload.get("token_count") or 0),
        score=float(getattr(point, "score", 0.0) or 0.0),
        chunk_type=str(payload.get("chunk_type") or "paragraph"),
        section_path=_payload_string_list(payload.get("section_path")),
        section_number=payload.get("section_number"),
        parent_section_number=payload.get("parent_section_number"),
        previous_chunk_id=payload.get("previous_chunk_id"),
        next_chunk_id=payload.get("next_chunk_id"),
        metadata=metadata,
    )


def _payload_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item) for item in value if item]


def _payload_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _period_value(metadata: dict[str, Any], key: str) -> Any:
    direct = metadata.get(f"inferred_{key}")
    if direct is not None:
        return direct
    period = metadata.get("period")
    return period.get(key) if isinstance(period, dict) else None
