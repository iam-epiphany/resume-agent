from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import INDEX_TASK_MAX_RETRIES, MAX_BATCH_UPLOAD_FILES, MAX_UPLOAD_BYTES
from backend.app.core.database import get_db
from backend.app.core.security import require_admin
from backend.app.models.document import Document, DocumentChunk, DocumentIndexTask
from backend.app.schemas.documents import (
    DocumentBatchUploadItem,
    DocumentBatchUploadResponse,
    DocumentBulkDeleteItem,
    DocumentBulkDeleteRequest,
    DocumentBulkDeleteResponse,
    ChunkSummary,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentProcessingResponse,
    DocumentSummary,
    DocumentConflictExistingDocument,
    DocumentUploadPreflightItem,
    DocumentUploadPreflightRequest,
    DocumentUploadPreflightResponse,
    DocumentUploadResponse,
    DocumentMetadataInput,
    DocumentMetadataConfirmResponse,
    DocumentMetadataUpdateResponse,
    DocumentManifestImportItem,
    DocumentManifestImportResponse,
    DocumentUrlImportRequest,
)
from backend.app.schemas.qa import ApiError
from backend.app.services.audit_service import log_action, record_event
from backend.app.services.document_storage import (
    EmptyDocumentError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
)
from backend.app.services.index_task_service import enqueue_document_index
from backend.app.services.document_lifecycle_service import (
    DocumentDeletionError,
    delete_document_record,
    original_file_exists,
    recover_documents_from_originals,
    remove_documents_with_missing_originals,
    restore_documents_with_available_originals,
)
from backend.app.services.document_upload_service import (
    DocumentUploadConflictError,
    ExactDuplicateDocumentError,
    FilenameConflictDocumentError,
    UploadTaskMissingError,
    UploadPreflightInput,
    build_upload_preflight,
    create_document_upload,
)
from backend.app.services.document_manifest_service import (
    ManifestImportError,
    import_manifest_records,
    parse_manifest,
)
from backend.app.services.document_metadata_service import (
    IDENTITY_METADATA_FIELDS,
    DocumentMetadataError,
    apply_document_metadata,
    confirm_document_identity,
    document_metadata_snapshot,
    identity_snapshot_hash,
    invalidate_identity_review,
    validate_document_identity,
)
from backend.app.services.document_metadata_index_service import refresh_document_metadata_indexes
from backend.app.services.vector_store_service import VectorStoreError
from backend.app.services.document_url_import_service import (
    DocumentUrlImportError,
    fetch_url_document,
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_admin)],
)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    file: UploadFile = File(...),
    filename_override: str | None = Form(None),
    overwrite_document_id: str | None = Form(None),
    metadata_json: str | None = Form(None),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    request_id = (idempotency_key or uuid4().hex).strip()[:128]
    try:
        metadata = _parse_metadata_json(metadata_json)
        document, task = await create_document_upload(
            db,
            file=file,
            request_id=request_id,
            filename_override=filename_override,
            overwrite_document_id=overwrite_document_id,
            metadata=metadata,
            max_bytes=MAX_UPLOAD_BYTES,
            enqueue_index=enqueue_document_index,
        )
        return _to_upload_response(document, task)
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (UnsupportedDocumentTypeError, EmptyDocumentError, DocumentMetadataError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadTaskMissingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DocumentUploadConflictError as exc:
        return _upload_conflict_response(exc)


@router.post("/manifest", response_model=DocumentManifestImportResponse)
async def import_document_manifest(
    manifest: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentManifestImportResponse:
    try:
        content = await manifest.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise ManifestImportError("manifest 超过上传大小限制")
        records = parse_manifest(content, manifest.filename or "manifest.json")
        results = import_manifest_records(db, records)
    except (ManifestImportError, DocumentMetadataError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    updated = [item for item in results if item.status == "updated" and item.document_id]
    # Manifest enrichment refreshes SQLite chunks and Qdrant payloads in place;
    # re-enqueuing full indexing here would recompute unchanged embeddings.
    return DocumentManifestImportResponse(
        total_count=len(results),
        updated_count=len(updated),
        failed_count=len(results) - len(updated),
        items=[
            DocumentManifestImportItem(
                record_index=item.record_index,
                status=item.status,
                document_id=item.document_id,
                filename=item.filename,
                message=item.message,
            )
            for item in results
        ],
    )


@router.post(
    "/url-import",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def import_document_from_url(
    payload: DocumentUrlImportRequest,
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    try:
        fetched = fetch_url_document(
            payload.url,
            max_bytes=MAX_UPLOAD_BYTES,
            filename_override=payload.filename,
        )
        upload = UploadFile(
            file=BytesIO(fetched.content),
            filename=fetched.filename,
            size=len(fetched.content),
            headers=Headers({"content-type": fetched.content_type}),
        )
        metadata = _metadata_input_dict(payload.metadata)
        metadata.setdefault("source_url", fetched.final_url)
        if fetched.content_type != "text/html":
            metadata.setdefault("attachment_url", fetched.final_url)
        metadata.setdefault("source_type", "official_url")
        document, task = await create_document_upload(
            db,
            file=upload,
            request_id=payload.client_request_id or uuid4().hex,
            filename_override=fetched.filename,
            metadata=metadata,
            metadata_source="user",
            max_bytes=MAX_UPLOAD_BYTES,
            enqueue_index=enqueue_document_index,
        )
        return _to_upload_response(document, task)
    except (DocumentUrlImportError, DocumentMetadataError, UnsupportedDocumentTypeError, EmptyDocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except DocumentUploadConflictError as exc:
        return _upload_conflict_response(exc)


@router.patch("/{document_id}/metadata", response_model=DocumentMetadataUpdateResponse)
def update_document_metadata(
    document_id: str,
    payload: DocumentMetadataInput,
    db: Session = Depends(get_db),
) -> DocumentMetadataUpdateResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")
    before = document_metadata_snapshot(document)
    refresh_warning: str | None = None
    metadata_refreshed = False
    try:
        metadata = _metadata_patch_dict(payload)
        if metadata:
            apply_document_metadata(
                document,
                metadata,
                source="user",
                confidence=1.0,
                allow_clear=True,
            )
            invalidate_identity_review(document)
            validate_document_identity(document)
            try:
                refresh_result = refresh_document_metadata_indexes(db, document)
                metadata_refreshed = bool(refresh_result.get("qdrant_refreshed"))
            except VectorStoreError as exc:
                refresh_warning = f"身份信息已保存，Qdrant payload 待刷新：{exc}"
        db.commit()
    except (DocumentMetadataError, ValueError) as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    after = document_metadata_snapshot(document)
    if metadata:
        _record_identity_audit(
            db,
            "document_identity_updated",
            document,
            before,
            after,
            metadata_refreshed=metadata_refreshed,
            refresh_warning=refresh_warning,
        )
    return DocumentMetadataUpdateResponse(
        document_id=document.document_id,
        metadata=after,
        reindex_queued=False,
        metadata_refreshed=metadata_refreshed,
        refresh_warning=refresh_warning,
    )


@router.post(
    "/{document_id}/metadata/confirm",
    response_model=DocumentMetadataConfirmResponse,
)
def confirm_document_metadata(
    document_id: str,
    db: Session = Depends(get_db),
) -> DocumentMetadataConfirmResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")
    before = document_metadata_snapshot(document)
    refresh_warning: str | None = None
    metadata_refreshed = False
    try:
        confirm_document_identity(document)
        refresh_targets = [document]
        try:
            refresh_results = []
            for target in refresh_targets:
                refresh_results.append(refresh_document_metadata_indexes(db, target))
            metadata_refreshed = any(bool(result.get("qdrant_refreshed")) for result in refresh_results)
        except VectorStoreError as exc:
            refresh_warning = f"身份卡已确认，Qdrant payload 待刷新：{exc}"
        db.commit()
    except (DocumentMetadataError, ValueError) as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    after = document_metadata_snapshot(document)
    _record_identity_audit(
        db,
        "document_identity_confirmed",
        document,
        before,
        after,
        metadata_refreshed=metadata_refreshed,
        refresh_warning=refresh_warning,
    )
    return DocumentMetadataConfirmResponse(
        document_id=document.document_id,
        metadata=after,
        metadata_refreshed=metadata_refreshed,
        refresh_warning=refresh_warning,
    )


@router.post(
    "/upload-preflight",
    response_model=DocumentUploadPreflightResponse,
)
def preflight_document_upload(
    payload: DocumentUploadPreflightRequest,
    db: Session = Depends(get_db),
) -> DocumentUploadPreflightResponse:
    results = build_upload_preflight(
        db,
        [
            UploadPreflightInput(
                client_file_id=item.client_file_id,
                filename=item.filename,
                size=item.size,
                file_sha256=item.file_sha256.lower(),
            )
            for item in payload.items
        ],
    )
    return DocumentUploadPreflightResponse(
        items=[
            DocumentUploadPreflightItem(
                client_file_id=item.client_file_id,
                filename=item.filename,
                status=item.status,
                existing_document=_to_conflict_existing_document(item.existing_document)
                if item.existing_document is not None
                else None,
                error_message=item.error_message,
            )
            for item in results
        ]
    )

@router.post(
    "/batch-upload",
    response_model=DocumentBatchUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_documents_batch(
    files: list[UploadFile] = File(...),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
) -> DocumentBatchUploadResponse:
    if len(files) > MAX_BATCH_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"单次批量上传最多支持 {MAX_BATCH_UPLOAD_FILES} 个文件",
        )

    batch_id = (idempotency_key or uuid4().hex).strip()[:96]
    items: list[DocumentBatchUploadItem] = []
    for index, file in enumerate(files):
        filename = file.filename or "unknown"
        request_id = _batch_file_request_id(batch_id, index, filename)
        try:
            document, task = await create_document_upload(
                db,
                file=file,
                request_id=request_id,
                max_bytes=MAX_UPLOAD_BYTES,
                enqueue_index=enqueue_document_index,
            )
            items.append(
                DocumentBatchUploadItem(
                    filename=filename,
                    status="accepted",
                    document_id=document.document_id,
                    task_id=task.task_id or str(task.id),
                    stage=task.stage,
                    size=document.size,
                    error_message=None,
                )
            )
        except ExactDuplicateDocumentError as exc:
            items.append(
                DocumentBatchUploadItem(
                    filename=filename,
                    status="duplicate",
                    document_id=exc.existing_document.document_id if exc.existing_document else None,
                    size=exc.existing_document.size if exc.existing_document else None,
                    error_message=str(exc),
                )
            )
        except FilenameConflictDocumentError as exc:
            items.append(
                DocumentBatchUploadItem(
                    filename=filename,
                    status="conflict",
                    document_id=exc.existing_document.document_id if exc.existing_document else None,
                    size=exc.existing_document.size if exc.existing_document else None,
                    error_message=str(exc),
                )
            )
        except DocumentUploadConflictError as exc:
            items.append(
                DocumentBatchUploadItem(
                    filename=filename,
                    status="conflict",
                    document_id=exc.existing_document.document_id if exc.existing_document else None,
                    size=exc.existing_document.size if exc.existing_document else None,
                    error_message=str(exc),
                )
            )
        except (UnsupportedDocumentTypeError, EmptyDocumentError, DocumentTooLargeError, UploadTaskMissingError) as exc:
            items.append(
                DocumentBatchUploadItem(
                    filename=filename,
                    status="failed",
                    error_message=str(exc),
                )
            )

    accepted_count = sum(1 for item in items if item.status == "accepted")
    return DocumentBatchUploadResponse(
        batch_id=batch_id,
        accepted_count=accepted_count,
        failed_count=sum(1 for item in items if item.status != "accepted"),
        items=items,
    )


@router.post("/{document_id}/index", response_model=DocumentDetailResponse)
def build_document_index(
    document_id: str,
    force_rebuild_chunks: bool = Query(
        False,
        description="Reparse the original file and rebuild SQLite chunks/cell index before vector indexing.",
    ),
    db: Session = Depends(get_db),
) -> DocumentDetailResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")

    enqueue_document_index(document_id, force_rebuild_chunks=force_rebuild_chunks)
    log_action(
        db,
        "document_index_queued",
        "document",
        document_id,
        f"chunks={document.chunk_count}; force_rebuild_chunks={force_rebuild_chunks}",
    )
    db.refresh(document)
    return _to_detail(document, db)


@router.get("/{document_id}/processing", response_model=DocumentProcessingResponse)
def get_document_processing(
    document_id: str,
    db: Session = Depends(get_db),
) -> DocumentProcessingResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")
    task = db.scalar(
        select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
    )
    if task is None:
        raise HTTPException(status_code=404, detail="未找到文档处理任务")
    return _to_processing_response(document, task)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    repair_sources: bool = Query(False, description="Scan original files and repair missing-source states."),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    restored_documents = restore_documents_with_available_originals(db)
    for document_id, result in restored_documents:
        detail = f"original file is available again; result: {result}"
        log_action(db, "document_source_restored", "document", document_id, detail[:500])

    if repair_sources:
        recovered_documents = recover_documents_from_originals(db)
        for document_id, result in recovered_documents:
            detail = f"original file recovered into SQLite; result: {result}"
            log_action(db, "document_recovered_from_original", "document", document_id, detail[:500])

        removed_documents = remove_documents_with_missing_originals(db)
        for document_id, result in removed_documents:
            detail = f"original file missing; result: {result}"
            log_action(db, "document_marked_source_missing", "document", document_id, detail[:500])

    statement = select(Document).order_by(Document.uploaded_at.desc())
    documents = db.scalars(statement).all()
    return DocumentListResponse(documents=[_to_summary(document) for document in documents])


@router.post("/bulk-delete", response_model=DocumentBulkDeleteResponse)
def bulk_delete_documents(
    payload: DocumentBulkDeleteRequest,
    db: Session = Depends(get_db),
) -> DocumentBulkDeleteResponse:
    document_ids = list(dict.fromkeys(payload.document_ids))
    items: list[DocumentBulkDeleteItem] = []

    for document_id in document_ids:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        if document is None:
            items.append(
                DocumentBulkDeleteItem(
                    document_id=document_id,
                    status="not_found",
                    message="未找到指定文档",
                )
            )
            continue

        filename = document.filename
        blocked_message = _document_delete_block_message(document)
        if blocked_message is not None:
            items.append(
                DocumentBulkDeleteItem(
                    document_id=document.document_id,
                    filename=filename,
                    status="blocked",
                    message=blocked_message,
                )
            )
            continue

        try:
            _delete_document_with_audit(db, document)
        except DocumentDeletionError as exc:
            items.append(
                DocumentBulkDeleteItem(
                    document_id=document_id,
                    filename=filename,
                    status="failed",
                    message=f"文档删除未完成，记录已保留为 delete_failed，可稍后重试：{exc}",
                )
            )
            continue

        items.append(
            DocumentBulkDeleteItem(
                document_id=document_id,
                filename=filename,
                status="deleted",
                message="删除成功",
            )
        )

    deleted_count = sum(1 for item in items if item.status == "deleted")
    return DocumentBulkDeleteResponse(
        requested_count=len(document_ids),
        deleted_count=deleted_count,
        failed_count=len(items) - deleted_count,
        items=items,
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: str,
    chunk_offset: int = Query(0, ge=0),
    chunk_limit: int = Query(50, ge=1, le=200),
    validate_source: bool = Query(False, description="Mark the document source_missing if the original file is gone."),
    db: Session = Depends(get_db),
) -> DocumentDetailResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")

    if document.status == "source_missing" and original_file_exists(document):
        for restored_document_id, result in restore_documents_with_available_originals(db):
            log_action(
                db,
                "document_source_restored",
                "document",
                restored_document_id,
                f"original file is available again; result: {result}"[:500],
            )
        db.refresh(document)

    if validate_source and not original_file_exists(document):
        document.status = "source_missing"
        document.index_error = "原始文件缺失，已暂停该文档参与知识库问答；请恢复文件或显式删除文档。"
        chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)).all()
        for chunk in chunks:
            chunk.index_status = "source_missing"
        db.commit()
        log_action(db, "document_marked_source_missing", "document", document_id, "original file missing")
        db.refresh(document)

    return _to_detail(document, db, chunk_offset=chunk_offset, chunk_limit=chunk_limit)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> DocumentDeleteResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")
    blocked_message = _document_delete_block_message(document)
    if blocked_message is not None:
        raise HTTPException(status_code=409, detail=blocked_message)

    try:
        _delete_document_with_audit(db, document)
    except DocumentDeletionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"文档删除未完成，记录已保留为 delete_failed，可稍后重试：{exc}",
        ) from exc

    return DocumentDeleteResponse(document_id=document_id, deleted=True, vector_warning=None)


def _document_delete_block_message(document: Document) -> str | None:
    if document.status in {"uploaded", "index_queued", "indexing"}:
        return "文档正在解析或构建索引，请等待处理完成后再删除"
    if document.status == "deleting":
        return "文档正在删除，请稍后刷新列表"
    return None


def _delete_document_with_audit(db: Session, document: Document) -> None:
    document_id = document.document_id
    try:
        delete_document_record(db, document)
    except DocumentDeletionError as exc:
        log_action(db, "document_delete_failed", "document", document_id, str(exc)[:500])
        raise
    log_action(db, "document_deleted", "document", document_id, "document deleted")


def _batch_file_request_id(batch_id: str, index: int, filename: str) -> str:
    fingerprint = hashlib.sha256(f"{index}:{filename}".encode("utf-8")).hexdigest()[:16]
    return f"batch:{batch_id}:{index}:{fingerprint}"[:128]


def _to_detail(
    document: Document,
    db: Session,
    *,
    chunk_offset: int = 0,
    chunk_limit: int = 50,
) -> DocumentDetailResponse:
    chunk_total = db.scalar(
        select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document.document_id)
    ) or 0
    chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.document_id)
        .order_by(DocumentChunk.id.asc())
        .offset(chunk_offset)
        .limit(chunk_limit)
    ).all()
    return DocumentDetailResponse(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=_iso_utc(document.uploaded_at),
        status=document.status,
        index_version=document.index_version,
        index_error=document.index_error,
        metadata=_document_metadata(document),
        chunks=[
            ChunkSummary(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                text_preview=chunk.text[:120],
                chunk_type=_chunk_type(chunk.text),
                is_truncated=len(chunk.text) > 120,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                token_count=chunk.token_count,
                index_status=chunk.index_status,
                index_version=chunk.index_version,
                metadata=_chunk_metadata(chunk),
                created_at=_iso_utc(chunk.created_at),
            )
            for chunk in chunks
        ],
        chunk_total=chunk_total,
        chunk_offset=chunk_offset,
        chunk_limit=chunk_limit,
    )


def _chunk_type(text: str) -> str:
    stripped = text.lstrip()
    table_prefixes = ("表格：", "表格摘要：", "表格行证据：", "琛ㄦ牸锛?", "琛ㄦ牸琛岃瘉鎹細")
    return "table" if stripped.startswith(table_prefixes) or "\n|" in text else "paragraph"


def _chunk_metadata(chunk: DocumentChunk) -> dict:
    if not chunk.chunk_metadata:
        return {}
    try:
        value = json.loads(chunk.chunk_metadata)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _to_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=_iso_utc(document.uploaded_at),
        status=document.status,
        index_version=document.index_version,
        index_error=document.index_error,
        metadata=_document_metadata(document),
    )


def _document_metadata(document: Document) -> dict:
    return document_metadata_snapshot(document)


def _parse_metadata_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json 必须是 JSON 对象") from exc
    if not isinstance(value, dict):
        raise ValueError("metadata_json 必须是 JSON 对象")
    payload = DocumentMetadataInput.model_validate(value)
    return _metadata_input_dict(payload)


def _metadata_input_dict(payload: DocumentMetadataInput) -> dict:
    value = payload.model_dump(exclude_none=True)
    extra = value.pop("extra", {})
    return {**value, **(extra if isinstance(extra, dict) else {})}


def _metadata_patch_dict(payload: DocumentMetadataInput) -> dict:
    """Preserve explicit nulls so PATCH can represent a reviewed unknown value."""

    value = payload.model_dump(exclude_unset=True)
    extra = value.pop("extra", {})
    return {**value, **(extra if isinstance(extra, dict) else {})}


def _record_identity_audit(
    db: Session,
    action: str,
    document: Document,
    before: dict,
    after: dict,
    *,
    metadata_refreshed: bool,
    refresh_warning: str | None,
) -> None:
    fields = (*IDENTITY_METADATA_FIELDS, "identity_review_status", "identity_reviewed_at")
    before_values = {key: before.get(key) for key in fields}
    after_values = {key: after.get(key) for key in fields}
    changed_fields = [key for key in fields if before_values.get(key) != after_values.get(key)]
    current_hash = identity_snapshot_hash(document)
    if metadata_refreshed:
        user_message = "身份信息已保存并同步到检索元数据。"
    elif refresh_warning:
        user_message = "身份信息已保存，检索元数据待刷新。"
    else:
        user_message = "身份信息已保存。"
    record_event(
        db,
        action,
        "document",
        document.document_id,
        detail=f"文档身份信息变更：{', '.join(changed_fields) if changed_fields else '确认状态刷新'}",
        event_key=f"{action}:{document.document_id}:{current_hash}",
        summary="文档身份信息已更新" if action.endswith("updated") else "文档身份卡已人工核对",
        user_message=user_message,
        details={
            "filename": document.filename,
            "changed_fields": changed_fields,
            "before": before_values,
            "after": after_values,
            "identity_snapshot_hash": current_hash,
            "metadata_refreshed": metadata_refreshed,
            "refresh_warning": refresh_warning,
        },
    )


def _to_upload_response(
    document: Document,
    task: DocumentIndexTask,
) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        document_id=document.document_id,
        task_id=task.task_id or str(task.id),
        status=task.status,
        stage=task.stage,
        filename=document.filename,
        content_type=document.content_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=_iso_utc(document.uploaded_at),
        metadata=_document_metadata(document),
    )


def _to_conflict_existing_document(
    document: Document | None,
) -> DocumentConflictExistingDocument | None:
    if document is None:
        return None
    return DocumentConflictExistingDocument(
        document_id=document.document_id,
        filename=document.filename,
        size=document.size,
        file_sha256=document.file_sha256,
        status=document.status,
        uploaded_at=_iso_utc(document.uploaded_at),
        chunk_count=document.chunk_count,
    )


def _upload_conflict_response(exc: DocumentUploadConflictError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "detail": str(exc),
            "error": {
                "code": exc.code,
                "message": str(exc),
                "stage": "upload_preflight",
                "retryable": False,
                "request_id": None,
            },
            "conflict": {
                "code": exc.code,
                "existing_document": (
                    _to_conflict_existing_document(exc.existing_document).model_dump()
                    if exc.existing_document is not None
                    else None
                ),
            },
        },
    )


def _to_processing_response(
    document: Document,
    task: DocumentIndexTask,
) -> DocumentProcessingResponse:
    error = None
    if task.error_code:
        error = ApiError(
            code=task.error_code,
            message=task.last_error or "文档处理失败",
            stage=task.stage or "failed",
            retryable=task.retry_count <= INDEX_TASK_MAX_RETRIES,
            request_id=task.task_id,
        )
    return DocumentProcessingResponse(
        document_id=document.document_id,
        task_id=task.task_id or str(task.id),
        status=task.status,
        stage=task.stage or "queued",
        completed_units=task.completed_units,
        total_units=task.total_units,
        error_code=task.error_code,
        error_message=task.last_error,
        error=error,
        retry_count=task.retry_count,
        updated_at=_iso_utc(task.updated_at),
    )


def _iso_utc(value: datetime) -> str:
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()
