import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Mapping
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.config import INDEX_VERSION, MAX_UPLOAD_BYTES
from backend.app.models.document import Document, DocumentIndexTask
from backend.app.services.audit_service import log_action
from backend.app.services.document_lifecycle_service import (
    delete_document_record,
    resolve_original_path,
)
from backend.app.services.document_storage import (
    next_document_id,
    save_original_document_stream,
)
from backend.app.services.document_metadata_service import (
    apply_document_metadata,
)
from backend.app.services.index_task_service import enqueue_document_index


class UploadTaskMissingError(RuntimeError):
    pass


class DocumentUploadConflictError(RuntimeError):
    def __init__(self, code: str, message: str, existing_document: Document | None = None):
        super().__init__(message)
        self.code = code
        self.existing_document = existing_document


class ExactDuplicateDocumentError(DocumentUploadConflictError):
    def __init__(self, existing_document: Document):
        super().__init__("exact_duplicate", "文件已存在于知识库中。", existing_document)


class FilenameConflictDocumentError(DocumentUploadConflictError):
    def __init__(self, existing_document: Document):
        super().__init__("name_conflict", "知识库中已存在同名文件。", existing_document)


class OverwriteDocumentBusyError(DocumentUploadConflictError):
    def __init__(self, existing_document: Document):
        super().__init__("overwrite_target_busy", "旧文档正在处理或删除，暂不能覆盖。", existing_document)


@dataclass(frozen=True)
class UploadPreflightInput:
    client_file_id: str
    filename: str
    size: int
    file_sha256: str


@dataclass(frozen=True)
class UploadPreflightResult:
    client_file_id: str
    filename: str
    status: str
    existing_document: Document | None = None
    error_message: str | None = None


async def create_document_upload(
    db: Session,
    *,
    file: UploadFile,
    request_id: str,
    filename_override: str | None = None,
    overwrite_document_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    metadata_source: str = "user",
    max_bytes: int = MAX_UPLOAD_BYTES,
    enqueue_index: Callable[[str], bool] = enqueue_document_index,
) -> tuple[Document, DocumentIndexTask]:
    filename = sanitize_upload_filename(filename_override or file.filename or "unknown")
    filename_norm = normalize_filename(filename)
    normalized_request_id = request_id.strip()[:128]
    existing = db.scalar(
        select(Document).where(Document.client_request_id == normalized_request_id)
    )
    if existing is not None:
        task = db.scalar(
            select(DocumentIndexTask).where(
                DocumentIndexTask.document_id == existing.document_id
            )
        )
        if task is None:
            raise UploadTaskMissingError("重复上传记录缺少处理任务，请重试索引")
        return existing, task

    overwrite_target: Document | None = None
    if overwrite_document_id:
        overwrite_target = db.scalar(
            select(Document).where(Document.document_id == overwrite_document_id)
        )
        if overwrite_target is None:
            raise DocumentUploadConflictError("overwrite_target_not_found", "要覆盖的旧文档不存在。")
        if overwrite_target.status in {"uploaded", "index_queued", "indexing", "deleting"}:
            raise OverwriteDocumentBusyError(overwrite_target)

    document_id = next_document_id(db)
    storage_path: Path | None = None
    try:
        stored = await save_original_document_stream(
            document_id=document_id,
            filename=filename,
            stream=file,
            content_type=file.content_type,
            max_bytes=max_bytes,
        )
        storage_path = stored.path
        exact_duplicate = find_exact_duplicate(
            db,
            size=stored.size,
            file_sha256=stored.file_sha256,
            exclude_document_id=overwrite_document_id,
        )
        if exact_duplicate is not None:
            raise ExactDuplicateDocumentError(exact_duplicate)
        filename_conflict = find_filename_conflict(
            db,
            filename_norm=filename_norm,
            exclude_document_id=overwrite_document_id,
        )
        if filename_conflict is not None:
            raise FilenameConflictDocumentError(filename_conflict)
        if overwrite_target is not None:
            delete_document_record(db, overwrite_target)
    except Exception:
        if storage_path is not None:
            storage_path.unlink(missing_ok=True)
        raise

    document = Document(
        document_id=document_id,
        client_request_id=normalized_request_id,
        filename=filename,
        filename_norm=filename_norm,
        content_type=file.content_type,
        file_type=Path(filename).suffix.lower().lstrip("."),
        size=stored.size,
        file_sha256=stored.file_sha256,
        storage_path=str(storage_path),
        title=Path(filename).stem.split("_", maxsplit=1)[-1],
        source_type="uploaded_file",
        metadata_status="inferred",
        document_metadata=json.dumps({}, ensure_ascii=False),
        status="uploaded",
        index_version=INDEX_VERSION,
        index_error=None,
        chunk_count=0,
    )
    apply_document_metadata(
        document,
        {"title": document.title, "source_type": "uploaded_file"},
        source="filename",
        confidence=0.4,
    )
    if metadata:
        apply_document_metadata(
            document,
            metadata,
            source=metadata_source,
            confidence=1.0 if metadata_source == "user" else 0.95,
        )
    task = DocumentIndexTask(
        task_id=uuid4().hex,
        document_id=document_id,
        status="queued",
        stage="queued",
    )
    db.add(document)
    db.add(task)
    db.flush()
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        storage_path.unlink(missing_ok=True)
        raise DocumentUploadConflictError("name_conflict", "知识库中已存在同名文件。") from exc
    except Exception:
        db.rollback()
        storage_path.unlink(missing_ok=True)
        raise
    db.refresh(document)
    db.refresh(task)
    log_action(db, "document_uploaded", "document", document_id, filename)
    enqueue_index(document_id)
    db.refresh(task)
    return document, task


def build_upload_preflight(
    db: Session,
    items: list[UploadPreflightInput],
) -> list[UploadPreflightResult]:
    ensure_existing_document_hashes(db)
    norm_counts: dict[str, int] = {}
    norm_by_file_id: dict[str, str] = {}
    for item in items:
        try:
            filename = sanitize_upload_filename(item.filename)
            filename_norm = normalize_filename(filename)
        except ValueError:
            filename_norm = ""
        norm_by_file_id[item.client_file_id] = filename_norm
        if filename_norm:
            norm_counts[filename_norm] = norm_counts.get(filename_norm, 0) + 1

    results: list[UploadPreflightResult] = []
    for item in items:
        try:
            filename = sanitize_upload_filename(item.filename)
            filename_norm = norm_by_file_id[item.client_file_id]
        except ValueError as exc:
            results.append(
                UploadPreflightResult(
                    client_file_id=item.client_file_id,
                    filename=item.filename,
                    status="name_conflict",
                    error_message=str(exc),
                )
            )
            continue
        if norm_counts.get(filename_norm, 0) > 1:
            results.append(
                UploadPreflightResult(
                    client_file_id=item.client_file_id,
                    filename=filename,
                    status="selection_name_conflict",
                    error_message="本次选择中存在同名文件，请重命名或跳过其中一个。",
                )
            )
            continue
        exact_duplicate = find_exact_duplicate(
            db,
            size=item.size,
            file_sha256=item.file_sha256.lower(),
        )
        if exact_duplicate is not None:
            results.append(
                UploadPreflightResult(
                    client_file_id=item.client_file_id,
                    filename=filename,
                    status="exact_duplicate",
                    existing_document=exact_duplicate,
                    error_message="文件已存在于知识库中。",
                )
            )
            continue
        filename_conflict = find_filename_conflict(db, filename_norm=filename_norm)
        if filename_conflict is not None:
            results.append(
                UploadPreflightResult(
                    client_file_id=item.client_file_id,
                    filename=filename,
                    status="name_conflict",
                    existing_document=filename_conflict,
                    error_message="知识库中已存在同名文件。",
                )
            )
            continue
        results.append(
            UploadPreflightResult(
                client_file_id=item.client_file_id,
                filename=filename,
                status="ready",
            )
        )
    return results


def sanitize_upload_filename(filename: str) -> str:
    basename = _basename(filename)
    normalized = unicodedata.normalize("NFC", basename).strip()
    if not normalized:
        raise ValueError("文件名不能为空。")
    if len(normalized) > 255:
        raise ValueError("文件名不能超过 255 个字符。")
    return normalized


def normalize_filename(filename: str) -> str:
    return sanitize_upload_filename(filename).casefold()


def find_exact_duplicate(
    db: Session,
    *,
    size: int,
    file_sha256: str,
    exclude_document_id: str | None = None,
) -> Document | None:
    statement = (
        select(Document)
        .where(Document.size == size)
        .where(Document.file_sha256 == file_sha256.lower())
        .order_by(Document.uploaded_at.asc(), Document.id.asc())
    )
    if exclude_document_id:
        statement = statement.where(Document.document_id != exclude_document_id)
    return db.scalar(statement)


def find_filename_conflict(
    db: Session,
    *,
    filename_norm: str,
    exclude_document_id: str | None = None,
) -> Document | None:
    statement = select(Document).where(Document.filename_norm == filename_norm)
    if exclude_document_id:
        statement = statement.where(Document.document_id != exclude_document_id)
    return db.scalar(statement)


def ensure_existing_document_hashes(db: Session) -> None:
    documents = db.scalars(
        select(Document).where(Document.file_sha256.is_(None)).order_by(Document.id.asc())
    ).all()
    changed = False
    for document in documents:
        path = resolve_original_path(document)
        if not path.exists() or not path.is_file():
            continue
        document.file_sha256 = _sha256_file(path)
        if not document.filename_norm:
            document.filename_norm = normalize_filename(document.filename)
        changed = True
    if changed:
        db.commit()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _basename(filename: str) -> str:
    if "\\" in filename or ":" in filename:
        return PureWindowsPath(filename).name
    return Path(filename).name
