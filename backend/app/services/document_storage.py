from datetime import datetime
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile, ZipFile
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import (
    DOCUMENT_DIR,
    MAX_OOXML_ENTRIES,
    MAX_OOXML_UNCOMPRESSED_BYTES,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    SUPPORTED_DOCUMENT_MIME_TYPES,
)
from backend.app.models.document import Document


class UnsupportedDocumentTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class DocumentTooLargeError(ValueError):
    pass


class AsyncUploadStream(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class StoredOriginalDocument:
    path: Path
    size: int
    file_sha256: str


def next_document_id(db: Session) -> str:
    """Generate a collision-resistant ID without a read-then-increment race.

    Existing sequential ``DOC-YYYYMMDD-0001`` IDs remain valid. New uploads use
    a UUID suffix, so concurrent workers do not need a shared sequence lock.
    """

    today = datetime.now().strftime("%Y%m%d")
    prefix = f"DOC-{today}-"
    for _ in range(5):
        document_id = f"{prefix}{uuid4().hex[:12].upper()}"
        existing = db.scalar(select(Document.document_id).where(Document.document_id == document_id))
        if existing is None and not any(DOCUMENT_DIR.glob(f"{document_id}.*")):
            return document_id
    raise RuntimeError("无法生成唯一文档编号，请重试上传")


def save_original_document(
    document_id: str,
    filename: str,
    content: bytes,
    content_type: str | None = None,
) -> StoredOriginalDocument:
    if not content:
        raise EmptyDocumentError("上传文档不能为空")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise UnsupportedDocumentTypeError("仅支持 .txt、.md、.doc、.docx、.pdf、.xls、.xlsx、.csv、.jsonl、.html 文档")
    _validate_mime_type(suffix, content_type)
    _validate_file_content(suffix, content)

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = DOCUMENT_DIR / f"{document_id}{suffix}"
    if storage_path.exists():
        raise UnsupportedDocumentTypeError("文档存储路径已存在，请刷新后重试上传")
    storage_path.write_bytes(content)
    return StoredOriginalDocument(
        path=storage_path,
        size=len(content),
        file_sha256=hashlib.sha256(content).hexdigest(),
    )


async def save_original_document_stream(
    *,
    document_id: str,
    filename: str,
    stream: AsyncUploadStream,
    content_type: str | None,
    max_bytes: int,
) -> StoredOriginalDocument:
    """Write an upload incrementally, validate it, then publish atomically."""

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise UnsupportedDocumentTypeError("仅支持 .txt、.md、.doc、.docx、.pdf、.xls、.xlsx、.csv、.jsonl、.html 文档")
    _validate_mime_type(suffix, content_type)
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = DOCUMENT_DIR / f"{document_id}{suffix}"
    temporary_path = storage_path.with_name(f"{storage_path.name}.upload")
    if storage_path.exists() or temporary_path.exists():
        raise UnsupportedDocumentTypeError("文档存储路径已存在，请刷新后重试上传")

    total = 0
    sha256 = hashlib.sha256()
    try:
        with temporary_path.open("xb") as target:
            while data := await stream.read(1024 * 1024):
                total += len(data)
                if total > max_bytes:
                    raise DocumentTooLargeError(
                        f"文件超过上传大小限制（{max_bytes // 1024 // 1024} MB）"
                    )
                sha256.update(data)
                target.write(data)
        if total == 0:
            raise EmptyDocumentError("上传文档不能为空")
        _validate_file_path(suffix, temporary_path)
        temporary_path.replace(storage_path)
        return StoredOriginalDocument(path=storage_path, size=total, file_sha256=sha256.hexdigest())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_mime_type(suffix: str, content_type: str | None) -> None:
    if not content_type:
        return

    normalized = content_type.split(";", maxsplit=1)[0].strip().lower()
    allowed = SUPPORTED_DOCUMENT_MIME_TYPES[suffix]
    if normalized not in allowed:
        raise UnsupportedDocumentTypeError(f"文件 MIME 类型与扩展名不匹配：{content_type}")


def _validate_file_content(suffix: str, content: bytes) -> None:
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise UnsupportedDocumentTypeError("PDF 文件内容校验失败")
        return

    if suffix in {".docx", ".xlsx"}:
        if not content.startswith(b"PK"):
            raise UnsupportedDocumentTypeError(f"{suffix.upper().lstrip('.')} 文件内容校验失败")
        try:
            with ZipFile(BytesIO(content)) as archive:
                _validate_ooxml_resource_limits(archive)
                names = set(archive.namelist())
        except BadZipFile as exc:
            raise UnsupportedDocumentTypeError(f"{suffix.upper().lstrip('.')} 文件内容校验失败") from exc
        if suffix == ".docx" and ("[Content_Types].xml" not in names or "word/document.xml" not in names):
            raise UnsupportedDocumentTypeError("DOCX 文件内容校验失败")
        if suffix == ".xlsx" and ("[Content_Types].xml" not in names or "xl/workbook.xml" not in names):
            raise UnsupportedDocumentTypeError("XLSX 文件内容校验失败")
        return

    if suffix in {".doc", ".xls"}:
        if not content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            raise UnsupportedDocumentTypeError(f"{suffix.upper().lstrip('.')} 文件内容校验失败")
        return

    if suffix in {".txt", ".md", ".csv", ".jsonl", ".html", ".htm"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnsupportedDocumentTypeError("文本文件必须使用 UTF-8 编码") from exc
        if "\x00" in text:
            raise UnsupportedDocumentTypeError("文本文件内容校验失败")


def _validate_file_path(suffix: str, path: Path) -> None:
    with path.open("rb") as stream:
        signature = stream.read(8)
    if suffix == ".pdf":
        if not signature.startswith(b"%PDF-"):
            raise UnsupportedDocumentTypeError("PDF 文件内容校验失败")
        return
    if suffix in {".docx", ".xlsx"}:
        if not signature.startswith(b"PK"):
            raise UnsupportedDocumentTypeError(f"{suffix.upper().lstrip('.')} 文件内容校验失败")
        try:
            with ZipFile(path) as archive:
                _validate_ooxml_resource_limits(archive)
                names = set(archive.namelist())
        except BadZipFile as exc:
            raise UnsupportedDocumentTypeError(f"{suffix.upper().lstrip('.')} 文件内容校验失败") from exc
        if suffix == ".docx" and ("[Content_Types].xml" not in names or "word/document.xml" not in names):
            raise UnsupportedDocumentTypeError("DOCX 文件内容校验失败")
        if suffix == ".xlsx" and ("[Content_Types].xml" not in names or "xl/workbook.xml" not in names):
            raise UnsupportedDocumentTypeError("XLSX 文件内容校验失败")
        return
    if suffix in {".doc", ".xls"}:
        if not signature.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            raise UnsupportedDocumentTypeError(f"{suffix.upper().lstrip('.')} 文件内容校验失败")
        return
    if suffix in {".txt", ".md", ".csv", ".jsonl", ".html", ".htm"}:
        try:
            with path.open("r", encoding="utf-8-sig") as stream:
                while text := stream.read(1024 * 1024):
                    if "\x00" in text:
                        raise UnsupportedDocumentTypeError("文本文件内容校验失败")
        except UnicodeDecodeError as exc:
            raise UnsupportedDocumentTypeError("文本文件必须使用 UTF-8 编码") from exc


def _validate_ooxml_resource_limits(archive: ZipFile) -> None:
    entries = archive.infolist()
    if len(entries) > MAX_OOXML_ENTRIES:
        raise DocumentTooLargeError(
            f"Office 文档包含过多压缩条目（上限 {MAX_OOXML_ENTRIES}）"
        )
    uncompressed_bytes = sum(max(0, entry.file_size) for entry in entries)
    if uncompressed_bytes > MAX_OOXML_UNCOMPRESSED_BYTES:
        limit_mb = MAX_OOXML_UNCOMPRESSED_BYTES // 1024 // 1024
        raise DocumentTooLargeError(
            f"Office 文档解压后超过大小限制（{limit_mb} MB）"
        )
