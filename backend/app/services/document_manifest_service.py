from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path, PureWindowsPath
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.models.document import Document
from backend.app.services.document_metadata_service import (
    DocumentMetadataError,
    apply_document_metadata,
    normalize_metadata_input,
)
from backend.app.services.document_metadata_index_service import refresh_document_metadata_indexes
from backend.app.services.vector_store_service import VectorStoreError


class ManifestImportError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestImportResult:
    record_index: int
    status: str
    document_id: str | None
    filename: str | None
    message: str | None = None


def parse_manifest(content: bytes, filename: str) -> list[dict[str, Any]]:
    if not content:
        raise ManifestImportError("manifest 不能为空")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ManifestImportError("manifest 必须使用 UTF-8 编码") from exc
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        records = [dict(row) for row in csv.DictReader(StringIO(text))]
    elif suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestImportError(f"JSONL 第 {line_number} 行不是有效 JSON") from exc
            if not isinstance(value, dict):
                raise ManifestImportError(f"JSONL 第 {line_number} 行必须是对象")
            records.append(value)
    elif suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestImportError("manifest JSON 格式错误") from exc
        if isinstance(payload, dict):
            candidates = payload.get("files") or payload.get("documents") or payload.get("records")
            records = candidates if isinstance(candidates, list) else [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ManifestImportError("manifest JSON 顶层必须是对象或数组")
    else:
        raise ManifestImportError("manifest 仅支持 .json、.jsonl、.csv")
    if not records or not all(isinstance(item, dict) for item in records):
        raise ManifestImportError("manifest 中没有有效记录")
    if any(_looks_like_qa_record(item) for item in records):
        raise ManifestImportError("检测到 QA/答案数据；评测数据禁止导入生产知识库")
    return [dict(item) for item in records]


def import_manifest_records(db: Session, records: list[dict[str, Any]]) -> list[ManifestImportResult]:
    results: list[ManifestImportResult] = []
    changed = False
    for index, record in enumerate(records, start=1):
        try:
            matches = _matching_documents(db, record)
            filename = _record_filename(record)
            if not matches:
                results.append(ManifestImportResult(index, "not_found", None, filename, "未找到 SHA、doc_id 或文件名匹配的文档"))
                continue
            if len(matches) > 1:
                results.append(ManifestImportResult(index, "ambiguous", None, filename, "匹配到多个文档，请补充 SHA-256 或官方 doc_id"))
                continue
            document = matches[0]
            sanitized_record = _sanitize_record_for_import(record, document=document)
            metadata = normalize_metadata_input(sanitized_record)
            _validate_integrity(document, record)
            apply_document_metadata(document, metadata, source="manifest", confidence=0.95)
            try:
                refresh_document_metadata_indexes(db, document)
                refresh_status = "updated"
                refresh_message = None
            except VectorStoreError as exc:
                # SQLite/Chunk metadata remains authoritative and committed;
                # the result makes the pending vector payload refresh visible.
                refresh_status = "updated"
                refresh_message = f"metadata 已提交，Qdrant payload 待刷新：{exc}"
            changed = True
            results.append(
                ManifestImportResult(
                    index,
                    refresh_status,
                    document.document_id,
                    document.filename,
                    refresh_message,
                )
            )
        except (ManifestImportError, DocumentMetadataError, ValueError) as exc:
            results.append(ManifestImportResult(index, "invalid", None, _record_filename(record), str(exc)))
    if changed:
        db.commit()
    return results


def _matching_documents(db: Session, record: dict[str, Any]) -> list[Document]:
    sha256 = str(record.get("sha256") or record.get("file_sha256") or "").strip().lower()
    external_doc_id = str(record.get("doc_id") or record.get("external_doc_id") or "").strip()
    filename = _record_filename(record)
    if sha256:
        rows = list(db.scalars(select(Document).where(Document.file_sha256 == sha256)).all())
        if rows:
            return rows
    if external_doc_id:
        rows = list(
            db.scalars(
                select(Document).where(
                    or_(Document.external_doc_id == external_doc_id, Document.document_id == external_doc_id)
                )
            ).all()
        )
        if rows:
            return rows
    if not filename:
        return []
    normalized = filename.casefold()
    return list(db.scalars(select(Document).where(Document.filename_norm == normalized)).all())


def _validate_integrity(document: Document, record: dict[str, Any]) -> None:
    expected_sha = str(record.get("sha256") or record.get("file_sha256") or "").strip().lower()
    if expected_sha and document.file_sha256 and expected_sha != document.file_sha256.lower():
        raise ManifestImportError("manifest SHA-256 与已入库文件不一致")
    expected_size = record.get("size") or record.get("file_size")
    if expected_size not in (None, "") and int(expected_size) != document.size:
        raise ManifestImportError("manifest 文件大小与已入库文件不一致")


def _sanitize_record_for_import(record: dict[str, Any], *, document: Document | None = None) -> dict[str, Any]:
    """Keep descriptive metadata and drop legacy source-review bookkeeping.

    Manifest records may carry review/provenance fields from older tooling;
    they are not part of resume knowledge base metadata, so drop them.
    """

    sanitized = dict(record)
    for key in (
        "match_status",
        "match_reason",
        "official_match_status",
        "provenance_status",
        "reviewer",
        "reviewed_at",
        "review_note",
        "candidate_score",
        "candidate_title",
        "candidate_attachment_name",
    ):
        sanitized.pop(key, None)
    return sanitized


def _json_object(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _record_filename(record: dict[str, Any]) -> str | None:
    raw = record.get("original_name") or record.get("filename") or record.get("local_path") or record.get("path")
    if not raw:
        return None
    text = str(raw)
    return PureWindowsPath(text).name if "\\" in text or ":" in text else Path(text).name


def _looks_like_qa_record(record: dict[str, Any]) -> bool:
    keys = {str(key).strip().lower() for key in record}
    return "question" in keys and bool(keys & {"answer", "answer_text", "evidence", "option_a"})
