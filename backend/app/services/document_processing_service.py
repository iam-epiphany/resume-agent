from __future__ import annotations

import json
from typing import Callable

from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend.app.core.config import INDEX_VERSION
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.chunk_service import build_chunks_from_parsed
from backend.app.services.document_lifecycle_service import resolve_original_path
from backend.app.services.document_parser import DocumentParseError, parse_document
from backend.app.services.document_metadata_service import (
    apply_document_metadata,
    infer_metadata_from_parsed,
    retrieval_metadata_snapshot,
)
from backend.app.services.performance_metrics import measure


class DocumentProcessingError(RuntimeError):
    pass


StageReporter = Callable[[str, int | None, int | None], None]


def ensure_document_chunks(
    db: Session,
    document: Document,
    *,
    stage_reporter: StageReporter | None = None,
    force_rebuild: bool = False,
) -> list[DocumentChunk]:
    """Parse and persist chunks when an uploaded document has none yet."""

    if document.chunks and not force_rebuild:
        return list(document.chunks)
    document_pk = document.id
    try:
        _report(stage_reporter, "parsing")
        with measure("index.parse"):
            # storage_path 可能来自容器绝对路径（/app/data/...）——统一走
            # resolve_original_path 的「原文路径 → DOCUMENT_DIR 兜底」解析，
            # 保证容器与宿主机直跑两种部署方式都能定位原始文件（2026-08-14 修复）
            parsed = parse_document(resolve_original_path(document), source_name=document.filename)
        apply_document_metadata(
            document,
            infer_metadata_from_parsed(parsed, document.filename),
            source="document_body",
            confidence=0.65,
        )
        _report(stage_reporter, "chunking")
        with measure("index.chunk_build"):
            drafts = build_chunks_from_parsed(
                document_id=document.document_id,
                parsed=parsed,
                source_file=document.filename,
                inherited_metadata=retrieval_metadata_snapshot(document),
            )
        if not drafts:
            raise DocumentParseError("文档没有可入库的文本片段")

        chunk_models = [
            DocumentChunk(
                chunk_id=draft.chunk_id,
                document_id=document.document_id,
                text=draft.text,
                embedding_text=draft.embedding_text,
                chunk_metadata=json.dumps(draft.metadata or {}, ensure_ascii=False),
                token_count=draft.token_count,
                index_status="uploaded",
                index_version=INDEX_VERSION,
                title=draft.title,
                section_title=draft.section_title,
                page_number=draft.page_number,
                source_file=document.filename,
            )
            for draft in drafts
        ]
        _report(stage_reporter, "metadata_indexing", 0, len(chunk_models))
        document.chunk_count = len(chunk_models)
        document.status = "uploaded"
        document.index_error = None
        with measure("index.sqlite_chunk_persist"):
            if force_rebuild:
                db.flush()
                db.execute(
                    delete(DocumentChunk)
                    .where(DocumentChunk.document_id == document.document_id)
                    .execution_options(synchronize_session=False)
                )
                db.commit()
                db.expunge_all()
                document = db.get(Document, document_pk)
                if document is None:
                    raise DocumentProcessingError("文档记录在重建 chunk 时已不存在")
                document.chunk_count = len(chunk_models)
                document.status = "uploaded"
                document.index_error = None
            db.add_all(chunk_models)
            db.flush()
            db.commit()
        _report(
            stage_reporter,
            "metadata_indexing",
            len(chunk_models),
            len(chunk_models),
        )
        return chunk_models
    except DocumentParseError as exc:
        db.rollback()
        document = db.get(Document, document.id) or document
        document.status = "index_failed"
        document.index_error = str(exc)[:1000]
        db.commit()
        raise DocumentProcessingError(str(exc)) from exc
    except Exception as exc:
        db.rollback()
        document = db.get(Document, document.id) or document
        document.status = "index_failed"
        document.index_error = f"文档处理失败：{exc}"[:1000]
        db.commit()
        raise DocumentProcessingError(str(exc)) from exc


def _report(
    reporter: StageReporter | None,
    stage: str,
    completed_units: int | None = None,
    total_units: int | None = None,
) -> None:
    if reporter is not None:
        reporter(stage, completed_units, total_units)
