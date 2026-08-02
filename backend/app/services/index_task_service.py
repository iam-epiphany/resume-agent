from __future__ import annotations

from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Lock, Thread
from uuid import uuid4

from sqlalchemy import select

from backend.app.core.config import INDEX_QUEUE_CAPACITY, INDEX_TASK_MAX_RETRIES
from backend.app.core.database import SessionLocal
from backend.app.models.document import Document, DocumentIndexTask
from backend.app.services.audit_service import log_action
from backend.app.services.document_indexing_service import DocumentIndexingError, index_document
from backend.app.services.document_processing_service import DocumentProcessingError, ensure_document_chunks
from backend.app.services.performance_metrics import trace_operation


_QUEUE: Queue[str] = Queue(maxsize=max(1, INDEX_QUEUE_CAPACITY))
_PENDING: set[str] = set()
_FORCE_REBUILD_REQUESTS: set[str] = set()
_LOCK = Lock()
_STARTED = False


def start_index_task_worker() -> None:
    """Start one bounded indexing worker and recover durable queued work."""

    global _STARTED
    newly_started = False
    with _LOCK:
        if not _STARTED:
            Thread(target=_worker_loop, name="resumemind-index-worker", daemon=True).start()
            _STARTED = True
            newly_started = True
    if newly_started:
        _recover_interrupted_tasks()
    _fill_queue_from_db()


def enqueue_document_index(document_id: str, *, force_rebuild_chunks: bool = False) -> bool:
    """Persist a task before attempting to place it in the bounded memory queue."""

    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        if document is None:
            return False
        if force_rebuild_chunks:
            with _LOCK:
                _FORCE_REBUILD_REQUESTS.add(document_id)
        task = db.scalar(
            select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
        )
        if task is None:
            task = DocumentIndexTask(document_id=document_id, task_id=uuid4().hex)
            db.add(task)
        elif not task.task_id:
            task.task_id = uuid4().hex
        document.status = "index_queued"
        document.index_error = None
        if task.status in {"queued", "running"}:
            if force_rebuild_chunks:
                task.stage = "queued_rebuild"
            task.updated_at = _now()
            db.commit()
            start_index_task_worker()
            return _schedule_in_memory(document_id)
        task.status = "queued"
        task.stage = "queued_rebuild" if force_rebuild_chunks else "queued"
        task.completed_units = None
        task.total_units = None
        task.retry_count = 0
        task.last_error = None
        task.error_code = None
        task.updated_at = _now()
        db.commit()
    start_index_task_worker()
    return _schedule_in_memory(document_id)


def index_task_status_counts() -> dict[str, int]:
    with SessionLocal() as db:
        rows = db.execute(
            select(DocumentIndexTask.status, DocumentIndexTask.id)
        ).all()
    counts: dict[str, int] = {}
    for status, _ in rows:
        counts[str(status)] = counts.get(str(status), 0) + 1
    counts["queue_capacity"] = _QUEUE.maxsize
    counts["queue_depth"] = _QUEUE.qsize()
    return counts


def _worker_loop() -> None:
    while True:
        try:
            document_id = _QUEUE.get(timeout=5)
        except Empty:
            try:
                _fill_queue_from_db()
            except Exception:
                pass
            continue
        retry = False
        try:
            retry = _run_task(document_id)
        except Exception as exc:
            _record_unexpected_failure(document_id, exc)
        finally:
            with _LOCK:
                _PENDING.discard(document_id)
            _QUEUE.task_done()
        if retry:
            _schedule_in_memory(document_id)
        try:
            _fill_queue_from_db()
        except Exception:
            pass


@trace_operation("document_index")
def _run_task(document_id: str) -> bool:
    with SessionLocal() as db:
        task = db.scalar(
            select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
        )
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        if task is None or document is None:
            return False
        with _LOCK:
            force_rebuild_chunks = document_id in _FORCE_REBUILD_REQUESTS
        force_rebuild_chunks = force_rebuild_chunks or task.stage == "queued_rebuild"
        task.status = "running"
        task.stage = task.stage or "queued"
        task.updated_at = _now()
        db.commit()
        task_pk = task.id

        def stage_reporter(
            stage: str,
            completed_units: int | None = None,
            total_units: int | None = None,
        ) -> None:
            _update_task_stage(db, task_pk, stage, completed_units, total_units)

        try:
            ensure_document_chunks(
                db,
                document,
                stage_reporter=stage_reporter,
                force_rebuild=force_rebuild_chunks,
            )
            document = db.scalar(select(Document).where(Document.document_id == document_id))
            if document is None:
                return False
            index_document(db, document, stage_reporter=stage_reporter)
        except (DocumentProcessingError, DocumentIndexingError) as exc:
            task = db.get(DocumentIndexTask, task_pk)
            task.retry_count += 1
            task.last_error = str(exc)[:2000]
            task.error_code = (
                "document_parse_failed"
                if isinstance(exc, DocumentProcessingError)
                else "document_index_failed"
            )
            task.updated_at = _now()
            should_retry = task.retry_count <= INDEX_TASK_MAX_RETRIES
            task.status = "queued" if should_retry else "failed"
            if force_rebuild_chunks and should_retry:
                task.stage = "queued_rebuild"
                with _LOCK:
                    _FORCE_REBUILD_REQUESTS.add(document_id)
            elif force_rebuild_chunks:
                with _LOCK:
                    _FORCE_REBUILD_REQUESTS.discard(document_id)
            db.commit()
            log_action(
                db,
                "document_index_retry" if should_retry else "document_index_failed",
                "document",
                document_id,
                f"retry={task.retry_count}; {str(exc)[:500]}",
            )
            return should_retry
        task = db.get(DocumentIndexTask, task_pk)
        if task is None:
            return False
        task.status = "completed"
        task.stage = "completed"
        task.completed_units = task.total_units
        task.last_error = None
        task.error_code = None
        task.updated_at = _now()
        db.commit()
        with _LOCK:
            _FORCE_REBUILD_REQUESTS.discard(document_id)
        log_action(db, "document_indexed", "document", document_id, "bounded queue completed")
        return False


def _recover_interrupted_tasks() -> None:
    with SessionLocal() as db:
        tasks = db.scalars(
            select(DocumentIndexTask).where(DocumentIndexTask.status == "running")
        ).all()
        for task in tasks:
            task.status = "queued"
            task.stage = task.stage or "queued"
            if not task.task_id:
                task.task_id = uuid4().hex
            task.last_error = "应用重启后自动恢复未完成的索引任务。"
            task.updated_at = _now()

        interrupted_documents = db.scalars(
            select(Document).where(
                Document.status == "index_failed",
                Document.index_error == "应用重启时检测到未完成的索引任务，请重试构建索引。",
            )
        ).all()
        known = {
            task.document_id
            for task in db.scalars(select(DocumentIndexTask)).all()
        }
        for document in interrupted_documents:
            if document.document_id not in known:
                db.add(
                    DocumentIndexTask(
                        document_id=document.document_id,
                        task_id=uuid4().hex,
                        status="queued",
                        stage="queued",
                        last_error="从旧版索引状态自动恢复。",
                    )
                )
        db.commit()


def _fill_queue_from_db() -> None:
    available = _QUEUE.maxsize - _QUEUE.qsize()
    if available <= 0:
        return
    with SessionLocal() as db:
        document_ids = db.scalars(
            select(DocumentIndexTask.document_id)
            .where(DocumentIndexTask.status == "queued")
            .order_by(DocumentIndexTask.updated_at.asc(), DocumentIndexTask.id.asc())
            .limit(available)
        ).all()
    for document_id in document_ids:
        _schedule_in_memory(document_id)


def _schedule_in_memory(document_id: str) -> bool:
    with _LOCK:
        if document_id in _PENDING:
            return True
        try:
            _QUEUE.put_nowait(document_id)
        except Full:
            return False
        _PENDING.add(document_id)
        return True


def _record_unexpected_failure(document_id: str, exc: Exception) -> None:
    try:
        with SessionLocal() as db:
            task = db.scalar(
                select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
            )
            document = db.scalar(select(Document).where(Document.document_id == document_id))
            if task is not None:
                task.status = "failed"
                task.stage = task.stage or "queued"
                task.retry_count += 1
                task.last_error = f"unexpected worker error: {exc}"[:2000]
                task.error_code = "document_worker_failed"
                task.updated_at = _now()
            if document is not None:
                document.status = "index_failed"
                document.index_error = f"索引工作线程异常：{exc}"[:1000]
            db.commit()
    except Exception:
        return


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _update_task_stage(
    db,
    task_id: int,
    stage: str,
    completed_units: int | None,
    total_units: int | None,
) -> None:
    task = db.get(DocumentIndexTask, task_id)
    if task is None:
        return
    task.stage = stage
    task.completed_units = completed_units
    task.total_units = total_units
    task.updated_at = _now()
    db.commit()
