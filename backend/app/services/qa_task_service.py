from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.config import QA_QUEUE_CAPACITY, QA_TASK_MAX_RETRIES
from backend.app.core.database import SessionLocal
from backend.app.models.document import QATask
from backend.app.schemas.qa import (
    ApiError,
    QAAnswerPreview,
    QAResponse,
    QATaskCreateResponse,
    QATaskRequest,
    QATaskStatusResponse,
    RagProgressEvent,
)
from backend.app.services.audit_service import record_event
from backend.app.services.rag_service import answer_question
from backend.app.services.retrieval_service import RetrievalServiceUnavailable


logger = logging.getLogger(__name__)

_QUEUE: Queue[str] = Queue(maxsize=max(1, QA_QUEUE_CAPACITY))
_PENDING: set[str] = set()
_LOCK = Lock()
_STARTED = False


class QATaskCancelled(Exception):
    """Stop the durable QA pipeline after a user cancellation request."""


def start_qa_task_worker() -> None:
    """Start one bounded worker and resume durable queued QA tasks."""

    global _STARTED
    newly_started = False
    with _LOCK:
        if not _STARTED:
            Thread(target=_worker_loop, name="resumemind-qa-worker", daemon=True).start()
            _STARTED = True
            newly_started = True
    if newly_started:
        _recover_interrupted_tasks()
    _fill_queue_from_db()


def create_qa_task(payload: QATaskRequest) -> QATaskCreateResponse:
    """Create or recover the task identified by the browser request id."""

    task: QATask | None = None
    with SessionLocal() as db:
        task = db.scalar(
            select(QATask).where(QATask.client_request_id == payload.client_request_id)
        )
        if task is None:
            task = QATask(
                task_id=uuid4().hex,
                client_request_id=payload.client_request_id,
                session_id=payload.session_id,
                question=payload.question.strip(),
                options_json=json.dumps(payload.options, ensure_ascii=False),
                include_debug=payload.include_debug,
                status="queued",
                progress_json="[]",
            )
            db.add(task)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                task = db.scalar(
                    select(QATask).where(
                        QATask.client_request_id == payload.client_request_id
                    )
                )
                if task is None:
                    raise
        response = QATaskCreateResponse(
            task_id=task.task_id,
            client_request_id=payload.client_request_id,
            status=task.status,
        )
    start_qa_task_worker()
    if response.status in {"queued", "running"}:
        _schedule_in_memory(response.task_id)
    return response


def get_qa_task_status(db: Session, task_id: str) -> QATaskStatusResponse | None:
    task = db.get(QATask, task_id)
    if task is None:
        return None
    return _task_to_response(task)


def cancel_qa_task(db: Session, task_id: str) -> QATaskStatusResponse | None:
    """Cancel a queued/running task and make the operation idempotent."""

    task = db.get(QATask, task_id)
    if task is None:
        return None
    if task.status in {"queued", "running"}:
        task.status = "cancelled"
        task.answer_preview_json = None
        task.error = None
        task.error_code = None
        task.error_stage = None
        task.error_retryable = False
        task.completed_at = _now()
        task.updated_at = task.completed_at
        db.commit()
        db.refresh(task)
        try:
            record_event(
                db,
                "qa_cancelled",
                "question",
                task.task_id,
                detail=f"用户停止生成。问题：{task.question}",
                severity="info",
                event_key=f"qa_cancelled:{task.task_id}",
                summary="问答生成已停止",
                user_message="用户主动停止了本次回答生成。",
                details={"question": task.question, "status": "cancelled"},
            )
        except Exception:
            db.rollback()
            logger.exception("Could not persist cancellation audit for %s", task.task_id)
        db.refresh(task)
    return _task_to_response(task)


def list_recent_qa_task_statuses(db: Session, limit: int = 5) -> list[QATaskStatusResponse]:
    safe_limit = min(max(limit, 1), 20)
    tasks = db.scalars(
        select(QATask).order_by(QATask.created_at.desc()).limit(safe_limit)
    ).all()
    return [_task_to_response(task) for task in tasks]


def qa_task_status_counts() -> dict[str, int]:
    with SessionLocal() as db:
        rows = db.execute(select(QATask.status, QATask.task_id)).all()
    counts: dict[str, int] = {}
    for status, _ in rows:
        counts[str(status)] = counts.get(str(status), 0) + 1
    counts["queue_capacity"] = _QUEUE.maxsize
    counts["queue_depth"] = _QUEUE.qsize()
    return counts


def _worker_loop() -> None:
    while True:
        try:
            task_id = _QUEUE.get(timeout=5)
        except Empty:
            try:
                _fill_queue_from_db()
            except Exception:
                logger.exception("Failed to refill QA queue")
            continue
        retry = False
        try:
            retry = _run_task(task_id)
        except Exception as exc:
            logger.exception("Unexpected QA worker failure for %s", task_id)
            _record_unexpected_failure(task_id, exc)
        finally:
            with _LOCK:
                _PENDING.discard(task_id)
            _QUEUE.task_done()
        if retry:
            _schedule_in_memory(task_id)
        try:
            _fill_queue_from_db()
        except Exception:
            logger.exception("Failed to refill QA queue")


def _run_task(task_id: str) -> bool:
    with SessionLocal() as db:
        task = db.get(QATask, task_id)
        if task is None or task.status not in {"queued", "running"}:
            return False
        task.status = "running"
        task.answer_preview_json = None
        task.attempt_count = int(task.attempt_count or 0) + 1
        task.error = None
        task.error_code = None
        task.error_stage = None
        task.error_retryable = False
        task.updated_at = _now()
        db.commit()

        options = _json_list(task.options_json)

        # 进度事件节流：每次问答 10-20 个进度事件 → 10-20 次 SQLite 写事务，
        # 与 /ask 路径并发时易触发 "database is locked"。合并为每 0.25s 最多
        # 落库一次（中间事件跳过，仅影响进度展示粒度）；失败事件必须落库供诊断。
        _progress_last_write = [0.0]

        def progress_reporter(event: dict[str, Any]) -> None:
            _raise_if_task_cancelled(db, task_id)
            now = monotonic()
            if event.get("status") != "failed" and now - _progress_last_write[0] < 0.25:
                return
            _progress_last_write[0] = now
            _append_progress_event(db, task_id, event)

        def answer_preview_reporter(preview: QAAnswerPreview) -> None:
            _raise_if_task_cancelled(db, task_id)
            _save_answer_preview(db, task_id, preview)

        # 节流版取消检查：LLM 流式每行都查库会把生成拖慢（几百行 × DB 往返），
        # 改为每 N 行或每 0.5s 检查一次；取消操作最多延迟 ~0.5s 生效，可接受。
        _cancel_check_rows = 0
        _last_cancel_check = monotonic()

        def cancellation_checker() -> None:
            nonlocal _cancel_check_rows, _last_cancel_check
            _cancel_check_rows += 1
            now = monotonic()
            if (
                _cancel_check_rows < 25
                and now - _last_cancel_check < 0.5
            ):
                return
            _cancel_check_rows = 0
            _last_cancel_check = now
            _raise_if_task_cancelled(db, task_id)

        try:
            response = answer_question(
                db,
                task.question,
                options=options,
                include_debug=task.include_debug,
                session_id=task.session_id,
                progress_reporter=progress_reporter,
                answer_preview_reporter=answer_preview_reporter,
                cancellation_checker=cancellation_checker,
            )
            db.expire_all()
            task = db.get(QATask, task_id)
            if task is None:
                return False
            if task.status == "cancelled":
                return False
            task.answer_json = response.model_dump_json()
            task.answer_preview_json = None
            task.status = "completed"
            task.error = None
            task.error_code = None
            task.error_stage = None
            task.error_retryable = False
            task.completed_at = _now()
            task.updated_at = task.completed_at
            db.commit()
            return False
        except QATaskCancelled:
            db.rollback()
            return False
        except RetrievalServiceUnavailable as exc:
            db.expire_all()
            task = db.get(QATask, task_id)
            if task is None or task.status == "cancelled":
                return False
            retryable = task.attempt_count <= QA_TASK_MAX_RETRIES
            _fail_task(
                db,
                task_id,
                str(exc),
                code="retrieval_unavailable",
                stage="retrieval",
                title="检索系统暂不可用",
                retryable=retryable,
            )
            return retryable
        except Exception:
            db.expire_all()
            task = db.get(QATask, task_id)
            if task is None or task.status == "cancelled":
                return False
            logger.exception("QA task %s failed", task_id)
            _fail_task(
                db,
                task_id,
                "问答任务执行失败。",
                code="qa_task_failed",
                stage="llm_generation",
                title="问答任务执行失败",
                retryable=False,
            )
            return False


def _append_progress_event(db: Session, task_id: str, event: dict[str, Any]) -> None:
    task = db.get(QATask, task_id)
    if task is None:
        return
    events = _json_events(task.progress_json)
    events.append(event)
    # Bound task snapshots so a long diagnostic run cannot grow without limit.
    task.progress_json = json.dumps(events[-100:], ensure_ascii=False)
    task.updated_at = _now()
    db.commit()


def _save_answer_preview(db: Session, task_id: str, preview: QAAnswerPreview) -> None:
    task = db.get(QATask, task_id)
    if task is None or task.status == "cancelled":
        return
    task.answer_preview_json = preview.model_dump_json()
    task.updated_at = _now()
    db.commit()


def _raise_if_task_cancelled(db: Session, task_id: str) -> None:
    db.expire_all()
    task = db.get(QATask, task_id)
    if task is not None and task.status == "cancelled":
        raise QATaskCancelled()


def _fail_task(
    db: Session,
    task_id: str,
    detail: str,
    *,
    code: str,
    stage: str,
    title: str,
    retryable: bool,
) -> None:
    _append_progress_event(
        db,
        task_id,
        {
            "stage": stage,
            "status": "failed",
            "title": title,
            "detail": detail,
        },
    )
    task = db.get(QATask, task_id)
    if task is None:
        return
    task.status = "queued" if retryable else "failed"
    task.answer_preview_json = None
    task.error = detail
    task.error_code = code
    task.error_stage = stage
    task.error_retryable = retryable
    task.completed_at = None if retryable else _now()
    task.updated_at = _now()
    db.commit()


def _recover_interrupted_tasks() -> None:
    with SessionLocal() as db:
        tasks = db.scalars(
            select(QATask).where(QATask.status.in_(("queued", "running")))
        ).all()
        for task in tasks:
            task.status = "queued"
            task.answer_preview_json = None
            task.error = "应用重启后正在恢复未完成的问答任务。"
            task.error_code = None
            task.error_stage = None
            task.error_retryable = False
            task.completed_at = None
            task.updated_at = _now()
        db.commit()


def _fill_queue_from_db() -> None:
    available = _QUEUE.maxsize - _QUEUE.qsize()
    if available <= 0:
        return
    with SessionLocal() as db:
        task_ids = db.scalars(
            select(QATask.task_id)
            .where(QATask.status == "queued")
            .order_by(QATask.updated_at.asc(), QATask.created_at.asc())
            .limit(available)
        ).all()
    for task_id in task_ids:
        _schedule_in_memory(task_id)


def _schedule_in_memory(task_id: str) -> bool:
    with _LOCK:
        if task_id in _PENDING:
            return True
        try:
            _QUEUE.put_nowait(task_id)
        except Full:
            return False
        _PENDING.add(task_id)
        return True


def _record_unexpected_failure(task_id: str, exc: Exception) -> None:
    try:
        with SessionLocal() as db:
            task = db.get(QATask, task_id)
            if task is None:
                return
            if task.status == "cancelled":
                return
            task.status = "failed"
            task.answer_preview_json = None
            task.error = "问答任务执行失败。"
            task.error_code = "qa_worker_failed"
            task.error_stage = "llm_generation"
            task.error_retryable = False
            task.completed_at = _now()
            task.updated_at = task.completed_at
            db.commit()
    except Exception:
        logger.exception("Could not persist QA worker failure: %s", exc)


def _task_to_response(task: QATask) -> QATaskStatusResponse:
    answer = None
    if task.answer_json:
        try:
            answer = QAResponse.model_validate_json(task.answer_json)
        except ValueError:
            answer = None
    progress_events = [
        RagProgressEvent.model_validate(event)
        for event in _json_events(task.progress_json)
        if isinstance(event, dict)
    ]
    answer_preview = None
    if task.answer_preview_json:
        try:
            answer_preview = QAAnswerPreview.model_validate_json(task.answer_preview_json)
        except ValueError:
            answer_preview = None
    error = None
    if task.error_code:
        error = ApiError(
            code=task.error_code,
            message=task.error or "问答任务执行失败。",
            stage=task.error_stage,
            retryable=bool(task.error_retryable),
            request_id=task.client_request_id or task.task_id,
        )
    return QATaskStatusResponse(
        task_id=task.task_id,
        client_request_id=task.client_request_id,
        question=task.question,
        options=_json_list(task.options_json),
        include_debug=task.include_debug,
        status=task.status,
        progress_events=progress_events,
        answer_preview=answer_preview,
        answer=answer,
        error=error,
        created_at=_iso(task.created_at),
        updated_at=_iso(task.updated_at),
        completed_at=_iso(task.completed_at) if task.completed_at else None,
    )


def _json_events(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()
