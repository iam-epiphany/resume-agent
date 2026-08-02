import json
from queue import Queue
from threading import Thread
from time import monotonic, sleep
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.core.database import SessionLocal, get_db
from backend.app.core.security import is_admin_request
from backend.app.schemas.qa import (
    LLMContextPackage,
    QARequest,
    QAResponse,
    QATaskRequest,
    QATaskCreateResponse,
    QATaskStatusResponse,
)
from backend.app.services.qa_task_service import (
    cancel_qa_task,
    create_qa_task,
    get_qa_task_status,
    list_recent_qa_task_statuses,
)
from backend.app.services.rag_service import answer_question, retrieve_context_package
from backend.app.services.readiness_service import public_qa_status
from backend.app.services.retrieval_service import RetrievalServiceUnavailable


router = APIRouter(prefix="/qa", tags=["qa"])


@router.get("/status")
def qa_status() -> dict[str, str | bool]:
    """公开轻量就绪状态：前台匿名轮询用，不暴露内部细节（限流中间件豁免此路径）。"""
    return public_qa_status()


@router.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest, request: Request, db: Session = Depends(get_db)) -> QAResponse:
    """Return a deterministic table answer or a generated, grounded text answer."""

    try:
        response = answer_question(
            db,
            payload.question,
            options=payload.options,
            include_debug=payload.include_debug,
            session_id=payload.session_id,
        )
    except RetrievalServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # 检索依据属后台信息：匿名一律剥离（include_debug 只对管理员生效，
    # 否则匿名可借 include_debug=true 拿到完整内部上下文）
    if not is_admin_request(request):
        return _strip_context_package(response)
    return response


@router.post("/tasks", response_model=QATaskCreateResponse)
def create_question_task(payload: QATaskRequest) -> QATaskCreateResponse:
    """Create a durable QA task that can be polled after page switches or reloads."""

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    return create_qa_task(payload)


@router.get("/tasks", response_model=list[QATaskStatusResponse])
def list_question_tasks(
    limit: int = 5,
    request: Request = None,  # type: ignore[assignment]  # FastAPI 注入
    db: Session = Depends(get_db),
) -> list[QATaskStatusResponse]:
    # 问答历史（问题与答案全文）仅管理员可见：匿名访客之间互不暴露提问内容。
    # 匿名用户仍可经 /tasks/{id} 与 /tasks/{id}/stream 访问自己刚提交的任务（id 不可枚举）。
    if not is_admin_request(request):
        return []
    return list_recent_qa_task_statuses(db, limit=limit)


@router.get("/tasks/{task_id}", response_model=QATaskStatusResponse)
def get_question_task(
    task_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> QATaskStatusResponse:
    task = get_qa_task_status(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="问答任务不存在或已过期")
    if is_admin_request(request):
        return task
    return _strip_context_package(task)


@router.get("/tasks/{task_id}/stream")
def stream_question_task(task_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    """Stream durable task snapshots, including verified answer previews."""

    if get_qa_task_status(db, task_id) is None:
        raise HTTPException(status_code=404, detail="问答任务不存在或已过期")
    return StreamingResponse(
        _stream_qa_task_snapshots(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tasks/{task_id}/cancel", response_model=QATaskStatusResponse)
def cancel_question_task(task_id: str, db: Session = Depends(get_db)) -> QATaskStatusResponse:
    """Cancel a queued or running QA task without losing its audit trail."""

    task = cancel_qa_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="问答任务不存在或已过期")
    return task


@router.post("/ask/stream")
def ask_question_stream(payload: QARequest, request: Request) -> StreamingResponse:
    """可信问答流式观测：通过 SSE 推送 RAG 阶段进度，最终返回 QAResponse。"""

    return StreamingResponse(
        _stream_qa_events(
            payload.question,
            payload.options,
            payload.include_debug,
            payload.session_id,
            is_admin=is_admin_request(request),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/retrieve", response_model=LLMContextPackage)
def retrieve_context(
    payload: QARequest,
    request: Request,
    db: Session = Depends(get_db),
) -> LLMContextPackage:
    """检索知识库并组装后续 LLM 可直接使用的上下文包。

    内部调试接口：返回完整内部提示词与检索片段全文，仅管理员可用；
    匿名调用一律 403（避免泄露系统提示词与知识库原文）。
    """

    if not is_admin_request(request):
        raise HTTPException(status_code=403, detail="检索调试接口仅管理员可用")
    try:
        return retrieve_context_package(db, payload.question, options=payload.options)
    except RetrievalServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _stream_qa_events(
    question: str,
    options: list[str] | None = None,
    include_debug: bool = False,
    session_id: str | None = None,
    is_admin: bool = False,
) -> Iterator[str]:
    events: Queue[tuple[str, dict[str, Any]] | None] = Queue()

    def progress_reporter(event: dict[str, Any]) -> None:
        events.put(("progress", event))

    def run_qa() -> None:
        db = SessionLocal()
        try:
            response = answer_question(
                db,
                question,
                options=options,
                include_debug=include_debug,
                session_id=session_id,
                progress_reporter=progress_reporter,
            )
            # 匿名视角剥离检索依据与内部提示词（SSE 无法带 Authorization 头）
            if not is_admin:
                stripped = response.model_copy(deep=False)
                stripped.context_package = None
                events.put(("final", stripped.model_dump(mode="json")))
            else:
                events.put(("final", response.model_dump(mode="json")))
        except RetrievalServiceUnavailable as exc:
            events.put(
                (
                    "progress",
                    {
                        "stage": "retrieval",
                        "status": "failed",
                        "title": "检索系统暂不可用",
                        "detail": str(exc),
                    },
                )
            )
            events.put(("error", {"detail": str(exc)}))
        except Exception as exc:
            events.put(
                (
                    "progress",
                    {
                        "stage": "prompt_build",
                        "status": "failed",
                        "title": "上下文构造失败",
                        "detail": str(exc),
                    },
                )
            )
            events.put(("error", {"detail": "问答接口暂未返回数据。"}))
        finally:
            db.close()
            events.put(None)

    Thread(target=run_qa, daemon=True).start()

    while True:
        item = events.get()
        if item is None:
            break
        event_name, payload = item
        yield _sse_frame(event_name, payload)


def _sse_frame(event_name: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


def _strip_context_package(task: QATaskStatusResponse | QAResponse) -> QATaskStatusResponse | QAResponse:
    """匿名视角剥离检索依据（context_chunks 全文与内部提示词），仅保留公开字段。"""
    if isinstance(task, QAResponse):
        if task.context_package is None:
            return task
        stripped = task.model_copy(deep=False)
        stripped.context_package = None
        return stripped
    if task.answer is None or task.answer.context_package is None:
        return task
    stripped = task.model_copy(deep=False)
    stripped.answer = task.answer.model_copy(deep=False)
    stripped.answer.context_package = None
    return stripped


def _stream_qa_task_snapshots(task_id: str) -> Iterator[str]:
    last_payload = ""
    last_heartbeat = monotonic()
    terminal_statuses = {"completed", "refused", "failed", "cancelled"}
    while True:
        with SessionLocal() as db:
            task = get_qa_task_status(db, task_id)
        if task is None:
            yield _sse_frame("error", {"detail": "问答任务不存在或已过期"})
            return
        # SSE（EventSource）无法携带 Authorization 头，统一按匿名视角剥离检索依据；
        # 管理员查看依据通过带 token 的 GET /api/qa/tasks/{id} 获取完整版。
        task = _strip_context_package(task)  # type: ignore[assignment]

        serialized = task.model_dump_json()
        if serialized != last_payload:
            last_payload = serialized
            last_heartbeat = monotonic()
            yield _sse_frame("task", task.model_dump(mode="json"))
        if task.status in terminal_statuses:
            return
        if monotonic() - last_heartbeat >= 15:
            last_heartbeat = monotonic()
            yield ": keep-alive\n\n"
        sleep(0.25)
