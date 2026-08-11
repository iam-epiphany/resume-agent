import json
from datetime import date, datetime, timezone
from queue import Queue
from threading import Thread
from time import monotonic, sleep
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core import config
from backend.app.core.database import SessionLocal, get_db
from backend.app.core.security import (
    QA_ACCESS_COOKIE,
    create_qa_access_token,
    has_qa_access,
    is_admin_request,
    qa_access_enabled,
    verify_qa_access_code,
)
from backend.app.models.document import QALog
from backend.app.schemas.qa import (
    LLMContextPackage,
    QARequest,
    QAResponse,
    QATaskRequest,
    QATaskCreateResponse,
    QATaskStatusResponse,
    QaPublicStatusResponse,
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

# 预算告警/熔断相关响应头（前端读取后弹窗提醒）
BUDGET_REMAINING_HEADER = "X-QA-Budget-Remaining"
BUDGET_WARNING_HEADER = "X-QA-Budget-Warning"


class QAAccessRequest(BaseModel):
    code: str = Field(min_length=1, max_length=64)


def _global_daily_qa_remaining(db: Session) -> tuple[int, int]:
    """今日全局已用提问数与剩余预算（跨 IP 的每日保险丝，防换 IP 刷爆额度）。

    以 qa_logs 当日记录数为准（含全部问答入口；redirected 零成本回答计入但占比极小）。
    """
    today = date.today()
    # created_at 存 UTC；用 SQLite 'localtime' 转本地日再比对，避免本地日期与 UTC 日期
    # 在 00:00-08:00 窗口错位导致"今日"预算漏计/多计（2026-08-12 修复）
    used = int(
        db.scalar(
            select(func.count())
            .select_from(QALog)
            .where(func.date(func.datetime(QALog.created_at, "localtime")) == today.isoformat())
        )
        or 0
    )
    limit = config.QA_GLOBAL_DAILY_LIMIT
    return max(limit - used, 0), limit


def _budget_warning(remaining: int, limit: int) -> bool:
    """预算告警：剩余 ≤ max(固定值, 预算×比例) 时提醒"今日预算即将超限"。"""
    threshold = max(config.QA_BUDGET_WARNING_REMAINING, int(limit * config.QA_BUDGET_WARNING_RATIO))
    return remaining <= threshold


def _apply_budget_headers(response: Response, db: Session) -> None:
    remaining, limit = _global_daily_qa_remaining(db)
    response.headers[BUDGET_REMAINING_HEADER] = str(remaining)
    if _budget_warning(remaining, limit):
        response.headers[BUDGET_WARNING_HEADER] = "1"


def _ensure_budget_available(db: Session) -> None:
    """全局每日预算熔断：预算用尽后拒绝新的问答（防恶意消耗，正常面试官远用不完）。"""
    remaining, _ = _global_daily_qa_remaining(db)
    if remaining <= 0:
        raise HTTPException(status_code=429, detail="今日问答预算已用完，请明天再试。")


def require_qa_access(request: Request) -> None:
    """访客问答访问码闸（fail-closed）：无有效访问凭证（cookie/Bearer/管理员）时 401。"""
    if not has_qa_access(request):
        raise HTTPException(status_code=401, detail="请输入访问码后使用问答功能")


@router.get("/status", response_model=QaPublicStatusResponse)
def qa_status() -> QaPublicStatusResponse:
    """公开轻量就绪状态：前台匿名轮询用，不暴露内部细节（限流中间件豁免此路径）。"""
    return QaPublicStatusResponse(**public_qa_status())


@router.post("/access")
def qa_access_login(payload: QAAccessRequest, response: Response) -> dict[str, Any]:
    """访客访问码登录：校验访问码后签发短期 JWT 存 httpOnly cookie（默认 24 小时）。"""
    if not qa_access_enabled():
        return {"granted": True, "access_enabled": False}
    if not verify_qa_access_code(payload.code):
        raise HTTPException(status_code=403, detail="访问码不正确")
    token, expires_at = create_qa_access_token()
    response.set_cookie(
        key=QA_ACCESS_COOKIE,
        value=token,
        max_age=int(config.QA_ACCESS_TOKEN_TTL_HOURS * 3600),
        httponly=True,
        samesite="lax",
        secure=config.COOKIE_SECURE,
        path="/",
    )
    return {"granted": True, "access_enabled": True, "expires_at": expires_at.isoformat()}


@router.get("/access/status")
def qa_access_status(request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    """访问闸与预算状态：前台用于决定显示访问码门还是问答界面，并轮询预算提醒。"""
    remaining, limit = _global_daily_qa_remaining(db)
    return {
        "access_enabled": qa_access_enabled(),
        "granted": has_qa_access(request),
        "daily_used": limit - remaining,
        "daily_remaining": remaining,
        "daily_limit": limit,
        "budget_warning": _budget_warning(remaining, limit),
    }


@router.post("/ask", response_model=QAResponse, dependencies=[Depends(require_qa_access)])
def ask_question(
    payload: QARequest,
    request: Request,
    http_response: Response,
    db: Session = Depends(get_db),
) -> QAResponse:
    """Return a deterministic table answer or a generated, grounded text answer."""

    _ensure_budget_available(db)
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
        result = _strip_context_package(response)
    else:
        result = response
    _apply_budget_headers(http_response, db)
    return result


@router.post("/tasks", response_model=QATaskCreateResponse, dependencies=[Depends(require_qa_access)])
def create_question_task(
    payload: QATaskRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> QATaskCreateResponse:
    """Create a durable QA task that can be polled after page switches or reloads."""

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    _ensure_budget_available(db)
    result = create_qa_task(payload)
    _apply_budget_headers(response, db)
    return result


@router.get("/tasks", response_model=list[QATaskStatusResponse], dependencies=[Depends(require_qa_access)])
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


@router.get("/tasks/{task_id}", response_model=QATaskStatusResponse, dependencies=[Depends(require_qa_access)])
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


@router.get("/tasks/{task_id}/stream", dependencies=[Depends(require_qa_access)])
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


@router.post("/tasks/{task_id}/cancel", response_model=QATaskStatusResponse, dependencies=[Depends(require_qa_access)])
def cancel_question_task(task_id: str, db: Session = Depends(get_db)) -> QATaskStatusResponse:
    """Cancel a queued or running QA task without losing its audit trail."""

    task = cancel_qa_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="问答任务不存在或已过期")
    return task


@router.post("/ask/stream", dependencies=[Depends(require_qa_access)])
def ask_question_stream(payload: QARequest, request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """可信问答流式观测：通过 SSE 推送 RAG 阶段进度，最终返回 QAResponse。"""

    _ensure_budget_available(db)
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
