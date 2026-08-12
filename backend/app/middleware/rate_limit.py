"""IP 限流 + QA 全局并发上限（纯 ASGI 中间件）。

设计要点：
- 限流范围：/api/qa/*（豁免 /api/qa/status 与 /api/qa/tasks/{id}/stream——
  EventSource 自动重连会耗尽配额）+ /api/auth/login（独立更严配额）
- 每 IP：1 分钟滑动窗口（deque）内存计数，单实例部署无需 Redis。
  提问总量不再按天限流，改由 API 层按 qa_logs 累计计数（QA_IP_MAX_QUESTIONS）
- 全局并发：仅 /api/qa/ask、/api/qa/ask/stream（真正跑 LLM 的路径），
  send wrapper 在 more_body=False 时释放配额（保证 SSE 期间持续占配额）
- 429 直接构造 JSONResponse：中间件层抛 HTTPException 不经 ExceptionMiddleware，
  body 与后端统一错误格式保持一致
- 必须用纯 ASGI 而非 BaseHTTPMiddleware：call_next 不等 StreamingResponse
  的 body 结束，会导致并发配额提前释放
- 配置每请求动态读取（config.*），测试 monkeypatch 即时生效
"""

import time
from collections import deque
from threading import Lock
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.app.core import config

_MINUTE_WINDOW_SECONDS = 60


def get_client_ip(request: Request) -> str:
    """解析客户端 IP：可信反向代理（Cloudflare Tunnel）时取 X-Forwarded-For 首段。

    中间件与 API 层（提问配额/审计记录）共用，保证同一次请求 IP 口径一致。
    """
    if config.RATE_LIMIT_TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client is not None else "unknown"


class RateLimitMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app
        self._minute_hits: dict[str, deque[float]] = {}
        self._lock = Lock()
        self._active_requests = 0
        self._concurrency_lock = Lock()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        path = scope["path"]

        if not config.RATE_LIMIT_ENABLED:
            await self.app(scope, receive, send)
            return

        client_ip = get_client_ip(request)

        if path == "/api/auth/login":
            if not self._check_rate(
                client_ip,
                per_minute=config.LOGIN_RATE_LIMIT_PER_MINUTE,
            ):
                await self._send_too_many_requests(scope, receive, send, request, retry_after=60)
                return
            await self.app(scope, receive, send)
            return

        if path.startswith("/api/qa/"):
            # 轻量就绪状态与 SSE 任务流豁免限流
            if path == "/api/qa/status" or path.startswith("/api/qa/tasks/"):
                await self.app(scope, receive, send)
                return
            if not self._check_rate(
                client_ip,
                per_minute=config.QA_IP_RATE_LIMIT_PER_MINUTE,
            ):
                await self._send_too_many_requests(scope, receive, send, request, retry_after=60)
                return
            if path in {"/api/qa/ask", "/api/qa/ask/stream"}:
                if not self._acquire_concurrency():
                    await self._send_too_many_requests(scope, receive, send, request, retry_after=30)
                    return
                await self._call_with_release(scope, receive, send)
                return

        await self.app(scope, receive, send)

    # ------------------------------------------------------------------
    # 每 IP 限流
    # ------------------------------------------------------------------
    def _check_rate(self, ip: str, per_minute: int) -> bool:
        """每 IP 1 分钟滑动窗口限流（per_minute <= 0 = 不限制）。

        滑动窗口 pop 即自然清理旧命中；不存在的 IP 首次访问时建空窗口。
        """
        if per_minute <= 0:
            return True
        now = time.time()
        with self._lock:
            window = self._minute_hits.setdefault(ip, deque())
            while window and now - window[0] > _MINUTE_WINDOW_SECONDS:
                window.popleft()
            if len(window) >= per_minute:
                return False
            window.append(now)
            return True

    # ------------------------------------------------------------------
    # 全局并发
    # ------------------------------------------------------------------
    def _acquire_concurrency(self) -> bool:
        with self._concurrency_lock:
            if self._active_requests >= config.QA_GLOBAL_CONCURRENCY:
                return False
            self._active_requests += 1
            return True

    async def _call_with_release(self, scope, receive, send) -> None:
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                with self._concurrency_lock:
                    self._active_requests -= 1

        async def wrapped_send(message: dict) -> None:
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                release()
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            release()

    # ------------------------------------------------------------------
    # 429 响应
    # ------------------------------------------------------------------
    async def _send_too_many_requests(self, scope, receive, send, request: Request, retry_after: int) -> None:
        request_id = getattr(request.state, "request_id", None)
        message = "请求过于频繁，请稍后再试。"
        response = JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={
                "detail": message,
                "error": {
                    "code": "too_many_requests",
                    "message": message,
                    "stage": None,
                    "retryable": True,
                    "request_id": request_id,
                },
            },
        )
        await response(scope, receive, send)
