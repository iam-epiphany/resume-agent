from contextlib import asynccontextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request as UrlRequest
from urllib.request import urlopen
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.api.audit import public_router as audit_public_router
from backend.app.api.audit import router as audit_router
from backend.app.api.auth import router as auth_router
from backend.app.api.documents import router as documents_router
from backend.app.api.health import admin_router as health_admin_router
from backend.app.api.health import router as health_router
from backend.app.api.qa import router as qa_router
from backend.app.middleware.rate_limit import RateLimitMiddleware
from backend.app.core.config import API_TITLE, CORS_ORIGINS, FRONTEND_DEV_SERVER
from backend.app.core.database import init_db
from backend.app.services.index_task_service import start_index_task_worker
from backend.app.services.qa_task_service import start_qa_task_worker
from backend.app.services.document_lifecycle_service import recover_interrupted_deletions
from backend.app.services.audit_service import archive_expired_audit_logs
from backend.app.services.model_warmup_service import start_background_model_warmup
from backend.app.services.performance_metrics import start_resource_sampling, stop_resource_sampling
from backend.app.core.database import SessionLocal


init_db()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    start_resource_sampling()
    start_index_task_worker()
    start_qa_task_worker()
    recover_interrupted_deletions()
    with SessionLocal() as db:
        archive_expired_audit_logs(db)
    start_background_model_warmup()
    try:
        yield
    finally:
        stop_resource_sampling()


# FastAPI 应用对象，应用启动入口。
app = FastAPI(title=API_TITLE, lifespan=_lifespan)


async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message = str(exc.detail)
    code_by_status = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        429: "too_many_requests",
        503: "service_unavailable",
    }
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "detail": message,
            "error": {
                "code": code_by_status.get(exc.status_code, "request_failed"),
                "message": message,
                "stage": None,
                "retryable": exc.status_code in {429, 503},
                "request_id": getattr(request.state, "request_id", None),
            },
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    message = "请求参数格式不正确。"
    return JSONResponse(
        status_code=422,
        content={
            "detail": message,
            "error": {
                "code": "validation_error",
                "message": message,
                "stage": None,
                "retryable": False,
                "request_id": getattr(request.state, "request_id", None),
            },
            "details": exc.errors(),
        },
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

# 中间件注册顺序（Starlette 后注册者包裹在外层），目标栈序从外到内：
#   request-id(最外) → CORS → rate_limit(最内)
# 限流中间件直接 send 的 429 因此会经过 CORS（补 CORS 头，浏览器跨域才能读到 429）
# 与 request-id（补 X-Request-ID 头）。
app.add_middleware(RateLimitMiddleware)
# 开发阶段允许前端页面直接调用后端接口；后期上线时再收紧 allowed origins。
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(BaseHTTPMiddleware, dispatch=add_request_id)

# 所有后端接口统一放在 /api 下，根路径留给 React 前端页面。
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(health_admin_router, prefix="/api", tags=["health"])
app.include_router(auth_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(qa_router, prefix="/api")
app.include_router(audit_router, prefix="/api")
app.include_router(audit_public_router, prefix="/api")

if not FRONTEND_DEV_SERVER and FRONTEND_ASSETS.exists():
    # Vite 构建后的 JS/CSS 会放在 dist/assets，FastAPI 负责按原路径托管。
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="frontend-assets")


def _frontend_proxy_headers(headers) -> dict[str, str]:
    allowed = {"content-type", "cache-control", "etag", "last-modified"}
    return {key: value for key, value in headers.items() if key.lower() in allowed}


def _proxy_frontend_dev_server(request: Request, full_path: str) -> Response:
    path = "/" + quote(full_path, safe="/@:-._~")
    url = f"{FRONTEND_DEV_SERVER}{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    proxy_request = UrlRequest(
        url,
        headers={
            "Accept": request.headers.get("accept", "*/*"),
            "User-Agent": request.headers.get("user-agent", "ResumeMind local dev proxy"),
        },
        method=request.method,
    )
    try:
        with urlopen(proxy_request, timeout=10) as proxy_response:
            body = b"" if request.method == "HEAD" else proxy_response.read()
            return Response(
                content=body,
                status_code=proxy_response.status,
                headers=_frontend_proxy_headers(proxy_response.headers),
            )
    except HTTPError as exc:
        body = b"" if request.method == "HEAD" else exc.read()
        return Response(
            content=body,
            status_code=exc.code,
            headers=_frontend_proxy_headers(exc.headers),
        )
    except URLError:
        return PlainTextResponse(
            f"Frontend dev server is not reachable at {FRONTEND_DEV_SERVER}. Restart run.sh or the frontend dev server.",
            status_code=503,
        )


@app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend(request: Request, full_path: str):
    """返回 React 单页应用；未知 /api 路径仍按后端 404 处理。"""

    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")

    if FRONTEND_DEV_SERVER:
        return _proxy_frontend_dev_server(request, full_path)

    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)

    return PlainTextResponse(
        "Frontend build not found. Run `npm install` and `npm run build` in the frontend directory first.",
        status_code=503,
    )
