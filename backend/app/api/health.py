from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.app.core import config
from backend.app.core.security import require_admin
from backend.app.core.config import APP_NAME
from backend.app.schemas.health import HealthResponse, RagHealthResponse
from backend.app.services.model_path_resolver import (
    ModelPathResolutionError,
    resolve_embedding_model_local_path,
    resolve_reranker_model_local_path,
    resolve_reranker_onnx_model_local_path,
)
from backend.app.services.office_conversion import office_tool_status
from backend.app.services.index_task_service import index_task_status_counts
from backend.app.services.model_device_service import get_model_device_info
from backend.app.services.qa_task_service import qa_task_status_counts
from backend.app.services.embedding_service import (
    embedding_runtime_status,
)
from backend.app.services.rerank_service import (
    reranker_runtime_status,
)
from backend.app.core.performance_profile import resolve_performance_profile
from backend.app.services.model_warmup_service import warmup_models_once, warmup_status
from backend.app.services.performance_metrics import (
    resource_metrics_snapshot,
    timing_metrics_snapshot,
    trace_history_snapshot,
)
from backend.app.services.vector_store_service import ensure_vector_collection


router = APIRouter()

# 富信息健康检查（模型路径/设备/性能/就绪状态）属内部信息，仅管理员可见；
# 公开的只有 GET /health（存活探测，供 docker healthcheck 与匿名前台使用）。
admin_router = APIRouter(
    dependencies=[Depends(require_admin)],
)


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """健康检查接口，前端工作台会用它判断后端是否可连接。"""

    return HealthResponse(
        status="ok",
        message=f"{APP_NAME} backend is healthy",
        build_id=config.BUILD_ID,
    )


@admin_router.get("/health/rag", response_model=RagHealthResponse)
def rag_health_check() -> RagHealthResponse:
    return _rag_health()


@admin_router.get("/health/ready", response_model=RagHealthResponse)
def readiness_check(response: Response) -> RagHealthResponse:
    health = _rag_health()
    if not health.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health


@admin_router.post("/health/warmup")
def warmup_models() -> dict[str, object]:
    """Load and exercise local inference models without writing business data."""

    try:
        status_payload = warmup_models_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "warmed": True,
        "warmup": status_payload,
        "model_runtime": _model_runtime_status(),
    }


def _rag_health() -> RagHealthResponse:
    embedding_ready, embedding_path, embedding_error = _resolve_model_for_health(
        resolver=resolve_embedding_model_local_path,
        configured_path=config.EMBEDDING_MODEL_PATH,
        default_path=config.DEFAULT_EMBEDDING_MODEL_DIR,
    )
    reranker_ready, reranker_path, reranker_error = _resolve_model_for_health(
        resolver=(
            resolve_reranker_onnx_model_local_path
            if config.MODEL_BACKEND == "onnx"
            else resolve_reranker_model_local_path
        ),
        configured_path=(
            config.RERANKER_ONNX_MODEL_PATH
            if config.MODEL_BACKEND == "onnx"
            else config.RERANKER_MODEL_PATH
        ),
        default_path=(
            config.DEFAULT_RERANKER_ONNX_INT8_DIR
            if config.MODEL_BACKEND == "onnx"
            else config.DEFAULT_RERANKER_MODEL_DIR
        ),
    )
    qdrant_ready, collection_ready, qdrant_error = _check_qdrant()
    sqlite_ready = _check_sqlite()
    office = office_tool_status()
    warmup = warmup_status()
    device_info = get_model_device_info()
    profile = resolve_performance_profile(device_info.selected_device)
    return RagHealthResponse(
        build_id=config.BUILD_ID,
        offline_mode=config.RESUME_OFFLINE_MODE,
        embedding_model_ready=embedding_ready,
        reranker_model_ready=reranker_ready,
        embedding_model_path=embedding_path,
        reranker_model_path=reranker_path,
        qdrant_ready=qdrant_ready,
        qdrant_collection=config.QDRANT_COLLECTION,
        qdrant_collection_ready=collection_ready,
        sqlite_ready=sqlite_ready,
        libreoffice_ready=bool(office["libreoffice_ready"]),
        antiword_ready=bool(office["antiword_ready"]),
        libreoffice_version=office["libreoffice_version"],
        antiword_version=office["antiword_version"],
        index_tasks=index_task_status_counts(),
        qa_tasks=qa_task_status_counts(),
        model_runtime=_model_runtime_status(),
        # LibreOffice 已从镜像中移除（瘦身），.doc 走 antiword 降级路径，
        # 不再作为就绪硬依赖；libreoffice_* 字段仍保留作信息性上报。
        ready=(
            embedding_ready
            and reranker_ready
            and qdrant_ready
            and collection_ready
            and sqlite_ready
        ),
        embedding_model_error=embedding_error,
        reranker_model_error=reranker_error,
        qdrant_error=qdrant_error,
        model_device=device_info.to_debug_dict(),
        performance={
            **profile.to_dict(),
            "warmup": warmup,
            "timings": timing_metrics_snapshot(),
            "resources": resource_metrics_snapshot(),
            "recent_traces": trace_history_snapshot(),
        },
    )


def _resolve_model_for_health(*, resolver, configured_path, default_path) -> tuple[bool, str, str | None]:
    try:
        resolved_path = resolver()
    except ModelPathResolutionError as exc:
        return False, str(configured_path or default_path), str(exc)
    return True, resolved_path, None


def _check_qdrant() -> tuple[bool, bool, str | None]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=config.QDRANT_URL)
        client.get_collections()
        collection_ready = False
        if client.collection_exists(config.QDRANT_COLLECTION):
            info = client.get_collection(config.QDRANT_COLLECTION)
            raw_status = getattr(info, "status", None)
            collection_status = (
                getattr(raw_status, "value", str(raw_status)) if raw_status is not None else None
            )
            points_count = int(getattr(info, "points_count", 0) or 0)
            collection_ready = collection_status in {"green", "yellow"} and points_count > 0
            if collection_ready:
                try:
                    ensure_vector_collection()
                except Exception as exc:
                    return True, False, str(exc)
    except Exception as exc:
        return False, False, str(exc)
    return True, collection_ready, None


def _check_sqlite() -> bool:
    try:
        from sqlalchemy import text
        from backend.app.core.database import engine

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


def _model_runtime_status() -> dict[str, object]:
    return {
        "embedding": embedding_runtime_status(),
        "reranker": reranker_runtime_status(),
    }
