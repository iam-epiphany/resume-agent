"""公开就绪状态：供前台（匿名）轮询，只暴露 ready + 一句中文 message + 负载分级，
不泄露内部细节。

与后台 /health/rag 的区别：不返回模型路径、设备、性能等内部信息，且不做 LibreOffice、
warmup 等重检查，保证轮询足够轻量。负载分级（load 字段）见 load_status_service.py，
信号全部来自内存/DB 快照，不引入额外开销。
"""

from sqlalchemy import func, select, text

from backend.app.core import config
from backend.app.models.document import Document
from backend.app.services.load_status_service import load_status_snapshot
from backend.app.services.model_path_resolver import (
    ModelPathResolutionError,
    resolve_embedding_model_local_path,
    resolve_reranker_model_local_path,
)


def public_qa_status() -> dict[str, str | bool | dict]:
    """返回 {ready, message, load}；message 为面向普通访客的中文一句说明。"""
    errors: list[str] = []

    if not _sqlite_ready():
        return {"ready": False, "message": "系统数据库暂不可用，请稍后再试。"}

    if not _models_ready():
        errors.append("本地模型未就绪")
    if not _qdrant_ready():
        errors.append("向量库暂不可用")
    indexed_docs = _indexed_document_count()
    if indexed_docs is None:
        errors.append("知识库状态不可用")
    elif indexed_docs == 0:
        errors.append("知识库暂无已索引文档")

    if errors:
        return {"ready": False, "message": "，".join(errors) + "，请稍后再试。"}
    result: dict[str, str | bool | dict] = {"ready": True, "message": "系统已就绪，可以开始提问。"}
    try:
        result["load"] = load_status_snapshot()
    except Exception:
        # 负载采集失败不阻塞状态接口：降级为绿（无压力信号），下次轮询再试
        result["load"] = {"level": "green", "signals": None}
    return result


def _sqlite_ready() -> bool:
    try:
        from sqlalchemy.orm import Session

        from backend.app.core.database import SessionLocal

        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


def _models_ready() -> bool:
    try:
        resolve_embedding_model_local_path()
        resolve_reranker_model_local_path()
    except ModelPathResolutionError:
        return False
    return True


def _qdrant_ready() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=config.QDRANT_URL)
        if not client.collection_exists(config.QDRANT_COLLECTION):
            return False
        info = client.get_collection(config.QDRANT_COLLECTION)
        raw_status = getattr(info, "status", None)
        collection_status = (
            getattr(raw_status, "value", str(raw_status)) if raw_status is not None else None
        )
        return collection_status in {"green", "yellow"} and int(getattr(info, "points_count", 0) or 0) > 0
    except Exception:
        return False


def _indexed_document_count() -> int | None:
    try:
        from sqlalchemy.orm import Session

        from backend.app.core.database import SessionLocal

        with SessionLocal() as db:
            return int(db.scalar(select(func.count()).select_from(Document).where(Document.status == "indexed")) or 0)
    except Exception:
        return None
