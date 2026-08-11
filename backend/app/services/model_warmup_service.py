from __future__ import annotations

from threading import Condition, Thread
from time import perf_counter_ns
from typing import Any

from backend.app.core import config
from backend.app.services.embedding_service import EmbeddingServiceError, embed_query
from backend.app.services.performance_metrics import observe_timing
from backend.app.services.rerank_service import RerankServiceError, rerank_candidates
from backend.app.services.vector_store_service import VectorSearchResult


_CONDITION = Condition()
_STATE = "not_started"
_ERROR: str | None = None
_ELAPSED_MS: float | None = None


def start_background_model_warmup() -> None:
    if config.MODEL_WARMUP_POLICY != "background":
        return
    with _CONDITION:
        if _STATE != "not_started":
            return
    Thread(target=_background_target, name="resumemind-model-warmup", daemon=True).start()


def warmup_models_at_startup() -> None:
    """Apply the configured startup policy without hiding a blocking-warmup failure."""

    if config.MODEL_WARMUP_POLICY == "blocking":
        warmup_models_once()
        return
    start_background_model_warmup()


def warmup_models_once() -> dict[str, Any]:
    global _STATE, _ERROR, _ELAPSED_MS
    with _CONDITION:
        if _STATE == "warmed":
            return warmup_status()
        if _STATE == "warming":
            _CONDITION.wait_for(lambda: _STATE != "warming")
            if _STATE == "warmed":
                return warmup_status()
            raise RuntimeError(_ERROR or "模型预热失败")
        _STATE = "warming"
        _ERROR = None

    started = perf_counter_ns()
    try:
        embed_query("简历材料问答预热")
        rerank_candidates(
            question="简历材料问答预热",
            candidates=[
                VectorSearchResult(
                    chunk_id="warmup",
                    document_id="warmup",
                    filename="warmup",
                    section_title=None,
                    page_number=None,
                    text="简历材料问答预热",
                    embedding_text="简历材料问答预热",
                    token_count=8,
                    score=1.0,
                )
            ],
            limit=1,
        )
    except (EmbeddingServiceError, RerankServiceError, RuntimeError) as exc:
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000
        with _CONDITION:
            _STATE = "failed"
            _ERROR = str(exc)
            _ELAPSED_MS = elapsed_ms
            _CONDITION.notify_all()
        observe_timing("model_warmup.failed", elapsed_ms)
        raise

    elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    with _CONDITION:
        _STATE = "warmed"
        _ERROR = None
        _ELAPSED_MS = elapsed_ms
        _CONDITION.notify_all()
    observe_timing("model_warmup.completed", elapsed_ms)
    return warmup_status()


def warmup_status() -> dict[str, Any]:
    with _CONDITION:
        return {
            "policy": config.MODEL_WARMUP_POLICY,
            "state": _STATE,
            "warmed": _STATE == "warmed",
            "warming": _STATE == "warming",
            "elapsed_ms": round(_ELAPSED_MS, 3) if _ELAPSED_MS is not None else None,
            "error": _ERROR,
        }


def _background_target() -> None:
    try:
        warmup_models_once()
    except Exception:
        return
