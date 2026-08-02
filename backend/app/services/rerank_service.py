from dataclasses import dataclass
from functools import lru_cache
import hashlib
from typing import Any

from backend.app.core import config
from backend.app.core.config import INDEX_VERSION, RERANK_TOP_K
from backend.app.core.performance_profile import resolve_performance_profile
from backend.app.services.model_path_resolver import ModelPathResolutionError, resolve_reranker_model_path
from backend.app.services.model_device_service import force_cpu_fallback, is_cuda_failure, selected_model_device
from backend.app.services.model_inference_lock import MODEL_INFERENCE_LOCK
from backend.app.services.performance_metrics import ByteLRUCache, measure
from backend.app.services.vector_store_service import VectorSearchResult


class RerankServiceError(RuntimeError):
    pass


_RERANKER_WARMED = False
_RERANK_INPUT_VERSION = "rerank-input-v2"


@dataclass
class RerankedChunk:
    candidate: VectorSearchResult
    rerank_score: float


@lru_cache(maxsize=1)
def _get_reranker() -> Any:
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RerankServiceError("缺少 FlagEmbedding 依赖，无法加载 BGE reranker") from exc

    try:
        reranker_model_path = resolve_reranker_model_path()
        with measure("rerank.model_load"):
            try:
                return FlagReranker(
                    reranker_model_path,
                    use_fp16=selected_model_device() == "cuda",
                    devices=selected_model_device(),
                )
            except TypeError:
                return FlagReranker(reranker_model_path, use_fp16=selected_model_device() == "cuda")
    except ModelPathResolutionError as exc:
        raise RerankServiceError(str(exc)) from exc
    except Exception as exc:
        raise RerankServiceError("BGE reranker 模型加载失败") from exc


def rerank_candidates(
    *,
    question: str,
    candidates: list[VectorSearchResult],
    limit: int = RERANK_TOP_K,
) -> list[RerankedChunk]:
    global _RERANKER_WARMED
    if not candidates:
        return []

    limited_candidates = candidates[: max(limit * 2, limit)]
    texts = [_rerank_text(candidate) for candidate in limited_candidates]
    cache = _rerank_score_cache()
    scores: list[float | None] = [None] * len(limited_candidates)
    missing_pairs: list[list[str]] = []
    missing_positions: list[int] = []
    missing_keys: list[str] = []
    for index, (candidate, text) in enumerate(zip(limited_candidates, texts, strict=True)):
        key = _rerank_cache_key(question, candidate.chunk_id, text)
        cached = cache.get(key)
        if cached is None:
            missing_pairs.append([question, text])
            missing_positions.append(index)
            missing_keys.append(key)
        else:
            scores[index] = cached

    if missing_pairs:
        computed = _compute_scores(missing_pairs)
        for position, key, value in zip(missing_positions, missing_keys, computed, strict=True):
            cache.put(key, value)
            scores[position] = value

    reranked = [
        RerankedChunk(candidate=candidate, rerank_score=float(score))
        for candidate, score in zip(limited_candidates, scores, strict=True)
        if score is not None
    ]
    reranked.sort(key=lambda item: item.rerank_score, reverse=True)
    _RERANKER_WARMED = True
    return reranked[:limit]


def reranker_runtime_status() -> dict[str, Any]:
    return {
        "loaded": _get_reranker.cache_info().currsize > 0,
        "warmed": _RERANKER_WARMED,
        "score_cache": _rerank_score_cache().snapshot(),
    }


def invalidate_rerank_score_cache() -> None:
    """Invalidate process-local scores after a knowledge-index mutation."""

    _rerank_score_cache().clear()


def _fallback_reranker_to_cpu(exc: BaseException) -> None:
    force_cpu_fallback(f"BGE reranker CUDA failure: {exc}")
    _get_reranker.cache_clear()
    _rerank_score_cache().clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _compute_scores(pairs: list[list[str]]) -> list[float]:
    profile = resolve_performance_profile(selected_model_device())
    try:
        with MODEL_INFERENCE_LOCK:
            reranker = _get_reranker()
            with measure("rerank.inference"):
                raw_scores = reranker.compute_score(
                    pairs,
                    normalize=True,
                    max_length=profile.rerank_max_length,
                    batch_size=profile.rerank_batch_size,
                )
    except TypeError:
        with MODEL_INFERENCE_LOCK:
            reranker = _get_reranker()
            with measure("rerank.inference"):
                raw_scores = reranker.compute_score(
                    pairs,
                    normalize=True,
                    max_length=profile.rerank_max_length,
                )
    except Exception as exc:
        if selected_model_device() == "cuda" and is_cuda_failure(exc):
            _fallback_reranker_to_cpu(exc)
            return _compute_scores(pairs)
        raise RerankServiceError("BGE reranker 评分失败") from exc
    if isinstance(raw_scores, (float, int)):
        return [float(raw_scores)]
    return [float(score) for score in raw_scores]


@lru_cache(maxsize=1)
def _rerank_score_cache() -> ByteLRUCache[float]:
    profile = resolve_performance_profile(selected_model_device())
    return ByteLRUCache(
        max_bytes=profile.rerank_score_cache_bytes,
        max_items=profile.rerank_score_cache_items,
        size_of=lambda _value: 96,
    )


def _rerank_cache_key(question: str, chunk_id: str, text: str) -> str:
    profile = resolve_performance_profile(selected_model_device())
    namespace = "|".join(
        [
            _RERANK_INPUT_VERSION,
            config.RERANKER_MODEL_PATH or config.RERANKER_MODEL_NAME,
            config.MODEL_BACKEND,
            selected_model_device(),
            "fp16" if selected_model_device() == "cuda" else "fp32",
            str(profile.rerank_max_length),
            profile.rerank_input_mode,
            INDEX_VERSION,
        ]
    )
    payload = f"{namespace}\0{question.strip()}\0{chunk_id}\0{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rerank_text(candidate: VectorSearchResult) -> str:
    if config.RERANK_INPUT_MODE != "compact":
        return candidate.embedding_text or candidate.text

    metadata = candidate.metadata or {}
    labels: list[str] = []
    _append_unique(labels, f"来源文件：{candidate.filename}" if candidate.filename else "")
    section_path = [str(item).strip() for item in candidate.section_path or [] if str(item).strip()]
    if section_path:
        _append_unique(labels, "章节路径：" + " > ".join(dict.fromkeys(section_path)))
    elif candidate.section_title:
        _append_unique(labels, f"章节：{candidate.section_title}")
    if candidate.section_number:
        _append_unique(labels, f"条款号：{candidate.section_number}")
    if candidate.parent_section_number:
        _append_unique(labels, f"父条款号：{candidate.parent_section_number}")
    if candidate.page_number is not None:
        _append_unique(labels, f"页码：{candidate.page_number}")
    if candidate.chunk_type == "table":
        for key, label in (
            ("sheet_name", "工作表"),
            ("table_title", "表格标题"),
            ("unit", "单位"),
        ):
            value = str(metadata.get(key) or "").strip()
            if value:
                _append_unique(labels, f"{label}：{value}")
        period = metadata.get("period")
        if isinstance(period, dict) and str(period.get("raw") or "").strip():
            _append_unique(labels, f"期间：{str(period['raw']).strip()}")
        headers = metadata.get("table_headers") or metadata.get("headers") or []
        header_text = "、".join(dict.fromkeys(str(item).strip() for item in headers if str(item).strip()))
        if header_text:
            _append_unique(labels, f"表头：{header_text}")
    body = candidate.text.strip()
    return "\n".join([*labels, "", body]).strip() if labels else body


def _append_unique(items: list[str], value: str) -> None:
    cleaned = value.strip()
    if cleaned and cleaned not in items:
        items.append(cleaned)
