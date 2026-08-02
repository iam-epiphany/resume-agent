import math
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from backend.app.core import config
from backend.app.core.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_BATCH_SIZE
from backend.app.core.performance_profile import resolve_performance_profile
from backend.app.services.model_path_resolver import ModelPathResolutionError, resolve_embedding_model_path
from backend.app.services.model_device_service import force_cpu_fallback, is_cuda_failure, selected_model_device
from backend.app.services.model_inference_lock import MODEL_INFERENCE_LOCK
from backend.app.services.performance_metrics import ByteLRUCache, measure


class EmbeddingServiceError(RuntimeError):
    pass


_EMBEDDING_WARMED = False
_QUERY_NORMALIZATION_VERSION = "query-v1-exact-strip"


@dataclass
class TextEmbedding:
    dense: list[float]


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    try:
        from FlagEmbedding import FlagModel
    except ImportError as exc:
        raise EmbeddingServiceError("缺少 FlagEmbedding 依赖，无法生成 BGE embedding") from exc

    try:
        embedding_model_path = resolve_embedding_model_path()
        with measure("embedding.model_load"):
            try:
                return FlagModel(
                    embedding_model_path,
                    pooling_method="cls",
                    normalize_embeddings=True,
                    use_fp16=selected_model_device() == "cuda",
                    devices=selected_model_device(),
                )
            except TypeError:
                return FlagModel(
                    embedding_model_path,
                    use_fp16=selected_model_device() == "cuda",
                )
    except ModelPathResolutionError as exc:
        raise EmbeddingServiceError(str(exc)) from exc
    except Exception as exc:
        raise EmbeddingServiceError("BGE embedding 模型加载失败") from exc


def _encode_texts(model: Any, texts: list[str], batch_size: int) -> Any:
    """FlagModel.encode 返回 (n, dim) 的 numpy 数组；部分版本 encode 不接收 batch_size。"""
    try:
        return model.encode(texts, batch_size=batch_size)
    except TypeError:
        return model.encode(texts)


def embed_texts(texts: list[str], *, batch_size: int = EMBEDDING_BATCH_SIZE) -> list[TextEmbedding]:
    global _EMBEDDING_WARMED
    cleaned = [text.strip() for text in texts]
    if any(not text for text in cleaned):
        raise EmbeddingServiceError("embedding 输入文本不能为空")
    safe_batch_size = min(max(int(batch_size or 1), 1), EMBEDDING_MAX_BATCH_SIZE)

    try:
        with MODEL_INFERENCE_LOCK:
            model = _get_embedding_model()
            with measure("embedding.inference"):
                encoded = _encode_texts(model, cleaned, safe_batch_size)
    except Exception as exc:
        if selected_model_device() == "cuda" and is_cuda_failure(exc):
            _fallback_embedding_to_cpu(exc)
            try:
                with MODEL_INFERENCE_LOCK:
                    model = _get_embedding_model()
                    with measure("embedding.inference"):
                        encoded = _encode_texts(model, cleaned, max(1, min(safe_batch_size, 2)))
            except Exception as retry_exc:
                raise EmbeddingServiceError("BGE embedding 生成失败") from retry_exc
        else:
            raise EmbeddingServiceError("BGE embedding 生成失败") from exc

    results: list[TextEmbedding] = []
    for vector in encoded:
        results.append(TextEmbedding(dense=_normalize_dense_vector(vector)))
    _EMBEDDING_WARMED = True
    return results


def embed_query(text: str) -> TextEmbedding:
    return embed_queries([text])[0]


def embed_queries(texts: list[str]) -> list[TextEmbedding]:
    """Embed reusable search queries while keeping document indexing uncached."""

    if not texts:
        return []
    cache = _query_embedding_cache()
    results: list[TextEmbedding | None] = [None] * len(texts)
    missing_by_key: dict[str, tuple[str, list[int]]] = {}
    for index, text in enumerate(texts):
        cleaned = text.strip()
        if not cleaned:
            raise EmbeddingServiceError("embedding 输入文本不能为空")
        key = _query_cache_key(cleaned)
        pending = missing_by_key.get(key)
        if pending is not None:
            pending[1].append(index)
            continue
        cached = cache.get(key)
        if cached is None:
            missing_by_key[key] = (cleaned, [index])
        else:
            results[index] = cached

    if missing_by_key:
        profile = resolve_performance_profile(selected_model_device())
        missing_keys = list(missing_by_key)
        missing_texts = [missing_by_key[key][0] for key in missing_keys]
        embedded = embed_texts(missing_texts, batch_size=profile.embedding_batch_size)
        for key, value in zip(missing_keys, embedded, strict=True):
            cache.put(key, value)
            for position in missing_by_key[key][1]:
                results[position] = value
    return [value for value in results if value is not None]


def embed_for_semantic_split(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [embedding.dense for embedding in embed_texts(texts)]


def embedding_runtime_status() -> dict[str, Any]:
    return {
        "loaded": _get_embedding_model.cache_info().currsize > 0,
        "warmed": _EMBEDDING_WARMED,
        "query_cache": _query_embedding_cache().snapshot(),
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize_dense_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _fallback_embedding_to_cpu(exc: BaseException) -> None:
    force_cpu_fallback(f"BGE embedding CUDA failure: {exc}")
    _get_embedding_model.cache_clear()
    _query_embedding_cache().clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@lru_cache(maxsize=1)
def _query_embedding_cache() -> ByteLRUCache[TextEmbedding]:
    profile = resolve_performance_profile(selected_model_device())
    return ByteLRUCache(
        max_bytes=profile.query_embedding_cache_bytes,
        max_items=profile.query_embedding_cache_items,
        size_of=lambda value: 64 + len(value.dense) * 32,
    )


def _query_cache_key(text: str) -> str:
    namespace = "|".join(
        [
            _QUERY_NORMALIZATION_VERSION,
            config.EMBEDDING_MODEL_PATH or config.EMBEDDING_MODEL_NAME,
            config.MODEL_BACKEND,
            selected_model_device(),
            "fp16" if selected_model_device() == "cuda" else "fp32",
        ]
    )
    return hashlib.sha256(f"{namespace}\0{text}".encode("utf-8")).hexdigest()
