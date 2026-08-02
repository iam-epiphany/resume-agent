from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from threading import Lock
from typing import Any

from backend.app.core.config import (
    MODEL_DEVICE,
    MODEL_GPU_MIN_FREE_MEMORY_GB,
    RESUME_PERFORMANCE_MODE,
)
from backend.app.core.performance_profile import configure_torch_runtime, resolve_performance_profile


logger = logging.getLogger(__name__)

_RUNTIME_CPU_FALLBACK_REASON: str | None = None
_RUNTIME_LOCK = Lock()
_DEVICE_LOGGED = False

# Configure native thread pools before PyTorch/FlagEmbedding is imported.
_BOOTSTRAP_PROFILE = resolve_performance_profile("cpu")
os.environ.setdefault("OMP_NUM_THREADS", str(_BOOTSTRAP_PROFILE.torch_num_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_BOOTSTRAP_PROFILE.torch_num_threads))


@dataclass(frozen=True)
class ModelDeviceInfo:
    requested_device: str
    selected_device: str
    torch_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_total_memory_gb: float | None = None
    cuda_free_memory_gb: float | None = None
    fallback_reason: str | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "selected_device": self.selected_device,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
            "cuda_total_memory_gb": self.cuda_total_memory_gb,
            "cuda_free_memory_gb": self.cuda_free_memory_gb,
            "fallback_reason": self.fallback_reason,
        }


def force_cpu_fallback(reason: str) -> None:
    """Switch subsequent model loads to CPU after a CUDA load/inference failure."""

    global _RUNTIME_CPU_FALLBACK_REASON
    with _RUNTIME_LOCK:
        if _RUNTIME_CPU_FALLBACK_REASON is None:
            _RUNTIME_CPU_FALLBACK_REASON = reason
            get_model_device_info.cache_clear()
            logger.warning("模型设备已降级为 CPU：%s。处理速度可能较慢。", reason)


def is_cuda_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "cuda",
            "cublas",
            "cudnn",
            "out of memory",
            "invalid device",
            "device-side assert",
        )
    )


@lru_cache(maxsize=1)
def get_model_device_info() -> ModelDeviceInfo:
    if RESUME_PERFORMANCE_MODE == "gpu":
        requested = "cuda"
    elif RESUME_PERFORMANCE_MODE in {"cpu_balanced", "cpu_low_resource"}:
        requested = "cpu"
    else:
        requested = MODEL_DEVICE if MODEL_DEVICE in {"auto", "cpu", "cuda"} else "auto"
    runtime_reason = _RUNTIME_CPU_FALLBACK_REASON
    try:
        import torch
    except Exception as exc:
        info = ModelDeviceInfo(
            requested_device=requested,
            selected_device="cpu",
            torch_version=None,
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_name=None,
            fallback_reason=f"torch unavailable: {exc}",
        )
        _log_device_once(info)
        return info

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count() or 0)
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    total_gb, free_gb = _cuda_memory_gb(torch, cuda_available)

    selected = "cpu"
    fallback_reason: str | None = runtime_reason
    if runtime_reason:
        selected = "cpu"
    elif requested == "cpu":
        selected = "cpu"
    elif requested == "cuda" and not cuda_available:
        selected = "cpu"
        fallback_reason = "MODEL_DEVICE=cuda requested but CUDA is unavailable"
    elif cuda_available and free_gb is not None and free_gb < MODEL_GPU_MIN_FREE_MEMORY_GB:
        selected = "cpu"
        fallback_reason = (
            f"CUDA free memory {free_gb:.2f}GB is below MODEL_GPU_MIN_FREE_MEMORY_GB="
            f"{MODEL_GPU_MIN_FREE_MEMORY_GB:.2f}GB"
        )
    elif cuda_available:
        selected = "cuda"
    else:
        selected = "cpu"
        fallback_reason = None

    configure_torch_runtime(torch, selected)

    info = ModelDeviceInfo(
        requested_device=requested,
        selected_device=selected,
        torch_version=str(getattr(torch, "__version__", "")) or None,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        cuda_total_memory_gb=total_gb,
        cuda_free_memory_gb=free_gb,
        fallback_reason=fallback_reason,
    )
    _log_device_once(info)
    return info


def selected_model_device() -> str:
    return get_model_device_info().selected_device


def reset_runtime_device_fallback_for_tests() -> None:
    global _RUNTIME_CPU_FALLBACK_REASON, _DEVICE_LOGGED
    with _RUNTIME_LOCK:
        _RUNTIME_CPU_FALLBACK_REASON = None
        _DEVICE_LOGGED = False
        get_model_device_info.cache_clear()


def _cuda_memory_gb(torch: Any, cuda_available: bool) -> tuple[float | None, float | None]:
    if not cuda_available:
        return None, None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return round(total_bytes / (1024**3), 2), round(free_bytes / (1024**3), 2)
    except Exception:
        try:
            props = torch.cuda.get_device_properties(0)
            return round(float(props.total_memory) / (1024**3), 2), None
        except Exception:
            return None, None


def _log_device_once(info: ModelDeviceInfo) -> None:
    global _DEVICE_LOGGED
    with _RUNTIME_LOCK:
        if _DEVICE_LOGGED:
            return
        _DEVICE_LOGGED = True
    message = (
        "模型设备选择：requested=%s selected=%s cuda_available=%s gpu=%s total_gb=%s free_gb=%s"
    )
    args = (
        info.requested_device,
        info.selected_device,
        info.cuda_available,
        info.cuda_device_name or "-",
        info.cuda_total_memory_gb,
        info.cuda_free_memory_gb,
    )
    if info.selected_device == "cpu" and info.fallback_reason:
        logger.warning(message + " fallback=%s。CPU 模式处理速度可能较慢。", *args, info.fallback_reason)
    else:
        logger.info(message, *args)
    profile = resolve_performance_profile(info.selected_device)
    logger.info(
        "性能配置：mode=%s backend=%s embedding_batch=%s rerank_batch=%s "
        "rerank_max_length=%s torch_threads=%s interop_threads=%s OMP=%s MKL=%s",
        profile.selected_mode,
        profile.backend,
        profile.embedding_batch_size,
        profile.rerank_batch_size,
        profile.rerank_max_length,
        profile.torch_num_threads,
        profile.torch_num_interop_threads,
        profile.omp_num_threads,
        profile.mkl_num_threads,
    )
