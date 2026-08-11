from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

from backend.app.core import config


@dataclass(frozen=True)
class PerformanceProfile:
    requested_mode: str
    selected_mode: str
    requested_backend: str
    backend: str
    backend_fallback_reason: str | None
    effective_cpu_cores: int
    memory_limit_bytes: int | None
    embedding_batch_size: int
    rerank_batch_size: int
    rerank_max_length: int
    rerank_input_mode: str
    torch_num_threads: int
    torch_num_interop_threads: int
    omp_num_threads: int
    mkl_num_threads: int
    query_embedding_cache_bytes: int
    query_embedding_cache_items: int
    rerank_score_cache_bytes: int
    rerank_score_cache_items: int
    warmup_policy: str
    experimental: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def effective_cpu_cores() -> int:
    """Return the CPU capacity visible to this process, including containers."""

    affinity_count = None
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    host_count = affinity_count or os.cpu_count() or 1
    quota_count = _cgroup_cpu_quota()
    return max(1, min(host_count, quota_count)) if quota_count else max(1, host_count)


def memory_limit_bytes() -> int | None:
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if raw and raw != "max":
                value = int(raw)
                if 0 < value < 1 << 60:
                    return value
        except (OSError, ValueError):
            continue
    return None


def resolve_performance_profile(selected_device: str) -> PerformanceProfile:
    requested = config.RESUME_PERFORMANCE_MODE
    if requested == "auto":
        selected = "gpu" if selected_device.startswith("cuda") else "cpu_balanced"
    elif requested == "gpu" and not selected_device.startswith("cuda"):
        selected = "cpu_balanced"
    else:
        selected = requested

    cores = effective_cpu_cores()
    low_resource = selected == "cpu_low_resource"
    default_rerank_batch = 24 if selected == "gpu" else (2 if low_resource else 4)
    default_threads = min(4 if low_resource else 16, cores)
    default_interop = 1 if selected != "gpu" else 0
    cache_scale = 0.25 if low_resource else 1.0

    return PerformanceProfile(
        requested_mode=requested,
        selected_mode=selected,
        requested_backend=config.MODEL_BACKEND,
        # ONNX uses the same BGE weights and is enabled only after parity checks.
        # OpenVINO remains intentionally gated and therefore falls back safely.
        backend="onnx" if config.MODEL_BACKEND == "onnx" else "pytorch",
        backend_fallback_reason=(
            None
            if config.MODEL_BACKEND in {"pytorch", "onnx"}
            else "openvino_backend_not_quality_qualified"
        ),
        effective_cpu_cores=cores,
        memory_limit_bytes=memory_limit_bytes(),
        embedding_batch_size=max(1, min(config.EMBEDDING_BATCH_SIZE, config.EMBEDDING_MAX_BATCH_SIZE)),
        rerank_batch_size=config.RERANK_BATCH_SIZE or default_rerank_batch,
        rerank_max_length=config.RERANK_MAX_LENGTH,
        rerank_input_mode=config.RERANK_INPUT_MODE,
        torch_num_threads=config.TORCH_NUM_THREADS or default_threads,
        torch_num_interop_threads=config.TORCH_NUM_INTEROP_THREADS or default_interop,
        omp_num_threads=_thread_env("OMP_NUM_THREADS", config.TORCH_NUM_THREADS or default_threads),
        mkl_num_threads=_thread_env("MKL_NUM_THREADS", config.TORCH_NUM_THREADS or default_threads),
        query_embedding_cache_bytes=int(config.QUERY_EMBEDDING_CACHE_BYTES * cache_scale),
        query_embedding_cache_items=(
            min(config.QUERY_EMBEDDING_CACHE_ITEMS, 512)
            if low_resource
            else config.QUERY_EMBEDDING_CACHE_ITEMS
        ),
        rerank_score_cache_bytes=int(config.RERANK_SCORE_CACHE_BYTES * cache_scale),
        rerank_score_cache_items=(
            min(config.RERANK_SCORE_CACHE_ITEMS, 10_000)
            if low_resource
            else config.RERANK_SCORE_CACHE_ITEMS
        ),
        warmup_policy=config.MODEL_WARMUP_POLICY,
        experimental=(
            low_resource
            or config.MODEL_BACKEND != "pytorch"
            or config.RERANK_INPUT_MODE != "embedding"
        ),
    )


def configure_torch_runtime(torch: Any, selected_device: str) -> PerformanceProfile:
    profile = resolve_performance_profile(selected_device)
    if selected_device == "cpu":
        try:
            torch.set_num_threads(profile.torch_num_threads)
        except (RuntimeError, AttributeError):
            pass
        if profile.torch_num_interop_threads > 0:
            try:
                torch.set_num_interop_threads(profile.torch_num_interop_threads)
            except (RuntimeError, AttributeError):
                # PyTorch permits setting inter-op threads only before parallel work starts.
                pass
    return profile


def _cgroup_cpu_quota() -> int | None:
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip().split()
        if len(raw) == 2 and raw[0] != "max":
            quota, period = int(raw[0]), int(raw[1])
            if quota > 0 and period > 0:
                return max(1, (quota + period - 1) // period)
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return max(1, (quota + period - 1) // period)
    except (OSError, ValueError):
        pass
    return None


def _thread_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return max(1, default)
