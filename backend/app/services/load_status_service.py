"""系统负载分级：供公开状态接口 /api/qa/status 的 load 字段使用。

信号按 2C4G 部署校准（任务管线单 worker 串行，模型推理全局锁串行）：
- in_flight（运行中 + 排队中的问答任务数）：1 人提问=1 → 绿；2 人=2 → 黄；3 人及以上 → 红
- 进程 CPU 使用率（归一化到核数，最近 30s 均值平滑）：rerank 突刺不误报，持续占满 → 黄/红
- 进程内存（仅作红色兜底）：接近容器 cgroup 上限时报警，防 OOM；不参与黄色判定
"""

from __future__ import annotations

from typing import Any

from backend.app.core import config
from backend.app.core.performance_profile import effective_cpu_cores, memory_limit_bytes
from backend.app.services.performance_metrics import resource_metrics_snapshot
from backend.app.services.qa_task_service import qa_task_status_counts

LOAD_LEVELS = ("green", "yellow", "red")

# CPU 平滑窗口：最近 N 条采样（采样周期 1s）的平均值
_CPU_SMOOTH_SAMPLES = 30


def classify_load(
    *,
    cpu_ratio: float,
    mem_ratio: float,
    in_flight: int,
    yellow_cpu_ratio: float | None = None,
    red_cpu_ratio: float | None = None,
    red_mem_ratio: float | None = None,
    yellow_in_flight: int | None = None,
    red_in_flight: int | None = None,
) -> str:
    """纯函数分级：任一信号达红 → red，否则任一达黄 → yellow，否则 green。

    阈值参数可显式传入以便单测；默认取 config 值。
    """
    yellow_cpu_ratio = (
        config.LOAD_YELLOW_CPU_RATIO if yellow_cpu_ratio is None else yellow_cpu_ratio
    )
    red_cpu_ratio = config.LOAD_RED_CPU_RATIO if red_cpu_ratio is None else red_cpu_ratio
    red_mem_ratio = config.LOAD_RED_MEM_RATIO if red_mem_ratio is None else red_mem_ratio
    yellow_in_flight = (
        config.LOAD_YELLOW_INFLIGHT if yellow_in_flight is None else yellow_in_flight
    )
    red_in_flight = config.LOAD_RED_INFLIGHT if red_in_flight is None else red_in_flight

    if (
        cpu_ratio >= red_cpu_ratio
        or mem_ratio >= red_mem_ratio
        or in_flight >= red_in_flight
    ):
        return "red"
    if cpu_ratio >= yellow_cpu_ratio or in_flight >= yellow_in_flight:
        return "yellow"
    return "green"


def load_status_snapshot() -> dict[str, Any]:
    """采集当前负载信号并分级；所有信号均为内存/DB 快照，保证轮询轻量。"""
    counts = qa_task_status_counts()
    running = int(counts.get("running", 0) or 0)
    queued = max(
        int(counts.get("queued", 0) or 0),
        int(counts.get("queue_depth", 0) or 0),
    )
    in_flight = running + queued

    resources = resource_metrics_snapshot()
    history = resources.get("history") or []
    recent = [sample for sample in history[-_CPU_SMOOTH_SAMPLES:] if isinstance(sample, dict)]
    cpu_values = [float(sample.get("process_cpu_percent") or 0.0) for sample in recent]
    cpu_percent = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
    cores = effective_cpu_cores()
    cpu_ratio = round(min(cpu_percent / max(cores * 100.0, 1e-9), 1.0), 4)

    current = resources.get("current") or {}
    rss_bytes = current.get("rss_bytes")
    if rss_bytes:
        mem_ratio = round(min(float(rss_bytes) / max(_memory_reference_bytes(), 1), 1.0), 4)
    else:
        mem_ratio = 0.0

    level = classify_load(cpu_ratio=cpu_ratio, mem_ratio=mem_ratio, in_flight=in_flight)
    return {
        "level": level,
        "signals": {
            "cpu_ratio": cpu_ratio,
            "mem_ratio": mem_ratio,
            "running": running,
            "queued": queued,
            "in_flight": in_flight,
            "cores": cores,
        },
    }


def _memory_reference_bytes() -> int:
    """内存基准：容器 cgroup 上限 → 服务器物理内存 → 配置兜底值。"""
    limit = memory_limit_bytes()
    if limit:
        return limit
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    if kib > 0:
                        return kib * 1024
    except (OSError, ValueError, IndexError):
        pass
    return config.LOAD_MEMORY_REFERENCE_BYTES
