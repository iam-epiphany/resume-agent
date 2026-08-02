from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
from threading import RLock
from threading import Event, Thread
from time import perf_counter, perf_counter_ns, process_time
from functools import wraps
from typing import Any, Callable, Generic, Iterator, ParamSpec, TypeVar
from uuid import uuid4


T = TypeVar("T")
P = ParamSpec("P")


@dataclass
class _Metric:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0


_METRICS: dict[str, _Metric] = {}
_METRICS_LOCK = RLock()
_CURRENT_TRACE: ContextVar[dict[str, Any] | None] = ContextVar("resumemind_performance_trace", default=None)
_TRACE_HISTORY: list[dict[str, Any]] = []
_TRACE_HISTORY_LIMIT = 50
_RESOURCE_LOCK = RLock()
_RESOURCE_STOP = Event()
_RESOURCE_THREAD: Thread | None = None
_RESOURCE_STATUS: dict[str, Any] = {
    "sampling": False,
    "interval_seconds": 1.0,
    "samples": 0,
    "current": {},
    "peaks": {},
    "history": [],
}


def observe_timing(name: str, elapsed_ms: float) -> None:
    with _METRICS_LOCK:
        metric = _METRICS.setdefault(name, _Metric())
        metric.count += 1
        metric.total_ms += elapsed_ms
        metric.max_ms = max(metric.max_ms, elapsed_ms)
        metric.last_ms = elapsed_ms
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        stage = trace["stages"].setdefault(name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
        stage["count"] += 1
        stage["total_ms"] += elapsed_ms
        stage["max_ms"] = max(stage["max_ms"], elapsed_ms)


@contextmanager
def measure(name: str) -> Iterator[None]:
    started = perf_counter_ns()
    try:
        yield
    finally:
        observe_timing(name, (perf_counter_ns() - started) / 1_000_000)


def timed(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with measure(name):
                return function(*args, **kwargs)

        return wrapper

    return decorator


def trace_operation(kind: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Capture all nested ``measure`` calls for one QA or indexing operation."""

    def decorator(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            trace: dict[str, Any] = {
                "trace_id": uuid4().hex,
                "kind": kind,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "stages": {},
            }
            started = perf_counter_ns()
            token = _CURRENT_TRACE.set(trace)
            try:
                result = function(*args, **kwargs)
            except Exception:
                trace["status"] = "failed"
                raise
            else:
                trace["status"] = "completed"
                return result
            finally:
                trace["total_ms"] = round((perf_counter_ns() - started) / 1_000_000, 3)
                trace["stages"] = {
                    name: {
                        "count": value["count"],
                        "total_ms": round(value["total_ms"], 3),
                        "max_ms": round(value["max_ms"], 3),
                    }
                    for name, value in sorted(trace["stages"].items())
                }
                _CURRENT_TRACE.reset(token)
                with _METRICS_LOCK:
                    _TRACE_HISTORY.append(trace)
                    del _TRACE_HISTORY[:-_TRACE_HISTORY_LIMIT]

        return wrapper

    return decorator


def timing_metrics_snapshot() -> dict[str, dict[str, float | int]]:
    with _METRICS_LOCK:
        return {
            name: {
                "count": metric.count,
                "last_ms": round(metric.last_ms, 3),
                "avg_ms": round(metric.total_ms / metric.count, 3) if metric.count else 0.0,
                "max_ms": round(metric.max_ms, 3),
            }
            for name, metric in sorted(_METRICS.items())
        }


def trace_history_snapshot(limit: int = 10) -> list[dict[str, Any]]:
    with _METRICS_LOCK:
        return [dict(item) for item in _TRACE_HISTORY[-max(0, limit):]]


def start_resource_sampling(interval_seconds: float = 1.0) -> None:
    """Sample lightweight process and PyTorch memory counters in the background."""

    global _RESOURCE_THREAD
    with _RESOURCE_LOCK:
        if _RESOURCE_THREAD is not None and _RESOURCE_THREAD.is_alive():
            return
        _RESOURCE_STOP.clear()
        _RESOURCE_STATUS.update({"sampling": True, "interval_seconds": interval_seconds})
        _RESOURCE_THREAD = Thread(
            target=_resource_sampling_loop,
            args=(max(0.1, interval_seconds),),
            name="resumemind-resource-sampler",
            daemon=True,
        )
        _RESOURCE_THREAD.start()


def stop_resource_sampling() -> None:
    global _RESOURCE_THREAD
    _RESOURCE_STOP.set()
    thread = _RESOURCE_THREAD
    if thread is not None:
        thread.join(timeout=2)
    with _RESOURCE_LOCK:
        _RESOURCE_STATUS["sampling"] = False
        _RESOURCE_THREAD = None


def resource_metrics_snapshot() -> dict[str, Any]:
    with _RESOURCE_LOCK:
        return {
            **_RESOURCE_STATUS,
            "current": dict(_RESOURCE_STATUS.get("current") or {}),
            "peaks": dict(_RESOURCE_STATUS.get("peaks") or {}),
            "history": list(_RESOURCE_STATUS.get("history") or []),
        }


def _resource_sampling_loop(interval_seconds: float) -> None:
    previous_wall = perf_counter()
    previous_cpu = process_time()
    while not _RESOURCE_STOP.wait(interval_seconds):
        current_wall = perf_counter()
        current_cpu = process_time()
        wall_delta = max(current_wall - previous_wall, 1e-9)
        cpu_percent = max(0.0, (current_cpu - previous_cpu) / wall_delta * 100.0)
        previous_wall, previous_cpu = current_wall, current_cpu
        current = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process_cpu_percent": round(cpu_percent, 2),
            "rss_bytes": _rss_bytes(),
        }
        current.update(_cuda_memory_metrics())
        with _RESOURCE_LOCK:
            _RESOURCE_STATUS["samples"] = int(_RESOURCE_STATUS.get("samples") or 0) + 1
            _RESOURCE_STATUS["current"] = current
            history = _RESOURCE_STATUS.setdefault("history", [])
            history.append(current)
            # Keep up to two hours at the default one-second cadence so a full
            # 300-case evaluation retains its complete resource envelope.
            del history[:-7200]
            peaks = _RESOURCE_STATUS.setdefault("peaks", {})
            for key, value in current.items():
                if isinstance(value, (int, float)):
                    peaks[key] = max(float(peaks.get(key) or 0), float(value))


def _rss_bytes() -> int | None:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _cuda_memory_metrics() -> dict[str, int]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except (AttributeError, RuntimeError):
        return {}


class ByteLRUCache(Generic[T]):
    """A bounded, process-local LRU with observable hit/miss/eviction counters."""

    def __init__(
        self,
        *,
        max_bytes: int,
        max_items: int,
        size_of: Callable[[T], int],
    ) -> None:
        self.max_bytes = max(0, max_bytes)
        self.max_items = max(0, max_items)
        self._size_of = size_of
        self._items: OrderedDict[str, tuple[T, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._items.pop(key, None)
            if item is None:
                self._misses += 1
                return None
            self._items[key] = item
            self._hits += 1
            return item[0]

    def put(self, key: str, value: T) -> None:
        if self.max_bytes <= 0 or self.max_items <= 0:
            return
        size = max(1, int(self._size_of(value)))
        if size > self.max_bytes:
            return
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            self._items[key] = (value, size)
            self._bytes += size
            while self._items and (
                len(self._items) > self.max_items or self._bytes > self.max_bytes
            ):
                _, (_, evicted_size) = self._items.popitem(last=False)
                self._bytes -= evicted_size
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "enabled": self.max_bytes > 0 and self.max_items > 0,
                "items": len(self._items),
                "bytes": self._bytes,
                "max_items": self.max_items,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
