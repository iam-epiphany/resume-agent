from threading import RLock
from time import perf_counter_ns

from backend.app.services.performance_metrics import observe_timing


# BGE-M3 and the reranker can share one GPU. Serializing their forward passes
# prevents concurrent upload indexing and QA requests from exhausting VRAM.
class InstrumentedRLock:
    def __init__(self) -> None:
        self._lock = RLock()
        self._acquired_at = 0

    def __enter__(self) -> "InstrumentedRLock":
        started = perf_counter_ns()
        self._lock.acquire()
        observe_timing("model_lock.wait", (perf_counter_ns() - started) / 1_000_000)
        self._acquired_at = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        observe_timing("model_lock.hold", (perf_counter_ns() - self._acquired_at) / 1_000_000)
        self._lock.release()


MODEL_INFERENCE_LOCK = InstrumentedRLock()
