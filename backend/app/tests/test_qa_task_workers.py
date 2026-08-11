"""问答任务 worker 池启动逻辑测试（QA_TASK_WORKERS 多线程启动与幂等性）。"""

from backend.app.services import qa_task_service


class DummyThread:
    """不真正启动线程：start 为 no-op，避免测试环境里跑真实 worker 循环。"""

    def __init__(self, *args, **kwargs) -> None:
        self.name = kwargs.get("name")

    def start(self) -> None:
        return None


def _install_dummy_thread(monkeypatch, started: list[str]) -> None:
    def factory(*args, **kwargs) -> DummyThread:
        started.append(kwargs.get("name"))
        return DummyThread(*args, **kwargs)

    monkeypatch.setattr(qa_task_service, "Thread", factory)
    monkeypatch.setattr(qa_task_service, "_recover_interrupted_tasks", lambda: None)
    monkeypatch.setattr(qa_task_service, "_fill_queue_from_db", lambda: None)


def test_start_qa_task_worker_starts_configured_worker_count(monkeypatch) -> None:
    started: list[str] = []
    _install_dummy_thread(monkeypatch, started)
    monkeypatch.setattr(qa_task_service, "_STARTED", False)

    qa_task_service.start_qa_task_worker()

    # 默认 QA_TASK_WORKERS=2 → 两个带序号的 worker 线程
    assert started == ["resumemind-qa-worker-1", "resumemind-qa-worker-2"]


def test_start_qa_task_worker_is_idempotent(monkeypatch) -> None:
    started: list[str] = []
    _install_dummy_thread(monkeypatch, started)
    monkeypatch.setattr(qa_task_service, "_STARTED", True)

    qa_task_service.start_qa_task_worker()

    # 已启动过则不再重复创建线程（幂等）
    assert started == []
