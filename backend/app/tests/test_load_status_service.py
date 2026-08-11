"""负载分级 classify_load 的边界测试 + load_status_snapshot 的信号采集测试。"""

import pytest

from backend.app.services import load_status_service
from backend.app.services.load_status_service import LOAD_LEVELS, classify_load


class TestClassifyLoad:
    """纯函数边界：全部绿色 / 单信号黄 / 单信号红 / 红优先于黄。"""

    def test_all_signals_low_is_green(self) -> None:
        assert classify_load(cpu_ratio=0.2, mem_ratio=0.4, in_flight=1) == "green"
        assert classify_load(cpu_ratio=0.0, mem_ratio=0.0, in_flight=0) == "green"

    def test_yellow_when_cpu_ratio_reaches_threshold(self) -> None:
        # 默认黄阈值 0.70：恰好等于 → 黄；略低于 → 绿
        assert classify_load(cpu_ratio=0.70, mem_ratio=0.3, in_flight=1) == "yellow"
        assert classify_load(cpu_ratio=0.69, mem_ratio=0.3, in_flight=1) == "green"

    def test_yellow_when_in_flight_reaches_threshold(self) -> None:
        # 默认双 worker（QA_TASK_WORKERS=2）：3 人提问（1 人排队）→ 黄；1-2 人 → 绿
        assert classify_load(cpu_ratio=0.1, mem_ratio=0.3, in_flight=3) == "yellow"
        assert classify_load(cpu_ratio=0.1, mem_ratio=0.3, in_flight=2) == "green"
        assert classify_load(cpu_ratio=0.1, mem_ratio=0.3, in_flight=1) == "green"

    def test_red_when_cpu_ratio_reaches_threshold(self) -> None:
        assert classify_load(cpu_ratio=0.90, mem_ratio=0.3, in_flight=1) == "red"
        # 0.70 ≤ cpu < 0.90 → 黄（未达红）
        assert classify_load(cpu_ratio=0.89, mem_ratio=0.3, in_flight=1) == "yellow"
        assert classify_load(cpu_ratio=0.69, mem_ratio=0.3, in_flight=1) == "green"

    def test_red_when_memory_ratio_reaches_threshold(self) -> None:
        # 内存只参与红色判定（防 OOM 兜底），不触发黄色
        assert classify_load(cpu_ratio=0.1, mem_ratio=0.90, in_flight=1) == "red"
        assert classify_load(cpu_ratio=0.1, mem_ratio=0.89, in_flight=1) == "green"

    def test_red_when_in_flight_reaches_threshold(self) -> None:
        # 默认双 worker（QA_TASK_WORKERS=2）：4 人及以上提问（in_flight=4）→ 红
        assert classify_load(cpu_ratio=0.1, mem_ratio=0.3, in_flight=4) == "red"
        # 3 人（1 人排队）→ 黄
        assert classify_load(cpu_ratio=0.1, mem_ratio=0.3, in_flight=3) == "yellow"
        # 2 人并行（都在跑）→ 绿
        assert classify_load(cpu_ratio=0.1, mem_ratio=0.3, in_flight=2) == "green"

    def test_red_priority_over_yellow(self) -> None:
        # 同时命中黄（in_flight=3）与红（cpu 0.95）→ 红
        assert classify_load(cpu_ratio=0.95, mem_ratio=0.3, in_flight=3) == "red"

    def test_explicit_thresholds_override_config(self) -> None:
        assert (
            classify_load(
                cpu_ratio=0.5,
                mem_ratio=0.5,
                in_flight=1,
                yellow_cpu_ratio=0.4,
                red_cpu_ratio=0.9,
                red_mem_ratio=0.9,
                yellow_in_flight=2,
                red_in_flight=3,
            )
            == "yellow"
        )

    def test_levels_are_ordered(self) -> None:
        assert LOAD_LEVELS == ("green", "yellow", "red")


class TestLoadStatusSnapshot:
    def test_snapshot_returns_level_and_signals(self, monkeypatch) -> None:
        monkeypatch.setattr(
            load_status_service,
            "qa_task_status_counts",
            lambda: {"running": 2, "queued": 1, "queue_depth": 1},
        )
        monkeypatch.setattr(
            load_status_service,
            "resource_metrics_snapshot",
            lambda: {
                "current": {"process_cpu_percent": 30.0, "rss_bytes": None},
                "history": [{"process_cpu_percent": 30.0, "rss_bytes": None}] * 10,
            },
        )
        monkeypatch.setattr(load_status_service, "effective_cpu_cores", lambda: 2)
        monkeypatch.setattr(load_status_service, "_memory_reference_bytes", lambda: 4096**3)

        snapshot = load_status_service.load_status_snapshot()
        assert snapshot["level"] == "yellow"  # in_flight=3（2 并行 + 1 排队）
        assert snapshot["signals"]["in_flight"] == 3
        assert snapshot["signals"]["running"] == 2
        assert snapshot["signals"]["queued"] == 1
        assert snapshot["signals"]["cpu_ratio"] == pytest.approx(0.15)  # 30% / (2核×100)
        assert snapshot["signals"]["cores"] == 2

    def test_snapshot_empty_history_does_not_crash(self, monkeypatch) -> None:
        monkeypatch.setattr(
            load_status_service, "qa_task_status_counts", lambda: {"running": 0}
        )
        monkeypatch.setattr(
            load_status_service,
            "resource_metrics_snapshot",
            lambda: {"current": {}, "history": []},
        )
        monkeypatch.setattr(load_status_service, "effective_cpu_cores", lambda: 2)
        monkeypatch.setattr(load_status_service, "_memory_reference_bytes", lambda: 4096**3)

        snapshot = load_status_service.load_status_snapshot()
        assert snapshot["level"] == "green"
        assert snapshot["signals"]["cpu_ratio"] == 0.0
        assert snapshot["signals"]["mem_ratio"] == 0.0
