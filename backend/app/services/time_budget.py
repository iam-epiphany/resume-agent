"""单问硬时间预算（2026-08-14）：monotonic deadline + 分段可负担检查。

重排是 2C4G 下的热路径（每次约 8-9s），LLM 生成约 2-3s。预算贯穿
answer_question 管线：重排前剩余预算不足即跳过重排（按融合分排序输出），
生成前不足即跳过 LLM 改用摘录兜底——保证单问总耗时被钳制在预算附近，
而不是靠每段超时各自兜底。
"""

from __future__ import annotations

from time import monotonic
from typing import Callable


class TimeBudget:
    """从创建时刻起算的硬时间预算；所有判断非抛异常（调用方显式分支）。"""

    def __init__(self, budget_seconds: float, now_fn: Callable[[], float] = monotonic) -> None:
        self._deadline = now_fn() + max(float(budget_seconds), 0.0)
        self._now_fn = now_fn

    def remaining(self) -> float:
        """剩余秒数（≤0 表示已超支）。"""
        return self._deadline - self._now_fn()

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def can_afford(self, cost_seconds: float) -> bool:
        """剩余预算是否足以承担 cost_seconds 的阶段性开销。"""
        return self.remaining() >= max(float(cost_seconds), 0.0)

    def snapshot(self) -> dict[str, float]:
        return {"remaining_seconds": round(self.remaining(), 3)}
