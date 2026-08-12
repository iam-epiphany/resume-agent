# -*- coding: utf-8 -*-
"""访问码输错阶梯锁定守卫（IP 维度，进程内存）。

访问码印在简历上不便更换，因此用「高熵 6 位码 + 输错锁定」抵御暴力破解：
- 连续输错 3 次 → 锁 1 分钟；解锁后第一次输入再错 → 升一档：
  1 → 5 → 10 → 30 → 60 分钟封顶（60 封顶后持续锁 1 小时）
- 锁定期内任何输入（无论对错）一律拒绝，由 API 层返回 429 + Retry-After
- 锁定期内失败不再累计计数；解锁后首试：输对清零，输错计数 +1 升档
- 状态存内存，重启清零（攻击者无法触发服务重启，可接受；
  即使每档都等满，36^6 组合也远超现实暴力窗口）
"""

from __future__ import annotations

import time
from threading import Lock

# 连续错误次数 → 锁定时长档位：错 3 次 1 分钟、4 次 5 分钟、5 次 10 分钟、
# 6 次 30 分钟、≥7 次 60 分钟封顶
_LOCKOUT_MINUTES = (1, 5, 10, 30, 60)

_state: dict[str, tuple[int, float | None]] = {}  # ip -> (连续错误次数, 锁定到期时间戳)
_lock = Lock()


def _now() -> float:
    """当前单调时钟（独立函数便于测试拨快）。"""
    return time.monotonic()


def _lockout_minutes(failures: int) -> int:
    if failures < 3:
        return 0
    return _LOCKOUT_MINUTES[min(failures - 3, len(_LOCKOUT_MINUTES) - 1)]


def locked_seconds(ip: str) -> int:
    """该 IP 剩余锁定秒数（向上取整）；0 = 未锁定（含锁定已到期）。"""
    with _lock:
        entry = _state.get(ip)
        if entry is None or entry[1] is None:
            return 0
        remaining = entry[1] - _now()
        if remaining <= 0:
            return 0
        return int(remaining) + 1


def record_failure(ip: str) -> int:
    """记一次连续输错；返回本次锁定时长（秒），0 = 未触发锁定。"""
    with _lock:
        failures, _ = _state.get(ip, (0, None))
        failures += 1
        minutes = _lockout_minutes(failures)
        until = _now() + minutes * 60 if minutes else None
        _state[ip] = (failures, until)
        return minutes * 60


def record_success(ip: str) -> None:
    """输对清零：连续错误计数与锁定状态一并清除。"""
    with _lock:
        _state.pop(ip, None)


def _clear() -> None:
    """清空全部状态（测试用）。"""
    with _lock:
        _state.clear()
