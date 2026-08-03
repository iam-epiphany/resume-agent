# -*- coding: utf-8 -*-
"""多轮追问记忆：轮次持久化与最近上下文读取。

设计动机（技术亮点 3，参考 Dify 会话记忆思路）：
面试本质是多轮深度追问，无状态 RAG 对"那怎么解决超卖的？"这类依赖前文的
追问必然检索失败。指代消解职责已并入意图路由（intent_router_service 的
classify_and_resolve 一次调用完成分类 + 补全），本模块只负责记忆的存取：
recent_turns 提供上一轮上下文，record_turn 持久化每轮对话。
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from backend.app.core.config import (
    CONVERSATION_MEMORY_ENABLED,
    CONVERSATION_MEMORY_MAX_TURNS,
    CONVERSATION_MEMORY_TTL_HOURS,
)
from backend.app.models.document import ConversationTurn

# 进程内 LRU：session_id -> deque of turn dicts（上限会话数）
_MEMORY_LRU: OrderedDict[str, list[dict]] = OrderedDict()
_MEMORY_MAX_SESSIONS = 200
_MEMORY_MAX_TURNS_PER_SESSION = 8


def _now() -> datetime:
    return datetime.now(timezone.utc)


def recent_turns(db, session_id: str, limit: int = 8) -> list[dict]:
    """读取会话最近轮次（内存优先，回源 SQLite）。"""
    if not session_id:
        return []
    cached = _MEMORY_LRU.get(session_id)
    if cached is not None:
        return list(cached)[-limit:]
    try:
        rows = db.scalars(
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.turn_seq.desc())
            .limit(limit)
        ).all()
    except Exception:
        return []
    turns = [
        {
            "question": row.question,
            "resolved_question": row.resolved_question,
            "intent": row.intent,
            "answer_mode": row.answer_mode,
            "answer_excerpt": row.answer_excerpt,
        }
        for row in reversed(rows)
    ]
    _cache_put(session_id, turns)
    return turns


def record_turn(
    db,
    session_id: str | None,
    *,
    question: str,
    resolved_question: str | None,
    intent: str | None,
    answer_mode: str | None,
    answer_excerpt: str | None = None,
) -> None:
    """记录一轮对话。内存 + SQLite 双写；SQLite 失败仅告警不阻塞。"""
    if not session_id or not CONVERSATION_MEMORY_ENABLED:
        return
    turn = {
        "question": question,
        "resolved_question": resolved_question or question,
        "intent": intent,
        "answer_mode": answer_mode,
        "answer_excerpt": (answer_excerpt or "")[:200],
    }
    _cache_put(session_id, _cache_get(session_id) + [turn])
    try:
        turns = recent_turns(db, session_id, limit=1)
        next_seq = 1
        if turns:
            last = db.scalars(
                select(ConversationTurn)
                .where(ConversationTurn.session_id == session_id)
                .order_by(ConversationTurn.turn_seq.desc())
                .limit(1)
            ).first()
            if last is not None:
                next_seq = last.turn_seq + 1
        row = ConversationTurn(
            session_id=session_id,
            turn_seq=next_seq,
            question=question,
            resolved_question=resolved_question or question,
            intent=intent,
            answer_mode=answer_mode,
            answer_excerpt=(answer_excerpt or "")[:200],
        )
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()


def _cache_get(session_id: str) -> list[dict]:
    _MEMORY_LRU.setdefault(session_id, [])
    _MEMORY_LRU.move_to_end(session_id)
    return _MEMORY_LRU[session_id]


def _cache_put(session_id: str, turns: list[dict]) -> None:
    _MEMORY_LRU[session_id] = turns[-_MEMORY_MAX_TURNS_PER_SESSION:]
    _MEMORY_LRU.move_to_end(session_id)
    while len(_MEMORY_LRU) > _MEMORY_MAX_SESSIONS:
        _MEMORY_LRU.popitem(last=False)


def purge_expired(db) -> int:
    """删除 TTL 之外的会话记录（应用启动时调用）。返回删除行数。"""
    cutoff = _now() - timedelta(hours=CONVERSATION_MEMORY_TTL_HOURS)
    try:
        result = db.execute(
            delete(ConversationTurn).where(ConversationTurn.created_at < cutoff)
        )
        db.commit()
        return result.rowcount or 0
    except Exception:
        db.rollback()
        return 0
