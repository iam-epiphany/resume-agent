# -*- coding: utf-8 -*-
"""多轮追问记忆：会话级指代消解与追问补全。

设计动机（技术亮点 3，参考 Dify 会话记忆思路）：
面试本质是多轮深度追问，无状态 RAG 对"那怎么解决超卖的？"这类依赖前文的
追问必然检索失败。本模块用规则预检指代词（零成本），命中才触发一次小 LLM
调用做指代消解，把追问补全为带上下文的完整问题。
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from backend.app.core.config import (
    CONVERSATION_MEMORY_ENABLED,
    CONVERSATION_MEMORY_MAX_TURNS,
    CONVERSATION_MEMORY_TTL_HOURS,
)
from backend.app.models.document import ConversationTurn
from backend.app.services.llm_client import ChatCompletionConfig, ChatCompletionError, chat_completion_content

# 进程内 LRU：session_id -> deque of turn dicts（上限会话数）
_MEMORY_LRU: OrderedDict[str, list[dict]] = OrderedDict()
_MEMORY_MAX_SESSIONS = 200
_MEMORY_MAX_TURNS_PER_SESSION = 8

# 指代词/弱引用检测（命中才考虑 LLM 消解）
_INTERPOLATION_MARKERS = (
    "它", "他", "她", "这个", "那个", "这", "那", "刚才", "上面", "上一步",
    "还有", "然后", "具体", "类似", "怎么实现", "怎么解决", "怎么处理",
    "为什么", "呢", "那怎么", "你说过", "你提到", "你刚才",
)

_DISAMBIGUATION_PROMPT = """你是简历问答系统的追问理解模块。用户（面试官）在连续追问中可能省略了上下文，
比如"那怎么解决超卖的？"实际指的是上一轮提到的项目。
根据最近一轮对话，判断当前问题是否需要补全，并输出补全后的问题。
只输出 JSON：{{"rewritten_question": "<补全后的问题，若无需补全则原样返回>", "needs_context": true/false, "relevant": true/false}}
relevant=false 表示当前问题是新话题、与上一轮无关（此时应放弃消解，用原问题）。"""


def _llm_config() -> ChatCompletionConfig:
    from backend.app.core.config import (
        INTENT_ROUTER_API_KEY,
        INTENT_ROUTER_BASE_URL,
        INTENT_ROUTER_MODEL,
        INTENT_ROUTER_PROVIDER,
        INTENT_ROUTER_RESPONSE_FORMAT,
        INTENT_ROUTER_TIMEOUT_SECONDS,
    )

    return ChatCompletionConfig(
        provider=INTENT_ROUTER_PROVIDER,
        api_key=INTENT_ROUTER_API_KEY,
        base_url=INTENT_ROUTER_BASE_URL,
        model=INTENT_ROUTER_MODEL,
        timeout_seconds=INTENT_ROUTER_TIMEOUT_SECONDS,
        response_format=INTENT_ROUTER_RESPONSE_FORMAT,
    )


def _needs_disambiguation(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question)
    return any(marker in normalized for marker in _INTERPOLATION_MARKERS)


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object")
    return content[start : end + 1]


def _llm_resolve(question: str, previous_turn: dict | None) -> str | None:
    """LLM 指代消解；失败返回 None（调用方用原文）。"""
    if previous_turn is None:
        return None
    context_block = (
        f"最近一轮：\nQ: {previous_turn.get('question') or ''}\n"
        f"A: {(previous_turn.get('answer_excerpt') or '')[:120]}"
    )
    messages = [
        {"role": "system", "content": _DISAMBIGUATION_PROMPT},
        {"role": "user", "content": f"{context_block}\n\n当前问题：{question}"},
    ]
    try:
        content = chat_completion_content(
            _llm_config(), messages, temperature=0, max_tokens=160
        )
        payload = json.loads(_extract_json_object(content))
    except (ChatCompletionError, json.JSONDecodeError, ValueError):
        return None
    if payload.get("relevant") is False:
        return None
    rewritten = str(payload.get("rewritten_question") or "").strip()
    return rewritten or None


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


def resolve_question(db, session_id: str | None, question: str):
    """指代消解入口。返回 (resolved_question, used_memory, memory_turns)。

    - 无 session / 无指代词 / 记忆关闭 → 原样返回
    - 命中指代词且上一轮存在 → LLM 消解；失败或 relevant=false → 原文
    """
    if not session_id or not CONVERSATION_MEMORY_ENABLED:
        return question, False, []
    memory_turns = recent_turns(db, session_id, limit=CONVERSATION_MEMORY_MAX_TURNS)
    if not memory_turns or not _needs_disambiguation(question):
        return question, False, memory_turns
    previous = memory_turns[-1]
    rewritten = _llm_resolve(question, previous)
    if rewritten:
        return rewritten, True, memory_turns
    return question, False, memory_turns


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
