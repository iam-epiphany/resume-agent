# -*- coding: utf-8 -*-
"""多轮追问记忆服务测试：轮次记录 / 会话隔离 / TTL 清理（指代消解已并入意图路由）。"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import ConversationTurn
from backend.app.services import conversation_memory_service
from backend.app.services.conversation_memory_service import (
    purge_expired,
    recent_turns,
    record_turn,
)


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'memory.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        conversation_memory_service._MEMORY_LRU.clear()
        yield db
    conversation_memory_service._MEMORY_LRU.clear()
    engine.dispose()


# ---------------------------------------------------------------------------
# record_turn / recent_turns
# ---------------------------------------------------------------------------

def test_record_turn_and_recent_turns_roundtrip(db_session) -> None:
    record_turn(
        db_session, "session-roundtrip",
        question="介绍一下项目经历",
        resolved_question="介绍一下项目经历",
        intent="resume_qa",
        answer_mode="answered",
        answer_excerpt="参与过外卖平台项目。",
    )
    record_turn(
        db_session, "session-roundtrip",
        question="那怎么处理高并发的？",
        resolved_question="外卖平台项目中如何处理高并发？",
        intent="resume_qa",
        answer_mode="hedged",
        answer_excerpt="根据现有知识库推测，使用 Redis 缓存。",
    )

    turns = recent_turns(db_session, "session-roundtrip")

    assert len(turns) == 2
    assert turns[0]["question"] == "介绍一下项目经历"
    assert turns[0]["intent"] == "resume_qa"
    assert turns[0]["answer_mode"] == "answered"
    assert turns[1]["question"] == "那怎么处理高并发的？"
    assert turns[1]["resolved_question"] == "外卖平台项目中如何处理高并发？"
    assert turns[1]["answer_excerpt"] == "根据现有知识库推测，使用 Redis 缓存。"


def test_record_turn_persists_to_sqlite(db_session) -> None:
    conversation_memory_service._MEMORY_LRU.clear()
    record_turn(
        db_session, "session-db",
        question="你的技术栈是什么",
        resolved_question=None,
        intent="resume_qa",
        answer_mode="answered",
        answer_excerpt="Python、Java 与 Redis。",
    )

    conversation_memory_service._MEMORY_LRU.clear()
    turns = recent_turns(db_session, "session-db")

    assert len(turns) == 1
    assert turns[0]["question"] == "你的技术栈是什么"
    assert turns[0]["resolved_question"] == "你的技术栈是什么"  # 未消解时回退原文


def test_turn_seq_increments_within_session(db_session) -> None:
    conversation_memory_service._MEMORY_LRU.clear()
    record_turn(db_session, "session-seq", question="第一轮", resolved_question="第一轮", intent="resume_qa", answer_mode="answered")
    record_turn(db_session, "session-seq", question="第二轮", resolved_question="第二轮", intent="resume_qa", answer_mode="answered")

    rows = db_session.scalars(
        select(ConversationTurn)
        .where(ConversationTurn.session_id == "session-seq")
        .order_by(ConversationTurn.turn_seq.asc())
    ).all()

    assert [row.turn_seq for row in rows] == [1, 2]


def test_record_turn_ignored_without_session(db_session) -> None:
    record_turn(
        db_session, None,
        question="你的技术栈是什么",
        resolved_question="你的技术栈是什么",
        intent="resume_qa",
        answer_mode="answered",
    )

    assert db_session.scalars(select(ConversationTurn)).all() == []


def test_session_isolation(db_session) -> None:
    record_turn(
        db_session, "session-a",
        question="A 的问题",
        resolved_question="A 的问题",
        intent="resume_qa",
        answer_mode="answered",
    )

    assert recent_turns(db_session, "session-a")
    assert recent_turns(db_session, "session-b") == []


def test_recent_turns_respects_limit(db_session) -> None:
    for index in range(10):
        record_turn(
            db_session, "session-limit",
            question=f"第{index}轮",
            resolved_question=f"第{index}轮",
            intent="resume_qa",
            answer_mode="answered",
        )

    turns = recent_turns(db_session, "session-limit", limit=3)

    assert [turn["question"] for turn in turns] == ["第7轮", "第8轮", "第9轮"]


# ---------------------------------------------------------------------------
# purge_expired
# ---------------------------------------------------------------------------

def test_purge_expired_deletes_stale_turns_only(db_session) -> None:
    now = datetime.now(timezone.utc)
    db_session.add(
        ConversationTurn(
            session_id="session-stale",
            turn_seq=1,
            question="旧问题",
            resolved_question="旧问题",
            intent="resume_qa",
            answer_mode="answered",
            created_at=now - timedelta(hours=48),
        )
    )
    db_session.add(
        ConversationTurn(
            session_id="session-fresh",
            turn_seq=1,
            question="新问题",
            resolved_question="新问题",
            intent="resume_qa",
            answer_mode="answered",
            created_at=now,
        )
    )
    db_session.commit()
    conversation_memory_service._MEMORY_LRU.clear()

    deleted = purge_expired(db_session)

    assert deleted == 1
    remaining = db_session.scalars(select(ConversationTurn)).all()
    assert [row.session_id for row in remaining] == ["session-fresh"]
