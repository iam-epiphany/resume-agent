# -*- coding: utf-8 -*-
"""事实台账服务测试（2026-08-14）：实体—属性—值错配校验 + 种子幂等导入。"""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import FactLedger
from backend.app.services.fact_ledger_service import check_answer, seed_fact_records


def _fact(
    fact_id: str,
    subject: str,
    predicate: str,
    value: str,
    status: str = "confirmed",
) -> FactLedger:
    return FactLedger(
        fact_id=fact_id,
        subject=subject,
        predicate=predicate,
        value=value,
        status=status,
        source_file="测试.md",
    )


FACTS = [
    _fact("ai_test", "AI 开放平台", "实测结果", "10用户并发抢购同一SKU恰好10单成功"),
    _fact("echo_test", "XDU-EchoGuide", "实测结果", "285项测试通过"),
    _fact("rev_param", "REV 密码算法", "环参数", "n=1152，q=2017素数"),
    _fact("edu_gpa", "河南大学", "本科绩点", "3.61/4"),
    _fact("award_pending", "大学生算法大赛", "获奖级别", "二等奖（年份待确认）", status="pending"),
]


def test_correct_attribution_no_mismatch() -> None:
    answer = "AI 开放平台实测 10用户并发抢购同一SKU恰好10单成功，无超卖。"
    result = check_answer(answer, FACTS)
    assert result.verified
    assert result.mismatches == []


def test_misattribution_detected() -> None:
    """REV 项目名下出现 AI 开放平台的实测结果 → 张冠李戴 mismatch。"""
    answer = "REV 密码算法压测结果：10用户并发抢购同一SKU恰好10单成功。"
    result = check_answer(answer, FACTS)
    assert not result.verified
    assert any("AI 开放平台" in item for item in result.mismatches)


def test_own_value_present_suppresses_window_check() -> None:
    """两个项目的实测结果同时正确出现时不误报。"""
    answer = (
        "AI 开放平台实测 10用户并发抢购同一SKU恰好10单成功；"
        "XDU-EchoGuide 有 285项测试通过。"
    )
    result = check_answer(answer, FACTS)
    assert result.verified


def test_pending_fact_citation_warning() -> None:
    answer = "大学生算法大赛拿了二等奖。"
    result = check_answer(answer, FACTS)
    assert any("待确认" in item for item in result.warnings)


def test_numbers_without_subject_are_ignored() -> None:
    """纯数字（如 3.61/4）单独出现但同属性无其他实体在场 → 不告警。"""
    result = check_answer("我的绩点是 3.61/4。", FACTS)
    assert result.warnings == []


def test_seed_fact_records_idempotent(tmp_path) -> None:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'ledger.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    records = [
        {
            "fact_id": "edu_gpa",
            "subject": "河南大学",
            "predicate": "本科绩点",
            "value": "3.61/4",
            "status": "confirmed",
            "source_file": "教育背景.md",
        },
        {
            "fact_id": "edu_rank",
            "subject": "河南大学",
            "predicate": "专业排名",
            "value": "6/177",
            "status": "confirmed",
            "source_file": "教育背景.md",
        },
    ]
    with session_factory() as db:
        assert seed_fact_records(db, records) == 2
        # 重复导入（修改了 value）仍是 2 条且值被更新
        records[0]["value"] = "3.62/4"
        assert seed_fact_records(db, records) == 2
        rows = db.scalars(select(FactLedger).order_by(FactLedger.fact_id.asc())).all()
    assert len(rows) == 2
    by_id = {row.fact_id: row for row in rows}
    assert by_id["edu_gpa"].value == "3.62/4"
    assert by_id["edu_rank"].value == "6/177"
    engine.dispose()
