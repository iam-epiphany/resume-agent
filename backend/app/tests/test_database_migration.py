# -*- coding: utf-8 -*-
"""数据库迁移测试（2026-08-15）：事实状态拆分回填只执行一次，不重置人工审核。

回归场景：_upgrade_sqlite_schema 的存量回填 UPDATE 此前每次启动都执行，
会把 review_status 强制写回 'pending'，抹掉人工审核结果；修复后回填仅在
两列首次补齐时执行一次。
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.core import database


def _legacy_engine(tmp_path):
    """旧版库：迁移看门表 document_chunks + 最小 documents + 旧版 fact_ledger。

    documents 只需迁移代码引用的列（id/filename/filename_norm/uploaded_at），
    其余列由迁移本身的 ALTER 补齐。
    """
    engine = create_engine(f"sqlite:///{(tmp_path / 'migrate.db').as_posix()}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE document_chunks (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE documents ("
                "id INTEGER PRIMARY KEY, document_id VARCHAR(64), filename VARCHAR(255), "
                "filename_norm VARCHAR(255), uploaded_at DATETIME)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE fact_ledger ("
                "id INTEGER PRIMARY KEY, fact_id VARCHAR(64), subject VARCHAR(255), "
                "predicate VARCHAR(255), value VARCHAR(500), status VARCHAR(20), "
                "source_file VARCHAR(255))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO fact_ledger (fact_id, subject, predicate, value, status, source_file) "
                "VALUES "
                "('legacy_1', 'XDU EchoGuide', '并发', '200 QPS', 'pending', '压测.md'), "
                "('legacy_2', '学校', '学历', '本科', 'confirmed', '教育背景.md')"
            )
        )
    return engine


def test_fact_status_backfill_runs_once_and_preserves_review(tmp_path, monkeypatch) -> None:
    engine = _legacy_engine(tmp_path)
    monkeypatch.setattr(database, "engine", engine)

    # 首次启动：补列 + 回填（pending→missing、confirmed→explicit，review 均 pending）
    database._upgrade_sqlite_schema()
    Session = sessionmaker(bind=engine)
    with Session() as db:
        rows = db.execute(
            text("SELECT fact_id, evidence_status, review_status FROM fact_ledger ORDER BY id")
        ).all()
        assert rows == [
            ("legacy_1", "missing", "pending"),
            ("legacy_2", "explicit", "pending"),
        ]

    # 模拟人工审核结果
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE fact_ledger SET review_status='approved' WHERE fact_id='legacy_1'")
        )

    # 再次启动：列已存在，不得回填、不得重置人工审核状态
    database._upgrade_sqlite_schema()
    with Session() as db:
        rows = db.execute(
            text("SELECT fact_id, evidence_status, review_status FROM fact_ledger ORDER BY id")
        ).all()
        assert rows == [
            ("legacy_1", "missing", "approved"),
            ("legacy_2", "explicit", "pending"),
        ]
