# -*- coding: utf-8 -*-
"""访客问答访问码闸与全局预算保险丝测试。

覆盖：
- 闸门默认关闭（QA_ACCESS_CODE 为空）时访客可直接提问
- 闸门开启：无访问凭证 401、错误访问码 403、正确访问码签发 cookie 后可提问
- 管理员 token 旁路闸门
- 全局每日预算：告警头 X-QA-Budget-Warning、预算用尽 429
- /qa/access/status 状态端点

隔离说明：本模块使用独立的 SQLite 引擎，并把 api.qa 的 get_db 与
qa_task_service 的 SessionLocal 都 patch 到本引擎——不与 test_rag_api 的
全局 SessionLocal 重绑冲突（两个模块共用同一 app/全局 SessionLocal 会互相踩库）。
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core import config
from backend.app.core.database import Base
from backend.app.models.document import QALog
from backend.app.schemas.qa import QAResponse
from backend.main import app

ACCESS_CODE = "Wbz_123"

TEST_DATABASE_PATH = Path(tempfile.gettempdir()) / f"resumemind-access-test-{os.getpid()}.db"
test_engine = create_engine(
    f"sqlite:///{TEST_DATABASE_PATH.as_posix()}",
    connect_args={"check_same_thread": False},
)
MySession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)


def reset_database() -> None:
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)


@pytest.fixture()
def client() -> TestClient:
    """每个用例独立 client：cookie 不跨用例残留（访问码 JWT 存 cookie）。"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def qa_env(monkeypatch):
    """隔离环境：独立数据库 + 快路径问答生成（不触碰本地模型）+ 闸门默认关闭。

    Depends(get_db) 在路由注册时已捕获 get_db 函数引用，patch 模块属性无效；
    正确做法是 patch 各模块持有的 SessionLocal 引用（database 模块全局 /
    api.qa 的流式会话 / qa_task_service 的任务会话）。
    """
    reset_database()
    monkeypatch.setattr(config, "QA_ACCESS_CODE", "")
    monkeypatch.setattr(config, "QA_GLOBAL_DAILY_LIMIT", 100000)
    monkeypatch.setattr("backend.app.core.database.SessionLocal", MySession)
    monkeypatch.setattr("backend.app.api.qa.SessionLocal", MySession)
    monkeypatch.setattr("backend.app.services.qa_task_service.SessionLocal", MySession)

    def fake_answer(db, question, **kwargs):
        return QAResponse(
            answer=f"answer:{question}",
            answer_mode="answered",
            evidence_sufficiency="sufficient",
            intent="resume_qa",
            generation_status="completed",
        )

    monkeypatch.setattr("backend.app.api.qa.answer_question", fake_answer)
    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", fake_answer)
    yield


def _enable_gate(monkeypatch, code: str = ACCESS_CODE) -> None:
    monkeypatch.setattr(config, "QA_ACCESS_CODE", code)


def _seed_qa_logs(count: int) -> None:
    with MySession() as db:
        for index in range(count):
            db.add(
                QALog(
                    question=f"seed-{index}",
                    answer="seed",
                    answer_mode="answered",
                    evidence_sufficiency="sufficient",
                    fallback_level=0,
                    used_chunks=1,
                )
            )
        db.commit()


class TestAccessGate:
    def test_gate_disabled_visitor_can_ask(self, client) -> None:
        response = client.post("/api/qa/ask", json={"question": "你好"})
        assert response.status_code == 200
        assert response.json()["answer"].startswith("answer:")

    def test_gate_blocks_unauthorized(self, monkeypatch, client) -> None:
        _enable_gate(monkeypatch)
        response = client.post("/api/qa/ask", json={"question": "你好"})
        assert response.status_code == 401
        assert "访问码" in response.json()["detail"]

    def test_wrong_code_rejected(self, monkeypatch, client) -> None:
        _enable_gate(monkeypatch)
        response = client.post("/api/qa/access", json={"code": "wrong-code"})
        assert response.status_code == 403

    def test_correct_code_grants_cookie_and_access(self, monkeypatch, client) -> None:
        _enable_gate(monkeypatch)
        # 提交访问码 → 签发 httpOnly cookie
        login = client.post("/api/qa/access", json={"code": ACCESS_CODE})
        assert login.status_code == 200
        assert login.json()["granted"] is True
        assert "qa_access" in login.cookies
        # 携带 cookie 后可以提问
        response = client.post("/api/qa/ask", json={"question": "你的项目经历"})
        assert response.status_code == 200
        assert response.json()["answer"].startswith("answer:")

    def test_admin_token_bypasses_gate(self, monkeypatch, client) -> None:
        _enable_gate(monkeypatch)
        login = client.post(
            "/api/auth/login", json={"password": os.environ["ADMIN_PASSWORD"]}
        )
        assert login.status_code == 200
        response = client.post(
            "/api/qa/ask",
            json={"question": "你好"},
            headers={"Authorization": f"Bearer {login.json()['token']}"},
        )
        assert response.status_code == 200

    def test_access_status_endpoint(self, monkeypatch, client) -> None:
        # 闸门开启且未登录
        _enable_gate(monkeypatch)
        status = client.get("/api/qa/access/status").json()
        assert status["access_enabled"] is True
        assert status["granted"] is False
        assert status["daily_remaining"] == 100000
        # 登录后 granted=True
        client.post("/api/qa/access", json={"code": ACCESS_CODE})
        status = client.get("/api/qa/access/status").json()
        assert status["granted"] is True


class TestBudgetFuse:
    def test_budget_warning_header(self, monkeypatch, client) -> None:
        _enable_gate(monkeypatch)
        client.post("/api/qa/access", json={"code": ACCESS_CODE})
        # 剩余 100 且阈值放得很大 → 恒告警
        monkeypatch.setattr(config, "QA_GLOBAL_DAILY_LIMIT", 100)
        monkeypatch.setattr(config, "QA_BUDGET_WARNING_REMAINING", 1000)
        response = client.post("/api/qa/ask", json={"question": "你好"})
        assert response.status_code == 200
        assert response.headers.get("X-QA-Budget-Remaining") == "100"
        assert response.headers.get("X-QA-Budget-Warning") == "1"

    def test_budget_exhausted_returns_429(self, monkeypatch, client) -> None:
        _enable_gate(monkeypatch)
        client.post("/api/qa/access", json={"code": ACCESS_CODE})
        _seed_qa_logs(2)
        monkeypatch.setattr(config, "QA_GLOBAL_DAILY_LIMIT", 2)
        response = client.post("/api/qa/ask", json={"question": "你好"})
        assert response.status_code == 429
        assert "预算" in response.json()["detail"]

    def test_budget_warning_on_task_creation(self, monkeypatch, client) -> None:
        _enable_gate(monkeypatch)
        client.post("/api/qa/access", json={"code": ACCESS_CODE})
        monkeypatch.setattr(config, "QA_GLOBAL_DAILY_LIMIT", 10)
        monkeypatch.setattr(config, "QA_BUDGET_WARNING_REMAINING", 1000)
        response = client.post(
            "/api/qa/tasks",
            json={"question": "你好", "client_request_id": "access-test-0001"},
        )
        assert response.status_code == 200
        assert response.headers.get("X-QA-Budget-Warning") == "1"


class TestDailyRemainingQuery:
    def test_count_only_today_rows(self) -> None:
        """预算计数只统计当日 qa_logs（跨天不累计）。"""
        from datetime import datetime, timedelta, timezone

        from backend.app.api.qa import _global_daily_qa_remaining

        with MySession() as db:
            yesterday = QALog(
                question="yesterday",
                answer="x",
                answer_mode="answered",
                evidence_sufficiency="sufficient",
                fallback_level=0,
                used_chunks=1,
                created_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
            today_1 = QALog(
                question="today-1",
                answer="x",
                answer_mode="answered",
                evidence_sufficiency="sufficient",
                fallback_level=0,
                used_chunks=1,
            )
            db.add_all([yesterday, today_1])
            db.commit()
            remaining, _ = _global_daily_qa_remaining(db)
        assert remaining == 99999  # 预算 100000 - 今日 1 条
