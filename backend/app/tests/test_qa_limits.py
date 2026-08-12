# -*- coding: utf-8 -*-
"""提问限制测试：问题长度上限 400、每 IP 累计提问配额 429、client_ip 落库。

覆盖：
- 问题超过 QA_MAX_QUESTION_CHARS（默认 500 字）→ 400
- 每 IP 累计提问数达 QA_IP_MAX_QUESTIONS（默认 20）→ 429（写 qa_logs 即消耗）
- 其他 IP 不受影响、管理员请求不受配额限制、配额可配置
- QALog / QATask 的 client_ip 落库（/ask 与任务路径）

隔离说明：独立 SQLite 引擎 + patch SessionLocal，不与同进程其他测试模块互踩库。
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from backend.app.core import config
from backend.app.core.database import Base
from backend.app.models.document import QALog, QATask
from backend.app.schemas.qa import QAResponse
from backend.app.services import rag_service
from backend.main import app

TEST_DATABASE_PATH = Path(tempfile.gettempdir()) / f"resumemind-limits-test-{os.getpid()}.db"
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
    """每个用例独立 client：cookie 不跨用例残留。"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def qa_env(monkeypatch):
    """隔离环境：独立数据库 + 快路径问答生成（不触碰本地模型）+ 预算放大。"""
    reset_database()
    monkeypatch.setattr(config, "QA_GLOBAL_DAILY_LIMIT", 100000)
    monkeypatch.setattr("backend.app.core.database.SessionLocal", MySession)
    monkeypatch.setattr("backend.app.api.qa.SessionLocal", MySession)
    monkeypatch.setattr("backend.app.services.qa_task_service.SessionLocal", MySession)

    def fake_answer(db, question, **kwargs):
        response = QAResponse(
            answer=f"answer:{question}",
            answer_mode="answered",
            evidence_sufficiency="sufficient",
            intent="resume_qa",
            generation_status="completed",
        )
        # 走真实日志写入（跳过 LLM/检索），验证 client_ip 从 API 层透传到 QALog
        rag_service._save_qa_log(db, question, response, client_ip=kwargs.get("client_ip"))
        return response

    monkeypatch.setattr("backend.app.api.qa.answer_question", fake_answer)
    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", fake_answer)
    yield


def _seed_qa_logs(count: int, client_ip: str = "testclient") -> None:
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
                    client_ip=client_ip,
                )
            )
        db.commit()


class TestQuestionLengthLimit:
    def test_question_over_500_chars_rejected(self, client) -> None:
        response = client.post("/api/qa/ask", json={"question": "很" * 501})
        assert response.status_code == 400
        assert "问题过长" in response.json()["detail"]

    def test_question_500_chars_accepted(self, client) -> None:
        response = client.post("/api/qa/ask", json={"question": "很" * 500})
        assert response.status_code == 200

    def test_task_creation_rejects_long_question(self, client) -> None:
        response = client.post(
            "/api/qa/tasks",
            json={"question": "很" * 501, "client_request_id": "limits-task-0001"},
        )
        assert response.status_code == 400


class TestIpQuestionQuota:
    def test_ip_exhausted_returns_429(self, client) -> None:
        _seed_qa_logs(20)  # testclient 已问满 20 个
        response = client.post("/api/qa/ask", json={"question": "你好"})
        assert response.status_code == 429
        assert "上限" in response.json()["detail"]

    def test_ip_below_quota_still_asks(self, client) -> None:
        _seed_qa_logs(19)
        response = client.post("/api/qa/ask", json={"question": "你好"})
        assert response.status_code == 200

    def test_other_ip_unaffected(self, client) -> None:
        _seed_qa_logs(20, client_ip="203.0.113.7")
        response = client.post("/api/qa/ask", json={"question": "你好"})
        assert response.status_code == 200

    def test_admin_request_exempt_from_quota(self, client) -> None:
        _seed_qa_logs(20)
        login = client.post("/api/auth/login", json={"password": os.environ["ADMIN_PASSWORD"]})
        assert login.status_code == 200
        response = client.post(
            "/api/qa/ask",
            json={"question": "你好"},
            headers={"Authorization": f"Bearer {login.json()['token']}"},
        )
        assert response.status_code == 200

    def test_quota_applies_to_task_creation(self, client) -> None:
        _seed_qa_logs(20)
        response = client.post(
            "/api/qa/tasks",
            json={"question": "你好", "client_request_id": "limits-task-0002"},
        )
        assert response.status_code == 429

    def test_quota_configurable(self, monkeypatch, client) -> None:
        monkeypatch.setattr(config, "QA_IP_MAX_QUESTIONS", 2)
        _seed_qa_logs(2)
        response = client.post("/api/qa/ask", json={"question": "你好"})
        assert response.status_code == 429


class TestClientIpLogging:
    def test_ask_logs_client_ip(self, client) -> None:
        response = client.post("/api/qa/ask", json={"question": "你的项目经历"})
        assert response.status_code == 200
        with MySession() as db:
            log = db.scalar(select(QALog))
            assert log is not None
            assert log.client_ip == "testclient"

    def test_task_creation_saves_client_ip(self, client) -> None:
        client.post(
            "/api/qa/tasks",
            json={"question": "你的项目经历", "client_request_id": "limits-task-0003"},
        )
        with MySession() as db:
            task = db.scalar(
                select(QATask).where(QATask.client_request_id == "limits-task-0003")
            )
            assert task is not None
            assert task.client_ip == "testclient"

    def test_task_execution_logs_client_ip_end_to_end(self, client) -> None:
        """任务在 worker 线程执行，QALog 的 client_ip 应来自创建时落库的 QATask.client_ip。"""
        import time

        response = client.post(
            "/api/qa/tasks",
            json={"question": "你的项目经历", "client_request_id": "limits-task-0004"},
        )
        assert response.status_code == 200
        task_id = response.json()["task_id"]
        status = None
        for _ in range(100):
            status = client.get(f"/api/qa/tasks/{task_id}").json()
            if status["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert status is not None and status["status"] == "completed", status
        with MySession() as db:
            log = db.scalar(select(QALog).order_by(QALog.id.desc()))
            assert log is not None
            assert log.client_ip == "testclient"
