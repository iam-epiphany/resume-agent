"""前台/后台权限分离的认证与限流测试。

数据库隔离说明：test_rag_api 模块级会把全局 SessionLocal 重新绑定到它自己的临时库，
本模块无法依赖全局绑定。因此用 autouse fixture 建立独立临时 SQLite，
并把各 API 模块的 get_db 依赖替换为独立库的 session —— 测试完全自洽，
且不影响 test_rag_api（它仍用自己绑定后的 SessionLocal）。
限流中间件运行时动态读取 config.*，故 monkeypatch 属性即可局部开启限流。
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import time

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core import config, security
from backend.app.core.database import Base
from backend.app.middleware.rate_limit import RateLimitMiddleware
from backend.app.models.audit import AuditLog
from backend.app.models.document import QATask
from backend.app.services.audit_service import list_audit_logs, record_event
from backend.main import app


client = TestClient(app)
TEST_DATABASE_PATH = Path(tempfile.gettempdir()) / f"resumemind-auth-test-{os.getpid()}.db"

_own_sessionmaker: sessionmaker | None = None


@pytest.fixture(autouse=True)
def _isolated_database(monkeypatch) -> None:
    """独立临时库：patch SessionLocal（各端点 get_db 内部调用它），fixture 结束自动还原。"""
    global _own_sessionmaker

    test_engine = create_engine(
        f"sqlite:///{TEST_DATABASE_PATH.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=test_engine)
    _own_sessionmaker = sessionmaker(bind=test_engine)

    monkeypatch.setattr("backend.app.core.database.SessionLocal", _own_sessionmaker)
    # 直接持有 SessionLocal 引用的服务模块也一并替换（/api/health/rag 链路）
    monkeypatch.setattr("backend.app.api.qa.SessionLocal", _own_sessionmaker)
    monkeypatch.setattr("backend.app.services.index_task_service.SessionLocal", _own_sessionmaker)
    monkeypatch.setattr("backend.app.services.qa_task_service.SessionLocal", _own_sessionmaker)


def _admin_headers() -> dict[str, str]:
    response = client.post("/api/auth/login", json={"password": os.environ["ADMIN_PASSWORD"]})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def _clear_rate_limit_state() -> None:
    """清空共享限流计数，保证限流测试的确定性（不受前面测试的登录/QA 请求污染）。"""
    node = app.middleware_stack
    for _ in range(20):
        if isinstance(node, RateLimitMiddleware):
            node._minute_hits.clear()
            node._daily_hits.clear()
            node._active_requests = 0
            return
        node = getattr(node, "app", None)
    raise AssertionError("RateLimitMiddleware not found in middleware stack")


# ---------------------------------------------------------------------------
# 认证
# ---------------------------------------------------------------------------

def test_admin_endpoints_reject_anonymous() -> None:
    for path in ["/api/documents", "/api/audit/logs", "/api/health/rag", "/api/health/ready"]:
        response = client.get(path)
        assert response.status_code == 401, path
        assert response.json()["error"]["code"] == "unauthorized", path


def test_admin_endpoints_accept_valid_token() -> None:
    headers = _admin_headers()
    for path in ["/api/documents", "/api/audit/logs", "/api/health/rag"]:
        response = client.get(path, headers=headers)
        assert response.status_code == 200, f"{path}: {response.text}"


def test_me_endpoint() -> None:
    assert client.get("/api/auth/me").status_code == 401
    response = client.get("/api/auth/me", headers=_admin_headers())
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_login_wrong_password_rejected_and_audited() -> None:
    response = client.post("/api/auth/login", json={"password": "definitely-wrong"})
    assert response.status_code == 401
    with _own_sessionmaker() as db:
        found = db.query(AuditLog).filter(AuditLog.action == "auth_login_failed").first()
        assert found is not None, "登录失败应写入审计日志"
        db.query(AuditLog).filter(AuditLog.action == "auth_login_failed").delete()
        db.commit()


def test_expired_token_rejected() -> None:
    now = datetime.now(timezone.utc)
    expired_payload = {
        "sub": "admin",
        "iat": now - timedelta(hours=2),
        "exp": now - timedelta(hours=1),
        "iss": security._TOKEN_ISSUER,
    }
    expired_token = pyjwt.encode(expired_payload, security._jwt_secret(), algorithm="HS256")
    response = client.get("/api/documents", headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == 401


def test_tampered_token_rejected() -> None:
    token = _admin_headers()["Authorization"].split(" ")[1]
    tampered = token[:-2] + ("A" if token[-2] != "A" else "B")
    response = client.get("/api/documents", headers={"Authorization": f"Bearer {tampered}"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def test_public_qa_status() -> None:
    response = client.get("/api/qa/status")
    assert response.status_code == 200
    body = response.json()
    assert "ready" in body and "message" in body


def test_public_qa_logs_endpoint_anonymous_ok() -> None:
    """匿名可访问问答日志端点，但（隐私）一律返回空列表；管理员可见。"""
    with _own_sessionmaker() as db:
        record_event(
            db, "qa_context_built", "question", "q1", detail="测试问答",
            event_key=f"test-qa-log-{time.time_ns()}",
        )
        db.commit()

    anonymous = client.get("/api/audit/qa-logs")
    assert anonymous.status_code == 200
    assert anonymous.json()["logs"] == []

    admin = client.get("/api/audit/qa-logs", headers=_admin_headers())
    assert admin.status_code == 200
    assert len(admin.json()["logs"]) == 1

    with _own_sessionmaker() as db:
        db.query(AuditLog).delete()
        db.commit()


def test_anonymous_task_list_is_empty() -> None:
    """问答历史（问题+答案全文）仅管理员可见：匿名列表为空。"""
    with _own_sessionmaker() as db:
        db.add(QATask(task_id="history-task-0001", question="你的弱点是什么", status="completed"))
        db.commit()

    anonymous = client.get("/api/qa/tasks")
    assert anonymous.status_code == 200
    assert anonymous.json() == []

    admin = client.get("/api/qa/tasks", headers=_admin_headers())
    assert admin.status_code == 200
    assert any(task["task_id"] == "history-task-0001" for task in admin.json())

    with _own_sessionmaker() as db:
        db.query(QATask).filter(QATask.task_id == "history-task-0001").delete()
        db.commit()


def test_retrieve_endpoint_requires_admin() -> None:
    """内部检索调试接口（含内部提示词与全文片段）仅管理员可用：匿名 403。"""
    anonymous = client.post("/api/qa/retrieve", json={"question": "介绍一下你的项目经历"})
    assert anonymous.status_code == 403

    admin = client.post("/api/qa/retrieve", json={"question": "介绍一下你的项目经历"}, headers=_admin_headers())
    assert admin.status_code in {200, 503}  # 503 属检索不可用（服务级测试覆盖），403 不应出现


def test_ask_include_debug_does_not_leak_context_for_anonymous(monkeypatch) -> None:
    """匿名即使显式 include_debug=true 也不得拿到检索依据（调试参数只对管理员生效）。"""
    from backend.app.schemas.qa import QAResponse

    def fake_answer_question(db, question, options=None, include_debug=False, **kwargs):
        return QAResponse(
            answer="测试回答",
            citations=[],
            confidence=0.9,
            refused=False,
            context_package={
                "query": question,
                "mode": "rag_context",
                "is_final_answer": False,
                "instruction": "",
                "retrieval_summary": {},
                "llm_prompt": "内部提示词",
                "context_chunks": [],
            },
            answer_type="llm_grounded",
            generation_status="completed",
        )

    monkeypatch.setattr("backend.app.api.qa.answer_question", fake_answer_question)

    anonymous = client.post("/api/qa/ask", json={"question": "介绍一下你的项目经历", "include_debug": True})
    assert anonymous.status_code == 200
    assert anonymous.json()["context_package"] is None

    admin = client.post("/api/qa/ask", json={"question": "介绍一下你的项目经历", "include_debug": True}, headers=_admin_headers())
    assert admin.status_code == 200
    assert admin.json()["context_package"] is not None


def test_task_retrieval_evidence_only_for_admin() -> None:
    """检索依据（context_package）属后台信息：匿名任务接口剥离，管理员可见。"""
    answer_json = json.dumps({
        "answer": "测试回答",
        "citations": [],
        "confidence": 0.9,
        "refused": False,
        "context_package": {
            "query": "你参与过哪些项目",
            "mode": "rag_context",
            "is_final_answer": False,
            "instruction": "",
            "retrieval_summary": {},
            "llm_prompt": "内部提示词",
            "context_chunks": [
                {
                    "chunk_id": "c1", "rank": 1, "score": 0.9,
                    "source_doc": "简历.pdf", "section_title": "项目经历",
                    "text": "参与过外卖平台项目。", "citation_label": "[1]",
                    "metadata": {"rerank_score": 0.9, "document_id": "D1"},
                }
            ],
        },
        "answer_type": "llm_grounded",
        "generation_status": "completed",
        "claims": [],
        "grounding_validation": {"passed": True},
    }, ensure_ascii=False)
    task_id = "evidence-test-task-0001"
    with _own_sessionmaker() as db:
        db.add(QATask(
            task_id=task_id,
            question="你参与过哪些项目",
            status="completed",
            answer_json=answer_json,
        ))
        db.commit()

    # 匿名：剥离 context_package
    anonymous = client.get(f"/api/qa/tasks/{task_id}")
    assert anonymous.status_code == 200
    assert anonymous.json()["answer"]["context_package"] is None

    # 管理员：保留检索依据
    admin = client.get(f"/api/qa/tasks/{task_id}", headers=_admin_headers())
    assert admin.status_code == 200
    package = admin.json()["answer"]["context_package"]
    assert package is not None
    assert package["context_chunks"][0]["text"] == "参与过外卖平台项目。"

    # SSE 快照（EventSource 无鉴权头）按匿名视角剥离
    with _own_sessionmaker() as db:
        db.query(QATask).filter(QATask.task_id == task_id).delete()
        db.commit()


def test_qa_logs_scope_filters_out_document_actions() -> None:
    with _own_sessionmaker() as db:
        record_event(
            db, "qa_context_built", "question", "q1", detail="测试问答",
            event_key=f"test-qa-{time.time_ns()}",
        )
        record_event(
            db, "document_uploaded", "document", "d1", detail="测试文档",
            event_key=f"test-doc-{time.time_ns()}",
        )
        all_actions = [log.action for log in list_audit_logs(db, scope="all")]
        qa_actions = [log.action for log in list_audit_logs(db, scope="qa")]
        db.query(AuditLog).delete()
        db.commit()

    assert "document_uploaded" in all_actions
    assert "qa_context_built" in all_actions
    assert "qa_context_built" in qa_actions
    assert "document_uploaded" not in qa_actions, "前台范围不应暴露知识库管理操作"


# ---------------------------------------------------------------------------
# 限流
# ---------------------------------------------------------------------------

def test_qa_rate_limit_rejects_excess_requests(monkeypatch) -> None:
    _clear_rate_limit_state()
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "QA_IP_RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(config, "QA_IP_DAILY_LIMIT", 100)

    statuses = [client.get("/api/qa/tasks?limit=1").status_code for _ in range(3)]
    assert statuses[0] != 429 and statuses[1] != 429
    assert statuses[2] == 429, f"第 3 次请求应被限流拒绝，实际 {statuses}"

    response = client.get("/api/qa/tasks?limit=1")
    assert response.status_code == 429
    assert response.headers.get("Retry-After") is not None
    assert response.headers.get("X-Request-ID") is not None
    assert response.json()["error"]["code"] == "too_many_requests"


def test_qa_rate_limit_per_minute_zero_means_unlimited(monkeypatch) -> None:
    """每分钟限流设为 0 = 不限制（仅保留每日上限，如 QA_GLOBAL_DAILY_LIMIT 全局预算）。"""
    _clear_rate_limit_state()
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "QA_IP_RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(config, "QA_IP_DAILY_LIMIT", 100000)

    statuses = [client.get("/api/qa/tasks?limit=1").status_code for _ in range(5)]
    assert all(status != 429 for status in statuses), f"0 = 不限分钟次数，实际 {statuses}"


def test_qa_status_exempt_from_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "QA_IP_RATE_LIMIT_PER_MINUTE", 1)
    for _ in range(3):
        response = client.get("/api/qa/status")
        assert response.status_code == 200


def test_login_rate_limit(monkeypatch) -> None:
    _clear_rate_limit_state()
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "LOGIN_RATE_LIMIT_PER_MINUTE", 3)
    statuses = [
        client.post("/api/auth/login", json={"password": "wrong"}).status_code for _ in range(4)
    ]
    assert statuses[:3] == [401, 401, 401]
    assert statuses[3] == 429


def test_qa_global_concurrency_limit(monkeypatch) -> None:
    """并发上限为 1 时，两个同时进行的 /api/qa/ask 请求至少一个被 429 拒绝。"""
    _clear_rate_limit_state()
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(config, "QA_GLOBAL_CONCURRENCY", 1)
    monkeypatch.setattr(config, "QA_IP_RATE_LIMIT_PER_MINUTE", 1000)
    monkeypatch.setattr(config, "QA_IP_DAILY_LIMIT", 100000)

    def slow_answer(*_args, **_kwargs):
        time.sleep(0.3)
        raise RuntimeError("slow_answer stub should never return")

    monkeypatch.setattr("backend.app.api.qa.answer_question", slow_answer)

    raw_client = TestClient(app, raise_server_exceptions=False)
    results: list[int] = []
    results_lock = threading.Lock()

    def hit() -> None:
        response = raw_client.post("/api/qa/ask", json={"question": "并发测试"})
        with results_lock:
            results.append(response.status_code)

    threads = [threading.Thread(target=hit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert 429 in results, f"并发上限应拒绝至少一个请求，实际状态码 {results}"
