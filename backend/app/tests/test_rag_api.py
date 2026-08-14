from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
import hashlib
import json
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
from threading import Event
import time
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy import text

from backend.app.core.config import AUDIT_ARCHIVE_DIR, INDEX_VERSION
from backend.app.core.database import Base, SessionLocal
from backend.app.models.audit import AuditLog
from backend.app.models.document import Document, DocumentChunk, DocumentIndexTask
from backend.app.schemas.qa import Citation, LLMContextPackage, QAAnswerPreview, QAResponse, RetrievalResult
from backend.app.schemas.health import RagHealthResponse
from backend.main import app
from backend.app.services.embedding_service import EmbeddingServiceError, TextEmbedding
from backend.app.services.document_storage import next_document_id
from backend.app.services.query_planner_service import QueryAspect, QueryPlan, QuerySearchQuery
from backend.app.services.retrieval_service import RetrievalDiagnostics, RetrievalMatch, RetrievalServiceUnavailable
from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.vector_store_service import VectorStoreError
from backend.app.services.vector_store_service import VectorSearchResult


client = TestClient(app)
# 前台/后台权限分离：知识库/日志/健康富信息接口已收进管理员后台，
# 模块级登录一次拿到 token 并注入默认头，覆盖全部现有调用（QA/公开接口不受影响）。
_admin_login = client.post("/api/auth/login", json={"password": os.environ["ADMIN_PASSWORD"]})
assert _admin_login.status_code == 200, f"admin login failed: {_admin_login.text}"
client.headers.update({"Authorization": f"Bearer {_admin_login.json()['token']}"})
TEST_DATABASE_PATH = Path(tempfile.gettempdir()) / f"resumemind-test-{os.getpid()}.db"
test_engine = create_engine(f"sqlite:///{TEST_DATABASE_PATH.as_posix()}", connect_args={"check_same_thread": False})
SessionLocal.configure(bind=test_engine)


def reset_database() -> None:
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)


@pytest.fixture(autouse=True)
def fake_indexing_services(monkeypatch, tmp_path):
    calls = []
    document_dir = tmp_path / "documents" / "originals"
    document_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("backend.app.services.document_storage.DOCUMENT_DIR", document_dir)
    monkeypatch.setattr("backend.app.services.document_lifecycle_service.DOCUMENT_DIR", document_dir)
    monkeypatch.setattr("backend.app.services.query_planner_service.QUERY_PLANNER_API_KEY", None)
    monkeypatch.setattr("backend.app.services.answer_generation_service.ANSWER_GENERATION_API_KEY", None)
    # 意图路由 LLM 兜底在测试环境不引入真实调用：全部问题按规则兜底 resume_detail
    monkeypatch.setattr("backend.app.services.intent_router_service.INTENT_ROUTER_ENABLED", False)

    def fake_embed_texts(texts: list[str]) -> list[TextEmbedding]:
        return [
            TextEmbedding(dense=[1.0, 0.0])
            for _ in texts
        ]

    def fake_upsert_chunk_embeddings(**kwargs) -> None:
        calls.append(kwargs)

    def fake_count_document_vectors(document_id: str) -> int:
        return len(calls[-1]["chunks"]) if calls else 0

    monkeypatch.setattr("backend.app.services.document_indexing_service.embed_texts", fake_embed_texts)
    monkeypatch.setattr("backend.app.services.document_indexing_service.upsert_chunk_embeddings", fake_upsert_chunk_embeddings)
    monkeypatch.setattr("backend.app.services.document_indexing_service.count_document_vectors", fake_count_document_vectors)
    monkeypatch.setattr(
        "backend.app.services.document_indexing_service.delete_document_vectors",
        lambda document_id, chunk_ids=None: None,
    )
    monkeypatch.setattr(
        "backend.app.services.document_lifecycle_service.delete_document_vectors",
        lambda document_id, chunk_ids=None: None,
    )

    def synchronous_test_enqueue(document_id: str, **kwargs) -> bool:
        with SessionLocal() as db:
            document = db.scalar(select(Document).where(Document.document_id == document_id))
            assert document is not None
            from backend.app.services.document_indexing_service import index_document
            from backend.app.services.document_processing_service import ensure_document_chunks
            from backend.app.models.document import DocumentIndexTask

            ensure_document_chunks(db, document, force_rebuild=bool(kwargs.get("force_rebuild_chunks")))
            index_document(db, document)
            task = db.scalar(
                select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
            )
            assert task is not None
            task.status = "completed"
            task.stage = "completed"
            db.commit()
        return True

    monkeypatch.setattr(
        "backend.app.api.documents.enqueue_document_index", synchronous_test_enqueue
    )
    yield calls


def fake_retrieval_match(document_id: str = "DOC-TEST-0001") -> RetrievalMatch:
    return RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="技能专长.md",
            section_title="综合成绩",
            page_number=None,
            excerpt="综合成绩应等于各科目分项成绩之和。",
            score=0.8,
            rerank_score=0.9,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.9,
        coverage_score=1.0,
        evidence_role="direct_evidence",
    )


def parse_sse_events(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in text.strip().split("\n\n"):
        event_name = "message"
        data = None
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            if line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if data is not None:
            events.append((event_name, data))
    return events


def assert_utc_iso_datetime(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def latest_progress_event(events: list[tuple[str, dict]], stage: str, *, aspect: bool | None = None) -> dict | None:
    for event_name, payload in reversed(events):
        if event_name != "progress" or payload.get("stage") != stage:
            continue
        if aspect is True and not payload.get("aspect_id"):
            continue
        if aspect is False and payload.get("aspect_id"):
            continue
        return payload
    return None


def test_health_check() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["message"] == "ResumeMind backend is healthy"


def test_rag_health_remains_diagnostic_while_readiness_returns_503(monkeypatch) -> None:
    health = RagHealthResponse(
        offline_mode=True,
        embedding_model_ready=True,
        reranker_model_ready=True,
        embedding_model_path="/models/bge-base-zh-v1.5",
        reranker_model_path="/models/reranker",
        qdrant_ready=True,
        qdrant_collection="resumemind_chunks",
        qdrant_collection_ready=False,
        sqlite_ready=True,
        libreoffice_ready=True,
        antiword_ready=True,
        index_tasks={"queued": 2, "queue_depth": 2, "queue_capacity": 8},
        ready=False,
    )
    monkeypatch.setattr("backend.app.api.health._rag_health", lambda: health)

    diagnostic = client.get("/api/health/rag")
    readiness = client.get("/api/health/ready")

    assert diagnostic.status_code == 200
    assert diagnostic.json()["index_tasks"]["queued"] == 2
    assert readiness.status_code == 503
    assert readiness.json()["qdrant_collection_ready"] is False


def test_openapi_only_exposes_rag_main_routes() -> None:
    paths = set(app.openapi()["paths"])

    assert "/api/health" in paths
    assert "/api/health/rag" in paths
    assert "/api/health/ready" in paths
    assert "/api/documents/upload" in paths
    assert "/api/documents/upload-preflight" in paths
    assert "/api/documents/batch-upload" in paths
    assert "/api/documents/bulk-delete" in paths
    assert "/api/documents/{document_id}/index" in paths
    assert "/api/documents" in paths
    assert "/api/documents/{document_id}" in paths
    assert "delete" in app.openapi()["paths"]["/api/documents/{document_id}"]
    assert "/api/qa/ask" in paths
    assert "/api/qa/ask/stream" in paths
    assert "/api/qa/retrieve" in paths
    assert "/api/audit/logs" in paths
    assert "/api/audit/archives" in paths
    assert "/api/audit/archives/{archive_date}" in paths
    assert "/api/reports/upload" not in paths
    assert app.openapi()["info"]["title"] == "ResumeMind API"


def test_upload_txt_document_creates_durable_processing_task(fake_indexing_services) -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "技能专长.md",
                "# 技能专长说明\n\n## 综合成绩\n综合成绩应等于各科目分项成绩之和。\n",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"].startswith("DOC-")
    assert body["task_id"]
    assert body["chunk_count"] in {0, 1}
    assert_utc_iso_datetime(body["uploaded_at"])
    assert len(fake_indexing_services) == 1
    documents = client.get("/api/documents").json()["documents"]
    assert documents[0]["status"] == "indexed"
    assert documents[0]["chunk_count"] == 1
    assert_utc_iso_datetime(documents[0]["uploaded_at"])
    detail = client.get(f"/api/documents/{body['document_id']}").json()
    assert_utc_iso_datetime(detail["uploaded_at"])
    assert_utc_iso_datetime(detail["chunks"][0]["created_at"])
    processing = client.get(f"/api/documents/{body['document_id']}/processing")
    assert processing.status_code == 200
    assert processing.json()["status"] == "completed"
    assert_utc_iso_datetime(processing.json()["updated_at"])


def test_upload_metadata_propagates_to_document_and_chunks(fake_indexing_services) -> None:
    reset_database()
    metadata = {
        "external_doc_id": "HENU-2026-001",
        "title": "技能专长说明",
        "issuing_authority": "河南大学",
        "publication_date": "2026-01-02",
        "source_url": "https://www.henu.edu.cn/skills/1",
        "attachment_url": "https://www.henu.edu.cn/skills/1.md",
        "material_topic": "技能掌握",
    }
    response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "第一条 应当按期更新。", "text/plain")},
        data={"metadata_json": json.dumps(metadata, ensure_ascii=False)},
    )

    assert response.status_code == 202
    detail = client.get(f"/api/documents/{response.json()['document_id']}").json()
    assert detail["metadata"]["external_doc_id"] == "HENU-2026-001"
    assert detail["metadata"]["source_url"] == metadata["source_url"]
    assert detail["chunks"][0]["metadata"]["issuing_authority"] == metadata["issuing_authority"]


def test_document_identity_patch_clear_confirm_and_invalidate(
    fake_indexing_services,
    monkeypatch,
) -> None:
    reset_database()
    refreshed: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "backend.app.services.document_metadata_index_service.refresh_document_metadata_payload",
        lambda document_id, metadata: refreshed.append((document_id, metadata)),
    )
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("identity.txt", "技能专长正文。", "text/plain")},
        data={
            "metadata_json": json.dumps(
                {"title": "原标题", "publication_date": "2026-01-02"},
                ensure_ascii=False,
            )
        },
    )
    document_id = upload.json()["document_id"]
    initial_index_calls = len(fake_indexing_services)

    patch = client.patch(
        f"/api/documents/{document_id}/metadata",
        json={"title": "人工标题", "publication_date": None},
    )

    assert patch.status_code == 200
    assert patch.json()["metadata"]["title"] == "人工标题"
    assert "publication_date" not in patch.json()["metadata"]
    assert patch.json()["metadata"]["identity_review_status"] == "unreviewed"
    assert patch.json()["reindex_queued"] is False
    assert patch.json()["metadata_refreshed"] is True
    assert len(fake_indexing_services) == initial_index_calls
    assert refreshed[-1][0] == document_id

    confirmed = client.post(f"/api/documents/{document_id}/metadata/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["metadata"]["identity_review_status"] == "confirmed"
    assert len(confirmed.json()["metadata"]["identity_reviewed_snapshot_hash"]) == 64
    assert confirmed.json()["metadata_refreshed"] is True

    changed = client.patch(
        f"/api/documents/{document_id}/metadata",
        json={"document_number": "河大〔2026〕1号"},
    )
    assert changed.status_code == 200
    assert changed.json()["metadata"]["identity_review_status"] == "unreviewed"
    assert changed.json()["metadata"]["identity_reviewed_snapshot_hash"] is None
    assert changed.json()["metadata_refreshed"] is True

    with SessionLocal() as db:
        identity_logs = [
            log
            for log in db.scalars(select(AuditLog).order_by(AuditLog.id.asc())).all()
            if log.action in {"document_identity_updated", "document_identity_confirmed"}
        ]
    actions = [log.action for log in identity_logs]
    assert "document_identity_updated" in actions
    assert "document_identity_confirmed" in actions
    assert all(log.user_message == "身份信息已保存并同步到检索元数据。" for log in identity_logs)
    assert all(json.loads(log.details_json or "{}")["metadata_refreshed"] is True for log in identity_logs)


def test_document_identity_refresh_failure_records_degraded_audit(
    fake_indexing_services,
    monkeypatch,
) -> None:
    reset_database()

    def failing_refresh_payload(document_id: str, metadata: dict) -> None:
        raise VectorStoreError("qdrant unavailable")

    monkeypatch.setattr(
        "backend.app.services.document_metadata_index_service.refresh_document_metadata_payload",
        failing_refresh_payload,
    )
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("identity.txt", "技能专长正文。", "text/plain")},
        data={
            "metadata_json": json.dumps(
                {"title": "原标题", "publication_date": "2026-01-02"},
                ensure_ascii=False,
            )
        },
    )
    document_id = upload.json()["document_id"]

    patch = client.patch(
        f"/api/documents/{document_id}/metadata",
        json={"title": "人工标题", "publication_date": None},
    )

    assert patch.status_code == 200
    assert patch.json()["metadata_refreshed"] is False
    assert "Qdrant payload 待刷新" in patch.json()["refresh_warning"]

    confirmed = client.post(f"/api/documents/{document_id}/metadata/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["metadata_refreshed"] is False
    assert "Qdrant payload 待刷新" in confirmed.json()["refresh_warning"]

    with SessionLocal() as db:
        identity_logs = [
            log
            for log in db.scalars(select(AuditLog).order_by(AuditLog.id.asc())).all()
            if log.action in {"document_identity_updated", "document_identity_confirmed"}
        ]

    assert identity_logs
    assert all(log.user_message == "身份信息已保存，检索元数据待刷新。" for log in identity_logs)
    details = [json.loads(log.details_json or "{}") for log in identity_logs]
    assert all(item["metadata_refreshed"] is False for item in details)
    assert all("Qdrant payload 待刷新" in item["refresh_warning"] for item in details)


def test_manifest_enriches_existing_document_and_rejects_qa_manifest(fake_indexing_services) -> None:
    reset_database()
    content = "技能专长正文。"
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", content, "text/plain")},
    )
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {
        "files": [
            {
                "filename": "rules.txt",
                "sha256": sha256,
                "doc_id": "HENU-M-001",
                "title": "正式说明",
                "source_url": "https://www.henu.edu.cn/materials/official",
            }
        ]
    }
    response = client.post(
        "/api/documents/manifest",
        files={"manifest": ("manifest.json", json.dumps(manifest, ensure_ascii=False), "application/json")},
    )
    assert response.status_code == 200
    assert response.json()["updated_count"] == 1
    detail = client.get(f"/api/documents/{upload.json()['document_id']}").json()
    assert detail["metadata"]["external_doc_id"] == "HENU-M-001"

    rejected = client.post(
        "/api/documents/manifest",
        files={"manifest": ("qa.jsonl", '{"question":"q","answer":"a"}\n', "application/x-ndjson")},
    )
    assert rejected.status_code == 400
    assert "评测数据禁止" in rejected.json()["detail"]


def test_url_import_persists_final_source_url(monkeypatch, fake_indexing_services) -> None:
    from backend.app.services.document_url_import_service import FetchedUrlDocument

    reset_database()
    monkeypatch.setattr(
        "backend.app.api.documents.fetch_url_document",
        lambda *args, **kwargs: FetchedUrlDocument(
            content=b"official resume text",
            filename="official.md",
            content_type="text/markdown",
            final_url="https://www.henu.edu.cn/official.md",
        ),
    )
    response = client.post(
        "/api/documents/url-import",
        json={
            "url": "https://www.henu.edu.cn/official.md",
            "metadata": {"title": "官方说明", "material_topic": "项目经历"},
        },
    )
    assert response.status_code == 202
    assert response.json()["metadata"]["source_url"] == "https://www.henu.edu.cn/official.md"
    assert response.json()["metadata"]["attachment_url"] == "https://www.henu.edu.cn/official.md"


def test_upload_is_idempotent_by_header(fake_indexing_services) -> None:
    reset_database()
    headers = {"Idempotency-Key": "document-upload-idempotency-0001"}
    files = {"file": ("rules.txt", "综合成绩应等于各分项成绩之和。", "text/plain")}

    first = client.post("/api/documents/upload", files=files, headers=headers)
    second = client.post("/api/documents/upload", files=files, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["task_id"] == second.json()["task_id"]


def test_upload_preflight_reports_exact_duplicate_and_upload_rejects_duplicate(fake_indexing_services) -> None:
    reset_database()
    content = "综合成绩应等于各分项成绩之和。"
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", content, "text/plain")},
    )
    assert upload_response.status_code == 202

    file_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    preflight_response = client.post(
        "/api/documents/upload-preflight",
        json={
            "items": [
                {
                    "client_file_id": "duplicate-1",
                    "filename": "renamed-rules.txt",
                    "size": len(content.encode("utf-8")),
                    "file_sha256": file_sha256,
                }
            ]
        },
    )

    assert preflight_response.status_code == 200
    item = preflight_response.json()["items"][0]
    assert item["status"] == "exact_duplicate"
    assert item["existing_document"]["document_id"] == upload_response.json()["document_id"]

    duplicate_response = client.post(
        "/api/documents/upload",
        files={"file": ("renamed-rules.txt", content, "text/plain")},
    )

    assert duplicate_response.status_code == 409
    assert duplicate_response.json()["error"]["code"] == "exact_duplicate"
    assert len(client.get("/api/documents").json()["documents"]) == 1


def test_same_filename_different_content_can_be_renamed_or_overwritten(fake_indexing_services) -> None:
    reset_database()
    first_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "旧表述。", "text/plain")},
    )
    assert first_response.status_code == 202
    first_document_id = first_response.json()["document_id"]
    new_content = "新表述。"

    preflight_response = client.post(
        "/api/documents/upload-preflight",
        json={
            "items": [
                {
                    "client_file_id": "conflict-1",
                    "filename": "rules.txt",
                    "size": len(new_content.encode("utf-8")),
                    "file_sha256": hashlib.sha256(new_content.encode("utf-8")).hexdigest(),
                }
            ]
        },
    )
    assert preflight_response.status_code == 200
    item = preflight_response.json()["items"][0]
    assert item["status"] == "name_conflict"
    assert item["existing_document"]["document_id"] == first_document_id

    conflict_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", new_content, "text/plain")},
    )
    assert conflict_response.status_code == 409
    assert conflict_response.json()["error"]["code"] == "name_conflict"

    renamed_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", new_content, "text/plain")},
        data={"filename_override": "rules (1).txt"},
    )
    assert renamed_response.status_code == 202
    documents = client.get("/api/documents").json()["documents"]
    assert {document["filename"] for document in documents} == {"rules.txt", "rules (1).txt"}

    overwrite_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "覆盖后的表述。", "text/plain")},
        data={"overwrite_document_id": first_document_id},
    )
    assert overwrite_response.status_code == 202
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 2
    assert first_document_id not in {document["document_id"] for document in documents}
    assert "rules.txt" in {document["filename"] for document in documents}


def test_batch_upload_reports_duplicate_and_conflict_without_rolling_back_ready_file(fake_indexing_services) -> None:
    reset_database()
    existing_content = "已有表述。"
    existing_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", existing_content, "text/plain")},
    )
    assert existing_response.status_code == 202

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("ready.txt", "可入库表述。", "text/plain")),
            ("files", ("duplicate.txt", existing_content, "text/plain")),
            ("files", ("rules.txt", "同名新表述。", "text/plain")),
        ],
        headers={"Idempotency-Key": "batch-upload-duplicate-conflict"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted_count"] == 1
    assert body["failed_count"] == 2
    assert [item["status"] for item in body["items"]] == ["accepted", "duplicate", "conflict"]
    assert len(client.get("/api/documents").json()["documents"]) == 2


def test_upload_preflight_reports_selection_name_conflict() -> None:
    reset_database()
    response = client.post(
        "/api/documents/upload-preflight",
        json={
            "items": [
                {
                    "client_file_id": "a",
                    "filename": "Rules.TXT",
                    "size": 1,
                    "file_sha256": hashlib.sha256(b"a").hexdigest(),
                },
                {
                    "client_file_id": "b",
                    "filename": "rules.txt",
                    "size": 1,
                    "file_sha256": hashlib.sha256(b"b").hexdigest(),
                },
            ]
        },
    )

    assert response.status_code == 200
    assert [item["status"] for item in response.json()["items"]] == [
        "selection_name_conflict",
        "selection_name_conflict",
    ]


def test_batch_upload_accepts_multiple_documents(fake_indexing_services) -> None:
    reset_database()

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("rules-a.txt", "综合成绩应等于各分项成绩之和。", "text/plain")),
            ("files", ("rules-b.md", "# 机试成绩\n笔试成绩应等于各分项成绩之和。", "text/markdown")),
        ],
        headers={"Idempotency-Key": "batch-upload-accepts-multiple"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["batch_id"] == "batch-upload-accepts-multiple"
    assert body["accepted_count"] == 2
    assert body["failed_count"] == 0
    assert [item["status"] for item in body["items"]] == ["accepted", "accepted"]
    assert all(item["document_id"].startswith("DOC-") for item in body["items"])
    assert len(fake_indexing_services) == 2
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 2
    assert {document["status"] for document in documents} == {"indexed"}


def test_batch_upload_keeps_partial_success_and_cleans_failed_file(fake_indexing_services, tmp_path) -> None:
    reset_database()

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("valid.txt", "课程成绩表述应以材料为准。", "text/plain")),
            ("files", ("empty.txt", b"", "text/plain")),
        ],
        headers={"Idempotency-Key": "batch-upload-partial-success"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted_count"] == 1
    assert body["failed_count"] == 1
    assert body["items"][0]["status"] == "accepted"
    assert body["items"][1]["status"] == "failed"
    assert body["items"][1]["document_id"] is None
    assert body["items"][1]["error_message"] == "上传文档不能为空"
    assert len(fake_indexing_services) == 1
    stored_files = list((tmp_path / "documents" / "originals").glob("*"))
    assert len(stored_files) == 1
    assert not list((tmp_path / "documents" / "originals").glob("*.upload"))


def test_batch_upload_rejects_too_many_files(monkeypatch) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents.MAX_BATCH_UPLOAD_FILES", 1)

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("a.txt", "A", "text/plain")),
            ("files", ("b.txt", "B", "text/plain")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "单次批量上传最多支持 1 个文件"
    assert client.get("/api/documents").json()["documents"] == []


def test_batch_upload_is_idempotent_by_header(fake_indexing_services) -> None:
    reset_database()
    headers = {"Idempotency-Key": "batch-upload-idempotency-0001"}
    files = [
        ("files", ("rules-a.txt", "综合成绩应等于各分项成绩之和。", "text/plain")),
        ("files", ("rules-b.txt", "笔试成绩应等于各分项成绩之和。", "text/plain")),
    ]

    first = client.post("/api/documents/batch-upload", files=files, headers=headers)
    second = client.post("/api/documents/batch-upload", files=files, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 202
    first_body = first.json()
    second_body = second.json()
    assert first_body["accepted_count"] == 2
    assert second_body["accepted_count"] == 2
    assert [item["document_id"] for item in first_body["items"]] == [
        item["document_id"] for item in second_body["items"]
    ]
    assert [item["task_id"] for item in first_body["items"]] == [
        item["task_id"] for item in second_body["items"]
    ]
    assert len(fake_indexing_services) == 2
    assert len(client.get("/api/documents").json()["documents"]) == 2


def test_next_document_id_is_collision_resistant_and_keeps_date_prefix() -> None:
    reset_database()
    today = datetime.now().strftime("%Y%m%d")
    with SessionLocal() as db:
        db.add(
            Document(
                document_id=f"DOC-{today}-0001",
                filename="existing.pdf",
                content_type="application/pdf",
                file_type="pdf",
                size=100,
                storage_path="existing.pdf",
                status="uploaded",
                chunk_count=0,
            )
        )
        db.commit()

        first = next_document_id(db)
        second = next_document_id(db)
        assert re.fullmatch(rf"DOC-{today}-[0-9A-F]{{12}}", first)
        assert re.fullmatch(rf"DOC-{today}-[0-9A-F]{{12}}", second)
        assert first != second


def test_sqlite_upgrade_backfills_duplicate_filename_norms_and_unique_index(monkeypatch, tmp_path) -> None:
    from backend.app.core import database as database_module

    legacy_engine = create_engine(
        f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    with legacy_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE documents ("
                "id INTEGER PRIMARY KEY, "
                "document_id VARCHAR(32), "
                "filename VARCHAR(255), "
                "content_type VARCHAR(100), "
                "file_type VARCHAR(20), "
                "size INTEGER, "
                "storage_path VARCHAR(500), "
                "status VARCHAR(30), "
                "chunk_count INTEGER, "
                "uploaded_at DATETIME"
                ")"
            )
        )
        connection.execute(text("CREATE TABLE document_chunks (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "INSERT INTO documents "
                "(id, document_id, filename, content_type, file_type, size, storage_path, status, chunk_count, uploaded_at) "
                "VALUES "
                "(1, 'DOC-OLD-1', 'Rules.TXT', 'text/plain', 'txt', 1, 'a.txt', 'indexed', 0, '2026-07-18 00:00:00'), "
                "(2, 'DOC-OLD-2', 'rules.txt', 'text/plain', 'txt', 2, 'b.txt', 'indexed', 0, '2026-07-18 00:01:00')"
            )
        )

    monkeypatch.setattr(database_module, "engine", legacy_engine)

    database_module._upgrade_sqlite_schema()

    with legacy_engine.connect() as connection:
        rows = connection.execute(
            text("SELECT filename, filename_norm FROM documents ORDER BY id")
        ).all()
        indexes = connection.execute(text("PRAGMA index_list('documents')")).mappings().all()

    assert rows[0] == ("Rules.TXT", "rules.txt")
    assert rows[1] == ("rules (历史重复-2).txt", "rules (历史重复-2).txt")
    assert any(index["name"] == "ix_documents_filename_norm" and index["unique"] for index in indexes)


def test_index_task_is_durable_when_memory_queue_is_full(monkeypatch) -> None:
    reset_database()
    document_id = "DOC-QUEUE-0001"
    with SessionLocal() as db:
        db.add(
            Document(
                document_id=document_id,
                filename="queued.txt",
                content_type="text/plain",
                file_type="txt",
                size=10,
                storage_path="queued.txt",
                status="uploaded",
                chunk_count=1,
            )
        )
        db.commit()

    monkeypatch.setattr("backend.app.services.index_task_service.start_index_task_worker", lambda: None)
    monkeypatch.setattr("backend.app.services.index_task_service._schedule_in_memory", lambda value: False)
    from backend.app.services.index_task_service import enqueue_document_index

    assert enqueue_document_index(document_id) is False
    with SessionLocal() as db:
        task = db.scalar(
            select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
        )
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert task is not None
        assert task.status == "queued"
        assert document is not None
        assert document.status == "index_queued"


def test_running_index_task_is_requeued_after_restart() -> None:
    reset_database()
    document_id = "DOC-QUEUE-0002"
    with SessionLocal() as db:
        db.add(
            Document(
                document_id=document_id,
                filename="recover.txt",
                content_type="text/plain",
                file_type="txt",
                size=10,
                storage_path="recover.txt",
                status="indexing",
                chunk_count=1,
            )
        )
        db.add(DocumentIndexTask(document_id=document_id, status="running"))
        db.commit()

    from backend.app.services.index_task_service import _recover_interrupted_tasks

    _recover_interrupted_tasks()
    with SessionLocal() as db:
        task = db.scalar(
            select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
        )
        assert task is not None
        assert task.status == "queued"


def test_build_document_index_persists_bounded_queue_task(monkeypatch) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.services.index_task_service.start_index_task_worker", lambda: None)
    monkeypatch.setattr("backend.app.services.index_task_service._schedule_in_memory", lambda value: False)
    from backend.app.services.index_task_service import enqueue_document_index
    monkeypatch.setattr("backend.app.api.documents.enqueue_document_index", enqueue_document_index)

    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.post(f"/api/documents/{document_id}/index")

    assert response.status_code == 200
    assert response.json()["status"] == "index_queued"
    with SessionLocal() as db:
        task = db.scalar(select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id))
        assert task is not None
        assert task.status == "queued"


def test_index_document_reloads_chunks_after_stage_commits(fake_indexing_services) -> None:
    reset_database()
    with SessionLocal() as db:
        document = Document(
            document_id="DOC-TEST-RELOAD",
            filename="reload.md",
            file_type="md",
            size=128,
            storage_path="reload.md",
            status="uploaded",
            chunk_count=1,
        )
        chunk = DocumentChunk(
            document_id=document.document_id,
            chunk_id="DOC-TEST-RELOAD-CHUNK-0001",
            text="综合成绩应当等于数据结构、操作系统、计算机网络和数据库四门科目成绩之和。",
            embedding_text="综合成绩应当等于数据结构、操作系统、计算机网络和数据库四门科目成绩之和。",
            token_count=20,
            index_status="uploaded",
            index_version=INDEX_VERSION,
            source_file=document.filename,
        )
        db.add_all([document, chunk])
        db.commit()

        from backend.app.services.document_indexing_service import index_document

        def stage_reporter(stage: str, completed: int | None = None, total: int | None = None) -> None:
            db.commit()

        index_document(db, document, stage_reporter=stage_reporter)

        indexed_chunk = db.scalar(
            select(DocumentChunk).where(DocumentChunk.chunk_id == "DOC-TEST-RELOAD-CHUNK-0001")
        )
        assert indexed_chunk is not None
        assert indexed_chunk.index_status == "indexed"
        assert document.status == "indexed"


def test_queued_index_failure_is_recorded_for_retry(monkeypatch) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.services.index_task_service.start_index_task_worker", lambda: None)
    monkeypatch.setattr("backend.app.services.index_task_service._schedule_in_memory", lambda value: False)
    from backend.app.services.index_task_service import _run_task, enqueue_document_index
    monkeypatch.setattr("backend.app.api.documents.enqueue_document_index", enqueue_document_index)

    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    def failing_embed_texts(texts: list[str]) -> list[TextEmbedding]:
        raise EmbeddingServiceError("embedding service unavailable")

    monkeypatch.setattr("backend.app.services.document_indexing_service.embed_texts", failing_embed_texts)

    response = client.post(f"/api/documents/{document_id}/index")

    assert response.status_code == 200
    assert _run_task(document_id) is True
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 1
    assert documents[0]["status"] == "index_failed"
    assert documents[0]["index_error"] == "embedding service unavailable"
    with SessionLocal() as db:
        task = db.scalar(select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id))
        assert task is not None
        assert task.status == "queued"
        assert task.retry_count == 1


def test_upload_empty_document_returns_400() -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "上传文档不能为空"


def test_upload_stream_rejects_limit_and_removes_temporary_file(monkeypatch, tmp_path) -> None:
    reset_database()
    monkeypatch.setattr("backend.app.api.documents.MAX_UPLOAD_BYTES", 8)

    response = client.post(
        "/api/documents/upload",
        files={"file": ("too-large.txt", b"123456789", "text/plain")},
    )

    assert response.status_code == 413
    assert list((tmp_path / "documents" / "originals").glob("*")) == []


def test_upload_rejects_pdf_extension_with_executable_content() -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={"file": ("a.pdf", b"MZ fake executable", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "PDF 文件内容校验失败"


def test_upload_rejects_mime_type_mismatch() -> None:
    reset_database()

    response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.pdf", b"%PDF-1.4\n", "application/x-msdownload")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "文件 MIME 类型与扩展名不匹配：application/x-msdownload"


def test_list_and_get_document_detail() -> None:
    reset_database()
    document_text = (
        "综合成绩应等于各科目分项成绩之和。\n"
        "数据质量要求包括字段完整。"
        + ("This sentence makes the chunk detail response longer than the preview limit. " * 3)
        + "完整文本应保留到最后一句。"
    )
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "rules.txt",
                document_text,
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    list_response = client.get("/api/documents")
    detail_response = client.get(f"/api/documents/{document_id}")

    assert list_response.status_code == 200
    assert list_response.json()["documents"][0]["document_id"] == document_id
    assert detail_response.status_code == 200
    chunk = detail_response.json()["chunks"][0]
    assert chunk["chunk_id"].startswith(document_id)
    assert chunk["text"] == document_text
    assert chunk["text_preview"] == document_text[:120]
    assert chunk["is_truncated"] is True
    assert chunk["chunk_type"] == "paragraph"
    assert "\n" in chunk["text"]
    assert detail_response.json()["chunk_total"] == 1
    assert detail_response.json()["chunk_offset"] == 0
    assert detail_response.json()["chunk_limit"] == 50


def test_document_detail_is_paginated_for_large_documents() -> None:
    reset_database()
    document_id = "DOC-LARGE-0001"
    with SessionLocal() as db:
        db.add(
            Document(
                document_id=document_id,
                filename="large.xls",
                content_type=None,
                file_type="xls",
                size=100,
                storage_path="large.xls",
                status="indexed",
                chunk_count=125,
            )
        )
        for index in range(125):
            db.add(
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-{index + 1:04d}",
                    document_id=document_id,
                    text=f"chunk {index + 1}",
                    token_count=2,
                    index_status="indexed",
                    source_file="large.xls",
                )
            )
        db.commit()

    response = client.get(f"/api/documents/{document_id}?chunk_offset=50&chunk_limit=50")

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_total"] == 125
    assert body["chunk_offset"] == 50
    assert body["chunk_limit"] == 50
    assert len(body["chunks"]) == 50
    assert body["chunks"][0]["text"] == "chunk 51"


def test_delete_document_removes_document_from_list() -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert client.get("/api/documents").json()["documents"] == []


def test_bulk_delete_documents_reports_partial_results() -> None:
    reset_database()
    first_upload = client.post(
        "/api/documents/upload",
        files={"file": ("rules-a.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    second_upload = client.post(
        "/api/documents/upload",
        files={"file": ("rules-b.txt", "课程成绩不得为负。", "text/plain")},
    )
    first_id = first_upload.json()["document_id"]
    second_id = second_upload.json()["document_id"]
    with SessionLocal() as db:
        second = db.scalar(select(Document).where(Document.document_id == second_id))
        assert second is not None
        second.status = "indexing"
        db.commit()

    response = client.post(
        "/api/documents/bulk-delete",
        json={"document_ids": [first_id, second_id, "DOC-NOT-FOUND"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requested_count"] == 3
    assert body["deleted_count"] == 1
    assert body["failed_count"] == 2
    statuses = {item["document_id"]: item["status"] for item in body["items"]}
    assert statuses[first_id] == "deleted"
    assert statuses[second_id] == "blocked"
    assert statuses["DOC-NOT-FOUND"] == "not_found"
    remaining_ids = {item["document_id"] for item in client.get("/api/documents").json()["documents"]}
    assert first_id not in remaining_ids
    assert second_id in remaining_ids


def test_delete_document_keeps_record_when_qdrant_cleanup_fails(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    def failing_delete_vectors(document_id: str, chunk_ids: list[str] | None = None) -> None:
        raise VectorStoreError("qdrant unavailable")

    monkeypatch.setattr("backend.app.services.document_lifecycle_service.delete_document_vectors", failing_delete_vectors)

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 503
    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 1
    assert documents[0]["document_id"] == document_id
    assert documents[0]["status"] == "delete_failed"
    assert "Qdrant" in documents[0]["index_error"]

    monkeypatch.setattr(
        "backend.app.services.document_lifecycle_service.delete_document_vectors",
        lambda document_id, chunk_ids=None: None,
    )
    retry_response = client.delete(f"/api/documents/{document_id}")

    assert retry_response.status_code == 200
    assert client.get("/api/documents").json()["documents"] == []


def test_delete_document_records_file_cleanup_failure_for_retry(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        target_path = Path(document.storage_path)

    original_unlink = Path.unlink

    def failing_unlink(path: Path, *args, **kwargs):
        if path == target_path:
            raise PermissionError("file is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 503
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        assert document.status == "delete_failed"
        assert document.lifecycle_stage == "deleting_file"
        assert "原文件删除失败" in (document.index_error or "")


def test_list_documents_marks_record_when_source_repair_is_requested() -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        storage_path = document.storage_path

    import os

    os.remove(storage_path)

    response = client.get("/api/documents?repair_sources=true")

    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 1
    assert documents[0]["document_id"] == document_id
    assert documents[0]["status"] == "source_missing"


def test_list_documents_restores_windows_storage_path_marked_source_missing(tmp_path, monkeypatch) -> None:
    reset_database()
    document_dir = tmp_path / "documents" / "originals"
    document_dir.mkdir(parents=True, exist_ok=True)
    stored = document_dir / "DOC-WINDOWS-0001.xls"
    stored.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "backend.app.services.document_lifecycle_service.document_vector_chunk_ids",
        lambda document_id: {"DOC-WINDOWS-0001-CHUNK-0001"},
    )

    with SessionLocal() as db:
        db.add(
            Document(
                document_id="DOC-WINDOWS-0001",
                filename="395_课程成绩统计表.xls",
                content_type=None,
                file_type="xls",
                size=100,
                storage_path=r"D:\Agent-Project\Resume-Agent\data\documents\originals\DOC-WINDOWS-0001.xls",
                status="source_missing",
                index_version=INDEX_VERSION,
                index_error="原始文件缺失",
                chunk_count=1,
            )
        )
        db.add(
            DocumentChunk(
                chunk_id="DOC-WINDOWS-0001-CHUNK-0001",
                document_id="DOC-WINDOWS-0001",
                text="chunk",
                token_count=1,
                index_status="source_missing",
                source_file="395_课程成绩统计表.xls",
            )
        )
        db.commit()

    response = client.get("/api/documents")

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["status"] == "indexed"
    assert document["index_error"] is None
    with SessionLocal() as db:
        chunk = db.scalar(select(DocumentChunk).where(DocumentChunk.document_id == "DOC-WINDOWS-0001"))
        assert chunk is not None
        assert chunk.index_status == "indexed"


def test_qa_returns_context_package_when_knowledge_matches(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "rules.txt",
                "综合成绩应等于各科目分项成绩之和。\n简历数据应保证字段完整、信息真实。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]

    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [fake_retrieval_match(document_id)],
    )

    response = client.post("/api/qa/ask", json={"question": "综合成绩和分项成绩怎么计算", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["answer_mode"] == "hedged"  # LLM 关闭 → 摘录兜底，按推测标注
    assert body["evidence_sufficiency"] == "partial"
    assert body["intent"] == "resume_qa"
    assert "综合成绩" in body["answer"]
    assert body["context_package"]["is_final_answer"] is False
    assert body["context_package"]["mode"] == "rag_context"
    assert body["context_package"]["query"] == "综合成绩和分项成绩怎么计算"
    assert body["context_package"]["context_chunks"][0]["citation_label"] == "[1]"
    assert body["context_package"]["context_chunks"][0]["chunk_id"] == f"{document_id}-CHUNK-0001"
    assert "综合成绩" in body["context_package"]["context_chunks"][0]["text"]
    assert "不得编造" in body["context_package"]["llm_prompt"]
    assert "综合成绩和分项成绩怎么计算" in body["context_package"]["llm_prompt"]
    assert "根据知识库引用，可归纳为" not in body["context_package"]["llm_prompt"]


def test_qa_stream_returns_progress_events_and_final_response(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "rules.txt",
                "综合成绩应等于各科目分项成绩之和。\n简历数据应保证字段完整、信息真实。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [fake_retrieval_match(document_id)],
    )

    response = client.post("/api/qa/ask/stream", json={"question": "综合成绩和分项成绩怎么计算", "include_debug": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse_events(response.text)
    event_names = [event_name for event_name, _payload in events]
    progress_stages = [
        payload["stage"]
        for event_name, payload in events
        if event_name == "progress"
    ]
    assert event_names[-1] == "final"
    assert "intent" in progress_stages
    assert "memory" in progress_stages
    assert "retrieval" in progress_stages
    assert "generation" in progress_stages
    retrieval_event = latest_progress_event(events, "retrieval", aspect=False)
    generation_event = latest_progress_event(events, "generation", aspect=False)
    assert retrieval_event is not None
    assert retrieval_event["status"] == "completed"
    assert generation_event is not None
    assert generation_event["status"] == "completed"
    assert any(
        payload["stage"] == "intent" and payload["status"] == "completed"
        for event_name, payload in events
        if event_name == "progress"
    )
    assert any(
        payload["stage"] == "memory" and payload["status"] == "completed"
        for event_name, payload in events
        if event_name == "progress"
    )
    final_payload = events[-1][1]
    assert final_payload["answer"]
    assert final_payload["answer_mode"] == "hedged"
    assert final_payload["context_package"]["query"] == "综合成绩和分项成绩怎么计算"
    assert final_payload["context_package"]["is_final_answer"] is False


def test_qa_stream_marks_empty_retrieval_and_failed_generation_as_handled(monkeypatch) -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [])

    response = client.post("/api/qa/ask/stream", json={"question": "火星基地如何审批", "include_debug": True})

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    retrieval_event = latest_progress_event(events, "retrieval", aspect=False)
    generation_event = latest_progress_event(events, "generation", aspect=False)
    assert retrieval_event is not None
    assert retrieval_event["status"] == "completed"
    assert generation_event is not None
    assert generation_event["status"] == "completed"
    assert "回答模式：failed" in generation_event["detail"]
    final_payload = events[-1][1]
    assert final_payload["answer_mode"] == "failed"
    assert final_payload["generation_status"] == "skipped"
    assert "知识库中还没有相关记录" in final_payload["answer"]


def test_qa_stream_handles_consecutive_questions(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [fake_retrieval_match(document_id)],
    )

    first_response = client.post("/api/qa/ask/stream", json={"question": "综合成绩怎么计算"})
    second_response = client.post("/api/qa/ask/stream", json={"question": "字段完整性如何检查"})

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first_events = parse_sse_events(first_response.text)
    second_events = parse_sse_events(second_response.text)
    assert first_events[-1][0] == "final"
    assert second_events[-1][0] == "final"
    assert latest_progress_event(first_events, "retrieval", aspect=False)["status"] == "completed"
    assert latest_progress_event(second_events, "retrieval", aspect=False)["status"] == "completed"


def test_qa_stream_returns_error_event_when_retrieval_fails(monkeypatch) -> None:
    reset_database()

    def failing_retrieve(question: str, progress_reporter=None):
        raise RetrievalServiceUnavailable("Qdrant hybrid 检索失败")

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", failing_retrieve)

    response = client.post("/api/qa/ask/stream", json={"question": "综合成绩"})

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert any(event_name == "error" for event_name, _payload in events)
    assert any(
        event_name == "progress" and payload["status"] == "failed"
        for event_name, payload in events
    )
    assert all(event_name != "final" for event_name, _payload in events)


def test_qa_context_package_keeps_full_evidence_block(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "project_material.txt",
                "4. 项目经历与竞赛奖项 项目经历统计以项目完成日期为基础。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    evidence = (
        "4. 项目经历与竞赛奖项 项目经历统计以项目完成日期为基础。"
        "项目开始时间超过计划完成日期仍未完成的，应根据延期情况进入项目经历统计。"
        "项目经历强调完成状态，竞赛奖项强调获奖等级，两者不能直接等同。"
        "若一个项目延期 10 天，但项目质量正常、成果完整且交付明确，本模拟材料不要求仅因延期 10 天就直接取消项目经历。"
        "若延期时间较长、质量明显下降或成果价值明显不足，应结合竞赛奖项规则重新判断。"
    )
    match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="project_material.txt",
            section_title="项目经历与竞赛奖项",
            page_number=None,
            excerpt=evidence,
            score=0.8,
            rerank_score=0.95,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.95,
        coverage_score=0.9,
        evidence_role="direct_evidence",
        evidence_text=evidence,
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [match])

    response = client.post("/api/qa/ask", json={"question": "项目经历与竞赛奖项", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    package = body["context_package"]
    assert package["is_final_answer"] is False
    assert package["retrieval_summary"]["used_chunks"] == 1
    assert package["context_chunks"][0]["source_doc"] == "project_material.txt"
    assert "项目开始时间超过计划完成日期仍未完成" in package["context_chunks"][0]["text"]
    assert "两者不能直接等同" in package["llm_prompt"]
    assert "延期 10 天" in package["llm_prompt"]
    assert "重新判断" in package["llm_prompt"]
    assert "1. 结论" not in json.dumps(body, ensure_ascii=False)


def test_qa_retrieve_returns_llm_context_package_and_cleans_repeated_title(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "complex_resume_sample.md",
                "2. 课程成绩统计口径\n课程成绩统计应同时满足科目范围、成绩类型和评分表述三类条件。",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="complex_resume_sample.md",
            section_title="2. 课程成绩统计口径",
            page_number=None,
            excerpt=(
                "2. 课程成绩统计口径\n"
                "课程成绩统计应同时满足科目范围、成绩类型和评分表述三类条件。"
            ),
            score=0.8,
            rerank_score=0.87,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.87,
        coverage_score=1.0,
        evidence_role="direct_evidence",
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [match, match])

    response = client.post("/api/qa/retrieve", json={"question": "哪些科目成绩可以纳入综合成绩统计，哪些不能？"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_final_answer"] is False
    assert body["retrieval_summary"]["used_chunks"] == 1
    assert body["retrieval_summary"]["query_count"] == 0
    assert body["retrieval_summary"]["candidate_count"] == 0
    assert body["retrieval_summary"]["filtered_count"] == 0
    assert body["retrieval_summary"]["citation_validation"]["invalid_chunks"] == 0
    assert body["context_chunks"][0]["section_title"] == "2. 课程成绩统计口径"
    assert body["context_chunks"][0]["text"].startswith("课程成绩统计应同时满足")
    assert body["context_chunks"][0]["citation_label"] == "[1]"
    assert "【知识片段 1】来源：complex_resume_sample.md / 2. 课程成绩统计口径" in body["llm_prompt"]
    assert "哪些科目成绩可以纳入综合成绩统计，哪些不能？" in body["llm_prompt"]
    assert "不得编造" in body["llm_prompt"]
    assert "不能纳入、暂不纳入、需补充材料" not in body["llm_prompt"]
    assert "根据知识库引用，可归纳为" not in json.dumps(body, ensure_ascii=False)


def test_qa_retrieve_no_longer_expands_neighbor_child_section(monkeypatch, fake_indexing_services) -> None:
    # 邻块扩展（A6b）已删除：命中父章节不再自动补入子章节 chunk
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "complex_resume_sample.md",
                "## 3. 综合成绩与校验关系\n"
                "综合成绩应等于各科目分项成绩之和。若综合成绩与分项成绩存在差异，"
                "应优先检查加权折算、四舍五入、科目映射和重复汇总问题。\n\n"
                "### 3.1 综合成绩差异处理\n"
                "当综合成绩差异小于等于系统允许的尾差阈值时，可以记录尾差说明；"
                "当差异超过尾差阈值时，应退回录入人员核对分项数据。"
                "若差异来自加权折算，应保留权重日期、折算规则和原分成绩来源。\n",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    index_response = client.post(f"/api/documents/{document_id}/index")
    assert index_response.status_code == 200
    parent_chunk_id = f"{document_id}-CHUNK-0001"
    parent_match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=parent_chunk_id,
            filename="complex_resume_sample.md",
            section_title="3. 综合成绩与校验关系",
            section_number="3",
            next_chunk_id=f"{document_id}-CHUNK-0002",
            page_number=None,
            excerpt=(
                "3. 综合成绩与校验关系\n"
                "综合成绩应等于各科目分项成绩之和。若综合成绩与分项成绩存在差异，"
                "应优先检查加权折算、四舍五入、科目映射和重复汇总问题。"
            ),
            score=0.8,
            rerank_score=0.92,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.92,
        coverage_score=0.8,
        evidence_role="direct_evidence",
    )
    retrieval_calls = []

    def fake_retrieve(question: str):
        retrieval_calls.append(question)
        return [parent_match]

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", fake_retrieve)

    response = client.post(
        "/api/qa/retrieve",
        json={"question": "综合成绩差异应该优先排查哪些问题？如果差异来自加权折算，需要保留什么依据？"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_final_answer"] is False
    # 邻块扩展已删除：aspect_2（"加权折算保留依据"）的内容仅在 3.1 子章节，
    # 不再被自动补入 → 该 aspect 有候选但 prompt 终检未覆盖，
    # has_sufficient_context=False（触发兜底链，符合预期）
    assert body["retrieval_summary"]["has_sufficient_context"] is False
    assert body["retrieval_summary"]["missing_aspects"] == []
    assert [
        aspect["aspect_id"]
        for aspect in body["retrieval_summary"]["query_plan"]["aspects"]
    ] == ["aspect_1", "aspect_2"]
    assert body["retrieval_summary"]["fusion_method"] == "aspect_query_rrf_then_bge_rerank"
    query_plan_aspects = body["retrieval_summary"]["query_plan"]["aspects"]
    assert query_plan_aspects[0]["search_queries"][0]["query_type"] == "semantic_question"
    assert query_plan_aspects[0]["search_queries"][1]["query_type"] == "keyword_anchor"
    assert any("综合成绩差异" in query and "排查" in query for query in retrieval_calls)
    assert any("加权折算" in query and "依据" in query for query in retrieval_calls)
    assert len(retrieval_calls) >= 4
    assert body["retrieval_summary"]["aspect_retrievals"][0]["covered"] is True
    assert body["retrieval_summary"]["aspect_retrievals"][1]["covered"] is False
    assert all(
        diagnostic["query_type"] in {"semantic_question", "document_style_statement", "keyword_anchor"}
        for aspect in body["retrieval_summary"]["aspect_retrievals"]
        for diagnostic in aspect["diagnostics"]
    )
    section_titles = [chunk["section_title"] for chunk in body["context_chunks"]]
    assert "3. 综合成绩与校验关系" in section_titles
    assert "3.1 综合成绩差异处理" not in section_titles
    assert "加权折算、四舍五入、科目映射和重复汇总" in body["llm_prompt"]
    assert "权重日期、折算规则和原分成绩来源" not in body["llm_prompt"]


def test_qa_retrieve_does_not_force_unrelated_prompt_chunk(monkeypatch, fake_indexing_services) -> None:
    # 邻块扩展（A6b）已删除：不再自动补入子章节，无关章节仍被 prompt 终检过滤

    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "complex_resume_sample.md",
                "## 3. 综合成绩与校验关系\n"
                "综合成绩应等于各科目分项成绩之和。若综合成绩与分项成绩存在差异，"
                "应优先检查加权折算、四舍五入、科目映射和重复汇总问题。\n\n"
                "### 3.1 综合成绩差异处理\n"
                "当综合成绩差异小于等于系统允许的尾差阈值时，可以记录尾差说明；"
                "若差异来自加权折算，应保留权重日期、折算规则和原分成绩来源。\n\n"
                "## 4. 项目经历与竞赛奖项\n"
                "项目经历统计以项目完成日期为基础，竞赛奖项强调获奖等级和成果质量。\n",
                "text/markdown",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    index_response = client.post(f"/api/documents/{document_id}/index")
    assert index_response.status_code == 200
    parent_match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="complex_resume_sample.md",
            section_title="3. 综合成绩与校验关系",
            section_number="3",
            next_chunk_id=f"{document_id}-CHUNK-0002",
            page_number=None,
            excerpt=(
                "3. 综合成绩与校验关系\n"
                "综合成绩应等于各科目分项成绩之和。若综合成绩与分项成绩存在差异，"
                "应优先检查加权折算、四舍五入、科目映射和重复汇总问题。"
            ),
            score=0.8,
            rerank_score=0.92,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.8,
        rerank_score=0.92,
        coverage_score=0.8,
        evidence_role="direct_evidence",
    )
    unrelated_match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0003",
            filename="complex_resume_sample.md",
            section_title="4. 项目经历与竞赛奖项",
            section_number="4",
            previous_chunk_id=f"{document_id}-CHUNK-0002",
            page_number=None,
            excerpt="4. 项目经历与竞赛奖项\n项目经历统计以项目完成日期为基础，竞赛奖项强调获奖等级和成果质量。",
            score=0.78,
            rerank_score=0.86,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.78,
        rerank_score=0.86,
        coverage_score=0.4,
        evidence_role="direct_evidence",
    )
    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [parent_match, unrelated_match],
    )

    response = client.post(
        "/api/qa/retrieve",
        json={"question": "综合成绩差异应该优先排查哪些问题？如果差异来自加权折算，需要保留什么依据？"},
    )

    assert response.status_code == 200
    body = response.json()
    section_titles = [chunk["section_title"] for chunk in body["context_chunks"]]
    assert section_titles == ["3. 综合成绩与校验关系"]
    assert body["retrieval_summary"]["used_chunks"] == 1
    assert body["retrieval_summary"]["top_k"] == 12
    assert body["retrieval_summary"]["prompt_filtered_count"] == 1
    assert body["retrieval_summary"]["prompt_selection"]["final_prompt_chunks"] == 1
    assert "4. 项目经历与竞赛奖项" not in body["llm_prompt"]


def test_qa_retrieve_marks_missing_context_aspect(monkeypatch) -> None:
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "complex_resume_sample.md",
                "3. 综合成绩与校验关系\n综合成绩应等于各科目分项成绩之和。若综合成绩与分项成绩存在差异，应优先检查加权折算、四舍五入、科目映射和重复汇总问题。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        document.status = "indexed"
        db.commit()
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [])

    response = client.post(
        "/api/qa/retrieve",
        json={"question": "综合成绩差异应该优先排查哪些问题？如果差异来自加权折算，需要保留什么依据？"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_summary"]["has_sufficient_context"] is False
    assert body["retrieval_summary"]["missing_aspects"] == [
        "未召回：综合成绩差异应该优先排查哪些问题",
        "未召回：差异来自加权折算，需要保留什么依据",
    ]


def test_qa_fails_gracefully_when_no_knowledge_matches(monkeypatch) -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [])

    response = client.post("/api/qa/ask", json={"question": "火星基地如何审批", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["answer_mode"] == "failed"
    assert "知识库中还没有相关记录" in body["answer"]
    assert body["generation_status"] == "skipped"
    assert body["intent"] == "resume_qa"
    assert body["context_package"]["retrieval_summary"]["used_chunks"] == 0
    assert body["context_package"]["retrieval_summary"]["has_sufficient_context"] is False


def test_qa_fails_off_topic_question_with_zero_relevance_candidates(monkeypatch) -> None:
    """Off-topic questions must fail politely, not dump unrelated excerpts.

    A retrieved candidate whose rerank score is far below the absolute bar
    (and has no lexical overlap with the question) used to be selected by the
    best-candidate "core" pass, marked as sufficient context, and later
    surfaced as the raw-excerpt fallback after the LLM failed grounding
    validation.  The relevance gate must turn this into a graceful failure.
    """
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "resume.txt",
                "连续三年获得河南大学奖学金和河南大学三好学生。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        document.status = "indexed"
        db.commit()

    evidence = "连续三年获得河南大学奖学金和河南大学三好学生。"
    match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="resume.txt",
            section_title="荣誉奖项",
            page_number=None,
            excerpt=evidence,
            score=0.02,
            rerank_score=0.012,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.02,
        rerank_score=0.012,
        coverage_score=0.0,
        evidence_role="direct_evidence",
        evidence_text=evidence,
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [match])

    response = client.post("/api/qa/ask", json={"question": "你爱弹吉他吗？", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["answer_mode"] == "failed"
    assert "知识库中还没有相关记录" in body["answer"]
    assert "教育背景" in body["answer"]
    assert body["context_package"]["retrieval_summary"]["used_chunks"] == 0
    assert body["context_package"]["retrieval_summary"]["has_sufficient_context"] is False


def test_qa_fails_off_topic_question_even_with_deployed_loose_thresholds(monkeypatch) -> None:
    """The relevance gate must hold under the deployed .env threshold values.

    The deployed config tunes RERANK_PROMPT_THRESHOLD down to 0.01 and
    MIN_EVIDENCE_COVERAGE to 0 (legacy 0.001-0.03 score distribution), which
    would let an off-topic candidate scoring 0.012 through a bare threshold.
    MIN_CORE_RERANK_SCORE (default 0.1) must keep the gate effective.
    """
    reset_database()
    upload_response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "resume.txt",
                "连续三年获得河南大学奖学金和河南大学三好学生。",
                "text/plain",
            )
        },
    )
    document_id = upload_response.json()["document_id"]
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        assert document is not None
        document.status = "indexed"
        db.commit()

    monkeypatch.setattr("backend.app.services.rag_service.RERANK_PROMPT_THRESHOLD", 0.01)
    monkeypatch.setattr("backend.app.services.rag_service.MIN_EVIDENCE_COVERAGE", 0)

    evidence = "连续三年获得河南大学奖学金和河南大学三好学生。"
    match = RetrievalMatch(
        citation=Citation(
            document_id=document_id,
            chunk_id=f"{document_id}-CHUNK-0001",
            filename="resume.txt",
            section_title="荣誉奖项",
            page_number=None,
            excerpt=evidence,
            score=0.02,
            rerank_score=0.012,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.02,
        rerank_score=0.012,
        coverage_score=0.0,
        evidence_role="direct_evidence",
        evidence_text=evidence,
    )
    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [match])

    response = client.post("/api/qa/ask", json={"question": "你爱弹吉他吗？", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["answer_mode"] == "failed"
    assert "知识库中还没有相关记录" in body["answer"]
    assert body["context_package"]["retrieval_summary"]["used_chunks"] == 0


def test_qa_passes_raw_question_to_retrieval_without_option_stripping(monkeypatch) -> None:
    """宽松管线不再做选项抽取/选择题澄清：原始问题（含内联选项）原样进入检索。"""
    reset_database()
    captured: dict[str, object] = {}

    def fake_plan_and_retrieve(
        db,
        question,
        progress_reporter=None,
        cancellation_checker=None,
        *,
        rewritten_queries=None,
        memory_context=None,
        budget=None,
        persona_id=None,
    ):
        captured["question"] = question
        captured["memory_context"] = memory_context
        aspect = QueryAspect(
            aspect_id="aspect_1",
            question=question,
            search_queries=(QuerySearchQuery(question, "semantic_question", ""),),
            evidence_need="相关材料依据",
            keywords=(),
        )
        query_plan = QueryPlan(original_question=question, aspects=(aspect,), planner="test")
        return query_plan, []

    monkeypatch.setattr(
        "backend.app.services.rag_service._plan_and_retrieve", fake_plan_and_retrieve
    )

    response = client.post(
        "/api/qa/ask",
        json={
            "question": "关于《材料》，以下哪项正确？ A 第一项事实 B 第二项事实 C 第三项事实",
            "include_debug": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer_mode"] == "failed"
    assert captured["question"] == "关于《材料》，以下哪项正确？ A 第一项事实 B 第二项事实 C 第三项事实"


def test_qa_task_persists_progress_and_final_answer(monkeypatch) -> None:
    reset_database()
    calls: list[str] = []

    def fake_answer_question(db, question, options=None, include_debug=False, progress_reporter=None, **kwargs):
        calls.append(question)
        assert include_debug is True
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "retrieval",
                    "status": "completed",
                    "title": "问题理解完成",
                    "detail": "测试进度",
                    "elapsed_ms": 1.0,
                }
            )
        return QAResponse(
            answer="综合成绩应等于各科目分项成绩之和。",
            answer_mode="answered",
            evidence_sufficiency="sufficient",
            intent="resume_qa",
            generation_status="completed",
        )

    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", fake_answer_question)

    created = client.post(
        "/api/qa/tasks",
        json={
            "question": "综合成绩如何校验？",
            "client_request_id": "qa-task-persistence-0001",
            "include_debug": True,
        },
    )

    assert created.status_code == 200
    task_id = created.json()["task_id"]

    body = None
    for _ in range(20):
        response = client.get(f"/api/qa/tasks/{task_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] == "completed":
            break
        time.sleep(0.05)

    assert body is not None
    assert body["question"] == "综合成绩如何校验？"
    assert body["status"] == "completed"
    assert body["progress_events"][0]["stage"] == "retrieval"
    assert body["answer"]["answer"] == "综合成绩应等于各科目分项成绩之和。"
    assert body["answer"]["answer_mode"] == "answered"
    assert calls == ["综合成绩如何校验？"]


def test_qa_task_stream_snapshot_contains_verified_preview_and_cancel_clears_it(monkeypatch) -> None:
    from backend.app.api.qa import _stream_qa_task_snapshots

    reset_database()
    preview_saved = Event()
    release = Event()

    def previewing_answer_question(
        db,
        question,
        options=None,
        include_debug=False,
        session_id=None,
        progress_reporter=None,
        answer_preview_reporter=None,
        cancellation_checker=None,
        client_ip=None,
    ):
        assert answer_preview_reporter is not None
        answer_preview_reporter(QAAnswerPreview(
            answer="综合成绩应等于各科目分项成绩之和。",
            revision=1,
        ))
        preview_saved.set()
        assert release.wait(timeout=3)
        if cancellation_checker is not None:
            cancellation_checker()
        return QAResponse(
            answer="综合成绩应等于各科目分项成绩之和。",
            answer_mode="answered",
            evidence_sufficiency="sufficient",
            intent="resume_qa",
            generation_status="completed",
        )

    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", previewing_answer_question)
    created = client.post("/api/qa/tasks", json={
        "question": "综合成绩如何校验？",
        "client_request_id": "qa-task-preview-0001",
        "include_debug": False,
    })
    assert created.status_code == 200
    task_id = created.json()["task_id"]
    assert preview_saved.wait(timeout=3)

    status = client.get(f"/api/qa/tasks/{task_id}").json()
    assert status["answer_preview"]["revision"] == 1
    assert status["answer_preview"]["answer"] == "综合成绩应等于各科目分项成绩之和。"
    assert status["answer"] is None

    stream = _stream_qa_task_snapshots(task_id)
    first_frame = next(stream)
    assert "event: task" in first_frame
    assert '"answer_preview"' in first_frame
    assert '"revision": 1' in first_frame
    stream.close()

    cancelled = client.post(f"/api/qa/tasks/{task_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["answer_preview"] is None
    release.set()


def test_running_qa_task_can_be_cancelled_without_saving_late_answer(monkeypatch) -> None:
    reset_database()
    started = Event()
    release = Event()

    def slow_answer_question(db, question, options=None, include_debug=False, progress_reporter=None, **kwargs):
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "generation",
                    "status": "running",
                    "title": "正在生成回答",
                    "detail": "测试中的慢任务",
                }
            )
        started.set()
        assert release.wait(timeout=3)
        return QAResponse(
            answer="这条迟到的回答不应被保存。",
            answer_mode="answered",
            evidence_sufficiency="sufficient",
            intent="resume_qa",
            generation_status="completed",
        )

    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", slow_answer_question)
    created = client.post(
        "/api/qa/tasks",
        json={
            "question": "请生成一个可以被停止的回答",
            "client_request_id": "qa-task-cancel-0001",
            "include_debug": False,
        },
    )
    assert created.status_code == 200
    task_id = created.json()["task_id"]
    assert started.wait(timeout=3)

    cancelled = client.post(f"/api/qa/tasks/{task_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["answer"] is None
    with SessionLocal() as db:
        audit = db.scalar(select(AuditLog).where(AuditLog.action == "qa_cancelled"))
        assert audit is not None
        assert audit.target_id == task_id

    release.set()
    for _ in range(20):
        body = client.get(f"/api/qa/tasks/{task_id}").json()
        if body["status"] == "cancelled":
            time.sleep(0.05)
            break
    body = client.get(f"/api/qa/tasks/{task_id}").json()
    assert body["status"] == "cancelled"
    assert body["answer"] is None

    repeated = client.post(f"/api/qa/tasks/{task_id}/cancel")
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "cancelled"


def test_qa_task_creation_is_idempotent_by_client_request_id(monkeypatch) -> None:
    reset_database()
    calls: list[str] = []

    def fake_answer_question(db, question, options=None, include_debug=False, progress_reporter=None, **kwargs):
        calls.append(question)
        return QAResponse(
            answer="依据已核验。",
            answer_mode="answered",
            evidence_sufficiency="sufficient",
            intent="resume_qa",
            generation_status="completed",
        )

    monkeypatch.setattr("backend.app.services.qa_task_service.answer_question", fake_answer_question)
    payload = {
        "question": "同一个请求只能执行一次",
        "client_request_id": "qa-idempotent-request-0001",
        "include_debug": False,
    }
    first = client.post("/api/qa/tasks", json=payload)
    second = client.post("/api/qa/tasks", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["task_id"] == second.json()["task_id"]
    assert second.json()["client_request_id"] == payload["client_request_id"]

    task_id = first.json()["task_id"]
    for _ in range(20):
        body = client.get(f"/api/qa/tasks/{task_id}").json()
        if body["status"] == "completed":
            break
        time.sleep(0.05)
    assert body["status"] == "completed"
    assert calls == [payload["question"]]


def test_qa_refuses_weak_single_token_match(monkeypatch) -> None:
    reset_database()
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "quality_material.txt",
                "简历材料应做到来源清楚、表述一致、可追溯。",
                "text/plain",
            )
        },
    )

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", lambda question: [])

    response = client.post("/api/qa/ask", json={"question": "数据中心机房如何审批", "include_debug": True})

    assert response.status_code == 200
    body = response.json()
    assert body["answer_mode"] == "failed"
    assert "知识库中还没有相关记录" in body["answer"]
    assert body["context_package"]["is_final_answer"] is False


def test_qa_returns_503_when_retrieval_system_unavailable(monkeypatch) -> None:
    reset_database()

    def failing_retrieve(question: str):
        raise RetrievalServiceUnavailable("Qdrant hybrid 检索失败")

    monkeypatch.setattr("backend.app.services.rag_service.retrieve_citations", failing_retrieve)

    response = client.post("/api/qa/ask", json={"question": "综合成绩"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Qdrant hybrid 检索失败"


def test_audit_logs_record_upload_and_qa(monkeypatch) -> None:
    reset_database()
    question = "total score"
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("rules.txt", "综合成绩应等于各科目分项成绩之和。", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]
    monkeypatch.setattr(
        "backend.app.services.rag_service.retrieve_citations",
        lambda question: [fake_retrieval_match(document_id)],
    )
    client.post("/api/qa/ask", json={"question": question})

    response = client.get("/api/audit/logs")

    assert response.status_code == 200
    logs = response.json()["logs"]
    actions = [log["action"] for log in logs]
    assert "document_uploaded" in actions
    assert "qa_answered" in actions

    qa_log = next(log for log in logs if log["action"] == "qa_answered")
    qa_detail = json.loads(qa_log["detail"])
    assert qa_detail["question"] == question
    assert qa_detail["answer"]
    assert qa_detail["is_final_answer"] is True
    assert qa_detail["used_chunks"] == 1
    assert qa_detail["mode"] == "hedged"
    assert qa_detail["generation_status"] == "degraded"
    assert "citations" not in qa_detail
    assert qa_log["summary"] == "问答完成"
    assert qa_log["user_message"].startswith("已回答：")
    assert question in qa_log["user_message"]
    qa_details_json = json.loads(qa_log["details_json"])
    assert qa_details_json["question"] == question
    assert qa_details_json["answer"] == qa_detail["answer"]
    assert qa_details_json["used_chunks"] == 1
    assert qa_details_json["intent"] == "resume_qa"


def test_audit_logs_aggregate_repeated_warning_events() -> None:
    reset_database()
    from backend.app.services.audit_service import record_event

    with SessionLocal() as db:
        first = record_event(
            db,
            "document_marked_source_missing",
            "document",
            "DOC-MISSING",
            detail="original file missing; result: marked_source_missing",
            severity="warning",
            event_key="source_missing:DOC-MISSING",
            summary="原文件缺失",
            user_message="系统检测到原文件不可用。",
        )
        second = record_event(
            db,
            "document_marked_source_missing",
            "document",
            "DOC-MISSING",
            detail="original file missing; result: marked_source_missing",
            severity="warning",
            event_key="source_missing:DOC-MISSING",
            summary="原文件缺失",
            user_message="系统检测到原文件不可用。",
        )

        assert first.id == second.id

    response = client.get("/api/audit/logs")

    assert response.status_code == 200
    logs = response.json()["logs"]
    assert len(logs) == 1
    assert logs[0]["severity"] == "warning"
    assert logs[0]["summary"] == "原文件缺失"
    assert logs[0]["occurrence_count"] == 2


def test_audit_logs_archive_expired_logs_by_day() -> None:
    reset_database()
    old_created_at = datetime.now(timezone.utc) - timedelta(days=2)
    old_date = old_created_at.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
    archive_path = AUDIT_ARCHIVE_DIR / f"audit-{old_date}.pdf"
    legacy_archive_path = AUDIT_ARCHIVE_DIR / f"audit-{old_date}.md"
    deleted_marker_path = AUDIT_ARCHIVE_DIR / ".deleted-audit-archives.json"
    marker_backup = deleted_marker_path.read_text(encoding="utf-8") if deleted_marker_path.exists() else None
    if archive_path.exists():
        archive_path.unlink()
    if legacy_archive_path.exists():
        legacy_archive_path.unlink()
    if deleted_marker_path.exists():
        deleted_marker_path.unlink()

    try:
        with SessionLocal() as db:
            db.add(
                AuditLog(
                        action="qa_answered",
                        target_type="question",
                        target_id=None,
                        detail=json.dumps({"question": "old question", "answer": "old answer"}, ensure_ascii=False),
                        created_at=old_created_at,
                    )
                )
            db.add(
                AuditLog(
                    action="document_uploaded",
                    target_type="document",
                    target_id="DOC-TODAY",
                    detail="today document",
                    created_at=datetime.now(timezone.utc),
                )
            )
            db.commit()

        from backend.app.services.audit_service import archive_expired_audit_logs

        with SessionLocal() as db:
            archive_expired_audit_logs(db)

        log_response = client.get("/api/audit/logs")
        archive_response = client.get("/api/audit/archives")

        assert log_response.status_code == 200
        assert [log["action"] for log in log_response.json()["logs"]] == ["document_uploaded"]
        assert archive_path.exists()
        assert not legacy_archive_path.exists()
        assert any(archive["date"] == old_date for archive in archive_response.json()["archives"])
        assert any(archive["filename"] == archive_path.name for archive in archive_response.json()["archives"])

        detail_response = client.get(f"/api/audit/archives/{old_date}")
        assert detail_response.status_code == 200
        assert "old question" in detail_response.json()["content"]
        assert "old answer" in detail_response.json()["content"]

        delete_response = client.delete(f"/api/audit/archives/{old_date}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"] is True
        assert not archive_path.exists()

        legacy_archive_path.write_text("legacy content should stay deleted", encoding="utf-8")
        with SessionLocal() as db:
            db.add(
                AuditLog(
                    action="qa_answered",
                    target_type="question",
                    target_id=None,
                    detail="old log should not be rearchived",
                    created_at=old_created_at,
                )
            )
            db.commit()
            archive_expired_audit_logs(db)

        assert not archive_path.exists()
        assert not legacy_archive_path.exists()
        assert old_date not in {archive["date"] for archive in client.get("/api/audit/archives").json()["archives"]}
        with SessionLocal() as db:
            old_logs = [
                log
                for log in db.scalars(select(AuditLog).order_by(AuditLog.created_at.asc())).all()
                if log.created_at.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat() == old_date
            ]
        assert old_logs == []
    finally:
        if archive_path.exists():
            archive_path.unlink()
        if legacy_archive_path.exists():
            legacy_archive_path.unlink()
        if marker_backup is None:
            if deleted_marker_path.exists():
                deleted_marker_path.unlink()
        else:
            deleted_marker_path.write_text(marker_backup, encoding="utf-8")


def test_aspect_retrieval_fuses_queries_before_single_rerank(monkeypatch) -> None:
    from backend.app.services import rag_service

    aspect = QueryAspect(
        aspect_id="skill_mastery_identification",
        question="技能掌握情况应如何处理",
        search_queries=(
            QuerySearchQuery("技能掌握情况的评价标准和方法有哪些", "semantic_question", "贴近用户意图"),
            QuerySearchQuery("技能掌握 掌握程度 分类 认定方法", "document_style_statement", "贴近材料原文"),
            QuerySearchQuery("技能掌握 掌握程度 分类", "keyword_anchor", "术语兜底"),
        ),
        evidence_need="技能掌握情况的掌握程度、分类和评价方法",
        keywords=("技能掌握", "掌握程度", "标准"),
    )
    candidates = [
        VectorSearchResult(
            chunk_id=f"DOC-TEST-0001-CHUNK-{index:04d}",
            document_id="DOC-TEST-0001",
            filename="skills.md",
            section_title="技能掌握情况",
            page_number=None,
            text="技能掌握情况应基于项目实践、参与程度和支持材料确定。",
            embedding_text="章节：技能掌握情况\n\n技能掌握情况应基于项目实践、参与程度和支持材料确定。",
            token_count=50,
            score=1.0 - index * 0.01,
            chunk_type="paragraph",
        )
        for index in range(6)
    ]
    rerank_calls = []

    def fake_collect(queries, query_metadata=None, diagnostics=None, metadata_filter=None):
        assert len(queries) == 3
        if diagnostics is not None:
            diagnostics.query_count = len(queries)
            diagnostics.raw_candidate_count = len(candidates) * len(queries)
            diagnostics.candidate_count = len(candidates)
        return (
            candidates,
            {
                candidate.chunk_id: [
                    {"query": queries[0], "query_type": "semantic_question", "rank": index + 1}
                ]
                for index, candidate in enumerate(candidates)
            },
            [
                {
                    "search_query": query,
                    "query_type": "semantic_question",
                    "rationale": "",
                    "query_count": 1,
                    "raw_candidate_count": len(candidates),
                    "candidate_count": len(candidates),
                    "rerank_input_count": 0,
                    "rerank_call_count": 0,
                    "reranked_count": 0,
                    "filtered_count": 0,
                    "match_count": 0,
                    "timings_ms": {},
                    "score_range": {},
                    "query_variants": [query],
                }
                for query in queries
            ],
        )

    def fake_rerank(question, candidates, limit):
        rerank_calls.append((question, len(candidates), limit))
        return [RerankedChunk(candidate=candidate, rerank_score=0.9) for candidate in candidates[:2]]

    def fake_matches_from_reranked(question, reranked, diagnostics, limit=5):
        diagnostics.reranked_count = len(reranked)
        return [
            RetrievalMatch(
                citation=Citation(
                    document_id=item.candidate.document_id,
                    chunk_id=item.candidate.chunk_id,
                    filename=item.candidate.filename,
                    section_title=item.candidate.section_title,
                    excerpt=item.candidate.text,
                    score=item.candidate.score,
                    rerank_score=item.rerank_score,
                    chunk_type=item.candidate.chunk_type,
                    evidence_role="direct_evidence",
                ),
                score=item.candidate.score,
                rerank_score=item.rerank_score,
                coverage_score=1.0,
                evidence_role="direct_evidence",
            )
            for item in reranked
        ]

    monkeypatch.setattr(rag_service, "collect_candidates_with_query_hits", fake_collect)
    monkeypatch.setattr(rag_service, "rerank_candidates", fake_rerank)
    monkeypatch.setattr(rag_service, "matches_from_reranked", fake_matches_from_reranked)
    monkeypatch.setattr(rag_service, "_filter_indexed_matches", lambda db, matches, persona_id=None: matches)

    matches, diagnostics = rag_service._retrieve_aspect_matches(object(), aspect)

    assert len(matches) == 2
    assert rerank_calls == [
        ("技能掌握情况应如何处理\n技能掌握 掌握程度 分类 认定方法", 6, 6)
    ]
    fused = diagnostics[-1]
    assert fused["query_type"] == "aspect_fused"
    assert fused["rerank_call_count"] == 1
    assert fused["query_count"] == 3


def test_prompt_budget_recomputes_real_aspect_coverage(monkeypatch) -> None:
    from backend.app.services import rag_service

    monkeypatch.setattr(rag_service, "MAX_PROMPT_TOKENS", 3600)
    aspects = tuple(
        QueryAspect(
            aspect_id=f"aspect_{index}",
            question=f"核对事项 {index}",
            search_queries=(QuerySearchQuery(f"事项 {index}", "semantic_question", ""),),
            evidence_need=f"事项 {index} 的直接证据",
            keywords=(f"事项 {index}",),
        )
        for index in (1, 2)
    )
    chunks = [
        RetrievalResult(
            chunk_id=f"DOC-BUDGET-CHUNK-{index:04d}",
            rank=index,
            score=0.9,
            source_doc="预算测试.pdf",
            section_title=f"事项 {index}",
            section_path=[f"事项 {index}"],
            text=f"事项 {index} 的独立证据内容。",
            citation_label=f"[{index}]",
            metadata={"rerank_score": 0.9, "token_count": 3000},
        )
        for index in (1, 2)
    ]
    retrievals = [
        rag_service.AspectRetrieval(
            aspect=aspect,
            candidates=[chunk],
            diagnostics=[],
            citation_validation={"valid_chunks": 1},
            selected_chunk_ids=[],
            retrieval_covered=True,
        )
        for aspect, chunk in zip(aspects, chunks, strict=True)
    ]

    selected, summary = rag_service._select_prompt_chunks(
        "同时核对事项 1 和事项 2。",
        rag_service.QueryPlan(
            original_question="同时核对事项 1 和事项 2。",
            aspects=aspects,
            planner="test",
        ),
        retrievals,
    )

    assert [chunk.chunk_id for chunk in selected] == ["DOC-BUDGET-CHUNK-0001"]
    assert summary["covered_aspects"] == ["aspect_1"]
    assert summary["covered_by_retrieval_but_not_prompted"] == ["aspect_2"]
    assert summary["prompt_capacity_limited"] is True
    assert summary["aspect_selected_chunk_ids"]["aspect_2"] == []


def test_prompt_coverage_sync_marks_late_recovered_context_as_sufficient() -> None:
    from backend.app.services import rag_service

    aspect = QueryAspect(
        aspect_id="certificate_verification_process_control_subject",
        question="核对证书核实全过程控制主体",
        search_queries=(
            QuerySearchQuery(
                "证书核实全过程控制 主体 责任部门 岗位职责",
                "document_style_statement",
                "材料原文锚点",
            ),
        ),
        evidence_need="证书核实全过程控制主体的定义、责任部门、岗位职责或材料说明",
        keywords=("证书核实", "全过程控制", "主体", "责任部门", "岗位职责"),
    )
    recovered_chunk = RetrievalResult(
        chunk_id="DOC-CERT-CHUNK-0002",
        rank=1,
        score=0.91,
        source_doc="证书核实说明.pdf",
        section_title="证书核实全过程控制",
        section_path=["证书核实全过程控制"],
        text=(
            "应聘者应当始终对证书核实的全过程保持控制，包括确定需要核实的信息、"
            "选择适当的核实机构、填写核实材料并予以跟进。"
        ),
        citation_label="[1]",
        metadata={"document_id": "DOC-CERT", "rerank_score": 0.91},
    )
    retrieval = rag_service.AspectRetrieval(
        aspect=aspect,
        candidates=[],
        diagnostics=[],
        citation_validation={"valid_chunks": 0},
        selected_chunk_ids=[],
        retrieval_covered=False,
        covered=False,
    )
    plan = rag_service.QueryPlan(
        original_question="核对证书核实全过程控制主体。",
        aspects=(aspect,),
        planner="test",
    )

    rag_service._sync_aspect_coverage_from_prompt([recovered_chunk], [retrieval])
    prompt_selection = rag_service._prompt_selection_summary(
        1,
        [recovered_chunk],
        plan,
        [retrieval],
        {retrieval.aspect.aspect_id for retrieval in [retrieval] if retrieval.covered},
    )
    summary = rag_service._build_retrieval_summary(
        "核对证书核实全过程控制主体。",
        [recovered_chunk],
        plan,
        [retrieval],
        RetrievalDiagnostics(),
        {"valid_chunks": 1, "invalid_chunks": 0},
        prompt_selection,
    )

    assert retrieval.retrieval_covered is True
    assert retrieval.covered is True
    assert retrieval.selected_chunk_ids == ["DOC-CERT-CHUNK-0002"]
    assert recovered_chunk.metadata["prompt_matched_aspects"] == [
        "certificate_verification_process_control_subject"
    ]
    assert summary["has_sufficient_context"] is True
    assert summary["missing_aspects"] == []
    assert summary["prompt_selection"]["covered_aspects"] == [
        "certificate_verification_process_control_subject"
    ]






def test_aspect_anchor_documents_become_retrieval_metadata_filter(monkeypatch) -> None:
    """简历领域结构：aspect 锚定的对象文档名解析为 document_ids 检索过滤。"""
    from backend.app.services import rag_service

    captured: dict = {}

    def fake_collect(queries, query_metadata=None, diagnostics=None, metadata_filter=None):
        captured["metadata_filter"] = metadata_filter
        captured["queries"] = queries
        return [], {}, []

    monkeypatch.setattr(rag_service, "collect_candidates_with_query_hits", fake_collect)

    aspect = QueryAspect(
        aspect_id="obj_1",
        question="介绍一下你的秒杀项目",
        search_queries=(QuerySearchQuery("秒杀项目", "semantic_question", ""),),
        evidence_need="项目经历",
        keywords=("秒杀",),
        anchor_documents=("项目介绍_高并发电商秒杀平台.md",),
    )
    rag_service._retrieve_aspect_matches(object(), aspect)

    # object() 作为 db 时 _anchor_document_ids 查询失败 → 返回空集 → 不设过滤（退化安全）
    assert captured["metadata_filter"] is None
    assert captured["queries"] == ["秒杀项目"]


def test_anchor_document_ids_resolves_indexed_filenames(monkeypatch) -> None:
    """anchor_documents 文件名 → document_id 集合（仅已索引文档）。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.app.core.database import Base
    from backend.app.models.document import Document
    from backend.app.services import rag_service

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(Document(
            document_id="DOC-PROJ-1",
            filename="项目介绍_高并发电商秒杀平台.md",
            filename_norm="项目介绍_高并发电商秒杀平台.md",
            file_type="md",
            size=10,
            storage_path="/tmp/x.md",
            status="indexed",
        ))
        db.add(Document(
            document_id="DOC-PROJ-2",
            filename="项目介绍_外卖平台.md",
            filename_norm="项目介绍_外卖平台.md",
            file_type="md",
            size=10,
            storage_path="/tmp/y.md",
            status="indexed",
        ))
        db.commit()
        resolved = rag_service._anchor_document_ids(
            db, ("项目介绍_高并发电商秒杀平台.md", "不存在的文档.md")
        )

    assert resolved == {"DOC-PROJ-1"}


def test_complement_question_excludes_documents_from_final_prompt(monkeypatch) -> None:
    """补集问句：被排除对象文档的 chunk 绝不进入最终 prompt（即使 coverage 逻辑会倾向选它）。"""
    from backend.app.services import rag_service

    monkeypatch.setattr(rag_service, "MIN_PROMPT_CHUNKS", 2)
    monkeypatch.setattr(rag_service, "FORCE_MIN_CHUNKS", True)

    excluded_filename = "项目介绍_外卖平台.md"
    included_filename = "项目介绍_高并发电商秒杀平台.md"
    aspects = tuple(
        QueryAspect(
            aspect_id=f"aspect_{index}",
            question=f"还有哪些项目 {index}",
            search_queries=(QuerySearchQuery(f"项目 {index}", "semantic_question", ""),),
            evidence_need="项目经历",
            keywords=(f"项目 {index}",),
        )
        for index in (1, 2)
    )
    chunks = [
        RetrievalResult(
            chunk_id=f"CHUNK-EXCL-{index}",
            rank=index,
            score=0.95,
            source_doc=excluded_filename,
            section_title=f"项目 {index}",
            section_path=[f"项目 {index}"],
            text=f"被排除项目 {index} 的介绍内容。",
            citation_label=f"[{index}]",
            metadata={"rerank_score": 0.95, "filename": excluded_filename, "token_count": 500},
        )
        for index in (1, 2)
    ] + [
        RetrievalResult(
            chunk_id=f"CHUNK-INC-{index}",
            rank=index,
            score=0.6,
            source_doc=included_filename,
            section_title=f"保留项目 {index}",
            section_path=[f"保留项目 {index}"],
            text=f"保留项目 {index} 的介绍内容。",
            citation_label=f"[{index}]",
            metadata={"rerank_score": 0.6, "filename": included_filename, "token_count": 500},
        )
        for index in (1, 2)
    ]
    retrievals = [
        rag_service.AspectRetrieval(
            aspect=aspect,
            candidates=chunks,
            diagnostics=[],
            citation_validation={"valid_chunks": len(chunks)},
            selected_chunk_ids=[],
            retrieval_covered=True,
        )
        for aspect in aspects
    ]
    plan = rag_service.QueryPlan(
        original_question="除了外卖平台还有哪些项目？",
        aspects=aspects,
        planner="test",
        enumerative=True,
        excluded_documents=(excluded_filename,),
    )

    selected, _summary = rag_service._select_prompt_chunks(plan.original_question, plan, retrievals)

    # 最终 prompt 中不得出现被排除文档的任何 chunk
    selected_docs = {chunk.metadata.get("filename") or chunk.source_doc for chunk in selected}
    assert excluded_filename not in selected_docs
    assert all(chunk.source_doc != excluded_filename for chunk in selected)
    # 保留项目文档有入选（确保不是全被排除）
    assert included_filename in selected_docs


def test_complement_exclusion_survives_force_min_chunks(monkeypatch) -> None:
    """即使 MIN_PROMPT_CHUNKS 强制补足，被排除文档也不得因 coverage 逻辑重新进入 prompt。"""
    from backend.app.services import rag_service

    monkeypatch.setattr(rag_service, "MIN_PROMPT_CHUNKS", 4)
    monkeypatch.setattr(rag_service, "FORCE_MIN_CHUNKS", True)

    excluded_filename = "项目介绍_REV密码算法.md"
    included_filename = "项目介绍_外卖平台.md"
    aspect = QueryAspect(
        aspect_id="aspect_1",
        question="还有哪些项目？",
        search_queries=(QuerySearchQuery("项目", "semantic_question", ""),),
        evidence_need="项目经历",
        keywords=("项目",),
    )
    chunks = [
        RetrievalResult(
            chunk_id=f"CHUNK-{index}",
            rank=index,
            score=0.9,
            source_doc=(excluded_filename if index % 2 else included_filename),
            section_title=f"项目 {index}",
            section_path=[f"项目 {index}"],
            text=f"项目 {index} 介绍。",
            citation_label=f"[{index}]",
            metadata={
                "rerank_score": 0.9,
                "filename": (excluded_filename if index % 2 else included_filename),
                "token_count": 400,
            },
        )
        for index in range(1, 7)
    ]
    retrievals = [
        rag_service.AspectRetrieval(
            aspect=aspect,
            candidates=chunks,
            diagnostics=[],
            citation_validation={"valid_chunks": len(chunks)},
            selected_chunk_ids=[],
            retrieval_covered=True,
        )
    ]
    plan = rag_service.QueryPlan(
        original_question="除了 REV 项目还有哪些项目？",
        aspects=(aspect,),
        planner="test",
        enumerative=True,
        excluded_documents=(excluded_filename,),
    )

    selected, _summary = rag_service._select_prompt_chunks(plan.original_question, plan, retrievals)

    selected_docs = {chunk.metadata.get("filename") or chunk.source_doc for chunk in selected}
    assert excluded_filename not in selected_docs
    assert included_filename in selected_docs


def test_qa_api_answer_patch_isolates_real_rag(monkeypatch) -> None:
    """验证 monkeypatch 目标正确：patch qa.py 模块绑定的 answer_question 后，
    QA API 端点走 fake，不再触发真实 RAG（模型/Qdrant 零调用）。"""
    from backend.app.api import qa as qa_module
    from backend.app.schemas.qa import QAResponse

    reset_database()
    real_rag_called = {"called": False}

    def fake_answer(db, question, **kwargs):
        # 若 patch 未生效，这里不会被调用；真实 RAG 会走模型链路
        return QAResponse(
            answer=f"fake:{question}",
            answer_mode="answered",
            evidence_sufficiency="sufficient",
            intent="resume_qa",
            generation_status="completed",
        )

    # 打桩：若真实 answer_question 被调用（patch 失败），必然经过这些模块
    monkeypatch.setattr(
        "backend.app.services.intent_router_service.chat_completion_content",
        lambda *a, **k: real_rag_called.update(called=True) or "{}",
    )
    monkeypatch.setattr(qa_module, "answer_question", fake_answer)

    response = client.post("/api/qa/ask", json={"question": "介绍一下你的项目"})

    assert response.status_code == 200
    assert response.json()["answer"].startswith("fake:")
    assert real_rag_called["called"] is False
