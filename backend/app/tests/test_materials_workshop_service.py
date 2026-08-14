# -*- coding: utf-8 -*-
"""人物工坊服务测试（2026-08-14）：隐私清洗、LLM 输出解析、入库与回滚。"""

from __future__ import annotations

import json
from io import BytesIO

import pytest
from fastapi import UploadFile
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers

from backend.app.core.database import Base
from backend.app.models.document import Document, FactLedger, Persona, WorkshopJob
from backend.app.services import materials_workshop_service as workshop
from backend.app.services.materials_workshop_service import (
    _extract_json_object,
    _safe_filename,
    _sanitize_privacy,
    _split_input_chunks,
    _transform_batch,
    rollback_job,
)


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        size=len(content),
        headers=Headers({"content-type": "text/plain"}),
    )


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'workshop.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    yield db
    db.close()
    engine.dispose()


# ------------------------------------------------------------------ 隐私清洗


def test_sanitize_privacy_removes_pii() -> None:
    raw = (
        "我的手机号是13812345678，身份证 11010119900101123X，"
        "邮箱 zhangsan@example.com，地址在北京市朝阳区某某路1号。"
        "项目用了 Redis 做缓存。"
    )
    cleaned = _sanitize_privacy(raw)
    assert "13812345678" not in cleaned
    assert "11010119900101123X" not in cleaned
    assert "zhangsan@example.com" not in cleaned
    assert "[已脱敏]" in cleaned
    assert "Redis" in cleaned  # 业务内容保留


def test_sanitize_privacy_keeps_regular_numbers() -> None:
    raw = "并发实测 10 个用户恰好 10 单成功；成绩 90.33 分。"
    cleaned = _sanitize_privacy(raw)
    assert "10" in cleaned and "90.33" in cleaned
    assert "[已脱敏]" not in cleaned


# ------------------------------------------------------------------ 工具函数


def test_extract_json_object_tolerates_wrapping() -> None:
    content = '前置说明 {"a": 1, "b": [1,2]} 后置说明'
    assert _extract_json_object(content) == '{"a": 1, "b": [1,2]}'


def test_safe_filename_normalizes() -> None:
    assert _safe_filename('a/b\c:*.md') == "a_b_c__.md"
    assert _safe_filename("简历.docx").endswith(".md")
    assert _safe_filename("") == "材料加工_文档.md"


def test_split_input_chunks_respects_max() -> None:
    chunks = _split_input_chunks("a" * 1000, max_chars=300)
    assert len(chunks) >= 4
    assert all(len(chunk) <= 300 for chunk in chunks)


# ------------------------------------------------------------------ LLM 解析


def test_transform_batch_parses_llm_output(monkeypatch) -> None:
    payload = {
        "persona_profile": {"name": "张三", "summary": "Java 后端"},
        "knowledge_documents": [
            {"filename": "综合简历_张三.md", "content": "> 材料主题：综合简历\n\n张三，Java 后端工程师。"}
        ],
        "facts": [
            {"subject": "张三", "predicate": "岗位", "value": "Java 后端", "status": "confirmed"}
        ],
    }
    monkeypatch.setattr(
        workshop,
        "chat_completion_content",
        lambda config, messages, **kwargs: json.dumps(payload, ensure_ascii=False),
    )
    result = _transform_batch("原始材料", batch_index=1, total_batches=1)
    assert result.persona_profile["name"] == "张三"
    assert len(result.documents) == 1
    assert result.documents[0]["filename"].endswith(".md")
    assert len(result.facts) == 1


def test_transform_batch_invalid_json_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        workshop, "chat_completion_content", lambda config, messages, **kwargs: "不是 JSON"
    )
    with pytest.raises(workshop.WorkshopError):
        _transform_batch("材料", batch_index=1, total_batches=1)


def test_transform_batch_schema_violation_raises(monkeypatch) -> None:
    """skill 契约校验：输出结构不合 output-schema.json 时整批拒绝，不静默降级。"""
    payload = {
        "persona_profile": {"name": "张三"},
        "knowledge_documents": [{"filename": "简历.docx", "content": "不是 markdown"}],
        "facts": [{"subject": "a", "predicate": "b", "value": "c", "status": "maybe"}],
    }
    monkeypatch.setattr(
        workshop, "chat_completion_content", lambda config, messages, **kwargs: json.dumps(payload)
    )
    with pytest.raises(workshop.WorkshopError, match="不合 skill 契约"):
        _transform_batch("材料", batch_index=1, total_batches=1)


def test_transform_batch_loads_prompt_from_skill(monkeypatch) -> None:
    """提示词以 skill 目录为单一事实来源：system 消息为 skill 提示词 + 占位符替换。"""
    captured: dict = {}

    def fake_llm(config, messages, **kwargs) -> str:
        captured["messages"] = messages
        return json.dumps(
            {
                "persona_profile": {"name": "张三"},
                "knowledge_documents": [],
                "facts": [],
            }
        )

    monkeypatch.setattr(workshop, "chat_completion_content", fake_llm)
    _transform_batch("原始材料内容", batch_index=2, total_batches=3)
    content = captured["messages"][0]["content"]
    assert "简历材料加工师" in content
    assert "第 2/3 批" in content
    assert "原始材料内容" in content
    assert "{batch_index}" not in content  # 占位符全部替换


# ------------------------------------------------------------------ 入库与回滚


def test_ingest_and_rollback_job(session, monkeypatch) -> None:
    monkeypatch.setattr(workshop, "enqueue_document_index", lambda document_id: True)
    session.add(
        Persona(
            persona_id="persona-张三",
            name="张三",
            display_name="张三",
            profile_json="{}",
            status="confirmed",
            is_active=True,
        )
    )
    session.commit()

    job = workshop.WorkshopJob(
        job_id="job-test-1",
        persona_id="persona-张三",
        status="completed",
        stage="completed",
        generated_document_ids_json=json.dumps(["DOC-ROLLBACK-1"]),
        generated_fact_count=1,
    )
    session.add(job)
    session.add(
        Document(
            document_id="DOC-ROLLBACK-1",
            persona_id="persona-张三",
            filename="材料加工_张三.md",
            filename_norm="材料加工_张三.md",
            file_type="md",
            size=10,
            storage_path="/tmp/x.md",
            status="indexed",
        )
    )
    session.add(
        FactLedger(
            fact_id="workshop-job-test-1-0",
            persona_id="persona-张三",
            subject="张三",
            predicate="岗位",
            value="Java 后端",
            status="pending",
        )
    )
    session.commit()

    rolled = rollback_job(session, "job-test-1")
    assert rolled.status == "rolled_back"
    assert session.scalar(select(Document).where(Document.document_id == "DOC-ROLLBACK-1")) is None
    assert session.scalar(select(FactLedger).where(FactLedger.fact_id == "workshop-job-test-1-0")) is None


def test_transform_end_to_end_mock_llm(session, monkeypatch) -> None:
    """端到端：上传材料 → LLM 返回结构化结果 → 文档/档案/事实自动入库。"""
    monkeypatch.setattr(workshop, "enqueue_document_index", lambda document_id: True)
    session.add(
        Persona(
            persona_id="persona-李四",
            name="李四",
            display_name="李四",
            profile_json="{}",
            status="confirmed",
            is_active=True,
        )
    )
    session.commit()

    payload = {
        "persona_profile": {"name": "李四", "summary": "前端开发"},
        "knowledge_documents": [
            {"filename": "综合简历_李四.md", "content": "> 材料主题：综合简历\n\n李四，前端开发，电话 13812345678。"}
        ],
        "facts": [{"subject": "李四", "predicate": "方向", "value": "前端", "status": "pending"}],
    }
    monkeypatch.setattr(
        workshop,
        "chat_completion_content",
        lambda config, messages, **kwargs: json.dumps(payload, ensure_ascii=False),
    )
    file = _upload("李四资料.txt", "李四的前端开发经历，联系方式 13812345678。".encode("utf-8"))
    import asyncio

    job = asyncio.run(
        workshop.transform_materials(
            session,
            [file],
            persona_id="persona-李四",
            max_files=10,
            max_input_chars=5000,
        )
    )
    assert job.status == "completed"
    assert job.generated_fact_count >= 1
    # 加工 skill 版本随任务落库（SKILL.md frontmatter 为唯一来源）
    assert job.skill_version == workshop.skill_loader.skill_version()
    docs = session.scalars(
        select(Document).where(Document.persona_id == "persona-李四")
    ).all()
    assert any("李四" in doc.filename for doc in docs)
    # 隐私清洗：生成文档内容不含手机号
    from backend.app.models.document import DocumentChunk

    chunks = session.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id.in_([d.document_id for d in docs]))
    ).all()
    for chunk in chunks:
        assert "13812345678" not in chunk.text
    # 档案 draft（人工确认前不激活）
    persona = session.scalar(select(Persona).where(Persona.persona_id == "persona-李四"))
    assert persona.status == "draft" or persona.name == "李四"


def test_transform_materials_skill_unavailable_raises(session, monkeypatch) -> None:
    """skill 目录不可用（缺版本/契约）时任务拒绝开工，错误信息可读。"""
    import asyncio

    from backend.app.services.skill_loader import SkillLoadError

    monkeypatch.setattr(workshop, "enqueue_document_index", lambda document_id: True)
    # 不依赖宿主环境变量：固定配置与模拟 skill 损坏
    monkeypatch.setattr(workshop, "WORKSHOP_ENABLED", True)
    monkeypatch.setattr(workshop, "WORKSHOP_API_KEY", "test-key")
    monkeypatch.setattr(workshop, "WORKSHOP_MODEL", "test-model")
    monkeypatch.setattr(
        workshop.skill_loader,
        "skill_version",
        lambda: (_ for _ in ()).throw(SkillLoadError("SKILL.md frontmatter 缺少 metadata.version")),
    )
    file = _upload("材料.txt", "张三的简介。".encode("utf-8"))
    with pytest.raises(workshop.WorkshopError, match="加工 skill 不可用"):
        asyncio.run(
            workshop.transform_materials(session, [file], persona_id="persona-张三")
        )
