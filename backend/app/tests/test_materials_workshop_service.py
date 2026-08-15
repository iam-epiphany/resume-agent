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
        "邮箱 zhangsan@example.com，家庭住址在北京市朝阳区某某路1号院。"
        "项目用了 Redis 做缓存。"
    )
    cleaned = _sanitize_privacy(raw)
    assert "13812345678" not in cleaned
    assert "11010119900101123X" not in cleaned
    assert "zhangsan@example.com" not in cleaned
    assert "北京市朝阳区某某路1号" not in cleaned  # 家庭住址（门牌号级）脱敏
    assert "[已脱敏]" in cleaned
    assert "Redis" in cleaned  # 业务内容保留


def test_sanitize_privacy_keeps_regular_numbers() -> None:
    raw = "并发实测 10 个用户恰好 10 单成功；成绩 90.33 分。项目部署在上海市。"
    cleaned = _sanitize_privacy(raw)
    assert "10" in cleaned and "90.33" in cleaned
    assert "上海市" in cleaned  # 无门牌号的地区表述不脱敏（避免误伤）
    assert "[已脱敏]" not in cleaned


def test_parse_document_safely_pre_sanitizes_pii() -> None:
    """PII 在调用 LLM 前本地预清洗：解析文本已无手机号/邮箱（不发给外部 LLM）。"""
    file = _upload(
        "材料.txt",
        "张三的联系方式 13812345678，邮箱 zhangsan@example.com，做过 Java 后端。".encode("utf-8"),
    )
    text = workshop.parse_document_safely(file)
    assert "13812345678" not in text
    assert "zhangsan@example.com" not in text
    assert "[已脱敏]" in text
    assert "Java 后端" in text


def test_sanitize_privacy_payload_covers_all_outputs() -> None:
    """三道出口统一清洗：文档 content、人物档案、事实字段都不留 PII。"""
    payload = {
        "persona_profile": {"name": "张三", "summary": "电话 13812345678"},
        "knowledge_documents": [
            {"filename": "综合简历_张三.md", "content": "> 材料主题：综合简历\n\n邮箱 zhangsan@example.com"}
        ],
        "facts": [
            {"subject": "张三", "predicate": "联系方式", "value": "13812345678", "source_file": "简历.pdf"}
        ],
    }
    cleaned = workshop._sanitize_privacy_payload(payload)
    serialized = json.dumps(cleaned, ensure_ascii=False)
    assert "13812345678" not in serialized
    assert "zhangsan@example.com" not in serialized
    assert cleaned["persona_profile"]["summary"].startswith("电话 [已脱敏]")
    assert "13812345678" not in cleaned["facts"][0]["value"]


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
            {
                "subject": "张三",
                "predicate": "岗位",
                "value": "Java 后端",
                "evidence_status": "explicit",
                "source_file": "简历.pdf",
            }
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
    assert result.facts[0]["evidence_status"] == "explicit"
    assert result.facts[0]["source_file"] == "简历.pdf"


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
        "facts": [
            {
                "subject": "a",
                "predicate": "b",
                "value": "c",
                "evidence_status": "maybe",
                "source_file": "简历.pdf",
            }
        ],
    }
    monkeypatch.setattr(
        workshop, "chat_completion_content", lambda config, messages, **kwargs: json.dumps(payload)
    )
    with pytest.raises(workshop.WorkshopError, match="不合 skill 契约"):
        _transform_batch("材料", batch_index=1, total_batches=1)


def test_transform_batch_loads_prompt_from_skill(monkeypatch) -> None:
    """提示词以 skill 目录为单一事实来源；System/User 分离防 prompt injection：
    system 只含固定指令（含「不可信」声明、无输入内容），原始材料只进 user 消息。"""
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
    assert [message["role"] for message in captured["messages"]] == ["system", "user"]
    system_content = captured["messages"][0]["content"]
    user_content = captured["messages"][1]["content"]
    assert "简历材料加工师" in system_content
    assert "不可信" in system_content  # 防御声明在 system 侧
    assert "原始材料内容" not in system_content  # 用户内容绝不进入 system
    for placeholder in ("{batch_index}", "{total_batches}", "{input_text}"):
        assert placeholder not in system_content
    assert "不可信原始材料" in user_content
    assert "第 2/3 批" in user_content
    assert "原始材料内容" in user_content
    assert "{input_text}" not in user_content  # 占位符全部替换


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
            evidence_status="explicit",
            review_status="pending",
        )
    )
    session.commit()

    rolled = rollback_job(session, "job-test-1")
    assert rolled.status == "rolled_back"
    assert session.scalar(select(Document).where(Document.document_id == "DOC-ROLLBACK-1")) is None
    assert session.scalar(select(FactLedger).where(FactLedger.fact_id == "workshop-job-test-1-0")) is None
    # 回滚后人物 Skill 包重建（文档/事实删除后包同步收缩；best-effort 不抛错）
    persona = session.scalar(select(Persona).where(Persona.persona_id == "persona-张三"))
    assert persona.skill_package_json is not None


def test_transform_end_to_end_mock_llm(session, monkeypatch) -> None:
    """端到端：上传材料 → LLM 返回结构化结果 → 文档/档案/事实自动入库。

    hermetic（2026-08-15）：不依赖宿主环境变量，固定配置与 mock LLM，
    与 test_transform_materials_skill_unavailable_raises 同法。
    """
    monkeypatch.setattr(workshop, "WORKSHOP_ENABLED", True)
    monkeypatch.setattr(workshop, "WORKSHOP_API_KEY", "test-key")
    monkeypatch.setattr(workshop, "WORKSHOP_MODEL", "test-model")
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
        "facts": [
            {
                "subject": "李四",
                "predicate": "方向",
                "value": "前端",
                "evidence_status": "explicit",
                "source_file": "李四资料.txt",
            }
        ],
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
    # 人物 Skill 包已自动生成：含加工文档（隐私清洗后的内容）与规范版本
    from backend.app.services.persona_skill_service import load_persona_skill_package

    package = load_persona_skill_package(persona)
    assert package is not None
    assert package["skill_version"] == job.skill_version
    doc_content = package["files"]["persona-李四/references/综合简历_李四.md"]
    assert doc_content.startswith("> 材料主题：综合简历")
    assert "13812345678" not in doc_content  # 包内内容与入库文档一致（已脱敏）


def test_transform_completes_without_skill_package_when_template_missing(session, monkeypatch) -> None:
    """人物 Skill 包模板缺失 → 任务仍成功（包是辅助产物，best-effort 跳过）。"""
    import asyncio

    from backend.app.services.skill_loader import SkillLoadError

    monkeypatch.setattr(workshop, "WORKSHOP_ENABLED", True)
    monkeypatch.setattr(workshop, "WORKSHOP_API_KEY", "test-key")
    monkeypatch.setattr(workshop, "WORKSHOP_MODEL", "test-model")
    monkeypatch.setattr(workshop, "enqueue_document_index", lambda document_id: True)
    session.add(
        Persona(
            persona_id="persona-王五",
            name="王五",
            display_name="王五",
            profile_json="{}",
            status="confirmed",
            is_active=True,
        )
    )
    session.commit()

    payload = {
        "persona_profile": {"name": "王五", "summary": "测试"},
        "knowledge_documents": [
            {"filename": "综合简历_王五.md", "content": "> 材料主题：综合简历\n\n王五的简介。"}
        ],
        "facts": [],
    }
    monkeypatch.setattr(
        workshop,
        "chat_completion_content",
        lambda config, messages, **kwargs: json.dumps(payload, ensure_ascii=False),
    )
    monkeypatch.setattr(
        workshop.skill_loader,
        "load_persona_skill_template",
        lambda: (_ for _ in ()).throw(SkillLoadError("加工 skill 缺少必需文件：persona-skill-template.md")),
    )
    file = _upload("王五资料.txt", "王五的简介。".encode("utf-8"))
    job = asyncio.run(
        workshop.transform_materials(
            session,
            [file],
            persona_id="persona-王五",
            max_files=10,
            max_input_chars=5000,
        )
    )
    assert job.status == "completed"
    assert job.generated_fact_count == 0
    persona = session.scalar(select(Persona).where(Persona.persona_id == "persona-王五"))
    assert persona.skill_package_json is None  # 模板缺失：跳过生成，不留空壳


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


# ------------------------------------------------------------------ Reduce 与原子性（2026-08-15）


def test_transform_multi_batch_reduce_and_provenance(session, monkeypatch) -> None:
    """跨批次 Reduce：输入带来源标注（Provenance）、同名异内容重命名并记冲突、
    事实冲突标 conflict 入库。"""
    import asyncio

    monkeypatch.setattr(workshop, "WORKSHOP_ENABLED", True)
    monkeypatch.setattr(workshop, "WORKSHOP_API_KEY", "test-key")
    monkeypatch.setattr(workshop, "WORKSHOP_MODEL", "test-model")
    monkeypatch.setattr(workshop, "enqueue_document_index", lambda document_id: True)
    session.add(
        Persona(
            persona_id="persona-赵六",
            name="赵六",
            display_name="赵六",
            profile_json="{}",
            status="confirmed",
            is_active=True,
        )
    )
    session.commit()

    calls = {"n": 0}
    captured: dict = {}

    def fake_llm(config, messages, **kwargs) -> str:
        calls["n"] += 1
        captured["user_message"] = messages[1]["content"]
        batch = calls["n"]
        payload = {
            "persona_profile": {"name": "赵六", "summary": f"批次{batch}"},
            "knowledge_documents": [
                {
                    "filename": "项目经历_平台.md",
                    "content": "> 材料主题：项目经历\n\n平台项目描述。",
                }
            ],
            "facts": [
                {
                    "subject": "平台",
                    "predicate": "并发",
                    "value": f"{batch}00 QPS",
                    "evidence_status": "explicit",
                    "source_file": "赵六资料.txt",
                }
            ],
        }
        if batch == 2:
            payload["knowledge_documents"].append(
                {
                    "filename": "项目经历_平台.md",
                    "content": "> 材料主题：项目经历\n\n平台项目另一版本描述。",
                }
            )
        return json.dumps(payload, ensure_ascii=False)

    def fake_split(text, max_chars):
        # 捕获带来源标注的组合文本（Source Provenance 入口），并强制拆成两批
        captured["combined"] = text
        return ["第1批", "第2批"]

    monkeypatch.setattr(workshop, "chat_completion_content", fake_llm)
    monkeypatch.setattr(workshop, "_split_input_chunks", fake_split)
    file = _upload("赵六资料.txt", "赵六项目经历，会拆成多批。".encode("utf-8"))
    job = asyncio.run(
        workshop.transform_materials(
            session, [file], persona_id="persona-赵六", max_files=10, max_input_chars=5000
        )
    )
    assert job.status == "completed"
    # Source Provenance：组合文本按来源标注，模型可知每句出处
    assert "【来源：赵六资料.txt】" in captured["combined"]
    # 同名异内容 → 重命名 _2 后缀，两篇都保留（不静默丢弃）
    docs = session.scalars(
        select(Document).where(Document.persona_id == "persona-赵六")
    ).all()
    names = sorted(doc.filename for doc in docs)
    assert names == ["项目经历_平台.md", "项目经历_平台_2.md"]
    # 冲突清单落库（同名异内容 + 事实多值冲突）
    conflicts = json.loads(job.conflicts_json or "[]")
    assert any("同名文档内容冲突" in item for item in conflicts)
    assert any("事实冲突" in item for item in conflicts)
    # 事实冲突 → evidence_status=conflict 入库（保留冲突不做选择）
    facts = session.scalars(
        select(FactLedger).where(FactLedger.persona_id == "persona-赵六")
    ).all()
    assert len(facts) == 2
    assert all(fact.evidence_status == "conflict" for fact in facts)
    assert all(fact.review_status == "pending" for fact in facts)


def test_transform_failure_rolls_back_ingested_documents(session, monkeypatch) -> None:
    """原子性：入库中途失败 → job=failed、已入库文档自动回滚、不留半成功数据。"""
    import asyncio

    monkeypatch.setattr(workshop, "WORKSHOP_ENABLED", True)
    monkeypatch.setattr(workshop, "WORKSHOP_API_KEY", "test-key")
    monkeypatch.setattr(workshop, "WORKSHOP_MODEL", "test-model")
    monkeypatch.setattr(workshop, "enqueue_document_index", lambda document_id: True)
    session.add(
        Persona(
            persona_id="persona-钱七",
            name="钱七",
            display_name="钱七",
            profile_json="{}",
            status="confirmed",
            is_active=True,
        )
    )
    session.commit()

    payload = {
        "persona_profile": {"name": "钱七"},
        "knowledge_documents": [
            {"filename": "综合简历_钱七.md", "content": "> 材料主题：综合简历\n\n钱七简介。"},
            {"filename": "技能专长_钱七.md", "content": "> 材料主题：技能专长\n\n钱七技能。"},
        ],
        "facts": [
            {
                "subject": "钱七",
                "predicate": "方向",
                "value": "后端",
                "evidence_status": "explicit",
                "source_file": "钱七资料.txt",
            }
        ],
    }
    monkeypatch.setattr(
        workshop,
        "chat_completion_content",
        lambda config, messages, **kwargs: json.dumps(payload, ensure_ascii=False),
    )
    real_ingest = workshop._ingest_markdown
    calls = {"n": 0}

    async def flaky_ingest(db, job, persona_id, filename, content, sources=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise workshop.WorkshopError("模拟第二篇入库失败")
        return await real_ingest(db, job, persona_id, filename, content, sources=sources)

    monkeypatch.setattr(workshop, "_ingest_markdown", flaky_ingest)
    file = _upload("钱七资料.txt", "钱七的后端经历。".encode("utf-8"))
    with pytest.raises(workshop.WorkshopError, match="模拟第二篇入库失败"):
        asyncio.run(
            workshop.transform_materials(
                session, [file], persona_id="persona-钱七", max_files=10, max_input_chars=5000
            )
        )
    job = session.scalar(select(workshop.WorkshopJob).order_by(workshop.WorkshopJob.id.desc()))
    assert job.status == "failed"
    assert "已自动回滚 1 篇" in job.error
    # 第一篇已入库的文档也被回滚：不留半成功数据
    remaining = session.scalars(
        select(Document).where(Document.persona_id == "persona-钱七")
    ).all()
    assert remaining == []
