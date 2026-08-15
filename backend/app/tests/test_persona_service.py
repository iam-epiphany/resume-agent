# -*- coding: utf-8 -*-
"""人物模型测试（2026-08-14）：默认人物种子、切换隔离、提示词动态化。"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base, DEFAULT_PERSONA_ID
from backend.app.models.document import Document, FactLedger, Persona
from backend.app.services import persona_service
from backend.app.services.persona_service import (
    activate_persona,
    confirm_persona_profile,
    get_active_persona,
    persona_prompt_context,
    public_persona_view,
    update_persona_metadata,
)
from backend.app.services.persona_skill_service import load_persona_skill_package


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'persona.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    yield db
    db.close()
    engine.dispose()


def test_get_active_persona_auto_seeds_default(session) -> None:
    persona = get_active_persona(session)
    assert persona.persona_id == DEFAULT_PERSONA_ID
    assert persona.name == ""  # 默认人物不内置任何姓名（由部署者配置或工坊提取）
    assert persona.is_active is True


def test_prompt_context_for_confirmed_persona(session) -> None:
    persona = get_active_persona(session)
    ctx = persona_prompt_context(persona)
    # 无姓名人物：persona_name 为空 → 提示词用中性"求职者"表述，不注入任何姓名
    assert ctx["persona_name"] == ""
    assert "求职者" in ctx["persona_description"]


def test_prompt_context_for_draft_persona_stays_neutral(session) -> None:
    draft = Persona(
        persona_id="persona-测试人物",
        name="测试人物",
        display_name="测试人物",
        profile_json="{}",
        status="draft",
        is_active=False,
    )
    session.add(draft)
    session.commit()
    ctx = persona_prompt_context(draft)
    # draft 人物不注入姓名，避免未确认信息进入提示词
    assert ctx["persona_name"] == ""
    assert "求职者" in ctx["persona_description"]


def test_activate_persona_switches_and_invalidates_cache(session) -> None:
    persona_service._ACTIVE_CACHE.clear()
    get_active_persona(session)  # 缓存默认人物
    other = Persona(
        persona_id="persona-李四",
        name="李四",
        display_name="李四",
        profile_json='{"name": "李四", "summary": "前端开发方向。"}',
        status="confirmed",
        is_active=False,
    )
    session.add(other)
    session.commit()
    activated = activate_persona(session, "persona-李四")
    assert activated.persona_id == "persona-李四"
    active = get_active_persona(session)
    assert active.persona_id == "persona-李四"
    assert active.is_active is True
    default = session.scalar(select(Persona).where(Persona.persona_id == DEFAULT_PERSONA_ID))
    assert default.is_active is False


def test_public_view_omits_full_profile(session) -> None:
    persona = get_active_persona(session)
    view = public_persona_view(persona)
    assert view["name"] == ""
    assert view["is_active"] is True
    assert "profile" not in view  # 不暴露完整档案
    assert view["skill_package"] is None  # 未生成包时为 None（元信息字段常驻）


def test_confirm_profile_regenerates_skill_package(session) -> None:
    """档案确认 → 人物 Skill 包重建：profile.md 反映确认后的档案。"""
    draft = Persona(
        persona_id="persona-赵六",
        name="赵六",
        display_name="赵六",
        profile_json='{"name": "赵六", "summary": "草稿摘要"}',
        status="draft",
        is_active=False,
    )
    session.add(draft)
    session.commit()

    confirmed = confirm_persona_profile(
        session, "persona-赵六", profile={"name": "赵六", "summary": "确认后的摘要：RAG 后端"}
    )
    assert confirmed.status == "confirmed"
    package = load_persona_skill_package(confirmed)
    assert package is not None
    profile_md = package["files"]["persona-赵六/references/profile.md"]
    assert "确认后的摘要：RAG 后端" in profile_md
    assert "草稿摘要" not in profile_md


def test_update_metadata_regenerates_skill_package(session) -> None:
    """姓名/称呼修改 → 人物 Skill 包重建：SKILL.md 中的姓名随之更新。"""
    persona = get_active_persona(session)
    update_persona_metadata(session, persona.persona_id, display_name="老王")
    package = load_persona_skill_package(persona)
    assert package is not None
    skill_md = package["files"]["persona-default/SKILL.md"]
    assert "老王" in skill_md


def test_persona_scoped_documents_and_facts(session) -> None:
    persona = get_active_persona(session)
    doc = Document(
        document_id="DOC-X1",
        persona_id=persona.persona_id,
        filename="测试.md",
        filename_norm="测试.md",
        file_type="md",
        size=10,
        storage_path="/tmp/x.md",
        status="indexed",
    )
    session.add(doc)
    fact = FactLedger(
        fact_id="fact-x1",
        persona_id=persona.persona_id,
        subject="测试",
        predicate="属性",
        value="值",
    )
    session.add(fact)
    session.commit()
    assert session.scalar(select(Document).where(Document.document_id == "DOC-X1")).persona_id == persona.persona_id
    assert session.scalar(select(FactLedger).where(FactLedger.fact_id == "fact-x1")).persona_id == persona.persona_id
