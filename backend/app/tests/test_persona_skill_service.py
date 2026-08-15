# -*- coding: utf-8 -*-
"""人物 Skill 包服务测试（2026-08-15）：确定性组装、主题归类、重建与 zip 序列化。"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import Document, FactLedger, Persona
from backend.app.services import persona_skill_service as skill_pkg
from backend.app.services import skill_loader
from backend.app.services.persona_skill_service import (
    build_persona_skill_package,
    load_persona_skill_package,
    package_dir_name,
    persona_skill_zip_bytes,
    regenerate_persona_skill,
    skill_package_info,
)


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'skill-pkg.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    yield db
    db.close()
    engine.dispose()


def _persona(db, persona_id="persona-张三", name="张三", profile=None) -> Persona:
    default_profile = {
        "name": name,
        "display_name": name,
        "summary": "AI 应用后端工程师。",
        "education": "西安电子科技大学",
        "job_intent": "AI 应用后端",
        "skills": ["FastAPI", "RAG"],
        "projects": ["XDU EchoGuide"],
    }
    persona = Persona(
        persona_id=persona_id,
        name=name,
        display_name=name,
        profile_json=json.dumps(profile if profile is not None else default_profile, ensure_ascii=False),
        status="confirmed",
        is_active=True,
    )
    db.add(persona)
    db.commit()
    return persona


def _document(db, tmp_path, document_id: str, persona_id: str, filename: str, content: str) -> Document:
    path = tmp_path / f"{document_id}.md"
    path.write_text(content, encoding="utf-8")
    doc = Document(
        document_id=document_id,
        persona_id=persona_id,
        filename=filename,
        filename_norm=filename,
        file_type="md",
        size=len(content.encode("utf-8")),
        storage_path=str(path),
        status="indexed",
    )
    db.add(doc)
    db.commit()
    return doc


# ------------------------------------------------------------------ 目录名


def test_package_dir_name_normalizes() -> None:
    persona = _persona_dbless("persona-张三")
    assert package_dir_name(persona) == "persona-张三"
    assert package_dir_name(_persona_dbless("default")) == "persona-default"
    assert package_dir_name(_persona_dbless("persona-李 四/郎")) == "persona-李_四_郎"


def _persona_dbless(persona_id: str) -> Persona:
    return Persona(persona_id=persona_id, name="x", display_name="x")


# ------------------------------------------------------------------ 组装


def test_build_package_structure(session, tmp_path) -> None:
    persona = _persona(session)
    _document(
        session, tmp_path, "DOC-P1", persona.persona_id, "项目经历_AI开放平台.md",
        "> 材料主题：项目经历\n\n负责检索链路，压测 200 QPS。",
    )
    _document(
        session, tmp_path, "DOC-S1", persona.persona_id, "技能专长.md",
        "> 材料主题：技能专长\n\n熟悉 FastAPI 与 RAG 检索。",
    )
    # 原始上传（无主题标记）不进入 Skill 包
    _document(
        session, tmp_path, "DOC-RAW", persona.persona_id, "原始笔记.txt.md",
        "零散笔记，未加工。",
    )
    session.add_all(
        [
            FactLedger(
                fact_id="f1", persona_id=persona.persona_id, subject="XDU EchoGuide",
                predicate="并发", value="200 QPS", evidence_status="explicit",
                review_status="pending", source_type="test_report",
            ),
            # 与 f1 同值：去重后只导出一条
            FactLedger(
                fact_id="f2", persona_id=persona.persona_id, subject="XDU EchoGuide",
                predicate="并发", value="200 QPS", evidence_status="explicit",
                review_status="pending", source_type="test_report",
            ),
            # 其他人物的事实不进入本包
            FactLedger(
                fact_id="f3", persona_id="persona-别人", subject="别人", predicate="方向", value="前端",
                evidence_status="missing", review_status="pending",
            ),
        ]
    )
    session.commit()

    package = build_persona_skill_package(session, persona)
    assert package is not None
    assert package["dir_name"] == "persona-张三"
    assert package["skill_version"] == skill_loader.skill_version()
    assert package["generated_at"]

    files = package["files"]
    prefix = "persona-张三/"
    assert set(files) == {
        f"{prefix}SKILL.md",
        f"{prefix}facts.json",
        f"{prefix}references/profile.md",
        f"{prefix}references/projects/项目经历_AI开放平台.md",
        f"{prefix}references/技能专长.md",
    }

    # SKILL.md：frontmatter 完整、占位符全部替换、包含回答规则
    skill_md = files[f"{prefix}SKILL.md"]
    assert skill_md.startswith("---\nname: persona-张三")
    assert "{{" not in skill_md and "}}" not in skill_md
    assert "张三" in skill_md
    assert "渐进式加载" in skill_md
    assert "references/projects/" in skill_md

    # profile.md：档案渲染（主题标记 + 各字段）
    profile_md = files[f"{prefix}references/profile.md"]
    assert profile_md.startswith("# 张三 · 人物档案")
    assert "> 材料主题：综合简历" in profile_md
    assert "AI 应用后端工程师" in profile_md
    assert "## 技能专长" in profile_md and "- FastAPI" in profile_md
    assert "## 项目经历" in profile_md and "- XDU EchoGuide" in profile_md

    # 项目经历文档归入 projects/，其余放根目录；原始上传被排除
    assert "原始笔记" not in "".join(files)

    # facts.json：导出该人物事实、按 实体-属性-值 去重（含证据/审核两维状态）
    facts = json.loads(files[f"{prefix}facts.json"])
    assert facts == [
        {
            "subject": "XDU EchoGuide",
            "predicate": "并发",
            "value": "200 QPS",
            "evidence_status": "explicit",
            "review_status": "pending",
            "source_type": "test_report",
        }
    ]


def test_build_package_without_documents(session) -> None:
    persona = _persona(session, persona_id="persona-新人", name="新人", profile={"name": "新人"})
    package = build_persona_skill_package(session, persona)
    assert package is not None
    prefix = "persona-新人/"
    assert set(package["files"]) == {
        f"{prefix}SKILL.md",
        f"{prefix}facts.json",
        f"{prefix}references/profile.md",
    }
    assert json.loads(package["files"][f"{prefix}facts.json"]) == []


def test_build_package_skips_when_template_missing(session, monkeypatch, tmp_path) -> None:
    """模板缺失（skill 目录损坏）→ 返回 None，不抛错（best-effort）。"""
    persona = _persona(session)
    monkeypatch.setattr(skill_pkg.skill_loader, "skill_root", lambda: tmp_path / "no-skill")
    assert build_persona_skill_package(session, persona) is None


# ------------------------------------------------------------------ 重建落库


def test_regenerate_stores_package_and_info(session) -> None:
    persona = _persona(session)
    assert regenerate_persona_skill(session, persona.persona_id) is True

    stored = session.get(Persona, persona.id)
    package = load_persona_skill_package(stored)
    assert package is not None
    assert package["dir_name"] == "persona-张三"
    assert stored.skill_package_updated_at is not None

    info = skill_package_info(stored)
    assert info == {
        "file_count": 3,
        "skill_version": skill_loader.skill_version(),
        "generated_at": package["generated_at"],
    }


def test_regenerate_updates_profile_md_after_confirm(session) -> None:
    """档案确认/修改后重建：profile.md 反映最新档案。"""
    persona = _persona(session, profile={"name": "张三", "summary": "旧摘要"})
    regenerate_persona_skill(session, persona.persona_id)

    stored = session.get(Persona, persona.id)
    persona.profile_json = json.dumps({"name": "张三", "summary": "新摘要：RAG 工程师"}, ensure_ascii=False)
    session.commit()
    assert regenerate_persona_skill(session, persona.persona_id) is True

    package = load_persona_skill_package(session.get(Persona, persona.id))
    profile_md = package["files"]["persona-张三/references/profile.md"]
    assert "新摘要：RAG 工程师" in profile_md
    assert "旧摘要" not in profile_md


def test_regenerate_returns_false_when_template_missing(session, monkeypatch, tmp_path) -> None:
    persona = _persona(session)
    monkeypatch.setattr(skill_pkg.skill_loader, "skill_root", lambda: tmp_path / "no-skill")
    assert regenerate_persona_skill(session, persona.persona_id) is False
    assert session.get(Persona, persona.id).skill_package_json is None


def test_regenerate_returns_false_for_unknown_persona(session) -> None:
    assert regenerate_persona_skill(session, "persona-不存在") is False


def test_skill_package_info_none_without_package(session) -> None:
    persona = _persona(session)
    assert skill_package_info(persona) is None


# ------------------------------------------------------------------ zip


def test_zip_roundtrip(session) -> None:
    persona = _persona(session)
    package = build_persona_skill_package(session, persona)
    assert package is not None
    raw = persona_skill_zip_bytes(package)
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        assert set(archive.namelist()) == set(package["files"])
        skill_md = archive.read("persona-张三/SKILL.md").decode("utf-8")
        assert skill_md == package["files"]["persona-张三/SKILL.md"]


def test_package_sanitizes_traversal_filenames(session, tmp_path) -> None:
    """恶意文件名（../、反斜杠）清洗后进包：zip 条目不得逃逸包目录（zip-slip）。"""
    persona = _persona(session)
    _document(
        session, tmp_path, "DOC-EVIL1", persona.persona_id, "../../evil.md",
        "> 材料主题：项目经历\n\n跨目录逃逸文件名。",
    )
    _document(
        session, tmp_path, "DOC-EVIL2", persona.persona_id, "a\\b.md",
        "> 材料主题：技能专长\n\n反斜杠分隔文件名。",
    )
    package = build_persona_skill_package(session, persona)
    assert package is not None
    prefix = "persona-张三/"
    for relative in package["files"]:
        parts = relative.split("/")
        assert ".." not in parts  # 任何路径段都不得是相对路径逃逸
        assert "\\" not in relative
        assert relative.startswith(prefix)
    # 清洗结果：../../evil.md → evil.md（projects/ 下）、a\b.md → a_b.md（根目录）
    assert f"{prefix}references/projects/evil.md" in package["files"]
    assert f"{prefix}references/a_b.md" in package["files"]


def test_package_deduplicates_same_filename(session, tmp_path) -> None:
    """清洗后重名的文档进包不互相覆盖：后续追加 _2/_3 后缀。

    数据库对原始文件名有唯一约束（filename_norm），同名原始文档进不了库；
    但「原始名不同、清洗后同名」（路径分隔符/非法字符 → 下划线）的场景
    由本包组装层兜底。
    """
    persona = _persona(session)
    _document(
        session, tmp_path, "DOC-D1", persona.persona_id, "技能专长.md",
        "> 材料主题：技能专长\n\n内容甲。",
    )
    _document(
        session, tmp_path, "DOC-D2", persona.persona_id, "a/b.md",
        "> 材料主题：项目经历\n\n路径分隔符重名。",
    )
    _document(
        session, tmp_path, "DOC-D3", persona.persona_id, "a\\b.md",
        "> 材料主题：项目经历\n\n清洗后重名。",
    )
    _document(
        session, tmp_path, "DOC-D4", persona.persona_id, "a:b.md",
        "> 材料主题：项目经历\n\n非法字符重名。",
    )
    package = build_persona_skill_package(session, persona)
    assert package is not None
    files = package["files"]
    # a/b.md、a\b.md、a:b.md 清洗后均为 a_b.md：第一篇保留原名，后续加 _2/_3
    assert "persona-张三/references/projects/a_b.md" in files
    assert "persona-张三/references/projects/a_b_2.md" in files
    assert "persona-张三/references/projects/a_b_3.md" in files
