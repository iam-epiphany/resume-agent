# -*- coding: utf-8 -*-
"""skill 加载器测试（2026-08-14）：版本解析、提示词/契约加载、缺失报错。"""

from __future__ import annotations

import pytest

from backend.app.services import skill_loader
from backend.app.services.skill_loader import (
    SkillLoadError,
    load_output_schema,
    load_transform_prompt,
    skill_version,
)


def test_skill_version_readable() -> None:
    version = skill_version()
    # 语义化版本（至少两段，如 1.0.0），且来自 SKILL.md frontmatter
    assert version.count(".") >= 1
    sk = (skill_loader.skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert version in sk.split("---", 2)[1]


def test_transform_prompt_has_placeholders() -> None:
    prompt = load_transform_prompt()
    assert "简历材料加工师" in prompt
    for placeholder in ("{batch_index}", "{total_batches}", "{input_text}"):
        assert placeholder in prompt


def test_output_schema_is_valid_contract() -> None:
    schema = load_output_schema()
    assert isinstance(schema, dict)
    assert "knowledge_documents" in schema["required"]
    assert schema["$schema"]


def test_missing_skill_dir_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(skill_loader, "skill_root", lambda: tmp_path / "no-such-skill")
    with pytest.raises(SkillLoadError, match="缺少必需文件"):
        skill_version()


def test_missing_version_in_frontmatter_raises(monkeypatch, tmp_path) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: resume-materials-workshop\n---\n正文", encoding="utf-8"
    )
    monkeypatch.setattr(skill_loader, "skill_root", lambda: skill_dir)
    with pytest.raises(SkillLoadError, match="metadata.version"):
        skill_version()


def test_corrupt_schema_raises(monkeypatch, tmp_path) -> None:
    skill_dir = tmp_path / "skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: resume-materials-workshop\nmetadata:\n  version: \"1.0.0\"\n---\n正文",
        encoding="utf-8",
    )
    (skill_dir / "references" / "transform-prompt.md").write_text("提示词", encoding="utf-8")
    (skill_dir / "references" / "output-schema.json").write_text("{ 不是 JSON", encoding="utf-8")
    monkeypatch.setattr(skill_loader, "skill_root", lambda: skill_dir)
    with pytest.raises(SkillLoadError, match="不是合法 JSON"):
        load_output_schema()
