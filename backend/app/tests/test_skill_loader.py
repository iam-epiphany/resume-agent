# -*- coding: utf-8 -*-
"""skill 加载器/注册表测试（2026-08-14）：版本解析、提示词/契约加载、缺失报错、
Registry 发现（2026-08-15）：扫描 */SKILL.md、frontmatter 解析、脚本加载、损坏跳过。"""

from __future__ import annotations

import pytest

from backend.app.services import skill_loader
from backend.app.services.skill_loader import (
    SkillLoadError,
    load_output_schema,
    load_persona_skill_template,
    load_transform_prompt,
    parse_skill_frontmatter,
    skill_version,
)


def test_skill_version_readable() -> None:
    version = skill_version()
    # 语义化版本（至少两段，如 1.0.0），且来自 SKILL.md frontmatter
    assert version.count(".") >= 1
    sk = (skill_loader.skill_root() / "SKILL.md").read_text(encoding="utf-8")
    assert version in sk.split("---", 2)[1]


def test_transform_prompt_has_rules_and_no_placeholders() -> None:
    """System Prompt = 固定指令 + 拼接的业务规则；不含任何用户输入占位符。"""
    prompt = load_transform_prompt()
    assert "简历材料加工师" in prompt
    assert "不可信" in prompt  # prompt injection 防御声明
    assert "忠于来源" in prompt  # transform-rules.md 已拼接
    for placeholder in ("{batch_index}", "{total_batches}", "{input_text}"):
        assert placeholder not in prompt


def test_user_message_template_has_placeholders() -> None:
    """User 消息模板：批次占位符 + 输入占位符 + 不可信声明（材料不进入 system）。"""
    template = skill_loader.load_user_message_template()
    assert "不可信原始材料" in template
    for placeholder in ("{batch_index}", "{total_batches}", "{input_text}"):
        assert placeholder in template


def test_output_schema_is_valid_contract() -> None:
    schema = load_output_schema()
    assert isinstance(schema, dict)
    assert "knowledge_documents" in schema["required"]
    assert schema["$schema"]


def test_persona_skill_template_has_placeholders() -> None:
    """人物 Skill 包模板（1.1.0）：可加载、frontmatter 完整、占位符齐全。"""
    template = load_persona_skill_template()
    assert template.startswith("---")
    for placeholder in (
        "{{package_name}}",
        "{{name}}",
        "{{display_name}}",
        "{{package_version}}",
        "{{generated_at}}",
    ):
        assert placeholder in template
    # 除已声明占位符外不得残留花括号（避免组装后 SKILL.md 出现未替换的模板痕迹）
    remaining = template
    for placeholder in (
        "{{package_name}}",
        "{{name}}",
        "{{display_name}}",
        "{{package_version}}",
        "{{generated_at}}",
    ):
        remaining = remaining.replace(placeholder, "")
    assert "{" not in remaining and "}" not in remaining


def test_missing_persona_skill_template_raises(monkeypatch, tmp_path) -> None:
    skill_dir = tmp_path / "skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: resume-materials-workshop\nmetadata:\n  version: \"1.1.0\"\n---\n正文",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_loader, "skill_root", lambda: skill_dir)
    with pytest.raises(SkillLoadError, match="缺少必需文件"):
        load_persona_skill_template()


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


# ------------------------------------------------------------------ Registry（2026-08-15）


def _write_skill_dir(root, name: str, version: str = "1.0.0", description: str = "") -> None:
    """在 root/name/ 下写一个最小可用 skill（SKILL.md frontmatter 齐全）。"""
    skill_dir = root / name
    (skill_dir / "references").mkdir(parents=True)
    desc = f"\n  {description}\n" if description else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: >-{desc}\nmetadata:\n  version: \"{version}\"\n---\n正文",
        encoding="utf-8",
    )


def test_parse_frontmatter_folded_description() -> None:
    text = (
        "---\n"
        "name: demo-skill\n"
        "description: >-\n"
        "  第一行描述\n"
        "  第二行描述\n"
        "metadata:\n"
        "  version: \"1.2.3\"\n"
        "---\n正文"
    )
    meta = parse_skill_frontmatter(text)
    assert meta["name"] == "demo-skill"
    assert meta["description"] == "第一行描述 第二行描述"
    assert meta["metadata.version"] == "1.2.3"


def test_list_skills_discovers_workshop() -> None:
    """注册表自动发现 resume-materials-workshop，版本与兼容绑定一致。"""
    names = [skill.name for skill in skill_loader.list_skills()]
    assert "resume-materials-workshop" in names
    workshop = skill_loader.get_skill("resume-materials-workshop")
    assert workshop.version == skill_version()
    assert workshop.has_self_test  # capability 检查：scripts/self_test.py 存在


def test_get_skill_missing_raises() -> None:
    with pytest.raises(SkillLoadError, match="未找到 skill"):
        skill_loader.get_skill("no-such-skill")


def test_scan_skips_broken_skills(monkeypatch, tmp_path) -> None:
    """缺 name/version 的 SKILL.md 不进入注册表，其余 skill 不受影响。"""
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    (good / "SKILL.md").write_text(
        '---\nname: good-skill\nmetadata:\n  version: "1.0.0"\n---\n正文', encoding="utf-8"
    )
    (bad / "SKILL.md").write_text("---\nname: bad-skill\n---\n正文", encoding="utf-8")  # 缺 version
    monkeypatch.setattr(skill_loader, "_skills_dir", lambda: tmp_path)
    names = [skill.name for skill in skill_loader.list_skills()]
    assert names == ["good-skill"]


def test_load_script_runs_skill_script(tmp_path) -> None:
    """scripts/ 下的确定性校验脚本可按路径加载并调用。"""
    skill_dir = tmp_path / "s"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "check.py").write_text(
        "def validate(payload, *, allowed):\n"
        "    return [] if payload in allowed else ['不在允许集合']\n",
        encoding="utf-8",
    )
    module = skill_loader.load_script(skill_dir, "scripts/check.py")
    assert module.validate("a", allowed={"a"}) == []
    assert module.validate("b", allowed={"a"}) == ["不在允许集合"]


def test_load_script_missing_raises(tmp_path) -> None:
    with pytest.raises(SkillLoadError, match="缺少必需脚本"):
        skill_loader.load_script(tmp_path, "scripts/check.py")
