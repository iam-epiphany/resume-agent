"""skill 加载器（2026-08-14）：resume-materials-workshop skill 的运行时读取。

后端按「规范驱动」调用 skill：加工提示词、输出 JSON 契约、版本均以
.agents/skills/resume-materials-workshop/ 目录为单一事实来源，规则只改 skill、
不动业务代码。skill 目录不可用时直接报错（可读信息），不静默降级——宁可让
工坊任务失败，也不让无版本、无契约的加工悄悄进行。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.app.core.config import WORKSHOP_SKILL_DIR

# SKILL.md frontmatter 轻量解析（不引 yaml 依赖）：只取 metadata.version
# （version 位于 metadata: 块内，允许前导缩进）
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_VERSION_RE = re.compile(r"^\s*version\s*:\s*[\"']?([^\"'\n]+)", re.MULTILINE)


class SkillLoadError(RuntimeError):
    pass


def skill_root() -> Path:
    path = Path(WORKSHOP_SKILL_DIR)
    if not path.is_dir():
        raise SkillLoadError(
            f"加工 skill 目录不存在：{path}（可配置 WORKSHOP_SKILL_DIR）"
        )
    return path


def _read_required(relative: str) -> str:
    path = skill_root() / relative
    if not path.is_file():
        raise SkillLoadError(f"加工 skill 缺少必需文件：{path}")
    return path.read_text(encoding="utf-8")


def skill_version() -> str:
    """SKILL.md frontmatter 的 metadata.version（唯一版本来源，随任务落库）。"""
    text = _read_required("SKILL.md")
    match = _FRONTMATTER_RE.search(text)
    if match:
        version = _VERSION_RE.search(match.group(1))
        if version:
            return version.group(1).strip()
    raise SkillLoadError("SKILL.md frontmatter 缺少 metadata.version")


def load_transform_prompt() -> str:
    """「材料加工师」System Prompt 全文（含 {batch_index}/{total_batches}/{input_text} 占位符）。"""
    return _read_required("references/transform-prompt.md").strip()


def load_output_schema() -> dict:
    """LLM 输出 JSON 契约（jsonschema 校验用）。"""
    try:
        schema = json.loads(_read_required("references/output-schema.json"))
    except json.JSONDecodeError as exc:
        raise SkillLoadError(f"output-schema.json 不是合法 JSON：{exc}") from exc
    if not isinstance(schema, dict):
        raise SkillLoadError("output-schema.json 顶层必须是 JSON 对象")
    return schema
