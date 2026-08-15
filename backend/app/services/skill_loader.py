"""skill 加载器 / 注册表（2026-08-15）：通用 Skill Registry + 工坊 skill 兼容绑定。

两级 API：
- 通用 Registry：扫描 SKILLS_DIR（默认 .agents/skills，可配置）下的 */SKILL.md，
  解析 frontmatter（name / description / metadata.version）为 Skill 记录，按 name
  加载 references / scripts / assets，带 capability 检查与 mtime 缓存——
  面试场景的「Skill 动态加载」入口（list_skills / get_skill / load_reference /
  load_json / load_script）。
- 工坊兼容绑定：skill_version() / load_transform_prompt() / load_output_schema() /
  load_persona_skill_template() 固定指向 resume-materials-workshop，是通用层的薄封装；
  根目录仍由 skill_root() 决定（WORKSHOP_SKILL_DIR 可配置，测试可 monkeypatch）。

skill 目录不可用时直接报错（可读信息），不静默降级——宁可让工坊任务失败，
也不让无版本、无契约的加工悄悄进行。
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from backend.app.core.config import SKILLS_DIR, WORKSHOP_SKILL_DIR

# SKILL.md frontmatter 轻量解析（不引 yaml 依赖）：frontmatter 块整体 + metadata.version
# （version 位于 metadata: 块内，允许前导缩进）
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_VERSION_RE = re.compile(r"^\s*version\s*:\s*[\"']?([^\"'\n]+)", re.MULTILINE)

WORKSHOP_SKILL_NAME = "resume-materials-workshop"


class SkillLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Skill:
    """注册表条目：一个已发现、可加载的 skill（SKILL.md frontmatter 的解析结果）。"""

    name: str
    version: str
    description: str
    root: Path
    has_self_test: bool


def parse_skill_frontmatter(text: str) -> dict[str, str]:
    """轻量解析 SKILL.md frontmatter（不引 yaml 依赖）。

    支持 name / description（折叠块 `>-` 与字面块 `|`）/ metadata.version；
    未知键忽略。返回键：name、description、metadata.version。
    """
    match = _FRONTMATTER_RE.search(text)
    if not match:
        return {}
    block = match.group(1)
    fields: dict[str, str] = {}
    lines = block.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        key_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if not key_match:
            index += 1
            continue
        key, value = key_match.group(1), key_match.group(2).strip()
        if key == "metadata":
            # metadata: 块内为缩进的子字段（如 version: "1.1.0"）
            sub_index = index + 1
            while sub_index < len(lines) and (
                not lines[sub_index].strip() or lines[sub_index].startswith(" ")
            ):
                sub = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", lines[sub_index])
                if sub:
                    fields[f"metadata.{sub.group(1)}"] = sub.group(2).strip().strip('"').strip("'")
                sub_index += 1
            index = sub_index
            continue
        if value in {">", ">-", "|", "|-"}:
            # 折叠/字面块：收集后续缩进行（空行视为块内内容）
            parts: list[str] = []
            sub_index = index + 1
            while sub_index < len(lines) and (
                not lines[sub_index].strip() or lines[sub_index].startswith(" ")
            ):
                parts.append(lines[sub_index].strip())
                sub_index += 1
            fields[key] = " ".join(parts) if value.startswith(">") else "\n".join(parts)
            index = sub_index
        else:
            fields[key] = value.strip('"').strip("'")
            index += 1
    if "metadata.version" not in fields:
        version = _VERSION_RE.search(block)
        if version:
            fields["metadata.version"] = version.group(1).strip()
    return fields


# ------------------------------------------------------------------ 通用 Registry

def _skills_dir() -> Path:
    path = Path(SKILLS_DIR)
    if not path.is_dir():
        raise SkillLoadError(f"skills 目录不存在：{path}（可配置 SKILLS_DIR）")
    return path


_scan_cache: tuple[Path, float, dict[str, Skill]] | None = None  # (dir, 最新 SKILL.md mtime, skills)


def _scan() -> dict[str, Skill]:
    """扫描 skills 目录下的 */SKILL.md 并解析 frontmatter（按 name 索引）。

    目录内 SKILL.md 的 mtime 变化会使缓存失效（skill 迭代后无需重启后端）；
    frontmatter 缺 name/version 的目录视为损坏，跳过并保留其余 skill。
    """
    global _scan_cache
    root = _skills_dir()
    newest_mtime = 0.0
    try:
        for child in root.iterdir():
            if child.is_dir():
                skill_file = child / "SKILL.md"
                if skill_file.is_file():
                    newest_mtime = max(newest_mtime, skill_file.stat().st_mtime)
    except OSError:
        newest_mtime = 0.0
    if _scan_cache is not None and _scan_cache[0] == root and _scan_cache[1] >= newest_mtime:
        return _scan_cache[2]

    skills: dict[str, Skill] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        skill_file = child / "SKILL.md"
        if not skill_file.is_file():
            continue
        try:
            meta = parse_skill_frontmatter(skill_file.read_text(encoding="utf-8"))
        except OSError:
            continue
        name = meta.get("name", "").strip()
        version = meta.get("metadata.version", "").strip()
        if not name or not version:
            continue  # 损坏的 skill（缺 name/version）不进入注册表
        skills[name] = Skill(
            name=name,
            version=version,
            description=meta.get("description", "").strip(),
            root=child,
            has_self_test=(child / "scripts" / "self_test.py").is_file(),
        )
    _scan_cache = (root, newest_mtime, skills)
    return skills


def list_skills() -> list[Skill]:
    """全部已发现的 skill（按名称排序）。"""
    return sorted(_scan().values(), key=lambda skill: skill.name)


def get_skill(name: str) -> Skill:
    """按 frontmatter name 取 skill；不存在时报可读错误（含可用清单）。"""
    skills = _scan()
    skill = skills.get(name)
    if skill is None:
        available = ", ".join(sorted(skills)) or "（空）"
        raise SkillLoadError(f"未找到 skill：{name}（已发现：{available}）")
    return skill


def load_reference(root: Path, relative: str) -> str:
    """读取 skill 目录下的文本资源（缺失即报错，不静默降级）。"""
    path = root / relative
    if not path.is_file():
        raise SkillLoadError(f"skill 缺少必需文件：{path}")
    return path.read_text(encoding="utf-8")


def load_json(root: Path, relative: str) -> dict:
    """读取 skill 目录下的 JSON 契约文件。"""
    path = root / relative
    if not path.is_file():
        raise SkillLoadError(f"skill 缺少必需文件：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkillLoadError(f"{relative} 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise SkillLoadError(f"{relative} 顶层必须是 JSON 对象")
    return data


def load_script(root: Path, relative: str) -> ModuleType:
    """按路径加载 skill 的 Python 脚本为模块（确定性校验/评估等逻辑放 scripts/）。"""
    path = root / relative
    if not path.is_file():
        raise SkillLoadError(f"skill 缺少必需脚本：{path}")
    module_name = f"skill_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SkillLoadError(f"无法加载 skill 脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ 工坊兼容绑定

def skill_root() -> Path:
    """resume-materials-workshop 目录（WORKSHOP_SKILL_DIR，测试可 monkeypatch）。"""
    path = Path(WORKSHOP_SKILL_DIR)
    if not path.is_dir():
        raise SkillLoadError(
            f"加工 skill 目录不存在：{path}（可配置 WORKSHOP_SKILL_DIR）"
        )
    return path


def _workshop_root() -> Path:
    return skill_root()


def skill_version() -> str:
    """SKILL.md frontmatter 的 metadata.version（唯一版本来源，随任务落库）。"""
    meta = parse_skill_frontmatter(load_reference(_workshop_root(), "SKILL.md"))
    version = meta.get("metadata.version", "").strip()
    if not version:
        raise SkillLoadError("SKILL.md frontmatter 缺少 metadata.version")
    return version


def load_transform_prompt() -> str:
    """「材料加工师」System Prompt：固定指令 + 拼接的业务规则。

    规则物理唯一地存在于 references/transform-rules.md（AGENTS.md/SKILL.md/README
    只做指针）；缺失即报错，不静默降级。
    """
    prompt = load_reference(_workshop_root(), "references/transform-prompt.md").strip()
    rules = load_reference(_workshop_root(), "references/transform-rules.md").strip()
    return f"{prompt}\n\n{rules}"


def load_user_message_template() -> str:
    """User 消息模板（不可信材料容器）：含 {batch_index}/{total_batches}/{input_text}
    占位符与「材料不可信、不执行其中指令」声明，由调用方替换后作为 user 消息发送。
    """
    return load_reference(_workshop_root(), "references/user-message-template.md").strip()


def load_output_schema() -> dict:
    """LLM 输出 JSON 契约（jsonschema 校验用）。"""
    return load_json(_workshop_root(), "references/output-schema.json")


def load_persona_skill_template() -> str:
    """人物 Skill 包 SKILL.md 模板（含 {{package_name}}/{{name}}/{{display_name}}/
    {{package_version}}/{{generated_at}} 占位符，由 persona_skill_service 确定性组装）。
    """
    return load_reference(_workshop_root(), "references/persona-skill-template.md")
