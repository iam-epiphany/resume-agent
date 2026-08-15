"""人物 Skill 包服务（2026-08-15）：把人物工坊的加工产物封装为可独立调用的人物 Skill。

产物结构（zip 下载，解压放入任意 Agent 的 skills 目录即可让 AI 按本人材料作答）：

```
persona-{人名}/
├── SKILL.md            # 回答规则 + 渐进式加载（references/persona-skill-template.md 组装）
├── facts.json          # 该人物全部事实台账（去重）
└── references/
    ├── profile.md      # 人物档案渲染（persona_profile → Markdown）
    ├── projects/       # 「项目经历」类知识文档（一项目一篇）
    └── *.md            # 其余主题文档（技能/教育/荣誉/证书/自我介绍…）
```

组装是「确定性模板 + 数据库查询」，零额外 LLM 调用、不改动加工输出契约；SKILL.md
模板以工坊 skill 目录为单一事实来源（skill_loader.load_persona_skill_template），
包版本 = 加工 skill 版本（metadata.version）。模板缺失时跳过生成（best-effort，
告警不阻断主流程——人物 Skill 包是辅助产物，LLM 契约校验才是 fail-closed 的硬约束）。
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import Document, FactLedger, Persona, now_utc
from backend.app.services import skill_loader
from backend.app.services.document_lifecycle_service import resolve_original_path

logger = logging.getLogger(__name__)

# 知识文档主题标记（输出契约强制首行，组装时据此归类）
TOPIC_MARKER = "> 材料主题："
PROJECT_TOPIC = "项目经历"

# SKILL.md 模板占位符（与 references/persona-skill-template.md 对应）
_PLACEHOLDERS = (
    "{{package_name}}",
    "{{name}}",
    "{{display_name}}",
    "{{package_version}}",
    "{{generated_at}}",
)

# facts.json 导出字段（台账列 → 包字段；不导内部 fact_id；status 拆为证据/审核两维）
_FACT_FIELDS = (
    "subject",
    "predicate",
    "value",
    "unit",
    "evidence_status",
    "review_status",
    "source_file",
    "source_type",
)


class PersonaSkillPackageError(RuntimeError):
    pass


def package_dir_name(persona: Persona) -> str:
    """人物 Skill 目录名：persona-{人名}（去掉 persona_id 前缀、清洗非法字符）。"""
    persona_id = persona.persona_id
    stem = persona_id[len("persona-") :] if persona_id.startswith("persona-") else persona_id
    stem = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", stem).strip("_")
    return f"persona-{stem or 'persona'}"


def build_persona_skill_package(db: Session, persona: Persona) -> dict[str, Any] | None:
    """组装人物 Skill 包（best-effort）：模板缺失/损坏返回 None，不抛错。

    返回 {"dir_name", "skill_version", "generated_at", "files": {相对路径: 内容}}。
    """
    try:
        template = skill_loader.load_persona_skill_template()
        skill_version = skill_loader.skill_version()
    except skill_loader.SkillLoadError as exc:
        logger.warning("人物 Skill 包跳过生成：%s", exc)
        return None

    dir_name = package_dir_name(persona)
    profile = _persona_profile_dict(persona)
    name = str(profile.get("name") or persona.name or "").strip()
    display_name = str(
        profile.get("display_name") or persona.display_name or name or "求职者"
    ).strip()

    files: dict[str, str] = {}
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    skill_md = (
        template.replace("{{package_name}}", dir_name)
        .replace("{{name}}", name or display_name)
        .replace("{{display_name}}", display_name)
        .replace("{{package_version}}", skill_version)
        .replace("{{generated_at}}", generated_at)
    )
    files[f"{dir_name}/SKILL.md"] = skill_md

    used_paths: set[str] = set()
    for filename, content in _persona_knowledge_documents(db, persona.persona_id):
        topic = _document_topic(content)
        relative = _package_document_path(dir_name, _safe_package_filename(filename), topic)
        if relative in used_paths:
            # 清洗后重名（如 a/b.md 与 a\b.md 都变 a_b.md）或原始同名文档：
            # 追加 _2/_3 后缀，避免 zip 内互相覆盖（静默丢内容）
            relative = _unique_package_path(relative, used_paths)
        used_paths.add(relative)
        files[relative] = content

    files[f"{dir_name}/references/profile.md"] = _render_profile_md(profile)
    files[f"{dir_name}/facts.json"] = json.dumps(
        _persona_facts(db, persona.persona_id), ensure_ascii=False, indent=2
    )

    return {
        "dir_name": dir_name,
        "skill_version": skill_version,
        "generated_at": generated_at,
        "files": files,
    }


def regenerate_persona_skill(db: Session, persona_id: str) -> bool:
    """重建并落库人物 Skill 包（best-effort）：模板缺失/异常时告警并返回 False。"""
    try:
        persona = db.scalar(select(Persona).where(Persona.persona_id == persona_id))
        if persona is None:
            logger.warning("人物 Skill 包重建跳过：人物不存在 %s", persona_id)
            return False
        package = build_persona_skill_package(db, persona)
        if package is None:
            return False
        persona.skill_package_json = json.dumps(package, ensure_ascii=False)
        persona.skill_package_updated_at = now_utc()
        db.commit()
        logger.info(
            "人物 Skill 包已重建：%s（%d 个文件，规范版本 %s）",
            package["dir_name"],
            len(package["files"]),
            package["skill_version"],
        )
        return True
    except Exception as exc:  # noqa: BLE001 — 辅助产物，任何异常都不阻断主流程
        logger.warning("人物 Skill 包重建失败（跳过）：%s", exc)
        db.rollback()
        return False


def skill_package_info(persona: Persona) -> dict[str, Any] | None:
    """包元信息（不泄露内容，供前端展示下载按钮）：文件数/规范版本/生成时间。"""
    package = load_persona_skill_package(persona)
    if package is None:
        return None
    return {
        "file_count": len(package.get("files") or {}),
        "skill_version": package.get("skill_version"),
        "generated_at": package.get("generated_at"),
    }


def load_persona_skill_package(persona: Persona) -> dict[str, Any] | None:
    """读取已落库的包（损坏/为空返回 None）。"""
    try:
        package = json.loads(persona.skill_package_json) if persona.skill_package_json else None
    except (json.JSONDecodeError, TypeError):
        package = None
    return package if isinstance(package, dict) else None


def persona_skill_zip_bytes(package: dict[str, Any]) -> bytes:
    """把包序列化为内存 zip（顶层目录内为 SKILL.md/facts.json/references/）。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, content in (package.get("files") or {}).items():
            archive.writestr(relative, content)
    return buffer.getvalue()


def _package_document_path(dir_name: str, filename: str, topic: str) -> str:
    if topic == PROJECT_TOPIC:
        return f"{dir_name}/references/projects/{filename}"
    return f"{dir_name}/references/{filename}"


def _safe_package_filename(filename: str) -> str:
    """包内文件名清洗：路径分隔符转下划线、剔除其余非法字符，杜绝 zip 路径穿越。

    zip-slip 防线：文档文件名来自上传（可能带 ../ 或分隔符），清洗后任何
    相对路径逃逸（..、/、\）都不可能进入 zip 条目路径。
    """
    cleaned = re.sub(r"[\\/]", "_", filename)
    # strip 掐掉前导/尾随点（防 ".." 相对路径逃逸）与非法字符
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.\-]", "_", cleaned).strip("_.")
    return (cleaned or "材料.md")[:255]


def _unique_package_path(path: str, used: set[str]) -> str:
    """包内路径冲突时追加 _2/_3 后缀（同名文档互不覆盖）。"""
    stem, suffix = path.rsplit(".", 1) if "." in path else (path, "")
    index = 2
    while True:
        candidate = f"{stem}_{index}.{suffix}" if suffix else f"{stem}_{index}"
        if candidate not in used:
            return candidate
        index += 1


def _persona_profile_dict(persona: Persona) -> dict[str, Any]:
    try:
        profile = json.loads(persona.profile_json) if persona.profile_json else {}
    except (json.JSONDecodeError, TypeError):
        profile = {}
    return profile if isinstance(profile, dict) else {}
def _persona_knowledge_documents(db: Session, persona_id: str) -> list[tuple[str, str]]:
    """该人物名下全部知识文档（文件名, 内容）：仅加工产物（`> 材料主题：` 开头），排除原始上传。"""
    documents = db.scalars(
        select(Document)
        .where(Document.persona_id == persona_id)
        .order_by(Document.uploaded_at.asc(), Document.id.asc())
    ).all()
    included: list[tuple[str, str]] = []
    for doc in documents:
        content = _read_document_content(doc)
        if content is None or not content.startswith(TOPIC_MARKER):
            continue
        included.append((doc.filename, content))
    return included


def _read_document_content(doc: Document) -> str | None:
    try:
        path = resolve_original_path(doc)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — 单篇读取失败跳过，不影响其余文档
        logger.warning("人物 Skill 包跳过文档 %s（读取失败）：%s", doc.filename, exc)
        return None


def _document_topic(content: str) -> str:
    first_line = content.splitlines()[0] if content else ""
    if first_line.startswith(TOPIC_MARKER):
        return first_line[len(TOPIC_MARKER) :].strip()
    return ""


def _persona_facts(db: Session, persona_id: str) -> list[dict[str, str]]:
    facts = db.scalars(
        select(FactLedger)
        .where(FactLedger.persona_id == persona_id)
        .order_by(FactLedger.id.asc())
    ).all()
    seen: set[tuple[str, ...]] = set()
    exported: list[dict[str, str]] = []
    for fact in facts:
        row = {
            field: str(getattr(fact, field))
            for field in _FACT_FIELDS
            if getattr(fact, field) is not None
        }
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        exported.append(row)
    return exported


def _render_profile_md(profile: dict[str, Any]) -> str:
    """人物档案 → references/profile.md（只渲染存在的字段，不补写）。"""
    name = str(profile.get("name") or "求职者")
    display_name = str(profile.get("display_name") or name)
    lines = [f"# {display_name} · 人物档案", f"> {TOPIC_MARKER}综合简历", ""]
    for label, key in (("姓名", "name"), ("摘要", "summary"), ("教育背景", "education"), ("求职意向", "job_intent")):
        value = profile.get(key)
        if value:
            lines.append(f"- {label}：{value}")
    skills = profile.get("skills")
    if isinstance(skills, list) and skills:
        lines += ["", "## 技能专长"] + [f"- {item}" for item in map(str, skills)]
    projects = profile.get("projects")
    if isinstance(projects, list) and projects:
        lines += ["", "## 项目经历"] + [f"- {item}" for item in map(str, projects)]
    return "\n".join(lines) + "\n"
