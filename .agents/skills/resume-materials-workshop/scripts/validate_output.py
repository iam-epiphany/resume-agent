#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resume-materials-workshop 输出业务契约校验（确定性、零 LLM）。

在 output-schema.json 的「结构校验」之上做「业务契约校验」——把不依赖模型判断、
可确定性验证的规则从提示词下沉到脚本，由后端（skill_loader 加载）与自测共用：

- validate()：批级业务校验（材料主题行/类别枚举/文件名前缀/source_file 有效/
  PII 残留/同名文档冲突）
- reconcile()：跨批次归并（Reduce）——文档去重与冲突重命名、事实去重与冲突标记，
  返回 (documents, facts, conflicts)
- merge_profiles()：多批人物档案归并（标量首值生效、数组并集去重）

用法（后端/评估）：
    import validate_output
    issues = validate_output.validate(payload, allowed_sources=["简历.pdf"])
    docs, facts, conflicts = validate_output.reconcile(docs, facts)

本脚本是 skill 的一部分，禁止 import 后端模块；PII 检测正则与后端
materials_workshop_service._sanitize_privacy 对应（清洗在端侧执行，检测在本脚本）。
"""

from __future__ import annotations

import re
from typing import Any

# 材料主题类别（与 SKILL.md / transform-rules.md 对齐，单一事实来源）
TOPICS = (
    "项目经历",
    "技能专长",
    "教育背景",
    "竞赛奖项",
    "荣誉奖励",
    "证书资格",
    "求职意向",
    "个人特质",
    "自我介绍",
    "综合简历",
)
TOPIC_MARKER = "> 材料主题："

# PII 检测正则（命中 = 输出残留 PII；与后端清洗规则保持一致）
_PII_PATTERNS = (
    re.compile(r"\b\d{17}[\dXx]\b"),  # 18 位身份证
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),  # 大陆手机号
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # 邮箱
    re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|6\d{3}|3[47]\d{2})\d{12,15}\b"),  # 银行卡
    # 家庭住址（保守：省市区 + 路/街/巷/弄/大道 + 门牌号，门牌号前允许空格，
    # 避免误伤一般表述）
    re.compile(
        r"[\u4e00-\u9fa5]{2,}(?:省|市|区|县|自治区|特别行政区)"
        r"[\u4e00-\u9fa5]{1,14}(?:路|街|大道|巷|弄)\s*[0-9一二三四五六七八九十百]+\s*号(?:院|楼|室|幢|栋)?"
    ),
)


def _has_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PII_PATTERNS)


def _topic_of_document(document: dict[str, Any]) -> str:
    content = str(document.get("content") or "")
    first_line = content.splitlines()[0] if content else ""
    if first_line.startswith(TOPIC_MARKER):
        return first_line[len(TOPIC_MARKER):].strip()
    return ""


def _source_matches(source_file: str, allowed_sources: list[str]) -> bool:
    """source_file 与上传文件名匹配：相等，或互为「完整文件名包含」关系。

    LLM 可能把「简历(1).pdf」写为「简历.pdf」，包含关系视为同一来源。
    """
    target = source_file.strip()
    if not target:
        return False
    if target in allowed_sources:
        return True
    for allowed in allowed_sources:
        if target in allowed or allowed in target:
            return True
    return False


def validate(payload: dict[str, Any], *, allowed_sources: list[str] | None = None) -> list[str]:
    """业务契约校验：返回问题列表（空 = 通过）；不抛异常、不修改输入。"""
    issues: list[str] = []
    allowed = [str(name) for name in (allowed_sources or [])]

    persona = payload.get("persona_profile")
    if isinstance(persona, dict):
        for key, value in persona.items():
            if isinstance(value, str) and _has_pii(value):
                issues.append(f"persona_profile.{key} 残留 PII")

    documents = payload.get("knowledge_documents") or []
    if not isinstance(documents, list):
        issues.append("knowledge_documents 必须是数组")
        documents = []
    seen_contents: dict[str, str] = {}
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            issues.append(f"knowledge_documents[{index}] 不是对象")
            continue
        filename = str(document.get("filename") or "")
        content = str(document.get("content") or "")
        topic = _topic_of_document(document)
        if not topic:
            issues.append(f"文档 {filename or index} 首行不是 `> 材料主题：类别`")
        elif topic not in TOPICS:
            issues.append(f"文档 {filename or index} 材料主题类别非法：{topic}")
        elif filename and not (filename.startswith(topic + "_") or filename == f"{topic}.md"):
            issues.append(f"文档 {filename} 文件名前缀与类别「{topic}」不一致")
        if content and _has_pii(content):
            issues.append(f"文档 {filename or index} 内容残留 PII")
        if filename in seen_contents and seen_contents[filename] != content:
            issues.append(f"同名文档内容冲突：{filename}（Reduce 将重命名并标记）")
        seen_contents[filename] = content

    facts = payload.get("facts") or []
    if not isinstance(facts, list):
        issues.append("facts 必须是数组")
        facts = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            issues.append(f"facts[{index}] 不是对象")
            continue
        for field in ("subject", "predicate", "value"):
            if _has_pii(str(fact.get(field) or "")):
                issues.append(f"facts[{index}].{field} 残留 PII")
        source_file = str(fact.get("source_file") or "")
        if allowed and not _source_matches(source_file, allowed):
            issues.append(f"facts[{index}].source_file 不在上传材料中：{source_file}")
        if _has_pii(source_file):
            issues.append(f"facts[{index}].source_file 残留 PII")
    return issues


def _rename_with_suffix(filename: str, count: int) -> str:
    stem, suffix = filename.rsplit(".", 1) if "." in filename else (filename, "md")
    return f"{stem}_{count}.{suffix}"


def reconcile(
    documents: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """跨批次归并（Reduce，确定性）：返回 (documents, facts, conflicts)。

    - 文档：同 filename 同 content 去重；同 content 不同名合并为先出现的名字；
      同 filename 异 content → 重命名加 _2 后缀并记冲突（同名文件重复治理）。
    - 事实：按 (subject, predicate, value) 去重；同 subject+predicate 异 value
      → 全部标 evidence_status=conflict 并记冲突（冲突保留，不做选择）。
    """
    conflicts: list[str] = []

    # ---- 文档 ----
    content_to_name: dict[str, str] = {}  # content → 首个 filename
    name_counts: dict[str, int] = {}      # filename → 已见次数
    merged_docs: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for document in documents:
        filename = str(document.get("filename") or "")
        content = str(document.get("content") or "")
        key = (filename, content)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if content and content in content_to_name:
            if content_to_name[content] != filename:
                conflicts.append(
                    f"重复文档：{filename} 与 {content_to_name[content]} 内容相同，已合并"
                )
            document = {**document, "filename": content_to_name[content]}
        elif filename in name_counts:
            count = name_counts[filename] + 1
            name_counts[filename] = count
            new_name = _rename_with_suffix(filename, count)
            conflicts.append(f"同名文档内容冲突：{filename} 内容不同，已重命名为 {new_name}")
            document = {**document, "filename": new_name}
        else:
            name_counts[filename] = 1
        if content:
            content_to_name.setdefault(content, str(document["filename"]))
        merged_docs.append(document)

    # ---- 事实 ----
    seen_spv: set[tuple[str, str, str]] = set()
    by_sp: dict[tuple[str, str], list[dict[str, Any]]] = {}
    merged_facts: list[dict[str, Any]] = []
    for fact in facts:
        subject = str(fact.get("subject") or "").strip()
        predicate = str(fact.get("predicate") or "").strip()
        value = str(fact.get("value") or "").strip()
        spv_key = (subject, predicate, value)
        if spv_key in seen_spv:
            continue
        seen_spv.add(spv_key)
        merged_facts.append(fact)
        by_sp.setdefault((subject, predicate), []).append(fact)

    for (subject, predicate), entries in by_sp.items():
        if len(entries) < 2:
            continue
        values = sorted({str(entry.get("value") or "") for entry in entries})
        if len(values) > 1:
            for entry in entries:
                entry["evidence_status"] = "conflict"
            conflicts.append(f"事实冲突：{subject}·{predicate} 存在多个值：{' / '.join(values)}")

    return merged_docs, merged_facts, conflicts


def merge_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """多批人物档案归并：标量字段首次出现生效（后批不覆盖）；skills/projects 并集去重。"""
    merged: dict[str, Any] = {}
    list_keys = {"skills", "projects"}
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for key, value in profile.items():
            if key in list_keys and isinstance(value, list):
                existing = merged.setdefault(key, [])
                for item in value:
                    if item not in existing:
                        existing.append(item)
            elif key not in merged:
                merged[key] = value
    return merged
