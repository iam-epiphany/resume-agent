#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resume-materials-workshop 离线自测（无 LLM 依赖）。

校验五件事：
1. references/output-schema.json 本身是合法 JSON Schema（draft 2020-12）
2. assets/sample-output.json（黄金样例）通过契约校验，且每篇 content 以 `> 材料主题：` 开头
3. 负例：若干典型非法输出必须被契约拒绝（缺字段 / 坏文件名 / 非法 status / 类型错误 /
   多余字段 / 材料主题行缺失）
4. scripts/validate_output.py（业务契约校验）可加载：黄金样例通过、PII/来源/主题行
   负例被检出、reconcile 归并（去重/冲突重命名/事实冲突标记）
5. references/persona-skill-template.md（人物 Skill 包模板）存在、frontmatter 完整、
   必需占位符齐全（{{package_name}}/{{name}}/{{display_name}}/{{package_version}}/{{generated_at}}）

用法：python self_test.py
退出码：0 全部通过；1 存在失败。
契约、提示词或模板改动后必须跑通本自测，再提升 SKILL.md 中的 metadata.version。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SKILL_ROOT = Path(__file__).resolve().parents[1]

# 人物 Skill 包模板（1.1.0）：SKILL.md 由该模板填充占位符确定性组装
PERSONA_TEMPLATE_REQUIRED_PLACEHOLDERS = (
    "{{package_name}}",
    "{{name}}",
    "{{display_name}}",
    "{{package_version}}",
    "{{generated_at}}",
)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def load_json(relative: str) -> object:
    return json.loads((SKILL_ROOT / relative).read_text(encoding="utf-8"))


def load_validate_module():
    """按后端同款方式（importlib）加载 scripts/validate_output.py。"""
    path = SKILL_ROOT / "scripts" / "validate_output.py"
    spec = importlib.util.spec_from_file_location("skill_validate_output", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    # 1. 契约本身是合法 JSON Schema
    try:
        schema = load_json("references/output-schema.json")
        Draft202012Validator.check_schema(schema)
        check("output-schema.json 是合法 JSON Schema", True)
    except Exception as exc:  # noqa: BLE001
        check("output-schema.json 是合法 JSON Schema", False, str(exc))
        print("\n自测中止：契约文件损坏，无法继续。")
        return 1

    validator = Draft202012Validator(schema)

    # 2. 黄金样例必须合规
    try:
        sample = load_json("assets/sample-output.json")
        errors = sorted(validator.iter_errors(sample), key=lambda e: list(e.absolute_path))
        check(
            "sample-output.json 通过契约校验",
            not errors,
            errors[0].message if errors else "",
        )
    except Exception as exc:  # noqa: BLE001
        check("sample-output.json 通过契约校验", False, str(exc))

    # 2b. 黄金样例的文档约定：每篇 content 以 `> 材料主题：` 开头
    try:
        sample_docs = sample["knowledge_documents"]
        bad = [d.get("filename") for d in sample_docs if not str(d.get("content", "")).startswith("> 材料主题：")]
        check("黄金样例每篇文档以 `> 材料主题：` 开头", not bad, ", ".join(bad) if bad else "")
    except (KeyError, TypeError):
        check("黄金样例每篇文档以 `> 材料主题：` 开头", False, "样例结构不合契约")

    # 3. 负例：全部必须被拒绝
    negative_cases: dict[str, object] = {
        "缺 knowledge_documents 键": {"persona_profile": {"name": "张三"}},
        "knowledge_documents 非数组": {"knowledge_documents": "not-array"},
        "文档缺 content": {"knowledge_documents": [{"filename": "综合简历_张三.md"}]},
        "文件名含路径分隔符": {"knowledge_documents": [{"filename": "a/b.md", "content": "> 材料主题：综合简历\n\nx"}]},
        "文件名非 .md 后缀": {"knowledge_documents": [{"filename": "简历.docx", "content": "> 材料主题：综合简历\n\nx"}]},
        "content 为空串": {"knowledge_documents": [{"filename": "综合简历_张三.md", "content": ""}]},
        "content 不以材料主题开头": {"knowledge_documents": [{"filename": "综合简历_张三.md", "content": "张三的简介\n\n正文"}]},
        "文档携带多余字段": {"knowledge_documents": [{"filename": "综合简历_张三.md", "content": "> 材料主题：综合简历\n\nx", "extra": 1}]},
        "persona_profile 携带多余字段": {"knowledge_documents": [], "persona_profile": {"name": "张三", "hobby": "篮球"}},
        "顶层携带多余字段": {"knowledge_documents": [], "extra_top": True},
        "事实缺 evidence_status": {"knowledge_documents": [], "facts": [{"subject": "a", "predicate": "b", "value": "c", "source_file": "x"}]},
        "evidence_status 非法值": {"knowledge_documents": [], "facts": [{"subject": "a", "predicate": "b", "value": "c", "evidence_status": "maybe", "source_file": "x"}]},
        "事实缺 source_file": {"knowledge_documents": [], "facts": [{"subject": "a", "predicate": "b", "value": "c", "evidence_status": "explicit"}]},
        "事实携带多余字段": {"knowledge_documents": [], "facts": [{"subject": "a", "predicate": "b", "value": "c", "evidence_status": "explicit", "source_file": "x", "extra": 1}]},
    }
    for name, payload in negative_cases.items():
        check(f"负例被拒绝：{name}", not validator.is_valid(payload))

    # 4. validate_output.py（业务契约校验，与后端共用）
    try:
        validate_mod = load_validate_module()
        check("validate_output.py 可加载", True)
    except Exception as exc:  # noqa: BLE001
        check("validate_output.py 可加载", False, str(exc))
        validate_mod = None

    if validate_mod is not None:
        issues = validate_mod.validate(sample, allowed_sources=["聊天记录节选"])
        check("黄金样例通过业务校验", not issues, "；".join(issues[:3]) if issues else "")

        bad_source = validate_mod.validate(sample, allowed_sources=["不存在.pdf"])
        check("source_file 不在上传材料中被检出", any("source_file" in item for item in bad_source))

        pii_payload = json.loads(
            json.dumps(sample, ensure_ascii=False).replace("800 余人（2 个月）", "13812345678")
        )
        issues = validate_mod.validate(pii_payload, allowed_sources=["聊天记录节选"])
        check("PII 残留被检出", any("PII" in item for item in issues))

        wrong_topic = {
            "knowledge_documents": [
                {"filename": "综合简历_张三.md", "content": "> 材料主题：不存在的类别\n\nx"}
            ],
            "facts": [],
        }
        issues = validate_mod.validate(wrong_topic)
        check("非法类别被检出", any("类别非法" in item for item in issues))

        bad_prefix = {
            "knowledge_documents": [
                {"filename": "技能专长_张三.md", "content": "> 材料主题：项目经历\n\nx"}
            ],
            "facts": [],
        }
        issues = validate_mod.validate(bad_prefix)
        check("文件名前缀与类别不一致被检出", any("不一致" in item for item in issues))

        # reconcile 归并：重复文档合并 / 同名异内容重命名 / 事实冲突标记
        docs = [
            {"filename": "项目经历_A.md", "content": "> 材料主题：项目经历\n\na"},
            {"filename": "项目经历_A.md", "content": "> 材料主题：项目经历\n\na"},
            {"filename": "项目经历_A.md", "content": "> 材料主题：项目经历\n\nb"},
            {"filename": "项目经历_B.md", "content": "> 材料主题：项目经历\n\na"},
        ]
        facts = [
            {"subject": "S", "predicate": "P", "value": "v1", "evidence_status": "explicit", "source_file": "x"},
            {"subject": "S", "predicate": "P", "value": "v1", "evidence_status": "explicit", "source_file": "x"},
            {"subject": "S", "predicate": "P", "value": "v2", "evidence_status": "explicit", "source_file": "x"},
        ]
        merged_docs, merged_facts, conflicts = validate_mod.reconcile(docs, facts)
        check("reconcile 去重同名同内容", len(merged_docs) == 3)
        check(
            "reconcile 同名异内容重命名",
            any("_2" in str(doc["filename"]) for doc in merged_docs),
            str([doc["filename"] for doc in merged_docs]),
        )
        check("reconcile 同内容不同名合并", any("项目经历_B" in item for item in conflicts))
        check("reconcile 事实去重", len(merged_facts) == 2)
        check("reconcile 事实冲突标记", all(fact["evidence_status"] == "conflict" for fact in merged_facts))

        profiles = [{"name": "张三", "skills": ["Java"]}, {"summary": "后端", "skills": ["Java", "Redis"]}]
        merged_profile = validate_mod.merge_profiles(profiles)
        check(
            "merge_profiles 标量首值生效 + 数组并集",
            merged_profile["name"] == "张三"
            and merged_profile["summary"] == "后端"
            and merged_profile["skills"] == ["Java", "Redis"],
        )

    # 5. 人物 Skill 包模板：存在、frontmatter 完整、必需占位符齐全
    template_path = SKILL_ROOT / "references" / "persona-skill-template.md"
    check("persona-skill-template.md 存在", template_path.is_file())
    if template_path.is_file():
        template = template_path.read_text(encoding="utf-8")
        has_frontmatter = template.startswith("---\n") and "\n---" in template.split("\n", 1)[1]
        check("模板含完整 YAML frontmatter", has_frontmatter)
        missing = [
            ph
            for ph in PERSONA_TEMPLATE_REQUIRED_PLACEHOLDERS
            if ph not in template
        ]
        check(
            "模板包含全部必需占位符",
            not missing,
            "缺失: " + ", ".join(missing) if missing else "",
        )
        remaining = template
        for ph in PERSONA_TEMPLATE_REQUIRED_PLACEHOLDERS:
            remaining = remaining.replace(ph, "")
        check(
            "模板无未声明花括号（全部为 {{占位符}}）",
            "{" not in remaining and "}" not in remaining,
        )
    else:
        check("模板含完整 YAML frontmatter", False, "模板缺失")
        check("模板包含全部必需占位符", False, "模板缺失")

    print()
    if failures:
        print(f"自测失败：{len(failures)} 项未通过")
        return 1
    print(
        f"自测通过（契约合法 + 黄金样例合规 + {len(negative_cases)} 个负例全部拒绝 + "
        f"业务校验/reconcile 归并 + 模板检查）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
