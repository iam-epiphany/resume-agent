#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resume-materials-workshop 离线自测（无 LLM 依赖）。

校验三件事：
1. references/output-schema.json 本身是合法 JSON Schema（draft 2020-12）
2. assets/sample-output.json（黄金样例）通过契约校验，且每篇 content 以 `> 材料主题：` 开头
3. 负例：若干典型非法输出必须被契约拒绝（缺字段 / 坏文件名 / 非法 status / 类型错误）

用法：python self_test.py
退出码：0 全部通过；1 存在失败。
契约或提示词改动后必须跑通本自测，再提升 SKILL.md 中的 metadata.version。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SKILL_ROOT = Path(__file__).resolve().parents[1]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def load_json(relative: str) -> object:
    return json.loads((SKILL_ROOT / relative).read_text(encoding="utf-8"))


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
        "事实缺 status": {"knowledge_documents": [], "facts": [{"subject": "a", "predicate": "b", "value": "c"}]},
        "status 非法值": {"knowledge_documents": [], "facts": [{"subject": "a", "predicate": "b", "value": "c", "status": "maybe"}]},
    }
    for name, payload in negative_cases.items():
        check(f"负例被拒绝：{name}", not validator.is_valid(payload))

    print()
    if failures:
        print(f"自测失败：{len(failures)} 项未通过")
        return 1
    print(f"自测通过（契约合法 + 黄金样例合规 + {len(negative_cases)} 个负例全部拒绝）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
