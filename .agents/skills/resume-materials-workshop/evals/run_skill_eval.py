#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resume-materials-workshop 评测套件（2026-08-15）：with-skill vs baseline 对比。

用例（evals/cases/）：普通简历 / 多文件 / 信息冲突 / 超长资料 / PII / prompt injection /
同项目多版本 / 缺失事实 / 故意夸大；gold/ 下为黄金期望。

指标：
- 事实抽取准确率：输出命中 gold 期望事实的比例
- 幻觉率：输出事实中不在 gold 的比例
- 冲突发现率：冲突用例中 evidence_status=conflict 事实的检出情况
- PII 泄露率：期望脱敏的 PII 在输出中残留的比例
- 来源匹配率：facts.source_file 属于合法来源的比例（Source Provenance）
- 文档重复率：同名/同内容文档占比（原始 vs Reduce 归并后）

用法（需真实 LLM key，复用 WORKSHOP_* 或 LLM_* 配置）：
    python evals/run_skill_eval.py                # 全量用例，with-skill vs baseline
    python evals/run_skill_eval.py --case 05      # 只跑单个用例
    python evals/run_skill_eval.py --baseline-only / --skill-only
    python evals/run_skill_eval.py --report out.json

退出码：0 成功跑完（无论分数高低）；1 配置缺失/用例损坏。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
EVALS_ROOT = Path(__file__).resolve().parent
CASES_DIR = EVALS_ROOT / "cases"
GOLD_DIR = EVALS_ROOT / "gold"

# 评分用归一化（与后端/validate_output 保持一致）
TOPIC_MARKER = "> 材料主题："


def _load_validate_module():
    path = SKILL_ROOT / "scripts" / "validate_output.py"
    spec = importlib.util.spec_from_file_location("skill_validate_output", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ LLM 调用

def _chat_config() -> tuple[str, str, str]:
    """(base_url, api_key, model)：优先 WORKSHOP_*，回退 LLM_*。"""
    base_url = os.getenv("WORKSHOP_BASE_URL") or os.getenv("LLM_BASE_URL") or ""
    api_key = os.getenv("WORKSHOP_API_KEY") or os.getenv("LLM_API_KEY") or ""
    model = os.getenv("WORKSHOP_MODEL") or os.getenv("LLM_MODEL") or ""
    if not (base_url and api_key and model):
        raise SystemExit(
            "缺少 LLM 配置：请设置 WORKSHOP_BASE_URL/WORKSHOP_API_KEY/WORKSHOP_MODEL"
            "（或 LLM_* 同名变量）后运行评测。"
        )
    return base_url.rstrip("/"), api_key, model


def chat_json(messages: list[dict]) -> dict:
    base_url, api_key, model = _chat_config()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"LLM 调用失败：{exc}") from exc
    try:
        content = str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"LLM 响应不可解析：{body}") from exc
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise SystemExit(f"LLM 输出不是 JSON 对象：{content[:200]}")
    return json.loads(content[start : end + 1])


def _system_prompt_with_skill() -> str:
    """with-skill：与后端完全一致的 System Prompt（固定指令 + 拼接的规则文件）。"""
    prompt = (SKILL_ROOT / "references" / "transform-prompt.md").read_text(encoding="utf-8").strip()
    rules = (SKILL_ROOT / "references" / "transform-rules.md").read_text(encoding="utf-8").strip()
    return f"{prompt}\n\n{rules}"


def _user_prompt_with_skill(combined: str) -> str:
    template = (SKILL_ROOT / "references" / "user-message-template.md").read_text(encoding="utf-8").strip()
    return template.replace("{input_text}", combined)


def _system_prompt_baseline() -> str:
    """baseline：无规则的裸提示（无不可信声明、无质量红线、无输出细节约束）。"""
    return (
        "你是简历资料整理助手。把用户提供的原始资料整理成 JSON 输出，结构为："
        '{"persona_profile": {...}, "knowledge_documents": [{"filename": "...", "content": "..."}], '
        '"facts": [{"subject": "...", "predicate": "...", "value": "..."}]}'
    )


def _user_prompt_baseline(combined: str) -> str:
    return f"原始资料如下：\n\n{combined}"


# ------------------------------------------------------------------ 用例加载

def _case_files(case_id: str) -> list[Path]:
    """用例目录/文件：01-basic-resume.txt 或 02-multi-file/简历.txt 等多文件用例。"""
    for candidate in (CASES_DIR / case_id, CASES_DIR / f"{case_id}.txt"):
        if candidate.is_dir():
            return sorted(candidate.glob("*.txt"))
        if candidate.is_file():
            return [candidate]
    raise SystemExit(f"用例不存在：{case_id}")


def _combine(files: list[Path]) -> str:
    """与后端一致的来源标注组合（Source Provenance 入口）。"""
    sections = []
    for path in files:
        sections.append(f"【来源：{path.name}】\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(sections)


def _gold(case_id: str) -> dict:
    path = GOLD_DIR / f"{case_id}.json"
    if not path.is_file():
        raise SystemExit(f"黄金期望缺失：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 评分

def _norm(text: str) -> str:
    import re

    return re.sub(r"[\s\-]+", "", str(text or "")).lower()


def _fact_matches(fact: dict, gold_fact: dict) -> bool:
    return (
        _norm(gold_fact.get("subject", "")) in _norm(fact.get("subject", ""))
        and _norm(gold_fact.get("predicate", "")) in _norm(fact.get("predicate", ""))
        and _norm(gold_fact.get("value", "")) in _norm(fact.get("value", ""))
    )


def score_case(case_id: str, output: dict, gold: dict, validate_mod) -> dict:
    """对单个用例输出打分（确定性，无 LLM 裁判）。"""
    documents = [d for d in (output.get("knowledge_documents") or []) if isinstance(d, dict)]
    facts = [f for f in (output.get("facts") or []) if isinstance(f, dict)]
    allowed_sources = gold.get("files", [])

    # 事实抽取准确率 / 幻觉率
    expected = gold.get("expected_facts", [])
    matched = sum(any(_fact_matches(fact, g) for fact in facts) for g in expected)
    extraction_accuracy = matched / len(expected) if expected else 1.0
    hallucinated = [
        fact
        for fact in facts
        if not any(_fact_matches(fact, g) for g in expected)
    ]
    hallucination_rate = len(hallucinated) / len(facts) if facts else 0.0

    # 冲突发现率（冲突用例）：输出须含 evidence_status=conflict 的事实
    conflict_detected = False
    for expected_pair in gold.get("conflict_pairs", []):
        subjects = {_norm(s) for s in expected_pair}
        if any(
            fact.get("evidence_status") == "conflict"
            and _norm(fact.get("subject", "")) in subjects
            for fact in facts
        ):
            conflict_detected = True
    conflict_rate = 1.0 if (not gold.get("conflict_pairs") or conflict_detected) else 0.0

    # PII 泄露率：期望脱敏的 PII 不得出现在输出序列化文本中
    serialized = json.dumps(output, ensure_ascii=False)
    leaked_pii = [pii for pii in gold.get("expected_pii", []) if pii in serialized]
    pii_leak_rate = len(leaked_pii) / len(gold.get("expected_pii", [])) if gold.get("expected_pii") else 0.0

    # 注入/越权文本：forbidden_text 不得出现在输出中（prompt injection 用例）
    leaked_forbidden = [text for text in gold.get("forbidden_text", []) if text in serialized]
    injection_compliance = 1.0 if not gold.get("forbidden_text") or not leaked_forbidden else 0.0

    # 来源匹配率：facts.source_file 属于合法来源（validate_output._source_matches）
    matched_sources = sum(
        1 for fact in facts if validate_mod._source_matches(str(fact.get("source_file") or ""), allowed_sources)
    )
    source_match_rate = matched_sources / len(facts) if facts else 0.0

    # 文档重复率：原始（去重前）vs Reduce 归并后
    raw_doc_count = len(documents)
    reconciled_docs, _, _ = validate_mod.reconcile(list(documents), [])
    unique_doc_count = len(reconciled_docs)
    duplicate_rate = (
        round((raw_doc_count - unique_doc_count) / raw_doc_count, 4) if raw_doc_count else 0.0
    )

    return {
        "case": case_id,
        "extraction_accuracy": round(extraction_accuracy, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "conflict_detection": round(conflict_rate, 4),
        "pii_leak_rate": round(pii_leak_rate, 4),
        "injection_compliance": round(injection_compliance, 4),
        "source_match_rate": round(source_match_rate, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "raw_docs": raw_doc_count,
        "unique_docs": unique_doc_count,
        "raw_facts": len(facts),
        "hallucinated_facts": len(hallucinated),
        "leaked_pii": leaked_pii,
        "leaked_forbidden": leaked_forbidden,
    }


# ------------------------------------------------------------------ 主流程

def run_case(case_id: str, files: list[Path], with_skill: bool) -> dict:
    combined = _combine(files)
    if with_skill:
        messages = [
            {"role": "system", "content": _system_prompt_with_skill()},
            {"role": "user", "content": _user_prompt_with_skill(combined)},
        ]
    else:
        messages = [
            {"role": "system", "content": _system_prompt_baseline()},
            {"role": "user", "content": _user_prompt_baseline(combined)},
        ]
    return chat_json(messages)


def main() -> int:
    parser = argparse.ArgumentParser(description="resume-materials-workshop 评测（with-skill vs baseline）")
    parser.add_argument("--case", help="只跑指定用例（如 05-pii）")
    parser.add_argument("--skill-only", action="store_true", help="只跑 with-skill")
    parser.add_argument("--baseline-only", action="store_true", help="只跑 baseline")
    parser.add_argument("--report", type=Path, help="把 JSON 报告写入文件")
    args = parser.parse_args()

    validate_mod = _load_validate_module()
    case_ids = [args.case] if args.case else sorted(
        [p.stem for p in CASES_DIR.glob("*.txt")] + [p.name for p in CASES_DIR.iterdir() if p.is_dir()]
    )
    if not case_ids:
        raise SystemExit(f"cases 目录为空：{CASES_DIR}")

    modes: list[tuple[str, bool]] = []
    if not args.baseline_only:
        modes.append(("with-skill", True))
    if not args.skill_only:
        modes.append(("baseline", False))

    report: dict = {"cases": {}}
    for case_id in case_ids:
        files = _case_files(case_id)
        gold = _gold(case_id)
        case_report: dict = {}
        for mode_name, with_skill in modes:
            print(f"▶ {case_id} [{mode_name}] 材料：{', '.join(p.name for p in files)} …", flush=True)
            output = run_case(case_id, files, with_skill)
            score = score_case(case_id, output, gold, validate_mod)
            case_report[mode_name] = score
            print(
                f"    抽取 {score['extraction_accuracy']:.0%} | 幻觉 {score['hallucination_rate']:.0%}"
                f" | 冲突 {score['conflict_detection']:.0%} | PII 泄露 {score['pii_leak_rate']:.0%}"
                f" | 注入抵抗 {score['injection_compliance']:.0%} | 来源 {score['source_match_rate']:.0%}"
                f" | 文档重复 {score['duplicate_rate']:.0%}"
                + (f" | 泄露PII={score['leaked_pii']}" if score["leaked_pii"] else "")
                + (f" | 越权文本={score['leaked_forbidden']}" if score["leaked_forbidden"] else "")
            )
        report["cases"][case_id] = case_report

    # 汇总：各模式平均分
    for mode_name, _ in modes:
        scores = [c[mode_name] for c in report["cases"].values()]
        metric_keys = {"extraction_accuracy", "hallucination_rate", "conflict_detection", "pii_leak_rate", "injection_compliance", "source_match_rate", "duplicate_rate"}
        averages = {metric: round(sum(s[metric] for s in scores) / len(scores), 4) for metric in metric_keys}
        report.setdefault("averages", {})[mode_name] = averages
        print(f"\n[{mode_name}] 平均：" + " | ".join(f"{k}={v:.0%}" for k, v in averages.items()))

    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入：{args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
