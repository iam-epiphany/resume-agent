"""关键事实台账服务（2026-08-14）：生成后「实体—属性—值」错配校验（零 LLM）。

解决 presence-only 校验的盲区：数字/日期在证据全文出现 ≠ 归属于正确对象。
台账以 fact_id 组织 subject/predicate/value/evidence_status/review_status/source，
生成后检查答案中的事实归属关系：
- A 实体的值出现在 B 实体附近（张冠李戴）→ mismatch
- 值出现但归属实体未出现、且同属性的其他实体出现 → mismatch
- 非 explicit 事实（missing/inferred）被直接引用 → warning（回答者引用了待确认信息）
- conflict 事实仅存档可见，不参与值校验（保留冲突不做选择）

2026-08-15 状态语义拆分：evidence_status（explicit/inferred/conflict/missing）
= 事实可信度；review_status（pending/approved/rejected）= 人工审核状态。
旧 status 列保留为兼容镜像（confirmed/pending/inferred/conflict），新代码写
evidence_status/review_status。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import FactLedger

# 旧 status 词表 → 新 evidence_status 词表（种子数据/存量兼容）
LEGACY_STATUS_TO_EVIDENCE = {
    "confirmed": "explicit",
    "pending": "missing",
    "inferred": "inferred",
    "conflict": "conflict",
}
# 新 evidence_status → 旧 status 镜像（保持旧列可读，供任何遗留消费方）
EVIDENCE_TO_LEGACY_STATUS = {
    "explicit": "confirmed",
    "missing": "pending",
    "inferred": "inferred",
    "conflict": "conflict",
}


@dataclass
class FactCheckResult:
    checked_facts: int = 0
    mismatches: list[str] = field(default_factory=list)      # 硬错配：必须降级 hedged
    warnings: list[str] = field(default_factory=list)        # 软提示：仅观测

    @property
    def verified(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_facts": self.checked_facts,
            "mismatches": list(self.mismatches),
            "warnings": list(self.warnings),
            "verified": self.verified,
        }


def load_fact_ledger(db: Session, persona_id: str | None = None) -> list[FactLedger]:
    """读取台账事实（台账规模小，逐请求读取即可）。

    persona_id 不为空时只读该人物的台账（多人物隔离，2026-08-14）。
    """
    try:
        statement = select(FactLedger).order_by(FactLedger.id.asc())
        if persona_id:
            statement = statement.where(FactLedger.persona_id == persona_id)
        return list(db.scalars(statement).all())
    except Exception:
        return []


def fact_status_for_source(source_file: str | None, facts: list[FactLedger]) -> str | None:
    """来源文件的台账事实证据状态（供公开出处标注）。

    explicit=该文件只有 explicit 事实；mixed=含 missing/inferred/conflict 事实；
    None=该文件未关联任何台账事实。审核状态（review_status）不参与本标注。
    """
    if not source_file:
        return None
    evidence = {fact.evidence_status for fact in facts if fact.source_file == source_file}
    if not evidence:
        return None
    if evidence <= {"explicit"}:
        return "explicit"
    return "mixed"


def seed_fact_records(db: Session, records: Iterable[dict[str, Any]]) -> int:
    """按 fact_id 幂等 upsert（scripts/seed_fact_ledger.py 调用）。返回写入条数。"""
    count = 0
    for record in records:
        fact_id = str(record["fact_id"]).strip()
        existing = db.scalar(select(FactLedger).where(FactLedger.fact_id == fact_id))
        evidence_status = str(
            record.get("evidence_status")
            or LEGACY_STATUS_TO_EVIDENCE.get(str(record.get("status") or ""), "explicit")
        )[:20]
        review_status = str(record.get("review_status") or "pending")[:20]
        values = {
            "subject": str(record["subject"])[:255],
            "predicate": str(record["predicate"])[:255],
            "value": str(record["value"])[:500],
            "unit": (str(record["unit"])[:50] if record.get("unit") else None),
            "persona_id": (str(record["persona_id"])[:40] if record.get("persona_id") else None),
            "evidence_status": evidence_status,
            "review_status": review_status,
            # 旧列兼容镜像（新代码的语义以 evidence/review 两列为准）
            "status": EVIDENCE_TO_LEGACY_STATUS.get(evidence_status, "confirmed"),
            "source_file": (str(record["source_file"])[:255] if record.get("source_file") else None),
            "source_section": (str(record["source_section"])[:255] if record.get("source_section") else None),
            "source_type": (str(record["source_type"])[:40] if record.get("source_type") else None),
        }
        if existing is not None:
            for key, value in values.items():
                setattr(existing, key, value)
        else:
            db.add(FactLedger(fact_id=fact_id, **values))
        count += 1
    db.commit()
    return count


def _normalize(text: str) -> str:
    """归一化：去空白与连字符、小写化（保留其余标点——日期/版本号中的 . / 有意义）。"""
    return re.sub(r"[\s\-]+", "", str(text or "")).lower()


_WINDOW_BEFORE = 40
_WINDOW_AFTER = 80


def check_answer(answer: str, facts: list[FactLedger]) -> FactCheckResult:
    """检查答案中的事实归属关系。

    mismatch（必须降级）：实体 S 出现在回答中，而未出现的另一实体 S2 的事实值
    落在 S 名称附近——"把 A 项目的指标安到 B 项目"类错配。
    warning（仅观测）：pending 事实被直接引用（回答者引用了待确认信息）。
    """
    result = FactCheckResult(checked_facts=len(facts))
    if not answer or not answer.strip():
        return result
    ans = _normalize(answer)

    by_subject: dict[str, list[FactLedger]] = {}
    for fact in facts:
        by_subject.setdefault(_normalize(fact.subject), []).append(fact)

    subjects_in_answer = {subject for subject in by_subject if subject and subject in ans}

    # 张冠李戴：实体 S 出现在回答中，而另一实体 S2 未出现、但 S2 的事实值
    # 落在了 S 名称附近的窗口内——"把 A 项目的指标安到 B 项目"。
    # S2 也出现在回答中时不告警（多对象并列回答是正常形态）。
    for subject, entries in by_subject.items():
        if subject not in subjects_in_answer:
            continue
        # 非 explicit 事实所属实体被讨论 → warning（回答者可能引用了待确认信息）
        for fact in entries:
            if fact.evidence_status != "explicit":
                result.warnings.append(
                    f"涉及待确认事实：{fact.subject}·{fact.predicate}={fact.value}"
                    f"（evidence={fact.evidence_status}，review={fact.review_status}）"
                )
        positions = [match.start() for match in re.finditer(re.escape(subject), ans)]
        for other_subject, other_entries in by_subject.items():
            if other_subject == subject or other_subject in subjects_in_answer:
                continue
            for other_fact in other_entries:
                other_value = _normalize(other_fact.value)
                if (
                    not other_value
                    or len(other_value) < 4
                    or other_fact.evidence_status in {"conflict", "missing"}
                ):
                    continue
                for position in positions:
                    window = ans[max(0, position - _WINDOW_BEFORE) : position + len(subject) + _WINDOW_AFTER]
                    if other_value in window:
                        result.mismatches.append(
                            f"疑似把「{other_fact.subject}」的{other_fact.predicate}"
                            f"（{other_fact.value}）安到了「{subject}」上"
                        )

    # 去重（同一错配可能被多个窗口重复检出）
    result.mismatches = list(dict.fromkeys(result.mismatches))
    result.warnings = list(dict.fromkeys(result.warnings))
    return result
