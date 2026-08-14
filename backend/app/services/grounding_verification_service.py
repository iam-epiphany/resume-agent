"""确定性硬事实校验器（零 LLM）。

解决"Self-RAG 自己生成、自己判断证据是否充足"的盲区：LLM 自评
evidence_sufficiency 可能高估证据覆盖。本模块在生成后追加一层纯规则的
硬事实核查——从答案中抽取数字、日期、书名号专名、以及知识库已知的简历
领域实体（学校/项目/奖项/证书/技术栈等），与检索证据文本做归一化匹配；
答案中的硬事实在证据中缺失时判定为"未落地"，由上层强制 hedged 标注并
记录缺失项。

设计约束：
- 零 LLM 调用、零网络开销，纯正则 + 子串匹配，单请求微秒级
- 保守优先：只对"高置信硬事实模式"（数字、日期、书名号实体、已知领域实体）
  核查，避免把 LLM 的自由措辞误判为幻觉
- 不重写答案、不做二次生成修复循环（避免过度设计）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# 数字：整数/小数/百分比/带单位数字。列表编号（"1." "2." "①"）单独排除——
# "1. EchoGuide 2. ReguMate"中的 1、2 是编号不是需证据支持的数字事实。
# 2026-08-14 扩展单位集合（k/K=千、w/W=万、月）："6到8k"类薪资数字必须
# 连同单位出现在证据中——裸数字在长证据里几乎必然撞上，带单位才可核实。
_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9年.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*[%％个万人元件项门次kKwW月]?"
)
# 列表编号/序数：1. 2. ① ② 第1 第2（不视为数字事实）
_LIST_ORDINAL_PATTERN = re.compile(
    r"(?:^|[：（(【\s，,、])(?:第\s*)?[①-⑳一二三四五六七八九十百\d]{1,3}\s*[.、)）】]" +
    r"|(?:^|\s)(?:\d{1,2})[.、](?=\s)"
)
# 4 位年份（1900-2099）
_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# 日期：2026-01-02 / 2026年1月2日
_DATE_PATTERN = re.compile(
    r"(?<!\d)(?:20\d{2})[-/年.]\d{1,2}[-/月.]\d{1,2}日?"
)
# 书名号专名（项目/证书/材料名）：《XXX》
_QUOTED_NAME_PATTERN = re.compile(r"[《「“]([^》」”]{2,40})[》」”]")

# 全角→半角映射（数字、百分号）
_FULLWIDTH_TO_HALFWIDTH = {
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "％": "%",
}


@dataclass
class GroundingVerificationResult:
    verified: bool                     # 全部硬事实在证据中找到 → True
    missing_numbers: list[str] = field(default_factory=list)
    missing_dates: list[str] = field(default_factory=list)
    missing_names: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    checked_numbers: int = 0
    checked_dates: int = 0
    checked_names: int = 0
    checked_entities: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "missing_numbers": self.missing_numbers,
            "missing_dates": self.missing_dates,
            "missing_names": self.missing_names,
            "missing_entities": self.missing_entities,
            "checked_numbers": self.checked_numbers,
            "checked_dates": self.checked_dates,
            "checked_names": self.checked_names,
            "checked_entities": self.checked_entities,
        }


def _normalize(text: str) -> str:
    """归一化：去空白、全角数字/百分号转半角、统一破折号，便于子串匹配。"""
    text = re.sub(r"\s+", "", text)
    for full, half in _FULLWIDTH_TO_HALFWIDTH.items():
        text = text.replace(full, half)
    return text


def _strip_list_ordinals(answer: str) -> str:
    """从答案中剔除列表编号/序数，避免把"1. EchoGuide"的 1 当成数字事实。"""
    return _LIST_ORDINAL_PATTERN.sub("", answer)


def verify_hard_facts(
    answer: str | None,
    evidence_texts: list[str],
    known_entities: Iterable[str] | None = None,
) -> GroundingVerificationResult:
    """核对答案中的硬事实是否能在检索证据中找到。

    answer: 生成的回答文本
    evidence_texts: 检索证据片段（context_chunks 的 text/embedding_text/section_title 拼接）
    known_entities: 知识库已知的简历领域实体名（学校/项目/奖项/证书/技术栈等，
        从文档 catalog/标题提取）。答案中出现这些实体但证据中缺失 → 未核实。
    """
    result = GroundingVerificationResult(verified=True)
    if not answer or not answer.strip():
        return result

    normalized_answer = _normalize(answer)
    # 剔除列表编号后再做数字提取（"1. EchoGuide 2. ReguMate"的编号不算数字事实）
    answer_without_ordinals = _strip_list_ordinals(normalized_answer)
    evidence = _normalize(" ".join(text for text in evidence_texts if text))

    # 1) 数字（跳过年份——年份作为独立类别单独核对）
    numbers = {num.strip() for num in _NUMBER_PATTERN.findall(answer_without_ordinals) if num.strip()}
    standalone_years = {year for year in numbers if re.fullmatch(r"(?:19|20)\d{2}", year)}
    numbers -= standalone_years
    result.checked_numbers = len(numbers)
    for number in numbers:
        if number not in evidence:
            result.missing_numbers.append(number)

    # 2) 年份（4 位）
    years = set(_YEAR_PATTERN.findall(normalized_answer))
    result.checked_dates = len(years)
    for year in years:
        if year not in evidence:
            result.missing_dates.append(year)

    # 3) 日期（含年月日完整格式）
    dates = set(_DATE_PATTERN.findall(normalized_answer))
    result.checked_dates += len(dates)
    for date_text in dates:
        normalized_date = _normalize(date_text)
        if normalized_date not in evidence:
            result.missing_dates.append(date_text)

    # 4) 书名号专名
    names = set(_QUOTED_NAME_PATTERN.findall(normalized_answer))
    result.checked_names = len(names)
    for name in names:
        normalized_name = _normalize(name)
        if normalized_name not in evidence:
            result.missing_names.append(name)

    # 5) 知识库已知领域实体（学校/项目/奖项/证书/技术栈等）
    # 答案中出现但证据中缺失 → 未核实。实体按长度降序匹配（更长/更具体的优先）。
    if known_entities:
        entity_pool = sorted(
            {str(entity).strip() for entity in known_entities if str(entity).strip()},
            key=len,
            reverse=True,
        )
        for entity in entity_pool:
            normalized_entity = _normalize(entity)
            if len(normalized_entity) < 2 or normalized_entity in evidence:
                continue
            if normalized_entity in normalized_answer:
                result.checked_entities += 1
                result.missing_entities.append(entity)

    result.verified = not (
        result.missing_numbers or result.missing_dates
        or result.missing_names or result.missing_entities
    )
    return result
