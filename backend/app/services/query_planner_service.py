from dataclasses import dataclass, replace
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable
from backend.app.services.performance_metrics import measure, timed

from backend.app.core.config import (
    QUERY_PLANNER_INCLUDE_THINKING,
    QUERY_PLANNER_API_KEY,
    QUERY_PLANNER_BASE_URL,
    QUERY_PLANNER_ENABLED,
    QUERY_PLANNER_MAX_ASPECTS,
    QUERY_PLANNER_MAX_SEARCH_QUERIES,
    QUERY_PLANNER_MODEL,
    QUERY_PLANNER_PROVIDER,
    QUERY_PLANNER_RESPONSE_FORMAT,
    QUERY_PLANNER_TIMEOUT_SECONDS,
    REWRITE_ENABLED,
    REWRITE_MAX_QUERIES,
)
from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
    chat_completion_content,
)


class QueryPlannerError(RuntimeError):
    pass


VALID_QUERY_TYPES = {"semantic_question", "document_style_statement", "keyword_anchor", "legacy", "fallback"}


@dataclass(frozen=True)
class QuerySearchQuery:
    query: str
    query_type: str
    rationale: str = ""

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class QueryAspect:
    aspect_id: str
    question: str
    search_queries: tuple[QuerySearchQuery, ...]
    evidence_need: str
    keywords: tuple[str, ...]

    @property
    def expected_evidence_type(self) -> str:
        return self.evidence_need

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "aspect_id": self.aspect_id,
            "question": self.question,
            "evidence_need": self.evidence_need,
            "expected_evidence_type": self.evidence_need,
            "search_queries": [query.to_debug_dict() for query in self.search_queries],
            "keywords": list(self.keywords),
        }


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    aspects: tuple[QueryAspect, ...]
    planner: str
    fallback_used: bool = False
    error: str | None = None
    budget: dict[str, Any] | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "planner": self.planner,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "budget": self.budget or {},
            "aspects": [aspect.to_debug_dict() for aspect in self.aspects],
        }


@dataclass(frozen=True)
class QueryBudget:
    detected_items: tuple[str, ...]
    max_aspects: int
    system_max_aspects: int
    capacity_limited: bool
    omitted_or_merged_items: tuple[str, ...] = ()

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "detected_items": list(self.detected_items),
            "detected_item_count": len(self.detected_items),
            "max_aspects": self.max_aspects,
            "system_max_aspects": self.system_max_aspects,
            "capacity_limited": self.capacity_limited,
            "omitted_or_merged_items": list(self.omitted_or_merged_items),
        }


@timed("query_planner.total")
def plan_query(
    question: str,
    options: list[str] | None = None,
    *,
    cancellation_checker: Callable[[], None] | None = None,
) -> QueryPlan:
    cleaned_question = question.strip()
    if not cleaned_question:
        return QueryPlan(original_question="", aspects=(), planner="empty", fallback_used=True)

    budget = plan_query_budget(cleaned_question)

    if QUERY_PLANNER_ENABLED and QUERY_PLANNER_API_KEY:
        try:
                aspects, omitted_or_merged_items = _plan_with_llm(cleaned_question, budget, cancellation_checker=cancellation_checker)
                if aspects:
                    aspects = _split_merged_llm_aspects(aspects, budget)
                    budget = _budget_with_omissions(budget, omitted_or_merged_items)
                    return QueryPlan(
                        original_question=cleaned_question,
                        aspects=tuple(aspects),
                        planner=f"{QUERY_PLANNER_PROVIDER}:{QUERY_PLANNER_MODEL}",
                    fallback_used=False,
                    budget=budget.to_debug_dict(),
                )
        except QueryPlannerError as exc:
            fallback = _fallback_aspects(cleaned_question, budget)
            return QueryPlan(
                original_question=cleaned_question,
                aspects=tuple(fallback),
                planner="fallback",
                fallback_used=True,
                error=str(exc),
                budget=budget.to_debug_dict(),
            )

    return QueryPlan(
        original_question=cleaned_question,
        aspects=tuple(_fallback_aspects(cleaned_question, budget)),
        planner="fallback",
        fallback_used=True,
        budget=budget.to_debug_dict(),
    )


def plan_query_budget(question: str) -> QueryBudget:
    explicit_items = _explicit_requested_items(question)
    detected_items = tuple(_dedupe(explicit_items))
    question_parts = _split_question_parts(question)
    nested_coordinated_count = sum(
        max(1, len(_coordinated_aspect_questions(part)))
        for part in question_parts
    )
    desired = max(len(detected_items), len(question_parts), nested_coordinated_count)
    desired = max(desired, len(_coordinated_aspect_questions(question)))
    desired = max(desired, _explicit_requested_aspect_count(question))
    desired = max(desired, 1)
    max_aspects = min(desired, QUERY_PLANNER_MAX_ASPECTS)
    return QueryBudget(
        detected_items=detected_items,
        max_aspects=max_aspects,
        system_max_aspects=QUERY_PLANNER_MAX_ASPECTS,
        capacity_limited=desired > QUERY_PLANNER_MAX_ASPECTS,
        omitted_or_merged_items=detected_items[QUERY_PLANNER_MAX_ASPECTS:],
    )


def _explicit_requested_items(question: str) -> list[str]:
    text = str(question or "").strip()
    if not text:
        return []
    compact = re.sub(r"\s+", "", text)
    chinese_counts = {
        "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
        "七": 7, "八": 8, "九": 9, "十": 10,
    }
    count_match = re.search(r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", compact)
    if not count_match:
        return []
    expected_count = int(count_match.group(1)) if count_match.group(1).isdigit() else chinese_counts.get(count_match.group(1), 0)
    if expected_count <= 1:
        return []
    original_match = re.search(r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", text)
    if not original_match:
        return []
    if original_match.end() < len(text) and "：" in text[original_match.end(): original_match.end() + 4]:
        segment = text[text.find("：", original_match.end()) + 1:]
    else:
        left = text[: original_match.start()]
        segment = re.split(r"[：:]", left)[-1]
    segment = re.split(r"[。！？?]", segment, maxsplit=1)[0]
    parts = [
        re.sub(r"^(?:形成审查摘要|联合核验|核验|说明|概括|回答|列出|请)+", "", part.strip(" ，,；;。:："))
        for part in re.split(r"[、；;,，]|以及|和|与|及", segment)
        if part.strip(" ，,；;。:：")
    ]
    parts = [part for part in parts if part and not re.fullmatch(r"(?:再)?(?:比较|查询|读取|计算|说明|最后)", part)]
    return _dedupe(parts)[:expected_count] if len(parts) >= expected_count else []


def _requested_items_overlap(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        text = re.sub(r"[\s，,。；;：:、？?]+", "", str(value or ""))
        for noise in ("是否可以", "是否", "可以", "直接", "生成", "正式", "违规", "报告"):
            text = text.replace(noise, "")
        return text

    left_norm = normalize(left)
    right_norm = normalize(right)
    return bool(left_norm and right_norm and (left_norm in right_norm or right_norm in left_norm))


def _explicit_requested_aspect_count(question: str) -> int:
    """Estimate only an upper budget for visibly enumerated user requests."""

    compact = re.sub(r"\s+", "", str(question or ""))
    chinese_counts = {
        "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
        "七": 7, "八": 8, "九": 9, "十": 10,
    }
    explicit = 1
    for match in re.finditer(r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", compact):
        token = match.group(1)
        explicit = max(explicit, int(token) if token.isdigit() else chinese_counts.get(token, 1))
    if any(marker in compact for marker in ("分别", "逐项", "汇总", "串联", "同时", "跨材料", "审查摘要")):
        # Enumeration commas are ambiguous in ordinary prose, so only use
        # them when the user explicitly asks for a multi-item response. This
        # raises the planner ceiling; it does not force artificial aspects.
        explicit = max(explicit, 1 + compact.count("、") + compact.count("；"))
    return min(explicit, QUERY_PLANNER_MAX_ASPECTS)


def _split_merged_llm_aspects(
    aspects: list[QueryAspect],
    budget: QueryBudget,
) -> list[QueryAspect]:
    """Restore clearly coordinated requirements merged by the remote planner.

    The split is intentionally narrow: strong “以及” coordination, or an
    “说明 A 和 B” construction where both sides name requirement categories.
    It never exceeds the deterministic budget and does not use answer data.
    """

    split_limit = max(budget.max_aspects, budget.system_max_aspects)
    if len(aspects) >= split_limit:
        return aspects
    result: list[QueryAspect] = []
    for aspect in aspects:
        remaining_capacity = split_limit - len(result)
        split_questions = _coordinated_aspect_questions(aspect.question)
        if len(split_questions) <= 1 or remaining_capacity < len(split_questions):
            result.append(aspect)
            continue
        for index, question in enumerate(split_questions, start=1):
            result.append(
                replace(
                    aspect,
                    aspect_id=f"{aspect.aspect_id}_part_{index}",
                    question=question,
                    search_queries=_fallback_search_queries(question),
                    evidence_need=f"与“{question}”直接对应的材料事实、条件或例外",
                    keywords=tuple(_keywords_from_text(question)),
                )
            )
    return result[:split_limit]


def _coordinated_aspect_questions(question: str) -> list[str]:
    text = str(question or "").strip()
    prefix_match = re.match(r"^(.*?(?:说明|概括|回答|列出|判断|核验|比较))", text)
    prefix = prefix_match.group(1) if prefix_match else ""
    titles = re.findall(r"《[^》]{2,80}》", text)

    summary_enumeration = _summary_enumerated_aspect_questions(text)
    if summary_enumeration:
        return summary_enumeration

    explicit_follow_up = re.match(
        r"^(.*?)[，,]并(说明|概括|回答|列出|判断|核验|比较)([^，。；]{4,80})([。？?]?)$",
        text,
    )
    if explicit_follow_up:
        left, verb, right, suffix = explicit_follow_up.groups()
        return [left.strip(" ，,"), f"{verb}{right}{suffix}"]

    strong_parts = re.split(r"，?以及", text, maxsplit=1)
    if len(strong_parts) == 2:
        left, right = (part.strip(" ，,。；;") for part in strong_parts)
        if prefix and not right.startswith(prefix):
            title_context = f"{''.join(titles)}中" if titles else ""
            right = f"{prefix}{title_context}{right}"
        if all(len(re.sub(r"\s+", "", part)) >= 6 for part in (left, right)):
            return [left, right]

    related_requirements = re.match(
        r"^(.*?与)([^，。；]{2,30}?)(?:和|以及)([^，。；]{2,30}?)(相关的(?:要求|规定|内容).*)$",
        text,
    )
    if related_requirements:
        common_prefix, left, right, suffix = related_requirements.groups()
        return [f"{common_prefix}{left}{suffix}", f"{common_prefix}{right}{suffix}"]

    category = r"(?:原则|对象|条件|期限|比例|定义|范围|要求|例外|情景|原因|方法|假设|阈值|门槛|顺序|效力|禁限|递延)"
    strong_bare_category = r"(?:原则|对象|条件|期限|时限|比例|例外|情景|原因|方法|假设|阈值|门槛|顺序|效力|禁限|递延|信息|版本|频率|留档要求)"
    coordinated = re.match(
        rf"^(.*?(?:说明|概括|回答|列出))(.{{2,24}}?{strong_bare_category})和(.{{2,24}}?{strong_bare_category})(.*)$",
        text,
    )
    if coordinated:
        common_prefix, left, right, suffix = coordinated.groups()
        # Preserve a shared interrogative tail such as “的依据是什么？”.  The
        # remote planner often merges two evidence needs immediately before
        # that tail, and dropping it makes the second query unnatural.
        return [f"{common_prefix}{left}{suffix}", f"{common_prefix}{right}{suffix}"]
    definition_and_requirement = re.match(
        r"^(.{2,30}?定义)(?:及|和)(.{2,30}?(?:要求|原则))(.*)$",
        text,
    )
    if definition_and_requirement:
        left, right, suffix = definition_and_requirement.groups()
        right = _inherit_left_subject(left, right)
        return [f"{left}{suffix}", f"{right}{suffix}"]
    # Remote planners sometimes compress an explicit “，并说明 …” request to
    # “门槛及…原则”.  Accept ``及/和`` only for two strong requirement
    # categories so ordinary lists such as “定义、口径、范围和表述要求” are
    # not split merely because they contain a conjunction.
    strong_bare_coordinated = re.match(
        rf"^(.{{2,40}}?{strong_bare_category})(?:及|和)(.{{2,40}}?{strong_bare_category})(.*)$",
        text,
    )
    if strong_bare_coordinated:
        left, right, suffix = strong_bare_coordinated.groups()
        right = _inherit_left_subject(left, right)
        return [f"{left}{suffix}", f"{right}{suffix}"]
    bare_coordinated = re.match(
        rf"^(.{{2,30}}?{category})与(.{{2,30}}?{category})(.*)$",
        text,
    )
    if bare_coordinated:
        left, right, suffix = bare_coordinated.groups()
        right = _inherit_left_subject(left, right)
        return [f"{left}{suffix}", f"{right}{suffix}"]
    principle_pair = re.match(
        r"^(.{2,30}?(?:项目|技能|成果|工具|策略|计划|要求|定义))(?:与|和)(.{2,20}?原则)(.*)$",
        text,
    )
    if principle_pair:
        left, right, suffix = principle_pair.groups()
        if "自救" in right and "风险暴露" in left and "处置" not in left:
            left = f"处置计划建议示例中的{left}"
        return [f"{left}{suffix}", f"{right}{suffix}"]
    return [text]


def _summary_enumerated_aspect_questions(question: str) -> list[str]:
    text = str(question or "").strip()
    if "、" not in text or not any(marker in text for marker in ("串联", "汇总", "逐项", "分别")):
        return []
    marker_positions = [
        text.rfind(marker)
        for marker in ("串联", "汇总", "逐项", "分别")
        if marker in text
    ]
    if not marker_positions:
        return []
    tail = text[max(marker_positions) + 2 :].strip(" ：:，,。；;？?")
    if not tail:
        return []
    parts = [
        part.strip(" ：:，,。；;？?")
        for part in re.split(r"、", tail)
        if part.strip(" ：:，,。；;？?")
    ]
    if len(parts) <= 1:
        return []
    expanded: list[str] = []
    for part in parts:
        nested = _split_compact_enumerated_part(part)
        expanded.extend(nested or [part])
    return _dedupe(expanded)[:QUERY_PLANNER_MAX_ASPECTS] if len(expanded) > 1 else []


def _split_compact_enumerated_part(part: str) -> list[str]:
    definition_and_requirement = re.match(
        r"^(.{2,30}?定义)(?:及|和)(.{2,30}?(?:要求|原则))(.*)$",
        part,
    )
    if definition_and_requirement:
        left, right, suffix = definition_and_requirement.groups()
        right = _inherit_left_subject(left, right)
        return [f"{left}{suffix}", f"{right}{suffix}"]
    principle_pair = re.match(
        r"^(.{2,30}?(?:项目|技能|成果|工具|策略|计划|要求|定义))(?:与|和)(.{2,20}?原则)(.*)$",
        part,
    )
    if principle_pair:
        left, right, suffix = principle_pair.groups()
        if "技能" in right and "项目" in left and "掌握" not in left:
            right = f"技能掌握情况的{right}"
        return [f"{left}{suffix}", f"{right}{suffix}"]
    return [part]


def _inherit_left_subject(left: str, right: str) -> str:
    if "的" in right:
        return right
    if not re.search(r"(?:定义|概念|含义)$", left):
        return right
    if re.search(r"(?:奖学金|软考|蓝桥杯|项目|技能|证书|实习|荣誉|奖项|开发)", right):
        return right
    bare_subject = re.sub(r"(?:的)?(?:定义|概念|含义)$", "", left).strip()
    bare_subject = re.sub(r"^(?:说明|概括|回答|列出|判断|核验|比较)", "", bare_subject).strip()
    if bare_subject and len(re.sub(r"\s+", "", bare_subject)) >= 2:
        return f"{bare_subject}的{right}"
    subject_match = re.match(r"^(.{2,40}?的).+", left)
    if not subject_match:
        return right
    subject = subject_match.group(1)
    if right.startswith(subject):
        return right
    return f"{subject}{right}"


def _budget_with_omissions(budget: QueryBudget, omitted_or_merged_items: list[str]) -> QueryBudget:
    merged = tuple(_dedupe([*budget.omitted_or_merged_items, *omitted_or_merged_items]))
    return QueryBudget(
        detected_items=budget.detected_items,
        max_aspects=budget.max_aspects,
        system_max_aspects=budget.system_max_aspects,
        capacity_limited=budget.capacity_limited or bool(merged),
        omitted_or_merged_items=merged,
    )


def _plan_with_llm(
    question: str,
    budget: QueryBudget,
    *,
    cancellation_checker: Callable[[], None] | None = None,
) -> tuple[list[QueryAspect], list[str]]:
    messages = [
            {
                "role": "system",
                "content": (
                    "你是 ResumeMind 的检索前 QueryPlanner。"
                    "你的唯一任务是把关于简历主人公（求职者）的提问拆成检索 aspect，生成检索计划。"
                    "禁止回答用户问题，禁止给出结论，禁止补充知识库外事实。"
                    "知识库内容是个人材料：简历、证书说明、荣誉奖项、教育背景、技能专长、求职意向、项目介绍等文字文档。"
                    "为向量检索生成贴近用户意图、可能命中材料原文 chunk 的自然语言查询；"
                    "不要使用公文、报告类术语改写用户问题（例如把“参与过哪些项目”改写成“支持材料/项目清单”）。"
                    "用户枚举多个方面时（如多个项目、多项技能），原则上一项一个 aspect，不要遗漏。"
                    "只输出 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请按本次预算把下面问题拆成 1 到 "
                    f"{budget.max_aspects} 个 aspect，系统安全上限为 {budget.system_max_aspects}。每个 aspect 包含："
                    "aspect_id、question、evidence_need、search_queries、keywords。"
                    "如果用户明确枚举了多个处理对象，一般每个处理对象单独成为一个 aspect；"
                    "如果因为预算必须合并或遗漏项目，把项目名称写入 omitted_or_merged_items。"
                    "输出 JSON 顶层必须包含 aspects 和 omitted_or_merged_items。"
                    "search_queries 必须是对象数组，每个对象包含 query、query_type、rationale。"
                    "query_type 只能是 semantic_question、document_style_statement、keyword_anchor。"
                    "每个 aspect 生成 2 到 "
                    f"{QUERY_PLANNER_MAX_SEARCH_QUERIES} 条 search_queries："
                    "semantic_question 贴近用户意图；document_style_statement 像简历、项目介绍、证书荣誉等材料中的自然表述；"
                    "keyword_anchor 只用少量关键术语兜底。"
                    "优先生成可能出现在个人材料原文中的自然语言查询，禁止只输出关键词串，也禁止套用公文/报告术语。"
                    "evidence_need 描述要找的材料内容（如项目经历、技能清单、获奖记录、教育背景、求职意向）。"
                    "如果问题只有一个主题，也返回一个 aspect。"
                    "示例输入：你参与过哪些项目？"
                    "示例输出：{\"aspects\":["
                    "{\"aspect_id\":\"project_experience\","
                    "\"question\":\"你参与过哪些项目？\","
                    "\"evidence_need\":\"在简历与项目介绍文档中查找参与过的项目经历\","
                    "\"search_queries\":["
                    "{\"query\":\"参与过哪些项目 项目经历\",\"query_type\":\"semantic_question\",\"rationale\":\"贴近用户意图\"},"
                    "{\"query\":\"项目介绍 项目背景 主要工作 技术栈 成果\",\"query_type\":\"document_style_statement\",\"rationale\":\"项目介绍文档的常见表述\"},"
                    "{\"query\":\"项目 经历 参与\",\"query_type\":\"keyword_anchor\",\"rationale\":\"术语兜底\"}],"
                    "\"keywords\":[\"项目\",\"经历\",\"参与\"]}"
                    "]}\n"
                    f"本次规则识别到的用户处理对象：{json.dumps(list(budget.detected_items), ensure_ascii=False)}\n"
                    "输出格式：{\"aspects\":[...],\"omitted_or_merged_items\":[]}\n\n"
                    f"用户问题：{question}"
                ),
            },
        ]
    try:
        if cancellation_checker is not None:
            cancellation_checker()
        with measure("query_planner.external_api"):
            content = chat_completion_content(
                ChatCompletionConfig(
                    provider=QUERY_PLANNER_PROVIDER,
                    api_key=QUERY_PLANNER_API_KEY,
                    base_url=QUERY_PLANNER_BASE_URL,
                    model=QUERY_PLANNER_MODEL,
                    timeout_seconds=QUERY_PLANNER_TIMEOUT_SECONDS,
                    include_thinking=QUERY_PLANNER_INCLUDE_THINKING,
                    response_format=QUERY_PLANNER_RESPONSE_FORMAT,
                ),
                messages,
                temperature=0,
                response_format=QUERY_PLANNER_RESPONSE_FORMAT,
                opener=urllib.request.urlopen,
            )
    except ChatCompletionError as exc:
        raise QueryPlannerError("LLM query planner 调用失败") from exc

    try:
        parsed = json.loads(_extract_json_object(content))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise QueryPlannerError("LLM query planner 返回格式不可解析") from exc

    return _aspects_from_payload(question, parsed, max_aspects=QUERY_PLANNER_MAX_ASPECTS), _clean_string_list(
        parsed.get("omitted_or_merged_items")
    )


def _aspects_from_payload(
    question: str,
    payload: dict[str, Any],
    max_aspects: int | None = None,
) -> list[QueryAspect]:
    raw_aspects = payload.get("aspects")
    if not isinstance(raw_aspects, list):
        raise QueryPlannerError("LLM query planner 未返回 aspects")

    aspects: list[QueryAspect] = []
    limit = max_aspects if max_aspects is not None else QUERY_PLANNER_MAX_ASPECTS
    for index, raw_aspect in enumerate(raw_aspects[:limit], start=1):
        if not isinstance(raw_aspect, dict):
            continue
        sub_question = _clean_text(raw_aspect.get("question")) or question
        if _is_execution_instruction_aspect(sub_question):
            # A user instruction such as “if sqrt is unsupported, refuse”
            # constrains execution. It is not a fact that must be retrieved
            # from the material, so it must not become a missing aspect.
            continue
        search_queries = _clean_search_queries(raw_aspect.get("search_queries"))
        evidence_need = (
            _clean_text(raw_aspect.get("evidence_need"))
            or _clean_text(raw_aspect.get("expected_evidence_type"))
            or "相关材料依据"
        )
        keywords = _clean_string_list(raw_aspect.get("keywords"))
        if not search_queries:
            search_queries = [QuerySearchQuery(query=sub_question, query_type="fallback", rationale="LLM 未返回检索查询")]
        if not keywords:
            keywords = _keywords_from_text(" ".join([sub_question, *[query.query for query in search_queries]]))
        aspects.append(
            QueryAspect(
                aspect_id=_clean_aspect_id(raw_aspect.get("aspect_id"), index),
                question=sub_question,
                search_queries=tuple(_dedupe_search_queries(search_queries)[:QUERY_PLANNER_MAX_SEARCH_QUERIES]),
                evidence_need=evidence_need,
                keywords=tuple(_dedupe(keywords)[:8]),
            )
        )

    if not aspects:
        raise QueryPlannerError("LLM query planner 未产生有效 aspect")
    return _expand_explicit_anchor_aspects(aspects, max_aspects=limit)


def _fallback_aspects(question: str, budget: QueryBudget | None = None) -> list[QueryAspect]:
    budget = budget or plan_query_budget(question)
    parts = _split_question_parts(question)
    if not parts:
        parts = [question]

    aspects: list[QueryAspect] = []
    for index, part in enumerate(parts[:budget.max_aspects], start=1):
        keywords = _keywords_from_text(part)
        aspects.append(
            QueryAspect(
                aspect_id=f"aspect_{index}",
                question=part,
                search_queries=_fallback_search_queries(part),
                evidence_need=_infer_evidence_type(part),
                keywords=tuple(keywords),
            )
        )
    return _expand_explicit_anchor_aspects(
        aspects,
        max_aspects=max(budget.max_aspects, budget.system_max_aspects),
    )


def _split_question_parts(question: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[？?。；;]|如果|若|以及|并且|同时|，且", question)
        if part.strip()
    ]


def _dedupe_aspects(aspects: list[QueryAspect]) -> list[QueryAspect]:
    deduped: list[QueryAspect] = []
    seen: set[str] = set()
    for aspect in aspects:
        if aspect.aspect_id in seen:
            continue
        seen.add(aspect.aspect_id)
        deduped.append(aspect)
    return deduped


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise json.JSONDecodeError("missing JSON object", stripped, 0)
    return match.group(0)


def _clean_aspect_id(value: Any, index: int) -> str:
    text = _clean_text(value)
    if not text:
        return f"aspect_{index}"
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")
    return cleaned or f"aspect_{index}"


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    return [cleaned for item in items if (cleaned := _clean_text(item))]


def _clean_search_queries(value: Any) -> list[QuerySearchQuery]:
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [QuerySearchQuery(query=cleaned, query_type="legacy", rationale="兼容旧字符串格式")] if cleaned else []
    if not isinstance(value, list):
        return []

    queries: list[QuerySearchQuery] = []
    for item in value:
        if isinstance(item, str):
            cleaned = _clean_text(item)
            if cleaned:
                queries.append(
                    QuerySearchQuery(query=cleaned, query_type="legacy", rationale="兼容旧字符串格式")
                )
            continue
        if not isinstance(item, dict):
            continue
        query = _clean_text(item.get("query"))
        if not query:
            continue
        query_type = _clean_text(item.get("query_type")) or "semantic_question"
        if query_type not in VALID_QUERY_TYPES:
            query_type = "semantic_question"
        queries.append(
            QuerySearchQuery(
                query=query,
                query_type=query_type,
                rationale=_clean_text(item.get("rationale")),
            )
        )
    return queries



def _expand_explicit_anchor_aspects(
    aspects: list[QueryAspect],
    *,
    max_aspects: int,
) -> list[QueryAspect]:
    """Split multiple explicitly quoted evidence needs into retrieval units.

    The planner may merge two requested clauses from the same document into a
    single aspect.  One high-scoring chunk can then make the aspect appear
    covered while the second clause is absent.  Quoted clauses are user-stated
    constraints, so preserving each as its own aspect is deterministic and
    does not infer an answer.
    """

    expanded: list[QueryAspect] = []
    for aspect in aspects:
        anchors = _quoted_evidence_anchors(aspect.question)
        if len(anchors) < 2:
            expanded.append(aspect)
            continue
        for index, anchor in enumerate(anchors, start=1):
            anchor_question = _question_for_single_anchor(aspect.question, anchor)
            expanded.append(
                replace(
                    aspect,
                    aspect_id=f"{aspect.aspect_id}_{index}",
                    question=anchor_question,
                    search_queries=(
                        QuerySearchQuery(anchor_question, "semantic_question", "拆分后的显式条件检索"),
                        QuerySearchQuery(anchor, "keyword_anchor", "显式条件术语兜底"),
                    ),
                    evidence_need=f"与“{anchor}”直接对应的材料原文及其条件、范围、期限或例外",
                    keywords=tuple(_dedupe([anchor, *_keywords_from_text(anchor)])),
                )
            )
    return expanded[:max_aspects]


def _quoted_evidence_anchors(question: str) -> list[str]:
    return _dedupe(
        [
            item.strip()
            for item in re.findall(r"[“‘\"]([^”’\"]+)[”’\"]", question)
            if len(re.sub(r"\s+", "", item)) >= 2
        ]
    )


def _question_for_single_anchor(question: str, anchor: str) -> str:
    title_match = re.search(r"《([^》]+)》", question)
    title_prefix = f"根据《{title_match.group(1)}》，" if title_match else ""
    quoted_matches = list(re.finditer(r"[“‘\"]([^”’\"]+)[”’\"]", question))
    if not title_match and len(quoted_matches) >= 2:
        prefix = question[: quoted_matches[0].start()].rstrip("，,、和与及")
        suffix = question[quoted_matches[-1].end() :].lstrip("，,、和与及")
        return f"{prefix}“{anchor}”{suffix}".strip()
    return f"{title_prefix}请说明与“{anchor}”相关的明确规定，并完整给出条件、范围、期限或例外。"





def _quoted_terms(text: str) -> list[str]:
    return _dedupe(
        [
            item.strip()
            for item in re.findall(r"[“‘\"]([^”’\"]+)[”’\"]", str(text or ""))
            if item.strip()
        ]
    )





def _is_execution_instruction_aspect(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return bool(
        re.search(
            r"^(?:若|如果).{0,30}(?:不支持|无法执行).{0,30}(?:必须|应当)?(?:明确)?拒答(?:而非|不得)?(?:估算|推算)?[。？?]?$",
            compact,
        )
    )


def _labelled_question_part(question: str, label: str) -> str:
    match = re.search(
        rf"{re.escape(label)}[：:]\s*(.+?)(?=\s*(?:材料侧|表格侧|证据边界)[：:]|$)",
        question,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else question


def _is_document_formula_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question).casefold()
    return "公式" in normalized and any(marker in normalized for marker in ("word", "pdf", "[公式]"))





def _fallback_search_queries(question: str) -> tuple[QuerySearchQuery, ...]:
    keywords = " ".join(_keywords_from_text(question))
    queries = [
        QuerySearchQuery(query=question, query_type="semantic_question", rationale="fallback 子问题检索"),
    ]
    document_style = _document_style_alias(question)
    if document_style != question:
        queries.append(
            QuerySearchQuery(
                query=document_style,
                query_type="document_style_statement",
                rationale="将用户概念转换为材料原文常见表达",
            )
        )
    if keywords and keywords != question:
        queries.append(
            QuerySearchQuery(query=keywords, query_type="keyword_anchor", rationale="fallback 关键术语兜底")
        )
    return tuple(queries[:QUERY_PLANNER_MAX_SEARCH_QUERIES])


def _document_style_alias(question: str) -> str:
    """Create a query-shaped alias only from the user's own wording.

    This transformation intentionally contains no material conclusions,
    thresholds, document titles, or benchmark-specific vocabulary.
    """

    text = re.sub(r"\s+", " ", str(question or "")).strip()
    text = re.sub(r"[\uFF1F?\u3002\uFF1B;\uFF0C,\uFF1A:]+", " ", text)
    text = re.sub(
        r"(?:\u8BF7\u95EE|\u8BF7\u8BF4\u660E|\u8BF7\u89E3\u91CA|"
        r"\u662F\u4EC0\u4E48|\u6709\u54EA\u4E9B|\u5982\u4F55|"
        r"\u4E3A\u4F55|\u591A\u5C11)$",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()





def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _dedupe_search_queries(items: list[QuerySearchQuery]) -> list[QuerySearchQuery]:
    deduped: list[QuerySearchQuery] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", "", item.query)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _keywords_from_text(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_./%-]{2,}|[\u4e00-\u9fff]{2,12}", str(text or ""))
    stopwords = {"请问", "请说明", "请解释", "是什么", "有哪些", "如何", "为何"}
    return [token for token in _dedupe(tokens) if token not in stopwords][:8]


def _infer_evidence_type(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    if any(term in normalized for term in ["保留", "依据", "材料"]):
        return "材料依据或支持材料"
    if any(term in normalized for term in ["怎么", "如何", "处理", "实现", "完成"]):
        return "做法或实现方式"
    if any(term in normalized for term in ["哪些", "情形", "条件", "口径"]):
        return "分类条件或表述口径"
    return "相关材料依据"


_REWRITE_PROMPT = """你是简历问答系统的查询改写器。面试官的问题可能是口语化的，需要改写成适合向量检索的查询。
输出 1-{max_queries} 条检索式查询（去口语、补全技术术语与项目/文档锚点，每条独立成行，不要编号）。
示例：
问题：秒杀系统那个超卖你们到底怎么防的？
改写：
高并发电商闪购平台 秒杀 防超卖 乐观锁 Redisson 分布式锁
Redis 预扣库存 Lua 原子扣减 一人一单
只输出改写后的查询，每行一条，不要任何解释。"""


def rewrite_search_queries(
    question: str,
    intent: str | None = None,
    cancellation_checker: Callable[[], None] | None = None,
) -> list[str]:
    """LLM 查询改写：把口语化问题改写成 1-3 条检索式查询。

    RAGFlow 语义路由简化版（技术亮点 2）：深挖/技术类问题或检索兜底链第 2 级触发。
    失败回退：`_fallback_search_queries` + 引号/书名号锚点。
    """
    if not (REWRITE_ENABLED and QUERY_PLANNER_API_KEY):
        return _rewrite_fallback(question)
    messages = [
        {"role": "system", "content": _REWRITE_PROMPT.format(max_queries=REWRITE_MAX_QUERIES)},
        {"role": "user", "content": f"问题：{question}"},
    ]
    config = ChatCompletionConfig(
        provider=QUERY_PLANNER_PROVIDER,
        api_key=QUERY_PLANNER_API_KEY,
        base_url=QUERY_PLANNER_BASE_URL,
        model=QUERY_PLANNER_MODEL,
        timeout_seconds=min(QUERY_PLANNER_TIMEOUT_SECONDS, 12.0),
        include_thinking=QUERY_PLANNER_INCLUDE_THINKING,
        response_format=QUERY_PLANNER_RESPONSE_FORMAT,
    )
    try:
        from backend.app.services.llm_client import chat_completion_content

        content = chat_completion_content(
            config, messages, temperature=0, max_tokens=240
        )
        queries = [
            re.sub(r"^[\s\d.\-、)）]+", "", line.strip()).strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith(("```", "改写", "查询"))
        ]
        queries = [q for q in queries if len(q) >= 4][:REWRITE_MAX_QUERIES]
        if queries:
            return queries
    except Exception:
        pass
    return _rewrite_fallback(question)


def _rewrite_fallback(question: str) -> list[str]:
    fallback = [
        item.query if isinstance(item, QuerySearchQuery) else str(item)
        for item in _fallback_search_queries(question)
    ]
    for anchor in _quoted_evidence_anchors(question):
        fallback.append(anchor)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in fallback:
        key = re.sub(r"\s+", "", item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:REWRITE_MAX_QUERIES] or [question]
