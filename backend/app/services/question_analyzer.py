# -*- coding: utf-8 -*-
"""确定性问题分析器：枚举/补集问句检测 + 对象类别识别 + 会话已知实体解析。

设计动机（检索规划层的确定性底座）：
检索规划此前完全依赖 LLM 猜测——拆几个 aspect、keywords 是什么、"除了这三个
还有哪些项目"里"这三个"指代哪些对象，全靠模型自由发挥；猜错（如臆测排除
名单）后错误会一路下传到召回与选择，这是列举/补集类问题答错的根因之一。

本模块把"问句形态分析"改为确定性规则：
- 是否在列举多个对象（枚举问句）
- 是否带排除语义（补集问句，"除了 X 之外还有哪些"）
- 排除对象是显式列出还是指代代词（"这三个/那三个"→ 需会话上下文解析）
- 对象属于哪个文档类别（项目/技能/奖项…），据此从文档清单归类对象文档
- 用上一轮回答摘录与文档名做重叠匹配，解析指代实体

LLM 只负责查询措辞增强，不再决定检索计划的结构。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 枚举问句的对象名词（与 query_planner 的领域词表保持一致）
OBJECT_NOUNS = r"(?:项目|技能|奖项|荣誉|课程|活动|经历|证书|竞赛|比赛|成就|特长|优点|作品)"

# 对象名词 → 文档类别（用于从文档清单归类对象文档；顺序敏感，先匹配更具体的）
NOUN_TO_CATEGORY: tuple[tuple[str, str], ...] = (
    ("项目", "项目"),
    ("技能", "技能"),
    ("竞赛", "奖项"),
    ("比赛", "奖项"),
    ("奖项", "奖项"),
    ("荣誉", "荣誉"),
    ("证书", "证书"),
    ("课程", "课程"),
    ("教育", "教育"),
    ("活动", "活动"),
    ("经历", "经历"),
)

# 文档清单分类规则：文件名/标题 子串 → 对象类别（顺序敏感：先匹配更具体的）
DOCUMENT_CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("项目介绍", "项目"),
    ("项目", "项目"),
    ("技能", "技能"),
    ("竞赛", "奖项"),
    ("奖项", "奖项"),
    ("荣誉", "荣誉"),
    ("证书", "证书"),
    ("课程", "课程"),
    ("教育", "教育"),
)

# 文档名中的干扰词：这些文档不是"对象文档"（简历里虽有项目经历，但不是项目文档）
NON_OBJECT_FILENAME_MARKERS = ("简历", "自我介绍", "求职", "个人特质")

# 指代代词："这三个/那三个/这三个项目/你刚才提到的三个项目" → 需要会话上下文解析。
# 匹配"量词短语 + 可选对象名词"形态（含/不含"这/那"），显式实体名（"秒杀项目"）不命中。
_PRONOMINAL_PATTERN = re.compile(
    r"^(?:[这那])?[一二三四五六七八九十两0-9]*(?:个|些|类|俩|几)个?(?:"
    + OBJECT_NOUNS
    + r")?$"
)
# 指代前的修饰成分（"你重点介绍的那三个项目"/"你刚才提到的三个项目" → "那三个项目"/"三个项目"）
_CORE_MODIFIER_PATTERN = re.compile(
    r"^(?:你|您|我)?(?:刚才|前面|之前|刚刚|上轮|上一轮)?(?:重点|详细|着重)?"
    r"(?:介绍|说|讲|提)(?:过|的|了|到|到的)?(?:的)?"
)
# 实体名中的通用词（匹配文档名时不计入，防止"项目/平台"等泛词误命中所有对象）
_GENERIC_ENTITY_TERMS = frozenset(
    "项目 平台 系统 介绍 经历 开发 技术 设计 实现 项目经历 项目介绍 内容".split()
)

_STOP_TERMS = frozenset(
    "怎么 什么 如何 为何 哪些 哪个 为什么 是否 可以 请问 还有 以及 因为 所以 但是 没有 "
    "就是 不是 都是 这个 那个 我们 你们 他们 一下 一个 一些 怎样 各自 分别 请问 "
    "说明 介绍 讲讲 说说 简单 具体 详细 大概 大约 目前 现在 之前 之后 已经 正在 "
    "除了 之外 以外 另外 其他 别的 剩下 其余 一个 三个 两个".split()
)


@dataclass(frozen=True)
class QuestionAnalysis:
    """确定性问句分析的输出，供检索规划（aspect 拆分）与选择层（配额优先级）使用。"""

    enumerative: bool = False                 # 枚举问句（多对象列举）
    complement: bool = False                  # 补集问句（"除了 X 之外还有哪些"）
    excluded_entities: tuple[str, ...] = ()   # 补集中显式列出的实体（文档名/对象名）
    needs_context_entities: bool = False      # 排除对象是"这三个/那些"等指代，需上下文解析
    object_category: str | None = None        # 对象类别（项目/技能/奖项/...）
    object_docs: tuple[str, ...] = ()         # 从文档清单按类别归类出的对象文档文件名
    keywords: tuple[str, ...] = ()            # 问题术语（去停用词）
    known_entities: tuple[str, ...] = ()      # 会话上下文中解析出的已知对象文档名

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "enumerative": self.enumerative,
            "complement": self.complement,
            "excluded_entities": list(self.excluded_entities),
            "needs_context_entities": self.needs_context_entities,
            "object_category": self.object_category,
            "object_docs": list(self.object_docs),
            "keywords": list(self.keywords),
            "known_entities": list(self.known_entities),
        }


def is_enumerative_question(question: str) -> bool:
    """枚举问句检测：识别"列举多个对象"的提问形态。

    比旧版正则多覆盖补集/剩余枚举形态：
    "除了…之外还有哪些"、"还有哪些其他X"、"还做过哪些X"、"剩下的X"。
    """
    compact = re.sub(r"\s+", "", str(question or ""))
    if not compact:
        return False
    # 1. "哪些/有哪/什么" + 对象（允许"其他/别的"插入："哪些其他项目"）
    if re.search(r"(?:哪些|有哪|什么)(?:的)?(?:其他|别的|其余)?(?:" + OBJECT_NOUNS + r")", compact):
        return True
    # 2. 对象 + "有哪些/有什么"
    if re.search(OBJECT_NOUNS + r"(?:有哪些|有什么)", compact):
        return True
    # 3. 列举类动词
    if re.search(r"(?:列举|罗列|列出)", compact):
        return True
    # 4. "介绍/说说…你的对象"（"介绍一下你的项目"）
    if re.search(
        r"(?:介绍|说说|讲讲|说一下)(?:一下|下)?(?:你的|您)?(?:全部|所有)?"
        r"(?:参与|做过|获得|掌握|取得|参加)?(?:过的?)?(?:" + OBJECT_NOUNS + r")",
        compact,
    ):
        return True
    # 5. 补集枚举："除了X之外(还)有哪些Y"、"除了X还有哪些Y"、"还有哪些其他Y"、"剩下的Y"
    if re.search(
        r"除(?:了)?.{0,24}?(?:之外|以外|外).{0,12}(?:还有|还|另外)?(?:哪些|什么)", compact
    ):
        return True
    if re.search(r"除(?:了)?[^。？?]{1,20}?还有(?:哪些|什么)", compact):
        return True
    # "还有哪些Y/还做过哪些Y/另外有哪些Y/剩下的Y"（Y 为对象名词，"还有什么想问的"不命中）
    if re.search(
        r"(?:还有|还做过|另外|其余|剩下|剩余).{0,8}(?:哪些|什么)(?:的)?(?:其他|别的)?"
        + OBJECT_NOUNS,
        compact,
    ):
        return True
    if re.search(r"(?:其余|剩下|剩余)(?:的)?" + OBJECT_NOUNS, compact):
        return True
    if re.search(r"哪些其他(?:" + OBJECT_NOUNS + r")", compact):
        return True
    return False


def is_complement_question(question: str) -> bool:
    """补集问句检测："除了 X 之外还有哪些" / "剩下的 X" 等排除语义。"""
    compact = re.sub(r"\s+", "", str(question or ""))
    if not compact:
        return False
    if re.search(r"除(?:了)?[^。？?]{1,30}?(?:之外|以外|外)", compact) and re.search(
        r"(?:还有|另外|还|剩下|其余|哪些|什么)", compact
    ):
        return True
    if re.search(r"除(?:了)?[^。？?]{1,20}?还有(?:哪些|什么)", compact):
        return True
    if re.search(r"(?:其余|剩下|剩余)(?:的)?" + OBJECT_NOUNS, compact):
        return True
    return False


def extract_excluded_entities(question: str) -> tuple[tuple[str, ...], bool]:
    """提取补集问句中被排除的对象（"除了高并发秒杀、外卖之外" → 两个实体）。

    返回 (实体名列表, 是否需要上下文解析)。指代代词（"这三个"或"你重点介绍的那三个项目"）
    不产出实体，置 needs_context=True，由调用方用会话已知实体解析。
    """
    compact = re.sub(r"\s+", "", str(question or ""))
    # 终结符需要边界："外卖平台"里的"外"不能作为"除…外"的终结（否则吞掉后半段实体）
    match = re.search(r"除(?:了)?(.*?)(?:之外|以外|外)(?=[，,。；;？?！!]|还有|还要|$)", compact)
    if not match:
        match = re.search(r"除(?:了)?(.*?)还有(?:哪些|什么)", compact)
    if not match:
        return (), False
    # 仅取"除…"的第一段（"你还有"前的逗号分隔部分），防止把"你"等代词并入实体
    span = re.split(r"[，,]", match.group(1), maxsplit=1)[0].strip(" ，,、")
    if not span:
        return (), False
    parts = [
        part.strip(" ，,、；;")
        for part in re.split(r"[、，,；;]|以及|和|与|及", span)
        if part.strip(" ，,、；;")
    ]
    entities: list[str] = []
    needs_context = False
    for part in parts:
        if _is_pronominal_span(part):
            needs_context = True
            continue
        entities.append(part)
    return tuple(dict.fromkeys(entities)), needs_context


def _is_pronominal_span(span: str) -> bool:
    """判断排除对象是否是"指代代词"（如"那三个项目"、带修饰语的"你刚才提到的三个项目"）。

    规则：剥掉"你刚才提到的/重点介绍的"等修饰后，若剩余只是"量词短语(+对象名词)"
    （"三个项目"/"那三个"/"几个"），即为指代——具体指代哪些对象需会话上下文解析。
    """
    cleaned = _CORE_MODIFIER_PATTERN.sub("", span)
    cleaned = re.sub(OBJECT_NOUNS + r"个?$", "", cleaned)
    return bool(cleaned) and _PRONOMINAL_PATTERN.fullmatch(cleaned) is not None


def detect_object_category(question: str) -> str | None:
    """识别问题中的对象类别（项目/技能/奖项/...），用于从文档清单归类对象文档。"""
    compact = re.sub(r"\s+", "", str(question or ""))
    for noun, category in NOUN_TO_CATEGORY:
        if noun in compact:
            return category
    return None


def classify_object_docs(catalog: list[tuple[str, str]] | None, category: str | None) -> tuple[str, ...]:
    """按类别从文档清单中归类对象文档（如项目类 → 项目介绍_*.md）。"""
    if not catalog or not category:
        return ()
    matched: list[str] = []
    for filename, title in catalog:
        if any(marker in filename for marker in NON_OBJECT_FILENAME_MARKERS):
            continue
        haystack = f"{filename} {title or ''}"
        if any(pattern in haystack for pattern, doc_category in DOCUMENT_CATEGORY_PATTERNS
               if doc_category == category):
            matched.append(filename)
    return tuple(matched)


def object_name_from_filename(filename: str) -> str:
    """从对象文档文件名提取对象名（"项目介绍_高并发电商秒杀平台.md" → "高并发电商秒杀平台"）。"""
    name = filename
    for prefix in ("项目介绍_", "技能", "奖项", "荣誉", "证书"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in (".md", ".txt", ".pdf", ".docx", ".doc"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


def resolve_known_entities(
    memory_context: dict | None,
    object_docs: tuple[str, ...],
) -> tuple[str, ...]:
    """用上一轮回答摘录与对象文档名做重叠匹配，解析指代实体（"这三个"→ 具体项目）。

    重叠度量：归一化后互为子串 → 1.0；否则公共双字符比例 ≥ 0.5 视为同一实体。
    """
    if not memory_context or not object_docs:
        return ()
    excerpt = re.sub(r"\s+", "", str(memory_context.get("answer_excerpt") or ""))
    if not excerpt:
        return ()
    known: list[str] = []
    for doc in object_docs:
        name = object_name_from_filename(doc)
        if not name or len(name) < 3:
            continue
        if _overlap_ratio(name, excerpt) >= 0.5:
            known.append(doc)
    return tuple(known)


def analyze_question(
    question: str,
    *,
    catalog: list[tuple[str, str]] | None = None,
    memory_context: dict | None = None,
) -> QuestionAnalysis:
    """对提问做确定性分析，输出供规划/选择层使用的结构化结论。"""
    enumerative = is_enumerative_question(question)
    complement = is_complement_question(question)
    excluded_entities, needs_context = extract_excluded_entities(question)
    category = detect_object_category(question)
    object_docs = classify_object_docs(catalog, category) if (enumerative or complement) else ()
    known_entities = resolve_known_entities(memory_context, object_docs)
    return QuestionAnalysis(
        enumerative=enumerative,
        complement=complement,
        excluded_entities=excluded_entities,
        needs_context_entities=needs_context,
        object_category=category,
        object_docs=object_docs,
        keywords=_keywords_from_text(question),
        known_entities=known_entities,
    )


def _overlap_ratio(left: str, right: str) -> float:
    normalized_left = re.sub(r"\s+", "", left)
    normalized_right = re.sub(r"\s+", "", right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 1.0
    left_bigrams = {
        normalized_left[index : index + 2] for index in range(max(len(normalized_left) - 1, 0))
    }
    right_bigrams = {
        normalized_right[index : index + 2] for index in range(max(len(normalized_right) - 1, 0))
    }
    if not left_bigrams:
        return 0.0
    return len(left_bigrams & right_bigrams) / max(len(left_bigrams), 1)


def _keywords_from_text(text: str) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z0-9_./%-]{2,}|[\u4e00-\u9fff]{2,12}", str(text or ""))
    return tuple(dict.fromkeys(token for token in tokens if token not in _STOP_TERMS))[:8]
