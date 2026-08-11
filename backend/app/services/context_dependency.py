"""追问依赖检测：判断当前问题是否依赖上一轮对话上下文。

简历面试场景中，追问形态有限且规律：指代代词/序数指代/省略主语/衔接词。
本模块用确定性规则判断"当前问题是否真的需要上一轮上下文才能理解"，
供 intent fast path 与 planner fast path 共用——有历史消息但问题独立时，
仍可走 fast path 跳过 LLM 消解/规划。

设计原则：
- 保守：拿不准就判为"需要上下文"（走 LLM 消解），避免追问消解错误
- 不依赖关键词表逐条命中，而是识别"问题是否自足"的结构特征
- 简历领域锚点（项目/技术/学校等）+ 完整表达的问题视为自足
"""

from __future__ import annotations

import re

# 指代代词：出现在问题中说明可能指代前文对象
_REFERENTIAL_TERMS = (
    "那", "这个", "这个项目", "它", "它们", "这些", "那些", "刚才", "之前说",
    "上述", "上轮", "之前提到", "刚才提到", "那样", "这样", "另一个", "那个",
    "另外", "其中", "其他的", "剩下的", "还有一个", "它的", "那里的",
)

# 明显衔接/追问词：问题以这些开头（或包含）且整体较短，几乎必是追问
_FOLLOWUP_PREFIXES = (
    "那", "然后", "接着", "再", "还有", "那怎么", "那为什么", "那什么",
    "为什么不用", "为什么没有", "怎么解决", "怎么办", "如何解决", "怎么防",
    "哪些是", "哪个是",
)

# 序数/指代词：第二个/第三个/另一个/那个…（对象省略）
_ORDINAL_REFERENCE = re.compile(r"(第[一二三四五六七八九十\d]+[个项门])|(另一个)|(那[个只座所])")

# 简历领域锚点：含这些词的问题通常是自足的技术/简历问题。
# 同时被 intent fast path 用于归类 resume_qa（单一事实源，避免两处维护）。
RESUME_ANCHOR_TERMS = (
    "项目", "简历", "技术", "教育", "学校", "专业", "技能", "证书", "荣誉",
    "奖项", "竞赛", "实习", "经历", "意向", "城市", "岗位", "成绩", "绩点",
    "毕业", "课程", "奖学金", "开发", "语言", "框架", "数据库", "架构",
    "HashMap", "Redis", "Kafka", "Java", "Python", "MySQL", "Redisson",
    "缓存", "线程", "锁", "超卖", "秒杀", "微信支付", "回调", "NTT", "Agent",
)

_GREETING_PHRASES = frozenset(
    {"你好", "您好", "哈喽", "嗨", "hello", "hi", "在吗", "在不在", "谢谢", "辛苦了", "再见"}
)


def needs_context_resolution(question: str) -> bool:
    """判断当前问题是否依赖上一轮对话上下文。

    True  → 需要上下文（追问），交给 LLM 消解/规划
    False → 问题自足（独立），可走 fast path
    """
    stripped = (question or "").strip()
    if not stripped:
        return True

    # 整句问候短语不依赖上下文（greeting 分支自足）
    if stripped in _GREETING_PHRASES:
        return False

    # 短句 + "呢/啊" 结尾（"那 Redis 呢？"、"那你呢？"）——典型追问省略
    if len(stripped) <= 12 and stripped.endswith(("呢", "啊", "吗")):
        return True
    # 极短问题（<= 4 字，如"爱好""籍贯""薪资"）——依赖前文指代的对象
    if len(stripped) <= 4:
        return True

    # 序数/对象指代（"第二个项目""另一个""那个"）——指向前文对象
    if _ORDINAL_REFERENCE.search(stripped):
        return True

    # 衔接/追问词开头
    if any(stripped.startswith(prefix) for prefix in _FOLLOWUP_PREFIXES):
        return True

    # 含指代代词且不含自足锚点 → 高度可疑
    has_referential = any(term in stripped for term in _REFERENTIAL_TERMS)
    has_anchor = any(term in stripped for term in RESUME_ANCHOR_TERMS)
    if has_referential and not has_anchor:
        return True

    # 其余（含简历锚点的完整问题）视为自足
    return False


def is_greeting_phrase(question: str) -> bool:
    """整句问候短语判定：仅"整句就是问候"的短句命中，避免误伤。

    "你好，介绍下你的项目"不是整句问候（长度/内容不匹配）→ 不算；
    "爱好""籍贯""项目"等短词不是问候 → 不算（应走 resume_qa 或 LLM）。
    """
    stripped = (question or "").strip()
    if not stripped:
        return False
    return stripped in _GREETING_PHRASES
