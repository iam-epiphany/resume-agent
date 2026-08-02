# -*- coding: utf-8 -*-
"""面试官意图路由：规则优先 + LLM 兜底，把问题分类到 7 类面试意图。

设计动机（技术亮点 2，参考 RAGFlow 语义路由思路）：
普通 RAG 对一切问题一视同仁地"检索+生成"，无关问题硬检索 → 高幻觉、高延迟；
本模块把 LLM 预算花在刀刃上——寒暄/无关话题零检索零生成，自我介绍单查询锚定，
项目深挖多查询召回，并在规则无法判定时用一次小 LLM 调用兜底。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from backend.app.core.config import (
    INTENT_ROUTER_API_KEY,
    INTENT_ROUTER_BASE_URL,
    INTENT_ROUTER_ENABLED,
    INTENT_ROUTER_MAX_TOKENS,
    INTENT_ROUTER_MODEL,
    INTENT_ROUTER_PROVIDER,
    INTENT_ROUTER_RESPONSE_FORMAT,
    INTENT_ROUTER_TIMEOUT_SECONDS,
)
from backend.app.services.llm_client import ChatCompletionConfig, ChatCompletionError, chat_completion_content

# 7 类意图
INTENT_SELF_INTRO = "self_intro"          # 自我介绍类
INTENT_PROJECT_DEEP_DIVE = "project_deep_dive"  # 项目深挖类
INTENT_TECH_QUIZ = "tech_quiz"            # 技术八股/原理类
INTENT_HR_QUALITY = "hr_quality"          # HR 素质/动机/规划类
INTENT_RESUME_DETAIL = "resume_detail"    # 简历细节类（默认兜底）
INTENT_GREETING = "greeting"              # 寒暄类
INTENT_OFF_TOPIC = "off_topic"            # 完全无关类

ALL_INTENTS = (
    INTENT_SELF_INTRO,
    INTENT_PROJECT_DEEP_DIVE,
    INTENT_TECH_QUIZ,
    INTENT_HR_QUALITY,
    INTENT_RESUME_DETAIL,
    INTENT_GREETING,
    INTENT_OFF_TOPIC,
)

# 一级强规则词表（零 LLM 成本；命中即返回）。顺序即优先级。
_RULE_ORDER = (
    INTENT_GREETING,
    INTENT_OFF_TOPIC,
    INTENT_SELF_INTRO,
    INTENT_RESUME_DETAIL,
)
_RULE_TERMS: dict[str, tuple[str, ...]] = {
    INTENT_GREETING: (
        "你好", "您好", "嗨", "哈喽", "hello", "hi", "谢谢", "感谢", "辛苦了",
        "在吗", "在不在", "晚上好", "早上好", "下午好", "再见", "拜拜", "好的", "嗯嗯",
    ),
    INTENT_OFF_TOPIC: (
        "天气", "股票", "彩票", "中奖", "政治", "明星", "八卦", "娱乐新闻", "电影推荐",
        "帮我写", "写一首", "讲个笑话", "菜谱", "做饭", "红烧肉", "几点", "现在时间",
        "路怎么走", "打车", "出租车", "房价", "菜价", "今天几号", "生日快乐",
    ),
    INTENT_SELF_INTRO: (
        "介绍你自己", "自我介绍", "介绍一下你", "你是谁", "讲讲你", "你的情况",
        "简单介绍", "自我介绍一下", "介绍下你", "做个自我介绍",
    ),
    INTENT_RESUME_DETAIL: (
        "学校", "专业", "成绩", "绩点", "排名", "证书", "荣誉", "获奖", "竞赛", "奖项",
        "软考", "六级", "四级", "英语", "年龄", "出生", "生日", "家乡", "籍贯",
        "毕业", "实习", "技能", "技术栈", "学历", "考研", "奖学金", "青训营",
        "电话", "邮箱", "github", "联系方式", "本科", "硕士", "研究生", "大学",
        "课程", "学分", "简历", "证书编号", "什么时候",
    ),
}


@dataclass(frozen=True)
class RetrievalStrategy:
    """意图 → 检索策略映射。"""

    anchor_documents: tuple[str, ...] = ()   # 锚定文档（文件名/书名号），空 = 不锚定
    rewrite: bool = False                    # 是否需要 LLM 查询改写
    min_prompt_chunks: int = 4               # prompt 最少 chunk 数
    direct_generation_allowed: bool = True   # 允许"直接生成 + 推测标注"
    polite_redirect: bool = False            # 跳过检索，直接礼貌转移


@dataclass(frozen=True)
class IntentResult:
    intent: str
    classifier: str          # "rule" | "llm" | "fallback"
    confidence: float
    reason: str | None = None
    strategy: RetrievalStrategy = field(default_factory=RetrievalStrategy)


_STRATEGIES: dict[str, RetrievalStrategy] = {
    INTENT_SELF_INTRO: RetrievalStrategy(
        anchor_documents=("自我介绍", "简历文字版"),
        min_prompt_chunks=4,
    ),
    INTENT_PROJECT_DEEP_DIVE: RetrievalStrategy(
        rewrite=True,
        min_prompt_chunks=6,
    ),
    INTENT_TECH_QUIZ: RetrievalStrategy(
        rewrite=True,
        min_prompt_chunks=4,
    ),
    INTENT_HR_QUALITY: RetrievalStrategy(
        anchor_documents=("求职动机与职业规划", "个人特质与兴趣爱好", "自我介绍"),
        min_prompt_chunks=4,
    ),
    INTENT_RESUME_DETAIL: RetrievalStrategy(min_prompt_chunks=4),
    INTENT_GREETING: RetrievalStrategy(
        polite_redirect=True,
        direct_generation_allowed=False,
    ),
    INTENT_OFF_TOPIC: RetrievalStrategy(
        polite_redirect=True,
        direct_generation_allowed=False,
    ),
}

_CLASSIFY_PROMPT = """你是简历问答系统的意图分类器。用户是面试官，问题都围绕简历主人公（张三，计算机方向求职者）。
把问题分类到以下 7 类之一：
- self_intro: 让候选人做自我介绍/概括自己的问题
- project_deep_dive: 深挖某个项目（秒杀/外卖/RAG 问答/校园助手/密码算法）的技术细节、难点、实现、复盘
- tech_quiz: 计算机技术原理/八股问题（Java/MySQL/Redis/Kafka/算法/网络等，与候选人本人经历无关）
- hr_quality: 个人素质、优缺点、动机、职业规划、薪资、爱好、抗压、团队等
- resume_detail: 询问简历上的具体事实（教育/成绩/证书/竞赛/技能/联系方式/时间线）
- greeting: 寒暄问候
- off_topic: 与简历和计算机完全无关的话题（天气/烹饪/政治/娱乐等）
只输出 JSON：{{"intent": "<类别>", "confidence": 0.0-1.0, "reason": "<一句话理由>"}}"""


def _rule_classify(question: str) -> IntentResult | None:
    normalized = re.sub(r"\s+", "", question).lower()
    for intent in _RULE_ORDER:
        for term in _RULE_TERMS[intent]:
            if term in normalized:
                return IntentResult(
                    intent=intent,
                    classifier="rule",
                    confidence=0.9,
                    reason=f"命中规则词：{term}",
                    strategy=_STRATEGIES[intent],
                )
    return None


def _llm_config() -> ChatCompletionConfig:
    return ChatCompletionConfig(
        provider=INTENT_ROUTER_PROVIDER,
        api_key=INTENT_ROUTER_API_KEY,
        base_url=INTENT_ROUTER_BASE_URL,
        model=INTENT_ROUTER_MODEL,
        timeout_seconds=INTENT_ROUTER_TIMEOUT_SECONDS,
        response_format=INTENT_ROUTER_RESPONSE_FORMAT,
    )


def _llm_classify(question: str) -> IntentResult | None:
    messages = [
        {"role": "system", "content": _CLASSIFY_PROMPT},
        {"role": "user", "content": f"问题：{question}"},
    ]
    try:
        content = chat_completion_content(
            _llm_config(), messages, temperature=0, max_tokens=INTENT_ROUTER_MAX_TOKENS
        )
    except ChatCompletionError:
        return None
    try:
        payload = json.loads(_extract_json_object(content))
    except (json.JSONDecodeError, ValueError):
        return None
    intent = str(payload.get("intent") or "").strip().lower()
    if intent not in ALL_INTENTS:
        return None
    confidence = float(payload.get("confidence") or 0.5)
    return IntentResult(
        intent=intent,
        classifier="llm",
        confidence=max(0.0, min(1.0, confidence)),
        reason=str(payload.get("reason") or "")[:200] or None,
        strategy=_STRATEGIES[intent],
    )


def _extract_json_object(content: str) -> str:
    """从 LLM 输出中截取首个 JSON 对象（容错：输出可能夹带前后缀）。"""
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object")
    return content[start : end + 1]


def classify_intent(question: str) -> IntentResult:
    """分类问题意图。规则优先；未命中走 LLM；LLM 失败兜底 resume_detail。"""
    if INTENT_ROUTER_ENABLED:
        rule_hit = _rule_classify(question)
        if rule_hit is not None:
            return rule_hit
        llm_hit = _llm_classify(question)
        if llm_hit is not None:
            return llm_hit
    return IntentResult(
        intent=INTENT_RESUME_DETAIL,
        classifier="fallback",
        confidence=0.3,
        reason="意图分类失败，回退简历细节类",
        strategy=_STRATEGIES[INTENT_RESUME_DETAIL],
    )
