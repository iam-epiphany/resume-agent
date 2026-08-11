"""context_dependency（追问依赖检测）测试。"""

import pytest

from backend.app.services.context_dependency import (
    is_greeting_phrase,
    needs_context_resolution,
)


@pytest.mark.parametrize(
    "question",
    [
        # 指代/省略/衔接 → 需要上下文
        "那怎么解决超卖的？",
        "那 Redis 呢？",
        "第二个项目用了什么技术？",
        "另一个项目呢？",
        "还有哪些？",
        "它怎么实现的？",
        "爱好",
        "籍贯",
        "薪资",
        "项目",
        "为什么不用数据库锁？",
    ],
)
def test_questions_that_need_context(question: str) -> None:
    assert needs_context_resolution(question) is True, question


@pytest.mark.parametrize(
    "question",
    [
        # 独立自足问题 → 不需要上下文
        "HashMap 为什么线程不安全",
        "你有哪些项目？",
        "介绍一下你的秒杀项目",
        "你的技术栈是什么",
        "为什么 B+ 树适合做索引",
        "Kafka 怎么保证消息不丢",
        "你好",
        "谢谢",
        "在吗",
        "蓝桥杯一等奖是什么水平",
    ],
)
def test_questions_that_are_self_contained(question: str) -> None:
    assert needs_context_resolution(question) is False, question


def test_empty_question_needs_context() -> None:
    assert needs_context_resolution("") is True
    assert needs_context_resolution("   ") is True


def test_greeting_phrase_matches_only_whole_sentence() -> None:
    assert is_greeting_phrase("你好") is True
    assert is_greeting_phrase("谢谢") is True
    assert is_greeting_phrase("在吗") is True
    # 短词不是问候
    assert is_greeting_phrase("爱好") is False
    assert is_greeting_phrase("项目") is False
    assert is_greeting_phrase("薪资") is False
    # 混合问题不是整句问候
    assert is_greeting_phrase("你好，介绍下你的项目") is False
    assert is_greeting_phrase("") is False
