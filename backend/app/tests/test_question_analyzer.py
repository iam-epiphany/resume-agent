# -*- coding: utf-8 -*-
"""确定性问题分析器与确定性检索规划测试。

覆盖：
- 枚举/补集问句识别（含"还有哪些其他X/除了…之外/剩下的X"，排除"还有什么想问的"误判）
- 排除实体提取（显式实体 vs 指代代词需上下文）
- 对象类别识别与文档清单归类
- 会话已知实体解析（"这三个"→ 上轮回答中出现的项目）
- planner 确定性路径：枚举问句 + 文档清单 → 每对象一个 aspect（anchor_documents）+ 排除文档
"""

from __future__ import annotations

import json

import pytest

from backend.app.services import query_planner_service
from backend.app.services.question_analyzer import (
    analyze_question,
    classify_object_docs,
    detect_object_category,
    extract_excluded_entities,
    is_complement_question,
    is_enumerative_question,
    object_name_from_filename,
    resolve_known_entities,
)

CATALOG = [
    ("项目介绍_高并发电商秒杀平台.md", "高并发电商秒杀平台"),
    ("项目介绍_外卖平台.md", "外卖平台"),
    ("项目介绍_REV密码算法.md", "REV 密码算法"),
    ("项目介绍_ReguMate.md", "ReguMate-Agent"),
    ("项目介绍_EchoGuide.md", "XDU EchoGuide"),
    ("简历文字版.md", ""),
    ("技能专长.md", ""),
    ("竞赛奖项.md", ""),
]


class TestEnumerativeDetection:
    @pytest.mark.parametrize(
        "question",
        [
            "你参与过哪些项目",
            "有哪些项目",
            "介绍一下你的项目",
            "还有哪些其他项目",
            "除了这三个还有哪些项目",
            "除了高并发秒杀、外卖之外还有哪些项目",
            "剩下的项目有哪些",
            "剩余的项目",
            "你还有什么技能",
            "列举一下你的奖项",
        ],
    )
    def test_enumerative_questions(self, question: str) -> None:
        assert is_enumerative_question(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "你会打篮球吗",
            "那超卖怎么防的",
            "你的技术栈是什么",
            "课程成绩差异应该优先检查哪些问题",
            "你还有什么想问的",
            "为什么不用数据库锁",
        ],
    )
    def test_non_enumerative_questions(self, question: str) -> None:
        assert is_enumerative_question(question) is False


class TestComplementDetection:
    def test_complement_with_pronoun(self) -> None:
        entities, needs_context = extract_excluded_entities("除了这三个还有哪些项目")
        assert entities == ()
        assert needs_context is True

    def test_complement_with_modifier_pronoun(self) -> None:
        # 修饰语"你重点介绍的"不应阻止指代识别
        entities, needs_context = extract_excluded_entities(
            "除了你重点介绍的那三个项目之外，你还有哪些其他项目经历？"
        )
        assert entities == ()
        assert needs_context is True

    def test_complement_with_rewritten_modifier_pronoun(self) -> None:
        # 意图层改写形态："你刚才提到的三个项目"（无"之外"、带"提到的"修饰）
        entities, needs_context = extract_excluded_entities(
            "除了你刚才提到的三个项目，你还有哪些其他项目经历？"
        )
        assert entities == ()
        assert needs_context is True

    def test_complement_with_explicit_entities(self) -> None:
        entities, needs_context = extract_excluded_entities(
            "除了高并发秒杀、外卖平台之外，还有哪些项目？"
        )
        assert needs_context is False
        assert entities == ("高并发秒杀", "外卖平台")

    def test_complement_flag(self) -> None:
        assert is_complement_question("除了这三个还有哪些项目") is True
        assert is_complement_question("剩下的项目有哪些") is True
        assert is_complement_question("你的项目经历") is False


class TestCategoryAndCatalog:
    def test_object_category(self) -> None:
        assert detect_object_category("还有哪些项目") == "项目"
        assert detect_object_category("有什么技能") == "技能"
        assert detect_object_category("获得过哪些奖项") == "奖项"
        assert detect_object_category("你的技术栈是什么") is None

    def test_classify_object_docs(self) -> None:
        docs = classify_object_docs(CATALOG, "项目")
        assert docs == (
            "项目介绍_高并发电商秒杀平台.md",
            "项目介绍_外卖平台.md",
            "项目介绍_REV密码算法.md",
            "项目介绍_ReguMate.md",
            "项目介绍_EchoGuide.md",
        )

    def test_classify_excludes_resume_documents(self) -> None:
        docs = classify_object_docs(CATALOG, "项目")
        assert "简历文字版.md" not in docs

    def test_object_name_from_filename(self) -> None:
        assert object_name_from_filename("项目介绍_外卖平台.md") == "外卖平台"


class TestKnownEntities:
    def test_resolve_known_entities_from_excerpt(self) -> None:
        memory = {
            "question": "你的项目经历",
            "answer_excerpt": (
                "第一个是高并发电商秒杀项目，2026年2月到6月。"
                "第二个是校际合作的抗量子加密算法项目（REV）。"
                "第三个是ReguMate项目，文档解析和问答系统。"
            ),
        }
        known = resolve_known_entities(memory, tuple(doc for doc, _ in CATALOG[:5]))
        assert known == (
            "项目介绍_高并发电商秒杀平台.md",
            "项目介绍_REV密码算法.md",
            "项目介绍_ReguMate.md",
        )

    def test_no_memory_returns_empty(self) -> None:
        assert resolve_known_entities(None, ()) == ()


class TestAnalyzeQuestion:
    def test_failure_case_analysis(self) -> None:
        memory = {
            "question": "你的项目经历",
            "answer_excerpt": "第一个是高并发电商秒杀项目。第二个是REV密码算法项目。第三个是ReguMate项目。",
        }
        analysis = analyze_question(
            "除了你重点介绍的那三个项目之外，你还有哪些其他项目经历？",
            catalog=CATALOG,
            memory_context=memory,
        )
        assert analysis.enumerative is True
        assert analysis.complement is True
        assert analysis.needs_context_entities is True
        assert analysis.object_category == "项目"
        assert len(analysis.object_docs) == 5
        assert len(analysis.known_entities) == 3
        debug = analysis.to_debug_dict()
        assert debug["enumerative"] is True
        assert debug["known_entities"][0].startswith("项目介绍_")


class TestDeterministicPlanning:
    def test_plan_query_builds_object_aspects_from_catalog(self, monkeypatch) -> None:
        monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
        plan = query_planner_service.plan_query(
            "除了这三个还有哪些项目",
            catalog=CATALOG,
        )
        assert plan.enumerative is True
        assert plan.planner == "deterministic_catalog"
        # 每个项目文档一个 aspect，且带 anchor_documents
        assert len(plan.aspects) == 5
        for aspect in plan.aspects:
            assert aspect.aspect_id.startswith("object_")
            assert len(aspect.anchor_documents) == 1
            assert aspect.anchor_documents[0].startswith("项目介绍_")
            assert aspect.keywords[0] == object_name_from_filename(aspect.anchor_documents[0])
        # 无会话上下文 → 排除文档为空（无法解析"这三个"）
        assert plan.excluded_documents == ()

    def test_plan_query_resolves_known_excluded_documents(self, monkeypatch) -> None:
        monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
        plan = query_planner_service.plan_query(
            "除了你重点介绍的那三个项目之外，你还有哪些其他项目经历？",
            catalog=CATALOG,
            memory_context={
                "question": "你的项目经历",
                "answer_excerpt": "第一个是高并发电商秒杀项目。第二个是REV密码算法项目。第三个是ReguMate项目。",
            },
        )
        assert plan.enumerative is True
        assert plan.excluded_documents == (
            "项目介绍_高并发电商秒杀平台.md",
            "项目介绍_REV密码算法.md",
            "项目介绍_ReguMate.md",
        )
        # 被排除对象也在 aspect 清单中（软降权而非硬排除），保证"还有哪些"的对照材料可用
        assert len(plan.aspects) == 5

    def test_plan_query_explicit_excluded_entities(self, monkeypatch) -> None:
        monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
        plan = query_planner_service.plan_query(
            "除了秒杀项目之外，还有哪些项目",
            catalog=CATALOG,
        )
        assert plan.excluded_documents == ("项目介绍_高并发电商秒杀平台.md",)

    def test_plan_query_llm_enhancement_merges_queries(self, monkeypatch) -> None:
        """LLM 增强只补充查询措辞，不改变对象结构（对象数量/顺序保持不变）。"""
        monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "test-key")
        llm_payload = {
            "aspects": [
                {
                    "aspect_id": "object_1",
                    "question": "高并发电商秒杀平台 项目介绍（项目介绍_高并发电商秒杀平台.md）",
                    "evidence_need": "高并发电商秒杀平台项目介绍",
                    "search_queries": [
                        {
                            "query": "秒杀平台 缓存 分布式锁 压测结果",
                            "query_type": "document_style_statement",
                            "rationale": "补充技术细节查询",
                        }
                    ],
                    "keywords": ["高并发电商秒杀平台"],
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {"choices": [{"message": {"content": json.dumps(llm_payload, ensure_ascii=False)}}]},
                    ensure_ascii=False,
                ).encode("utf-8")

        monkeypatch.setattr(
            query_planner_service.urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: FakeResponse(),
        )
        plan = query_planner_service.plan_query(
            "还有哪些项目",
            catalog=[("项目介绍_高并发电商秒杀平台.md", "高并发电商秒杀平台")],
        )
        assert len(plan.aspects) == 1
        aspect = plan.aspects[0]
        # 确定性查询（3 条）+ LLM 补充（1 条），共 4 条
        assert len(aspect.search_queries) == 4
        assert aspect.search_queries[-1].query == "秒杀平台 缓存 分布式锁 压测结果"
        assert aspect.anchor_documents == ("项目介绍_高并发电商秒杀平台.md",)
        assert plan.planner.startswith("openai_compatible:")
        assert plan.fallback_used is False

    def test_plan_query_llm_enhancement_failure_falls_back_to_deterministic(self, monkeypatch) -> None:
        monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "test-key")

        def fake_urlopen(*_args, **_kwargs):
            raise OSError("network down")

        monkeypatch.setattr(query_planner_service.urllib.request, "urlopen", fake_urlopen)
        plan = query_planner_service.plan_query(
            "还有哪些项目",
            catalog=[("项目介绍_外卖平台.md", "外卖平台")],
        )
        # LLM 增强失败 → 降级为纯确定性 aspect，仍然完整可用
        assert plan.planner == "deterministic_catalog"
        assert len(plan.aspects) == 1
        assert plan.aspects[0].anchor_documents == ("项目介绍_外卖平台.md",)
        assert len(plan.aspects[0].search_queries) == 3
