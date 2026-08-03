import json

from backend.app.services import query_planner_service
from backend.app.services.query_planner_service import plan_query


def test_llm_postprocessor_splits_two_coordinated_requirement_categories() -> None:
    assert query_planner_service._coordinated_aspect_questions(
        "项目介绍还应说明技术选型原则和沟通对象"
    ) == ["项目介绍还应说明技术选型原则", "项目介绍还应说明沟通对象"]
    assert query_planner_service._coordinated_aspect_questions(
        "证书等级相对考核标准的门槛及掌握程度基本原则"
    ) == ["证书等级相对考核标准的门槛", "掌握程度基本原则"]
    assert query_planner_service._coordinated_aspect_questions(
        "说明蓝桥杯大赛奖项定义及获奖要求"
    ) == ["说明蓝桥杯大赛奖项定义", "蓝桥杯大赛奖项的获奖要求"]


def test_llm_postprocessor_splits_bare_threshold_and_order_categories() -> None:
    question = "技能掌握程度触发阈值与证书等级顺序是什么？"

    assert query_planner_service._coordinated_aspect_questions(question) == [
        "技能掌握程度触发阈值是什么？",
        "证书等级顺序是什么？",
    ]


def test_query_budget_counts_nested_coordination_across_semicolon_parts() -> None:
    budget = query_planner_service.plan_query_budget(
        "分别回答：技能掌握程度触发阈值与证书等级顺序是什么；"
        "项目经历描述范围及面试要求是什么？"
    )

    assert budget.max_aspects == 3


def test_llm_postprocessor_splits_explicit_follow_up_requirement() -> None:
    assert query_planner_service._coordinated_aspect_questions(
        "证书等级相对考核标准的门槛是多少，并说明掌握程度评分基本原则。"
    ) == [
        "证书等级相对考核标准的门槛是多少",
        "说明掌握程度评分基本原则。",
    ]


def test_llm_postprocessor_splits_strong_categories_after_planner_rephrasing() -> None:
    assert query_planner_service._coordinated_aspect_questions(
        "证书等级相对考核标准的门槛及掌握程度基本原则"
    ) == [
        "证书等级相对考核标准的门槛",
        "掌握程度基本原则",
    ]
    assert query_planner_service._coordinated_aspect_questions(
        "说明定义、口径、范围和表述要求"
    ) == ["说明定义、口径、范围和表述要求"]


def test_llm_postprocessor_preserves_shared_question_tail_when_splitting() -> None:
    assert query_planner_service._coordinated_aspect_questions(
        "项目介绍还应说明技术选型原则和沟通对象的依据是什么？"
    ) == [
        "项目介绍还应说明技术选型原则的依据是什么？",
        "项目介绍还应说明沟通对象的依据是什么？",
    ]


def test_llm_postprocessor_splits_two_topics_inside_related_requirements() -> None:
    parts = query_planner_service._coordinated_aspect_questions(
        "请综合概括《技能专长.md》中与技能掌握程度和证书等级相关的要求，"
        "不能漏掉比例、期限及禁止对象。"
    )

    assert len(parts) == 2
    assert "技能掌握程度相关的要求" in parts[0]
    assert "证书等级相关的要求" in parts[1]
    assert all("不能漏掉比例、期限及禁止对象" in part for part in parts)
    assert query_planner_service.plan_query_budget(
        "请综合概括《技能专长.md》中与技能掌握程度和证书等级相关的要求，"
        "不能漏掉比例、期限及禁止对象。"
    ).max_aspects == 2


def test_single_short_anchor_keeps_shared_classification_context() -> None:
    question = "区分竞赛奖项‘一等奖’与‘二等奖’的评定条件。"

    assert query_planner_service._question_for_single_anchor(question, "一等奖") == (
        "区分竞赛奖项“一等奖”的评定条件。"
    )
    assert query_planner_service._question_for_single_anchor(question, "二等奖") == (
        "区分竞赛奖项“二等奖”的评定条件。"
    )


def test_coordinated_definition_and_pricing_inherits_subject() -> None:
    assert query_planner_service._coordinated_aspect_questions("蓝桥杯奖项的定义与评奖原则是什么？") == [
        "蓝桥杯奖项的定义是什么？",
        "蓝桥杯奖项的评奖原则是什么？",
    ]


def test_llm_postprocessor_splits_definition_and_pricing_but_preserves_title_scope() -> None:
    parts = query_planner_service._coordinated_aspect_questions(
        "请用一段话概括《竞赛奖项.md》对蓝桥杯奖项的定义，"
        "以及获奖评定所遵循的评分标准与奖项等级。"
    )

    assert len(parts) == 2
    assert "蓝桥杯奖项的定义" in parts[0]
    assert "《竞赛奖项.md》中获奖评定" in parts[1]


def test_fallback_plan_splits_multiple_quoted_anchors_in_same_document_aspect(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
    question = (
        "请跨文件分别回答以下两个事项，并为每项给出对应来源。\n"
        "关于前一份文件：根据《项目介绍_高并发电商秒杀平台.md》，请说明与"
        "“秒杀系统支持高并发请求”相关的内容。\n"
        "关于后一份文件：根据《项目介绍_ReguMate.md》，请说明与"
        "“ReguMate基于多Agent与RAG架构”、"
        "“ReguMate支持文档问答与知识库检索”相关的内容。"
    )

    plan = plan_query(question)
    questions = [aspect.question for aspect in plan.aspects]

    assert any("秒杀系统支持高并发请求" in item for item in questions)
    assert any("ReguMate基于多Agent与RAG架构" in item for item in questions)
    assert any("ReguMate支持文档问答与知识库检索" in item for item in questions)
    assert len(plan.aspects) >= 4


def test_fallback_plan_splits_two_quoted_material_levels(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "联合核验。材料侧：区分竞赛奖项‘一等奖’与‘二等奖’的评定条件。"
        "表格侧：读取2023年10月能力评估情况表中掌握程度累计值。"
    )

    questions = [aspect.question for aspect in plan.aspects]
    assert any("一等奖" in question for question in questions)
    assert any("二等奖" in question for question in questions)
    assert any("一等奖" not in question and "二等奖" not in question for question in questions)


def test_query_planner_llm_returns_structured_multi_view_queries(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "test-key")

    llm_payload = {
        "aspects": [
            {
                "aspect_id": "grade_difference_check",
                "question": "课程成绩差异应该优先检查哪些问题？",
                "evidence_need": "课程成绩差异处理流程、优先检查原因或校验规则",
                "search_queries": [
                    {
                        "query": "课程成绩与分项成绩存在差异时应优先检查哪些原因",
                        "query_type": "semantic_question",
                        "rationale": "贴近用户意图",
                    },
                    {
                        "query": "课程成绩校验差异处理流程 成绩核算 评分标准 汇总口径",
                        "query_type": "document_style_statement",
                        "rationale": "贴近材料原文证据句",
                    },
                    {
                        "query": "课程成绩 差异 核算 评分 汇总",
                        "query_type": "keyword_anchor",
                        "rationale": "术语兜底",
                    },
                ],
                "keywords": ["课程成绩", "差异", "核算"],
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

    monkeypatch.setattr(query_planner_service.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    plan = plan_query("课程成绩差异应该优先检查哪些问题？")

    assert plan.fallback_used is False
    assert plan.planner.startswith("openai_compatible:")
    aspect = plan.aspects[0]
    assert aspect.evidence_need == "课程成绩差异处理流程、优先检查原因或校验规则"
    assert [query.query_type for query in aspect.search_queries] == [
        "semantic_question",
        "document_style_statement",
        "keyword_anchor",
    ]
    assert aspect.search_queries[0].query.endswith("应优先检查哪些原因")
    assert aspect.search_queries[0].query != "课程成绩 差异"


def test_plan_query_injects_dynamic_document_catalog_into_prompt(monkeypatch) -> None:
    """运行时知识库清单注入 planner prompt（动态数据，非硬编码）。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "test-key")
    captured: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            payload = {
                "aspects": [
                    {
                        "aspect_id": "project_exp",
                        "question": "你参与过哪些项目？",
                        "evidence_need": "项目经历",
                        "search_queries": [
                            {"query": "参与过的项目 项目经历", "query_type": "semantic_question", "rationale": "贴近用户意图"}
                        ],
                        "keywords": ["项目"],
                    }
                ]
            }
            return json.dumps({"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]}, ensure_ascii=False).encode()

    def fake_urlopen(request, *args, **kwargs):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(query_planner_service.urllib.request, "urlopen", fake_urlopen)

    catalog = "- 项目介绍_高并发电商秒杀平台.md（高并发电商秒杀平台）\n- 证书说明.md"
    plan_query("你参与过哪些项目？", catalog=catalog)

    user_content = captured["body"]["messages"][-1]["content"]
    assert "项目介绍_高并发电商秒杀平台.md" in user_content
    assert "证书说明.md" in user_content
    assert "仅用于检索规划" in user_content
    assert "输出到回答内容中" in user_content


def test_plan_query_omits_catalog_block_when_empty(monkeypatch) -> None:
    """空清单时不注入目录段落（测试环境/空库兼容）。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "test-key")
    captured: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            payload = {
                "aspects": [
                    {
                        "aspect_id": "intro",
                        "question": "介绍一下自己",
                        "evidence_need": "自我介绍",
                        "search_queries": [{"query": "自我介绍", "query_type": "semantic_question", "rationale": "贴近用户意图"}],
                        "keywords": ["自我介绍"],
                    }
                ]
            }
            return json.dumps({"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]}, ensure_ascii=False).encode()

    def fake_urlopen(request, *args, **kwargs):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(query_planner_service.urllib.request, "urlopen", fake_urlopen)

    plan_query("介绍一下自己", catalog="")
    user_content = captured["body"]["messages"][-1]["content"]
    assert "知识库文档清单" not in user_content


def test_query_planner_accepts_legacy_string_search_queries() -> None:
    aspects = query_planner_service._aspects_from_payload(
        "课程成绩怎么表述",
        {
            "aspects": [
                {
                    "aspect_id": "course_grade",
                    "question": "课程成绩怎么表述",
                    "expected_evidence_type": "表述口径",
                    "search_queries": ["课程成绩表述口径", "课程成绩表述口径"],
                    "keywords": ["课程成绩"],
                }
            ]
        },
    )

    assert len(aspects) == 1
    assert aspects[0].evidence_need == "表述口径"
    assert len(aspects[0].search_queries) == 1
    assert aspects[0].search_queries[0].query == "课程成绩表述口径"
    assert aspects[0].search_queries[0].query_type == "legacy"


def test_query_planner_budget_honors_explicit_three_item_summary(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_MAX_ASPECTS", 8)

    budget = query_planner_service.plan_query_budget(
        "请概括总结：技能掌握程度、项目贡献、实习经历三项内容分别是什么？"
    )

    assert len(budget.detected_items) == 3
    assert budget.max_aspects >= 3


def test_query_planner_budget_honors_explicit_four_item_joint_check(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_MAX_ASPECTS", 8)

    budget = query_planner_service.plan_query_budget(
        "联合核验四项内容：证书信息真实性、材料有效性核实、技能掌握情况比较、依据不足边界。"
    )

    assert len(budget.detected_items) == 4
    assert budget.max_aspects >= 4


def test_execution_constraint_is_not_converted_to_retrieval_aspect() -> None:
    aspects = query_planner_service._aspects_from_payload(
        "依据 Word 公式计算 GPA；若执行器不支持平方根，必须明确拒答而非估算。",
        {
            "aspects": [
                {
                    "aspect_id": "formula",
                    "question": "课程成绩规则中 GPA 的 Word 公式是什么？",
                    "search_queries": ["GPA 公式"],
                },
                {
                    "aspect_id": "instruction",
                    "question": "若执行器不支持平方根运算，必须明确拒答而非估算。",
                    "search_queries": ["拒答要求"],
                },
            ]
        },
    )

    assert [aspect.aspect_id for aspect in aspects] == ["formula"]
