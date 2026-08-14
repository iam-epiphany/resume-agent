import json

from backend.app.schemas.qa import RetrievalResult
from backend.app.services import answer_generation_service
from backend.app.services.answer_generation_service import (
    FALLBACK_NO_CONTEXT,
    _greeting_copy,
    _off_topic_copy,
    GeneratedAnswer,
    generate_answer,
)
from backend.app.services.intent_router_service import (
    INTENT_GREETING,
    INTENT_OFF_TOPIC,
    INTENT_RESUME_QA,
    INTENT_RESUME_QA,
)


def _chunk(
    *,
    metadata=None,
    text="证书有效期至2025年12月31日。",
    citation_label="[1]",
    chunk_id="C1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, rank=1, score=0.9, source_doc="技能专长.md",
        section_title="测试", section_path=["测试"], text=text,
        citation_label=citation_label, metadata=metadata or {},
    )


def _mock_llm_answer(monkeypatch, *, answer, evidence_sufficiency="sufficient", degraded=False) -> None:
    """Mock the internal LLM call to return a fixed GeneratedAnswer."""

    def fake_call(*args, **kwargs):
        return GeneratedAnswer(
            answer=answer,
            evidence_sufficiency=evidence_sufficiency,
            generation_status="completed",
            degraded=degraded,
        )

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_call)


# ---------------------------------------------------------------------------
# 意图转移（零 LLM）
# ---------------------------------------------------------------------------

def test_greeting_intent_redirects_without_llm(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("greeting intent must not invoke the LLM")

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fail_if_called)

    result = generate_answer("你好", [], intent=INTENT_GREETING)

    assert result.answer == _greeting_copy("")
    assert result.answer_mode == "redirected"
    assert result.evidence_sufficiency is None
    assert result.generation_status == "skipped"


def test_off_topic_intent_redirects_without_llm(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("off_topic intent must not invoke the LLM")

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fail_if_called)

    result = generate_answer("今天天气怎么样", [], intent=INTENT_OFF_TOPIC)

    assert result.answer == _off_topic_copy("")
    assert result.answer_mode == "redirected"
    assert result.generation_status == "skipped"


# ---------------------------------------------------------------------------
# LLM 自评分级
# ---------------------------------------------------------------------------

def test_sufficient_evidence_returns_answered_without_prefix(monkeypatch) -> None:
    _mock_llm_answer(monkeypatch, answer="证书有效期至2025年12月31日。", evidence_sufficiency="sufficient")

    result = generate_answer("证书有效期是什么？", [_chunk()], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "answered"
    assert result.evidence_sufficiency == "sufficient"
    assert result.answer == "证书有效期至2025年12月31日。"
    assert not result.answer.startswith("根据现有知识库推测")


def test_partial_sufficiency_maps_to_hedged_without_prefix(monkeypatch) -> None:
    _mock_llm_answer(monkeypatch, answer="可能参与了秒杀项目的核心开发。", evidence_sufficiency="partial")

    result = generate_answer("秒杀项目你做了什么？", [_chunk()], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.evidence_sufficiency == "partial"
    # 分级仅内部记录：回答正文原样保留，不再追加推测前缀
    assert result.answer == "可能参与了秒杀项目的核心开发。"


def test_insufficient_evidence_is_hedged_without_prefix(monkeypatch) -> None:
    _mock_llm_answer(monkeypatch, answer="简历中未明确提及薪资预期。", evidence_sufficiency="insufficient")

    result = generate_answer("期望薪资是多少？", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.answer == "简历中未明确提及薪资预期。"


def test_llm_provided_hedge_prefix_is_preserved_as_is(monkeypatch) -> None:
    _mock_llm_answer(
        monkeypatch,
        answer="根据现有知识库推测，简历中未明确提及薪资预期。",
        evidence_sufficiency="partial",
    )

    result = generate_answer("期望薪资是多少？", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.answer == "根据现有知识库推测，简历中未明确提及薪资预期。"


def test_missing_sufficiency_defaults_to_hedged(monkeypatch) -> None:
    _mock_llm_answer(monkeypatch, answer="推测性回答。", evidence_sufficiency=None)

    result = generate_answer("这个问题没有明确依据", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.evidence_sufficiency == "partial"
    assert result.answer == "推测性回答。"


def test_invalid_sufficiency_value_defaults_to_hedged(monkeypatch) -> None:
    _mock_llm_answer(monkeypatch, answer="推测性回答。", evidence_sufficiency="unknown")

    result = generate_answer("这个问题没有明确依据", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.evidence_sufficiency == "partial"


def test_sufficient_answer_with_matching_evidence_returns_answered(monkeypatch) -> None:
    """LLM 自评 sufficient 且硬事实在证据中 → answered（grounding 校验通过）。"""
    _mock_llm_answer(monkeypatch, answer="该材料规定本人必须在2099年完成技能评估。", evidence_sufficiency="sufficient")

    result = generate_answer(
        "请概述《技能专长.md》的主要内容。",
        [_chunk(text="技能专长包括Python开发与Java开发等技能。该材料规定本人必须在2099年完成技能评估。")],
        intent=INTENT_RESUME_QA,
    )

    assert result.answer_mode == "answered"
    assert result.generation_status == "completed"
    assert "2099年" in result.answer


def test_sufficient_answer_with_unverified_hard_fact_downgrades_to_hedged(monkeypatch) -> None:
    """LLM 自评 sufficient 但答案中的硬事实（2099）不在证据中 → grounding 降级 hedged。"""
    _mock_llm_answer(monkeypatch, answer="该材料规定本人必须在2099年完成技能评估。", evidence_sufficiency="sufficient")

    result = generate_answer(
        "请概述《技能专长.md》的主要内容。",
        [_chunk(text="技能专长包括Python开发与Java开发等技能。")],
        intent=INTENT_RESUME_QA,
    )

    assert result.answer_mode == "hedged"
    assert result.evidence_sufficiency == "partial"
    # grounding 降级只改变分级，不改写回答正文
    assert result.answer == "该材料规定本人必须在2099年完成技能评估。"
    assert "硬事实未在检索证据中核实" in (result.hedge_note or "")
    grounding = result.extra.get("grounding_verification") or {}
    assert grounding.get("verified") is False
    assert "2099" in grounding.get("missing_dates", [])


def test_grounding_verification_disabled_keeps_sufficient_answered(monkeypatch) -> None:
    """关闭 GROUNDING_VERIFY_ENABLED 时，sufficient 自评直接 answered（兼容旧行为）。"""
    monkeypatch.setattr(answer_generation_service, "GROUNDING_VERIFY_ENABLED", False)
    _mock_llm_answer(monkeypatch, answer="该材料规定本人必须在2099年完成技能评估。", evidence_sufficiency="sufficient")

    result = generate_answer(
        "请概述《技能专长.md》的主要内容。",
        [_chunk(text="技能专长包括Python开发与Java开发等技能。")],
        intent=INTENT_RESUME_QA,
    )

    assert result.answer_mode == "answered"
    assert result.evidence_sufficiency == "sufficient"
    assert "2099年" in result.answer


# ---------------------------------------------------------------------------
# 兜底路径（无 LLM / LLM 异常）
# ---------------------------------------------------------------------------

def test_api_unavailable_uses_extractive_fallback(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    result = generate_answer("证书有效期是什么？", [_chunk()], intent=INTENT_RESUME_QA)

    assert result.degraded is True
    assert result.answer_mode == "hedged"
    assert result.evidence_sufficiency == "partial"
    assert result.generation_status == "degraded"
    assert result.answer == "证书有效期至2025年12月31日。"


def test_empty_context_with_llm_disabled_returns_failed_fallback(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer("不存在的问题", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "failed"
    assert result.answer == FALLBACK_NO_CONTEXT
    assert result.generation_status == "skipped"
    assert result.degraded is False


def test_empty_context_with_llm_exception_returns_failed_fallback(monkeypatch) -> None:
    def failing_call(*args, **kwargs):
        raise OSError("answer generation unavailable")

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", failing_call)

    result = generate_answer("不存在的问题", [], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "failed"
    assert result.answer == FALLBACK_NO_CONTEXT


def test_llm_exception_with_context_uses_extractive_fallback(monkeypatch) -> None:
    def failing_call(*args, **kwargs):
        raise OSError("answer generation unavailable")

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", failing_call)

    result = generate_answer(
        "证书有效期是什么？",
        [_chunk(text="证书有效期至2025年12月31日。")],
        intent=INTENT_RESUME_QA,
    )

    assert result.answer_mode == "hedged"
    assert result.degraded is True
    assert result.generation_status == "degraded"
    assert "2025年12月31日" in result.answer


def test_llm_returns_empty_answer_falls_back_to_extract(monkeypatch) -> None:
    _mock_llm_answer(monkeypatch, answer="")

    result = generate_answer("证书有效期是什么？", [_chunk()], intent=INTENT_RESUME_QA)

    assert result.answer_mode == "hedged"
    assert result.degraded is True
    assert "2025年12月31日" in result.answer


def test_extractive_fallback_uses_top_chunk_excerpt() -> None:
    result = answer_generation_service._extractive_fallback(
        [
            _chunk(
                metadata={
                    "aspect_id": "award_query_requirement_2",
                    "aspect_question": "本人不得向特定群体成员以外的个人提供获奖名额",
                    "prompt_matched_aspects": ["award_query_requirement_2"],
                },
                text=(
                    "本人参加竞赛活动应遵守奖项说明要求。"
                    "第十九条 本人应为获奖者提供电话、邮箱、微信或其他方式的获奖信息查询服务，"
                    "并在奖项说明后告知获奖者查询途径。"
                    "获奖信息查询服务应至少保留至赛事结束后三个月。"
                ),
                citation_label="[1]",
                chunk_id="R1",
            )
        ],
    )

    assert result.answer_mode == "hedged"
    assert result.degraded is True
    assert result.hedge_note == "LLM 生成失败，直接摘录知识库原文"
    assert "获奖信息查询服务应至少保留至赛事结束后三个月" in result.answer


def test_extractive_fallback_without_chunks_returns_none() -> None:
    assert answer_generation_service._extractive_fallback([]) is None


# ---------------------------------------------------------------------------
# _call_llm 请求载荷与解析
# ---------------------------------------------------------------------------

def test_call_llm_reuses_context_package_prompt_in_request_payload(monkeypatch) -> None:
    captured = {}
    response_content = json.dumps(
        {
            "answer": "证书有效期至2025年12月31日。",
            "evidence_sufficiency": "sufficient",
            "reason": "知识片段直接包含有效期信息",
        },
        ensure_ascii=False,
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": response_content}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(answer_generation_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")

    llm_prompt = (
        "你是 ResumeMind，一个基于个人简历、证书、荣誉与项目文档的可信问答助手。\n\n"
        "【用户问题】\n证书有效期是什么？\n\n"
        "【检索到的知识片段】\n[1] 来源：技能专长.md / 测试\n证书有效期至2025年12月31日。"
    )

    result = answer_generation_service._call_llm(
        "证书有效期是什么？",
        [_chunk()],
        llm_prompt=llm_prompt,
    )

    user_message = captured["payload"]["messages"][1]["content"]
    system_message = captured["payload"]["messages"][0]["content"]
    assert llm_prompt in user_message
    assert "evidence_sufficiency" in system_message
    assert result.answer == "证书有效期至2025年12月31日。"
    assert result.evidence_sufficiency == "sufficient"
    assert result.answer_mode == "answered"


def test_parse_generated_content_extracts_answer_and_sufficiency() -> None:
    result = answer_generation_service._parse_generated_content(
        json.dumps(
            {"answer": "简历中未明确提及薪资预期。", "evidence_sufficiency": "insufficient", "reason": "无直接依据"},
            ensure_ascii=False,
        )
    )

    assert result.answer == "简历中未明确提及薪资预期。"
    assert result.evidence_sufficiency == "insufficient"
    assert result.hedge_note == "无直接依据"
    assert result.generation_status == "completed"


def test_parse_generated_content_rejects_invalid_sufficiency() -> None:
    result = answer_generation_service._parse_generated_content(
        json.dumps({"answer": "回答。", "evidence_sufficiency": "unknown"})
    )

    assert result.evidence_sufficiency is None
    assert result.answer == "回答。"


def test_call_llm_recovers_partial_answer_after_stream_parse_failure(monkeypatch) -> None:
    calls = []
    partial_content = (
        '{"answer": "Requirement one must be retained for three years. [1]",'
        '"evidence_sufficiency": "sufficient"'
    )

    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(["data: [DONE]\n"])

    class FakeBodyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": partial_content}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(payload)
        if payload.get("stream") is True:
            return FakeStreamResponse()
        return FakeBodyResponse()

    monkeypatch.setattr(answer_generation_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_STREAM", True)

    result = answer_generation_service._call_llm(
        "What is the retention requirement?",
        [_chunk(text="Requirement one must be retained for three years.")],
    )

    assert [payload.get("stream") for payload in calls] == [True, None, None]
    assert result.answer == "Requirement one must be retained for three years. [1]"
    assert result.degraded is True
    assert result.evidence_sufficiency == "partial"
    assert result.hedge_note == "流式内容截断，按推测处理"


# ---------------------------------------------------------------------------
# 摘录辅助函数
# ---------------------------------------------------------------------------

def test_relevant_extractive_excerpt_keeps_exception_when_question_requests_exceptions() -> None:
    text = (
        "二、说明内容。"
        "（一）本人以表格形式进行技能掌握情况说明，包括固定表格和可变表格。"
        "本人应按说明填写固定表格。"
        "本人应按要求评估说明内容，并确保说明信息对信息使用者具有参考价值。"
        "本人应根据表格要求，分别按照季度、半年和年度的频率说明信息。"
        "本人应按照简历材料范围说明相关信息，表格中另有规定的除外。"
        "说明报告应由指导教师审阅。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "技能掌握说明频率和表格例外",
    )

    assert "季度、半年和年度" in excerpt
    assert "表格中另有规定的除外" in excerpt


def test_relevant_extractive_excerpt_keeps_discount_curve_definition() -> None:
    text = (
        "根据《项目介绍_ReguMate.md》规定，"
        "计算项目综合成绩所采用的评分规则由基础得分加奖励得分形成，具体计算方法如下："
        "一、基础得分为项目分数，由以下三段组成。"
        "其中，t表示维度；加权规则采用加权平均方法计算得到；基础权重暂定为4.5%。"
        "二、加权规则采用加权平均方法计算得到。"
        "t为维度；r为在t维度下项目评分的数值；Rt为在t维度下加权得分的数值。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "项目综合成绩计算所采用的评分规则",
    )

    assert "评分规则由基础得分加奖励得分形成" in excerpt


def test_relevant_extractive_excerpt_keeps_leading_normative_sentence() -> None:
    text = (
        "本人应当在其个人主页、简历等公开渠道就办理证书核实相关事项进行说明。"
        "本人说明的内容应当符合本说明具体要求。"
        "用人单位在此基础上根据本人说明信息进行核实。"
        "说明信息包括：1.各种核实方式下办理核验工作的机构及其联系方式。"
        "在实现集约化或数字化的情况下，本人应当就核验范围以及所采用的核验用章的适用范围进行说明。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "根据《证书说明.md》，请说明与“证书核实”相关的明确规定。",
        max_chars=180,
    )

    assert "本人应当在其个人主页、简历等公开渠道就办理证书核实相关事项进行说明" in excerpt


def test_extract_answer_from_partial_extracts_escaped_json_string() -> None:
    # 恢复逻辑保留原始转义序列（不反解码），避免截断导致转义损坏
    content = '{"answer": "含引号\\"与转义\\\\的内容", "extra": 1}'
    extracted = answer_generation_service._extract_answer_from_partial(content)

    assert extracted == '含引号\\"与转义\\\\的内容'


def test_extract_json_handles_prefix_and_suffix_noise() -> None:
    content = 'prefix {"answer": "x", "evidence_sufficiency": "sufficient"} suffix'

    assert answer_generation_service._extract_json(content) == (
        '{"answer": "x", "evidence_sufficiency": "sufficient"}'
    )
