"""grounding 确定性硬事实校验器测试：数字/日期/年份/书名号专名命中与缺失。"""

from backend.app.services.grounding_verification_service import (
    verify_hard_facts,
)


def test_all_hard_facts_found_in_evidence_verifies() -> None:
    answer = "秒杀项目支撑 10000 并发，2024 年上线，采用《高并发电商秒杀平台》方案。"
    evidence = [
        "秒杀平台支持 10000 并发，2024 年上线。",
        "《高并发电商秒杀平台》是主要项目。",
    ]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is True
    assert result.missing_numbers == []
    assert result.missing_dates == []
    assert result.missing_names == []


def test_missing_number_marks_hedged() -> None:
    answer = "这个项目跑过 500000 并发。"
    evidence = ["秒杀平台支撑 10000 并发。"]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is False
    assert "500000" in result.missing_numbers
    assert result.checked_numbers == 1


def test_missing_year_marks_hedged() -> None:
    answer = "项目于 2023 年启动。"
    evidence = ["项目 2024 年启动。"]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is False
    assert "2023" in result.missing_dates


def test_missing_quoted_project_name_marks_hedged() -> None:
    answer = "我参与了《外卖平台》项目。"
    evidence = ["介绍的是秒杀系统。"]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is False
    assert "外卖平台" in result.missing_names
    assert result.checked_names == 1


def test_empty_answer_verifies_vacuously() -> None:
    assert verify_hard_facts("", ["任何证据"]).verified is True
    assert verify_hard_facts(None, []).verified is True


def test_hedged_answer_with_prefix_is_still_checked() -> None:
    """已带推测前缀的回答仍需过校验（前缀本身不是豁免理由）。"""
    answer = "根据现有知识库推测，成绩是 98 分。"
    evidence = ["成绩 85 分。"]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is False
    assert "98" in result.missing_numbers


def test_full_width_digits_normalized() -> None:
    answer = "并发量 １００００。"
    evidence = ["支撑 10000 并发。"]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is True


def test_contextual_year_is_not_flagged_when_present() -> None:
    """答案中的 4 位年份只要在证据中出现即通过（如"2026 届毕业生"）。"""
    answer = "我是 2026 届毕业生。"
    evidence = ["2026 届毕业生"]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is True


# ---------------------------------------------------------------------------
# 二轮增强：实体校验 / 列表编号误杀 / normalization
# ---------------------------------------------------------------------------

def test_list_ordinals_are_not_treated_as_numbers() -> None:
    """"1. EchoGuide 2. ReguMate"中的编号 1、2 不是需证据支持的数字事实。"""
    answer = "我的项目包括：1. EchoGuide 2. ReguMate 3. 外卖平台。"
    evidence = ["EchoGuide、ReguMate、外卖平台都是我的项目。"]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is True
    assert result.missing_numbers == []


def test_known_entity_missing_from_evidence_marks_hedged() -> None:
    """答案出现知识库已知学校但证据中没有 → 实体未核实。"""
    answer = "我本科毕业于西安电子科技大学。"
    evidence = ["我的教育背景。"]
    known_entities = ["河南大学", "西安电子科技大学", "网络与信息安全"]

    result = verify_hard_facts(answer, evidence, known_entities=known_entities)

    assert result.verified is False
    assert "西安电子科技大学" in result.missing_entities
    assert result.checked_entities == 1


def test_known_entity_present_in_evidence_verifies() -> None:
    """答案出现的已知实体在证据中 → 通过。"""
    answer = "我本科毕业于河南大学。"
    evidence = ["本科毕业于河南大学，网络与信息安全专业。"]
    known_entities = ["河南大学", "西安电子科技大学"]

    result = verify_hard_facts(answer, evidence, known_entities=known_entities)

    assert result.verified is True
    assert result.missing_entities == []


def test_project_name_missing_from_evidence_marks_hedged() -> None:
    answer = "我参与了《高并发电商秒杀平台》开发。"
    evidence = ["外卖平台的介绍。"]
    known_entities = ["高并发电商秒杀平台", "外卖平台"]

    result = verify_hard_facts(answer, evidence, known_entities=known_entities)

    assert result.verified is False
    assert "高并发电商秒杀平台" in result.missing_entities


def test_fullwidth_percent_normalized() -> None:
    """全角百分号 ％ 与半角 % 归一化后匹配。"""
    answer = "提升 30％ 的性能。"
    evidence = ["性能提升 30%"]

    result = verify_hard_facts(answer, evidence)

    assert result.verified is True


def test_entity_pool_empty_is_safe() -> None:
    answer = "我毕业于河南大学。"
    evidence = ["随便一段证据文本。"]

    result = verify_hard_facts(answer, evidence, known_entities=[])

    assert result.checked_entities == 0
    # 无已知实体时不做实体校验（数字/专名仍按各自规则）
    assert "河南大学" not in result.missing_entities
