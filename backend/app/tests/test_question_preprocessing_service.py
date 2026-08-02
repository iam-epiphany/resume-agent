from backend.app.services.question_preprocessing_service import preprocess_qa_request


def test_preprocess_keeps_explicit_options() -> None:
    result = preprocess_qa_request(
        "关于《材料》，以下哪项正确？",
        ["选项A", "选项B"],
    )

    assert result.question == "关于《材料》，以下哪项正确？"
    assert result.options == ["选项A", "选项B"]
    assert result.extracted_options is False


def test_preprocess_extracts_three_labeled_inline_options() -> None:
    result = preprocess_qa_request(
        "关于《材料》，以下哪项正确？ A 第一项事实 B 第二项事实 C 第三项事实"
    )

    assert result.question == "关于《材料》，以下哪项正确？"
    assert result.options == ["第一项事实", "第二项事实", "第三项事实"]
    assert result.option_labels == ["A", "B", "C"]
    assert result.extracted_options is True
    assert result.option_source == "inline_labeled"


def test_preprocess_extracts_circled_options_without_punctuation() -> None:
    result = preprocess_qa_request(
        "检索《技能专长.md》，哪组表述属于材料内容？①熟练掌握 Java 后端开发与 Spring Boot 框架 ②技能掌握情况详见技能专长文档"
    )

    assert result.question == "检索《技能专长.md》，哪组表述属于材料内容？"
    assert result.options == [
        "熟练掌握 Java 后端开发与 Spring Boot 框架",
        "技能掌握情况详见技能专长文档",
    ]
    assert result.option_labels == ["①", "②"]


def test_preprocess_extracts_unlabeled_option_block_split_by_newlines_and_pipes() -> None:
    result = preprocess_qa_request(
        "关于《材料》，以下哪组正确？选项：第一组选项事实\n第二组选项事实｜第三组选项事实"
    )

    assert result.question == "关于《材料》，以下哪组正确？"
    assert result.options == ["第一组选项事实", "第二组选项事实", "第三组选项事实"]
    assert result.option_labels == [None, None, None]
    assert result.option_source == "inline_unlabeled_block"


def test_preprocess_extracts_tabular_option_groups_after_choice_stem() -> None:
    common_fact = "证书说明用于记录职业资格证书的获得时间、颁发机构和证书编号等信息。"
    result = preprocess_qa_request(
        "关于《证书说明.md》，下列哪一组选项中的两项表述均属于该材料内容？\t"
        f"{common_fact}；软件设计师证书由人力资源和社会保障部、工业和信息化部联合颁发。\t"
        f"{common_fact}；蓝桥杯全国软件和信息技术专业人才大赛面向全国高校在校学生举办。\t"
        f"{common_fact}；获得奖学金记录与获奖等级详见个人荣誉文档。\t"
        f"{common_fact}；软考中级软件设计师证书需要通过综合知识与应用技术两科考试。"
    )

    assert result.question == "关于《证书说明.md》，下列哪一组选项中的两项表述均属于该材料内容？"
    assert len(result.options) == 4
    assert result.options[2].endswith("详见个人荣誉文档")
    assert result.extracted_options is True
    assert result.option_source == "inline_tabular_options"


def test_preprocess_does_not_split_single_semicolon_facts_as_options() -> None:
    result = preprocess_qa_request(
        "关于《材料》，以下哪组正确？选项：第一条事实；第二条事实；第三条事实"
    )

    assert result.options == []
