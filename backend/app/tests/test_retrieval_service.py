from dataclasses import replace

import pytest

from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.retrieval_service import (
    get_last_retrieval_diagnostics,
    limit_rerank_candidates,
    retrieval_queries,
    retrieve_citations,
)
from backend.app.services.vector_store_service import VectorSearchResult


@pytest.fixture(autouse=True)
def keep_synthetic_candidates_active(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.filter_active_candidates",
        lambda candidates: candidates,
    )


def candidate() -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0001",
        document_id="DOC-TEST-0001",
        filename="rules.txt",
        section_title="综合成绩",
        page_number=1,
        text="综合成绩应等于分项成绩合计。",
        embedding_text="章节：综合成绩\n\n综合成绩应等于分项成绩合计。",
        token_count=18,
        score=0.7,
        chunk_type="paragraph",
    )


def fake_embed_texts(monkeypatch):
    calls = []

    def _fake(texts: list[str]):
        calls.append(texts)
        return [f"query-vector-{index}" for index, _text in enumerate(texts)]

    monkeypatch.setattr("backend.app.services.retrieval_service.embed_texts", _fake)
    return calls


def test_limit_rerank_candidates_can_preserve_rrf_order(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.services.retrieval_service.RERANK_CANDIDATE_LIMIT", 2)
    items = [
        replace(candidate(), chunk_id="rrf-first", score=0.2),
        replace(candidate(), chunk_id="rrf-second", score=0.3),
        replace(candidate(), chunk_id="raw-high", score=0.99),
    ]

    selected = limit_rerank_candidates(items, preserve_order=True)

    assert [item.chunk_id for item in selected] == ["rrf-first", "rrf-second"]


def test_limit_rerank_candidates_accepts_effective_limit(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.services.retrieval_service.RERANK_CANDIDATE_LIMIT", 24)
    items = [
        replace(candidate(), chunk_id="rrf-first", score=0.2),
        replace(candidate(), chunk_id="rrf-second", score=0.3),
        replace(candidate(), chunk_id="raw-high", score=0.99),
    ]

    selected = limit_rerank_candidates(items, preserve_order=True, limit=2)

    assert [item.chunk_id for item in selected] == ["rrf-first", "rrf-second"]


def test_retrieve_citations_runs_hybrid_search_and_rerank(monkeypatch) -> None:
    calls = []
    item = candidate()

    embed_calls = fake_embed_texts(monkeypatch)

    def fake_hybrid_search(query_embedding, *, limit: int):
        calls.append((query_embedding, limit))
        return [item]

    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=candidates[0], rerank_score=0.91)],
    )

    matches = retrieve_citations("综合成绩怎么计算")

    assert embed_calls == [["综合成绩怎么计算"]]
    assert calls == [("query-vector-0", 50)]
    assert matches[0].citation.chunk_id == "DOC-TEST-0001-CHUNK-0001"
    assert matches[0].rerank_score == 0.91
    assert matches[0].evidence_role == "direct_evidence"
    diagnostics = get_last_retrieval_diagnostics()
    assert diagnostics.query_count == 1
    assert diagnostics.candidate_count == 1
    assert diagnostics.reranked_count == 1


def test_retrieve_citations_accepts_low_rerank_scores(monkeypatch) -> None:
    item = candidate()

    fake_embed_texts(monkeypatch)
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [item])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=candidates[0], rerank_score=-0.2)],
    )

    matches = retrieve_citations("综合成绩怎么计算")

    assert len(matches) == 1
    assert matches[0].citation.chunk_id == "DOC-TEST-0001-CHUNK-0001"
    assert matches[0].rerank_score == -0.2


def test_retrieve_citations_ranks_by_rerank_score_after_lenient_gate(monkeypatch) -> None:
    paragraph = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0003",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="课程成绩统计口径",
        page_number=None,
        text=(
            "课程成绩统计应同时满足课程范围、考核方式和计分口径三类条件。"
            "同一学生在同一学期存在多门课程时，应按课程维度汇总判断成绩口径。"
            "若学生部分课程符合加分要求、部分课程不符合加分要求，应仅统计符合加分要求的课程成绩。"
        ),
        embedding_text="章节：课程成绩统计口径\n\n同一学生在同一学期存在多门课程时，应按课程维度汇总判断成绩口径。",
        token_count=120,
        score=0.7,
        chunk_type="paragraph",
    )
    table = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0005",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="不应纳入课程成绩统计的情形",
        page_number=None,
        text="表格：\n| 情形 | 处理口径 |\n| --- | --- |\n| 缺勤超过三分之一 | 不纳入课程成绩 |",
        embedding_text="章节：不应纳入课程成绩统计的情形\n内容类型：表格\n\n表格：\n| 情形 | 处理口径 |",
        token_count=80,
        score=0.9,
        chunk_type="table",
    )

    fake_embed_texts(monkeypatch)
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [table, paragraph])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [
            RerankedChunk(candidate=table, rerank_score=0.96),
            RerankedChunk(candidate=paragraph, rerank_score=0.9),
        ],
    )

    matches = retrieve_citations("同一个学生有多门课程时，课程成绩应该怎么统计？")

    # The structural table_evidence machinery is gone: ranking is driven by
    # rerank score, and the neighbour table is no longer downgraded.
    assert [match.citation.chunk_id for match in matches] == [
        "DOC-TEST-0001-CHUNK-0005",
        "DOC-TEST-0001-CHUNK-0003",
    ]
    # 通用分词器下：paragraph 文本与问题 bigram 覆盖高 → direct；table 覆盖低 → related
    assert matches[1].evidence_role == "direct_evidence"
    assert matches[0].evidence_role == "related_context"
    assert all(match.citation.evidence_role != "table_context" for match in matches)


def test_retrieve_citations_accepts_high_rerank_low_coverage(monkeypatch) -> None:
    item = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0010",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="奖学金评定标识",
        page_number=None,
        text="奖学金评定应基于综合表现、获奖记录和支撑材料确定。",
        embedding_text="章节：奖学金评定标识\n\n奖学金评定应基于综合表现、获奖记录和支撑材料确定。",
        token_count=50,
        score=0.9,
        chunk_type="paragraph",
    )

    fake_embed_texts(monkeypatch)
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [item])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=item, rerank_score=1.0)],
    )

    matches = retrieve_citations("同一个学生有多门课程时，课程成绩应该怎么统计？")

    # The lexical coverage gate is gone: a high-rerank candidate passes even
    # with zero term coverage, and is kept as related_context.
    assert len(matches) == 1
    assert matches[0].citation.chunk_id == "DOC-TEST-0001-CHUNK-0010"
    assert matches[0].rerank_score == 1.0
    assert matches[0].evidence_role == "related_context"


def test_retrieve_citations_keeps_table_chunk_with_related_context_role(monkeypatch) -> None:
    table = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0020",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="课程成绩与实习评价区别",
        page_number=None,
        text=(
            "表格行证据：比较项目为“课程成绩与实习评价”时，区别为“课程成绩强调考核得分，"
            "实习评价强调实践表现与能力，两者不能直接等同”。"
        ),
        embedding_text=(
            "章节：课程成绩与实习评价区别\n内容类型：表格\n内容形态：表格行级证据\n\n"
            "表格行证据：比较项目为“课程成绩与实习评价”时，区别为“课程成绩强调考核得分，"
            "实习评价强调实践表现与能力，两者不能直接等同”。"
        ),
        token_count=80,
        score=0.88,
        chunk_type="table",
    )

    fake_embed_texts(monkeypatch)
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [table])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=table, rerank_score=0.94)],
    )

    matches = retrieve_citations("课程成绩与实习评价的区别")

    # The table_evidence role no longer exists; the table chunk survives the
    # lenient gate and is labelled by lexical coverage instead.
    assert len(matches) == 1
    assert matches[0].citation.chunk_id == "DOC-TEST-0001-CHUNK-0020"
    assert matches[0].evidence_role == "direct_evidence"


def test_retrieval_queries_decompose_score_difference_question() -> None:
    question = "综合成绩差异应该优先排查哪些问题？如果差异来自分项加总，需要保留什么依据？"
    queries = retrieval_queries(question)

    assert queries == [
        question,
        "综合成绩差异应该优先排查哪些问题",
        "差异来自分项加总",
        "需要保留什么依据",
    ]
    assert not any("综合成绩 差异" in query for query in queries)


def test_retrieve_citations_batches_expanded_query_embeddings(monkeypatch) -> None:
    item = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0030",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="综合成绩差异处理",
        page_number=None,
        text=(
            "综合成绩差异应优先检查分项加总、四舍五入、课程映射和重复汇总。"
            "若差异来自分项加总，应保留统计日期、计算规则和成绩来源。"
        ),
        embedding_text=(
            "章节：综合成绩差异处理\n\n综合成绩差异应优先检查分项加总、四舍五入、课程映射和重复汇总。"
            "若差异来自分项加总，应保留统计日期、计算规则和成绩来源。"
        ),
        token_count=100,
        score=0.9,
        chunk_type="paragraph",
    )
    embed_calls = fake_embed_texts(monkeypatch)
    hybrid_calls = []

    def fake_hybrid_search(query_embedding, *, limit: int):
        hybrid_calls.append((query_embedding, limit))
        return [item]

    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=candidates[0], rerank_score=0.93)],
    )

    matches = retrieve_citations("综合成绩差异应该优先排查哪些问题？如果差异来自分项加总，需要保留什么依据？")

    assert matches
    assert len(embed_calls) == 1
    assert len(embed_calls[0]) > 1
    assert len(hybrid_calls) == len(embed_calls[0])
    diagnostics = get_last_retrieval_diagnostics()
    assert diagnostics.query_count == len(embed_calls[0])
    assert diagnostics.raw_candidate_count == len(embed_calls[0])
    assert diagnostics.candidate_count == 1


def test_retrieve_citations_limits_candidates_before_rerank(monkeypatch) -> None:
    items = [
        VectorSearchResult(
            chunk_id=f"DOC-TEST-0001-CHUNK-{index:04d}",
            document_id="DOC-TEST-0001",
            filename="quality_rules.md",
            section_title="综合成绩差异处理",
            page_number=None,
            text=(
                "综合成绩差异应优先检查分项加总、四舍五入、课程映射和重复汇总问题。"
                "若差异来自分项加总，应保留统计日期、计算规则和成绩来源。"
            ),
            embedding_text=(
                "章节：综合成绩差异处理\n\n"
                "综合成绩差异应优先检查分项加总、四舍五入、课程映射和重复汇总问题。"
                "若差异来自分项加总，应保留统计日期、计算规则和成绩来源。"
            ),
            token_count=100,
            score=1.0 - index * 0.001,
            chunk_type="paragraph",
        )
        for index in range(40)
    ]
    rerank_inputs = []

    fake_embed_texts(monkeypatch)
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: items)

    def fake_rerank_candidates(question, candidates, limit):
        rerank_inputs.append(candidates)
        return [
            RerankedChunk(candidate=candidate, rerank_score=0.95 - index * 0.001)
            for index, candidate in enumerate(candidates[:limit])
        ]

    monkeypatch.setattr("backend.app.services.retrieval_service.rerank_candidates", fake_rerank_candidates)

    matches = retrieve_citations("综合成绩差异应该优先排查哪些问题？如果差异来自分项加总，需要保留什么依据？")

    assert matches
    assert len(rerank_inputs) == 1
    assert len(rerank_inputs[0]) == 24
    assert [item.chunk_id for item in rerank_inputs[0]] == [item.chunk_id for item in items[:24]]
    diagnostics = get_last_retrieval_diagnostics()
    assert diagnostics.query_count > 1
    assert diagnostics.raw_candidate_count == 40 * diagnostics.query_count
    assert diagnostics.candidate_count == 40
    assert diagnostics.rerank_input_count == 24
    assert diagnostics.reranked_count == 20
