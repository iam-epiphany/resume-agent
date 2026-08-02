from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import re
from time import perf_counter
from typing import Any, Callable

from backend.app.core.config import (
    DIRECT_EVIDENCE_COVERAGE,
    FINAL_CITATION_LIMIT,
    INDEX_VERSION,
    RERANK_CANDIDATE_LIMIT,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
)
from backend.app.core.database import SessionLocal
from backend.app.models.document import Document
from backend.app.schemas.qa import Citation
from backend.app.services.embedding_service import EmbeddingServiceError, embed_queries, embed_texts
from backend.app.services.rerank_service import RerankServiceError, RerankedChunk, rerank_candidates
from backend.app.services.vector_store_service import (
    VectorSearchResult,
    VectorStoreError,
    hybrid_search,
    hybrid_search_batch,
)


class RetrievalServiceUnavailable(RuntimeError):
    pass


ProgressReporter = Callable[[dict[str, Any]], None]
_DEFAULT_EMBED_TEXTS = embed_texts
_DEFAULT_HYBRID_SEARCH = hybrid_search


@dataclass
class RetrievalMatch:
    citation: Citation
    score: float
    rerank_score: float
    coverage_score: float = 0.0
    evidence_role: str = "related_context"
    evidence_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalDiagnostics:
    query_count: int = 0
    candidate_count: int = 0
    raw_candidate_count: int = 0
    rerank_input_count: int = 0
    rerank_call_count: int = 0
    reranked_count: int = 0
    filtered_count: int = 0
    reliable_count: int = 0
    selected_count: int = 0
    query_variants: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    score_range: dict[str, float | None] = field(default_factory=dict)
    metadata_filter: dict[str, Any] = field(default_factory=dict)

    def to_summary_fields(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "raw_candidate_count": self.raw_candidate_count,
            "candidate_count": self.candidate_count,
            "rerank_input_count": self.rerank_input_count,
            "rerank_call_count": self.rerank_call_count,
            "rerank_candidate_limit": RERANK_CANDIDATE_LIMIT,
            "reranked_count": self.reranked_count,
            "filtered_count": self.filtered_count,
            "timings_ms": self.timings_ms,
            "score_range": self.score_range,
            "query_variants": self.query_variants,
            "metadata_filter": self.metadata_filter,
        }


_LAST_RETRIEVAL_DIAGNOSTICS: ContextVar[RetrievalDiagnostics] = ContextVar(
    "resumemind_last_retrieval_diagnostics",
    default=RetrievalDiagnostics(),
)


def get_last_retrieval_diagnostics() -> RetrievalDiagnostics:
    return _LAST_RETRIEVAL_DIAGNOSTICS.get()


def reset_retrieval_diagnostics() -> None:
    _LAST_RETRIEVAL_DIAGNOSTICS.set(RetrievalDiagnostics())


def filter_active_candidates(
    candidates: list[VectorSearchResult],
    *,
    include_inactive: bool = False,
) -> list[VectorSearchResult]:
    """Keep only candidates backed by the current active SQLite index."""

    document_ids = {candidate.document_id for candidate in candidates if candidate.document_id}
    if not document_ids:
        return []
    with SessionLocal() as db:
        query = db.query(Document).filter(
            Document.document_id.in_(document_ids),
            Document.status == "indexed",
            Document.index_version == INDEX_VERSION,
        )
        rows = query.all()
    active_documents = {document.document_id: document for document in rows}
    active_ids = set(active_documents)
    active_candidates = [candidate for candidate in candidates if candidate.document_id in active_ids]
    for candidate in active_candidates:
        if candidate.metadata is None:
            candidate.metadata = {}
        document = active_documents[candidate.document_id]
        authoritative_metadata = {
            "source_title": document.title or document.filename,
            "external_doc_id": document.external_doc_id,
            "issuing_authority": document.issuing_authority,
            "publication_date": document.publication_date,
            "effective_date": document.effective_date,
            "expiration_date": document.expiration_date,
            "document_number": document.document_number,
            "material_topic": document.material_topic,
            "business_domain": document.business_domain,
            "source_url": document.source_url,
            "attachment_url": document.attachment_url,
            "version_status": document.version_status,
            "file_type": document.file_type,
            "source_format": document.file_type,
        }
        extension_metadata = _json_object(document.document_metadata)
        candidate.metadata.update(
            {
                key: value
                for key, value in {**extension_metadata, **authoritative_metadata}.items()
                if value not in (None, "")
            }
        )
        candidate.metadata["active_index_verified"] = True
    return active_candidates


def _json_object(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def filter_candidates_by_metadata(
    candidates: list[VectorSearchResult],
    filters: dict[str, Any] | None,
) -> list[VectorSearchResult]:
    """Apply auditable document filters after hybrid recall.

    Table-specific fields remain handled by SpreadsheetCell. This function is
    deliberately strict only for explicit document metadata constraints.
    """

    filters = filters or {}
    mappings = {
        "source_title": ("title", "source_title", "filename"),
        "filename": ("source_filename", "filename"),
        "external_doc_id": ("external_doc_id",),
        "issuing_authority": ("issuing_authority",),
        "publication_date": ("publication_date",),
        "document_number": ("document_number",),
        "material_topic": ("material_topic",),
        "business_domain": ("business_domain",),
        "article_number": ("article_number", "section_number"),
        "version_status": ("version_status",),
        "file_type": ("source_format", "file_type"),
    }
    active_filters = {
        key: value
        for key, value in filters.items()
        if key in mappings and value not in (None, "", [])
    }
    if not active_filters:
        return candidates

    def matches(candidate: VectorSearchResult) -> bool:
        metadata = candidate.metadata or {}
        for key, expected in active_filters.items():
            actual_values: list[Any] = []
            for field_name in mappings[key]:
                if field_name == "filename":
                    actual_values.append(candidate.filename)
                elif field_name == "section_number":
                    actual_values.append(candidate.section_number)
                else:
                    actual_values.append(metadata.get(field_name))
            expected_norm = _normalize_for_match(str(expected))
            if key in {"publication_date", "version_status", "external_doc_id", "file_type"}:
                accepted = any(_normalize_for_match(str(actual)) == expected_norm for actual in actual_values if actual)
            else:
                accepted = any(expected_norm in _normalize_for_match(str(actual)) for actual in actual_values if actual)
            if not accepted:
                return False
        return True

    return [candidate for candidate in candidates if matches(candidate)]


def retrieve_citations(question: str, progress_reporter: ProgressReporter | None = None) -> list[RetrievalMatch]:
    diagnostics = RetrievalDiagnostics()
    _LAST_RETRIEVAL_DIAGNOSTICS.set(diagnostics)
    total_started_at = perf_counter()
    try:
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "running",
                "title": "正在检索相关依据",
                "detail": "正在从知识库召回候选片段……",
                "summary": {"query": question},
            },
        )
        candidates = filter_active_candidates(_collect_candidates(question, diagnostics))
        diagnostics.candidate_count = len(candidates)
        rerank_candidates_input = _limit_rerank_candidates(candidates)
        diagnostics.rerank_input_count = len(rerank_candidates_input)
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "completed",
                "title": "依据检索完成",
                "detail": f"已召回 {diagnostics.candidate_count} 个候选片段，用时 {_format_elapsed_seconds(diagnostics.timings_ms.get('qdrant'))}",
                "elapsed_ms": diagnostics.timings_ms.get("qdrant"),
                "summary": {
                    "candidate_count": diagnostics.candidate_count,
                    "raw_candidate_count": diagnostics.raw_candidate_count,
                    "rerank_input_count": diagnostics.rerank_input_count,
                    "query_count": diagnostics.query_count,
                },
            },
        )
        if not rerank_candidates_input:
            diagnostics.timings_ms["total"] = _elapsed_ms(total_started_at)
            return []
        rerank_started_at = perf_counter()
        _report_progress(
            progress_reporter,
            {
                "stage": "rerank",
                "status": "running",
                "title": "正在重排候选片段",
                "detail": f"正在重排 {diagnostics.rerank_input_count} 个候选片段……",
                "summary": {
                    "candidate_count": diagnostics.candidate_count,
                    "rerank_input_count": diagnostics.rerank_input_count,
                    "rerank_candidate_limit": RERANK_CANDIDATE_LIMIT,
                },
            },
        )
        reranked = rerank_candidates(question=question, candidates=rerank_candidates_input, limit=RERANK_TOP_K)
        diagnostics.rerank_call_count = 1 if rerank_candidates_input else 0
        diagnostics.timings_ms["rerank"] = _elapsed_ms(rerank_started_at)
        diagnostics.reranked_count = len(reranked)
        diagnostics.score_range = _score_range(candidates, reranked)
        _report_progress(
            progress_reporter,
            {
                "stage": "rerank",
                "status": "completed",
                "title": "候选重排完成",
                "detail": f"已完成 {diagnostics.reranked_count} 个片段重排，用时 {_format_elapsed_seconds(diagnostics.timings_ms.get('rerank'))}",
                "elapsed_ms": diagnostics.timings_ms["rerank"],
                "summary": {
                    "rerank_input_count": diagnostics.rerank_input_count,
                    "reranked_count": diagnostics.reranked_count,
                },
            },
        )
    except (EmbeddingServiceError, VectorStoreError, RerankServiceError) as exc:
        diagnostics.timings_ms["total"] = _elapsed_ms(total_started_at)
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "failed",
                "title": "依据检索失败",
                "detail": str(exc),
                "elapsed_ms": diagnostics.timings_ms["total"],
            },
        )
        raise RetrievalServiceUnavailable(str(exc)) from exc

    matches = matches_from_reranked(question=question, reranked=reranked, diagnostics=diagnostics)
    diagnostics.timings_ms["total"] = _elapsed_ms(total_started_at)
    return matches


def _report_progress(progress_reporter: ProgressReporter | None, event: dict[str, Any]) -> None:
    if progress_reporter is not None:
        progress_reporter(event)


def _collect_candidates(question: str, diagnostics: RetrievalDiagnostics) -> list[VectorSearchResult]:
    queries = retrieval_queries(question)
    return collect_candidates_for_queries(queries, diagnostics)


def collect_candidates_for_queries(
    queries: list[str],
    diagnostics: RetrievalDiagnostics | None = None,
    *,
    metadata_filter: dict[str, Any] | None = None,
) -> list[VectorSearchResult]:
    diagnostics = diagnostics or RetrievalDiagnostics()
    diagnostics.query_variants = queries
    diagnostics.query_count = len(queries)
    if not queries:
        return []

    candidates_by_chunk_id: dict[str, VectorSearchResult] = {}
    embedding_started_at = perf_counter()
    query_embeddings = _embed_search_queries(queries)
    diagnostics.timings_ms["embedding"] = _elapsed_ms(embedding_started_at)

    qdrant_started_at = perf_counter()
    raw_results_by_query = _hybrid_search_many(
        query_embeddings,
        limit=RETRIEVAL_TOP_K,
        metadata_filter=metadata_filter,
    )
    diagnostics.timings_ms["qdrant"] = _elapsed_ms(qdrant_started_at)
    merge_started_at = perf_counter()
    for raw_results in raw_results_by_query:
        for candidate in raw_results:
            diagnostics.raw_candidate_count += 1
            existing = candidates_by_chunk_id.get(candidate.chunk_id)
            if existing is None or candidate.score > existing.score:
                candidates_by_chunk_id[candidate.chunk_id] = candidate
    diagnostics.timings_ms["candidate_merge"] = _elapsed_ms(merge_started_at)
    diagnostics.candidate_count = len(candidates_by_chunk_id)
    return list(candidates_by_chunk_id.values())


def collect_candidates_with_query_hits(
    queries: list[str],
    *,
    query_metadata: list[dict[str, Any]] | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[list[VectorSearchResult], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    diagnostics = diagnostics or RetrievalDiagnostics()
    diagnostics.query_variants = queries
    diagnostics.query_count = len(queries)
    if not queries:
        return [], {}, []

    metadata_items = query_metadata or [{} for _ in queries]
    candidates_by_chunk_id: dict[str, VectorSearchResult] = {}
    query_hits_by_chunk_id: dict[str, list[dict[str, Any]]] = {}
    per_query_diagnostics: list[dict[str, Any]] = []

    embedding_started_at = perf_counter()
    query_embeddings = _embed_search_queries(queries)
    diagnostics.timings_ms["embedding"] = _elapsed_ms(embedding_started_at)

    qdrant_started_at = perf_counter()
    diagnostics.metadata_filter = dict(metadata_filter or {})
    raw_results_by_query = _hybrid_search_many(
        query_embeddings,
        limit=RETRIEVAL_TOP_K,
        metadata_filter=metadata_filter,
    )
    diagnostics.timings_ms["qdrant"] = _elapsed_ms(qdrant_started_at)
    merge_started_at = perf_counter()
    for query, raw_results, metadata in zip(queries, raw_results_by_query, metadata_items, strict=True):
        for candidate in raw_results:
            candidate.score += _query_anchor_boost(query, candidate)
        raw_results.sort(key=lambda candidate: candidate.score, reverse=True)
        seen_for_query: set[str] = set()
        for rank, candidate in enumerate(raw_results, start=1):
            diagnostics.raw_candidate_count += 1
            seen_for_query.add(candidate.chunk_id)
            existing = candidates_by_chunk_id.get(candidate.chunk_id)
            if existing is None or candidate.score > existing.score:
                candidates_by_chunk_id[candidate.chunk_id] = candidate
            query_hits_by_chunk_id.setdefault(candidate.chunk_id, []).append(
                {
                    "query": query,
                    "rank": rank,
                    "vector_score": candidate.score,
                    **metadata,
                }
            )
        per_query_diagnostics.append(
            {
                "search_query": query,
                "query_count": 1,
                "raw_candidate_count": len(raw_results),
                "candidate_count": len(seen_for_query),
                "rerank_input_count": 0,
                "rerank_call_count": 0,
                "reranked_count": 0,
                "filtered_count": 0,
                "match_count": 0,
                "query_variants": [query],
                "timings_ms": {"qdrant_batch_total": diagnostics.timings_ms["qdrant"]},
                "score_range": _score_range(raw_results, []),
                **metadata,
            }
        )

    diagnostics.timings_ms["candidate_merge"] = _elapsed_ms(merge_started_at)
    diagnostics.candidate_count = len(candidates_by_chunk_id)
    return list(candidates_by_chunk_id.values()), query_hits_by_chunk_id, per_query_diagnostics


def _embed_search_queries(queries: list[str]) -> list[Any]:
    # Keep the long-standing test/extension hook while using the query cache in production.
    if embed_texts is not _DEFAULT_EMBED_TEXTS:
        return embed_texts(queries)
    return embed_queries(queries)


def _hybrid_search_many(
    query_embeddings: list[Any],
    *,
    limit: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[list[VectorSearchResult]]:
    # A patched single-search hook is intentionally honored for deterministic tests.
    if hybrid_search is not _DEFAULT_HYBRID_SEARCH:
        return [hybrid_search(embedding, limit=limit) for embedding in query_embeddings]
    return hybrid_search_batch(
        query_embeddings,
        limit=limit,
        metadata_filter=metadata_filter,
    )


def _query_anchor_boost(query: str, candidate: VectorSearchResult) -> float:
    """Reward exact auditable anchors without replacing semantic retrieval."""

    filename = _normalize_for_match(candidate.filename)
    boost = 0.0
    for title in re.findall(r"《([^》]+)》", query):
        normalized = _normalize_for_match(title)
        if normalized and normalized in filename:
            boost = max(boost, 0.30)
    for document_number in re.findall(r"[\u4e00-\u9fffA-Za-z]+〔\d{4}〕\d+号", query):
        if _normalize_for_match(document_number) in _normalize_for_match(
            f"{candidate.filename} {candidate.embedding_text}"
        ):
            boost = max(boost, 0.20)
    return boost


def matches_from_reranked(
    *,
    question: str,
    reranked: list[RerankedChunk],
    diagnostics: RetrievalDiagnostics | None = None,
    limit: int = FINAL_CITATION_LIMIT,
) -> list[RetrievalMatch]:
    diagnostics = diagnostics or RetrievalDiagnostics()
    terms = question_terms(question)
    filter_started_at = perf_counter()
    reliable = [_annotate(item, terms, question) for item in reranked if _is_reliable(item, terms, question)]
    diagnostics.timings_ms["filter"] = _elapsed_ms(filter_started_at)
    diagnostics.reliable_count = len(reliable)
    diagnostics.filtered_count = max(diagnostics.reranked_count - len(reliable), 0)
    reliable.sort(key=_rank_key, reverse=True)
    selected = _select_final_chunks(reliable, question, limit=limit)
    diagnostics.selected_count = len(selected)
    return [
        RetrievalMatch(
            citation=_to_citation(item),
            score=item.candidate.score,
            rerank_score=item.rerank_score,
            coverage_score=item.coverage_score,
            evidence_role=item.evidence_role,
            evidence_text=_candidate_evidence_text(item.candidate),
            metadata=item.candidate.metadata or {},
        )
        for item in selected
    ]


def limit_rerank_candidates(
    candidates: list[VectorSearchResult],
    *,
    preserve_order: bool = False,
    limit: int | None = None,
) -> list[VectorSearchResult]:
    candidate_limit = RERANK_CANDIDATE_LIMIT if limit is None else max(0, int(limit))
    if len(candidates) <= candidate_limit:
        return candidates
    if candidate_limit <= 0:
        return []
    if preserve_order:
        return candidates[:candidate_limit]
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:candidate_limit]


def _limit_rerank_candidates(candidates: list[VectorSearchResult]) -> list[VectorSearchResult]:
    return limit_rerank_candidates(candidates)


def retrieval_queries(question: str) -> list[str]:
    queries = [question]
    split_parts = [part.strip() for part in re.split(r"[？?。；;，,]|如果|若", question) if part.strip()]
    for part in split_parts:
        if part != question:
            queries.append(part)
    return list(dict.fromkeys(queries))


@dataclass
class _AnnotatedChunk:
    candidate: VectorSearchResult
    rerank_score: float
    coverage_score: float
    evidence_role: str


def _is_reliable(item: RerankedChunk, terms: list[str], question: str) -> bool:
    del terms, question
    # Lenient gate: only structural sanity is enforced. Rerank score and
    # lexical coverage are no longer hard filters; rerank order carries the
    # relevance decision.
    if not item.candidate.chunk_id or not item.candidate.text.strip():
        return False
    return True


def _annotate(item: RerankedChunk, terms: list[str], question: str) -> _AnnotatedChunk:
    coverage = evidence_coverage(terms, _candidate_evidence_text(item.candidate))
    return _AnnotatedChunk(
        candidate=item.candidate,
        rerank_score=item.rerank_score,
        coverage_score=coverage,
        evidence_role=_evidence_role(item.candidate, question, coverage),
    )


def _rank_key(item: _AnnotatedChunk) -> tuple[float, float, float]:
    role_weight = {
        "direct_evidence": 3.0,
        "related_context": 1.0,
    }.get(item.evidence_role, 0.0)
    return (role_weight, item.coverage_score, item.rerank_score)


def _select_final_chunks(items: list[_AnnotatedChunk], question: str, limit: int) -> list[_AnnotatedChunk]:
    if _should_enforce_diversity(question):
        return _limit_document_repetition(items, limit=limit)
    return items[:limit]


def _limit_document_repetition(items: list[_AnnotatedChunk], limit: int) -> list[_AnnotatedChunk]:
    selected: list[_AnnotatedChunk] = []
    per_document: dict[str, int] = {}
    for item in items:
        document_id = item.candidate.document_id
        if per_document.get(document_id, 0) >= 2 and len(items) > limit:
            continue
        selected.append(item)
        per_document[document_id] = per_document.get(document_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _should_enforce_diversity(question: str) -> bool:
    normalized = _normalize_for_match(question)
    return any(term in normalized for term in ["综合", "总结", "对比", "比较", "区别", "关系", "不同"])


def _to_citation(item: _AnnotatedChunk) -> Citation:
    candidate = item.candidate
    metadata = candidate.metadata or {}
    return Citation(
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        filename=candidate.filename,
        source_url=_optional_text(metadata.get("source_url")),
        attachment_url=_optional_text(metadata.get("attachment_url")),
        source_title=_optional_text(metadata.get("title") or metadata.get("source_title")),
        issuing_authority=_optional_text(metadata.get("issuing_authority")),
        publication_date=_optional_text(metadata.get("publication_date")),
        document_number=_optional_text(metadata.get("document_number")),
        version_status=_optional_text(metadata.get("version_status")),
        section_title=candidate.section_title,
        section_path=candidate.section_path or ([candidate.section_title] if candidate.section_title else []),
        section_number=candidate.section_number,
        parent_section_number=candidate.parent_section_number,
        previous_chunk_id=candidate.previous_chunk_id,
        next_chunk_id=candidate.next_chunk_id,
        page_number=candidate.page_number,
        excerpt=_safe_excerpt(candidate.text),
        score=candidate.score,
        rerank_score=item.rerank_score,
        chunk_type=candidate.chunk_type,
        evidence_role=item.evidence_role,
        metadata=metadata,
    )


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def question_terms(question: str) -> list[str]:
    normalized = _normalize_for_match(question)
    domain_phrases = [
        # 简历场景领域词：教育背景 / 项目经历 / 技能 / 荣誉证书 / 求职意向
        "项目介绍",
        "项目经历",
        "项目背景",
        "项目职责",
        "项目成果",
        "技术栈",
        "开发语言",
        "参与情况",
        "负责内容",
        "掌握程度",
        "熟悉程度",
        "专业方向",
        "主修课程",
        "核心课程",
        "研究方向",
        "毕业时间",
        "入学时间",
        "就读",
        "学位",
        "绩点",
        "排名",
        "综合成绩",
        "课程成绩",
        "证书编号",
        "证书名称",
        "取得时间",
        "获得条件",
        "荣誉奖项",
        "奖学金",
        "三好学生",
        "获奖记录",
        "评优记录",
        "竞赛",
        "蓝桥杯",
        "软考",
        "四级",
        "六级",
        "实习时长",
        "实习单位",
        "实习经历",
        "社团",
        "学生工作",
        "论文",
        "课题",
        "团队",
        "分工",
        "任职能力",
        "个人特质",
        "自我评价",
        "求职意向",
        "期望薪资",
        "工作地点",
        "到岗时间",
        "职业规划",
    ]
    terms = [phrase for phrase in domain_phrases if phrase in normalized]
    terms.extend(_expanded_domain_terms(normalized))
    terms.extend(token for token in re.findall(r"[A-Za-z0-9_]{2,}", normalized))
    if not terms:
        terms.extend(_fallback_chinese_terms(normalized))
    return list(dict.fromkeys(terms))


def evidence_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    normalized_text = _normalize_for_match(text)
    matched = sum(1 for term in terms if term in normalized_text)
    return matched / len(terms)


def _evidence_role(candidate: VectorSearchResult, question: str, coverage: float) -> str:
    del candidate, question
    return "direct_evidence" if coverage >= DIRECT_EVIDENCE_COVERAGE else "related_context"


def _is_comparison_question(question: str) -> bool:
    normalized = _normalize_for_match(question)
    return any(term in normalized for term in ["区别", "差异", "关系", "等同", "不同", "比较", "与", "和"])


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    replacements = {
        "同一个": "同一",
        "怎样": "怎么",
        "如何": "怎么",
        "应该": "",
        "应当": "",
        "怎么": "",
        "如何": "",
        "什么": "",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _fallback_chinese_terms(text: str) -> list[str]:
    stop_words = ["应该", "应当", "怎么", "如何", "什么", "是否", "可以", "进行", "统计"]
    terms: list[str] = []
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        cleaned = sequence
        for word in stop_words:
            cleaned = cleaned.replace(word, "")
        if 2 <= len(cleaned) <= 8:
            terms.append(cleaned)
        elif len(cleaned) > 8:
            terms.extend(cleaned[index : index + 4] for index in range(0, len(cleaned) - 3, 4))
    return terms


def _expanded_domain_terms(normalized_question: str) -> list[str]:
    expansions: list[str] = []
    if "获奖" in normalized_question:
        expansions.extend(["获奖", "获奖记录", "获奖时间", "获奖等级", "奖项名称"])
    if "荣誉" in normalized_question:
        expansions.extend(["荣誉", "荣誉奖项", "奖学金", "评优", "荣誉称号"])
    if "技能" in normalized_question:
        expansions.extend(["技能", "技能专长", "掌握程度", "熟练程度"])
    if any(term in normalized_question for term in ["区别", "差异", "关系", "等同", "不同", "比较"]):
        expansions.extend(["区别", "差异", "关系", "等同"])
    if "项目" in normalized_question:
        expansions.extend(["项目", "项目经历", "项目背景", "技术栈", "项目成果"])
    if "证书" in normalized_question or "资格" in normalized_question:
        expansions.extend(["证书", "证书编号", "证书名称", "发证机构"])
    if "教育" in normalized_question and "专业" in normalized_question:
        expansions.extend(["教育背景", "学校", "专业", "学位", "毕业"])
    return expansions


def _candidate_evidence_text(candidate: VectorSearchResult) -> str:
    return "\n".join(
        part
        for part in [candidate.section_title or "", candidate.embedding_text or "", candidate.text]
        if part.strip()
    )


def _safe_excerpt(text: str, limit: int = 1200) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    sentences = re.findall(r".+?(?:[。！？!?；;]|$)", cleaned, flags=re.S)
    excerpt = ""
    for sentence in sentences:
        if len(excerpt) + len(sentence) > limit:
            break
        excerpt += sentence
    return excerpt.strip() or cleaned[:limit].strip()


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def _format_elapsed_seconds(elapsed_ms: float | None) -> str:
    if elapsed_ms is None:
        return "0.000s"
    return f"{elapsed_ms / 1000:.3f}s"


def _score_range(candidates: list[VectorSearchResult], reranked: list[RerankedChunk]) -> dict[str, float | None]:
    vector_scores = [candidate.score for candidate in candidates]
    rerank_scores = [item.rerank_score for item in reranked]
    return {
        "vector_min": round(min(vector_scores), 4) if vector_scores else None,
        "vector_max": round(max(vector_scores), 4) if vector_scores else None,
        "rerank_min": round(min(rerank_scores), 4) if rerank_scores else None,
        "rerank_max": round(max(rerank_scores), 4) if rerank_scores else None,
    }
