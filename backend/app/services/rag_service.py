from collections import OrderedDict
from dataclasses import dataclass
import json
import re
from threading import RLock
from time import monotonic, perf_counter
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, load_only

from backend.app.core.config import (
    DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS,
    DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS,
    FALLBACK_DIRECT_GENERATION_ENABLED,
    FALLBACK_LOWER_THRESHOLD_ENABLED,
    FALLBACK_REWRITE_RETRY_ENABLED,
    FINAL_CITATION_LIMIT,
    FORCE_MIN_CHUNKS,
    MAX_PROMPT_CHUNKS,
    MAX_PROMPT_TOKENS,
    MIN_CORE_RERANK_SCORE,
    MIN_EVIDENCE_COVERAGE,
    MIN_LEXICAL_RERANK_SCORE,
    MIN_LEXICAL_SCORE,
    MIN_PROMPT_CHUNKS,
    RELAXED_MIN_PROMPT_CHUNKS,
    RELAXED_RERANK_THRESHOLD,
    RELATIVE_SCORE_RATIO,
    RERANK_TOP_K,
    RERANK_PROMPT_THRESHOLD,
)
from backend.app.models.document import Document, DocumentChunk, QALog
from backend.app.schemas.qa import (
    Citation,
    LLMContextPackage,
    QAAnswerPreview,
    QAResponse,
    RetrievalResult,
)
from backend.app.services.audit_service import record_event
from backend.app.services.answer_generation_service import generate_answer
from backend.app.services.conversation_memory_service import record_turn, resolve_question
from backend.app.services.document_identity_query_service import answer_document_identity_question
from backend.app.services.intent_router_service import classify_intent
from backend.app.services.query_planner_service import (
    QueryAspect,
    QueryPlan,
    QuerySearchQuery,
    plan_query,
    rewrite_search_queries,
)
from backend.app.services.prompt_builder import RAGPromptBuilder
from backend.app.services.question_preprocessing_service import preprocess_qa_request
from backend.app.services.retrieval_service import (
    RetrievalDiagnostics,
    RetrievalMatch,
    RetrievalServiceUnavailable,
    ProgressReporter,
    collect_candidates_with_query_hits,
    evidence_coverage,
    get_last_retrieval_diagnostics,
    limit_rerank_candidates,
    matches_from_reranked,
    question_terms,
    reset_retrieval_diagnostics,
    retrieve_citations,
)
from backend.app.services.embedding_service import EmbeddingServiceError
from backend.app.services.model_device_service import get_model_device_info
from backend.app.services.performance_metrics import measure, trace_operation
from backend.app.services.rerank_service import RerankServiceError, rerank_candidates
from backend.app.services.vector_store_service import VectorStoreError


CONTEXT_INSTRUCTION = (
    "请严格根据检索到的知识片段回答用户问题。不得编造知识库中不存在的材料依据。"
    "若依据不足，请明确说明无法根据当前知识库判断。"
)
CONTEXT_CHUNK_CHAR_LIMIT = 1200
EMPTY_ANSWER_LOG_TEXT = "[RAG_CONTEXT_PACKAGE_ONLY] 当前阶段未接入 LLM，接口仅返回检索上下文包。"
ASPECT_QUERY_FUSION_METHOD = "aspect_query_rrf_then_bge_rerank"
RRF_K = 60
CONSTRAINED_RERANK_CANDIDATE_LIMIT = 20
MCQ_RERANK_CANDIDATE_LIMIT = 24
QUERY_TYPE_WEIGHTS = {
    "semantic_question": 1.0,
    "document_style_statement": 1.15,
    "keyword_anchor": 0.75,
    "legacy": 0.9,
    "fallback": 0.85,
}
_DEFAULT_RETRIEVE_CITATIONS = retrieve_citations
@dataclass(frozen=True)
class _DocumentChunkSnapshot:
    id: int
    chunk_id: str
    document_id: str
    text: str
    embedding_text: str | None
    chunk_metadata: str | None
    index_status: str
    source_file: str
    page_number: int | None
    section_title: str | None


_DOCUMENT_SNAPSHOT_CACHE: OrderedDict[str, tuple[float, list[_DocumentChunkSnapshot]]] = OrderedDict()
_DOCUMENT_SNAPSHOT_CACHE_LOCK = RLock()


def clear_document_snapshot_cache(document_ids: set[str] | list[str] | tuple[str, ...] | None = None) -> None:
    """Drop cached chunk snapshots after indexing, deletion, recovery, or metadata changes."""

    with _DOCUMENT_SNAPSHOT_CACHE_LOCK:
        if document_ids is None:
            _DOCUMENT_SNAPSHOT_CACHE.clear()
            return
        for document_id in document_ids:
            _DOCUMENT_SNAPSHOT_CACHE.pop(str(document_id), None)


@dataclass
class AspectRetrieval:
    aspect: QueryAspect
    candidates: list[RetrievalResult]
    diagnostics: list[dict[str, Any]]
    citation_validation: dict[str, Any]
    selected_chunk_ids: list[str]
    retrieval_covered: bool = False
    covered: bool = False


@trace_operation("qa")
def answer_question(
    db: Session,
    question: str,
    options: list[str] | None = None,
    include_debug: bool = False,
    session_id: str | None = None,
    progress_reporter: ProgressReporter | None = None,
    answer_preview_reporter: Callable[[QAAnswerPreview], None] | None = None,
    cancellation_checker: Callable[[], None] | None = None,
) -> QAResponse:
    """简历面试问答主流程：意图路由 → 追问记忆 → 检索 → 单次自评生成 → 置信度分级。"""
    cleaned_question = question.strip()
    if not cleaned_question:
        return QAResponse(answer=None, answer_mode="failed", generation_status="skipped", context_package=None)
    original_question = cleaned_question

    # 身份快通道（证书真实性/有效期类问题，保留确定性回答）
    identity_response = answer_document_identity_question(db, cleaned_question)
    if identity_response is not None:
        _save_qa_log(db, original_question, identity_response)
        _log_qa_audit(db, "qa_identity_answered", original_question, identity_response)
        return identity_response

    # ① 意图分类（规则优先 + LLM 兜底）
    intent_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {"stage": "intent", "status": "running", "title": "正在识别问题意图", "detail": "正在判断问题类型并选择回答策略……"},
    )
    intent_result = classify_intent(cleaned_question)
    _report_progress(
        progress_reporter,
        {
            "stage": "intent",
            "status": "completed",
            "title": "意图识别完成",
            "detail": f"意图：{intent_result.intent}（{intent_result.classifier}）",
            "elapsed_ms": _elapsed_ms(intent_started_at),
            "summary": {
                "intent": intent_result.intent,
                "classifier": intent_result.classifier,
                "confidence": intent_result.confidence,
                "reason": intent_result.reason,
            },
        },
    )

    # ② 多轮追问记忆（指代消解）
    memory_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {"stage": "memory", "status": "running", "title": "正在结合上下文理解追问", "detail": "正在检查追问是否依赖上一轮对话……"},
    )
    resolved_question, used_memory, _memory_turns = resolve_question(db, session_id, cleaned_question)
    _report_progress(
        progress_reporter,
        {
            "stage": "memory",
            "status": "completed",
            "title": "追问理解完成",
            "detail": "已结合上一轮对话补全问题" if used_memory else "无需指代消解",
            "elapsed_ms": _elapsed_ms(memory_started_at),
            "summary": {"used_memory": used_memory, "resolved_question": resolved_question},
        },
    )

    # ③ 礼貌转移（寒暄/无关话题：零检索零生成）
    if intent_result.strategy.polite_redirect:
        generated = generate_answer(resolved_question, [], intent=intent_result.intent)
        response = QAResponse(
            answer=generated.answer,
            answer_mode=generated.answer_mode,
            intent=intent_result.intent,
            resolved_question=(resolved_question if used_memory else None),
            generation_status=generated.generation_status,
        )
        _record_turn_if_needed(db, session_id, original_question, resolved_question, intent_result.intent, response)
        _save_qa_log(db, original_question, response)
        _log_qa_audit(db, "qa_answered", original_question, response)
        return response

    # ④ 检索与上下文（兜底链：标准 → 降阈 → 改写 → 直接生成）
    fallback_level = 0
    package = build_context_package(
        db,
        resolved_question,
        options=options or [],
        progress_reporter=progress_reporter,
        cancellation_checker=cancellation_checker,
    )
    grade = _evidence_grade(package.context_chunks)
    if grade == "none" and FALLBACK_DIRECT_GENERATION_ENABLED:
        # level 3：无检索直接生成（LLM 基于 persona + 会话历史推理，强制推测标注）
        fallback_level = 3
        package = _empty_context_package(resolved_question)
    elif grade == "weak":
        # level 1：降阈重选（放宽门槛，按 rerank 分数补足候选）
        if FALLBACK_LOWER_THRESHOLD_ENABLED:
            relaxed_package = build_context_package(
                db,
                resolved_question,
                options=options or [],
                progress_reporter=progress_reporter,
                cancellation_checker=cancellation_checker,
                relaxed=True,
            )
            if relaxed_package.context_chunks:
                package = relaxed_package
                fallback_level = 1
        # level 2：LLM 改写查询后重检索
        if (
            fallback_level == 1
            and FALLBACK_REWRITE_RETRY_ENABLED
            and _evidence_grade(package.context_chunks) == "weak"
        ):
            _report_progress(
                progress_reporter,
                {
                    "stage": "rewrite",
                    "status": "running",
                    "title": "正在改写查询重试",
                    "detail": "证据较弱，正在把口语化问题改写为检索查询……",
                },
            )
            queries = rewrite_search_queries(
                resolved_question, intent_result.intent, cancellation_checker
            )
            if queries:
                rewritten_package = build_context_package(
                    db,
                    resolved_question,
                    options=options or [],
                    progress_reporter=progress_reporter,
                    cancellation_checker=cancellation_checker,
                    relaxed=True,
                    rewritten_queries=queries,
                )
                _report_progress(
                    progress_reporter,
                    {
                        "stage": "rewrite",
                        "status": "completed",
                        "title": "改写检索完成",
                        "detail": f"改写为 {len(queries)} 条查询",
                        "summary": {"rewritten_queries": queries},
                    },
                )
                if rewritten_package.context_chunks:
                    package = rewritten_package
                    fallback_level = 2
            else:
                _report_progress(
                    progress_reporter,
                    {
                        "stage": "rewrite",
                        "status": "skipped",
                        "title": "改写检索跳过",
                        "detail": "查询改写未返回结果",
                    },
                )

    # ⑤ 单次自评生成
    generation_started_at = perf_counter()
    preview_revision = 0

    def report_preview(text: str) -> None:
        nonlocal preview_revision
        preview_revision += 1
        if answer_preview_reporter is not None:
            answer_preview_reporter(QAAnswerPreview(answer=text, revision=preview_revision))

    _report_progress(
        progress_reporter,
        {"stage": "generation", "status": "running", "title": "正在生成回答", "detail": "正在基于检索内容组织面试回答……"},
    )
    generated = generate_answer(
        resolved_question,
        package.context_chunks,
        intent=intent_result.intent,
        llm_prompt=package.llm_prompt,
        cancellation_checker=cancellation_checker,
        preview_reporter=report_preview,
    )
    _report_progress(
        progress_reporter,
        {
            "stage": "generation",
            "status": "completed",
            "title": "回答生成完成",
            "detail": f"回答模式：{generated.answer_mode}",
            "elapsed_ms": _elapsed_ms(generation_started_at),
            "summary": {
                "answer_mode": generated.answer_mode,
                "evidence_sufficiency": generated.evidence_sufficiency,
                "degraded": generated.degraded,
            },
        },
    )

    response = QAResponse(
        answer=generated.answer,
        answer_mode=generated.answer_mode,
        evidence_sufficiency=generated.evidence_sufficiency,
        hedge_note=generated.hedge_note,
        intent=intent_result.intent,
        resolved_question=(resolved_question if used_memory else None),
        retrieval_fallback_level=fallback_level,
        context_package=package,
        degraded=generated.degraded,
        generation_status=generated.generation_status,
    )
    _record_turn_if_needed(db, session_id, original_question, resolved_question, intent_result.intent, response)
    _save_qa_log(db, original_question, response)
    _log_qa_audit(db, "qa_answered", original_question, response)
    return response


def _record_turn_if_needed(
    db: Session,
    session_id: str | None,
    question: str,
    resolved_question: str,
    intent: str,
    response: QAResponse,
) -> None:
    if not session_id:
        return
    record_turn(
        db,
        session_id,
        question=question,
        resolved_question=resolved_question,
        intent=intent,
        answer_mode=response.answer_mode,
        answer_excerpt=response.answer,
    )


def retrieve_context_package(
    db: Session,
    question: str,
    options: list[str] | None = None,
) -> LLMContextPackage:
    cleaned_question = question.strip()
    if not cleaned_question:
        return _empty_context_package("")
    preprocessed = preprocess_qa_request(cleaned_question, options)
    return build_context_package(db, preprocessed.question, options=preprocessed.options)


def build_context_package(
    db: Session,
    question: str,
    options: list[str] | None = None,
    progress_reporter: ProgressReporter | None = None,
    cancellation_checker: Callable[[], None] | None = None,
    *,
    relaxed: bool = False,
    rewritten_queries: list[str] | None = None,
) -> LLMContextPackage:
    planning_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "running",
            "title": "正在理解问题",
            "detail": "正在拆分问题并生成检索计划……",
        },
    )
    if rewritten_queries:
        # 兜底链第 2 级：用 LLM 改写后的查询构造单方面检索计划
        aspect = QueryAspect(
            aspect_id="rewritten",
            question=question,
            search_queries=tuple(
                QuerySearchQuery(query, "rewritten") for query in rewritten_queries
            ),
            evidence_need="相关材料依据",
            keywords=(),
        )
        query_plan = QueryPlan(
            original_question=question, aspects=(aspect,), planner="rewrite"
        )
    else:
        query_plan = plan_query(question, options=options, cancellation_checker=cancellation_checker)
    planning_elapsed_ms = _elapsed_ms(planning_started_at)
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "completed",
            "title": "问题理解完成",
            "detail": f"已拆分为 {len(query_plan.aspects)} 个方面，用时 {_format_elapsed_seconds(planning_elapsed_ms)}",
            "elapsed_ms": planning_elapsed_ms,
            "summary": {
                "aspect_count": len(query_plan.aspects),
                "aspects": [aspect.to_debug_dict() for aspect in query_plan.aspects],
                "planner": query_plan.planner,
                "fallback_used": query_plan.fallback_used,
            },
        },
    )
    aspect_retrievals = _retrieve_aspects(db, query_plan, progress_reporter=progress_reporter)
    citation_validation = {
        "checked_chunks": sum(int(item.citation_validation.get("checked_chunks") or 0) for item in aspect_retrievals),
        "valid_chunks": sum(int(item.citation_validation.get("valid_chunks") or 0) for item in aspect_retrievals),
        "invalid_chunks": sum(int(item.citation_validation.get("invalid_chunks") or 0) for item in aspect_retrievals),
        "invalid_chunk_ids": [
            chunk_id
            for item in aspect_retrievals
            for chunk_id in item.citation_validation.get("invalid_chunk_ids", [])
        ],
    }
    context_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "running",
            "title": "正在精选最终上下文",
            "detail": "正在按问题方面选择可引用依据……",
        },
    )
    context_chunks, prompt_selection = _select_prompt_chunks(
        question, query_plan, aspect_retrievals, relaxed=relaxed
    )
    _sync_aspect_coverage_from_prompt(context_chunks, aspect_retrievals)
    prompt_selection["final_prompt_chunks"] = len(context_chunks)
    prompt_selection["final_prompt_chunk_ids"] = [chunk.chunk_id for chunk in context_chunks]
    prompt_selection["covered_aspects"] = [
        aspect.aspect_id for aspect in query_plan.aspects if any(
            aspect.aspect_id == retrieval.aspect.aspect_id and retrieval.covered
            for retrieval in aspect_retrievals
        )
    ]
    prompt_selection["retrieval_covered_aspects"] = [
        retrieval.aspect.aspect_id for retrieval in aspect_retrievals if retrieval.retrieval_covered
    ]
    prompt_selection["covered_by_retrieval_but_not_prompted"] = [
        retrieval.aspect.aspect_id
        for retrieval in aspect_retrievals
        if retrieval.retrieval_covered and not retrieval.covered
    ]
    prompt_selection["aspect_selected_chunk_ids"] = {
        retrieval.aspect.aspect_id: retrieval.selected_chunk_ids
        for retrieval in aspect_retrievals
    }
    prompt_selection["prompt_capacity_limited"] = bool(
        prompt_selection["covered_by_retrieval_but_not_prompted"]
    )
    context_elapsed_ms = _elapsed_ms(context_started_at)
    prompt_covered_aspect_count = sum(1 for item in aspect_retrievals if item.covered)
    retrieval_covered_aspect_count = sum(1 for item in aspect_retrievals if item.retrieval_covered)
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "completed",
            "title": "上下文精选完成",
            "detail": (
                f"最终使用 {len(context_chunks)}/{MAX_PROMPT_CHUNKS} 个片段，"
                f"检索覆盖 {retrieval_covered_aspect_count}/{len(query_plan.aspects)} 个方面，"
                f"进入 Prompt {prompt_covered_aspect_count}/{len(query_plan.aspects)} 个方面"
            ),
            "elapsed_ms": context_elapsed_ms,
            "summary": {
                "used_chunks": len(context_chunks),
                "max_prompt_chunks": MAX_PROMPT_CHUNKS,
                "covered_aspects": prompt_covered_aspect_count,
                "retrieval_covered_aspects": retrieval_covered_aspect_count,
                "total_aspects": len(query_plan.aspects),
            },
        },
    )
    # 列举类问题（"介绍你的项目/哪些项目/项目经历"）文档级覆盖补全：
    # 向量检索可能只命中部分项目文档，把缺失的《项目介绍_*.md》文档 chunk 补进上下文，
    # 保证 LLM 能列出全部项目（技术亮点：枚举召回兜底）
    context_chunks, _project_covered_added = _ensure_project_documents_covered(db, question, context_chunks)
    diagnostics = _aggregate_aspect_diagnostics(aspect_retrievals)
    retrieval_summary = _build_retrieval_summary(
        question,
        context_chunks,
        query_plan,
        aspect_retrievals,
        diagnostics,
        citation_validation,
        prompt_selection,
    )
    prompt_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "running",
            "title": "正在构造 LLM Prompt",
            "detail": "正在将问题和依据片段组装为后续 LLM 输入……",
        },
    )
    prompt = RAGPromptBuilder().build(question, context_chunks)
    prompt_elapsed_ms = _elapsed_ms(prompt_started_at)
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "completed",
            "title": "Prompt 构造完成",
            "detail": f"已完成 Prompt 构造，用时 {_format_elapsed_seconds(prompt_elapsed_ms)}",
            "elapsed_ms": prompt_elapsed_ms,
            "summary": {"prompt_chunk_count": len(context_chunks)},
        },
    )
    return LLMContextPackage(
        query=question,
        instruction=CONTEXT_INSTRUCTION,
        retrieval_summary=retrieval_summary,
        context_chunks=context_chunks,
        llm_prompt=prompt,
    )


_PROJECT_DOC_PREFIX = "项目介绍_"
_ENUMERATION_PATTERNS = (
    "哪些项目", "项目经历", "做过哪些", "项目有哪些", "介绍你的项目", "介绍下你的项目",
    "所有项目", "参与过哪些", "项目都有", "做过什么项目", "有什么项目", "介绍一下你的项目",
)


def _is_enumeration_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", str(question or ""))
    return any(pattern in normalized for pattern in _ENUMERATION_PATTERNS)


def _ensure_project_documents_covered(
    db: Session,
    question: str,
    context_chunks: list[RetrievalResult],
) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
    """列举类问题：确保每个项目文档的"开头简介块"都进入上下文。

    向量检索对"介绍你的项目"这类宽泛问题常只召回部分项目文档（且召回的
    可能是 FAQ/选型等非简介块），导致 LLM 只能列出被检索到的项目。
    此处对每个《项目介绍_*.md》文档取头部 2 块（文档开头=项目简介），
    若不在上下文中则直接补入（chunk_id 去重）。
    """
    if not _is_enumeration_question(question):
        return context_chunks, []
    existing_ids = {chunk.chunk_id for chunk in context_chunks}
    project_docs = db.scalars(
        select(Document).where(
            Document.status == "indexed",
            Document.filename.like(f"{_PROJECT_DOC_PREFIX}%"),
        )
    ).all()
    added: list[RetrievalResult] = []
    for doc in project_docs:
        chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc.document_id)
            .order_by(DocumentChunk.id.asc())
            .limit(2)
        ).all()
        for chunk in chunks[:2]:
            if chunk.chunk_id in existing_ids:
                continue
            existing_ids.add(chunk.chunk_id)
            added.append(_project_chunk_to_result(doc, chunk))
    if not added:
        return context_chunks, []
    merged = list(context_chunks) + added
    _renumber_context_chunks(merged)
    return merged, added


def _project_chunk_to_result(document: Document, chunk: DocumentChunk) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        rank=0,
        score=0.5,
        source_doc=document.filename,
        section_title=chunk.section_title or None,
        section_path=[],
        text=chunk.text,
        citation_label=f"[{document.filename}]",
        metadata={"evidence_role": "project_enumeration_cover", "rerank_score": 0.5},
    )


def _evidence_grade(context_chunks: list[RetrievalResult]) -> str:
    """证据强度评估：none=零候选 / weak=分数过低或数量不足 / strong=足够。

    兜底链触发依据——weak 与 none 不拒答，而是降级放宽或直接生成 + 推测标注。
    """
    if not context_chunks:
        return "none"
    top = max((_prompt_score(chunk) for chunk in context_chunks), default=0.0)
    if top < RELAXED_RERANK_THRESHOLD or len(context_chunks) < RELAXED_MIN_PROMPT_CHUNKS:
        return "weak"
    return "strong"


def _empty_context_package(question: str) -> LLMContextPackage:
    prompt = RAGPromptBuilder().build(question, [])
    query_plan = plan_query(question) if question else QueryPlan("", (), "empty", fallback_used=True)
    return LLMContextPackage(
        query=question,
        instruction=CONTEXT_INSTRUCTION,
        retrieval_summary={
            "top_k": FINAL_CITATION_LIMIT,
            "used_chunks": 0,
            "has_sufficient_context": False,
            "coverage_notes": [],
            "missing_aspects": ["未召回可用知识片段"],
            "query_count": 0,
            "candidate_count": 0,
            "reranked_count": 0,
            "filtered_count": 0,
            "timings_ms": {},
            "score_range": {},
            "model_device": get_model_device_info().to_debug_dict(),
            "prompt_filtered_count": 0,
            "prompt_selection": _empty_prompt_selection(),
            "query_plan": query_plan.to_debug_dict(),
            "aspect_retrievals": [],
            "final_prompt_chunk_ids": [],
        },
        context_chunks=[],
        llm_prompt=prompt,
    )


def _retrieve_aspects(
    db: Session,
    query_plan: QueryPlan,
    progress_reporter: ProgressReporter | None = None,
) -> list[AspectRetrieval]:
    retrieval_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "running",
            "title": "正在检索相关依据",
            "detail": f"正在按 {len(query_plan.aspects)} 个问题方面召回知识库片段……",
            "summary": {"total_aspects": len(query_plan.aspects)},
        },
    )
    aspect_retrievals: list[AspectRetrieval] = []
    document_chunk_cache: dict[str, list[DocumentChunk]] = {}
    for aspect_index, aspect in enumerate(query_plan.aspects, start=1):
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "running",
                "title": f"正在检索方面 {aspect_index}",
                "detail": f"正在为“{aspect.question}”检索相关依据……",
                "aspect_id": aspect.aspect_id,
                "summary": {
                    "aspect_index": aspect_index,
                    "total_aspects": len(query_plan.aspects),
                    "aspect_question": aspect.question,
                },
            },
        )
        matches, diagnostics = _retrieve_aspect_matches(
            db,
            aspect,
            progress_reporter=progress_reporter,
            document_chunk_cache=document_chunk_cache,
        )
        matches = _expand_neighbor_matches(
            db,
            aspect.question,
            matches,
            document_chunk_cache=document_chunk_cache,
        )
        candidates = _to_retrieval_results(matches)
        for candidate in candidates:
            candidate.metadata["aspect_id"] = aspect.aspect_id
            candidate.metadata["aspect_question"] = aspect.question
            candidate.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
            candidate.metadata["expected_evidence_type"] = aspect.expected_evidence_type
            candidate.metadata["evidence_need"] = aspect.evidence_need
            candidate.metadata.setdefault("prompt_matched_aspects", [aspect.aspect_id])
        valid_candidates, citation_validation = _validate_context_chunks(db, candidates)
        aspect_retrieval = AspectRetrieval(
            aspect=aspect,
            candidates=valid_candidates,
            diagnostics=diagnostics,
            citation_validation=citation_validation,
            selected_chunk_ids=[],
            retrieval_covered=bool(valid_candidates),
            covered=False,
        )
        aspect_retrievals.append(aspect_retrieval)
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "completed" if valid_candidates else "failed",
                "title": f"方面 {aspect_index} 检索完成" if valid_candidates else f"方面 {aspect_index} 未找到足够依据",
                "detail": (
                    f"已为“{aspect.question}”召回 {len(valid_candidates)} 个可回溯候选片段"
                    if valid_candidates
                    else f"“{aspect.question}”暂未召回可回溯依据"
                ),
                "aspect_id": aspect.aspect_id,
                "summary": {
                    "aspect_index": aspect_index,
                    "total_aspects": len(query_plan.aspects),
                    "candidate_count": len(valid_candidates),
                    "aspect_question": aspect.question,
                    "retrieved_sections": [
                        chunk.section_title for chunk in valid_candidates if chunk.section_title
                    ],
                },
            },
        )
    _recover_missing_aspects_from_sibling_documents(
        db,
        aspect_retrievals,
        document_chunk_cache=document_chunk_cache,
    )
    retrieval_elapsed_ms = _elapsed_ms(retrieval_started_at)
    retrieved_aspect_count = sum(1 for item in aspect_retrievals if item.retrieval_covered)
    candidate_count = sum(len(item.candidates) for item in aspect_retrievals)
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "completed",
            "title": "检索相关依据完成",
            "detail": (
                f"已完成 {len(query_plan.aspects)} 个方面检索，"
                f"召回 {candidate_count} 个可回溯候选片段。"
            ),
            "elapsed_ms": retrieval_elapsed_ms,
            "summary": {
                "total_aspects": len(query_plan.aspects),
                "retrieved_aspects": retrieved_aspect_count,
                "candidate_count": candidate_count,
            },
        },
    )
    _report_rerank_stage_summary(progress_reporter, aspect_retrievals)
    return aspect_retrievals


def _recover_missing_aspects_from_sibling_documents(
    db: Session,
    aspect_retrievals: list[AspectRetrieval],
    *,
    document_chunk_cache: dict[str, list[DocumentChunk]],
) -> None:
    """Supplement a sub-question inside documents found for its siblings.

    Cross-document questions often identify a source through one sub-question
    while a short definition sub-question has no metadata of its own.  A weak
    but non-empty retrieval is not proof that the needed clause was found, so
    this pass also supplements already-covered aspects when a sibling document
    contains stronger direct lexical support.  It searches only documents
    already retrieved for the same request; it never scans the corpus or
    evaluator answers.
    """

    if len(aspect_retrievals) < 2:
        return
    documents_by_aspect: list[list[str]] = []
    for retrieval in aspect_retrievals:
        documents_by_aspect.append(
            list(
                dict.fromkeys(
                    str(candidate.metadata.get("document_id") or "")
                    for candidate in retrieval.candidates
                    if candidate.metadata.get("document_id")
                )
            )[:4]
        )
    # Round-robin selection prevents the first broad aspect from consuming the
    # whole bounded scope before later, more source-specific aspects contribute.
    request_document_ids: list[str] = []
    for rank in range(4):
        for document_ids in documents_by_aspect:
            if rank < len(document_ids) and document_ids[rank] not in request_document_ids:
                request_document_ids.append(document_ids[rank])
            if len(request_document_ids) >= 12:
                break
        if len(request_document_ids) >= 12:
            break
    if not request_document_ids:
        return
    for retrieval, own_document_ids in zip(aspect_retrievals, documents_by_aspect, strict=True):
        sibling_document_ids = set(request_document_ids) - set(own_document_ids)
        if not sibling_document_ids:
            continue
        matches = _bounded_document_lexical_support_matches(
            db,
            retrieval.aspect,
            [],
            document_ids=sibling_document_ids,
            document_chunk_cache=document_chunk_cache,
        )
        if not matches:
            continue
        matches = _expand_neighbor_matches(
            db,
            retrieval.aspect.question,
            matches,
            document_chunk_cache=document_chunk_cache,
        )
        candidates = _to_retrieval_results(matches)
        for candidate in candidates:
            candidate.metadata["aspect_id"] = retrieval.aspect.aspect_id
            candidate.metadata["aspect_question"] = retrieval.aspect.question
            candidate.metadata["aspect_search_queries"] = _search_query_debug_list(
                retrieval.aspect
            )
            candidate.metadata["expected_evidence_type"] = retrieval.aspect.expected_evidence_type
            candidate.metadata["evidence_need"] = retrieval.aspect.evidence_need
            candidate.metadata.setdefault(
                "prompt_matched_aspects", [retrieval.aspect.aspect_id]
            )
        valid_candidates, citation_validation = _validate_context_chunks(db, candidates)
        if not valid_candidates:
            continue
        recovered_chunk_ids = {candidate.chunk_id for candidate in valid_candidates}
        retrieval.candidates = valid_candidates + [
            candidate
            for candidate in retrieval.candidates
            if candidate.chunk_id not in recovered_chunk_ids
        ]
        retrieval.citation_validation = {
            "checked_chunks": int(retrieval.citation_validation.get("checked_chunks") or 0)
            + int(citation_validation.get("checked_chunks") or 0),
            "valid_chunks": int(retrieval.citation_validation.get("valid_chunks") or 0)
            + int(citation_validation.get("valid_chunks") or 0),
            "invalid_chunks": int(retrieval.citation_validation.get("invalid_chunks") or 0)
            + int(citation_validation.get("invalid_chunks") or 0),
            "invalid_chunk_ids": [
                *retrieval.citation_validation.get("invalid_chunk_ids", []),
                *citation_validation.get("invalid_chunk_ids", []),
            ],
        }
        retrieval.retrieval_covered = True
        retrieval.diagnostics.append(
            {
                "query_type": "sibling_document_lexical_recovery",
                "search_query": retrieval.aspect.question,
                "document_scope_count": len(sibling_document_ids),
                "match_count": len(valid_candidates),
                "recovery_mode": "supplemented" if own_document_ids else "recovered",
            }
        )


def _retrieve_aspect_matches(
    db: Session,
    aspect: QueryAspect,
    progress_reporter: ProgressReporter | None = None,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> tuple[list[RetrievalMatch], list[dict[str, Any]]]:
    if retrieve_citations is not _DEFAULT_RETRIEVE_CITATIONS:
        return _retrieve_aspect_matches_legacy_hook(db, aspect, progress_reporter)

    fusion_scores: dict[str, float] = {}
    diagnostics = RetrievalDiagnostics()
    total_started_at = perf_counter()

    search_queries = [search_query.query for search_query in aspect.search_queries]
    query_metadata = [
        {
            "query_type": search_query.query_type,
            "rationale": search_query.rationale,
        }
        for search_query in aspect.search_queries
    ]
    try:
        candidates, query_hits_by_chunk_id, diagnostics_by_query = collect_candidates_with_query_hits(
            search_queries,
            query_metadata=query_metadata,
            diagnostics=diagnostics,
        )
    except (EmbeddingServiceError, VectorStoreError) as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc
    for candidate in candidates:
        for hit in query_hits_by_chunk_id.get(candidate.chunk_id, []):
            query_type = str(hit.get("query_type") or "semantic_question")
            query_weight = QUERY_TYPE_WEIGHTS.get(query_type, 1.0)
            rank = int(hit.get("rank") or 1)
            contribution = query_weight / (RRF_K + rank)
            fusion_scores[candidate.chunk_id] = fusion_scores.get(candidate.chunk_id, 0.0) + contribution
            hit["rrf_contribution"] = round(contribution, 6)

    pre_filter_candidate_count = len(candidates)
    diagnostics.candidate_count = len(candidates)
    document_style_scores = {
        candidate.chunk_id: _document_style_evidence_score(candidate, aspect)
        for candidate in candidates
    }
    for candidate in candidates:
        if candidate.metadata is None:
            candidate.metadata = {}
        candidate.metadata["document_style_evidence_score"] = round(
            document_style_scores.get(candidate.chunk_id, 0.0), 6
        )
    candidates.sort(
        key=lambda candidate: (
            document_style_scores.get(candidate.chunk_id, 0.0),
            fusion_scores.get(candidate.chunk_id, 0.0),
            candidate.score,
        ),
        reverse=True,
    )
    # ``candidates`` is already ordered by weighted RRF above.  Re-sorting it
    # by one raw vector score here would undo multi-query fusion and starve
    # exact option/statement hits in long documents before cross-encoding.
    effective_rerank_limit = _effective_rerank_candidate_limit(candidates)
    rerank_input = limit_rerank_candidates(
        candidates,
        preserve_order=True,
        limit=effective_rerank_limit,
    )
    diagnostics.rerank_input_count = len(rerank_input)
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "running",
            "title": "正在重排候选片段",
            "detail": f"方面“{aspect.question}”融合后进入重排 {diagnostics.rerank_input_count} 个片段……",
            "aspect_id": aspect.aspect_id,
            "summary": {
                "aspect_id": aspect.aspect_id,
                "candidate_count": diagnostics.candidate_count,
                "rerank_input_count": diagnostics.rerank_input_count,
                "effective_rerank_candidate_limit": effective_rerank_limit,
            },
        },
    )
    rerank_started_at = perf_counter()
    try:
        reranked = rerank_candidates(
            question=_rerank_query_for_aspect(aspect),
            candidates=rerank_input,
            limit=RERANK_TOP_K,
        )
    except RerankServiceError as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc
    diagnostics.rerank_call_count = 1 if rerank_input else 0
    diagnostics.timings_ms["rerank"] = _elapsed_ms(rerank_started_at)
    diagnostics.reranked_count = len(reranked)
    diagnostics.score_range = _score_range_for_candidates(candidates, reranked)
    matches = matches_from_reranked(
        question=aspect.question,
        reranked=reranked,
        diagnostics=diagnostics,
    )
    diagnostics.timings_ms["total"] = _elapsed_ms(total_started_at)
    matches = _filter_indexed_matches(db, matches)
    for match in matches:
        chunk_id = match.citation.chunk_id
        match.metadata["aspect_id"] = aspect.aspect_id
        match.metadata["aspect_question"] = aspect.question
        match.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
        match.metadata["aspect_search_query_hits"] = query_hits_by_chunk_id.get(chunk_id, [])
        match.metadata["aspect_query_fusion_score"] = round(fusion_scores.get(chunk_id, 0.0), 6)
        match.metadata["fusion_method"] = ASPECT_QUERY_FUSION_METHOD
        match.metadata["expected_evidence_type"] = aspect.expected_evidence_type
        match.metadata["evidence_need"] = aspect.evidence_need

    lexical_scope_document_ids: set[str] = set()
    lexical_scope_document_ids.update(
        _candidate_document_id(item.candidate)
        for item in reranked[:24]
        if _candidate_document_id(item.candidate)
    )
    # Keep the clause-recovery pass inside documents already recalled by
    # dense/sparse/hybrid retrieval.  Missing aspects are recovered by the
    # generic planner split and their own hybrid queries; lexical support only
    # repairs within-document clause ranking and must not become a second
    # unbounded retriever.
    lexical_support_matches = _bounded_document_lexical_support_matches(
        db,
        aspect,
        matches,
        document_ids=lexical_scope_document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    if lexical_support_matches:
        best_fusion = max(fusion_scores.values(), default=0.0)
        for offset, supplement in enumerate(lexical_support_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = best_fusion + 0.01 - offset * 0.0001
        existing_chunk_ids = {match.citation.chunk_id for match in lexical_support_matches}
        matches = lexical_support_matches + [
            match for match in matches if match.citation.chunk_id not in existing_chunk_ids
        ]

    for item in diagnostics_by_query:
        chunk_ids_for_query = {
            chunk_id
            for chunk_id, hits in query_hits_by_chunk_id.items()
            if any(hit.get("query") == item.get("search_query") for hit in hits)
        }
        item["match_count"] = sum(
            1 for match in matches if match.citation.chunk_id in chunk_ids_for_query
        )

    diagnostics_by_query.append(
        {
            "search_query": aspect.question,
            "query_type": "aspect_fused",
            "rationale": "同一 aspect 的多条 search query 先召回融合，再单次 BGE rerank",
            "match_count": len(matches),
            "pre_filter_candidate_count": pre_filter_candidate_count,
            "post_filter_candidate_count": len(candidates),
            "rerank_input_ranking": [
                {
                    "rank": rank,
                    "chunk_id": candidate.chunk_id,
                    "fusion_score": round(fusion_scores.get(candidate.chunk_id, 0.0), 6),
                    "vector_score": round(float(candidate.score), 6),
                }
                for rank, candidate in enumerate(rerank_input, start=1)
            ],
            "reranked_ranking": [
                {
                    "rank": rank,
                    "chunk_id": item.candidate.chunk_id,
                    "rerank_score": round(float(item.rerank_score), 8),
                }
                for rank, item in enumerate(reranked, start=1)
            ],
            "effective_rerank_candidate_limit": effective_rerank_limit,
            **diagnostics.to_summary_fields(),
        }
    )
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "completed",
            "title": "候选重排完成",
            "detail": f"方面“{aspect.question}”完成 1 次融合重排，输出 {diagnostics.reranked_count} 个片段",
            "aspect_id": aspect.aspect_id,
            "elapsed_ms": diagnostics.timings_ms.get("rerank"),
            "summary": {
                "aspect_id": aspect.aspect_id,
                "rerank_call_count": diagnostics.rerank_call_count,
                "rerank_input_count": diagnostics.rerank_input_count,
                "reranked_count": diagnostics.reranked_count,
            },
        },
    )
    matches.sort(
        key=lambda match: (
            fusion_scores.get(match.citation.chunk_id, 0.0),
            match.rerank_score,
            match.score,
        ),
        reverse=True,
    )
    return matches, diagnostics_by_query


def _retrieve_aspect_matches_legacy_hook(
    db: Session,
    aspect: QueryAspect,
    progress_reporter: ProgressReporter | None = None,
) -> tuple[list[RetrievalMatch], list[dict[str, Any]]]:
    fused_by_chunk_id: dict[str, RetrievalMatch] = {}
    fusion_scores: dict[str, float] = {}
    fusion_query_hits: dict[str, list[dict[str, Any]]] = {}
    diagnostics_by_query: list[dict[str, Any]] = []

    for search_query in aspect.search_queries:
        reset_retrieval_diagnostics()
        matches = _filter_indexed_matches(
            db,
            _retrieve_citations_with_optional_progress(search_query.query, progress_reporter),
        )
        diagnostics = get_last_retrieval_diagnostics()
        diagnostics_by_query.append(
            {
                "search_query": search_query.query,
                "query_type": search_query.query_type,
                "rationale": search_query.rationale,
                "match_count": len(matches),
                **diagnostics.to_summary_fields(),
            }
        )
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "completed",
                "title": "候选重排完成",
                "detail": f"已完成 {diagnostics.reranked_count} 个片段重排",
                "aspect_id": aspect.aspect_id,
                "elapsed_ms": diagnostics.timings_ms.get("rerank"),
                "summary": {
                    "search_query": search_query.query,
                    "query_type": search_query.query_type,
                    "reranked_count": diagnostics.reranked_count,
                },
            },
        )
        query_weight = QUERY_TYPE_WEIGHTS.get(search_query.query_type, 1.0)
        for rank, match in enumerate(matches, start=1):
            chunk_id = match.citation.chunk_id
            if not chunk_id:
                continue
            fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0.0) + query_weight / (RRF_K + rank)
            fusion_query_hits.setdefault(chunk_id, []).append(
                {
                    "query": search_query.query,
                    "query_type": search_query.query_type,
                    "rationale": search_query.rationale,
                    "rank": rank,
                    "rrf_contribution": round(query_weight / (RRF_K + rank), 6),
                }
            )
            existing = fused_by_chunk_id.get(chunk_id)
            if existing is None or _match_sort_key(match) > _match_sort_key(existing):
                fused_by_chunk_id[chunk_id] = match

    fused_matches = list(fused_by_chunk_id.values())
    for match in fused_matches:
        chunk_id = match.citation.chunk_id
        match.metadata["aspect_id"] = aspect.aspect_id
        match.metadata["aspect_question"] = aspect.question
        match.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
        match.metadata["aspect_search_query_hits"] = fusion_query_hits.get(chunk_id, [])
        match.metadata["aspect_query_fusion_score"] = round(fusion_scores.get(chunk_id, 0.0), 6)
        match.metadata["fusion_method"] = "query_plan_rrf"
        match.metadata["expected_evidence_type"] = aspect.expected_evidence_type
        match.metadata["evidence_need"] = aspect.evidence_need

    fused_matches.sort(
        key=lambda match: (
            fusion_scores.get(match.citation.chunk_id, 0.0),
            match.rerank_score,
            match.score,
        ),
        reverse=True,
    )
    return fused_matches, diagnostics_by_query


def _retrieve_citations_with_optional_progress(
    search_query: str,
    progress_reporter: ProgressReporter | None,
) -> list[RetrievalMatch]:
    if progress_reporter is None:
        return retrieve_citations(search_query)
    try:
        return retrieve_citations(search_query, progress_reporter=progress_reporter)
    except TypeError as exc:
        if "progress_reporter" not in str(exc):
            raise
        return retrieve_citations(search_query)


def _report_rerank_stage_summary(
    progress_reporter: ProgressReporter | None,
    aspect_retrievals: list[AspectRetrieval],
) -> None:
    diagnostics = [
        diagnostic
        for aspect_retrieval in aspect_retrievals
        for diagnostic in aspect_retrieval.diagnostics
    ]
    rerank_input_count = sum(int(item.get("rerank_input_count") or 0) for item in diagnostics)
    rerank_call_count = sum(int(item.get("rerank_call_count") or 0) for item in diagnostics)
    reranked_count = sum(int(item.get("reranked_count") or 0) for item in diagnostics)
    candidate_count = sum(len(item.candidates) for item in aspect_retrievals)
    if rerank_input_count or rerank_call_count or reranked_count:
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "completed",
                "title": "重排候选片段完成",
                "detail": f"已完成 {rerank_call_count} 次重排，输出 {reranked_count} 个候选片段。",
                "summary": {
                    "candidate_count": candidate_count,
                    "rerank_input_count": rerank_input_count,
                    "rerank_call_count": rerank_call_count,
                    "reranked_count": reranked_count,
                },
            },
        )
        return

    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "skipped",
            "title": "重排候选片段已跳过",
            "detail": (
                "检索未产生需要重排的候选片段。"
                if candidate_count == 0
                else "当前候选片段已由结构化检索或兼容检索直接排序，无需额外重排。"
            ),
            "summary": {
                "candidate_count": candidate_count,
                "rerank_input_count": rerank_input_count,
                "rerank_call_count": rerank_call_count,
                "reranked_count": reranked_count,
            },
        },
    )


def _report_progress(progress_reporter: ProgressReporter | None, event: dict[str, Any]) -> None:
    if progress_reporter is not None:
        progress_reporter(event)


def _match_sort_key(match: RetrievalMatch) -> tuple[float, float, float]:
    return (
        float(match.rerank_score or 0.0),
        float(match.coverage_score or 0.0),
        float(match.score or 0.0),
    )


def _search_query_debug_list(aspect: QueryAspect) -> list[dict[str, Any]]:
    return [search_query.to_debug_dict() for search_query in aspect.search_queries]


def _rerank_query_for_aspect(aspect: QueryAspect) -> str:
    evidence_queries = [
        search_query.query
        for search_query in aspect.search_queries
        if search_query.query_type == "document_style_statement" and search_query.query.strip()
    ]
    if not evidence_queries:
        return aspect.question
    return "\n".join([aspect.question, *evidence_queries])


def _effective_rerank_candidate_limit(candidates: list[Any]) -> int:
    if len(candidates) <= CONSTRAINED_RERANK_CANDIDATE_LIMIT:
        return len(candidates)
    return max(CONSTRAINED_RERANK_CANDIDATE_LIMIT, min(len(candidates), MCQ_RERANK_CANDIDATE_LIMIT))


def _document_style_evidence_score(candidate: Any, aspect: QueryAspect) -> float:
    statements = [
        statement.strip()
        for search_query in aspect.search_queries
        if search_query.query_type == "document_style_statement"
        for statement in search_query.query.splitlines()
        if statement.strip()
    ]
    if not statements:
        return 0.0
    evidence_text = "\n".join(
        part
        for part in (
            candidate.filename,
            candidate.section_title or "",
            candidate.embedding_text or "",
            candidate.text,
        )
        if part
    )
    return max(
        evidence_coverage(question_terms(statement), evidence_text)
        for statement in statements
    )


def _direct_exact_support_match(
    chunk: Any,
    document_chunks: list[Any],
    aspect: QueryAspect,
    anchor: str,
    support_score: float,
) -> RetrievalMatch:
    previous_chunk_id, next_chunk_id = _adjacent_chunk_ids(chunk, document_chunks)
    section_number = _section_number(chunk.section_title)
    text = _clean_chunk_text(chunk.text, chunk.section_title)
    normalized_score = min(0.99, 0.88 + min(support_score, 2.0) * 0.05)
    return RetrievalMatch(
        citation=Citation(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            filename=chunk.source_file or "",
            section_title=chunk.section_title,
            section_path=_inferred_section_path(chunk, document_chunks),
            section_number=section_number,
            parent_section_number=_parent_section_number(section_number),
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            page_number=chunk.page_number,
            excerpt=text,
            score=normalized_score,
            rerank_score=normalized_score,
            chunk_type=_chunk_type(chunk.text),
            evidence_role="exact_anchor_support",
        ),
        score=normalized_score,
        rerank_score=normalized_score,
        coverage_score=evidence_coverage(
            question_terms(aspect.question),
            f"{chunk.section_title or ''}\n{chunk.embedding_text or ''}\n{text}",
        ),
        evidence_role="exact_anchor_support",
        evidence_text=text,
        metadata={
            **_chunk_metadata(chunk),
            "aspect_id": aspect.aspect_id,
            "aspect_question": aspect.question,
            "aspect_search_queries": _search_query_debug_list(aspect),
            "expected_evidence_type": aspect.expected_evidence_type,
            "evidence_need": aspect.evidence_need,
            "evidence_role": "exact_anchor_support",
            "exact_support_anchor": anchor,
            "exact_support_score": round(support_score, 4),
            "fusion_method": "bounded_document_exact_anchor",
        },
    )


def _bounded_document_lexical_support_matches(
    db: Session,
    aspect: QueryAspect,
    matches: list[RetrievalMatch],
    *,
    document_ids: set[str],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    """Recover strongly overlapping clauses inside metadata-resolved documents.

    This is intentionally bounded by document identity.  It is a generic
    safeguard for long materials where a title filter resolves correctly
    but vector retrieval misses one clause of a compound question.  It never
    searches evaluator answers and does not broaden an explicit document
    scope to the full corpus.
    """

    if not hasattr(db, "scalars"):
        return []
    scope_document_ids = set(document_ids) or {
        match.citation.document_id for match in matches[:24] if match.citation.document_id
    }
    if not scope_document_ids:
        return []
    phrases = _bounded_lexical_phrases(aspect)
    if not phrases:
        return []
    required_numeric_terms = _bounded_required_numeric_terms(aspect)
    chunks_by_document = _chunks_by_document(
        db,
        scope_document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    existing_ids = {match.citation.chunk_id for match in matches}
    supplements: list[tuple[float, int, RetrievalMatch]] = []
    seen_ids: set[str] = set()
    for phrase in phrases:
        normalized_phrase = _normalize_exact_support_text(phrase)
        if len(normalized_phrase) < 6:
            continue
        best: tuple[float, Any, list[Any]] | None = None
        for document_chunks in chunks_by_document.values():
            for chunk in document_chunks:
                if chunk.index_status != "indexed":
                    continue
                searchable = "\n".join(
                    part
                    for part in (
                        chunk.section_title or "",
                        chunk.text or chunk.embedding_text or "",
                    )
                    if part
                )
                normalized_searchable = _normalize_exact_support_text(searchable)
                if required_numeric_terms and not all(
                    _normalize_exact_support_text(term) in normalized_searchable
                    for term in required_numeric_terms
                ):
                    continue
                recall = _exact_support_recall(phrase, searchable)
                exact = normalized_phrase in normalized_searchable
                definition = bool(
                    exact
                    and any(
                        marker in normalized_searchable
                        for marker in (
                            f"本办法所称{normalized_phrase}",
                            f"所称{normalized_phrase}",
                            f"{normalized_phrase}是指",
                            f"{normalized_phrase}是以",
                        )
                    )
                )
                critical = _critical_lexical_terms(phrase)
                critical_hits = sum(
                    _normalize_exact_support_text(term) in normalized_searchable for term in critical
                )
                critical_ratio = critical_hits / len(critical) if critical else 0.0
                qualifies = exact or recall >= 0.72 or (recall >= 0.58 and critical_ratio >= 0.75)
                if not qualifies:
                    continue
                score = (
                    recall
                    + (0.35 if exact else 0.0)
                    + (1.0 if definition else 0.0)
                    + critical_ratio * 0.2
                )
                if best is None or score > best[0]:
                    best = (score, chunk, document_chunks)
        if best is None:
            continue
        score, chunk, document_chunks = best
        if chunk.chunk_id in seen_ids:
            continue
        # Existing direct matches do not need to be duplicated, unless the
        # bounded pass found an exact clause that should be promoted ahead of
        # a low-ranked vector result.
        seen_ids.add(chunk.chunk_id)
        supplement = _direct_exact_support_match(chunk, document_chunks, aspect, phrase, score)
        supplement.citation.evidence_role = "bounded_lexical_support"
        supplement.evidence_role = "bounded_lexical_support"
        supplement.metadata["evidence_role"] = "bounded_lexical_support"
        supplement.metadata["fusion_method"] = "bounded_document_lexical_support"
        supplement.metadata["lexical_support_phrase"] = phrase
        supplement.metadata["promoted_existing_match"] = chunk.chunk_id in existing_ids
        supplements.append((score, len(normalized_phrase), supplement))
    supplements.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in supplements[:6]]
def _bounded_lexical_phrases(aspect: QueryAspect) -> list[str]:
    candidates = [
        search_query.query
        for search_query in aspect.search_queries
        if search_query.query_type not in {"document_title", "metadata_filter"}
    ]
    candidates.append(aspect.question)
    candidates.extend(
        keyword
        for keyword in aspect.keywords
        if 4 <= len(_normalize_exact_support_text(keyword)) <= 40
        and keyword not in {"定义", "概念", "含义", "说明文档", "相关规定"}
    )
    phrases: list[str] = []
    for candidate in candidates:
        text = re.sub(r"《[^》]{2,80}》", " ", str(candidate or ""))
        text = re.sub(r"^[A-DＡ-Ｄ][、.．：:]\s*", "", text.strip(), flags=re.IGNORECASE)
        spaced_terms = [
            re.sub(
                r"^(?:请|根据|依据|结合|说明|判断|概括|比较|计算|回答|指出)+",
                "",
                term.strip(" ，,：:"),
            ).strip(" ，,：:")
            for term in re.split(r"\s+", text)
            if term.strip()
        ]
        spaced_terms = [
            term
            for term in spaced_terms
            if 2 <= len(_normalize_exact_support_text(term)) <= 20
            and term not in {"说明", "判断", "比较", "计算", "回答", "指出"}
            and not term.endswith("版")
            and "示例" not in term
        ]
        if len(spaced_terms) >= 2:
            for window in (4, 3, 2):
                for start in range(0, max(0, len(spaced_terms) - window + 1)):
                    phrase = "".join(spaced_terms[start : start + window])
                    normalized_phrase = _normalize_exact_support_text(phrase)
                    if 6 <= len(normalized_phrase) <= 60:
                        phrases.append(phrase)
        for term in spaced_terms:
            normalized_term = _normalize_exact_support_text(term)
            if 6 <= len(normalized_term) <= 40:
                phrases.append(term)
        # Document-style queries can leave an unknown argument inside an
        # otherwise exact predicate, e.g. ``与哪些主体开展有效沟通``.  That
        # interrogative slot is absent from the source clause, so also retain
        # its evidence-shaped predicate tail.  The caller still constrains the
        # scan to resolved/candidate documents; this is not a corpus shortcut.
        interrogative_tail = re.search(
            r"(?:哪些|什么|何种|何类)(?:主体|对象|机构|人员|部门|条件|要求|方式|措施)?(?P<tail>[\u4e00-\u9fff]{6,40})",
            text,
        )
        if interrogative_tail:
            tail = interrogative_tail.group("tail").strip()
            if 6 <= len(_normalize_exact_support_text(tail)) <= 40:
                phrases.append(tail)
        # Threshold questions often express the unknown as ``X 相对 Y 的门槛``
        # while the source uses a definition such as ``X 是指……超过 Y Z%``.
        # Preserve both relation anchors, plus bounded suffixes of a long left
        # noun phrase, so an already-resolved sibling document can recover the
        # defining clause without knowing the missing numeric answer.
        for relation in re.finditer(
            r"(?P<left>[\u4e00-\u9fff]{4,40})相对(?P<right>[\u4e00-\u9fff]{4,30}?)(?:的)?(?:门槛|阈值|比例|限额)",
            text,
        ):
            left = relation.group("left")
            right = relation.group("right")
            phrases.extend((left, right))
            if len(left) > 8:
                phrases.extend((left[-8:], left[-6:]))
        parts = re.split(r"[；;。？?]|以及|并且|并同时|同时|分别", text)
        for part in parts:
            cleaned = re.sub(
                r"^(?:请|根据|依据|结合|说明|判断|概括|比较|计算|回答|指出)+",
                "",
                part.strip(" ，,：:"),
            )
            cleaned = re.sub(
                r"(?:是什么|有哪些|如何规定|是否正确|是否相符|请说明|请回答)$",
                "",
                cleaned,
            ).strip(" ，,：:")
            normalized = _normalize_exact_support_text(cleaned)
            if 6 <= len(normalized) <= 120:
                phrases.append(cleaned)
    return list(dict.fromkeys(phrases))[:12]


def _bounded_required_numeric_terms(aspect: QueryAspect) -> list[str]:
    texts = [aspect.question, *(query.query for query in aspect.search_queries)]
    terms: list[str] = []
    for text in texts:
        terms.extend(re.findall(r"\d+(?:\.\d+)?%", str(text or "")))
    return list(dict.fromkeys(terms))
def _critical_lexical_terms(text: str) -> list[str]:
    terms = re.findall(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:年|个月|日|万元|亿元|倍)", text)
    terms.extend(
        match.group(0)
        for match in re.finditer(
            r"[\u4e00-\u9fff]{2,12}(?:不得|应当|必须|可以|属于|包括|不低于|不高于|不超过|至少)",
            text,
        )
    )
    return list(dict.fromkeys(term for term in terms if len(_normalize_exact_support_text(term)) >= 2))
def _candidate_document_id(candidate: Any) -> str:
    direct = getattr(candidate, "document_id", None)
    if direct:
        return str(direct)
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("document_id")
        if value:
            return str(value)
    return ""


def _normalize_exact_support_text(value: str) -> str:
    without_breaks = re.sub(r"<br\s*/?>", "", str(value or ""), flags=re.IGNORECASE)
    normalized = re.sub(r"[\s\"'“”‘’=：:；;，,。()（）]+", "", without_breaks).lower()
    for source, target in (
        ("XDU EchoGuide 项目", "XDU EchoGuide"),
        ("ResumeMind 简历问答助手", "ResumeMind"),
        ("REV 密码算法项目", "REV 密码算法"),
        ("REV 密码算法项目", "REV 密码算法"),
        ("高并发电商秒杀平台", "秒杀平台"),
        ("外卖平台项目", "外卖平台"),
    ):
        normalized = normalized.replace(source, target)
    return normalized
def _exact_support_recall(expected: str, actual: str) -> float:
    expected_text = _normalize_exact_support_text(expected)
    actual_text = _normalize_exact_support_text(actual)
    if not expected_text:
        return 0.0
    if expected_text in actual_text:
        return 1.0
    if len(expected_text) == 1:
        return 1.0 if expected_text in actual_text else 0.0
    bigrams = [expected_text[index : index + 2] for index in range(len(expected_text) - 1)]
    return sum(bigram in actual_text for bigram in bigrams) / len(bigrams)


def _to_retrieval_results(matches: list[RetrievalMatch]) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    seen_evidence: set[str] = set()
    seen_text: set[str] = set()
    for match in matches:
        citation = match.citation
        text = _clean_chunk_text(citation.excerpt, citation.section_title)
        normalized_text = _normalize_for_dedupe(text)
        evidence_id = str(match.metadata.get("evidence_id") or citation.chunk_id)
        if evidence_id in seen_evidence or normalized_text in seen_text:
            continue
        seen_evidence.add(evidence_id)
        seen_text.add(normalized_text)
        rank = len(results) + 1
        results.append(
            RetrievalResult(
                chunk_id=citation.chunk_id,
                rank=rank,
                score=match.rerank_score or citation.rerank_score or match.score or citation.score,
                source_doc=citation.filename,
                section_title=citation.section_title,
                section_path=_section_path(citation),
                text=text,
                citation_label=f"[{rank}]",
                metadata={
                    "document_id": citation.document_id,
                    "page_number": citation.page_number,
                    "chunk_type": citation.chunk_type,
                    "evidence_role": match.evidence_role,
                    "vector_score": citation.score,
                    "rerank_score": citation.rerank_score,
                    "coverage_score": match.coverage_score,
                    "section_number": citation.section_number or _section_number(citation.section_title),
                    "parent_section_number": citation.parent_section_number
                    or _parent_section_number(_section_number(citation.section_title)),
                    "previous_chunk_id": citation.previous_chunk_id,
                    "next_chunk_id": citation.next_chunk_id,
                    **citation.metadata,
                    **match.metadata,
                },
            )
        )
    return results


def _validate_context_chunks(
    db: Session,
    context_chunks: list[RetrievalResult],
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    if not context_chunks:
        return (
            [],
            {
                "checked_chunks": 0,
                "valid_chunks": 0,
                "invalid_chunks": 0,
                "invalid_chunk_ids": [],
            },
        )

    chunk_ids = [chunk.chunk_id for chunk in context_chunks]
    rows = db.execute(
        select(DocumentChunk, Document.status)
        .join(Document, DocumentChunk.document_id == Document.document_id)
        .where(DocumentChunk.chunk_id.in_(chunk_ids))
    ).all()
    source_by_chunk_id = {chunk.chunk_id: (chunk, status) for chunk, status in rows}

    valid_chunks: list[RetrievalResult] = []
    invalid_chunk_ids: list[str] = []
    for result in context_chunks:
        source = source_by_chunk_id.get(result.chunk_id)
        if source is None:
            invalid_chunk_ids.append(result.chunk_id)
            continue

        source_chunk, document_status = source
        if document_status not in {"indexed", "table_indexed"}:
            invalid_chunk_ids.append(result.chunk_id)
            continue

        valid_chunks.append(result)

    _renumber_context_chunks(valid_chunks)
    return (
        valid_chunks,
        {
            "checked_chunks": len(context_chunks),
            "valid_chunks": len(valid_chunks),
            "invalid_chunks": len(invalid_chunk_ids),
            "invalid_chunk_ids": invalid_chunk_ids,
        },
    )


def _select_prompt_chunks(
    question: str,
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
    *,
    relaxed: bool = False,
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    selected: list[RetrievalResult] = []
    candidate_count = len(
        {
            chunk.chunk_id
            for item in aspect_retrievals
            for chunk in item.candidates
        }
    )

    for aspect_retrieval in aspect_retrievals:
        if len(selected) >= MAX_PROMPT_CHUNKS:
            break
        shared_chunk = _best_shared_candidate(
            aspect_retrieval.candidates,
            selected,
            aspect_retrieval.aspect,
        )
        if shared_chunk is not None:
            _mark_chunk_for_aspect(shared_chunk, aspect_retrieval.aspect)
            aspect_retrieval.selected_chunk_ids.append(shared_chunk.chunk_id)
            aspect_retrieval.covered = True
            continue
        core_chunk = _best_non_duplicate_candidate(
            aspect_retrieval.candidates,
            selected,
            aspect_retrieval.aspect,
        )
        if core_chunk is None:
            aspect_retrieval.covered = False
            continue
        if not _passes_core_relevance(core_chunk, aspect_retrieval.aspect):
            # Best candidate still has essentially no relevance (rerank far
            # below the absolute bar and no lexical overlap with the question).
            # Marking the aspect uncovered makes the request refuse politely
            # instead of pushing unrelated evidence into the prompt and later
            # failing grounding validation with a raw-excerpt fallback.
            aspect_retrieval.covered = False
            aspect_retrieval.retrieval_covered = False
            aspect_retrieval.diagnostics.append(
                {
                    "query_type": "relevance_gate",
                    "search_query": aspect_retrieval.aspect.question,
                    "match_count": 1,
                    "match_status": "below_rerank_threshold",
                    "reason": "best candidate below absolute relevance bar",
                    "rerank_score": round(_prompt_score(core_chunk), 6),
                    "rerank_prompt_threshold": RERANK_PROMPT_THRESHOLD,
                    "query_count": 0,
                    "raw_candidate_count": 0,
                    "candidate_count": 0,
                    "rerank_input_count": 0,
                    "rerank_call_count": 0,
                    "reranked_count": 0,
                    "filtered_count": 0,
                    "timings_ms": {},
                    "score_range": {},
                }
            )
            continue
        core_chunk.metadata["prompt_selection_reason"] = "core"
        _mark_chunk_for_aspect(core_chunk, aspect_retrieval.aspect)
        selected.append(core_chunk)
        aspect_retrieval.selected_chunk_ids.append(core_chunk.chunk_id)
        aspect_retrieval.covered = True

    # A single planner aspect can contain several explicit conditions.  Preserve
    # at least one directly matching evidence block for each condition before
    # spending the remaining budget on generic neighbours.
    for aspect_retrieval in aspect_retrievals:
        added_for_anchors = 0
        for anchor in _aspect_anchor_phrases(aspect_retrieval.aspect):
            if len(selected) >= MAX_PROMPT_CHUNKS or added_for_anchors >= 3:
                break
            aspect_selected = [
                chunk
                for chunk in selected
                if aspect_retrieval.aspect.aspect_id
                in (chunk.metadata.get("prompt_matched_aspects") or [])
            ]
            if any(_anchor_coverage(anchor, chunk.text) >= 0.82 for chunk in aspect_selected):
                continue
            candidates = [
                chunk
                for chunk in aspect_retrieval.candidates
                if _anchor_coverage(anchor, chunk.text) >= 0.72
                and not _is_duplicate_or_redundant(chunk, selected)
            ]
            if not candidates:
                continue
            chunk = max(
                candidates,
                key=lambda item: (
                    _anchor_coverage(anchor, item.text),
                    _aspect_lexical_score(item, aspect_retrieval.aspect),
                    _prompt_score(item),
                ),
            )
            chunk.metadata["prompt_selection_reason"] = "anchor"
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)
            added_for_anchors += 1

    # A planner may deliberately keep several evidence-seeking queries inside
    # one aspect (for example, a rule plus its exception). Preserve the best
    # direct candidate for each substantive query before generic neighbours;
    # otherwise one high rerank score can crowd out another requested clause.
    for aspect_retrieval in aspect_retrievals:
        added_for_queries = 0
        for search_query in aspect_retrieval.aspect.search_queries:
            if search_query.query_type not in {"semantic_question", "document_style_statement"}:
                continue
            if len(selected) >= MAX_PROMPT_CHUNKS or added_for_queries >= 4:
                break
            candidates: list[tuple[float, int, RetrievalResult]] = []
            for chunk in aspect_retrieval.candidates:
                if not _passes_prompt_relevance_bar(chunk, aspect_retrieval.aspect):
                    continue
                if not (
                    _chunk_matches_query_aspect(chunk, aspect_retrieval.aspect)
                    or _aspect_lexical_score(chunk, aspect_retrieval.aspect) > 0
                    or _chunk_question_coverage(chunk, aspect_retrieval.aspect.question) >= MIN_EVIDENCE_COVERAGE
                ):
                    continue
                hits = chunk.metadata.get("aspect_search_query_hits") or []
                matching_hits = [
                    hit
                    for hit in hits
                    if isinstance(hit, dict) and hit.get("query") == search_query.query
                ]
                if not matching_hits:
                    continue
                best_hit = max(
                    matching_hits,
                    key=lambda hit: (float(hit.get("vector_score") or 0.0), -int(hit.get("rank") or 9999)),
                )
                candidates.append(
                    (
                        float(best_hit.get("vector_score") or 0.0),
                        -int(best_hit.get("rank") or 9999),
                        chunk,
                    )
                )
            candidates.sort(key=lambda item: (item[0], item[1], _prompt_score(item[2])), reverse=True)
            chosen = next(
                (chunk for _, _, chunk in candidates if not _is_duplicate_or_redundant(chunk, selected)),
                None,
            )
            if chosen is None:
                continue
            chosen.metadata["prompt_selection_reason"] = "query"
            _mark_chunk_for_aspect(chosen, aspect_retrieval.aspect)
            selected.append(chosen)
            aspect_retrieval.selected_chunk_ids.append(chosen.chunk_id)
            added_for_queries += 1

    for aspect_retrieval in aspect_retrievals:
        if len(selected) >= MAX_PROMPT_CHUNKS:
            break
        top_score = max((_prompt_score(chunk) for chunk in aspect_retrieval.candidates), default=0.0)
        for chunk in aspect_retrieval.candidates:
            if len(selected) >= MAX_PROMPT_CHUNKS:
                break
            if chunk.chunk_id in aspect_retrieval.selected_chunk_ids:
                continue
            if not _passes_prompt_score(chunk, top_score):
                continue
            if _is_duplicate_or_redundant(chunk, selected):
                continue
            if not (_chunk_matches_query_aspect(chunk, aspect_retrieval.aspect) or _is_structural_support(chunk, selected, question)):
                continue
            chunk.metadata["prompt_selection_reason"] = "generic"
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)

    if FORCE_MIN_CHUNKS and len(selected) < MIN_PROMPT_CHUNKS:
        for aspect_retrieval in aspect_retrievals:
            top_score = max((_prompt_score(chunk) for chunk in aspect_retrieval.candidates), default=0.0)
            for chunk in aspect_retrieval.candidates:
                if len(selected) >= min(MIN_PROMPT_CHUNKS, MAX_PROMPT_CHUNKS):
                    break
                if _is_duplicate_or_redundant(chunk, selected):
                    continue
                # The minimum is a lower-bound preference, not permission to
                # inject unrelated evidence. Apply the same score and aspect
                # relevance gate used by ordinary prompt selection.
                if not _passes_prompt_score(chunk, top_score):
                    continue
                if not (
                    _chunk_matches_query_aspect(chunk, aspect_retrieval.aspect)
                    or _is_structural_support(chunk, selected, question)
                ):
                    continue
                chunk.metadata["prompt_selection_reason"] = "forced_minimum"
                _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
                selected.append(chunk)
                aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)

    if relaxed and len(selected) < RELAXED_MIN_PROMPT_CHUNKS:
        # 兜底链降阈重试：放宽门槛，按 rerank 分数直接补足候选（相关甚至不太相关，
        # 交给 LLM 推理 + 推测标注兜底，而不是拒答）
        pool: list[tuple[RetrievalResult, AspectRetrieval]] = []
        seen_pool: set[str] = set()
        for aspect_retrieval in aspect_retrievals:
            for chunk in aspect_retrieval.candidates:
                if chunk.chunk_id in seen_pool or chunk.chunk_id in {
                    item.chunk_id for item in selected
                }:
                    continue
                seen_pool.add(chunk.chunk_id)
                pool.append((chunk, aspect_retrieval))
        pool.sort(key=lambda item: _prompt_score(item[0]), reverse=True)
        for chunk, aspect_retrieval in pool:
            if len(selected) >= min(RELAXED_MIN_PROMPT_CHUNKS, MAX_PROMPT_CHUNKS):
                break
            if _is_duplicate_or_redundant(chunk, selected):
                continue
            chunk.metadata["prompt_selection_reason"] = "relaxed_fallback"
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)

    selected = _filter_prompt_relevance(selected, query_plan)
    selected = _apply_prompt_token_budget(selected)
    selected = _sort_prompt_chunks(selected, query_plan)
    _sync_aspect_coverage_from_prompt(selected, aspect_retrievals)
    _renumber_context_chunks(selected)
    covered_aspects = {item.aspect.aspect_id for item in aspect_retrievals if item.covered}
    return selected, _prompt_selection_summary(
        candidate_count,
        selected,
        query_plan,
        aspect_retrievals,
        covered_aspects,
    )
def _filter_prompt_relevance(
    selected: list[RetrievalResult],
    query_plan: QueryPlan,
) -> list[RetrievalResult]:
    """Apply a final trust gate after all quota and neighbour passes."""

    always_keep_roles = {
        "exact_anchor_support",
        "bounded_lexical_support",
        "formula_target_support",
    }
    relevant: list[RetrievalResult] = []
    for chunk in selected:
        selection_reason = chunk.metadata.get("prompt_selection_reason")
        if selection_reason not in {"core", "generic", "forced_minimum"}:
            relevant.append(chunk)
            continue
        if chunk.metadata.get("dynamic_table_evidence") or chunk.metadata.get("evidence_role") in always_keep_roles:
            relevant.append(chunk)
            continue
        if any(_chunk_matches_query_aspect(chunk, aspect) for aspect in query_plan.aspects):
            relevant.append(chunk)
            continue
        if selection_reason == "core" and any(
            _aspect_lexical_score(chunk, aspect) > 0
            or _chunk_question_coverage(chunk, aspect.question) >= MIN_EVIDENCE_COVERAGE
            for aspect in query_plan.aspects
        ):
            relevant.append(chunk)
            continue
        if _is_structural_support(chunk, relevant, query_plan.original_question):
            relevant.append(chunk)
    return relevant


def _sync_aspect_coverage_from_prompt(
    selected: list[RetrievalResult],
    aspect_retrievals: list[AspectRetrieval],
) -> None:
    """Reconcile aspect coverage after final prompt filtering and expansion.

    Some bounded recovery passes add same-document support directly to the
    """

    final_chunk_ids = {item.chunk_id for item in selected}
    for item in aspect_retrievals:
        item.selected_chunk_ids = [
            chunk_id for chunk_id in item.selected_chunk_ids if chunk_id in final_chunk_ids
        ]
        for chunk in selected:
            if chunk.chunk_id in item.selected_chunk_ids:
                continue
            matched = chunk.metadata.get("prompt_matched_aspects")
            matched_aspects = matched if isinstance(matched, list) else []
            if item.aspect.aspect_id in matched_aspects or _prompt_chunk_covers_aspect(
                chunk,
                item.aspect,
            ):
                _mark_chunk_for_aspect(chunk, item.aspect)
                item.selected_chunk_ids.append(chunk.chunk_id)
        if item.selected_chunk_ids:
            if not item.retrieval_covered:
                item.diagnostics.append(
                    {
                        "query_type": "prompt_coverage_sync",
                        "search_query": item.aspect.question,
                        "match_count": len(item.selected_chunk_ids),
                        "match_status": "covered_by_final_prompt_context",
                        "query_count": 0,
                        "raw_candidate_count": 0,
                        "candidate_count": len(item.selected_chunk_ids),
                        "rerank_input_count": 0,
                        "rerank_call_count": 0,
                        "reranked_count": 0,
                        "filtered_count": 0,
                        "timings_ms": {},
                        "score_range": {},
                    }
                )
            item.retrieval_covered = True
        item.covered = bool(item.selected_chunk_ids)


def _prompt_chunk_covers_aspect(chunk: RetrievalResult, aspect: QueryAspect) -> bool:
    if chunk.metadata.get("dynamic_table_evidence") or chunk.metadata.get("evidence_role") in {
        "exact_anchor_support",
        "bounded_lexical_support",
        "formula_target_support",
    }:
        return _chunk_matches_query_aspect(chunk, aspect) or _aspect_lexical_score(chunk, aspect) > 0

    text = _chunk_match_text(chunk)
    if not text:
        return False
    for query in aspect.search_queries:
        if query.query_type == "document_style_statement" and _anchor_coverage(query.query, text) >= 0.62:
            return True
    if _chunk_question_coverage(chunk, aspect.question) >= max(MIN_EVIDENCE_COVERAGE, 0.5):
        return True

    terms = _prompt_aspect_coverage_terms(aspect)
    hits = [term for term in terms if term and term in text]
    if len(set(hits)) >= 3:
        return True
    if len(set(hits)) >= 2 and _has_normative_or_definition_marker(text):
        return True
    return False


def _prompt_aspect_coverage_terms(aspect: QueryAspect) -> list[str]:
    candidates = [
        aspect.question,
        aspect.evidence_need,
        aspect.expected_evidence_type,
        *aspect.keywords,
        *[query.query for query in aspect.search_queries],
    ]
    terms: list[str] = []
    stop_terms = {
        "核对",
        "说明",
        "列出",
        "判断",
        "概括",
        "要求",
        "规定",
        "主体",
        "部门",
        "岗位",
        "职责",
        "依据",
        "定义",
        "相关",
        "哪些",
        "什么",
    }
    for candidate in candidates:
        normalized = _normalize_exact_support_text(str(candidate or ""))
        if not normalized:
            continue
        for raw in re.split(r"[^\u4e00-\u9fff0-9A-Za-z.%]+", str(candidate or "")):
            term = _normalize_exact_support_text(raw)
            if 2 <= len(term) <= 18 and term not in stop_terms:
                terms.append(term)
            if len(term) >= 6:
                terms.extend(_split_compound_coverage_term(term))
        for match in re.finditer(r"[\u4e00-\u9fff]{2,12}(?:全过程|开发|部署|上线|负责|参与|荣获|获得|一等奖|奖学金|软考|设计|实现)", str(candidate or "")):
            term = _normalize_exact_support_text(match.group(0))
            if 2 <= len(term) <= 18:
                terms.append(term)
    return list(dict.fromkeys(term for term in terms if term not in stop_terms))[:24]


def _split_compound_coverage_term(term: str) -> list[str]:
    pieces: list[str] = []
    for marker in (
        "全过程",
        "开发",
        "部署",
        "上线",
        "负责",
        "参与",
        "荣获",
        "获得",
        "一等奖",
        "奖学金",
        "软考",
        "设计",
        "实现",
    ):
        if marker in term:
            pieces.append(marker)
    return pieces


def _has_normative_or_definition_marker(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "应当",
            "不得",
            "必须",
            "可以",
            "应",
            "包括",
            "是指",
            "所称",
            "具有",
            "负责",
            "暂定为",
            "频率",
            "按照",
        )
    )


def _empty_prompt_selection() -> dict[str, Any]:
    return {
        "max_prompt_chunks": MAX_PROMPT_CHUNKS,
        "min_prompt_chunks": MIN_PROMPT_CHUNKS,
        "force_min_chunks": FORCE_MIN_CHUNKS,
        "rerank_prompt_threshold": RERANK_PROMPT_THRESHOLD,
        "relative_score_ratio": RELATIVE_SCORE_RATIO,
        "candidate_prompt_chunks": 0,
        "final_prompt_chunks": 0,
        "retrieval_covered_aspects": [],
        "covered_aspects": [],
        "covered_by_retrieval_but_not_prompted": [],
        "prompt_capacity_limited": False,
        "expected_aspects": [],
        "final_prompt_chunk_ids": [],
    }


def _prompt_selection_summary(
    candidate_count: int,
    selected: list[RetrievalResult],
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
    covered_aspects: set[str],
) -> dict[str, Any]:
    summary = _empty_prompt_selection()
    retrieval_covered_aspects = [
        item.aspect.aspect_id for item in aspect_retrievals if item.retrieval_covered
    ]
    prompt_covered_aspects = [
        aspect.aspect_id for aspect in query_plan.aspects if aspect.aspect_id in covered_aspects
    ]
    covered_by_retrieval_but_not_prompted = [
        aspect_id for aspect_id in retrieval_covered_aspects if aspect_id not in prompt_covered_aspects
    ]
    summary.update(
        {
            "candidate_prompt_chunks": candidate_count,
            "final_prompt_chunks": len(selected),
            "retrieval_covered_aspects": retrieval_covered_aspects,
            "covered_aspects": prompt_covered_aspects,
            "covered_by_retrieval_but_not_prompted": covered_by_retrieval_but_not_prompted,
            "prompt_capacity_limited": bool(covered_by_retrieval_but_not_prompted),
            "expected_aspects": [
                {
                    "aspect_id": aspect.aspect_id,
                    "description": aspect.question,
                    "evidence_need": aspect.evidence_need,
                    "search_queries": _search_query_debug_list(aspect),
                    "expected_evidence_type": aspect.expected_evidence_type,
                }
                for aspect in query_plan.aspects
            ],
            "final_prompt_chunk_ids": [chunk.chunk_id for chunk in selected],
            "aspect_selected_chunk_ids": {
                item.aspect.aspect_id: item.selected_chunk_ids for item in aspect_retrievals
            },
        }
    )
    return summary


def _apply_prompt_token_budget(chunks: list[RetrievalResult]) -> list[RetrievalResult]:
    """Keep complete evidence blocks within a global token budget."""

    selected: list[RetrievalResult] = []
    used = 0
    for chunk in chunks:
        token_count = _int_or_none(chunk.metadata.get("token_count")) or max(len(chunk.text) // 2, 1)
        if selected and used + token_count > MAX_PROMPT_TOKENS:
            continue
        selected.append(chunk)
        used += token_count
    return selected


def _first_non_duplicate_candidate(
    candidates: list[RetrievalResult],
    selected: list[RetrievalResult],
) -> RetrievalResult | None:
    for chunk in candidates:
        if _is_duplicate_or_redundant(chunk, selected):
            continue
        return chunk
    return None


def _best_non_duplicate_candidate(
    candidates: list[RetrievalResult],
    selected: list[RetrievalResult],
    aspect: QueryAspect,
) -> RetrievalResult | None:
    available = [
        chunk for chunk in candidates if not _is_duplicate_or_redundant(chunk, selected)
    ]
    if not available:
        return None
    return max(
        available,
        key=lambda chunk: (
            _aspect_lexical_score(chunk, aspect),
            _prompt_score(chunk),
            -(chunk.rank or 999),
        ),
    )


def _best_shared_candidate(
    candidates: list[RetrievalResult],
    selected: list[RetrievalResult],
    aspect: QueryAspect,
) -> RetrievalResult | None:
    candidate_ids = {chunk.chunk_id for chunk in candidates}
    reusable = [chunk for chunk in selected if chunk.chunk_id in candidate_ids]
    if not reusable:
        return None
    anchors = _aspect_anchor_phrases(aspect)
    strongly_matching = [
        chunk
        for chunk in reusable
        if any(_anchor_coverage(anchor, chunk.text) >= 0.72 for anchor in anchors)
    ]
    if not strongly_matching:
        return None
    return max(
        strongly_matching,
        key=lambda chunk: (
            _aspect_lexical_score(chunk, aspect),
            _prompt_score(chunk),
            -(chunk.rank or 999),
        ),
    )


def _aspect_anchor_phrases(aspect: QueryAspect) -> list[str]:
    phrases = [
        re.sub(r"\s+", "", item)
        for item in re.findall(r"[“‘'\"]([^”’'\"]+)[”’'\"]", aspect.question)
    ]
    return [
        phrase
        for phrase in dict.fromkeys(phrases)
        if len(phrase) >= 6 and phrase not in {"相关规定", "明确规定"}
    ]


def _anchor_coverage(anchor: str, text: str) -> float:
    normalized_anchor = re.sub(r"\s+", "", str(anchor or ""))
    normalized_text = re.sub(r"\s+", "", str(text or ""))
    if not normalized_anchor or not normalized_text:
        return 0.0
    if normalized_anchor in normalized_text:
        return 1.0
    anchor_bigrams = {
        normalized_anchor[index : index + 2]
        for index in range(max(len(normalized_anchor) - 1, 0))
    }
    text_bigrams = {
        normalized_text[index : index + 2]
        for index in range(max(len(normalized_text) - 1, 0))
    }
    return len(anchor_bigrams & text_bigrams) / max(len(anchor_bigrams), 1)


def _aspect_lexical_score(chunk: RetrievalResult, aspect: QueryAspect) -> float:
    text = re.sub(r"\s+", "", chunk.text)
    anchors = _aspect_anchor_phrases(aspect)
    anchor_score = sum(
        8.0 * _anchor_coverage(anchor, text)
        + (2.0 if anchor in text and any(marker in text for marker in ("是指", "是以", "包括")) else 0.0)
        for anchor in anchors
    )
    keywords = [re.sub(r"\s+", "", keyword) for keyword in aspect.keywords if keyword]
    keyword_score = sum(1.0 for keyword in keywords if keyword in text)
    return anchor_score + keyword_score


def _mark_chunk_for_aspect(chunk: RetrievalResult, aspect: QueryAspect) -> None:
    matched = chunk.metadata.get("prompt_matched_aspects")
    matched_aspects = matched if isinstance(matched, list) else []
    if aspect.aspect_id not in matched_aspects:
        matched_aspects.append(aspect.aspect_id)
    chunk.metadata["prompt_matched_aspects"] = matched_aspects
    chunk.metadata["aspect_id"] = aspect.aspect_id
    chunk.metadata["aspect_question"] = aspect.question
    chunk.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
    chunk.metadata["expected_evidence_type"] = aspect.expected_evidence_type
    chunk.metadata["evidence_need"] = aspect.evidence_need


def _chunk_matches_query_aspect(chunk: RetrievalResult, aspect: QueryAspect) -> bool:
    if chunk.metadata.get("fusion_method") == "query_plan_rrf" and not re.search(r"[\u4e00-\u9fff]", aspect.question):
        return True
    if chunk.metadata.get("evidence_role") in {
        "exact_anchor_support",
        "bounded_lexical_support",
    }:
        return True
    if not aspect.keywords:
        return True
    text = _chunk_match_text(chunk)
    keyword_hits = sum(1 for keyword in aspect.keywords if re.sub(r"\s+", "", keyword) in text)
    if keyword_hits >= min(2, len(aspect.keywords)):
        return True
    question_text = re.sub(r"\s+", "", aspect.question)
    return bool(question_text and question_text in text)


def _passes_prompt_score(chunk: RetrievalResult, top_score: float) -> bool:
    score = _prompt_score(chunk)
    if score < max(RERANK_PROMPT_THRESHOLD, MIN_CORE_RERANK_SCORE):
        return False
    return not top_score or score >= top_score * RELATIVE_SCORE_RATIO


def _prompt_score(chunk: RetrievalResult) -> float:
    metadata_score = _float_or_none(chunk.metadata.get("rerank_score"))
    if metadata_score is not None:
        return metadata_score
    return float(chunk.score or 0.0)


def _passes_core_relevance(chunk: RetrievalResult, aspect: QueryAspect) -> bool:
    """Best-candidate core evidence must clear the absolute relevance bar.

    The generic prompt pass already enforces RERANK_PROMPT_THRESHOLD, but the
    core pass used to accept the best candidate unconditionally, so an
    off-topic question (rerank scores near zero) still entered the prompt and
    later failed grounding validation, surfacing a raw-excerpt fallback.
    The bar is max(RERANK_PROMPT_THRESHOLD, MIN_CORE_RERANK_SCORE): deployed
    configs tune RERANK_PROMPT_THRESHOLD down to ~0.01 for the old 0.001-0.03
    score distribution, while the actual bge-reranker-base scores are 0-1
    (off-topic ≈ 0.012, real matches ≥ 0.48), so a bare threshold would let
    off-topic evidence through.  Lexical overlap with the question is an
    alternative proof of relevance: chunks recovered by the exact/bounded
    support passes always satisfy it, and it keeps genuine keyword hits
    usable even when the reranker under-scores them.  Question-coverage is
    deliberately not used here: for broad Chinese questions it is unreliable
    (MIN_EVIDENCE_COVERAGE is 0 in deployed configs), and the rerank bar
    above already separates off-topic from real evidence.
    """

    return _passes_prompt_relevance_bar(chunk, aspect)


def _passes_prompt_relevance_bar(chunk: RetrievalResult, aspect: QueryAspect) -> bool:
    """Shared absolute relevance bar for the unconditional selection passes.

    Core and per-query selection must both refuse candidates below the bar;
    the query pass additionally gates on ``MIN_EVIDENCE_COVERAGE``, which is 0
    in deployed configs and would otherwise admit any candidate with a query
    hit.  See ``_passes_core_relevance`` for the threshold rationale.
    """

    if _prompt_score(chunk) >= max(RERANK_PROMPT_THRESHOLD, MIN_CORE_RERANK_SCORE):
        return True
    # 词法逃生通道收紧（2026-08-02 拒答校准）：此前仅“词法重叠 > 0”即可绕过门槛，
    # “你的银行卡密码是什么”（rerank 0.004-0.03，命中“密码”关键词）与“你爱弹吉他吗”
    # （0.06-0.33，命中“爱好”类词）均混入并产出跑题摘录。
    # 现要求：词法命中数 ≥ MIN_LEXICAL_SCORE（引号锚点命中或 ≥2 个关键词），
    # 且 rerank ≥ MIN_LEXICAL_RERANK_SCORE（近零分不认）。
    if _aspect_lexical_score(chunk, aspect) < MIN_LEXICAL_SCORE:
        return False
    return _prompt_score(chunk) >= MIN_LEXICAL_RERANK_SCORE


def _is_structural_support(chunk: RetrievalResult, selected: list[RetrievalResult], question: str) -> bool:
    if not selected:
        return False
    if _chunk_question_coverage(chunk, question) < MIN_EVIDENCE_COVERAGE:
        return False

    chunk_document_id = str(chunk.metadata.get("document_id") or "")
    chunk_section = str(chunk.metadata.get("section_number") or "")
    chunk_parent = str(chunk.metadata.get("parent_section_number") or "")
    chunk_previous = str(chunk.metadata.get("previous_chunk_id") or "")
    chunk_next = str(chunk.metadata.get("next_chunk_id") or "")

    for selected_chunk in selected:
        selected_document_id = str(selected_chunk.metadata.get("document_id") or "")
        if chunk_document_id and selected_document_id and chunk_document_id != selected_document_id:
            continue
        selected_section = str(selected_chunk.metadata.get("section_number") or "")
        selected_parent = str(selected_chunk.metadata.get("parent_section_number") or "")
        if chunk_parent and chunk_parent == selected_section:
            return True
        if selected_parent and selected_parent == chunk_section:
            return True
        same_section_family = not chunk_section or not selected_section or (
            chunk_section.split(".", maxsplit=1)[0] == selected_section.split(".", maxsplit=1)[0]
        )
        if same_section_family and (
            chunk_previous == selected_chunk.chunk_id or chunk_next == selected_chunk.chunk_id
        ):
            return True
    return False


def _chunk_question_coverage(chunk: RetrievalResult, question: str) -> float:
    return evidence_coverage(question_terms(question), _chunk_match_text(chunk))


def _is_duplicate_or_redundant(chunk: RetrievalResult, selected: list[RetrievalResult]) -> bool:
    normalized = _normalize_for_dedupe(chunk.text)
    if not normalized:
        return True
    for selected_chunk in selected:
        selected_normalized = _normalize_for_dedupe(selected_chunk.text)
        if chunk.chunk_id == selected_chunk.chunk_id:
            return True
        if normalized == selected_normalized:
            return True
        shorter, longer = sorted([normalized, selected_normalized], key=len)
        if len(shorter) >= 80 and shorter in longer:
            return True
    return False


def _sort_prompt_chunks(chunks: list[RetrievalResult], query_plan: QueryPlan) -> list[RetrievalResult]:
    aspect_order = {aspect.aspect_id: index for index, aspect in enumerate(query_plan.aspects)}

    def sort_key(chunk: RetrievalResult) -> tuple[Any, ...]:
        section_number = str(chunk.metadata.get("section_number") or "") or _section_number(chunk.section_title) or ""
        section_key = _section_sort_key(section_number)
        matched = chunk.metadata.get("prompt_matched_aspects")
        matched_ids = matched if isinstance(matched, list) else []
        aspect_index = min((aspect_order.get(str(aspect_id), 999) for aspect_id in matched_ids), default=999)
        if section_number:
            return (
                0,
                str(chunk.metadata.get("document_id") or chunk.source_doc),
                section_key,
                aspect_index,
                chunk.rank,
            )
        return (1, aspect_index, -_prompt_score(chunk), chunk.rank)

    return sorted(chunks, key=sort_key)


def _section_sort_key(section_number: str) -> tuple[int, ...]:
    if not section_number:
        return (9999,)
    return tuple(int(part) for part in section_number.split(".") if part.isdigit()) or (9999,)


def _chunk_match_text(chunk: RetrievalResult) -> str:
    return re.sub(
        r"\s+",
        "",
        "\n".join(
            part
            for part in [
                chunk.source_doc,
                chunk.section_title or "",
                " ".join(chunk.section_path),
                chunk.text,
            ]
            if part
        ),
    )
def _renumber_context_chunks(context_chunks: list[RetrievalResult]) -> None:
    for index, chunk in enumerate(context_chunks, start=1):
        chunk.rank = index
        chunk.citation_label = f"[{index}]"


def _clean_chunk_text(text: str, section_title: str | None) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    if section_title:
        cleaned = _drop_repeated_leading_title(cleaned, section_title)
    return _truncate_preserving_sentence(cleaned, CONTEXT_CHUNK_CHAR_LIMIT)


def _drop_repeated_leading_title(text: str, section_title: str) -> str:
    lines = text.splitlines()
    while lines and lines[0].strip() == section_title.strip():
        lines = lines[1:]
    return "\n".join(lines).strip() or text


def _truncate_preserving_sentence(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    excerpt = ""
    for sentence in re.findall(r".+?(?:[。！？!?；;]|\n|$)", text, flags=re.S):
        if len(excerpt) + len(sentence) > limit:
            break
        excerpt += sentence
    excerpt = excerpt.strip()
    return excerpt if excerpt else text[:limit].strip()


def _section_path(citation: Citation) -> list[str]:
    if citation.section_path:
        return citation.section_path
    return [citation.section_title] if citation.section_title else []


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _expand_neighbor_matches(
    db: Session,
    question: str,
    matches: list[RetrievalMatch],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    if not matches:
        return []

    document_ids = {match.citation.document_id for match in matches}
    chunks_by_document = _chunks_by_document(
        db,
        document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    selected: list[RetrievalMatch] = []
    selected_chunk_ids: set[str] = set()

    for match in matches:
        if match.metadata.get("dynamic_table_evidence") or match.citation.chunk_type in {"table_cell", "table_calculation"}:
            selected.append(match)
            selected_chunk_ids.add(match.citation.chunk_id)
            continue
        if match.citation.chunk_id in selected_chunk_ids:
            continue
        selected.append(match)
        selected_chunk_ids.add(match.citation.chunk_id)

        document_chunks = chunks_by_document.get(match.citation.document_id, [])
        added_neighbors = 0
        for chunk, reason in _neighbor_candidates(match.citation.chunk_id, document_chunks, question):
            if chunk.chunk_id in selected_chunk_ids:
                continue
            selected.append(_match_from_chunk(chunk, match, reason, document_chunks, question))
            selected_chunk_ids.add(chunk.chunk_id)
            added_neighbors += 1
            if added_neighbors >= 2:
                break

    return selected[: max(MAX_PROMPT_CHUNKS * 3, MAX_PROMPT_CHUNKS)]


def _chunks_by_document(
    db: Session,
    document_ids: set[str],
    document_chunk_cache: dict[str, list[_DocumentChunkSnapshot]] | None = None,
) -> dict[str, list[_DocumentChunkSnapshot]]:
    if not document_ids:
        return {}
    cache = document_chunk_cache if document_chunk_cache is not None else {}
    now = monotonic()
    missing_ids = set(document_ids).difference(cache)
    with _DOCUMENT_SNAPSHOT_CACHE_LOCK:
        for document_id in list(missing_ids):
            cached = _DOCUMENT_SNAPSHOT_CACHE.get(document_id)
            if cached is None:
                continue
            cached_at, chunks = cached
            if now - cached_at > DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS:
                _DOCUMENT_SNAPSHOT_CACHE.pop(document_id, None)
                continue
            _DOCUMENT_SNAPSHOT_CACHE.move_to_end(document_id)
            cache[document_id] = chunks
            missing_ids.remove(document_id)
    if missing_ids:
        with measure("rag.document_snapshot_fetch"):
            chunk_models = db.scalars(
                select(DocumentChunk)
                .options(
                    load_only(
                        DocumentChunk.id,
                        DocumentChunk.chunk_id,
                        DocumentChunk.document_id,
                        DocumentChunk.text,
                        DocumentChunk.embedding_text,
                        DocumentChunk.chunk_metadata,
                        DocumentChunk.index_status,
                        DocumentChunk.source_file,
                        DocumentChunk.page_number,
                        DocumentChunk.section_title,
                    )
                )
                .where(DocumentChunk.document_id.in_(missing_ids))
                .order_by(DocumentChunk.document_id.asc(), DocumentChunk.id.asc())
            ).all()
        chunks = [
            _DocumentChunkSnapshot(
                id=chunk.id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                embedding_text=chunk.embedding_text,
                chunk_metadata=chunk.chunk_metadata,
                index_status=chunk.index_status,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
            )
            for chunk in chunk_models
        ]
        for document_id in missing_ids:
            cache[document_id] = []
        for chunk in chunks:
            cache.setdefault(chunk.document_id, []).append(chunk)
        with _DOCUMENT_SNAPSHOT_CACHE_LOCK:
            for document_id in missing_ids:
                _DOCUMENT_SNAPSHOT_CACHE[document_id] = (now, cache.get(document_id, []))
                _DOCUMENT_SNAPSHOT_CACHE.move_to_end(document_id)
            while len(_DOCUMENT_SNAPSHOT_CACHE) > DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS:
                _DOCUMENT_SNAPSHOT_CACHE.popitem(last=False)
    return {document_id: cache.get(document_id, []) for document_id in document_ids}


def _neighbor_candidates(
    anchor_chunk_id: str,
    chunks: list[DocumentChunk],
    question: str,
) -> list[tuple[DocumentChunk, str]]:
    by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}
    anchor = by_chunk_id.get(anchor_chunk_id)
    if anchor is None:
        return []

    anchor_section_number = _section_number(anchor.section_title)
    candidates: list[tuple[DocumentChunk, str]] = []
    if anchor_section_number and "." not in anchor_section_number:
        child_prefix = f"{anchor_section_number}."
        candidates.extend(
            (chunk, "child_section")
            for chunk in chunks
            if (_section_number(chunk.section_title) or "").startswith(child_prefix)
        )
    if anchor_section_number and "." in anchor_section_number:
        parent_number = _parent_section_number(anchor_section_number)
        candidates.extend(
            (chunk, "parent_section")
            for chunk in chunks
            if parent_number and _section_number(chunk.section_title) == parent_number
        )

    anchor_index = chunks.index(anchor)
    if anchor_index > 0:
        candidates.append((chunks[anchor_index - 1], "previous_chunk"))
    if anchor_index + 1 < len(chunks):
        candidates.append((chunks[anchor_index + 1], "next_chunk"))

    scored = [
        (chunk, reason, _neighbor_relevance(question, chunk, reason))
        for chunk, reason in candidates
        if chunk.chunk_id != anchor_chunk_id
    ]
    scored = [item for item in scored if item[2] > 0 or item[1] in {"child_section", "parent_section"}]
    scored.sort(key=lambda item: (item[2], item[1] in {"child_section", "parent_section"}), reverse=True)

    deduped: list[tuple[DocumentChunk, str]] = []
    seen: set[str] = set()
    for chunk, reason, _score in scored:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append((chunk, reason))
    return deduped


def _neighbor_relevance(question: str, chunk: DocumentChunk, reason: str) -> float:
    text = "\n".join(part for part in [chunk.section_title or "", chunk.embedding_text or "", chunk.text] if part.strip())
    score = evidence_coverage(question_terms(question), text)
    if reason == "child_section" and _question_prefers_child_section(question):
        score += 0.2
    if reason in {"previous_chunk", "next_chunk"}:
        score -= 0.05
    return score


def _question_prefers_child_section(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question)
    # 简历场景：追问细节类问题倾向子章节（项目详情/技能分项/获奖说明）
    return any(term in normalized for term in ["详情", "细节", "具体", "分别", "分项", "详细", "哪些"])


def _match_from_chunk(
    chunk: DocumentChunk,
    anchor: RetrievalMatch,
    reason: str,
    document_chunks: list[DocumentChunk],
    question: str,
) -> RetrievalMatch:
    previous_chunk_id, next_chunk_id = _adjacent_chunk_ids(chunk, document_chunks)
    section_number = _section_number(chunk.section_title)
    parent_section_number = _parent_section_number(section_number)
    text = _clean_chunk_text(chunk.text, chunk.section_title)
    coverage = evidence_coverage(question_terms(question), f"{chunk.section_title or ''}\n{chunk.embedding_text or ''}\n{text}")
    return RetrievalMatch(
        citation=Citation(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            filename=chunk.source_file,
            section_title=chunk.section_title,
            section_path=_inferred_section_path(chunk, document_chunks),
            section_number=section_number,
            parent_section_number=parent_section_number,
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            page_number=chunk.page_number,
            excerpt=text,
            score=anchor.score,
            rerank_score=max(anchor.rerank_score - 0.05, 0.0),
            chunk_type=_chunk_type(chunk.text),
            evidence_role="expanded_context",
        ),
        score=anchor.score,
        rerank_score=max(anchor.rerank_score - 0.05, 0.0),
        coverage_score=coverage,
        evidence_role="expanded_context",
        evidence_text=text,
        metadata={
            **_chunk_metadata(chunk),
            "expansion_reason": reason,
            "expanded_from_chunk_id": anchor.citation.chunk_id,
        },
    )


def _adjacent_chunk_ids(chunk: DocumentChunk, chunks: list[DocumentChunk]) -> tuple[str | None, str | None]:
    index = chunks.index(chunk)
    previous_chunk_id = chunks[index - 1].chunk_id if index > 0 else None
    next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
    return previous_chunk_id, next_chunk_id


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    if not chunk.chunk_metadata:
        return {}
    try:
        value = json.loads(chunk.chunk_metadata)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _inferred_section_path(chunk: DocumentChunk, chunks: list[DocumentChunk]) -> list[str]:
    section_number = _section_number(chunk.section_title)
    parent_section_number = _parent_section_number(section_number)
    path: list[str] = []
    if parent_section_number:
        parent = next((item for item in chunks if _section_number(item.section_title) == parent_section_number), None)
        if parent and parent.section_title:
            path.append(parent.section_title)
    if chunk.section_title:
        path.append(chunk.section_title)
    return path


def _chunk_type(text: str) -> str:
    stripped = text.lstrip()
    table_prefixes = ("表格：", "表格摘要：", "表格行证据：", "琛ㄦ牸锛?", "琛ㄦ牸琛岃瘉鎹細")
    return "table" if stripped.startswith(table_prefixes) or "\n|" in text else "paragraph"


def _section_number(section_title: str | None) -> str | None:
    if not section_title:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[\.、\s]", section_title)
    return match.group(1) if match else None


def _parent_section_number(section_number: str | None) -> str | None:
    if not section_number or "." not in section_number:
        return None
    return section_number.rsplit(".", maxsplit=1)[0]


def _build_retrieval_summary(
    question: str,
    context_chunks: list[RetrievalResult],
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
    diagnostics: RetrievalDiagnostics,
    citation_validation: dict[str, Any],
    prompt_selection: dict[str, Any],
) -> dict[str, Any]:
    coverage_notes = [
        _covered_aspect_note(item.aspect)
        for item in aspect_retrievals
        if item.retrieval_covered
    ]
    missing_aspects = [
        _missing_aspect_note(item.aspect)
        for item in aspect_retrievals
        if not item.retrieval_covered
    ]
    covered_by_retrieval_but_not_prompted = [
        item.aspect.aspect_id
        for item in aspect_retrievals
        if item.retrieval_covered and not item.covered
    ]
    prompt_capacity_limited = bool(covered_by_retrieval_but_not_prompted)
    metadata_filters = [
        diagnostic["metadata_filter"]
        for item in aspect_retrievals
        for diagnostic in item.diagnostics
        if isinstance(diagnostic.get("metadata_filter"), dict)
    ]
    metadata_filter_no_match = any(
        diagnostic.get("match_status") == "metadata_filter_no_match"
        for item in aspect_retrievals
        for diagnostic in item.diagnostics
    )

    has_sufficient_context = bool(context_chunks) and not missing_aspects and not prompt_capacity_limited
    summary = {
        "top_k": MAX_PROMPT_CHUNKS,
        "used_chunks": len(context_chunks),
        "has_sufficient_context": has_sufficient_context,
        "aspect_count": len(query_plan.aspects),
        "retrieval_covered_aspect_count": sum(1 for item in aspect_retrievals if item.retrieval_covered),
        "prompt_covered_aspect_count": sum(1 for item in aspect_retrievals if item.covered),
        "prompt_capacity_limited": prompt_capacity_limited,
        "metadata_filters": metadata_filters,
        "metadata_filter_no_match": metadata_filter_no_match,
        "metadata_filter_no_match_reason": (
            "用户明确元数据条件未匹配到已发布文档，未执行无约束回退"
            if metadata_filter_no_match
            else None
        ),
        "covered_by_retrieval_but_not_prompted": covered_by_retrieval_but_not_prompted,
        "coverage_notes": coverage_notes,
          "missing_aspects": missing_aspects,
        "citation_validation": citation_validation,
        "prompt_filtered_count": max(prompt_selection["candidate_prompt_chunks"] - len(context_chunks), 0),
        "prompt_selection": prompt_selection,
        "query_plan": query_plan.to_debug_dict(),
        "aspect_retrievals": [_aspect_retrieval_debug(item) for item in aspect_retrievals],
        "final_prompt_chunk_ids": [chunk.chunk_id for chunk in context_chunks],
        "fusion_method": ASPECT_QUERY_FUSION_METHOD,
        "model_device": get_model_device_info().to_debug_dict(),
    }
    summary.update(diagnostics.to_summary_fields())
    return summary
def _aggregate_aspect_diagnostics(aspect_retrievals: list[AspectRetrieval]) -> RetrievalDiagnostics:
    aggregate = RetrievalDiagnostics()
    vector_min: float | None = None
    vector_max: float | None = None
    rerank_min: float | None = None
    rerank_max: float | None = None

    for aspect_retrieval in aspect_retrievals:
        for diagnostics in aspect_retrieval.diagnostics:
            aggregate.query_count += int(diagnostics.get("query_count") or 0)
            aggregate.raw_candidate_count += int(diagnostics.get("raw_candidate_count") or 0)
            aggregate.candidate_count += int(diagnostics.get("candidate_count") or 0)
            aggregate.rerank_input_count += int(diagnostics.get("rerank_input_count") or 0)
            aggregate.rerank_call_count += int(diagnostics.get("rerank_call_count") or 0)
            aggregate.reranked_count += int(diagnostics.get("reranked_count") or 0)
            aggregate.filtered_count += int(diagnostics.get("filtered_count") or 0)
            search_query = diagnostics.get("search_query")
            if isinstance(search_query, str) and search_query:
                aggregate.query_variants.append(search_query)
            for key, value in (diagnostics.get("timings_ms") or {}).items():
                if isinstance(value, (int, float)):
                    aggregate.timings_ms[key] = round(aggregate.timings_ms.get(key, 0.0) + float(value), 2)
            score_range = diagnostics.get("score_range") or {}
            vector_min = _min_optional(vector_min, _float_or_none(score_range.get("vector_min")))
            vector_max = _max_optional(vector_max, _float_or_none(score_range.get("vector_max")))
            rerank_min = _min_optional(rerank_min, _float_or_none(score_range.get("rerank_min")))
            rerank_max = _max_optional(rerank_max, _float_or_none(score_range.get("rerank_max")))

    aggregate.query_variants = list(dict.fromkeys(aggregate.query_variants))
    aggregate.score_range = {
        "vector_min": vector_min,
        "vector_max": vector_max,
        "rerank_min": rerank_min,
        "rerank_max": rerank_max,
    }
    return aggregate


def _score_range_for_candidates(candidates: list[Any], reranked: list[Any]) -> dict[str, float | None]:
    vector_scores = [float(candidate.score) for candidate in candidates]
    rerank_scores = [float(item.rerank_score) for item in reranked]
    return {
        "vector_min": round(min(vector_scores), 4) if vector_scores else None,
        "vector_max": round(max(vector_scores), 4) if vector_scores else None,
        "rerank_min": round(min(rerank_scores), 4) if rerank_scores else None,
        "rerank_max": round(max(rerank_scores), 4) if rerank_scores else None,
    }


def _score_range_for_matches(matches: list[RetrievalMatch]) -> dict[str, float | None]:
    scores = [float(match.score) for match in matches]
    rerank_scores = [float(match.rerank_score) for match in matches]
    return {
        "vector_min": round(min(scores), 4) if scores else None,
        "vector_max": round(max(scores), 4) if scores else None,
        "rerank_min": round(min(rerank_scores), 4) if rerank_scores else None,
        "rerank_max": round(max(rerank_scores), 4) if rerank_scores else None,
    }
def _aspect_retrieval_debug(item: AspectRetrieval) -> dict[str, Any]:
    return {
        **item.aspect.to_debug_dict(),
        "evidence_need": item.aspect.evidence_need,
        "retrieval_covered": item.retrieval_covered,
        "covered": item.covered,
        "missing": not item.retrieval_covered,
        "covered_by_retrieval_but_not_prompted": item.retrieval_covered and not item.covered,
        "candidate_count": len(item.candidates),
        "selected_chunk_ids": item.selected_chunk_ids,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source_doc": chunk.source_doc,
                "section_title": chunk.section_title,
                "score": chunk.score,
                "rerank_score": chunk.metadata.get("rerank_score"),
                "fusion_score": chunk.metadata.get("aspect_query_fusion_score"),
                "query_hits": chunk.metadata.get("aspect_search_query_hits", []),
                "evidence_role": chunk.metadata.get("evidence_role"),
                "selected_for_prompt": chunk.chunk_id in item.selected_chunk_ids,
            }
            for chunk in item.candidates
        ],
        "diagnostics": item.diagnostics,
    }


def _covered_aspect_note(aspect: QueryAspect) -> str:
    return f"已覆盖：{aspect.question}"


def _missing_aspect_note(aspect: QueryAspect) -> str:
    return f"未召回：{aspect.question}"


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def _format_elapsed_seconds(elapsed_ms: float | None) -> str:
    if elapsed_ms is None:
        return "0.000s"
    return f"{elapsed_ms / 1000:.3f}s"


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


def _min_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    return candidate if current is None else min(current, candidate)


def _max_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    return candidate if current is None else max(current, candidate)


def _filter_indexed_matches(db: Session, matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    if not matches:
        return []
    document_ids = {match.citation.document_id for match in matches}
    verified_ids = {
        match.citation.document_id
        for match in matches
        if bool(match.metadata.get("active_index_verified"))
    }
    unchecked_ids = document_ids.difference(verified_ids)
    indexed_ids = set(verified_ids)
    if unchecked_ids:
        indexed_ids.update(
            db.scalars(
                select(Document.document_id).where(
                    Document.document_id.in_(unchecked_ids),
                    Document.status.in_(["indexed", "table_indexed"]),
                )
            ).all()
        )
    return [match for match in matches if match.citation.document_id in indexed_ids]


def _save_qa_log(db: Session, question: str, response: QAResponse) -> None:
    log = QALog(
        question=question,
        answer=response.answer or EMPTY_ANSWER_LOG_TEXT,
        refused=response.answer_mode == "failed",
        confidence=0.0,
        citation_count=0,
    )
    db.add(log)
    db.commit()


def _log_qa_audit(db: Session, action: str, question: str, response: QAResponse) -> None:
    package = response.context_package
    used_chunks = (
        int(package.retrieval_summary.get("used_chunks") or 0)
        if package
        else 0
    )
    detail_payload = {
        "question": question,
        "answer": response.answer,
        "is_final_answer": bool(response.answer),
        "mode": response.answer_mode,
        "generation_status": response.generation_status,
        "used_chunks": used_chunks,
        "intent": response.intent,
        "evidence_sufficiency": response.evidence_sufficiency,
        "retrieval_fallback_level": response.retrieval_fallback_level,
    }
    details_payload = {**detail_payload}
    detail = json.dumps(detail_payload, ensure_ascii=False)
    status_text = "已转移" if response.answer_mode == "redirected" else "已回答"
    summary = "问答完成"
    user_message = f"{status_text}：{_compact_audit_text(question, 80)}"
    try:
        record_event(
            db,
            action,
            "question",
            None,
            detail=detail,
            severity="info",
            event_key=f"{action}:question:{uuid4().hex}",
            summary=summary,
            user_message=user_message,
            details=details_payload,
        )
    except SQLAlchemyError:
        # The answer has already been produced.  A non-critical audit write
        # must not turn that successful answer into HTTP 500.  Roll back the
        # failed transaction so the request-scoped Session remains usable;
        # normal audit records stay small enough to persist because citation
        # metadata is compacted below.
        db.rollback()


def _compact_audit_value(value: Any, max_chars: int = 500) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _compact_audit_text(value, max_chars) if isinstance(value, str) else value
    if isinstance(value, dict):
        return {
            str(key): _compact_audit_value(item, max_chars=max_chars)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, (list, tuple)):
        return [_compact_audit_value(item, max_chars=max_chars) for item in list(value)[:24]]
    return _compact_audit_text(str(value), max_chars)


def _compact_audit_text(value: str | None, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max(max_chars - head - 1, 0)
    return f"{text[:head]}…{text[-tail:]}"
