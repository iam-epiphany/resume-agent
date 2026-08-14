from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import re
from threading import RLock
from time import monotonic, perf_counter
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)

from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, load_only

from backend.app.core.config import (
    CONVERSATION_MEMORY_MAX_TURNS,
    DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS,
    DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS,
    FALLBACK_DIRECT_GENERATION_ENABLED,
    FALLBACK_LOWER_THRESHOLD_ENABLED,
    FALLBACK_REWRITE_RETRY_ENABLED,
    FINAL_CITATION_LIMIT,
    FORCE_MIN_CHUNKS,
    GENERATION_MIN_BUDGET_SECONDS,
    GROUNDING_VERIFY_ENABLED,
    KEYWORD_RECALL_LIMIT,
    MAX_PROMPT_CHUNKS,
    MAX_PROMPT_TOKENS,
    MIN_CORE_RERANK_SCORE,
    MIN_EVIDENCE_COVERAGE,
    MIN_LEXICAL_RERANK_SCORE,
    MIN_LEXICAL_SCORE,
    MIN_PROMPT_CHUNKS,
    PER_DOCUMENT_PROMPT_CAP,
    PLANNER_MIN_BUDGET_SECONDS,
    QA_HARD_BUDGET_SECONDS,
    RELAXED_MIN_PROMPT_CHUNKS,
    RELAXED_RERANK_THRESHOLD,
    RELATIVE_SCORE_RATIO,
    RERANK_BUDGET_SECONDS,
    RERANK_CANDIDATE_LIMIT,
    RERANK_TOP_K,
    RERANK_PROMPT_THRESHOLD,
    SKIP_RERANK_ENABLED,
    SKIP_RERANK_MARGIN_RATIO,
    SKIP_RERANK_MIN_KEYWORD_HITS,
    SKIP_RERANK_TOP_FUSION_MIN,
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
from backend.app.services.conversation_memory_service import record_turn, recent_turns
from backend.app.services.persona_service import get_active_persona, persona_prompt_context
from backend.app.services.context_dependency import RESUME_ANCHOR_TERMS
from backend.app.services.intent_router_service import classify_and_resolve
from backend.app.services.llm_client import (
    llm_call_count as _request_llm_call_count,
    reset_llm_call_count as _reset_llm_call_count,
)
from backend.app.services.query_planner_service import (
    QueryAspect,
    QueryPlan,
    QuerySearchQuery,
    object_name_from_filename,
    plan_query,
    rewrite_search_queries,
)
from backend.app.services.prompt_builder import RAGPromptBuilder
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
from backend.app.services import qa_cache_service
from backend.app.services.fact_ledger_service import fact_status_for_source, load_fact_ledger
from backend.app.services.rerank_service import RerankServiceError, RerankedChunk, rerank_candidates
from backend.app.services.time_budget import TimeBudget
from backend.app.services.vector_store_service import VectorSearchResult, VectorStoreError


CONTEXT_INSTRUCTION = (
    "请严格根据检索到的知识片段回答用户问题。不得编造知识库中不存在的材料依据。"
    "若依据不足，请明确说明无法根据当前知识库判断。"
)
CONTEXT_CHUNK_CHAR_LIMIT = 1200
EMPTY_ANSWER_LOG_TEXT = "[RAG_CONTEXT_PACKAGE_ONLY] 当前阶段未接入 LLM，接口仅返回检索上下文包。"
ASPECT_QUERY_FUSION_METHOD = "aspect_query_rrf_then_bge_rerank"
RRF_K = 60
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



def answer_question(
    db: Session,
    question: str,
    options: list[str] | None = None,
    include_debug: bool = False,
    session_id: str | None = None,
    progress_reporter: ProgressReporter | None = None,
    answer_preview_reporter: Callable[[QAAnswerPreview], None] | None = None,
    cancellation_checker: Callable[[], None] | None = None,
    client_ip: str | None = None,
) -> QAResponse:
    """简历面试问答主流程：意图路由 → 追问记忆 → 检索 → 单次自评生成 → 置信度分级。"""
    cleaned_question = question.strip()
    if not cleaned_question:
        return QAResponse(answer=None, answer_mode="failed", generation_status="skipped", context_package=None)
    original_question = cleaned_question

    # 请求级 LLM 调用计数清零：统计本问真实发起的 API 请求数（llm_client 层计数）
    _reset_llm_call_count()

    # 单问硬时间预算（自意图路由起算）：重排前不足跳过重排、生成前不足摘录兜底
    budget = TimeBudget(QA_HARD_BUDGET_SECONDS)

    # 当前人物（2026-08-14）：驱动提示词个性化与检索/台账隔离
    persona = get_active_persona(db)
    persona_ctx = persona_prompt_context(persona)
    persona_name = persona_ctx["persona_name"]

    # 问答答案缓存（独立问题才查；追问链依赖前文，禁止缓存）：
    # 命中直接返回完整答案（含证据上下文），零 LLM 调用、秒回
    if session_id is None:
        cached_response = qa_cache_service.lookup(db, cleaned_question, persona_id=persona.persona_id)
        if cached_response is not None:
            cached_response.llm_call_count = 0
            cached_response.cached = True
            _report_progress(
                progress_reporter,
                {
                    "stage": "cache",
                    "status": "completed",
                    "title": "命中缓存，直接返回答案",
                    "detail": "该问题已有缓存答案，跳过检索与生成。",
                    "summary": {"cached": True},
                },
            )
            _save_qa_log(db, original_question, cached_response, client_ip=client_ip)
            return cached_response

    # ① 意图分类 + 追问补全（一次 LLM 调用双职责；失败保守回退 resume_qa 原问）
    intent_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {"stage": "intent", "status": "running", "title": "正在识别问题意图", "detail": "正在判断问题类型并选择回答策略……"},
    )
    memory_turns = recent_turns(db, session_id, limit=CONVERSATION_MEMORY_MAX_TURNS) if session_id else []
    previous_turn = memory_turns[-1] if memory_turns else None
    intent_result = classify_and_resolve(
        cleaned_question,
        previous_turn,
        persona_description=persona_ctx["persona_description"],
    )
    resolved_question = cleaned_question
    if (
        intent_result.needs_context
        and intent_result.rewritten_question
        and intent_result.rewritten_question != cleaned_question
    ):
        resolved_question = intent_result.rewritten_question
    used_memory = resolved_question != cleaned_question
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
                "rewritten_question": intent_result.rewritten_question,
                "used_memory": used_memory,
            },
        },
    )

    # ② 多轮追问记忆（消解已在意图调用中完成，此处仅汇报上下文使用情况）
    memory_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {"stage": "memory", "status": "running", "title": "正在结合上下文理解追问", "detail": "正在检查追问是否依赖上一轮对话……"},
    )
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
        generated = generate_answer(
            resolved_question, [], intent=intent_result.intent, persona_name=persona_name
        )
        response = QAResponse(
            answer=generated.answer,
            answer_mode=generated.answer_mode,
            intent=intent_result.intent,
            resolved_question=(resolved_question if used_memory else None),
            generation_status=generated.generation_status,
            llm_call_count=_request_llm_call_count(),
        )
        _record_turn_if_needed(db, session_id, original_question, resolved_question, intent_result.intent, response)
        _save_qa_log(db, original_question, response, client_ip=client_ip)
        _log_qa_audit(db, "qa_answered", original_question, response, client_ip=client_ip)
        return response

    # ④ 检索与上下文（兜底链：标准 → 降阈 → 改写 → 直接生成）
    fallback_level = 0
    memory_context = (
        {
            "question": previous_turn.get("question") or "",
            "answer_excerpt": previous_turn.get("answer_excerpt") or "",
        }
        if previous_turn
        else None
    )
    query_plan, aspect_retrievals = _plan_and_retrieve(
        db,
        resolved_question,
        progress_reporter=progress_reporter,
        cancellation_checker=cancellation_checker,
        memory_context=memory_context,
        budget=budget,
        persona_id=persona.persona_id,
    )
    package = _package_from_retrieval(
        db,
        resolved_question,
        query_plan,
        aspect_retrievals,
        progress_reporter=progress_reporter,
        cancellation_checker=cancellation_checker,
        persona_name=persona_name,
    )
    grade = _evidence_grade(package.context_chunks)
    if grade == "none" and FALLBACK_DIRECT_GENERATION_ENABLED:
        # level 3：无检索直接生成（LLM 基于 persona + 会话历史推理，强制推测标注）
        fallback_level = 3
        package = _empty_context_package(resolved_question, persona_name=persona_name)
    elif grade == "weak":
        # level 1：降阈重选（复用同一 plan 与候选池，仅放宽选择门槛，不再整链重跑）
        if FALLBACK_LOWER_THRESHOLD_ENABLED:
            relaxed_package = _package_from_retrieval(
                db,
                resolved_question,
                query_plan,
                aspect_retrievals,
                relaxed=True,
                progress_reporter=progress_reporter,
                cancellation_checker=cancellation_checker,
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
                resolved_question,
                intent_result.intent,
                catalog=_document_catalog_summary(db),
                cancellation_checker=cancellation_checker,
            )
            if queries:
                rewritten_plan, rewritten_retrievals = _plan_and_retrieve(
                    db,
                    resolved_question,
                    progress_reporter=progress_reporter,
                    cancellation_checker=cancellation_checker,
                    rewritten_queries=queries,
                    budget=budget,
                    persona_id=persona.persona_id,
                )
                rewritten_package = _package_from_retrieval(
                    db,
                    resolved_question,
                    rewritten_plan,
                    rewritten_retrievals,
                    relaxed=True,
                    progress_reporter=progress_reporter,
                    cancellation_checker=cancellation_checker,
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
    # 对话上下文（上轮 Q/A 摘要）注入生成 prompt：多轮衔接（"除了这三个"类指代）不依赖模型记忆
    conversation_context = _conversation_context_text(memory_turns)
    if conversation_context and package.llm_prompt:
        package.llm_prompt = (
            f"{package.llm_prompt.rstrip()}\n\n"
            f"【对话上下文】（仅用于理解指代与衔接，回答不得照抄原文）\n{conversation_context}"
        )
    # 事实台账（与 grounding 并存）：检测"实体—属性—值"错配（张冠李戴），
    # 同时为最终上下文块标注 fact_status 供公开出处展示
    ledger_facts = load_fact_ledger(db, persona_id=persona.persona_id) if GROUNDING_VERIFY_ENABLED else None
    if ledger_facts is not None:
        for chunk in package.context_chunks:
            fact_status = fact_status_for_source(chunk.source_doc, ledger_facts)
            if fact_status:
                chunk.metadata["fact_status"] = fact_status
    generated = generate_answer(
        resolved_question,
        package.context_chunks,
        intent=intent_result.intent,
        persona_name=persona_name,
        llm_prompt=package.llm_prompt,
        cancellation_checker=cancellation_checker,
        preview_reporter=report_preview,
        no_evidence=(fallback_level == 3),
        known_entities=_known_resume_entities(db, persona_id=persona.persona_id),
        ledger_facts=ledger_facts,
        # 硬时间预算：剩余不足 LLM 生成下限 → 跳过生成摘录兜底；否则按剩余预算收紧 LLM 超时
        force_extractive=not budget.can_afford(GENERATION_MIN_BUDGET_SECONDS),
        timeout_override=max(budget.remaining(), 1.0) if budget.remaining() > 0 else None,
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

    # 真实 LLM 调用计数：由 llm_client 请求层统计（意图/规划/改写/生成所有模块的
    # 真实 API 请求数，含超时/失败后 fallback 的调用；fast path 跳过 LLM 时不计）
    llm_call_count = _request_llm_call_count()
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
        llm_call_count=llm_call_count,
    )
    _record_turn_if_needed(db, session_id, original_question, resolved_question, intent_result.intent, response)
    _save_qa_log(db, original_question, response, client_ip=client_ip)
    _log_qa_audit(db, "qa_answered", original_question, response, client_ip=client_ip)
    # 写答案缓存：仅独立问题的 answered+sufficient 答案（推测/拒答/失败不缓存）
    if (
        session_id is None
        and response.answer_mode == "answered"
        and response.evidence_sufficiency == "sufficient"
    ):
        qa_cache_service.store(db, original_question, response, persona_id=persona.persona_id)
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


def _conversation_context_text(memory_turns: list[dict]) -> str:
    """最近两轮的 Q/A 摘要文本，注入生成 prompt 帮助多轮指代衔接。"""
    lines: list[str] = []
    for turn in memory_turns[-2:]:
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer_excerpt") or "").strip()
        if question:
            lines.append(f"Q: {question}")
        if answer:
            lines.append(f"A: {answer[:200]}")
    return "\n".join(lines)


def retrieve_context_package(
    db: Session,
    question: str,
    options: list[str] | None = None,
) -> LLMContextPackage:
    cleaned_question = question.strip()
    if not cleaned_question:
        return _empty_context_package("", persona_name=persona_name)
    return build_context_package(db, cleaned_question)


def _plan_and_retrieve(
    db: Session,
    question: str,
    progress_reporter: ProgressReporter | None = None,
    cancellation_checker: Callable[[], None] | None = None,
    *,
    rewritten_queries: list[str] | None = None,
    memory_context: dict | None = None,
    budget: TimeBudget | None = None,
    persona_id: str | None = None,
) -> tuple[QueryPlan, list[AspectRetrieval]]:
    """检索规划 + 候选召回（标准与改写重试两条路径共用）。

    规划层已接入确定性问句分析（question_analyzer）：枚举/补集问句由文档清单
    确定性拆 aspect，LLM 只做查询措辞增强；会话上下文（上轮 Q/A）贯穿 planner，
    用于解析"这三个/那三个"类指代。
    """
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
    elif budget is not None and not budget.can_afford(PLANNER_MIN_BUDGET_SECONDS):
        # 硬时间预算不足：跳过规划 LLM，零 LLM 单方面计划直接检索
        query_plan = _budget_synthetic_plan(question)
    else:
        query_plan = plan_query(
            question,
            catalog=_document_catalog_entries(db, persona_id=persona_id),
            memory_context=memory_context,
            cancellation_checker=cancellation_checker,
        )
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
    aspect_retrievals = _retrieve_aspects(db, query_plan, progress_reporter=progress_reporter, budget=budget, persona_id=persona_id)
    return query_plan, aspect_retrievals


def _package_from_retrieval(
    db: Session,
    question: str,
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
    *,
    relaxed: bool = False,
    progress_reporter: ProgressReporter | None = None,
    cancellation_checker: Callable[[], None] | None = None,
    persona_name: str = "",
) -> LLMContextPackage:
    """基于已完成的检索结果构造最终上下文包（选择/摘要/prompt）。

    relaxed=True 时复用同一候选池降阈重选——兜底链第 1 级不再重新规划、
    重新检索、重新重排（旧实现整链重跑，单问多花一次 LLM 规划 + 10s+）。
    """
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
    prompt = RAGPromptBuilder(persona_name=persona_name).build(question, context_chunks)
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


def build_context_package(
    db: Session,
    question: str,
    options: list[str] | None = None,
    progress_reporter: ProgressReporter | None = None,
    cancellation_checker: Callable[[], None] | None = None,
    *,
    relaxed: bool = False,
    rewritten_queries: list[str] | None = None,
    memory_context: dict | None = None,
) -> LLMContextPackage:
    """检索规划 → 召回 → 选择 → 上下文包（保留的公开入口，供 retrieve API 等使用）。"""
    persona = get_active_persona(db)
    persona_id = persona.persona_id
    persona_name = persona_prompt_context(persona)["persona_name"]
    query_plan, aspect_retrievals = _plan_and_retrieve(
        db,
        question,
        progress_reporter=progress_reporter,
        cancellation_checker=cancellation_checker,
        rewritten_queries=rewritten_queries,
        memory_context=memory_context,
        persona_id=persona_id,
    )
    return _package_from_retrieval(
        db,
        question,
        query_plan,
        aspect_retrievals,
        relaxed=relaxed,
        progress_reporter=progress_reporter,
        cancellation_checker=cancellation_checker,
        persona_name=persona_name,
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


def _document_catalog_entries(
    db: Session, limit: int = 60, persona_id: str | None = None
) -> list[tuple[str, str, str]]:
    """运行时读取知识库文档清单 [(filename, title, material_topic)]，供 planner 结构化使用。

    动态数据（每次从 Document 表读取），非硬编码：文档增删后自动反映；
    persona_id 不为空时只列当前人物的文档（多人物隔离，2026-08-14）。
    material_topic 是简历领域类别（项目经历/技能掌握/竞赛奖项/证书资格/教育背景…），
    供枚举检索按类别确定对象文档集合（文件名 pattern 仅作旧数据兼容 fallback）。
    """
    try:
        statement = (
            select(Document)
            .where(Document.status == "indexed")
            .order_by(Document.filename.asc())
            .limit(limit)
        )
        if persona_id:
            statement = statement.where(Document.persona_id == persona_id)
        documents = db.scalars(statement).all()
    except Exception:
        return []
    return [
        (
            document.filename,
            (document.title or "").strip(),
            (document.material_topic or "").strip(),
        )
        for document in documents
    ]


def _document_catalog_summary(db: Session, limit: int = 60) -> str:
    """运行时生成知识库文档清单（文件名 + 标题），供 planner/改写 prompt 使用。

    动态数据（每次从 Document 表读取），非硬编码：文档增删后自动反映；
    仅作 LLM 检索规划的提示上下文，不参与任何分数计算。
    """
    lines: list[str] = []
    for filename, title, _material_topic in _document_catalog_entries(db, limit=limit):
        if title and title != filename:
            lines.append(f"- {filename}（{title}）")
        else:
            lines.append(f"- {filename}")
    return "\n".join(lines)


def _empty_context_package(question: str, persona_name: str = "") -> LLMContextPackage:
    prompt = RAGPromptBuilder(persona_name=persona_name).build(question, [])
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


def _indexed_document_ids(db: Session, persona_id: str | None = None) -> set[str]:
    statement = select(Document.document_id).where(
        Document.status.in_(["indexed", "table_indexed"])
    )
    if persona_id:
        statement = statement.where(Document.persona_id == persona_id)
    return set(db.scalars(statement).all())


def _anchor_document_ids(
    db: Session,
    anchor_documents: tuple[str, ...],
    persona_id: str | None = None,
) -> set[str]:
    """把 aspect 锚定的对象文档名解析为 document_id 集合（用于检索过滤）。

    仅解析已索引且属于当前人物的文档；解析不到时返回空集（调用方不设过滤，退化为全库检索）。
    """
    if not anchor_documents:
        return set()
    try:
        statement = select(Document.document_id, Document.filename).where(
            Document.status.in_(["indexed", "table_indexed"]),
            Document.filename.in_(list(anchor_documents)),
        )
        if persona_id:
            statement = statement.where(Document.persona_id == persona_id)
        rows = db.execute(statement).all()
    except Exception:
        return set()
    return {document_id for document_id, _filename in rows}


def _known_resume_entities(
    db: Session, limit: int = 60, persona_id: str | None = None
) -> list[str]:
    """从知识库文档提取简历领域已知实体（学校/项目/奖项/证书/技术栈等）。

    用于 grounding 硬事实校验：答案中出现但检索证据中缺失的已知实体被标记为
    未核实。实体来源：文档标题、material_topic、颁发机构、文件名去前缀后的对象名。
    全部动态读取 Document 表（非硬编码）；persona_id 不为空时按当前人物过滤。
    """
    try:
        statement = (
            select(Document)
            .where(Document.status == "indexed")
            .order_by(Document.filename.asc())
            .limit(limit)
        )
        if persona_id:
            statement = statement.where(Document.persona_id == persona_id)
        documents = db.scalars(statement).all()
    except Exception:
        return []
    entities: list[str] = []
    seen: set[str] = set()
    for document in documents:
        candidates = [
            (document.title or "").strip(),
            (document.issuing_authority or "").strip(),
            (document.material_topic or "").strip(),
            _object_name_from_filename(document.filename),
        ]
        for candidate in candidates:
            if not candidate or len(candidate) < 2:
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            entities.append(candidate)
    return entities


def _object_name_from_filename(filename: str) -> str:
    """从对象文档文件名提取对象名（"项目介绍_高并发电商秒杀平台.md" → "高并发电商秒杀平台"）。"""
    name = filename or ""
    for prefix in ("项目介绍_", "技能", "奖项", "荣誉", "证书"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in (".md", ".txt", ".pdf", ".docx", ".doc"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


def _keyword_recall_candidates(
    db: Session,
    terms: list[str],
    limit: int = KEYWORD_RECALL_LIMIT,
    persona_id: str | None = None,
) -> list[tuple[_DocumentChunkSnapshot, int]]:
    """关键词精确召回：小库优势——全量扫描已索引 chunk，术语子串命中计数取 top-N。

    dense 向量对列举对象/精确术语召回弱（语义查询下对象文档常排到 40 名开外），
    本路径用词面子串匹配保证对象文档的片段进入候选池，相关性最终由 rerank 裁决。
    """
    normalized_terms = [
        re.sub(r"\s+", "", term)
        for term in terms
        if term and re.sub(r"\s+", "", term)
    ]
    if not normalized_terms or limit <= 0:
        return []
    try:
        chunks_by_document = _chunks_by_document(db, _indexed_document_ids(db, persona_id=persona_id))
    except Exception as exc:
        # 关键词召回是最佳努力路径：SQLite 扫描失败不阻断 dense 检索主链
        logger.warning("keyword recall skipped: %s", exc)
        return []
    scored: list[tuple[int, _DocumentChunkSnapshot]] = []
    for chunks in chunks_by_document.values():
        for chunk in chunks:
            text = re.sub(
                r"\s+",
                "",
                "\n".join(
                    part
                    for part in (
                        chunk.source_file,
                        chunk.section_title or "",
                        chunk.embedding_text or "",
                        chunk.text or "",
                    )
                    if part
                ),
            )
            if not text:
                continue
            hits = sum(1 for term in normalized_terms if term in text)
            if hits:
                scored.append((hits, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [(chunk, hits) for hits, chunk in scored[:limit]]


def _keyword_snapshot_candidate(snapshot: _DocumentChunkSnapshot, hits: int) -> VectorSearchResult:
    """把关键词命中的 chunk 快照包装为候选（合成分数仅作排序占位，相关性由 rerank 裁决）。"""
    return VectorSearchResult(
        chunk_id=snapshot.chunk_id,
        document_id=snapshot.document_id,
        filename=snapshot.source_file,
        section_title=snapshot.section_title,
        page_number=snapshot.page_number,
        text=snapshot.text,
        embedding_text=snapshot.embedding_text or snapshot.text,
        token_count=max(len(snapshot.text or "") // 2, 1),
        score=min(0.5 * hits, 1.0),
        chunk_type="paragraph",
        section_path=None,
        section_number=None,
        parent_section_number=None,
        previous_chunk_id=None,
        next_chunk_id=None,
        metadata={"recall_path": "keyword", "keyword_hits": hits},
    )


# ---- 跳重排分级（2026-08-14）：融合分分差 / 关键词锚定 / 时间预算 → 跳过 CPU 重排 ----


def _skip_rerank_reason(
    candidates: list[Any],
    fusion_scores: dict[str, float],
) -> str:
    """重排前判断融合排序是否已决定性，返回跳过原因（"" = 不跳过）。

    信号（保守取并集，任一命中即跳过）：
    - 候选 ≤1：无需重排
    - top1 融合分 ≥ 下限 且 与 top2 分差比例 ≥ SKIP_RERANK_MARGIN_RATIO（实体明确、单对象碾压）
    - top1 候选关键词精确命中 ≥ SKIP_RERANK_MIN_KEYWORD_HITS（词面锚定强信号）
    - 词面锚定块接近榜首（融合分 ≥ top1×0.5）——口语化/专名问句的重排主要
      只是给锚定块排序，词面已给出答案方向，跳过可省 4-7s（2026-08-14）
    """
    if not SKIP_RERANK_ENABLED:
        return ""
    if len(candidates) <= 1:
        return "single_candidate"
    ranked = sorted(candidates, key=lambda candidate: fusion_scores.get(candidate.chunk_id, 0.0), reverse=True)
    top1 = fusion_scores.get(ranked[0].chunk_id, 0.0)
    top2 = fusion_scores.get(ranked[1].chunk_id, 0.0)
    if top1 < SKIP_RERANK_TOP_FUSION_MIN:
        return ""
    if top2 <= 0.0 or (top1 - top2) / top1 >= SKIP_RERANK_MARGIN_RATIO:
        return "fusion_margin"
    if _keyword_hit_count(ranked[0]) >= SKIP_RERANK_MIN_KEYWORD_HITS:
        return "keyword_anchor"
    for candidate in ranked[1:8]:
        if _keyword_hit_count(candidate) >= SKIP_RERANK_MIN_KEYWORD_HITS:
            fusion = fusion_scores.get(candidate.chunk_id, 0.0)
            if fusion >= top1 * 0.5:
                return "keyword_near_top"
    return ""


def _keyword_hit_count(candidate: Any) -> int:
    hits = (candidate.metadata or {}).get("keyword_hits", 0)
    try:
        return int(hits)
    except (TypeError, ValueError):
        return 0


def _fusion_ordered_reranked(
    candidates: list[Any],
    fusion_scores: dict[str, float],
) -> list[RerankedChunk]:
    """跳重排时的合成分数。

    词面锚定块（keyword_hits ≥ 1，含 dense 召回后补标命中数的块）排最前，
    给高置信分（0.55 + 0.15×命中数，封顶 0.95）——它们是对象文档的精确锚点；
    其余 dense 块按融合分相对 top1 归一化映射到 0.05~0.95。
    下游 prompt 选择以 rerank 分数为相关性信号（_prompt_score 读
    metadata["rerank_score"]），合成分数必须落在与真实 rerank 相同的语义区间。
    """
    anchored = sorted(
        (c for c in candidates if _keyword_hit_count(c) > 0),
        key=lambda c: (-_keyword_hit_count(c), -fusion_scores.get(c.chunk_id, 0.0)),
    )
    dense = sorted(
        (c for c in candidates if _keyword_hit_count(c) == 0),
        key=lambda c: -fusion_scores.get(c.chunk_id, 0.0),
    )
    top1_fusion = max((fusion_scores.get(c.chunk_id, 0.0) for c in candidates), default=0.0) or 1.0
    ordered: list[RerankedChunk] = []
    for candidate in [*anchored, *dense]:
        if _keyword_hit_count(candidate) > 0:
            synthetic = min(0.95, 0.55 + 0.15 * _keyword_hit_count(candidate))
        else:
            ratio = fusion_scores.get(candidate.chunk_id, 0.0) / top1_fusion
            synthetic = min(0.95, 0.05 + 0.90 * ratio)
        ordered.append(RerankedChunk(candidate=candidate, rerank_score=synthetic))
    return ordered


def _rerank_or_skip(
    *,
    rerank_question: str,
    rerank_input: list[Any],
    fusion_scores: dict[str, float],
    budget: TimeBudget | None,
    rerank_limit: int,
) -> tuple[list[RerankedChunk], str]:
    """重排/跳过分流：返回 (reranked, skip_reason)；skip_reason 为空 = 真实重排。

    跳过顺序：① 融合分分差/关键词锚定判定 ② 时间预算不足。
    真实重排异常（RerankServiceError）向上抛出，由检索不可用兜底处理。
    """
    if not rerank_input:
        return [], ""
    reason = _skip_rerank_reason(rerank_input, fusion_scores)
    if reason:
        return _fusion_ordered_reranked(rerank_input, fusion_scores), reason
    if budget is not None and not budget.can_afford(RERANK_BUDGET_SECONDS):
        return _fusion_ordered_reranked(rerank_input, fusion_scores), "time_budget"
    # 真实重排：记录融合分分差，供跳重排阈值校准（top1/top2 分差不足则无法跳过）
    ranked = sorted(fusion_scores.values(), reverse=True)
    top1 = ranked[0] if ranked else 0.0
    top2 = ranked[1] if len(ranked) > 1 else 0.0
    logger.info(
        "rerank engaged: top1_fusion=%.4f top2_fusion=%.4f margin=%.2f candidates=%d",
        top1, top2, (top1 - top2) / top1 if top1 > 0 else 0.0, len(rerank_input),
    )
    return (
        rerank_candidates(question=rerank_question, candidates=rerank_input, limit=rerank_limit),
        "",
    )


def _budget_synthetic_plan(question: str) -> QueryPlan:
    """预算不足时的零 LLM 检索计划：单 aspect，原问直接作 semantic 查询。"""
    aspect = QueryAspect(
        aspect_id="budget",
        question=question,
        search_queries=(QuerySearchQuery(question, "semantic_question"),),
        evidence_need="相关材料依据",
        keywords=(),
    )
    return QueryPlan(original_question=question, aspects=(aspect,), planner="budget")


def _retrieve_aspects(
    db: Session,
    query_plan: QueryPlan,
    progress_reporter: ProgressReporter | None = None,
    budget: TimeBudget | None = None,
    persona_id: str | None = None,
) -> list[AspectRetrieval]:
    if len(query_plan.aspects) > 1 and retrieve_citations is _DEFAULT_RETRIEVE_CITATIONS:
        # 多角度问题（枚举/补集/复合）：对象文档跨 aspect 高度重叠（5 个项目文档被
        # 5 个 aspect 各查一遍），走融合检索——全部查询一次召回、一次重排，
        # 再按 aspect 切分（省 4/5 重排计算；歧义低时进一步跳过重排）。
        # retrieve_citations 被替换（测试/自定义检索钩子）时保持逐 aspect 串行，
        # 保留 legacy hook 的既有语义。
        return _retrieve_aspects_fused(db, query_plan, progress_reporter=progress_reporter, budget=budget, persona_id=persona_id)
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
            enumerative=query_plan.enumerative,
            budget=budget,
            persona_id=persona_id,
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


def _retrieve_aspects_fused(
    db: Session,
    query_plan: QueryPlan,
    progress_reporter: ProgressReporter | None = None,
    budget: TimeBudget | None = None,
    persona_id: str | None = None,
) -> list[AspectRetrieval]:
    """多角度融合检索：全部 aspect 的查询一次 dense 召回 + 关键词精确召回 → 至多一次 BGE rerank。

    多角度问句（枚举/补集/复合）的对象文档跨 aspect 高度重叠：5 个项目文档会被
    5 个 aspect 各查一遍、各重排一遍（CPU 上每次重排 20 对 ≈ 8-9s），浪费约 4/5 计算。
    融合路径把召回与重排合并为一次，再按"查询命中"把候选切分回各 aspect，
    随后统一走配额式多样性选择。相关性信号与单 aspect 路径完全一致
    （加权 RRF + rerank；融合分差足够大时进一步跳过重排）。
    """
    aspects = query_plan.aspects
    retrieval_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "running",
            "title": "正在检索相关依据",
            "detail": f"正在按 {len(aspects)} 个问题方面融合召回知识库片段……",
            "summary": {"total_aspects": len(aspects), "fusion": "multi_aspect_fused"},
        },
    )

    # 1. 全部检索查询（跨 aspect 去重）
    seen_queries: set[str] = set()
    queries: list[str] = []
    query_metadata: list[dict[str, str]] = []
    for aspect in aspects:
        for search_query in aspect.search_queries:
            key = re.sub(r"\s+", "", search_query.query)
            if not key or key in seen_queries:
                continue
            seen_queries.add(key)
            queries.append(search_query.query)
            query_metadata.append(
                {"query_type": search_query.query_type, "rationale": search_query.rationale}
            )

    # 2. 一次 dense 召回 + 关键词精确召回（对象名锚定，保证对象文档的片段必然入池）
    diagnostics = RetrievalDiagnostics()
    try:
        candidates, query_hits_by_chunk_id, _diagnostics_by_query = collect_candidates_with_query_hits(
            queries,
            query_metadata=query_metadata,
            diagnostics=diagnostics,
        )
    except (EmbeddingServiceError, VectorStoreError) as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc
    if KEYWORD_RECALL_LIMIT > 0:
        all_keywords: list[str] = []
        for aspect in aspects:
            all_keywords.extend(_recall_terms(aspect))
        all_keywords = list(dict.fromkeys(all_keywords))
        existing_by_chunk_id = {candidate.chunk_id: candidate for candidate in candidates}
        for snapshot, hits in _keyword_recall_candidates(db, all_keywords, persona_id=persona_id):
            existing = existing_by_chunk_id.get(snapshot.chunk_id)
            if existing is not None:
                if existing.metadata is None:
                    existing.metadata = {}
                existing.metadata["keyword_hits"] = int(existing.metadata.get("keyword_hits") or 0) + int(hits)
                continue
            candidates.append(_keyword_snapshot_candidate(snapshot, hits))
            existing_by_chunk_id[snapshot.chunk_id] = candidates[-1]

    # 3. 加权 RRF 融合（与单 aspect 路径的 QUERY_TYPE_WEIGHTS/RRF_K 完全一致）
    fusion_scores: dict[str, float] = {}
    for candidate in candidates:
        for hit in query_hits_by_chunk_id.get(candidate.chunk_id, []):
            query_type = str(hit.get("query_type") or "semantic_question")
            query_weight = QUERY_TYPE_WEIGHTS.get(query_type, 1.0)
            rank = int(hit.get("rank") or 1)
            fusion_scores[candidate.chunk_id] = (
                fusion_scores.get(candidate.chunk_id, 0.0) + query_weight / (RRF_K + rank)
            )
    document_style_scores = _fused_document_style_scores(candidates, aspects)
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

    # 4. 至多一次重排（rerank 查询 = 原问题 + 全部对象文档名 + 各 aspect 问题；
    #    融合分差足够大或时间预算不足时跳过，按融合分排序直接输出）
    rerank_started_at = perf_counter()
    object_names = [
        object_name_from_filename(doc)
        for aspect in aspects
        for doc in aspect.anchor_documents
    ]
    aspect_questions = [
        aspect.question
        for aspect in aspects
        if aspect.question and aspect.question != query_plan.original_question
    ]
    rerank_question = "\n".join(
        dict.fromkeys(
            value
            for value in [query_plan.original_question, *object_names, *aspect_questions]
            if value and str(value).strip()
        )
    )
    # 重排输入：融合排序后的 dense 候选取前 RERANK_CANDIDATE_LIMIT，
    # 关键词精确召回的对象块（排序在融合末尾）无条件保留——它们是对象文档的可靠锚点，
    # 不能被截断挤出（否则"外卖平台"等对象文档会像旧实现一样从 prompt 中消失）
    dense_pool = [
        candidate
        for candidate in candidates
        if candidate.metadata.get("recall_path") != "keyword"
    ]
    keyword_pool = [
        candidate
        for candidate in candidates
        if candidate.metadata.get("recall_path") == "keyword"
    ]
    rerank_input = limit_rerank_candidates(
        dense_pool,
        preserve_order=True,
        limit=RERANK_CANDIDATE_LIMIT,
    )
    rerank_input = rerank_input + [
        candidate
        for candidate in keyword_pool
        if not any(candidate.chunk_id == item.chunk_id for item in rerank_input)
    ]
    diagnostics.rerank_input_count = len(rerank_input)
    diagnostics.candidate_count = len(candidates)
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "running",
            "title": "正在重排候选片段",
            "detail": f"融合候选 {len(rerank_input)} 个片段进入重排……",
            "summary": {"rerank_input_count": len(rerank_input), "fusion": "multi_aspect_fused"},
        },
    )
    try:
        reranked, skip_reason = _rerank_or_skip(
            rerank_question=rerank_question,
            rerank_input=rerank_input,
            fusion_scores=fusion_scores,
            budget=budget,
            rerank_limit=RERANK_TOP_K,
        )
    except RerankServiceError as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc
    diagnostics.timings_ms["rerank"] = _elapsed_ms(rerank_started_at)
    diagnostics.rerank_call_count = 0 if skip_reason else (1 if rerank_input else 0)
    diagnostics.reranked_count = len(reranked)
    # 枚举/补集问句：保留全部重排结果（不只 top-12），由选择层的对象配额做最终取舍，
    # 避免高分文档（如自我介绍/简历）挤掉对象文档（如外卖平台/EchoGuide）
    matches = matches_from_reranked(
        question=query_plan.original_question,
        reranked=reranked,
        diagnostics=diagnostics,
        limit=max(FINAL_CITATION_LIMIT, len(reranked)),
    )
    matches = _filter_indexed_matches(db, matches, persona_id=persona_id)

    # 5. 按查询命中把候选切分回各 aspect（关键词候选归属所有 aspect，选择层按块去重）
    query_texts_by_aspect = {
        aspect.aspect_id: {re.sub(r"\s+", "", sq.query) for sq in aspect.search_queries}
        for aspect in aspects
    }
    matches_by_aspect: dict[str, list[RetrievalMatch]] = {aspect.aspect_id: [] for aspect in aspects}
    for match in matches:
        hit_queries = {
            re.sub(r"\s+", "", hit.get("query") or "")
            for hit in query_hits_by_chunk_id.get(match.citation.chunk_id, [])
        }
        for aspect in aspects:
            if match.metadata.get("recall_path") == "keyword" or (
                hit_queries & query_texts_by_aspect[aspect.aspect_id]
            ):
                matches_by_aspect[aspect.aspect_id].append(match)

    aspect_retrievals: list[AspectRetrieval] = []
    for aspect in aspects:
        aspect_matches = matches_by_aspect[aspect.aspect_id]
        for match in aspect_matches:
            chunk_id = match.citation.chunk_id
            match.metadata["aspect_id"] = aspect.aspect_id
            match.metadata["aspect_question"] = aspect.question
            match.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
            match.metadata["aspect_search_query_hits"] = query_hits_by_chunk_id.get(chunk_id, [])
            match.metadata["aspect_query_fusion_score"] = round(
                fusion_scores.get(chunk_id, 0.0), 6
            )
            match.metadata["fusion_method"] = ASPECT_QUERY_FUSION_METHOD
            match.metadata["expected_evidence_type"] = aspect.expected_evidence_type
            match.metadata["evidence_need"] = aspect.evidence_need
        aspect_matches.sort(
            key=lambda match: (
                fusion_scores.get(match.citation.chunk_id, 0.0),
                match.rerank_score,
                match.score,
            ),
            reverse=True,
        )
        candidates_for_aspect = _to_retrieval_results(aspect_matches)
        valid_candidates, citation_validation = _validate_context_chunks(db, candidates_for_aspect)
        fused_diagnostics = diagnostics.to_summary_fields()
        fused_diagnostics.update(
            {
                "query_type": "multi_aspect_fused",
                "search_query": query_plan.original_question,
                "rationale": "多角度融合检索：全部查询一次召回、至多一次重排后按查询命中切分",
                "match_count": len(valid_candidates),
                "rerank_skipped": skip_reason,
            }
        )
        aspect_retrievals.append(
            AspectRetrieval(
                aspect=aspect,
                candidates=valid_candidates,
                diagnostics=[fused_diagnostics],
                citation_validation=citation_validation,
                selected_chunk_ids=[],
                retrieval_covered=bool(valid_candidates),
                covered=False,
            )
        )
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "completed" if valid_candidates else "failed",
                "title": f"方面检索完成" if valid_candidates else f"方面未找到足够依据",
                "detail": (
                    f"已为“{aspect.question}”召回 {len(valid_candidates)} 个可回溯候选片段"
                    if valid_candidates
                    else f"“{aspect.question}”暂未召回可回溯依据"
                ),
                "aspect_id": aspect.aspect_id,
                "summary": {"aspect_index": aspects.index(aspect) + 1, "total_aspects": len(aspects),
                            "candidate_count": len(valid_candidates), "aspect_question": aspect.question},
            },
        )

    diagnostics.timings_ms["total"] = _elapsed_ms(retrieval_started_at)
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "completed",
            "title": "融合检索完成",
            "detail": (
                f"重排已跳过（{skip_reason}），按融合分输出 {diagnostics.reranked_count} 个片段，切分给 {len(aspects)} 个方面"
                if skip_reason
                else f"一次重排输出 {diagnostics.reranked_count} 个片段，切分给 {len(aspects)} 个方面"
            ),
            "elapsed_ms": diagnostics.timings_ms.get("total"),
            "summary": {
                "total_aspects": len(aspects),
                "rerank_call_count": 0 if skip_reason else 1,
                "reranked_count": diagnostics.reranked_count,
                "rerank_skipped": skip_reason,
                "fusion": "multi_aspect_fused",
            },
        },
    )
    return aspect_retrievals


def _fused_document_style_scores(
    candidates: list[Any],
    aspects: tuple[QueryAspect, ...],
) -> dict[str, float]:
    """融合路径的文档风格证据分：取所有 aspect 的 document_style 语句在候选文本中的最大覆盖。"""
    statements = [
        statement.strip()
        for aspect in aspects
        for search_query in aspect.search_queries
        if search_query.query_type == "document_style_statement"
        for statement in search_query.query.splitlines()
        if statement.strip()
    ]
    if not statements:
        return {}
    scores: dict[str, float] = {}
    for candidate in candidates:
        text = "\n".join(
            part
            for part in (
                candidate.filename,
                candidate.section_title or "",
                candidate.embedding_text or "",
                candidate.text,
            )
            if part
        )
        scores[candidate.chunk_id] = max(
            evidence_coverage(question_terms(statement), text) for statement in statements
        )
    return scores


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



def _recall_terms(aspect: QueryAspect) -> list[str]:
    """关键词精确召回的术语集：aspect.keywords + 简历领域锚点命中 + 《》/引号专名。

    fast-path 规划的中文关键词是 2-12 字连跑切分（"你有哪些可以写进简历的证"），
    对全库子串扫描几乎没有召回力；补充领域锚点（证书/项目/学校/技术栈等，单一
    事实源见 context_dependency.RESUME_ANCHOR_TERMS）与书名号专名，
    保证口语化问句（"REV 项目是你一个人做的吗"）也能把对象文档召回入池。
    """
    terms: list[str] = []
    for keyword in aspect.keywords or ():
        keyword = str(keyword).strip()
        if keyword:
            terms.append(keyword)
    for anchor in RESUME_ANCHOR_TERMS:
        if anchor in aspect.question:
            terms.append(anchor)
    for quoted in re.findall(r"[《「“]([^》」”]{2,40})[》」”]", aspect.question):
        terms.append(quoted)
    return list(dict.fromkeys(terms))


def _retrieve_aspect_matches(
    db: Session,
    aspect: QueryAspect,
    progress_reporter: ProgressReporter | None = None,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
    *,
    enumerative: bool = False,
    budget: TimeBudget | None = None,
    persona_id: str | None = None,
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
    # 简历领域结构：确定性枚举/补集路径的 aspect 锚定了对象文档（如"项目介绍_高并发秒杀平台.md"），
    # 把对象文档名解析为 document_id 集合作为 Qdrant 检索过滤——列举对象不依赖 embedding 相似度，
    # 直接从对象文档召回（与关键词精确召回互补：dense 侧限定范围、应用层注入锚点）。
    metadata_filter = None
    if aspect.anchor_documents:
        anchor_ids = _anchor_document_ids(db, aspect.anchor_documents, persona_id=persona_id)
        if anchor_ids:
            metadata_filter = {"document_ids": sorted(anchor_ids)}
    try:
        candidates, query_hits_by_chunk_id, diagnostics_by_query = collect_candidates_with_query_hits(
            search_queries,
            query_metadata=query_metadata,
            diagnostics=diagnostics,
            metadata_filter=metadata_filter,
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
    # 关键词精确召回（2026-08-14 起全问题开放，不再限于枚举问句）：
    # 改写式/口语化问句（"REV 项目是你一个人做的吗"）下 dense 会把目标文档排到
    # 50 名开外，词面子串命中（项目名/证书/学校等专名）保证对象文档的片段必然入池，
    # 相关性最终由 rerank/融合分裁决。小库全量扫，CPU 开销可忽略。
    if KEYWORD_RECALL_LIMIT > 0:
        keyword_candidates = _keyword_recall_candidates(db, _recall_terms(aspect), persona_id=persona_id)
        existing_by_chunk_id = {candidate.chunk_id: candidate for candidate in candidates}
        for snapshot, hits in keyword_candidates:
            existing = existing_by_chunk_id.get(snapshot.chunk_id)
            if existing is not None:
                # dense 已召回同一块：保留 dense 身份，但标注词面命中数——
                # 供选择层逃生门槛与跳重排判定使用（证书/项目名等精确锚点）
                if existing.metadata is None:
                    existing.metadata = {}
                existing.metadata["keyword_hits"] = int(existing.metadata.get("keyword_hits") or 0) + int(hits)
                continue
            candidates.append(_keyword_snapshot_candidate(snapshot, hits))
            existing_by_chunk_id[snapshot.chunk_id] = candidates[-1]
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
    effective_rerank_limit = _effective_rerank_candidate_limit(candidates, enumerative=enumerative)
    # 关键词精确召回块（排序在融合末尾）无条件保留（与融合路径一致的 dense/keyword 双池）：
    # 口语化问句下它们是对象文档的唯一锚点，不能被 RERANK_CANDIDATE_LIMIT 截断挤出
    dense_pool = [
        candidate
        for candidate in candidates
        if (candidate.metadata or {}).get("recall_path") != "keyword"
    ]
    keyword_pool = [
        candidate
        for candidate in candidates
        if (candidate.metadata or {}).get("recall_path") == "keyword"
    ]
    rerank_input = limit_rerank_candidates(
        dense_pool,
        preserve_order=True,
        limit=effective_rerank_limit,
    )
    rerank_input = rerank_input + [
        candidate
        for candidate in keyword_pool
        if not any(candidate.chunk_id == item.chunk_id for item in rerank_input)
    ]
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
        reranked, skip_reason = _rerank_or_skip(
            rerank_question=_rerank_query_for_aspect(aspect),
            rerank_input=rerank_input,
            fusion_scores=fusion_scores,
            budget=budget,
            rerank_limit=RERANK_TOP_K,
        )
    except RerankServiceError as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc
    diagnostics.timings_ms["rerank"] = _elapsed_ms(rerank_started_at)
    diagnostics.rerank_call_count = 0 if skip_reason else (1 if rerank_input else 0)
    diagnostics.reranked_count = len(reranked)
    diagnostics.score_range = _score_range_for_candidates(candidates, reranked)
    matches = matches_from_reranked(
        question=aspect.question,
        reranked=reranked,
        diagnostics=diagnostics,
    )
    diagnostics.timings_ms["total"] = _elapsed_ms(total_started_at)
    matches = _filter_indexed_matches(db, matches, persona_id=persona_id)
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
            "rerank_skipped": skip_reason,
            **diagnostics.to_summary_fields(),
        }
    )
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "completed",
            "title": "候选重排完成",
            "detail": (
                f"方面“{aspect.question}”重排已跳过（{skip_reason}），按融合分取 {diagnostics.reranked_count} 个片段"
                if skip_reason
                else f"方面“{aspect.question}”完成 1 次融合重排，输出 {diagnostics.reranked_count} 个片段"
            ),
            "aspect_id": aspect.aspect_id,
            "elapsed_ms": diagnostics.timings_ms.get("rerank"),
            "summary": {
                "aspect_id": aspect.aspect_id,
                "rerank_call_count": diagnostics.rerank_call_count,
                "rerank_input_count": diagnostics.rerank_input_count,
                "reranked_count": diagnostics.reranked_count,
                "rerank_skipped": skip_reason,
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


def _effective_rerank_candidate_limit(candidates: list[Any], *, enumerative: bool = False) -> int:
    if enumerative:
        # 枚举问句（多对象列举）：候选截断会把部分对象的材料挤出 rerank，
        # 全量参与重排，保证每个被列举对象都有机会进入 prompt
        return len(candidates)
    # 过去这里被 20/24 的常量下限覆盖，导致环境变量 RERANK_CANDIDATE_LIMIT 在普通问题上失效。
    return min(len(candidates), RERANK_CANDIDATE_LIMIT)

def _normalize_exact_support_text(value: str) -> str:
    without_breaks = re.sub(r"<br\s*/?>", "", str(value or ""), flags=re.IGNORECASE)
    return re.sub(r"[\s\"'“”‘’=：:；;，,。()（）]+", "", without_breaks).lower()
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
                    "filename": citation.filename,
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
    """配额式多样性选择：统一以 rerank 分数为相关性信号。

    替代旧的多 pass 特判（core/anchor/query/generic/forced_minimum/relaxed）：
    1. 每个 aspect 至少保留 1 块最优候选（问题各个方面都有依据）
    2. 每个文档最多 PER_DOCUMENT_PROMPT_CAP 块（杜绝单一文档霸屏挤掉其他对象）
    3. 枚举/补集问句：对象文档优先填充配额，被排除文档降权到最后（软约束，仅对照）
    4. relaxed 兜底：同一候选池降阈重选（兜底链第 1 级不再整链重跑）
    """
    candidate_count = len(
        {
            chunk.chunk_id
            for item in aspect_retrievals
            for chunk in item.candidates
        }
    )
    # 绝对相关性门槛：标准 = max(RERANK_PROMPT_THRESHOLD, MIN_CORE_RERANK_SCORE)；
    # 枚举问句补选放宽一半（宽泛问句下 rerank 分数整体偏低，与旧行为一致）；
    # relaxed 兜底降到 RELAXED_RERANK_THRESHOLD
    if relaxed:
        bar = RELAXED_RERANK_THRESHOLD
    elif query_plan.enumerative:
        bar = max(RERANK_PROMPT_THRESHOLD, MIN_CORE_RERANK_SCORE) * 0.5
    else:
        bar = max(RERANK_PROMPT_THRESHOLD, MIN_CORE_RERANK_SCORE)
    target = MAX_PROMPT_CHUNKS if query_plan.enumerative else MIN_PROMPT_CHUNKS
    if relaxed:
        target = max(target, RELAXED_MIN_PROMPT_CHUNKS)

    excluded_docs = set(query_plan.excluded_documents)
    object_docs = {
        doc
        for item in aspect_retrievals
        for doc in item.aspect.anchor_documents
    }

    def _doc_key(chunk: RetrievalResult) -> str:
        # 统一用文档文件名作为文档身份标识（与 excluded_documents / anchor_documents 一致；
        # document_id 是内部主键，filename 才是 planner/选择层共用的领域标识）。
        # source_doc 恒为 citation.filename，是选择层可用的文件名。
        return str(chunk.metadata.get("filename") or "") or chunk.source_doc

    def _doc_priority(doc_key: str) -> int:
        # 对象文档（非排除）最高 → 其他文档 → 被排除对象文档（仅对照，最终不入选）
        if doc_key in object_docs:
            return 0 if doc_key not in excluded_docs else 2
        return 1

    def _passes_fill_bar(chunk: RetrievalResult) -> bool:
        """填充阶段门槛：绝对分数为主；词面锚定块（keyword_hits 或精确召回）允许低分逃生。"""
        score = _prompt_score(chunk)
        if score >= bar:
            return True
        if chunk.metadata.get("recall_path") == "keyword" or chunk.metadata.get("keyword_hits"):
            return score >= MIN_LEXICAL_RERANK_SCORE
        return False

    def _passes_bar(chunk: RetrievalResult, aspect: QueryAspect) -> bool:
        """aspect 覆盖阶段门槛：填充门槛 + 词法逃生通道（词面命中 ≥2 且 rerank 不低于下限）。"""
        if _passes_fill_bar(chunk):
            return True
        if _aspect_lexical_score(chunk, aspect) < MIN_LEXICAL_SCORE:
            return False
        return _prompt_score(chunk) >= MIN_LEXICAL_RERANK_SCORE

    pool: list[tuple[RetrievalResult, AspectRetrieval]] = [
        (chunk, item)
        for item in aspect_retrievals
        for chunk in item.candidates
    ]
    pool.sort(
        key=lambda pair: (
            _doc_priority(_doc_key(pair[0])),
            -_prompt_score(pair[0]),
            pair[0].rank or 999,
        )
    )

    selected: list[RetrievalResult] = []
    selected_ids: set[str] = set()
    doc_used: dict[str, int] = {}
    in_phase_one = True

    def _select(chunk: RetrievalResult, aspect: AspectRetrieval) -> None:
        nonlocal in_phase_one
        if len(selected) >= target or chunk.chunk_id in selected_ids:
            return
        # 补集问句硬排除：被排除对象文档的 chunk 绝不进入最终 prompt
        # （"除了这三个项目还有哪些"——已介绍的项目不得重复进回答依据）
        if excluded_docs and _doc_key(chunk) in excluded_docs:
            return
        if _is_duplicate_or_redundant(chunk, selected):
            return
        doc_key = _doc_key(chunk)
        if doc_used.get(doc_key, 0) >= PER_DOCUMENT_PROMPT_CAP:
            return
        chunk.metadata["prompt_selection_reason"] = (
            "relaxed_fallback"
            if relaxed
            else ("core" if in_phase_one else "generic")
        )
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        doc_used[doc_key] = doc_used.get(doc_key, 0) + 1
        # 覆盖标记：只标记该块被选中的 aspect（多 aspect 共享块由覆盖同步兜底判定）
        _mark_chunk_for_aspect(chunk, aspect.aspect)
        aspect.selected_chunk_ids.append(chunk.chunk_id)

    # 阶段 1：每个 aspect 至少 1 块最优候选——词法命中优先（引号锚点/对象名命中优先于高分泛化块），
    # 分数其次；保证问题各个方面都有依据，且锚定对象（《材料》/项目名）优先入选
    for item in aspect_retrievals:
        if len(selected) >= target:
            break
        best = max(
            (chunk for chunk in item.candidates if _passes_bar(chunk, item.aspect)),
            key=lambda chunk: (
                _aspect_lexical_score(chunk, item.aspect),
                _prompt_score(chunk),
                -(chunk.rank or 999),
            ),
            default=None,
        )
        if best is not None:
            _select(best, item)
    in_phase_one = False

    # 阶段 2：配额填充——按 (文档优先级, rerank 分) 降序，每文档最多 PER_DOCUMENT_PROMPT_CAP 块
    for chunk, item in pool:
        if len(selected) >= target:
            break
        if chunk.chunk_id in selected_ids or not _passes_fill_bar(chunk):
            continue
        _select(chunk, item)

    # 阶段 3：文档配额兜底——当所有有候选的文档都已入池仍不足目标时，放宽配额按分数补足。
    # 单文档/小知识库场景不会被多样性约束饿死证据；多文档场景阶段 2 已达标，不会触发。
    represented_docs = {_doc_key(chunk) for chunk in selected}
    available_docs = {_doc_key(chunk) for chunk, _item in pool if _passes_fill_bar(chunk)}
    if len(selected) < target and represented_docs and represented_docs == available_docs:
        for chunk, item in pool:
            if len(selected) >= target:
                break
            if chunk.chunk_id in selected_ids or not _passes_fill_bar(chunk):
                continue
            if excluded_docs and _doc_key(chunk) in excluded_docs:
                continue
            if _is_duplicate_or_redundant(chunk, selected):
                continue
            chunk.metadata["prompt_selection_reason"] = (
                "relaxed_fallback" if relaxed else "generic"
            )
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
            _mark_chunk_for_aspect(chunk, item.aspect)
            item.selected_chunk_ids.append(chunk.chunk_id)

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
    """最终信任门：拦截与问句明显无关的块（非枚举问句）。

    保留词法/结构门槛的理由：rerank 分数偏高但词面无关联的块（如"综合成绩"
    问句召回"项目经历与竞赛奖项"章节，mock 场景 rerank 0.86）必须被拦下。
    枚举/补集问句放行：被列举对象（如各项目文档的"技术栈"块）与宽泛问句无
    词法重叠，靠 rerank 分数与对象配额裁决（旧实现曾因词法门挤掉 0.61 分
    的对象块——"除了这三个还有哪些项目"案例）。
    """

    relevant: list[RetrievalResult] = []
    for chunk in selected:
        selection_reason = chunk.metadata.get("prompt_selection_reason")
        if selection_reason not in {"core", "generic", "forced_minimum"}:
            relevant.append(chunk)
            continue
        if chunk.metadata.get("dynamic_table_evidence"):
            relevant.append(chunk)
            continue
        if query_plan.enumerative:
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
            continue
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
    if chunk.metadata.get("dynamic_table_evidence"):
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
    return len(set(hits)) >= 3


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
    return list(dict.fromkeys(term for term in terms if term not in stop_terms))[:24]


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
    if not aspect.keywords:
        return True
    text = _chunk_match_text(chunk)
    keyword_hits = sum(1 for keyword in aspect.keywords if re.sub(r"\s+", "", keyword) in text)
    if keyword_hits >= min(2, len(aspect.keywords)):
        return True
    question_text = re.sub(r"\s+", "", aspect.question)
    return bool(question_text and question_text in text)


def _prompt_score(chunk: RetrievalResult) -> float:
    metadata_score = _float_or_none(chunk.metadata.get("rerank_score"))
    if metadata_score is not None:
        return metadata_score
    return float(chunk.score or 0.0)


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


def _filter_indexed_matches(
    db: Session, matches: list[RetrievalMatch], persona_id: str | None = None
) -> list[RetrievalMatch]:
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
        statement = select(Document.document_id).where(
            Document.document_id.in_(unchecked_ids),
            Document.status.in_(["indexed", "table_indexed"]),
        )
        if persona_id:
            statement = statement.where(Document.persona_id == persona_id)
        indexed_ids.update(db.scalars(statement).all())
    return [match for match in matches if match.citation.document_id in indexed_ids]


def _save_qa_log(
    db: Session, question: str, response: QAResponse, client_ip: str | None = None
) -> None:
    package = response.context_package
    used_chunks = (
        int(package.retrieval_summary.get("used_chunks") or 0)
        if package
        else 0
    )
    log = QALog(
        question=question,
        answer=response.answer or EMPTY_ANSWER_LOG_TEXT,
        refused=response.answer_mode == "failed",
        confidence=0.0,
        citation_count=used_chunks,
        answer_mode=response.answer_mode,
        evidence_sufficiency=response.evidence_sufficiency,
        fallback_level=response.retrieval_fallback_level,
        used_chunks=used_chunks,
        client_ip=client_ip,
    )
    db.add(log)
    db.commit()


def _log_qa_audit(
    db: Session,
    action: str,
    question: str,
    response: QAResponse,
    client_ip: str | None = None,
) -> None:
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
        "client_ip": client_ip,
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
