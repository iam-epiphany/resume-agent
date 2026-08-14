from typing import Any, Literal

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    options: list[str] = Field(default_factory=list, max_length=8)  # 兼容保留，不再参与判断
    include_debug: bool = False
    session_id: str | None = Field(
        default=None, min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )


class QATaskRequest(QARequest):
    client_request_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class QATaskCreateResponse(BaseModel):
    task_id: str
    client_request_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]


class QaPublicStatusResponse(BaseModel):
    """公开就绪状态（/api/qa/status）：ready + 中文 message + 负载分级。

    load.level 为 green/yellow/red；signals 仅含数值信号（CPU/内存占比、并发与排队数），
    不暴露模型路径、设备等内部信息。
    """

    ready: bool
    message: str
    load: dict[str, Any] | None = None


class ApiError(BaseModel):
    code: str
    message: str
    stage: str | None = None
    retryable: bool = False
    request_id: str | None = None


class RagProgressEvent(BaseModel):
    stage: Literal["intent", "memory", "rewrite", "retrieval", "generation", "cache"]
    status: Literal["pending", "running", "completed", "skipped", "failed"]
    title: str
    detail: str
    elapsed_ms: float | None = None
    summary: dict[str, Any] | None = None
    aspect_id: str | None = None


class Citation(BaseModel):
    """内部载体：检索管线内传递 chunk 来源元数据（不再出现在问答响应中）。"""

    document_id: str
    chunk_id: str
    filename: str
    source_url: str | None = None
    attachment_url: str | None = None
    source_title: str | None = None
    issuing_authority: str | None = None
    publication_date: str | None = None
    document_number: str | None = None
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    section_number: str | None = None
    parent_section_number: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    page_number: int | None = None
    excerpt: str
    score: float | None = None
    rerank_score: float | None = None
    chunk_type: str = "paragraph"
    evidence_role: str = "related_context"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    rank: int
    score: float | None = None
    source_doc: str
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    citation_label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMContextPackage(BaseModel):
    query: str
    mode: str = "rag_context"
    is_final_answer: bool = False
    instruction: str
    retrieval_summary: dict[str, Any]
    context_chunks: list[RetrievalResult]
    llm_prompt: str


class PublicCitation(BaseModel):
    """公开出处（2026-08-14）：匿名视角可见的最小来源信息。

    只含 文件名 + 章节标题 + 一小段原文摘录 + 分数 + 事实状态；
    不含内部 prompt、文档全文、内部 chunk_id / document_id。
    事实状态由事实台账标注：confirmed=该来源文件的事实已确认；
    mixed=该文件含待确认/冲突事实；None=未关联台账。
    """

    source_doc: str
    section_title: str | None = None
    excerpt: str
    score: float | None = None
    fact_status: str | None = None


class QAResponse(BaseModel):
    """简历面试问答响应：纯文本 + 置信度分级，无引用标注（出处走结构化 citations）。"""

    answer: str | None
    answer_mode: Literal["answered", "hedged", "redirected", "failed"] = "answered"
    evidence_sufficiency: Literal["sufficient", "partial", "insufficient"] | None = None
    hedge_note: str | None = None
    intent: str | None = None
    resolved_question: str | None = None
    retrieval_fallback_level: int = 0
    context_package: LLMContextPackage | None = None
    # 公开出处（2026-08-14）：匿名视角由 API 层从 context_package 构造；管理员仍见完整包
    citations: list[PublicCitation] | None = None
    degraded: bool = False
    generation_status: str = "completed"
    # 本次请求的 LLM 调用次数（意图分类/规划/改写/生成合计），供评测与观测
    llm_call_count: int = 1
    # 答案是否直接命中问答缓存（无 LLM 调用、秒回；2026-08-12）
    cached: bool = False


class QAAnswerPreview(BaseModel):
    answer: str
    revision: int = 0


class QATaskStatusResponse(BaseModel):
    task_id: str
    client_request_id: str | None = None
    question: str
    options: list[str] = Field(default_factory=list)
    include_debug: bool = False
    session_id: str | None = None
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    progress_events: list[RagProgressEvent] = Field(default_factory=list)
    answer_preview: QAAnswerPreview | None = None
    answer: QAResponse | None = None
    error: ApiError | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None
