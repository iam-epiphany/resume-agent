export interface ApiErrorBody {
  detail?: string;
  error_code?: string;
  message?: string;
  details?: unknown[];
  error?: ApiError;
}

export interface HealthResponse {
  status: string;
  message: string;
  build_id: string;
}

export type DocumentStatus =
  | "uploaded"
  | "index_queued"
  | "indexing"
  | "indexed"
  | "index_failed"
  | "deleting"
  | "delete_failed"
  | "source_missing";

export type DocumentTaskStatus = "queued" | "running" | "completed" | "failed";

export type DocumentStage =
  | "queued"
  | "parsing"
  | "chunking"
  | "metadata_indexing"
  | "embedding"
  | "vector_upsert"
  | "verifying"
  | "completed"
  | "failed";

export interface RagHealthResponse {
  build_id: string;
  offline_mode: boolean;
  embedding_model_ready: boolean;
  reranker_model_ready: boolean;
  embedding_model_path: string;
  reranker_model_path: string;
  qdrant_ready: boolean;
  qdrant_collection: string;
  qdrant_collection_ready: boolean;
  sqlite_ready: boolean;
  libreoffice_ready: boolean;
  antiword_ready: boolean;
  libreoffice_version: string | null;
  antiword_version: string | null;
  index_tasks: Record<string, number>;
  qa_tasks: Record<string, number>;
  model_runtime: {
    embedding?: { loaded?: boolean; warmed?: boolean; query_cache?: Record<string, number | boolean> };
    reranker?: { loaded?: boolean; warmed?: boolean; score_cache?: Record<string, number | boolean> };
  };
  ready: boolean;
  model_device: {
    requested_device: string;
    selected_device: string;
    torch_version: string | null;
    cuda_available: boolean;
    cuda_device_count: number;
    cuda_device_name: string | null;
    cuda_total_memory_gb?: number | null;
    cuda_free_memory_gb?: number | null;
    fallback_reason: string | null;
  };
  performance?: {
    requested_mode: "auto" | "gpu" | "cpu_balanced" | "cpu_low_resource" | string;
    selected_mode: "gpu" | "cpu_balanced" | "cpu_low_resource" | string;
    requested_backend?: string;
    backend: string;
    backend_fallback_reason?: string | null;
    effective_cpu_cores: number;
    memory_limit_bytes: number | null;
    embedding_batch_size: number;
    rerank_batch_size: number;
    rerank_max_length: number;
    rerank_input_mode?: string;
    torch_num_threads: number;
    torch_num_interop_threads: number;
    omp_num_threads?: number;
    mkl_num_threads?: number;
    warmup_policy: string;
    experimental: boolean;
    warmup?: { state?: string; warmed?: boolean; warming?: boolean; elapsed_ms?: number | null; error?: string | null };
    timings?: Record<string, { count?: number; last_ms?: number; avg_ms?: number; max_ms?: number }>;
    resources?: Record<string, unknown>;
    recent_traces?: Array<Record<string, unknown>>;
  };
}

export interface DocumentUploadResponse {
  document_id: string;
  task_id: string;
  status: DocumentTaskStatus;
  stage: DocumentStage;
  filename: string;
  content_type: string | null;
  size: number;
  chunk_count: number;
  uploaded_at: string;
  metadata: DocumentMetadata;
}

export interface MetadataProvenanceEntry {
  source?: string;
  confidence?: number;
  priority?: number;
  updated_at?: string;
  related_document_id?: string;
}

export interface DocumentMetadata extends Record<string, unknown> {
  external_doc_id?: string;
  title?: string;
  issuing_authority?: string;
  publication_date?: string;
  expiration_date?: string;
  document_number?: string;
  material_topic?: string;
  source_url?: string;
  attachment_url?: string;
  source_type?: string;
  source_filename?: string;
  file_sha256?: string;
  metadata_status?: string;
  metadata_provenance?: Record<string, MetadataProvenanceEntry>;
  identity_review_status?: "unreviewed" | "confirmed";
  identity_reviewed_at?: string | null;
  identity_reviewed_snapshot_hash?: string | null;
  identity_warnings?: string[];
}

export interface DocumentMetadataPatch {
  title?: string | null;
  issuing_authority?: string | null;
  publication_date?: string | null;
  expiration_date?: string | null;
  document_number?: string | null;
  material_topic?: string | null;
  source_url?: string | null;
}

export interface DocumentMetadataUpdateResponse {
  document_id: string;
  metadata: DocumentMetadata;
  reindex_queued?: boolean;
  metadata_refreshed: boolean;
  refresh_warning: string | null;
}

export interface DocumentBatchUploadItem {
  filename: string;
  status: "accepted" | "failed" | "duplicate" | "conflict";
  document_id: string | null;
  task_id: string | null;
  stage: DocumentStage | null;
  size: number | null;
  error_message: string | null;
}

export interface DocumentBatchUploadResponse {
  batch_id: string;
  accepted_count: number;
  failed_count: number;
  items: DocumentBatchUploadItem[];
}

export interface DocumentConflictExistingDocument {
  document_id: string;
  filename: string;
  size: number;
  file_sha256: string | null;
  status: DocumentStatus;
  uploaded_at: string;
  chunk_count: number;
}

export interface DocumentUploadPreflightRequestItem {
  client_file_id: string;
  filename: string;
  size: number;
  file_sha256: string;
}

export interface DocumentUploadPreflightItem {
  client_file_id: string;
  filename: string;
  status: "ready" | "exact_duplicate" | "name_conflict" | "selection_name_conflict";
  existing_document: DocumentConflictExistingDocument | null;
  error_message: string | null;
}

export interface DocumentUploadPreflightResponse {
  items: DocumentUploadPreflightItem[];
}

export interface DocumentProcessingResponse {
  document_id: string;
  task_id: string;
  status: DocumentTaskStatus;
  stage: DocumentStage;
  completed_units: number | null;
  total_units: number | null;
  error_code: string | null;
  error_message: string | null;
  error: ApiError | null;
  retry_count: number;
  updated_at: string;
}

export interface DocumentSummary {
  document_id: string;
  filename: string;
  file_type: string;
  size: number;
  chunk_count: number;
  uploaded_at: string;
  status: DocumentStatus;
  index_version: string | null;
  index_error: string | null;
  metadata: DocumentMetadata;
}

export interface DocumentListResponse {
  documents: DocumentSummary[];
}

export interface DocumentDeleteResponse {
  document_id: string;
  deleted: boolean;
  vector_warning: string | null;
}

export interface DocumentBulkDeleteItem {
  document_id: string;
  filename: string | null;
  status: "deleted" | "not_found" | "blocked" | "failed";
  message: string | null;
}

export interface DocumentBulkDeleteResponse {
  requested_count: number;
  deleted_count: number;
  failed_count: number;
  items: DocumentBulkDeleteItem[];
}

export interface ChunkSummary {
  chunk_id: string;
  text: string;
  text_preview: string;
  chunk_type: string;
  is_truncated: boolean;
  section_title: string | null;
  page_number: number | null;
  token_count: number;
  index_status: string;
  index_version: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface DocumentDetailResponse {
  document_id: string;
  filename: string;
  file_type: string;
  size: number;
  chunk_count: number;
  uploaded_at: string;
  status: DocumentStatus;
  index_version: string | null;
  index_error: string | null;
  metadata: DocumentMetadata;
  chunks: ChunkSummary[];
  chunk_total: number;
  chunk_offset: number;
  chunk_limit: number;
}

export interface Citation {
  document_id: string;
  chunk_id: string;
  filename: string;
  source_url?: string | null;
  attachment_url?: string | null;
  source_title?: string | null;
  issuing_authority?: string | null;
  publication_date?: string | null;
  document_number?: string | null;
  version_status?: string | null;
  section_title: string | null;
  page_number: number | null;
  excerpt: string;
  score: number | null;
  rerank_score: number | null;
  chunk_type: string;
  evidence_role: string;
  metadata: Record<string, unknown>;
}

export interface RetrievalResult {
  chunk_id: string;
  rank: number;
  score: number | null;
  source_doc: string;
  section_title: string | null;
  section_path: string[];
  text: string;
  citation_label: string;
  metadata: Record<string, unknown>;
}

export interface LLMContextPackage {
  query: string;
  mode: "rag_context";
  is_final_answer: false;
  instruction: string;
  retrieval_summary: {
    top_k: number;
    used_chunks: number;
    has_sufficient_context: boolean;
    aspect_count?: number;
    retrieval_covered_aspect_count?: number;
    prompt_covered_aspect_count?: number;
    prompt_capacity_limited?: boolean;
    covered_by_retrieval_but_not_prompted?: string[];
    query_count?: number;
    raw_candidate_count?: number;
    candidate_count?: number;
    rerank_input_count?: number;
    rerank_call_count?: number;
    rerank_candidate_limit?: number;
    reranked_count?: number;
    filtered_count?: number;
    prompt_filtered_count?: number;
    timings_ms?: Record<string, number>;
    score_range?: Record<string, number | null>;
    query_variants?: string[];
    missing_aspects?: string[];
    coverage_notes?: string[];
    fusion_method?: string;
    model_device?: {
      requested_device: string;
      selected_device: string;
      torch_version: string | null;
      cuda_available: boolean;
      cuda_device_count: number;
      cuda_device_name: string | null;
      cuda_total_memory_gb?: number | null;
      cuda_free_memory_gb?: number | null;
      fallback_reason: string | null;
    };
    query_plan?: {
      original_question: string;
      planner: string;
      fallback_used: boolean;
      error: string | null;
      budget?: Record<string, unknown>;
      aspects: Array<{
        aspect_id: string;
        question: string;
        evidence_need?: string;
        modality?: "text" | "table" | "mixed" | string;
        table_task?: "lookup" | "compare" | "calculate" | "locate" | "none" | string;
        table_filters?: Record<string, unknown>;
        operation?: "max" | "min" | "difference" | "sum" | "ratio" | "none" | string;
        search_queries: QueryPlanSearchQuery[];
        expected_evidence_type: string;
        keywords: string[];
      }>;
    };
    aspect_retrievals?: Array<{
      aspect_id: string;
      question: string;
      evidence_need?: string;
      modality?: "text" | "table" | "mixed" | string;
      table_task?: "lookup" | "compare" | "calculate" | "locate" | "none" | string;
      table_filters?: Record<string, unknown>;
      operation?: "max" | "min" | "difference" | "sum" | "ratio" | "none" | string;
      search_queries: QueryPlanSearchQuery[];
      expected_evidence_type: string;
      keywords: string[];
      covered: boolean;
      retrieval_covered?: boolean;
      missing: boolean;
      covered_by_retrieval_but_not_prompted?: boolean;
      candidate_count: number;
      selected_chunk_ids: string[];
      retrieved_chunks: Array<{
        chunk_id: string;
        source_doc: string;
        section_title: string | null;
        score: number | null;
        rerank_score: unknown;
        fusion_score?: unknown;
        query_hits?: Array<Record<string, unknown>>;
        evidence_role: unknown;
        selected_for_prompt: boolean;
      }>;
      diagnostics: Array<Record<string, unknown>>;
    }>;
    final_prompt_chunk_ids?: string[];
    prompt_selection?: {
      max_prompt_chunks: number;
      min_prompt_chunks: number;
      force_min_chunks: boolean;
      rerank_prompt_threshold: number;
      relative_score_ratio: number;
      candidate_prompt_chunks: number;
      final_prompt_chunks: number;
      retrieval_covered_aspects?: string[];
      covered_aspects: string[];
      covered_by_retrieval_but_not_prompted?: string[];
      prompt_capacity_limited?: boolean;
      expected_aspects: Array<{
        aspect_id: string;
        description: string;
        evidence_need?: string;
        search_queries?: QueryPlanSearchQuery[];
        expected_evidence_type?: string;
      }>;
      final_prompt_chunk_ids?: string[];
      aspect_selected_chunk_ids?: Record<string, string[]>;
    };
    citation_validation?: {
      checked_chunks: number;
      valid_chunks: number;
      invalid_chunks: number;
      invalid_chunk_ids: string[];
    };
  };
  context_chunks: RetrievalResult[];
  llm_prompt: string;
}

export interface QueryPlanSearchQuery {
  query: string;
  query_type:
    | "semantic_question"
    | "document_style_statement"
    | "keyword_anchor"
    | "table_locator"
    | "legacy"
    | "fallback"
    | string;
  rationale: string;
}

export interface QARequest {
  question: string;
  options: string[];
  include_debug: boolean;
  session_id: string | null;
}

export interface QATaskRequest extends QARequest {
  client_request_id: string;
}

export interface QAResponse {
  answer: string | null;
  answer_mode: "answered" | "hedged" | "redirected" | "failed";
  evidence_sufficiency: "sufficient" | "partial" | "insufficient" | null;
  hedge_note: string | null;
  intent: string | null;
  resolved_question: string | null;
  retrieval_fallback_level: number;
  context_package: LLMContextPackage | null;
  degraded: boolean;
  generation_status: string;
}

export interface QAAnswerPreview {
  answer: string;
  revision: number;
}

export interface QATaskCreateResponse {
  task_id: string;
  client_request_id: string;
  status: QATaskStatus;
}

export type QATaskStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface ApiError {
  code: string;
  message: string;
  stage: string | null;
  retryable: boolean;
  request_id: string | null;
}

export type RagProgressStage = "intent" | "memory" | "rewrite" | "retrieval" | "generation";

export type RagProgressStatus = "running" | "completed" | "failed" | "skipped" | "pending";

export interface RagProgressEvent {
  stage: RagProgressStage;
  status: RagProgressStatus;
  title: string;
  detail: string;
  elapsed_ms?: number | null;
  summary?: Record<string, unknown>;
  aspect_id?: string;
}

export interface QATaskStatusResponse {
  task_id: string;
  client_request_id: string | null;
  question: string;
  options: string[];
  include_debug: boolean;
  session_id: string | null;
  status: QATaskStatus;
  progress_events: RagProgressEvent[];
  answer_preview?: QAAnswerPreview | null;
  answer: QAResponse | null;
  error: ApiError | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface AuditLogItem {
  id: number;
  action: string;
  target_type: string;
  target_id: string | null;
  detail: string;
  severity: "info" | "warning" | "error";
  event_key: string | null;
  summary: string | null;
  user_message: string | null;
  details_json: string | null;
  first_seen_at: string | null;
  last_seen_at: string | null;
  occurrence_count: number;
  resolved: boolean;
  created_at: string;
}

export interface AuditLogListResponse {
  logs: AuditLogItem[];
  limit: number;
  offset: number;
  returned: number;
}

export interface AuditArchiveSummary {
  date: string;
  filename: string;
  size: number;
  updated_at: string;
}

export interface AuditArchiveListResponse {
  archives: AuditArchiveSummary[];
}

export interface AuditArchiveDetailResponse {
  date: string;
  filename: string;
  content: string;
}

export interface AuditArchiveDeleteResponse {
  date: string;
  deleted: boolean;
}
