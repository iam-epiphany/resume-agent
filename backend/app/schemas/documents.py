from typing import Any, Literal

from pydantic import BaseModel, Field
from backend.app.schemas.qa import ApiError


DocumentStatus = Literal[
    "uploaded",
    "index_queued",
    "indexing",
    "indexed",
    "index_failed",
    "deleting",
    "delete_failed",
    "source_missing",
]
DocumentTaskStatus = Literal["queued", "running", "completed", "failed"]
DocumentStage = Literal[
    "queued",
    "queued_rebuild",
    "parsing",
    "chunking",
    "metadata_indexing",
    "embedding",
    "vector_upsert",
    "verifying",
    "completed",
    "failed",
]

VersionStatus = Literal["unknown", "current", "future", "repealed", "superseded", "draft"]


class DocumentMetadataInput(BaseModel):
    external_doc_id: str | None = Field(default=None, max_length=128)
    title: str | None = Field(default=None, max_length=500)
    issuing_authority: str | None = Field(default=None, max_length=255)
    publication_date: str | None = Field(default=None, max_length=10)
    effective_date: str | None = Field(default=None, max_length=10)
    expiration_date: str | None = Field(default=None, max_length=10)
    document_number: str | None = Field(default=None, max_length=255)
    material_topic: str | None = Field(default=None, max_length=255)
    business_domain: str | None = Field(default=None, max_length=255)
    source_column: str | None = Field(default=None, max_length=255)
    source_url: str | None = Field(default=None, max_length=2000)
    attachment_url: str | None = Field(default=None, max_length=2000)
    source_type: str | None = Field(default=None, max_length=40)
    version_label: str | None = Field(default=None, max_length=120)
    version_status: VersionStatus | None = None
    supersedes_document_id: str | None = Field(default=None, max_length=128)
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadataUpdateResponse(BaseModel):
    document_id: str
    metadata: dict[str, Any]
    reindex_queued: bool = False
    metadata_refreshed: bool = False
    refresh_warning: str | None = None


class DocumentMetadataConfirmResponse(BaseModel):
    document_id: str
    metadata: dict[str, Any]
    metadata_refreshed: bool = False
    refresh_warning: str | None = None


class DocumentManifestImportItem(BaseModel):
    record_index: int
    status: Literal["updated", "not_found", "ambiguous", "invalid"]
    document_id: str | None = None
    filename: str | None = None
    message: str | None = None


class DocumentManifestImportResponse(BaseModel):
    total_count: int
    updated_count: int
    failed_count: int
    items: list[DocumentManifestImportItem]


class DocumentUrlImportRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    filename: str | None = Field(default=None, max_length=255)
    metadata: DocumentMetadataInput = Field(default_factory=DocumentMetadataInput)
    client_request_id: str | None = Field(default=None, min_length=8, max_length=128)


class DocumentUploadResponse(BaseModel):
    document_id: str
    task_id: str
    status: DocumentTaskStatus
    stage: DocumentStage
    filename: str
    content_type: str | None = None
    size: int
    chunk_count: int
    uploaded_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentConflictExistingDocument(BaseModel):
    document_id: str
    filename: str
    size: int
    file_sha256: str | None = None
    status: DocumentStatus
    uploaded_at: str
    chunk_count: int


class DocumentUploadPreflightItemRequest(BaseModel):
    client_file_id: str
    filename: str
    size: int = Field(ge=0)
    file_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")


class DocumentUploadPreflightRequest(BaseModel):
    items: list[DocumentUploadPreflightItemRequest] = Field(min_length=1)


class DocumentUploadPreflightItem(BaseModel):
    client_file_id: str
    filename: str
    status: Literal["ready", "exact_duplicate", "name_conflict", "selection_name_conflict"]
    existing_document: DocumentConflictExistingDocument | None = None
    error_message: str | None = None


class DocumentUploadPreflightResponse(BaseModel):
    items: list[DocumentUploadPreflightItem]


class DocumentBatchUploadItem(BaseModel):
    filename: str
    status: Literal["accepted", "failed", "duplicate", "conflict"]
    document_id: str | None = None
    task_id: str | None = None
    stage: DocumentStage | None = None
    size: int | None = None
    error_message: str | None = None


class DocumentBatchUploadResponse(BaseModel):
    batch_id: str
    accepted_count: int
    failed_count: int
    items: list[DocumentBatchUploadItem]


class DocumentProcessingResponse(BaseModel):
    document_id: str
    task_id: str
    status: DocumentTaskStatus
    stage: DocumentStage
    completed_units: int | None = None
    total_units: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    error: ApiError | None = None
    retry_count: int = 0
    updated_at: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size: int
    chunk_count: int
    uploaded_at: str
    status: DocumentStatus
    index_version: str | None = None
    index_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    vector_warning: str | None = None


class DocumentBulkDeleteRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=100)


class DocumentBulkDeleteItem(BaseModel):
    document_id: str
    filename: str | None = None
    status: Literal["deleted", "not_found", "blocked", "failed"]
    message: str | None = None


class DocumentBulkDeleteResponse(BaseModel):
    requested_count: int
    deleted_count: int
    failed_count: int
    items: list[DocumentBulkDeleteItem]


class ChunkSummary(BaseModel):
    chunk_id: str
    text: str
    text_preview: str
    chunk_type: str = "paragraph"
    is_truncated: bool = False
    section_title: str | None = None
    page_number: int | None = None
    token_count: int = 0
    index_status: str
    index_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size: int
    chunk_count: int
    uploaded_at: str
    status: DocumentStatus
    index_version: str | None = None
    index_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkSummary]
    chunk_total: int = 0
    chunk_offset: int = 0
    chunk_limit: int = 50
