from pydantic import BaseModel


class AuditLogItem(BaseModel):
    id: int
    action: str
    target_type: str
    target_id: str | None = None
    detail: str
    severity: str = "info"
    event_key: str | None = None
    summary: str | None = None
    user_message: str | None = None
    details_json: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    occurrence_count: int = 1
    resolved: bool = False
    created_at: str


class AuditLogListResponse(BaseModel):
    logs: list[AuditLogItem]
    limit: int = 50
    offset: int = 0
    returned: int = 0


class AuditArchiveSummary(BaseModel):
    date: str
    filename: str
    size: int
    updated_at: str


class AuditArchiveListResponse(BaseModel):
    archives: list[AuditArchiveSummary]


class AuditArchiveDetailResponse(BaseModel):
    date: str
    filename: str
    content: str


class AuditArchiveDeleteResponse(BaseModel):
    date: str
    deleted: bool
