from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import require_admin
from backend.app.schemas.audit import (
    AuditArchiveDeleteResponse,
    AuditArchiveDetailResponse,
    AuditArchiveListResponse,
    AuditArchiveSummary,
    AuditLogItem,
    AuditLogListResponse,
)
from backend.app.services.audit_service import (
    delete_audit_archive,
    list_audit_archives,
    list_audit_logs,
    read_audit_archive,
    read_audit_archive_content,
)


# 完整操作日志与归档仅管理员可见；前台（匿名）只能通过 public_router 看问答类日志。
router = APIRouter(
    prefix="/audit",
    tags=["audit"],
    dependencies=[Depends(require_admin)],
)

public_router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/logs", response_model=AuditLogListResponse)
def get_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> AuditLogListResponse:
    logs = list_audit_logs(db, limit=limit, offset=offset)
    return AuditLogListResponse(
        logs=[
            AuditLogItem(
                id=log.id,
                action=log.action,
                target_type=log.target_type,
                target_id=log.target_id,
                detail=log.detail,
                severity=log.severity or "info",
                event_key=log.event_key,
                summary=log.summary,
                user_message=log.user_message,
                details_json=log.details_json,
                first_seen_at=log.first_seen_at.isoformat() if log.first_seen_at else None,
                last_seen_at=log.last_seen_at.isoformat() if log.last_seen_at else None,
                occurrence_count=log.occurrence_count or 1,
                resolved=bool(log.resolved),
                created_at=log.created_at.isoformat(),
            )
            for log in logs
        ],
        limit=limit,
        offset=offset,
        returned=len(logs),
    )


@router.get("/archives", response_model=AuditArchiveListResponse)
def get_audit_archives() -> AuditArchiveListResponse:
    archives = list_audit_archives()
    return AuditArchiveListResponse(
        archives=[
            AuditArchiveSummary(
                date=archive.date,
                filename=archive.path.name,
                size=archive.size,
                updated_at=archive.updated_at.isoformat(),
            )
            for archive in archives
        ]
    )


@router.get("/archives/{archive_date}", response_model=AuditArchiveDetailResponse)
def get_audit_archive(archive_date: str) -> AuditArchiveDetailResponse:
    try:
        archive = read_audit_archive(archive_date)
        content = read_audit_archive_content(archive_date)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="未找到指定日期的日志归档") from exc
    return AuditArchiveDetailResponse(date=archive.date, filename=archive.path.name, content=content)


@public_router.get("/qa-logs", response_model=AuditLogListResponse)
def get_public_qa_logs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    request: Request = None,  # type: ignore[assignment]  # FastAPI 注入
    db: Session = Depends(get_db),
) -> AuditLogListResponse:
    """公开路径保留（限流豁免/兼容），但问答日志（含完整问答内容）仅管理员可见，
    匿名一律返回空列表——前端隐藏只是体验，安全边界在后端。"""
    from backend.app.core.security import is_admin_request

    if not is_admin_request(request):
        return AuditLogListResponse(logs=[], limit=limit, offset=offset, returned=0)
    logs = list_audit_logs(db, limit=limit, offset=offset, scope="qa")
    return AuditLogListResponse(
        logs=[
            AuditLogItem(
                id=log.id,
                action=log.action,
                target_type=log.target_type,
                target_id=log.target_id,
                detail=log.detail,
                severity=log.severity or "info",
                event_key=log.event_key,
                summary=log.summary,
                user_message=log.user_message,
                details_json=log.details_json,
                first_seen_at=log.first_seen_at.isoformat() if log.first_seen_at else None,
                last_seen_at=log.last_seen_at.isoformat() if log.last_seen_at else None,
                occurrence_count=log.occurrence_count or 1,
                resolved=bool(log.resolved),
                created_at=log.created_at.isoformat(),
            )
            for log in logs
        ],
        limit=limit,
        offset=offset,
        returned=len(logs),
    )


@router.delete("/archives/{archive_date}", response_model=AuditArchiveDeleteResponse)
def remove_audit_archive(archive_date: str, db: Session = Depends(get_db)) -> AuditArchiveDeleteResponse:
    try:
        delete_audit_archive(archive_date, db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="未找到指定日期的日志归档") from exc
    return AuditArchiveDeleteResponse(date=archive_date, deleted=True)
