from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import json
import re
from threading import Lock
from zoneinfo import ZoneInfo

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Preformatted, SimpleDocTemplate
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import AUDIT_ARCHIVE_DIR
from backend.app.models.audit import AuditLog

LOCAL_TZ = ZoneInfo("Asia/Shanghai")
AGGREGATION_WINDOW = timedelta(minutes=10)
_ARCHIVE_LOCK = Lock()
_ARCHIVE_ATTACHMENT_NAME = "audit-content.txt"
_DELETED_ARCHIVE_MARKER_NAME = ".deleted-audit-archives.json"
_PDF_FONT_NAME = "ResumeMindAuditArchiveFont"
_PDF_FALLBACK_CID_FONT_NAME = "STSong-Light"
_PDF_FONT_REGISTERED = False
_PDF_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simfang.ttf"),
    Path(r"C:\Windows\Fonts\simkai.ttf"),
    Path(r"C:\Windows\Fonts\simsunb.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    Path("/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
)


@dataclass
class AuditArchive:
    date: str
    path: Path
    size: int
    updated_at: datetime


def record_event(
    db: Session,
    action: str,
    target_type: str,
    target_id: str | None = None,
    *,
    detail: str = "",
    severity: str = "info",
    event_key: str | None = None,
    summary: str | None = None,
    user_message: str | None = None,
    details: dict | None = None,
) -> AuditLog:
    """Persist or aggregate an operator-facing audit event."""

    now = datetime.now(timezone.utc)
    safe_severity = severity if severity in {"info", "warning", "error"} else "info"
    safe_event_key = event_key or _default_event_key(action, target_type, target_id, detail)
    existing = db.scalar(
        select(AuditLog)
        .where(
            AuditLog.event_key == safe_event_key,
            AuditLog.target_type == target_type,
            AuditLog.target_id == target_id,
            AuditLog.severity == safe_severity,
            AuditLog.resolved == False,  # noqa: E712
        )
        .order_by(AuditLog.last_seen_at.desc().nullslast(), AuditLog.created_at.desc())
    )
    if existing is not None:
        last_seen = existing.last_seen_at or existing.created_at
        if last_seen is None or _ensure_aware(last_seen) >= now - AGGREGATION_WINDOW:
            existing.detail = detail or existing.detail
            existing.summary = summary or existing.summary
            existing.user_message = user_message or existing.user_message
            existing.details_json = json.dumps(details, ensure_ascii=False) if details else existing.details_json
            existing.last_seen_at = now
            existing.occurrence_count = int(existing.occurrence_count or 1) + 1
            db.commit()
            db.refresh(existing)
            return existing

    display = _display_fields(action, detail)
    log = AuditLog(
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=detail,
        severity=safe_severity,
        event_key=safe_event_key,
        summary=summary or display["summary"],
        user_message=user_message or display["user_message"],
        details_json=json.dumps(details, ensure_ascii=False) if details else None,
        first_seen_at=now,
        last_seen_at=now,
        occurrence_count=1,
        resolved=False,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def log_action(db: Session, action: str, target_type: str, target_id: str | None = None, detail: str = "") -> AuditLog:
    """Backward-compatible audit writer used by existing call sites."""

    severity = "error" if action.endswith("_failed") or "failed" in action else "warning" if "source_missing" in action else "info"
    return record_event(
        db,
        action,
        target_type,
        target_id,
        detail=detail,
        severity=severity,
        event_key=_default_event_key(action, target_type, target_id, detail),
    )


def list_audit_logs(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    scope: str = "all",
) -> list[AuditLog]:
    today = datetime.now(LOCAL_TZ).date()
    logs = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc())).all()
    today_logs = [log for log in logs if _local_date(log.created_at) == today]
    if scope == "qa":
        # 前台公开范围：仅问答类记录（target_type=question 或 qa_* 动作）
        today_logs = [
            log
            for log in today_logs
            if log.target_type == "question" or (log.action or "").startswith("qa_")
        ]
    return today_logs[max(0, offset) : max(0, offset) + limit]


def archive_expired_audit_logs(db: Session) -> None:
    with _ARCHIVE_LOCK:
        today = datetime.now(LOCAL_TZ).date()
        logs = db.scalars(select(AuditLog).order_by(AuditLog.created_at.asc())).all()
        expired_logs = [log for log in logs if _local_date(log.created_at) < today]
        if not expired_logs:
            return

        grouped: dict[str, list[AuditLog]] = {}
        for log in expired_logs:
            grouped.setdefault(_local_date(log.created_at).isoformat(), []).append(log)

        AUDIT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        for archive_date, day_logs in grouped.items():
            if _is_archive_deleted(archive_date):
                _purge_archive_files(archive_date)
                for log in day_logs:
                    db.delete(log)
                continue
            _migrate_legacy_markdown_archive(archive_date)
            path = _archive_path(archive_date)
            existing = read_audit_archive_content(archive_date) if path.exists() else ""
            archived_ids = {
                int(value)
                for value in re.findall(r"<!-- audit-id:(\d+) -->", existing)
            }
            new_logs = [log for log in day_logs if log.id not in archived_ids]
            if not new_logs:
                continue
            rendered = _render_archive_markdown(archive_date, new_logs)
            content = f"{existing.rstrip()}\n\n---\n\n{rendered}" if existing else rendered
            _write_archive_pdf(path, archive_date, content)

        for log in expired_logs:
            db.delete(log)
        db.commit()


def list_audit_archives() -> list[AuditArchive]:
    AUDIT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    _purge_deleted_archive_files()
    _migrate_legacy_markdown_archives()
    archives: list[AuditArchive] = []
    for path in AUDIT_ARCHIVE_DIR.glob("audit-*.pdf"):
        archive_date = path.stem.removeprefix("audit-")
        if _is_archive_deleted(archive_date):
            _unlink_if_exists(path)
            continue
        stat = path.stat()
        archives.append(
            AuditArchive(
                date=archive_date,
                path=path,
                size=stat.st_size,
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    archives.sort(key=lambda archive: archive.date, reverse=True)
    return archives


def read_audit_archive(archive_date: str) -> AuditArchive:
    if _is_archive_deleted(archive_date):
        _purge_archive_files(archive_date)
        raise FileNotFoundError(archive_date)
    _migrate_legacy_markdown_archive(archive_date)
    path = _archive_path(archive_date)
    if not path.exists():
        raise FileNotFoundError(archive_date)
    stat = path.stat()
    return AuditArchive(
        date=archive_date,
        path=path,
        size=stat.st_size,
        updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )


def read_audit_archive_content(archive_date: str) -> str:
    archive = read_audit_archive(archive_date)
    return _read_archive_pdf_content(archive.path)


def _read_archive_pdf_content(path: Path) -> str:
    reader = PdfReader(str(path))
    attachments = getattr(reader, "attachments", {})
    embedded = attachments.get(_ARCHIVE_ATTACHMENT_NAME)
    if embedded:
        return embedded[0].decode("utf-8")
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def delete_audit_archive(archive_date: str, db: Session | None = None) -> None:
    path = _archive_path(archive_date)
    legacy_path = _legacy_archive_path(archive_date)
    if not path.exists() and not legacy_path.exists() and not _is_archive_deleted(archive_date):
        raise FileNotFoundError(archive_date)

    _mark_archive_deleted(archive_date)
    _purge_archive_files(archive_date)
    if db is not None:
        for log in db.scalars(select(AuditLog).order_by(AuditLog.created_at.asc())).all():
            if _local_date(log.created_at).isoformat() == archive_date:
                db.delete(log)
        db.commit()


def _archive_path(archive_date: str) -> Path:
    try:
        parsed = date.fromisoformat(archive_date)
    except ValueError as exc:
        raise FileNotFoundError(archive_date) from exc
    if parsed.isoformat() != archive_date:
        raise FileNotFoundError(archive_date)
    return AUDIT_ARCHIVE_DIR / f"audit-{archive_date}.pdf"


def _legacy_archive_path(archive_date: str) -> Path:
    try:
        parsed = date.fromisoformat(archive_date)
    except ValueError as exc:
        raise FileNotFoundError(archive_date) from exc
    if parsed.isoformat() != archive_date:
        raise FileNotFoundError(archive_date)
    return AUDIT_ARCHIVE_DIR / f"audit-{archive_date}.md"


def _migrate_legacy_markdown_archives() -> None:
    for markdown_path in AUDIT_ARCHIVE_DIR.glob("audit-*.md"):
        archive_date = markdown_path.stem.removeprefix("audit-")
        try:
            _migrate_legacy_markdown_archive(archive_date)
        except FileNotFoundError:
            continue


def _migrate_legacy_markdown_archive(archive_date: str) -> None:
    markdown_path = _legacy_archive_path(archive_date)
    if not markdown_path.exists():
        return
    if _is_archive_deleted(archive_date):
        markdown_path.unlink()
        _unlink_if_exists(_archive_path(archive_date))
        return
    pdf_path = _archive_path(archive_date)
    content = markdown_path.read_text(encoding="utf-8")
    if pdf_path.exists():
        existing = _read_archive_pdf_content(pdf_path)
        existing_ids = set(re.findall(r"<!-- audit-id:(\d+) -->", existing))
        markdown_ids = set(re.findall(r"<!-- audit-id:(\d+) -->", content))
        if not markdown_ids.issubset(existing_ids):
            merged = f"{existing.rstrip()}\n\n---\n\n{content}" if existing else content
            _write_archive_pdf(pdf_path, archive_date, merged)
    else:
        _write_archive_pdf(pdf_path, archive_date, content)
    markdown_path.unlink()


def _write_archive_pdf(path: Path, archive_date: str, content: str) -> None:
    if _is_archive_deleted(archive_date):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    render_buffer = BytesIO()
    document = SimpleDocTemplate(
        render_buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"ResumeMind 审计日志归档 {archive_date}",
    )
    document.build([Preformatted(content, _archive_pdf_text_style())])
    render_buffer.seek(0)

    reader = PdfReader(render_buffer)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": f"ResumeMind 审计日志归档 {archive_date}",
            "/Subject": "ResumeMind audit archive",
            "/Creator": "ResumeMind",
        }
    )
    writer.add_attachment(_ARCHIVE_ATTACHMENT_NAME, content.encode("utf-8"))

    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as file:
        writer.write(file)
    temporary.replace(path)


def _archive_pdf_text_style() -> ParagraphStyle:
    global _PDF_FONT_REGISTERED
    if not _PDF_FONT_REGISTERED:
        _register_archive_pdf_font()
        _PDF_FONT_REGISTERED = True
    return ParagraphStyle(
        "AuditArchiveText",
        fontName=_registered_archive_pdf_font_name(),
        fontSize=8.5,
        leading=11.5,
        splitLongWords=True,
    )


def _register_archive_pdf_font() -> None:
    for font_path in _PDF_FONT_CANDIDATES:
        if not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(_PDF_FONT_NAME, str(font_path)))
            return
        except Exception:
            continue
    pdfmetrics.registerFont(UnicodeCIDFont(_PDF_FALLBACK_CID_FONT_NAME))


def _registered_archive_pdf_font_name() -> str:
    return _PDF_FONT_NAME if _PDF_FONT_NAME in pdfmetrics.getRegisteredFontNames() else _PDF_FALLBACK_CID_FONT_NAME


def _local_date(value: datetime) -> object:
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(LOCAL_TZ).date()


def _local_datetime(value: datetime) -> datetime:
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(LOCAL_TZ)


def _deleted_archives_path() -> Path:
    return AUDIT_ARCHIVE_DIR / _DELETED_ARCHIVE_MARKER_NAME


def _read_deleted_archive_dates() -> set[str]:
    path = _deleted_archives_path()
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(raw, list):
        return set()
    return {value for value in raw if isinstance(value, str) and _is_valid_archive_date(value)}


def _write_deleted_archive_dates(dates: set[str]) -> None:
    AUDIT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    _deleted_archives_path().write_text(
        json.dumps(sorted(dates), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mark_archive_deleted(archive_date: str) -> None:
    if not _is_valid_archive_date(archive_date):
        raise FileNotFoundError(archive_date)
    dates = _read_deleted_archive_dates()
    dates.add(archive_date)
    _write_deleted_archive_dates(dates)


def _is_archive_deleted(archive_date: str) -> bool:
    return archive_date in _read_deleted_archive_dates()


def _purge_deleted_archive_files() -> None:
    for archive_date in _read_deleted_archive_dates():
        _purge_archive_files(archive_date)


def _purge_archive_files(archive_date: str) -> None:
    _unlink_if_exists(_archive_path(archive_date))
    _unlink_if_exists(_legacy_archive_path(archive_date))


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _is_valid_archive_date(archive_date: str) -> bool:
    try:
        parsed = date.fromisoformat(archive_date)
    except ValueError:
        return False
    return parsed.isoformat() == archive_date


def _render_archive_markdown(archive_date: str, logs: list[AuditLog]) -> str:
    lines = [
        f"# ResumeMind 审计日志归档：{archive_date}",
        "",
        f"归档时间：{datetime.now(LOCAL_TZ).isoformat(timespec='seconds')}",
        "",
    ]
    for log in logs:
        created_at = _local_datetime(log.created_at).isoformat(timespec="seconds")
        lines.extend(
            [
                f"<!-- audit-id:{log.id} -->",
                f"## {created_at} · {log.action}",
                "",
                f"- 对象类型：{log.target_type}",
                f"- 对象编号：{log.target_id or '无'}",
                f"- 级别：{log.severity or 'info'}",
                f"- 摘要：{log.summary or log.action}",
                f"- 出现次数：{log.occurrence_count or 1}",
                "- 详情：",
                "",
                "```text",
                (log.detail or "无补充说明").replace("```", "'''"),
                "```",
                "",
            ]
        )
        if log.details_json:
            lines.extend(
                [
                    "- 结构化详情：",
                    "",
                    "```json",
                    log.details_json.replace("```", "'''"),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _default_event_key(action: str, target_type: str, target_id: str | None, detail: str) -> str:
    normalized_detail = " ".join((detail or "").split())[:120]
    if "source_missing" in action:
        normalized_detail = "source_missing"
    return f"{action}:{target_type}:{target_id or ''}:{normalized_detail}"


def _display_fields(action: str, detail: str) -> dict[str, str]:
    labels = {
        "document_uploaded": ("文档上传", "文档已上传并进入解析/索引流程。"),
        "document_indexed": ("文档入库完成", "文档已完成解析和索引，可用于问答。"),
        "document_index_failed": ("文档入库失败", "文档处理失败，需要检查文件格式、解析器或模型状态。"),
        "document_deleted": ("文档删除", "文档及其索引已删除。"),
        "document_delete_failed": ("文档删除失败", "文档删除未完成，需要稍后重试或检查向量库状态。"),
        "document_marked_source_missing": ("原文件缺失", "系统检测到原文件不可用，该文档已暂停源文件校验。"),
        "document_source_restored": ("原文件已恢复", "系统重新找到原文件，文档状态已恢复。"),
        "qa_context_built": ("问答完成", "系统已完成一次可信问答。"),
        "qa_cancelled": ("问答生成已停止", "用户主动停止了本次回答生成。"),
    }
    summary, message = labels.get(action, (action, detail or "系统记录了一次操作。"))
    return {"summary": summary, "user_message": message}


def _ensure_aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
