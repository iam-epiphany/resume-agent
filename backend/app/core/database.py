from collections.abc import Generator
import json
from pathlib import Path, PureWindowsPath
import unicodedata

from sqlalchemy import create_engine, event, func, inspect, select, text, update
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core import config
from backend.app.core.config import DATABASE_PATH, ensure_runtime_dirs

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

# 默认人物（2026-08-14）：单当前人物模型下，存量库零配置升级的兜底人物。
# 不内置任何姓名——默认人物姓名由部署者通过 DEFAULT_PERSONA_NAME 配置，
# 留空即中性"我/求职者"表述。
DEFAULT_PERSONA_ID = "default"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """SQLAlchemy model base for the local SQLite database."""


def init_db() -> None:
    """Create SQLite tables on startup; Alembic can replace this later."""

    ensure_runtime_dirs()
    from backend.app.models import audit, document  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _upgrade_sqlite_schema()
    _seed_default_persona()
    _recover_interrupted_states()


def _seed_default_persona() -> None:
    """存量/全新库的默认人物种子：零配置升级、向后兼容。

    默认人物不携带任何具体姓名（DEFAULT_PERSONA_NAME 环境变量可配置部署者自己的
    名字，留空 = 中性"我/求职者"表述，绝不内置任何人的姓名）。
    已有文档/事实回填到默认人物（persona_id=DEFAULT_PERSONA_ID），
    保证旧库升级后检索过滤不丢数据。
    """
    from backend.app.models.document import Document, FactLedger, Persona

    with SessionLocal() as db:
        existing = db.scalar(select(Persona).where(Persona.persona_id == DEFAULT_PERSONA_ID))
        if existing is None:
            db.add(
                Persona(
                    persona_id=DEFAULT_PERSONA_ID,
                    name=config.DEFAULT_PERSONA_NAME,
                    display_name=config.DEFAULT_PERSONA_NAME or "我",
                    profile_json=json.dumps(
                        {
                            "name": config.DEFAULT_PERSONA_NAME,
                            "summary": "简历问答系统的默认人物。",
                        },
                        ensure_ascii=False,
                    ),
                    status="confirmed",
                    is_active=True,
                )
            )
            db.flush()
            db.execute(
                update(Document).where(Document.persona_id.is_(None)).values(persona_id=DEFAULT_PERSONA_ID)
            )
            db.execute(
                update(FactLedger).where(FactLedger.persona_id.is_(None)).values(persona_id=DEFAULT_PERSONA_ID)
            )
            db.commit()
        elif existing.is_active is False:
            # 默认人物被切换走但仍在：保证至少一个 active（单当前人物模型）
            active_count = db.scalar(
                select(func.count()).select_from(Persona).where(Persona.is_active.is_(True))
            )
            if not active_count:
                existing.is_active = True
                db.commit()


def _upgrade_sqlite_schema() -> None:
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "document_chunks" not in table_names:
        return

    document_columns = {column["name"] for column in inspector.get_columns("documents")} if "documents" in table_names else set()
    index_task_columns = {column["name"] for column in inspector.get_columns("document_index_tasks")} if "document_index_tasks" in table_names else set()
    audit_columns = {column["name"] for column in inspector.get_columns("audit_logs")} if "audit_logs" in table_names else set()
    qa_task_columns = {column["name"] for column in inspector.get_columns("qa_tasks")} if "qa_tasks" in table_names else set()
    workshop_job_columns = {column["name"] for column in inspector.get_columns("workshop_jobs")} if "workshop_jobs" in table_names else set()
    document_migrations = {
        "client_request_id": "ALTER TABLE documents ADD COLUMN client_request_id VARCHAR(128)",
        "document_metadata": "ALTER TABLE documents ADD COLUMN document_metadata TEXT",
        "index_version": "ALTER TABLE documents ADD COLUMN index_version VARCHAR(80)",
        "index_error": "ALTER TABLE documents ADD COLUMN index_error TEXT",
        "lifecycle_stage": "ALTER TABLE documents ADD COLUMN lifecycle_stage VARCHAR(40)",
        "filename_norm": "ALTER TABLE documents ADD COLUMN filename_norm VARCHAR(255)",
        "file_sha256": "ALTER TABLE documents ADD COLUMN file_sha256 VARCHAR(64)",
        "external_doc_id": "ALTER TABLE documents ADD COLUMN external_doc_id VARCHAR(128)",
        "title": "ALTER TABLE documents ADD COLUMN title VARCHAR(500)",
        "issuing_authority": "ALTER TABLE documents ADD COLUMN issuing_authority VARCHAR(255)",
        "publication_date": "ALTER TABLE documents ADD COLUMN publication_date VARCHAR(10)",
        "expiration_date": "ALTER TABLE documents ADD COLUMN expiration_date VARCHAR(10)",
        "document_number": "ALTER TABLE documents ADD COLUMN document_number VARCHAR(255)",
        "material_topic": "ALTER TABLE documents ADD COLUMN material_topic VARCHAR(255)",
        "source_url": "ALTER TABLE documents ADD COLUMN source_url VARCHAR(2000)",
        "attachment_url": "ALTER TABLE documents ADD COLUMN attachment_url VARCHAR(2000)",
        "source_type": "ALTER TABLE documents ADD COLUMN source_type VARCHAR(40)",
        "metadata_status": "ALTER TABLE documents ADD COLUMN metadata_status VARCHAR(30) DEFAULT 'inferred'",
        "metadata_provenance": "ALTER TABLE documents ADD COLUMN metadata_provenance TEXT",
        "identity_review_status": "ALTER TABLE documents ADD COLUMN identity_review_status VARCHAR(30) DEFAULT 'unreviewed'",
        "identity_reviewed_at": "ALTER TABLE documents ADD COLUMN identity_reviewed_at DATETIME",
        "identity_reviewed_snapshot_hash": "ALTER TABLE documents ADD COLUMN identity_reviewed_snapshot_hash VARCHAR(64)",
    }
    chunk_columns = {column["name"] for column in inspector.get_columns("document_chunks")}
    chunk_migrations = {
        "embedding_text": "ALTER TABLE document_chunks ADD COLUMN embedding_text TEXT",
        "chunk_metadata": "ALTER TABLE document_chunks ADD COLUMN chunk_metadata TEXT",
        "token_count": "ALTER TABLE document_chunks ADD COLUMN token_count INTEGER DEFAULT 0",
        "index_status": "ALTER TABLE document_chunks ADD COLUMN index_status VARCHAR(30) DEFAULT 'indexed'",
        "index_version": "ALTER TABLE document_chunks ADD COLUMN index_version VARCHAR(80)",
    }
    with engine.begin() as connection:
        for column_name, statement in document_migrations.items():
            if column_name not in document_columns:
                connection.execute(text(statement))
        if "documents" in table_names:
            _backfill_document_filename_norms(connection)
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_client_request_id "
                    "ON documents(client_request_id)"
                )
            )
            for field_name in (
                "external_doc_id",
                "title",
                "issuing_authority",
                "publication_date",
                "expiration_date",
                "document_number",
                "material_topic",
                "source_type",
                "metadata_status",
                "identity_review_status",
            ):
                connection.execute(
                    text(f"CREATE INDEX IF NOT EXISTS ix_documents_{field_name} ON documents({field_name})")
                )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_filename_norm "
                    "ON documents(filename_norm)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_documents_file_sha256 "
                    "ON documents(file_sha256)"
                )
            )
        if "document_index_tasks" in table_names:
            index_task_migrations = {
                "task_id": "ALTER TABLE document_index_tasks ADD COLUMN task_id VARCHAR(32)",
                "stage": "ALTER TABLE document_index_tasks ADD COLUMN stage VARCHAR(40) DEFAULT 'queued'",
                "completed_units": "ALTER TABLE document_index_tasks ADD COLUMN completed_units INTEGER",
                "total_units": "ALTER TABLE document_index_tasks ADD COLUMN total_units INTEGER",
                "error_code": "ALTER TABLE document_index_tasks ADD COLUMN error_code VARCHAR(100)",
            }
            for column_name, statement in index_task_migrations.items():
                if column_name not in index_task_columns:
                    connection.execute(text(statement))
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_document_index_tasks_task_id "
                    "ON document_index_tasks(task_id)"
                )
            )
        for column_name, statement in chunk_migrations.items():
            if column_name not in chunk_columns:
                connection.execute(text(statement))
        audit_migrations = {
            "severity": "ALTER TABLE audit_logs ADD COLUMN severity VARCHAR(20) DEFAULT 'info'",
            "event_key": "ALTER TABLE audit_logs ADD COLUMN event_key VARCHAR(255)",
            "summary": "ALTER TABLE audit_logs ADD COLUMN summary VARCHAR(500)",
            "user_message": "ALTER TABLE audit_logs ADD COLUMN user_message TEXT",
            "details_json": "ALTER TABLE audit_logs ADD COLUMN details_json TEXT",
            "first_seen_at": "ALTER TABLE audit_logs ADD COLUMN first_seen_at DATETIME",
            "last_seen_at": "ALTER TABLE audit_logs ADD COLUMN last_seen_at DATETIME",
            "occurrence_count": "ALTER TABLE audit_logs ADD COLUMN occurrence_count INTEGER DEFAULT 1",
            "resolved": "ALTER TABLE audit_logs ADD COLUMN resolved BOOLEAN DEFAULT 0",
        }
        if "audit_logs" in table_names:
            for column_name, statement in audit_migrations.items():
                if column_name not in audit_columns:
                    connection.execute(text(statement))
        if "qa_tasks" in table_names:
            qa_task_migrations = {
                "client_request_id": "ALTER TABLE qa_tasks ADD COLUMN client_request_id VARCHAR(128)",
                "session_id": "ALTER TABLE qa_tasks ADD COLUMN session_id VARCHAR(128)",
                "error_code": "ALTER TABLE qa_tasks ADD COLUMN error_code VARCHAR(100)",
                "error_stage": "ALTER TABLE qa_tasks ADD COLUMN error_stage VARCHAR(50)",
                "error_retryable": "ALTER TABLE qa_tasks ADD COLUMN error_retryable BOOLEAN DEFAULT 0",
                "attempt_count": "ALTER TABLE qa_tasks ADD COLUMN attempt_count INTEGER DEFAULT 0",
                "answer_preview_json": "ALTER TABLE qa_tasks ADD COLUMN answer_preview_json TEXT",
                "client_ip": "ALTER TABLE qa_tasks ADD COLUMN client_ip VARCHAR(64)",
            }
            for column_name, statement in qa_task_migrations.items():
                if column_name not in qa_task_columns:
                    connection.execute(text(statement))
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_qa_tasks_client_request_id "
                    "ON qa_tasks(client_request_id)"
                )
            )
        if "qa_logs" in table_names:
            qa_log_columns = {column["name"] for column in inspector.get_columns("qa_logs")}
            qa_log_migrations = {
                "answer_mode": "ALTER TABLE qa_logs ADD COLUMN answer_mode VARCHAR(20)",
                "evidence_sufficiency": "ALTER TABLE qa_logs ADD COLUMN evidence_sufficiency VARCHAR(20)",
                "fallback_level": "ALTER TABLE qa_logs ADD COLUMN fallback_level INTEGER DEFAULT 0",
                "used_chunks": "ALTER TABLE qa_logs ADD COLUMN used_chunks INTEGER DEFAULT 0",
                "client_ip": "ALTER TABLE qa_logs ADD COLUMN client_ip VARCHAR(64)",
            }
            for column_name, statement in qa_log_migrations.items():
                if column_name not in qa_log_columns:
                    connection.execute(text(statement))
            # 按 IP 累计提问配额查询走该索引（qa_logs 行数少，查询按 client_ip 精确过滤）
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_qa_logs_client_ip "
                    "ON qa_logs(client_ip)"
                )
            )
        # 人物模型（2026-08-14）：persona_id 列迁移（存量数据回填由 _seed_default_persona 完成）
        document_columns_after = {column["name"] for column in inspector.get_columns("documents")}
        if "persona_id" not in document_columns_after:
            connection.execute(text("ALTER TABLE documents ADD COLUMN persona_id VARCHAR(40)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_documents_persona_id ON documents(persona_id)"))
        if "fact_ledger" in table_names:
            fact_columns = {column["name"] for column in inspector.get_columns("fact_ledger")}
            if "persona_id" not in fact_columns:
                connection.execute(text("ALTER TABLE fact_ledger ADD COLUMN persona_id VARCHAR(40)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS ix_fact_ledger_persona_id ON fact_ledger(persona_id)"))
            # 事实状态拆分（2026-08-15）：evidence_status（事实可信度）+ review_status（人工审核）。
            # 存量回填映射：confirmed→explicit/pending、pending→missing/pending、
            # inferred→inferred/pending、conflict→conflict/pending；其余→missing/pending。
            # 回填只在两列首次补齐时执行一次：后续每次启动不得再触碰
            # review_status，避免抹掉人工审核结果（approved/rejected）。
            columns_added = False
            if "evidence_status" not in fact_columns:
                connection.execute(
                    text("ALTER TABLE fact_ledger ADD COLUMN evidence_status VARCHAR(20) DEFAULT 'explicit'")
                )
                columns_added = True
            if "review_status" not in fact_columns:
                connection.execute(
                    text("ALTER TABLE fact_ledger ADD COLUMN review_status VARCHAR(20) DEFAULT 'pending'")
                )
                columns_added = True
            if columns_added:
                connection.execute(
                    text(
                        "UPDATE fact_ledger SET evidence_status = CASE status "
                        "WHEN 'confirmed' THEN 'explicit' "
                        "WHEN 'pending' THEN 'missing' "
                        "WHEN 'inferred' THEN 'inferred' "
                        "WHEN 'conflict' THEN 'conflict' "
                        "ELSE 'missing' END, "
                        "review_status = 'pending' "
                        "WHERE status IS NOT NULL AND status != ''"
                    )
                )
        # 人物工坊（2026-08-14）：skill 版本随任务落库（加工规范单一事实来源）
        if "workshop_jobs" in table_names:
            workshop_job_migrations = {
                "skill_version": "ALTER TABLE workshop_jobs ADD COLUMN skill_version VARCHAR(40)",
                "conflicts_json": "ALTER TABLE workshop_jobs ADD COLUMN conflicts_json TEXT",
            }
            for column_name, statement in workshop_job_migrations.items():
                if column_name not in workshop_job_columns:
                    connection.execute(text(statement))
        # 人物 Skill 包（2026-08-15）：工坊产物二次封装为可独立调用的人物 Skill（zip 下载）
        if "personas" in table_names:
            persona_columns = {column["name"] for column in inspector.get_columns("personas")}
            persona_migrations = {
                "skill_package_json": "ALTER TABLE personas ADD COLUMN skill_package_json TEXT",
                "skill_package_updated_at": "ALTER TABLE personas ADD COLUMN skill_package_updated_at DATETIME",
            }
            for column_name, statement in persona_migrations.items():
                if column_name not in persona_columns:
                    connection.execute(text(statement))
        # 银行场景遗留表已不再建模（模型已移除 SpreadsheetCell），幂等清理旧库残留
        connection.execute(text("DROP TABLE IF EXISTS spreadsheet_cells"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _backfill_document_filename_norms(connection) -> None:
    rows = connection.execute(
        text("SELECT id, filename FROM documents ORDER BY uploaded_at ASC, id ASC")
    ).mappings().all()
    used: set[str] = set()
    for row in rows:
        original_filename = str(row["filename"] or f"未命名文档-{row['id']}")
        candidate_filename = _sanitize_existing_filename(original_filename)
        candidate_norm = _normalize_existing_filename(candidate_filename)
        if candidate_norm in used:
            duplicate_index = 2
            while True:
                renamed = _duplicate_filename(candidate_filename, duplicate_index)
                renamed_norm = _normalize_existing_filename(renamed)
                if renamed_norm not in used:
                    candidate_filename = renamed
                    candidate_norm = renamed_norm
                    break
                duplicate_index += 1
        used.add(candidate_norm)
        connection.execute(
            text("UPDATE documents SET filename=:filename, filename_norm=:filename_norm WHERE id=:id"),
            {"id": row["id"], "filename": candidate_filename, "filename_norm": candidate_norm},
        )


def _sanitize_existing_filename(filename: str) -> str:
    if "\\" in filename or ":" in filename:
        basename = PureWindowsPath(filename).name
    else:
        basename = Path(filename).name
    normalized = unicodedata.normalize("NFC", basename).strip()
    return normalized[:255] or "未命名文档"


def _normalize_existing_filename(filename: str) -> str:
    return _sanitize_existing_filename(filename).casefold()


def _duplicate_filename(filename: str, duplicate_index: int) -> str:
    path = Path(filename)
    suffix = path.suffix
    stem = filename[: -len(suffix)] if suffix else filename
    marker = f" (历史重复-{duplicate_index})"
    max_stem_len = 255 - len(marker) - len(suffix)
    return f"{stem[:max_stem_len]}{marker}{suffix}"


def _recover_interrupted_states() -> None:
    """Make process-interrupted index jobs visible and safely retryable."""

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE documents SET status='index_failed', "
                "index_error='应用重启时检测到未完成的索引任务，请重试构建索引。' "
                "WHERE status='indexing'"
            )
        )
        connection.execute(
            text("UPDATE document_chunks SET index_status='index_failed' WHERE index_status='indexing'")
        )
        inspector = inspect(engine)
        if "qa_tasks" in inspector.get_table_names():
            connection.execute(
                text(
                    "UPDATE qa_tasks SET status='queued', "
                    "error='应用重启后正在恢复未完成的问答任务。', "
                    "error_code=NULL, error_stage=NULL, error_retryable=0, "
                    "answer_preview_json=NULL, "
                    "completed_at=NULL, "
                    "updated_at=CURRENT_TIMESTAMP "
                    "WHERE status IN ('queued', 'running')"
                )
            )
