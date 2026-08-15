"""人物工坊服务（2026-08-14）：任意简历资料 → LLM 加工为检索友好的知识库文档。

流程：
1. 上传原始资料（管理员，多文件）→ document_parser 解析取文本，原始文件留存溯源
2. LLM「材料加工师」按批转换（json_object 输出）：
   - persona_profile：人物档案（姓名/教育/意向/技能摘要…）→ Persona(draft)，人工确认后生效
   - knowledge_documents[]：一主题一文件 Markdown（> 材料主题：类别 开头、单事实段落、
     事实/观点分离、无法确认标 [待确认]、隐私清洗）
   - facts[]：结构化事实（subject/predicate/value/status）→ fact_ledger（status=pending，
     只告警不硬校验，人工确认后升 confirmed）
3. 自动入库：生成文档经 create_document_upload → 索引队列（persona_id=当前人物）
4. WorkshopJob 落库（审计 + 一键回滚兜底）

隐私清洗：身份证/手机号/银行卡/邮箱/家庭住址等 PII 在解析期本地预清洗
（LLM 之前）+ 生成后二次清洗（三道出口：文档/档案/事实），绝不进入知识库。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import jsonschema
from fastapi import UploadFile
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.core.config import (
    WORKSHOP_API_KEY,
    WORKSHOP_BASE_URL,
    WORKSHOP_ENABLED,
    WORKSHOP_MAX_FILES_PER_JOB,
    WORKSHOP_MAX_INPUT_CHARS,
    WORKSHOP_MODEL,
    WORKSHOP_PROVIDER,
    WORKSHOP_TIMEOUT_SECONDS,
)
from backend.app.models.document import Document, DocumentChunk, FactLedger, Persona, WorkshopJob
from backend.app.services import skill_loader
from backend.app.services.document_parser import parse_document
from backend.app.services.document_storage import (
    EmptyDocumentError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
)
from backend.app.services.document_upload_service import (
    DocumentUploadConflictError,
    ExactDuplicateDocumentError,
    create_document_upload,
)
from backend.app.services.fact_ledger_service import seed_fact_records
from backend.app.services.index_task_service import enqueue_document_index
from backend.app.services.llm_client import ChatCompletionConfig, chat_completion_content
from backend.app.services.persona_skill_service import regenerate_persona_skill
from backend.app.services.skill_loader import SkillLoadError

logger = logging.getLogger(__name__)

# 加工提示词不硬编码在业务代码里（2026-08-14 skill 化）：以
# .agents/skills/resume-materials-workshop/ 为单一事实来源，由 skill_loader
# 运行时加载提示词与输出契约；修改加工规则请改 skill 并提升 metadata.version。
# 历史遗留：旧常量 WORKSHOP_TRANSFORM_PROMPT 中的 {{"filename"}} 双花括号转义
# 已随迁移修正为 {"filename"}（见 skill README 版本记录）。


class WorkshopError(RuntimeError):
    pass


@dataclass
class WorkshopResult:
    persona_profile: dict[str, Any] = field(default_factory=dict)
    documents: list[dict[str, str]] = field(default_factory=list)
    facts: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SourceText:
    """一份原始材料的解析结果（保留来源边界，供溯源标注与事实追溯）。"""

    source_id: str
    source_filename: str
    text: str


_validator_cache: Any | None = None


def _load_business_validator() -> Any:
    """加载 skill 的 scripts/validate_output.py（业务契约校验 + Reduce 归并）。

    确定性规则以 skill 目录为单一事实来源：后端不重复实现去重/冲突/类别校验逻辑。
    """
    global _validator_cache
    if _validator_cache is None:
        try:
            _validator_cache = skill_loader.load_script(
                skill_loader.skill_root(), "scripts/validate_output.py"
            )
        except SkillLoadError as exc:
            raise WorkshopError(f"加工 skill 不可用：{exc}") from exc
    return _validator_cache


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object")
    return content[start : end + 1]


# 家庭住址（保守：省市区 + 路/街/巷/弄/大道 + 门牌号。要求门牌号结尾，
# 避免误伤「项目部署在上海市」「高新区」等一般表述；门牌号前允许空格）
_ADDRESS_RE = re.compile(
    r"[\u4e00-\u9fa5]{2,}(?:省|市|区|县|自治区|特别行政区)"
    r"[\u4e00-\u9fa5]{1,14}(?:路|街|大道|巷|弄)\s*[0-9一二三四五六七八九十百]+\s*号(?:院|楼|室|幢|栋)?"
)


def _sanitize_privacy(text: str) -> str:
    """隐私清洗：PII 模式删除/脱敏（身份证/手机/邮箱/银行卡/家庭住址）。"""
    text = re.sub(r"\b\d{17}[\dXx]\b", "[已脱敏]", text)                    # 18 位身份证
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[已脱敏]", text)            # 大陆手机号
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[已脱敏]", text)
    text = re.sub(r"\b(?:4\d{3}|5[1-5]\d{2}|6\d{3}|3[47]\d{2})\d{12,15}\b", "[已脱敏]", text)
    text = _ADDRESS_RE.sub("[已脱敏]", text)
    return text


def _sanitize_privacy_payload(value: Any) -> Any:
    """对 LLM 输出整体递归清洗：知识文档、人物档案、事实三个出口统一脱敏。

    与解析期的预清洗构成两道防线：先本地预清洗再送 LLM，生成结果再洗一遍。
    """
    if isinstance(value, str):
        return _sanitize_privacy(value)
    if isinstance(value, dict):
        return {key: _sanitize_privacy_payload(sub) for key, sub in value.items()}
    if isinstance(value, list):
        return [_sanitize_privacy_payload(item) for item in value]
    return value


async def transform_materials(
    db: Session,
    files: list[UploadFile],
    *,
    persona_id: str,
    max_files: int = WORKSHOP_MAX_FILES_PER_JOB,
    max_input_chars: int = WORKSHOP_MAX_INPUT_CHARS,
) -> WorkshopJob:
    """转换主入口：解析（PII 预清洗、保留来源）→ 分批 LLM 加工 → Reduce 归并 →
    业务校验 → 原子入库（文档 + 档案 + 事实）→ 任务落库。

    原子性（2026-08-15）：所有校验（schema + 业务契约）在写库前完成；
    入库期间每成功一篇立即把 document_id 追加进任务台账并 commit，
    任何一步失败 → status=failed + 自动回滚已入库产物，不留半成功数据。

    返回 WorkshopJob（含生成文档 id、事实数、冲突清单，供回滚与审计）。
    """
    if not WORKSHOP_ENABLED:
        raise WorkshopError("人物工坊未启用（WORKSHOP_ENABLED=false）")
    if not files:
        raise WorkshopError("请至少上传一份原始材料")
    if len(files) > max_files:
        raise WorkshopError(f"单次最多 {max_files} 份材料")
    if not (WORKSHOP_API_KEY and WORKSHOP_MODEL):
        raise WorkshopError("人物工坊 LLM 未配置（WORKSHOP_API_KEY/WORKSHOP_MODEL）")
    # skill 规范前置检查：版本、契约与业务校验脚本以 skill 目录为单一事实来源，缺失即拒绝开工
    try:
        skill_version = skill_loader.skill_version()
        skill_loader.load_output_schema()
        _load_business_validator()
    except SkillLoadError as exc:
        raise WorkshopError(f"加工 skill 不可用：{exc}") from exc

    job = WorkshopJob(
        job_id=uuid4().hex[:16],
        persona_id=persona_id,
        status="running",
        stage="parsing",
        skill_version=skill_version,
        raw_filenames_json=json.dumps([file.filename or "未命名" for file in files], ensure_ascii=False),
    )
    db.add(job)
    db.commit()

    try:
        # ① 解析全部原始材料：保留来源边界（source_filename），PII 本地预清洗
        sources: list[SourceText] = []
        for file in files:
            filename = file.filename or "未命名"
            text = parse_document_safely(file)
            if text.strip():
                sources.append(
                    SourceText(source_id=uuid4().hex[:8], source_filename=filename, text=text)
                )
        if not sources:
            raise WorkshopError("所有材料解析后均为空")
        source_names = [source.source_filename for source in sources]
        # 每份材料前带来源标注，模型可知每一句的出处（Source Provenance）
        combined = "\n\n".join(
            f"【来源：{source.source_filename}】\n{source.text}" for source in sources
        )

        # ② 分批调 LLM 加工（此阶段不写任何业务表）
        chunks = _split_input_chunks(combined, max_input_chars)
        job.stage = "transforming"
        db.commit()
        llm_calls = 0
        batch_documents: list[dict[str, Any]] = []
        batch_facts: list[dict[str, Any]] = []
        batch_profiles: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            result = _transform_batch(chunk, batch_index=index, total_batches=len(chunks))
            llm_calls += 1
            batch_documents.extend(result.documents)
            batch_facts.extend(result.facts)
            if result.persona_profile:
                batch_profiles.append(result.persona_profile)

        # ③ Reduce：跨批次归并（文档去重/冲突重命名、事实去重/冲突标记、档案合并）
        validator = _load_business_validator()
        generated_docs, total_facts, conflicts = validator.reconcile(batch_documents, batch_facts)
        merged_profile = validator.merge_profiles(batch_profiles)
        job.conflicts_json = json.dumps(conflicts, ensure_ascii=False)

        # ④ 业务契约校验（写库前最后一道门：材料主题行/类别/文件名前缀/source_file 合法/PII 残留）
        issues = validator.validate(
            {
                "persona_profile": merged_profile,
                "knowledge_documents": generated_docs,
                "facts": total_facts,
            },
            allowed_sources=source_names,
        )
        if issues:
            raise WorkshopError(
                f"加工输出不合业务契约（{len(issues)} 项，首项：{issues[0]}）"
            )

        # ⑤ 原子入库：每成功一篇立即记入任务台账（失败可完整回滚）
        job.stage = "ingesting"
        db.commit()
        generated_document_ids: list[str] = []
        for doc in generated_docs:
            filename = _safe_filename(doc.get("filename") or "材料加工_文档.md")
            content = doc.get("content") or ""
            if not content.strip():
                raise WorkshopError(f"文档 {filename} 内容为空，拒绝入库")
            sources_meta = doc.get("sources")
            document_id = await _ingest_markdown(
                db, job, persona_id, filename, content,
                sources=[str(name) for name in sources_meta] if isinstance(sources_meta, list) else None,
            )
            if document_id:
                generated_document_ids.append(document_id)

        # ⑥ 人物档案（draft，人工确认后生效）
        if merged_profile.get("name"):
            _upsert_persona_profile(db, persona_id, merged_profile)

        # ⑦ 事实入台账：evidence_status 由 LLM 证据分级；review_status=pending 待人工审核
        fact_count = 0
        if total_facts:
            fact_count = seed_fact_records(
                db,
                [
                    {
                        "fact_id": f"workshop-{job.job_id}-{index}",
                        "persona_id": persona_id,
                        "subject": str(fact.get("subject") or "")[:255],
                        "predicate": str(fact.get("predicate") or "")[:255],
                        "value": str(fact.get("value") or "")[:500],
                        "evidence_status": str(fact.get("evidence_status") or "explicit")[:20],
                        "review_status": "pending",
                        "source_file": str(fact.get("source_file") or "")[:255] or None,
                        "source_section": str(fact.get("source_section") or "")[:255] or None,
                    }
                    for index, fact in enumerate(total_facts)
                    if fact.get("subject") and fact.get("value")
                ],
            )
            job.generated_fact_count = fact_count
            db.commit()

        # ⑧ 人物 Skill 包重建（best-effort：加工产物二次封装为可独立调用的人物 Skill，
        #    模板缺失时跳过，不影响任务成功）
        regenerate_persona_skill(db, persona_id)

        job.status = "completed"
        job.stage = "completed"
        job.generated_document_ids_json = json.dumps(generated_document_ids, ensure_ascii=False)
        job.llm_call_count = llm_calls
        from datetime import datetime, timezone

        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        return job
    except Exception as exc:
        logger.exception("人物工坊转换失败")
        # 失败不静默：标记 failed 并自动回滚已入库产物（增量台账保证可完整撤销）
        auto_note = ""
        try:
            ingested = json.loads(job.generated_document_ids_json or "[]")
            if ingested or job.generated_fact_count:
                rollback_job(db, job.job_id)
                auto_note = f"；已自动回滚 {len(ingested)} 篇已入库文档"
        except Exception as rollback_exc:  # noqa: BLE001 — 回滚本身失败也要保留任务失败状态
            logger.warning("自动回滚失败：%s", rollback_exc)
        job.status = "failed"
        job.error = f"{exc}{auto_note}"[:1000]
        db.commit()
        if isinstance(exc, WorkshopError):
            raise
        raise WorkshopError(f"转换失败：{exc}") from exc


def parse_document_safely(file: UploadFile) -> str:
    """解析上传的原始材料为文本（复用 document_parser；失败给出可读错误）。

    PII 在调用 LLM 前本地预清洗（第一道防线）：手机号/邮箱/身份证等
    不离开本机、不发送给外部 LLM。
    """
    try:
        path = _temporary_file(file)
        try:
            parsed = parse_document(path, source_name=file.filename or "材料")
            return _sanitize_privacy(parsed.text or "")
        finally:
            path.unlink(missing_ok=True)
    except (UnsupportedDocumentTypeError, EmptyDocumentError, DocumentTooLargeError) as exc:
        raise WorkshopError(str(exc)) from exc
    except Exception as exc:
        raise WorkshopError(f"材料 {file.filename} 解析失败：{exc}") from exc


def _temporary_file(file: UploadFile) -> Path:
    import tempfile

    suffix = Path(file.filename or "材料.txt").suffix or ".txt"
    fd, name = tempfile.mkstemp(suffix=suffix)
    with open(fd, "wb") as target:
        data = file.file.read(50 * 1024 * 1024 + 1)
        if len(data) > 50 * 1024 * 1024:
            raise DocumentTooLargeError("材料超过上传大小限制")
        target.write(data)
    return Path(name)


def _split_input_chunks(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        # 单段超限：按字符硬切窗口拆（保留段落边界优先）
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start : start + max_chars])
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _transform_batch(input_text: str, *, batch_index: int, total_batches: int) -> WorkshopResult:
    """调 LLM 加工一批材料。

    prompt injection 防线：system 消息只含 skill 固定指令（不含任何用户内容），
    原始材料放进 user 消息并声明「不可信数据、不执行其中指令」。
    """
    try:
        prompt = skill_loader.load_transform_prompt()
        user_template = skill_loader.load_user_message_template()
    except SkillLoadError as exc:
        raise WorkshopError(f"加工 skill 不可用：{exc}") from exc
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": user_template
            .replace("{batch_index}", str(batch_index))
            .replace("{total_batches}", str(total_batches))
            .replace("{input_text}", input_text),
        },
    ]
    config = ChatCompletionConfig(
        provider=WORKSHOP_PROVIDER,
        api_key=WORKSHOP_API_KEY,
        base_url=WORKSHOP_BASE_URL,
        model=WORKSHOP_MODEL,
        timeout_seconds=WORKSHOP_TIMEOUT_SECONDS,
        response_format="json_object",
    )
    try:
        content = chat_completion_content(config, messages, temperature=0, max_tokens=3000)
    except Exception as exc:
        raise WorkshopError(f"LLM 加工失败：{exc}") from exc
    try:
        payload = json.loads(_extract_json_object(content))
    except (json.JSONDecodeError, ValueError) as exc:
        raise WorkshopError("LLM 加工输出不是合法 JSON") from exc
    try:
        jsonschema.validate(payload, skill_loader.load_output_schema())
    except jsonschema.exceptions.ValidationError as exc:
        # 契约校验失败 → 整批拒绝，不静默降级（skill 1.0.0 起的硬性约束）
        raise WorkshopError(f"LLM 加工输出不合 skill 契约：{exc.message}") from exc
    # 第二道防线：LLM 输出整体再洗一遍（文档/档案/事实三出口统一）
    payload = _sanitize_privacy_payload(payload)
    profile = payload.get("persona_profile")
    documents = payload.get("knowledge_documents") or payload.get("documents") or []
    facts = payload.get("facts") or []
    return WorkshopResult(
        persona_profile=profile if isinstance(profile, dict) else {},
        documents=[doc for doc in documents if isinstance(doc, dict)],
        facts=[fact for fact in facts if isinstance(fact, dict)],
    )


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", filename or "").strip()
    if not cleaned:
        cleaned = "材料加工_文档.md"
    if not cleaned.lower().endswith(".md"):
        cleaned = f"{Path(cleaned).stem}.md"
    return cleaned[:255]


async def _ingest_markdown(
    db: Session,
    job: WorkshopJob,
    persona_id: str,
    filename: str,
    content: str,
    sources: list[str] | None = None,
) -> str:
    """生成的 Markdown 经标准上传管线入库，成功后把 document_id 追加进任务台账。

    失败 → 抛 WorkshopError 中断任务（不静默跳过，由调用方统一回滚）；
    与知识库内容完全重复（同 sha）→ 视为已存在，跳过写入并返回空串。
    文档溯源（sources）随 metadata 落库（workshop_sources）。
    """
    from io import BytesIO

    from starlette.datastructures import Headers

    upload = UploadFile(
        file=BytesIO(content.encode("utf-8")),
        filename=filename,
        size=len(content.encode("utf-8")),
        headers=Headers({"content-type": "text/markdown"}),
    )
    try:
        document, _task = await create_document_upload(
            db,
            file=upload,
            request_id=f"workshop-{uuid4().hex[:16]}",
            filename_override=filename,
            max_bytes=50 * 1024 * 1024,
            metadata={"workshop_sources": sources} if sources else None,
            metadata_source="system",
            enqueue_index=enqueue_document_index,
        )
    except ExactDuplicateDocumentError:
        logger.info("工坊文档与知识库内容重复，跳过写入：%s", filename)
        return ""
    except DocumentUploadConflictError as exc:
        raise WorkshopError(f"工坊文档入库失败 {filename}：{exc.message}") from exc
    except Exception as exc:
        raise WorkshopError(f"工坊文档入库失败 {filename}：{exc}") from exc
    document.persona_id = persona_id
    # 增量台账：每成功一篇立即落库，任何时刻失败都能完整回滚（原子性的兜底）
    generated_ids = json.loads(job.generated_document_ids_json or "[]")
    generated_ids.append(document.document_id)
    job.generated_document_ids_json = json.dumps(generated_ids, ensure_ascii=False)
    db.commit()
    return document.document_id


def _upsert_persona_profile(db: Session, persona_id: str, profile: dict[str, Any]) -> None:
    persona = db.scalar(
        select(Persona).where(Persona.persona_id == persona_id)
    )
    name = str(profile.get("name") or "").strip()
    if persona is None:
        db.add(
            Persona(
                persona_id=persona_id,
                name=name or "新人物",
                display_name=str(profile.get("display_name") or name or "新人物"),
                profile_json=json.dumps(profile, ensure_ascii=False),
                status="draft",
                is_active=False,
            )
        )
    else:
        persona.name = name or persona.name
        persona.display_name = str(profile.get("display_name") or persona.display_name or persona.name)
        persona.profile_json = json.dumps(profile, ensure_ascii=False)
    db.commit()


def rollback_job(db: Session, job_id: str) -> WorkshopJob:
    """回滚任务：删除该任务生成的文档与事实（自动入库的兜底）。"""
    job = db.scalar(
        select(WorkshopJob).where(WorkshopJob.job_id == job_id)
    )
    if job is None:
        raise ValueError("任务不存在")
    try:
        generated_ids = json.loads(job.generated_document_ids_json or "[]")
    except json.JSONDecodeError:
        generated_ids = []
    from backend.app.services.document_lifecycle_service import delete_document_record

    for document_id in generated_ids:
        document = db.scalar(
            select(Document).where(Document.document_id == document_id)
        )
        if document is not None:
            try:
                delete_document_record(db, document)
            except Exception as exc:
                # 生命周期删除失败（如 Qdrant 不可用）时直接删 SQLite 记录与
                # 关联 chunk——工坊回滚是"兜底撤销"，宁可留向量残留也不留文档
                logger.warning("回滚生命周期删除失败 %s: %s，降级直接删记录", document_id, exc)
                db.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
                )
                db.delete(document)
    # 事实台账按 job 前缀删除（不依赖 generated_fact_count——任务中途失败时
    # 计数可能尚未落库，但事实已写入，回滚同样要清掉）
    facts = db.scalars(
        select(FactLedger).where(
            FactLedger.fact_id.like(f"workshop-{job.job_id}-%")
        )
    ).all()
    for fact in facts:
        db.delete(fact)
    job.status = "rolled_back"
    job.error = "已回滚：生成文档与事实已删除"
    db.commit()
    # 回滚后重建人物 Skill 包，包内容与知识库保持一致（best-effort）
    regenerate_persona_skill(db, job.persona_id)
    return job


def list_jobs(db: Session, limit: int = 20) -> list[WorkshopJob]:
    return list(
        db.scalars(
            select(WorkshopJob)
            .order_by(WorkshopJob.id.desc())
            .limit(limit)
        ).all()
    )
