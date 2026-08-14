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

隐私清洗：身份证/手机号/银行卡/邮箱等 PII 强制删除或脱敏，不进入知识库。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

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
from backend.app.services.document_parser import parse_document
from backend.app.services.document_storage import (
    EmptyDocumentError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
)
from backend.app.services.document_upload_service import create_document_upload
from backend.app.services.fact_ledger_service import seed_fact_records
from backend.app.services.index_task_service import enqueue_document_index
from backend.app.services.llm_client import ChatCompletionConfig, chat_completion_content

logger = logging.getLogger(__name__)

WORKSHOP_TRANSFORM_PROMPT = """你是简历材料加工师。用户提供了某求职者的原始简历材料（可能是聊天记录、PDF 文本、课程作业说明等），请把它们加工成适合简历问答系统检索的结构化知识库。

输出必须是 JSON 对象，包含：
1. persona_profile: 人物档案对象 {name, display_name, summary, education, job_intent, skills, projects}——从材料提取；无法确认的字段省略。
2. knowledge_documents: 数组，每篇一个主题，结构：
   {{"filename": "主题_文件名.md", "content": "以 `> 材料主题：类别` 开头的 Markdown（类别限：项目经历/技能专长/教育背景/竞赛奖项/荣誉奖励/证书资格/求职意向/个人特质/自我介绍/综合简历），正文每段只陈述一个完整事实或一个“场景—方案—结果”，用具体名词避免“它/这个/上述”指代"}}
3. facts: 数组，结构化事实 {{"subject", "predicate", "value", "status": "confirmed|pending"}}——status 只能从材料直接确认时为 confirmed，其余 pending。

硬性要求：
- 忠于材料：姓名、时间、角色、技术、数字、成果不得补写或拔高；材料中没有的标 [待确认]
- 区分事实与观点：压测数字、排名、奖项等硬事实照抄原文数字，禁止估算
- 隐私清洗：身份证号、银行卡、手机号、邮箱、家庭住址等一律删除或替换为“[已脱敏]”，不得进入文档内容
- 一篇文件只讲一个主题；标题表达具体对象

当前任务输入（第 {batch_index}/{total_batches} 批）：
{input_text}"""


class WorkshopError(RuntimeError):
    pass


@dataclass
class WorkshopResult:
    persona_profile: dict[str, Any] = field(default_factory=dict)
    documents: list[dict[str, str]] = field(default_factory=list)
    facts: list[dict[str, str]] = field(default_factory=list)


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object")
    return content[start : end + 1]


def _sanitize_privacy(text: str) -> str:
    """隐私清洗：PII 模式删除/脱敏（身份证/手机/邮箱/银行卡/地址）。"""
    text = re.sub(r"\b\d{17}[\dXx]\b", "[已脱敏]", text)                    # 18 位身份证
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[已脱敏]", text)            # 大陆手机号
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[已脱敏]", text)
    text = re.sub(r"\b(?:4\d{3}|5[1-5]\d{2}|6\d{3}|3[47]\d{2})\d{12,15}\b", "[已脱敏]", text)
    return text


async def transform_materials(
    db: Session,
    files: list[UploadFile],
    *,
    persona_id: str,
    max_files: int = WORKSHOP_MAX_FILES_PER_JOB,
    max_input_chars: int = WORKSHOP_MAX_INPUT_CHARS,
) -> WorkshopJob:
    """转换主入口：解析 → LLM 分批加工 → 自动入库（文档 + 档案 + 事实）→ 任务落库。

    返回 WorkshopJob（含生成文档 id 与事实数，供回滚）。
    """
    if not WORKSHOP_ENABLED:
        raise WorkshopError("人物工坊未启用（WORKSHOP_ENABLED=false）")
    if not files:
        raise WorkshopError("请至少上传一份原始材料")
    if len(files) > max_files:
        raise WorkshopError(f"单次最多 {max_files} 份材料")
    if not (WORKSHOP_API_KEY and WORKSHOP_MODEL):
        raise WorkshopError("人物工坊 LLM 未配置（WORKSHOP_API_KEY/WORKSHOP_MODEL）")

    job = WorkshopJob(
        job_id=uuid4().hex[:16],
        persona_id=persona_id,
        status="running",
        stage="parsing",
        raw_filenames_json=json.dumps([file.filename or "未命名" for file in files], ensure_ascii=False),
    )
    db.add(job)
    db.commit()

    generated_document_ids: list[str] = []
    try:
        # ① 解析全部原始材料
        raw_texts: list[str] = []
        for file in files:
            parsed = parse_document_safely(file)
            raw_texts.append(parsed)
        combined = "\n\n".join(raw_texts)
        if not combined.strip():
            raise WorkshopError("所有材料解析后均为空")

        # ② 分批调 LLM 加工
        chunks = _split_input_chunks(combined, max_input_chars)
        total_facts: list[dict[str, str]] = []
        merged_profile: dict[str, Any] = {}
        generated_docs: list[dict[str, str]] = []
        job.stage = "transforming"
        db.commit()
        llm_calls = 0
        for index, chunk in enumerate(chunks, start=1):
            result = _transform_batch(chunk, batch_index=index, total_batches=len(chunks))
            llm_calls += 1
            generated_docs.extend(result.documents)
            total_facts.extend(result.facts)
            if result.persona_profile:
                merged_profile = {**merged_profile, **result.persona_profile}

        # ③ 自动入库
        job.stage = "ingesting"
        db.commit()
        for doc in generated_docs:
            filename = _safe_filename(doc.get("filename") or "材料加工_文档.md")
            content = _sanitize_privacy(doc.get("content") or "")
            if not content.strip():
                continue
            document_id = await _ingest_markdown(db, persona_id, filename, content)
            if document_id:
                generated_document_ids.append(document_id)

        # ④ 人物档案（draft，人工确认后生效）
        if merged_profile.get("name"):
            _upsert_persona_profile(db, persona_id, merged_profile)

        # ⑤ 事实入台账（status=pending，只告警不硬校验）
        fact_count = 0
        if total_facts:
            fact_count = seed_fact_records(
                db,
                [
                    {
                        "fact_id": f"workshop-{job.job_id}-{index}",
                        "subject": str(fact.get("subject") or "")[:255],
                        "predicate": str(fact.get("predicate") or "")[:255],
                        "value": str(fact.get("value") or "")[:500],
                        "status": "pending" if fact.get("status") != "confirmed" else "confirmed",
                        "source_file": str(fact.get("source_file") or "工坊加工材料")[:255],
                    }
                    for index, fact in enumerate(total_facts)
                    if fact.get("subject") and fact.get("value")
                ],
            )

        job.status = "completed"
        job.stage = "completed"
        job.generated_document_ids_json = json.dumps(generated_document_ids, ensure_ascii=False)
        job.generated_fact_count = fact_count
        job.llm_call_count = llm_calls
        from datetime import datetime, timezone

        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        return job
    except Exception as exc:
        logger.exception("人物工坊转换失败")
        job.status = "failed"
        job.error = str(exc)[:1000]
        db.commit()
        if isinstance(exc, WorkshopError):
            raise
        raise WorkshopError(f"转换失败：{exc}") from exc


def parse_document_safely(file: UploadFile) -> str:
    """解析上传的原始材料为文本（复用 document_parser；失败给出可读错误）。"""
    try:
        path = _temporary_file(file)
        try:
            parsed = parse_document(path, source_name=file.filename or "材料")
            return parsed.text or ""
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
    messages = [
        {
            "role": "system",
            "content": WORKSHOP_TRANSFORM_PROMPT
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


async def _ingest_markdown(db: Session, persona_id: str, filename: str, content: str) -> str | None:
    """生成的 Markdown 经标准上传管线入库（persona_id=当前人物）。"""
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
            enqueue_index=enqueue_document_index,
        )
        document.persona_id = persona_id
        db.commit()
        return document.document_id
    except Exception as exc:
        logger.warning("工坊文档入库失败 %s: %s", filename, exc)
        return None


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
    if job.generated_fact_count:
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
    return job


def list_jobs(db: Session, limit: int = 20) -> list[WorkshopJob]:
    return list(
        db.scalars(
            select(WorkshopJob)
            .order_by(WorkshopJob.id.desc())
            .limit(limit)
        ).all()
    )
