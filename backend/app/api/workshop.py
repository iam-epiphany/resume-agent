"""人物工坊接口（2026-08-14，管理员）。

POST /api/workshop/transform  上传原始资料 → LLM 加工 → 自动入库（人物档案 draft）
GET  /api/workshop/jobs       转换任务列表（状态/产物/日志）
POST /api/workshop/jobs/{job_id}/rollback  一键回滚（删除生成文档与事实）
"""

import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.core.config import WORKSHOP_MAX_FILES_PER_JOB
from backend.app.core.database import get_db
from backend.app.core.security import require_admin
from backend.app.models.document import WorkshopJob
from backend.app.services.materials_workshop_service import (
    WorkshopError,
    list_jobs,
    rollback_job,
    transform_materials,
)
from backend.app.services.persona_service import get_active_persona

router = APIRouter(
    prefix="/workshop",
    tags=["workshop"],
    dependencies=[Depends(require_admin)],
)


@router.post("/transform")
async def transform(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> dict:
    """上传原始简历材料 → LLM 加工 → 自动入库（人物档案 draft 待确认）。"""
    if len(files) > WORKSHOP_MAX_FILES_PER_JOB:
        raise HTTPException(status_code=400, detail=f"单次最多 {WORKSHOP_MAX_FILES_PER_JOB} 份材料")
    persona = get_active_persona(db)
    try:
        job = await transform_materials(
            db,
            files,
            persona_id=persona.persona_id,
        )
    except WorkshopError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "persona_id": job.persona_id,
        "skill_version": job.skill_version,
        "generated_document_ids": json.loads(job.generated_document_ids_json or "[]"),
        "generated_fact_count": job.generated_fact_count,
        "llm_call_count": job.llm_call_count,
        "error": job.error,
    }


@router.get("/jobs")
def jobs(limit: int = 20, db: Session = Depends(get_db)) -> list[dict]:
    records = list_jobs(db, limit=limit)
    return [
        {
            "job_id": job.job_id,
            "persona_id": job.persona_id,
            "status": job.status,
            "stage": job.stage,
            "skill_version": job.skill_version,
            "raw_filenames": json.loads(job.raw_filenames_json or "[]"),
            "generated_document_ids": json.loads(job.generated_document_ids_json or "[]"),
            "generated_fact_count": job.generated_fact_count,
            "llm_call_count": job.llm_call_count,
            "error": job.error,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
        for job in records
    ]


@router.post("/jobs/{job_id}/rollback")
def rollback(job_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        job = rollback_job(db, job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
    }
