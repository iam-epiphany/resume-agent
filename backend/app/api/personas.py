"""人物档案接口（2026-08-14）。

公开：GET /api/personas/active（匿名安全视图——面试官视角，驱动前台文案）。
管理员：列表 / 创建 / 切换 / 确认档案 / 更新姓名称呼 / 下载人物 Skill 包。
"""

import json
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import require_admin
from backend.app.models.document import Persona
from backend.app.services.persona_service import (
    activate_persona,
    confirm_persona_profile,
    get_active_persona,
    persona_profile,
    public_persona_view,
    update_persona_metadata,
)
from backend.app.services.persona_skill_service import (
    load_persona_skill_package,
    persona_skill_zip_bytes,
)

router = APIRouter(prefix="/personas", tags=["personas"])


class PersonaCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    display_name: str | None = Field(default=None, max_length=100)
    profile: dict = Field(default_factory=dict)


class PersonaUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    display_name: str | None = Field(default=None, max_length=100)
    profile: dict | None = None
    confirm: bool = False


class SkillPackageInfo(BaseModel):
    """人物 Skill 包元信息（不含内容；完整内容走管理员下载端点）。"""

    file_count: int
    skill_version: str | None
    generated_at: str | None


class PersonaPublicResponse(BaseModel):
    persona_id: str
    name: str
    display_name: str
    status: str
    profile_summary: str
    is_active: bool
    skill_package: SkillPackageInfo | None = None


@router.get("/active", response_model=PersonaPublicResponse)
def active_persona(db: Session = Depends(get_db)) -> PersonaPublicResponse:
    """公开：当前激活人物（匿名安全视图，供前台文案与面试官展示）。"""
    return PersonaPublicResponse(**public_persona_view(get_active_persona(db)))


@router.get("", response_model=list[PersonaPublicResponse])
def list_personas(db: Session = Depends(get_db)) -> list[PersonaPublicResponse]:
    personas = db.scalars(select(Persona).order_by(Persona.id.asc())).all()
    return [PersonaPublicResponse(**public_persona_view(persona)) for persona in personas]


@router.post("", response_model=PersonaPublicResponse)
def create_persona(
    payload: PersonaCreateRequest,
    db: Session = Depends(get_db),
) -> PersonaPublicResponse:
    persona = Persona(
        persona_id=f"persona-{payload.name.strip()[:12]}",
        name=payload.name.strip(),
        display_name=(payload.display_name or payload.name).strip(),
        profile_json=json.dumps(payload.profile, ensure_ascii=False),
        status="draft",
        is_active=False,
    )
    db.add(persona)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=409, detail="人物已存在，请使用其他姓名")
    db.refresh(persona)
    return PersonaPublicResponse(**public_persona_view(persona))


@router.post("/{persona_id}/activate", response_model=PersonaPublicResponse)
def activate(
    persona_id: str,
    db: Session = Depends(get_db),
) -> PersonaPublicResponse:
    try:
        persona = activate_persona(db, persona_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PersonaPublicResponse(**public_persona_view(persona))


@router.patch("/{persona_id}", response_model=PersonaPublicResponse)
def update_persona(
    persona_id: str,
    payload: PersonaUpdateRequest,
    db: Session = Depends(get_db),
) -> PersonaPublicResponse:
    persona = db.scalar(select(Persona).where(Persona.persona_id == persona_id))
    if persona is None:
        raise HTTPException(status_code=404, detail="人物不存在")
    if payload.name is not None or payload.display_name is not None:
        persona = update_persona_metadata(
            db,
            persona_id,
            name=payload.name,
            display_name=payload.display_name,
        )
    if payload.profile is not None or payload.confirm:
        # 合并更新档案（保留未提供的字段）
        current = persona_profile(persona)
        merged = {**current, **(payload.profile or {})} if payload.profile else current
        persona = confirm_persona_profile(db, persona_id, profile=merged)
    return PersonaPublicResponse(**public_persona_view(persona))


@router.get(
    "/{persona_id}/skill-package",
    dependencies=[Depends(require_admin)],
    response_class=Response,
    responses={404: {"description": "人物不存在或尚未生成人物 Skill 包"}},
)
def download_persona_skill_package(
    persona_id: str,
    db: Session = Depends(get_db),
) -> Response:
    """管理员：下载人物 Skill 包（zip，含 SKILL.md + references/ + facts.json）。

    包内容是该人物的全部个人材料，必须管理员鉴权——下载端点独立挂 require_admin
    （人物档案常规端点目前未鉴权，见路由注册）。
    """
    persona = db.scalar(select(Persona).where(Persona.persona_id == persona_id))
    if persona is None:
        raise HTTPException(status_code=404, detail="人物不存在")
    package = load_persona_skill_package(persona)
    if package is None:
        raise HTTPException(
            status_code=404,
            detail="该人物还没有生成人物 Skill 包，请先运行一次人物工坊转换",
        )
    zip_bytes = persona_skill_zip_bytes(package)
    filename = f"{(package.get('dir_name') or 'persona')}.skill.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )
