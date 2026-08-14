"""人物档案服务（2026-08-14）：单当前人物模型的查询、激活与提示词渲染。

persona 驱动全链路个性化：提示词中的姓名/主人公描述、礼貌转移文案、
检索过滤（documents/fact_ledger 按 persona_id 隔离）、问答缓存签名
（切人自动失效）。status=draft 时（工坊 LLM 提取待人工确认）不参与个性化。
"""

from __future__ import annotations

import json
from threading import RLock

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.app.core.database import DEFAULT_PERSONA_ID
from backend.app.models.document import Persona
from backend.app.services.qa_cache_service import clear as clear_qa_cache

# 进程内 active persona_id 缓存（写入/激活/更新时失效；ORM 实例不入缓存——
# 会话关闭后实例会 Detached，跨请求不可用）
_ACTIVE_CACHE: dict[str, str] = {}
_ACTIVE_CACHE_LOCK = RLock()


def get_active_persona(db: Session) -> Persona:
    """当前激活人物（进程内缓存 persona_id；激活/档案变更时失效）。

    库里完全没有人物时自动创建默认人物（测试库/全新库自愈，幂等）。
    """
    cached_id = _ACTIVE_CACHE.get("active")
    if cached_id:
        persona = db.scalar(select(Persona).where(Persona.persona_id == cached_id).limit(1))
        if persona is not None:
            return persona
    persona = db.scalar(select(Persona).where(Persona.is_active.is_(True)).order_by(Persona.id.asc()).limit(1))
    if persona is None:
        # 兜底：保证至少有一个可用人物（默认人物种子在 init_db 时保证）
        persona = db.scalar(select(Persona).where(Persona.persona_id == DEFAULT_PERSONA_ID).limit(1))
    if persona is None:
        # 测试库/全新库懒加载种子：与 _seed_default_persona 行为一致（幂等）
        persona = Persona(
            persona_id=DEFAULT_PERSONA_ID,
            name="张三",
            display_name="张三",
            profile_json=json.dumps(
                {"name": "张三", "summary": "AI 应用后端开发方向，计算机相关专业背景。"},
                ensure_ascii=False,
            ),
            status="confirmed",
            is_active=True,
        )
        db.add(persona)
        db.commit()
    with _ACTIVE_CACHE_LOCK:
        _ACTIVE_CACHE["active"] = persona.persona_id
    return persona


def _invalidate_active_persona() -> None:
    with _ACTIVE_CACHE_LOCK:
        _ACTIVE_CACHE.pop("active", None)


def activate_persona(db: Session, persona_id: str) -> Persona:
    """切换当前人物：置 is_active 唯一，清空问答缓存与文档快照缓存（防串人物）。"""
    persona = db.scalar(select(Persona).where(Persona.persona_id == persona_id))
    if persona is None:
        raise ValueError(f"人物不存在：{persona_id}")
    db.execute(update(Persona).values(is_active=False))
    persona.is_active = True
    db.commit()
    _invalidate_active_persona()
    clear_qa_cache()          # model_signature 含 persona_id，内存缓存一并清掉
    try:
        # 文档快照缓存（延迟导入避免与 rag_service 循环依赖）
        from backend.app.services.rag_service import clear_document_snapshot_cache

        clear_document_snapshot_cache()
    except Exception:
        pass
    return persona


def confirm_persona_profile(db: Session, persona_id: str, profile: dict | None = None) -> Persona:
    """人工确认人物档案（status → confirmed；profile 可选更新）。"""
    persona = db.scalar(select(Persona).where(Persona.persona_id == persona_id))
    if persona is None:
        raise ValueError(f"人物不存在：{persona_id}")
    if profile is not None:
        persona.profile_json = json.dumps(profile, ensure_ascii=False)
        persona.name = str(profile.get("name") or persona.name)
        persona.display_name = str(profile.get("display_name") or profile.get("name") or persona.name)
    persona.status = "confirmed"
    db.commit()
    _invalidate_active_persona()
    return persona


def update_persona_metadata(db: Session, persona_id: str, *, name: str | None = None, display_name: str | None = None) -> Persona:
    persona = db.scalar(select(Persona).where(Persona.persona_id == persona_id))
    if persona is None:
        raise ValueError(f"人物不存在：{persona_id}")
    if name is not None:
        persona.name = name
    if display_name is not None:
        persona.display_name = display_name
    db.commit()
    _invalidate_active_persona()
    return persona


def persona_profile(persona: Persona) -> dict:
    """人物档案字典（解析 profile_json，缺失字段回退默认）。"""
    try:
        profile = json.loads(persona.profile_json) if persona.profile_json else {}
    except (json.JSONDecodeError, TypeError):
        profile = {}
    if not isinstance(profile, dict):
        profile = {}
    profile.setdefault("name", persona.name or "求职者")
    profile.setdefault("display_name", persona.display_name or persona.name or "求职者")
    return profile


def persona_prompt_context(persona: Persona) -> dict[str, str]:
    """提示词渲染上下文：姓名、主人公描述（按档案状态分级）。

    draft 人物（工坊提取待人工确认）不参与个性化：用中性"简历主人公"表述，
    避免未确认的姓名/描述注入提示词造成误导。
    """
    if persona.status != "confirmed":
        return {"persona_name": "", "persona_description": "简历主人公（求职者）"}
    profile = persona_profile(persona)
    name = str(profile.get("name") or persona.name or "").strip()
    summary = str(profile.get("summary") or "").strip()
    if name and summary:
        description = f"简历主人公（{name}，{summary}）"
    elif name:
        description = f"简历主人公（{name}）"
    else:
        description = "简历主人公（求职者）"
    return {"persona_name": name, "persona_description": description}


def public_persona_view(persona: Persona) -> dict:
    """匿名安全视图：仅姓名/称呼/确认状态（面试官视角，不含完整档案）。"""
    profile = persona_profile(persona)
    return {
        "persona_id": persona.persona_id,
        "name": persona.name,
        "display_name": persona.display_name or persona.name,
        "status": persona.status,
        "profile_summary": profile.get("summary") or "",
        "is_active": bool(persona.is_active),
    }
