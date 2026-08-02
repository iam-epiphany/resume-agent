"""管理员认证端点：登录签发 JWT、校验当前会话。"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import create_access_token, get_admin_identity, verify_admin_password
from backend.app.services.audit_service import record_event


router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    expires_at: datetime


class MeResponse(BaseModel):
    authenticated: bool
    role: str


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    if not verify_admin_password(payload.password):
        # event_key 含小时级时间窗：避免 record_event 的 10 分钟聚合把多次失败并成一条
        hour_window = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        record_event(
            db,
            "auth_login_failed",
            "auth",
            None,
            detail="管理员登录失败：密码错误",
            severity="warning",
            event_key=f"auth_login_failed:{hour_window}",
            summary="登录失败",
            user_message="管理员登录失败：密码错误。",
        )
        raise HTTPException(status_code=401, detail="密码错误")
    token, expires_at = create_access_token()
    return LoginResponse(token=token, expires_at=expires_at)


@router.get("/me", response_model=MeResponse)
def me(identity: dict[str, str] = Depends(get_admin_identity)) -> MeResponse:
    return MeResponse(authenticated=True, role=identity["role"])
