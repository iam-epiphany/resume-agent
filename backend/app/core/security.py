"""管理员认证：密码校验、JWT 签发/校验、FastAPI 依赖。

单管理员、无用户表：密码来自 .env（ADMIN_PASSWORD），token 为 HS256 JWT。
ADMIN_PASSWORD 缺失时模块导入即报错（fail-closed），阻止未配置就启动。
"""

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.core import config

if not config.ADMIN_PASSWORD:
    raise RuntimeError(
        "ADMIN_PASSWORD 未配置：请在 .env 中设置管理员密码后启动（前台/后台权限分离要求）。"
    )

_TOKEN_ISSUER = "resumemind"
_BEARER_SCHEME = HTTPBearer(auto_error=False)


def _jwt_secret() -> str:
    return config.ADMIN_JWT_SECRET or hashlib.sha256(config.ADMIN_PASSWORD.encode("utf-8")).hexdigest()


def verify_admin_password(password: str) -> bool:
    """常量时间比较，避免时序侧信道。"""
    if not password:
        return False
    return hmac.compare_digest(
        password.encode("utf-8"),
        config.ADMIN_PASSWORD.encode("utf-8"),
    )


def create_access_token() -> tuple[str, datetime]:
    """签发管理员 token，返回 (token, expires_at)。"""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=config.ADMIN_TOKEN_EXPIRY_HOURS)
    payload = {
        "sub": "admin",
        "iat": now,
        "exp": expires_at,
        "iss": _TOKEN_ISSUER,
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm="HS256")
    return token, expires_at


def decode_access_token(token: str) -> dict | None:
    """校验并解码 token；任何异常（过期/篡改/格式错误）返回 None。"""
    try:
        payload = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            issuer=_TOKEN_ISSUER,
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.PyJWTError:
        return None
    return payload if payload.get("sub") == "admin" else None


def get_admin_identity(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER_SCHEME),
) -> dict[str, str]:
    """FastAPI dependency：校验管理员身份；AUTH_REQUIRED=false 时直接放行（开发用）。"""
    if not config.AUTH_REQUIRED:
        return {"role": "admin", "bypass": "true"}
    if credentials is None or decode_access_token(credentials.credentials) is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return {"role": "admin"}


require_admin = get_admin_identity


def is_admin_request(request: Request | None) -> bool:
    """可选鉴权：请求带有效管理员 token 时为 True（匿名/无 token 均不影响接口本身）。

    用于"默认公开、有 token 升级为完整视图"的接口（问答任务、SSE 等），
    与 get_admin_identity 的强制鉴权（fail-closed 的 401）互补。
    """
    if request is None:
        return False
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False
    return decode_access_token(auth[len("Bearer "):].strip()) is not None


# ---- 访客问答访问码闸：看过简历的面试官凭访问码提问，签发短期 JWT 存 httpOnly cookie ----

QA_ACCESS_COOKIE = "qa_access"
_QA_ACCESS_SUB = "qa_visitor"


def qa_access_enabled() -> bool:
    """访问码闸开关：QA_ACCESS_CODE 为空 = 关闭（开发/测试默认）。"""
    return bool(config.QA_ACCESS_CODE)


def verify_qa_access_code(code: str) -> bool:
    """校验访问码（常量时间比较，避免时序侧信道）。"""
    if not config.QA_ACCESS_CODE:
        return True
    if not code:
        return False
    return hmac.compare_digest(
        code.encode("utf-8"),
        config.QA_ACCESS_CODE.encode("utf-8"),
    )


def create_qa_access_token() -> tuple[str, datetime]:
    """签发访客问答 token（独立 sub，与管理员 token 互不通用），返回 (token, expires_at)。"""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=config.QA_ACCESS_TOKEN_TTL_HOURS)
    payload = {
        "sub": _QA_ACCESS_SUB,
        "iat": now,
        "exp": expires_at,
        "iss": _TOKEN_ISSUER,
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm="HS256")
    return token, expires_at


def decode_qa_access_token(token: str) -> dict | None:
    """校验并解码访客 token；任何异常（过期/篡改/格式错误）返回 None。"""
    try:
        payload = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            issuer=_TOKEN_ISSUER,
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.PyJWTError:
        return None
    return payload if payload.get("sub") == _QA_ACCESS_SUB else None


def has_qa_access(request: Request | None) -> bool:
    """访客问答访问闸：有效访客 token（cookie 或 Bearer）或管理员 token 均放行；
    未配置访问码时恒放行。"""
    if not qa_access_enabled():
        return True
    if request is None:
        return False
    token = request.cookies.get(QA_ACCESS_COOKIE, "")
    if not token:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[len("Bearer "):].strip()
    if token and decode_qa_access_token(token) is not None:
        return True
    return is_admin_request(request)
