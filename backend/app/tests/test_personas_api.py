# -*- coding: utf-8 -*-
"""人物档案 API 测试（2026-08-15）：人物 Skill 包下载（管理员鉴权）与元信息字段。

注意：不重绑全局 SessionLocal（test_rag_api 在模块导入期重绑它；谁后导入谁生效，
再增加一个重绑者会互相踩踏）。本文件改用 app.dependency_overrides[get_db]
按测试隔离自己的临时库，测试结束即移除覆盖——任何执行顺序下都互不影响。
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base, get_db
from backend.app.models.document import Persona
from backend.app.services.persona_skill_service import regenerate_persona_skill
from backend.main import app

client = TestClient(app)
_admin_login = client.post("/api/auth/login", json={"password": os.environ["ADMIN_PASSWORD"]})
assert _admin_login.status_code == 200, f"admin login failed: {_admin_login.text}"
client.headers.update({"Authorization": f"Bearer {_admin_login.json()['token']}"})

TEST_DATABASE_PATH = Path(tempfile.gettempdir()) / f"resumemind-personas-test-{os.getpid()}.db"
test_engine = create_engine(
    f"sqlite:///{TEST_DATABASE_PATH.as_posix()}", connect_args={"check_same_thread": False}
)
test_session_factory = sessionmaker(bind=test_engine)


@pytest.fixture()
def api_db():
    """按测试重建临时库，并把应用的 get_db 依赖指向该库（结束即还原）。"""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db() -> Generator:
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


def _seed_persona(persona_id: str, name: str, with_package: bool) -> None:
    with test_session_factory() as db:
        persona = Persona(
            persona_id=persona_id,
            name=name,
            display_name=name,
            profile_json=json.dumps({"name": name, "summary": "测试人物摘要。"}, ensure_ascii=False),
            status="confirmed",
            is_active=False,
        )
        db.add(persona)
        db.commit()
        if with_package:
            assert regenerate_persona_skill(db, persona_id) is True


def test_download_skill_package_requires_admin(api_db) -> None:
    """下载端点独立挂 require_admin：未登录返回 401（包内含全部个人材料）。"""
    anonymous = TestClient(app)
    _seed_persona("persona-张三", "张三", with_package=True)
    response = anonymous.get("/api/personas/persona-张三/skill-package")
    assert response.status_code == 401


def test_download_skill_package_returns_zip(api_db) -> None:
    _seed_persona("persona-张三", "张三", with_package=True)
    response = client.get("/api/personas/persona-张三/skill-package")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment")
    assert "persona-%E5%BC%A0%E4%B8%89.skill.zip" in disposition  # RFC 5987 UTF-8 文件名
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert {
            "persona-张三/SKILL.md",
            "persona-张三/facts.json",
            "persona-张三/references/profile.md",
        } <= names
        skill_md = archive.read("persona-张三/SKILL.md").decode("utf-8")
        assert "张三" in skill_md and "{{" not in skill_md


def test_download_skill_package_without_package_404(api_db) -> None:
    _seed_persona("persona-张三", "张三", with_package=False)
    response = client.get("/api/personas/persona-张三/skill-package")
    assert response.status_code == 404
    assert "人物 Skill 包" in response.json()["detail"]


def test_download_skill_package_unknown_persona_404(api_db) -> None:
    response = client.get("/api/personas/persona-不存在/skill-package")
    assert response.status_code == 404


def test_persona_list_includes_skill_package_info(api_db) -> None:
    """persona 响应带 skill_package 元信息（无包为 null，有包含文件数/版本/时间）。"""
    _seed_persona("persona-张三", "张三", with_package=True)
    _seed_persona("persona-新人", "新人", with_package=False)
    response = client.get("/api/personas")
    assert response.status_code == 200
    by_id = {item["persona_id"]: item for item in response.json()}
    package_info = by_id["persona-张三"]["skill_package"]
    assert package_info["file_count"] == 3
    assert package_info["skill_version"]
    assert package_info["generated_at"]
    assert by_id["persona-新人"]["skill_package"] is None
