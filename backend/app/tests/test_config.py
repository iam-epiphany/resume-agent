import pytest

from backend.app.core import config


def test_env_int_reports_variable_name(monkeypatch) -> None:
    monkeypatch.setenv("RESUME_TEST_INTEGER", "not-a-number")

    with pytest.raises(RuntimeError, match="RESUME_TEST_INTEGER must be an integer"):
        config._env_int("RESUME_TEST_INTEGER", 1)


def test_env_int_enforces_minimum(monkeypatch) -> None:
    monkeypatch.setenv("RESUME_TEST_INTEGER", "0")

    with pytest.raises(RuntimeError, match="must be >= 1"):
        config._env_int("RESUME_TEST_INTEGER", 1, minimum=1)
