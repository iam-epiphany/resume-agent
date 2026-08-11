import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.core import config
from backend.app.services import embedding_service, rerank_service
from backend.app.services.model_path_resolver import (
    ModelPathResolutionError,
    resolve_embedding_model_path,
    resolve_reranker_model_path,
)


def make_model_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("weights", encoding="utf-8")
    return path


def test_explicit_model_paths_have_priority(monkeypatch, tmp_path) -> None:
    embedding_path = make_model_dir(tmp_path / "explicit-embedding")
    reranker_path = make_model_dir(tmp_path / "explicit-reranker")
    default_embedding = make_model_dir(tmp_path / "default" / "bge-base-zh-v1.5")
    default_reranker = make_model_dir(tmp_path / "default" / "bge-reranker-base")

    monkeypatch.setattr(config, "RESUME_OFFLINE_MODE", True)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_PATH", str(embedding_path))
    monkeypatch.setattr(config, "RERANKER_MODEL_PATH", str(reranker_path))
    monkeypatch.setattr(config, "DEFAULT_EMBEDDING_MODEL_DIR", default_embedding)
    monkeypatch.setattr(config, "DEFAULT_RERANKER_MODEL_DIR", default_reranker)
    monkeypatch.setattr(config, "HF_HUB_CACHE", tmp_path / "missing-cache")

    assert resolve_embedding_model_path() == str(embedding_path)
    assert resolve_reranker_model_path() == str(reranker_path)


def test_default_model_dirs_are_used_before_hf_cache(monkeypatch, tmp_path) -> None:
    default_embedding = make_model_dir(tmp_path / "data" / "models" / "bge-base-zh-v1.5")
    default_reranker = make_model_dir(tmp_path / "data" / "models" / "bge-reranker-base")
    make_model_dir(tmp_path / "hf" / "models--BAAI--bge-base-zh-v1.5" / "snapshots" / "rev1")
    make_model_dir(tmp_path / "hf" / "models--BAAI--bge-reranker-base" / "snapshots" / "rev1")

    monkeypatch.setattr(config, "RESUME_OFFLINE_MODE", True)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_PATH", None)
    monkeypatch.setattr(config, "RERANKER_MODEL_PATH", None)
    monkeypatch.setattr(config, "DEFAULT_EMBEDDING_MODEL_DIR", default_embedding)
    monkeypatch.setattr(config, "DEFAULT_RERANKER_MODEL_DIR", default_reranker)
    monkeypatch.setattr(config, "HF_HUB_CACHE", tmp_path / "hf")

    assert resolve_embedding_model_path() == str(default_embedding)
    assert resolve_reranker_model_path() == str(default_reranker)


def test_hf_hub_cache_snapshot_is_used_when_default_missing(monkeypatch, tmp_path) -> None:
    embedding_snapshot = make_model_dir(
        tmp_path / "hf" / "models--BAAI--bge-small-zh-v1.5" / "snapshots" / "rev1"
    )
    reranker_snapshot = make_model_dir(
        tmp_path / "hf" / "models--BAAI--bge-reranker-base" / "snapshots" / "rev1"
    )

    monkeypatch.setattr(config, "RESUME_OFFLINE_MODE", True)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_PATH", None)
    monkeypatch.setattr(config, "RERANKER_MODEL_PATH", None)
    monkeypatch.setattr(config, "DEFAULT_EMBEDDING_MODEL_DIR", tmp_path / "missing" / "bge-base-zh-v1.5")
    monkeypatch.setattr(config, "DEFAULT_RERANKER_MODEL_DIR", tmp_path / "missing" / "reranker")
    monkeypatch.setattr(config, "HF_HUB_CACHE", tmp_path / "hf")

    assert resolve_embedding_model_path() == str(embedding_snapshot)
    assert resolve_reranker_model_path() == str(reranker_snapshot)


def test_offline_mode_raises_chinese_error_when_model_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "RESUME_OFFLINE_MODE", True)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_PATH", None)
    monkeypatch.setattr(config, "DEFAULT_EMBEDDING_MODEL_DIR", tmp_path / "data" / "models" / "bge-base-zh-v1.5")
    monkeypatch.setattr(config, "HF_HUB_CACHE", tmp_path / "missing-cache")

    with pytest.raises(ModelPathResolutionError) as exc_info:
        resolve_embedding_model_path()

    message = str(exc_info.value)
    assert "离线模式已开启" in message
    assert "data" in message
    assert "EMBEDDING_MODEL_PATH" in message


def test_online_fallback_allowed_only_when_offline_mode_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "RESUME_OFFLINE_MODE", False)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_PATH", None)
    monkeypatch.setattr(config, "DEFAULT_EMBEDDING_MODEL_DIR", tmp_path / "missing" / "bge-base-zh-v1.5")
    monkeypatch.setattr(config, "HF_HUB_CACHE", tmp_path / "missing-cache")
    monkeypatch.setattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-base-zh-v1.5")

    assert resolve_embedding_model_path() == "BAAI/bge-base-zh-v1.5"


def test_embedding_service_passes_local_path_to_flag_embedding(monkeypatch, tmp_path) -> None:
    model_path = make_model_dir(tmp_path / "models" / "bge-base-zh-v1.5")
    calls: list[str] = []

    class FakeFlagModel:
        def __init__(self, model_name_or_path: str, use_fp16: bool):
            calls.append(model_name_or_path)
            self.use_fp16 = use_fp16

    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(FlagModel=FakeFlagModel))
    monkeypatch.setattr(config, "RESUME_OFFLINE_MODE", True)
    monkeypatch.setattr(config, "EMBEDDING_MODEL_PATH", str(model_path))
    embedding_service._get_embedding_model.cache_clear()

    embedding_service._get_embedding_model()

    assert calls == [str(model_path)]
    assert calls[0] != "BAAI/bge-base-zh-v1.5"
    embedding_service._get_embedding_model.cache_clear()


def test_rerank_service_passes_local_path_to_flag_reranker(monkeypatch, tmp_path) -> None:
    model_path = make_model_dir(tmp_path / "models" / "bge-reranker-base")
    calls: list[str] = []

    class FakeFlagReranker:
        def __init__(self, model_name_or_path: str, use_fp16: bool):
            calls.append(model_name_or_path)
            self.use_fp16 = use_fp16

    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(FlagReranker=FakeFlagReranker))
    monkeypatch.setattr(config, "RESUME_OFFLINE_MODE", True)
    monkeypatch.setattr(config, "RERANKER_MODEL_PATH", str(model_path))
    rerank_service._get_reranker.cache_clear()

    rerank_service._get_reranker()

    assert calls == [str(model_path)]
    assert calls[0] != "BAAI/bge-reranker-base"
    rerank_service._get_reranker.cache_clear()
