import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.api import health
from backend.app.services import model_path_resolver


def _write_minimal_model_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"test")


def test_local_model_resolution_does_not_treat_online_name_as_ready(tmp_path, monkeypatch) -> None:
    missing_default = tmp_path / "missing-bge-base-zh-v1.5"
    monkeypatch.setattr(model_path_resolver.config, "EMBEDDING_MODEL_PATH", None)
    monkeypatch.setattr(model_path_resolver.config, "DEFAULT_EMBEDDING_MODEL_DIR", missing_default)
    monkeypatch.setattr(model_path_resolver.config, "HF_HUB_CACHE", tmp_path / "hub")
    monkeypatch.setattr(model_path_resolver.config, "EMBEDDING_MODEL_NAME", "BAAI/bge-base-zh-v1.5")

    with pytest.raises(model_path_resolver.ModelPathResolutionError):
        model_path_resolver.resolve_embedding_model_local_path()


def test_local_model_resolution_accepts_downloaded_model_dir(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "bge-base-zh-v1.5"
    _write_minimal_model_dir(model_dir)
    monkeypatch.setattr(model_path_resolver.config, "EMBEDDING_MODEL_PATH", None)
    monkeypatch.setattr(model_path_resolver.config, "DEFAULT_EMBEDDING_MODEL_DIR", model_dir)
    monkeypatch.setattr(model_path_resolver.config, "HF_HUB_CACHE", tmp_path / "hub")

    assert model_path_resolver.resolve_embedding_model_local_path() == str(model_dir)


def test_onnx_reranker_resolution_requires_model_and_tokenizer(tmp_path, monkeypatch) -> None:
    artifact_dir = tmp_path / "reranker-onnx"
    artifact_dir.mkdir()
    monkeypatch.setattr(model_path_resolver.config, "RERANKER_ONNX_MODEL_PATH", artifact_dir / "model.onnx")

    with pytest.raises(model_path_resolver.ModelPathResolutionError):
        model_path_resolver.resolve_reranker_onnx_model_local_path()

    (artifact_dir / "model.onnx").write_bytes(b"onnx")
    (artifact_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert model_path_resolver.resolve_reranker_onnx_model_local_path() == str(artifact_dir / "model.onnx")


def test_qdrant_service_available_even_when_collection_is_empty(monkeypatch) -> None:
    class FakeQdrantClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def get_collections(self) -> list[object]:
            return []

        def collection_exists(self, collection: str) -> bool:
            return True

        def get_collection(self, collection: str) -> object:
            return SimpleNamespace(status=SimpleNamespace(value="green"), points_count=0)

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=FakeQdrantClient))

    assert health._check_qdrant() == (True, False, None)


def test_qdrant_ready_check_initializes_payload_indexes(monkeypatch) -> None:
    class FakeQdrantClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def get_collections(self) -> list[object]:
            return []

        def collection_exists(self, collection: str) -> bool:
            return True

        def get_collection(self, collection: str) -> object:
            return SimpleNamespace(status=SimpleNamespace(value="green"), points_count=10)

    calls: list[bool] = []
    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=FakeQdrantClient))
    monkeypatch.setattr(health, "ensure_vector_collection", lambda: calls.append(True))

    assert health._check_qdrant() == (True, True, None)
    assert calls == [True]


def test_qdrant_payload_index_failure_marks_collection_unready(monkeypatch) -> None:
    class FakeQdrantClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def get_collections(self) -> list[object]:
            return []

        def collection_exists(self, collection: str) -> bool:
            return True

        def get_collection(self, collection: str) -> object:
            return SimpleNamespace(status=SimpleNamespace(value="green"), points_count=10)

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=FakeQdrantClient))

    def fail_initialization() -> None:
        raise RuntimeError("payload index migration failed")

    monkeypatch.setattr(health, "ensure_vector_collection", fail_initialization)

    assert health._check_qdrant() == (True, False, "payload index migration failed")


def test_qdrant_unavailable_reports_error(monkeypatch) -> None:
    class FakeQdrantClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def get_collections(self) -> None:
            raise RuntimeError("connection refused")

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=FakeQdrantClient))

    qdrant_ready, collection_ready, error = health._check_qdrant()

    assert qdrant_ready is False
    assert collection_ready is False
    assert "connection refused" in str(error)


def test_rag_ready_does_not_wait_for_background_warmup(monkeypatch) -> None:
    monkeypatch.setattr(health, "_resolve_model_for_health", lambda **kwargs: (True, "model-path", None))
    monkeypatch.setattr(health, "_check_qdrant", lambda: (True, True, None))
    monkeypatch.setattr(health, "_check_sqlite", lambda: True)
    monkeypatch.setattr(
        health,
        "office_tool_status",
        lambda: {
            "libreoffice_ready": True,
            "antiword_ready": True,
            "libreoffice_version": "24.2",
            "antiword_version": "0.37",
        },
    )
    monkeypatch.setattr(health, "warmup_status", lambda: {"state": "not_started", "warmed": False})
    monkeypatch.setattr(
        health,
        "get_model_device_info",
        lambda: SimpleNamespace(selected_device="cpu", to_debug_dict=lambda: {"selected_device": "cpu"}),
    )
    monkeypatch.setattr(
        health,
        "resolve_performance_profile",
        lambda selected_device: SimpleNamespace(
            warmup_policy="background",
            to_dict=lambda: {"warmup_policy": "background"},
        ),
    )
    monkeypatch.setattr(health, "_model_runtime_status", lambda: {})
    monkeypatch.setattr(health, "index_task_status_counts", lambda: {})
    monkeypatch.setattr(health, "qa_task_status_counts", lambda: {})

    payload = health._rag_health()

    assert payload.ready is True
    assert payload.performance["warmup"]["warmed"] is False


def test_rag_ready_is_true_even_without_libreoffice(monkeypatch) -> None:
    """LibreOffice 已从镜像移除（瘦身），ready 判定不再依赖它。"""

    monkeypatch.setattr(health, "_resolve_model_for_health", lambda **kwargs: (True, "model-path", None))
    monkeypatch.setattr(health, "_check_qdrant", lambda: (True, True, None))
    monkeypatch.setattr(health, "_check_sqlite", lambda: True)
    monkeypatch.setattr(
        health,
        "office_tool_status",
        lambda: {
            "libreoffice_ready": False,
            "antiword_ready": True,
            "libreoffice_version": "",
            "antiword_version": "0.37",
        },
    )
    monkeypatch.setattr(health, "warmup_status", lambda: {"state": "not_started", "warmed": False})
    monkeypatch.setattr(
        health,
        "get_model_device_info",
        lambda: SimpleNamespace(selected_device="cpu", to_debug_dict=lambda: {"selected_device": "cpu"}),
    )
    monkeypatch.setattr(
        health,
        "resolve_performance_profile",
        lambda selected_device: SimpleNamespace(
            warmup_policy="background",
            to_dict=lambda: {"warmup_policy": "background"},
        ),
    )
    monkeypatch.setattr(health, "_model_runtime_status", lambda: {})
    monkeypatch.setattr(health, "index_task_status_counts", lambda: {})
    monkeypatch.setattr(health, "qa_task_status_counts", lambda: {})

    payload = health._rag_health()

    assert payload.ready is True
    assert payload.libreoffice_ready is False
