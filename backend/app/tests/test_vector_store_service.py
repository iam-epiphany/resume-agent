from types import SimpleNamespace

import pytest
from qdrant_client import models

from backend.app.services import vector_store_service
from backend.app.services.vector_store_service import VectorStoreError


@pytest.fixture(autouse=True)
def reset_vector_collection_readiness() -> None:
    vector_store_service.reset_vector_collection_readiness()
    yield
    vector_store_service.reset_vector_collection_readiness()


def test_formal_app_does_not_auto_create_missing_public_collection(monkeypatch) -> None:
    client = SimpleNamespace(collection_exists=lambda name: False)
    monkeypatch.setattr(vector_store_service, "QDRANT_AUTO_CREATE_COLLECTION", False)
    monkeypatch.setattr(vector_store_service, "_qdrant", lambda: (client, models))

    with pytest.raises(VectorStoreError, match="尚未发布"):
        vector_store_service.ensure_vector_collection()


def test_existing_collection_only_creates_missing_payload_indexes(monkeypatch) -> None:
    index_names = {
        "document_id",
        "index_version",
        "source_file",
        "chunk_type",
        "external_doc_id",
        "issuing_authority",
        "publication_date",
        "expiration_date",
        "document_number",
        "material_topic",
        "source_url",
        "attachment_url",
        "source_type",
    }

    class FakePayloadIndexClient:
        def __init__(self) -> None:
            self.created: list[tuple[str, object, bool, int]] = []

        def collection_exists(self, name: str) -> bool:
            return True

        def get_collection(self, name: str) -> object:
            existing = index_names - {"material_topic"}
            return SimpleNamespace(payload_schema={key: object() for key in existing})

        def create_payload_index(
            self,
            *,
            collection_name: str,
            field_name: str,
            field_schema: object,
            wait: bool,
            timeout: int,
        ) -> None:
            self.created.append((field_name, field_schema, wait, timeout))

    client = FakePayloadIndexClient()
    monkeypatch.setattr(vector_store_service, "_qdrant", lambda: (client, models))

    vector_store_service.ensure_vector_collection(force=True)

    assert len(client.created) == 1
    assert client.created[0][0] == "material_topic"
    assert client.created[0][2:] == (True, 60)
