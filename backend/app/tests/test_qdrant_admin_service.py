from types import SimpleNamespace

import pytest
from qdrant_client import models

from backend.app.services import qdrant_admin_service, vector_store_service
from backend.app.services.qdrant_admin_service import CollectionStatus, QdrantAdminError
from backend.app.services.vector_store_service import VectorStoreError


@pytest.fixture(autouse=True)
def reset_vector_collection_readiness() -> None:
    vector_store_service.reset_vector_collection_readiness()
    yield
    vector_store_service.reset_vector_collection_readiness()


class FakeQdrantAdminClient:
    def __init__(self, blocking_points: int):
        self.blocking_points = blocking_points
        self.deleted: list[str] = []
        self.operations = []

    def get_aliases(self):
        return SimpleNamespace(aliases=[])

    def collection_exists(self, name: str) -> bool:
        return True

    def get_collection(self, name: str):
        return SimpleNamespace(points_count=self.blocking_points)

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def update_collection_aliases(self, *, change_aliases_operations) -> None:
        self.operations = change_aliases_operations


def test_alias_promotion_removes_only_empty_blocking_collection(monkeypatch) -> None:
    client = FakeQdrantAdminClient(blocking_points=0)
    monkeypatch.setattr(qdrant_admin_service, "_client_and_models", lambda: (client, models))
    monkeypatch.setattr(
        qdrant_admin_service,
        "collection_status",
        lambda name: CollectionStatus(name=name, exists=True, status="green", points_count=100),
    )

    qdrant_admin_service.promote_collection_alias(
        collection_name="contest_build", alias_name="contest_public"
    )

    assert client.deleted == ["contest_public"]
    assert len(client.operations) == 1


def test_alias_promotion_preserves_nonempty_physical_collection(monkeypatch) -> None:
    client = FakeQdrantAdminClient(blocking_points=12)
    monkeypatch.setattr(qdrant_admin_service, "_client_and_models", lambda: (client, models))
    monkeypatch.setattr(
        qdrant_admin_service,
        "collection_status",
        lambda name: CollectionStatus(name=name, exists=True, status="green", points_count=100),
    )

    with pytest.raises(QdrantAdminError, match="non-empty physical collection"):
        qdrant_admin_service.promote_collection_alias(
            collection_name="contest_build", alias_name="contest_public"
        )

    assert client.deleted == []
    assert client.operations == []


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
        "sheet_name",
        "year",
        "month",
        "quarter",
        "external_doc_id",
        "issuing_authority",
        "publication_date",
        "effective_date",
        "document_number",
        "material_topic",
        "business_domain",
        "version_status",
        "article_number",
    }

    class FakePayloadIndexClient:
        def __init__(self) -> None:
            self.created: list[tuple[str, object, bool, int]] = []

        def collection_exists(self, name: str) -> bool:
            return True

        def get_collection(self, name: str) -> object:
            existing = index_names - {"article_number"}
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
    assert client.created[0][0] == "article_number"
    assert client.created[0][2:] == (True, 60)
