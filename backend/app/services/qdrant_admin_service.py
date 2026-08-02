from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.core.config import QDRANT_URL


class QdrantAdminError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionStatus:
    name: str
    exists: bool
    status: str | None
    points_count: int


def collection_status(name: str) -> CollectionStatus:
    client, _ = _client_and_models()
    try:
        if not client.collection_exists(name):
            return CollectionStatus(name=name, exists=False, status=None, points_count=0)
        info = client.get_collection(name)
        status = getattr(info, "status", None)
        return CollectionStatus(
            name=name,
            exists=True,
            status=getattr(status, "value", str(status)) if status is not None else None,
            points_count=int(getattr(info, "points_count", 0) or 0),
        )
    except Exception as exc:
        raise QdrantAdminError(f"Qdrant collection status failed: {exc}") from exc


def promote_collection_alias(*, collection_name: str, alias_name: str) -> None:
    """Atomically point the public alias at a fully built collection."""

    if collection_name == alias_name:
        raise QdrantAdminError("build collection and public alias must be different")
    client, models = _client_and_models()
    status = collection_status(collection_name)
    if not status.exists or status.status not in {"green", "yellow"}:
        raise QdrantAdminError(
            f"build collection is not ready: exists={status.exists}, status={status.status}"
        )
    try:
        aliases = getattr(client.get_aliases(), "aliases", [])
        alias_names = {getattr(item, "alias_name", None) for item in aliases}
        if alias_name not in alias_names and client.collection_exists(alias_name):
            blocking = client.get_collection(alias_name)
            blocking_points = int(getattr(blocking, "points_count", 0) or 0)
            if blocking_points:
                raise QdrantAdminError(
                    f"public alias name is occupied by a non-empty physical collection: "
                    f"{alias_name} ({blocking_points} points)"
                )
            # A pre-ingestion query from an older build may have created an
            # empty collection under the public alias name. Removing only the
            # verified-empty collection is safe and frees the alias name.
            client.delete_collection(alias_name)
        operations: list[Any] = []
        if alias_name in alias_names:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection_name,
                    alias_name=alias_name,
                )
            )
        )
        client.update_collection_aliases(change_aliases_operations=operations)
        _invalidate_vector_readiness()
    except QdrantAdminError:
        raise
    except Exception as exc:
        raise QdrantAdminError(f"Qdrant alias promotion failed: {exc}") from exc


def delete_collection_if_exists(name: str) -> None:
    client, _ = _client_and_models()
    try:
        if client.collection_exists(name):
            client.delete_collection(name)
            _invalidate_vector_readiness()
    except Exception as exc:
        raise QdrantAdminError(f"Qdrant collection deletion failed: {exc}") from exc


def _client_and_models() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise QdrantAdminError("qdrant-client is not installed") from exc
    return QdrantClient(url=QDRANT_URL), models


def _invalidate_vector_readiness() -> None:
    from backend.app.services.vector_store_service import reset_vector_collection_readiness

    reset_vector_collection_readiness()
