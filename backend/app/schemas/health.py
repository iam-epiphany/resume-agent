from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查接口的统一响应结构。"""

    status: str
    message: str
    build_id: str = "dev"


class RagHealthResponse(BaseModel):
    build_id: str = "dev"
    offline_mode: bool
    embedding_model_ready: bool
    reranker_model_ready: bool
    embedding_model_path: str
    reranker_model_path: str
    qdrant_ready: bool
    qdrant_collection: str
    qdrant_collection_ready: bool = False
    sqlite_ready: bool = False
    libreoffice_ready: bool = False
    antiword_ready: bool = False
    libreoffice_version: str | None = None
    antiword_version: str | None = None
    index_tasks: dict[str, int] = Field(default_factory=dict)
    qa_tasks: dict[str, int] = Field(default_factory=dict)
    model_runtime: dict[str, object] = Field(default_factory=dict)
    ready: bool = False
    embedding_model_error: str | None = None
    reranker_model_error: str | None = None
    qdrant_error: str | None = None
    model_device: dict[str, object] = Field(default_factory=dict)
    performance: dict[str, object] = Field(default_factory=dict)
