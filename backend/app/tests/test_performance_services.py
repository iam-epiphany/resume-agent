from types import SimpleNamespace

import numpy as np

from qdrant_client import models

from backend.app.services import embedding_service, rerank_service, vector_store_service
from backend.app.services.embedding_service import TextEmbedding
from backend.app.services.performance_metrics import (
    ByteLRUCache,
    measure,
    trace_history_snapshot,
    trace_operation,
)
from backend.app.services.vector_store_service import VectorSearchResult


def _candidate(chunk_id: str = "chunk-1", text: str = "材料证据") -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="document-1",
        filename="rules.md",
        section_title="第一条",
        page_number=1,
        text=text,
        embedding_text=text,
        token_count=10,
        score=0.9,
    )


def test_byte_lru_cache_honors_item_limit_and_reports_hits() -> None:
    cache = ByteLRUCache(max_bytes=100, max_items=2, size_of=lambda value: len(value))
    cache.put("a", "123")
    cache.put("b", "456")
    assert cache.get("a") == "123"
    cache.put("c", "789")

    assert cache.get("b") is None
    snapshot = cache.snapshot()
    assert snapshot["items"] == 2
    assert snapshot["hits"] == 1
    assert snapshot["evictions"] == 1


def test_query_embedding_cache_batches_only_misses(monkeypatch) -> None:
    embedding_service._query_embedding_cache.cache_clear()
    calls: list[list[str]] = []

    def fake_embed_texts(texts, *, batch_size):
        calls.append(list(texts))
        return [TextEmbedding(dense=[float(index)]) for index, _ in enumerate(texts)]

    monkeypatch.setattr(embedding_service, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(embedding_service, "selected_model_device", lambda: "cpu")

    first = embedding_service.embed_queries(["同一问题", "另一个问题", "同一问题"])
    second = embedding_service.embed_queries(["同一问题"])

    assert len(first) == 3
    assert first[0] is first[2]
    assert len(second) == 1
    assert calls == [["同一问题", "另一个问题"]]
    assert embedding_service.embedding_runtime_status()["query_cache"]["hits"] >= 1
    embedding_service._query_embedding_cache.cache_clear()


def test_reranker_uses_cpu_profile_batch_and_reuses_scores(monkeypatch) -> None:
    rerank_service._rerank_score_cache.cache_clear()
    kwargs_seen: list[dict] = []

    class FakeReranker:
        def compute_score(self, pairs, **kwargs):
            kwargs_seen.append(kwargs)
            return [0.8 for _ in pairs]

    monkeypatch.setattr(rerank_service, "_get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(rerank_service, "selected_model_device", lambda: "cpu")
    candidates = [_candidate(f"chunk-{index}", f"证据-{index}") for index in range(6)]

    first = rerank_service.rerank_candidates(question="问题", candidates=candidates, limit=6)
    second = rerank_service.rerank_candidates(question="问题", candidates=candidates, limit=6)

    assert len(first) == len(second) == 6
    assert len(kwargs_seen) == 1
    assert kwargs_seen[0]["batch_size"] == 4
    assert kwargs_seen[0]["max_length"] == 1024
    assert rerank_service._rerank_score_cache().snapshot()["hits"] >= 6
    rerank_service._rerank_score_cache.cache_clear()


def test_rerank_score_cache_can_be_invalidated_after_index_change() -> None:
    rerank_service._rerank_score_cache.cache_clear()


def test_onnx_reranker_uses_session_logits_and_normalizes_scores(monkeypatch) -> None:
    rerank_service._rerank_score_cache.cache_clear()

    class FakeTokenizer:
        def __call__(self, questions, texts, **_kwargs):
            return {"input_ids": np.ones((len(questions), 2), dtype=np.int64)}

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input_ids")]

        def run(self, _outputs, inputs):
            assert "input_ids" in inputs
            return [np.asarray([[0.0], [np.log(3.0)]])]

    monkeypatch.setattr(rerank_service.config, "MODEL_BACKEND", "onnx")
    monkeypatch.setattr(rerank_service, "selected_model_device", lambda: "cpu")
    monkeypatch.setattr(
        rerank_service,
        "_get_onnx_reranker",
        lambda: SimpleNamespace(tokenizer=FakeTokenizer(), session=FakeSession(), model_path="model.onnx"),
    )
    candidates = [_candidate("chunk-a"), _candidate("chunk-b")]

    reranked = rerank_service.rerank_candidates(question="question", candidates=candidates, limit=2)

    assert [item.candidate.chunk_id for item in reranked] == ["chunk-b", "chunk-a"]
    assert [round(item.rerank_score, 2) for item in reranked] == [0.75, 0.5]
    rerank_service._rerank_score_cache.cache_clear()
    cache = rerank_service._rerank_score_cache()
    cache.put("score-key", 0.75)

    rerank_service.invalidate_rerank_score_cache()

    assert cache.snapshot()["items"] == 0
    rerank_service._rerank_score_cache.cache_clear()


def test_compact_rerank_text_keeps_critical_metadata_once(monkeypatch) -> None:
    monkeypatch.setattr(rerank_service.config, "RERANK_INPUT_MODE", "compact")
    candidate = _candidate(text="材料正文")
    candidate.filename = "简历材料.pdf"
    candidate.section_path = ["第二章", "第二章", "第十条"]
    candidate.section_number = "第十条"
    candidate.page_number = 8
    candidate.embedding_text = "内容类型：正文\n内容类型：正文\n材料正文"

    text = rerank_service._rerank_text(candidate)

    assert "来源文件：简历材料.pdf" in text
    assert "章节路径：第二章 > 第十条" in text
    assert "条款号：第十条" in text
    assert "页码：8" in text
    assert text.count("材料正文") == 1
    assert "内容类型：正文" not in text


def test_vector_collection_readiness_runs_once(monkeypatch) -> None:
    class FakeClient:
        def __init__(self):
            self.exists_calls = 0
            self.index_calls = 0

        def collection_exists(self, _name):
            self.exists_calls += 1
            return True

        def create_payload_index(self, **_kwargs):
            self.index_calls += 1

    client = FakeClient()
    monkeypatch.setattr(vector_store_service, "_qdrant", lambda: (client, models))
    vector_store_service.reset_vector_collection_readiness()

    vector_store_service.ensure_vector_collection()
    vector_store_service.ensure_vector_collection()

    assert client.exists_calls == 1
    # Core retrieval indexes plus resume-material provenance metadata indexes.
    assert client.index_calls == 13
    vector_store_service.reset_vector_collection_readiness()


def test_hybrid_search_batch_preserves_response_order(monkeypatch) -> None:
    class FakeClient:
        def query_batch_points(self, *, collection_name, requests):
            assert collection_name == vector_store_service.QDRANT_COLLECTION
            assert len(requests) == 2
            return [
                SimpleNamespace(points=[SimpleNamespace(payload=_payload("chunk-a"), score=0.9)]),
                SimpleNamespace(points=[SimpleNamespace(payload=_payload("chunk-b"), score=0.8)]),
            ]

    monkeypatch.setattr(vector_store_service, "ensure_vector_collection", lambda: None)
    monkeypatch.setattr(vector_store_service, "_qdrant", lambda: (FakeClient(), models))
    embeddings = [
        TextEmbedding(dense=[0.1] * 768),
        TextEmbedding(dense=[0.2] * 768),
    ]

    results = vector_store_service.hybrid_search_batch(embeddings, limit=3)

    assert [[item.chunk_id for item in group] for group in results] == [["chunk-a"], ["chunk-b"]]


def test_trace_operation_collects_nested_stage_timings() -> None:
    @trace_operation("test")
    def operation() -> str:
        with measure("test.stage"):
            return "done"

    assert operation() == "done"
    trace = trace_history_snapshot(limit=1)[0]
    assert trace["kind"] == "test"
    assert trace["status"] == "completed"
    assert trace["total_ms"] >= 0
    assert trace["stages"]["test.stage"]["count"] == 1


def _payload(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "document-1",
        "filename": "rules.md",
        "text": "材料证据",
        "embedding_text": "材料证据",
        "token_count": 10,
        "chunk_type": "paragraph",
        "index_version": vector_store_service.INDEX_VERSION,
    }
