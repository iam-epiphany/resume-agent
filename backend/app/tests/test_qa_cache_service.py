"""问答答案缓存服务测试：归一化 / 精确与语义命中 / 阈值 / 签名失效 / 清空 / LRU。"""

import pytest

from backend.app.core import config
from backend.app.core.database import Base
from backend.app.models.document import QAAnswerCache
from backend.app.schemas.qa import QAResponse
from backend.app.services import qa_cache_service
from backend.app.services.qa_cache_service import normalize_question


@pytest.fixture()
def test_db(monkeypatch):
    """独立内存库（StaticPool 共享连接，避免 Windows 文件锁）+ 模块级缓存复位。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    test_session = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    # clear() 的 SessionLocal 指向测试库：清空只影响测试数据，不碰真实 app.db
    monkeypatch.setattr(qa_cache_service, "SessionLocal", test_session)
    qa_cache_service.clear()
    with test_session() as db:
        yield db
    qa_cache_service.clear()


def make_response(answer: str = "我是张三。") -> QAResponse:
    return QAResponse(
        answer=answer,
        answer_mode="answered",
        evidence_sufficiency="sufficient",
        llm_call_count=2,
    )


class TestNormalize:
    def test_strips_whitespace_punctuation_and_fullwidth(self) -> None:
        assert normalize_question("  请介绍 一下，你 自己？！") == "请介绍一下你自己"
        assert normalize_question("请介绍一下你自己。") == "请介绍一下你自己"
        assert normalize_question("ＡＢＣｄｅｆ") == "abcdef"
        assert normalize_question("What is your tech stack?") == "whatisyourtechstack"

    def test_keeps_discriminating_words(self) -> None:
        # 面试问题的"怎么/为什么/哪些/多少"是关键差异词，归一化不得删除
        assert normalize_question("秒杀怎么防超卖") != normalize_question("秒杀为什么用Lua")
        assert normalize_question("你有什么证书") != normalize_question("英语六级多少分")


class TestLookup:
    def test_exact_match_hits(self, test_db) -> None:
        qa_cache_service.store(test_db, "请介绍一下你自己", make_response())
        hit = qa_cache_service.lookup(test_db, " 请介绍一下，你 自己？")
        assert hit is not None
        assert hit.answer == "我是张三。"
        assert hit.cached is False  # 命中后由 rag_service 置位，服务本身不改

    def test_exact_match_increments_hit_count(self, test_db) -> None:
        qa_cache_service.store(test_db, "你的技术栈是什么", make_response())
        qa_cache_service.lookup(test_db, "你的技术栈是什么")
        qa_cache_service.lookup(test_db, "你的技术栈是什么")
        row = test_db.scalar(
            __import__("sqlalchemy").select(QAAnswerCache).where(
                QAAnswerCache.norm_question == "你的技术栈是什么"
            )
        )
        assert row.hit_count == 2

    def test_exact_match_survives_embedding_failure(self, test_db, monkeypatch) -> None:
        monkeypatch.setattr(qa_cache_service, "embed_query", lambda question: (_ for _ in ()).throw(RuntimeError("offline")))
        qa_cache_service.store(test_db, "请介绍一下你自己", make_response())
        hit = qa_cache_service.lookup(test_db, "请介绍一下你自己")
        assert hit is not None
        assert hit.answer == "我是张三。"

    def test_semantic_match_hits_above_threshold(self, test_db, monkeypatch) -> None:
        # 模拟同义改写：向量相似度 0.98 ≥ 0.93
        vectors = {"介绍一下你自己": [1.0, 0.0], "请做自我介绍": [0.98, 0.02]}
        monkeypatch.setattr(
            qa_cache_service,
            "embed_query",
            lambda question: type("E", (), {"dense": vectors[question]})(),
        )
        qa_cache_service.store(test_db, "介绍一下你自己", make_response())
        hit = qa_cache_service.lookup(test_db, "请做自我介绍")
        assert hit is not None
        assert hit.answer == "我是张三。"

    def test_semantic_miss_below_threshold(self, test_db, monkeypatch) -> None:
        vectors = {"介绍一下你自己": [1.0, 0.0], "秒杀怎么防超卖": [0.6, 0.8]}
        monkeypatch.setattr(
            qa_cache_service,
            "embed_query",
            lambda question: type("E", (), {"dense": vectors[question]})(),
        )
        qa_cache_service.store(test_db, "介绍一下你自己", make_response())
        assert qa_cache_service.lookup(test_db, "秒杀怎么防超卖") is None

    def test_different_signature_naturally_misses(self, test_db, monkeypatch) -> None:
        qa_cache_service.store(test_db, "你的优缺点是什么", make_response())
        monkeypatch.setattr(config, "INDEX_VERSION", "other-version")
        monkeypatch.setattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-other")
        assert qa_cache_service.lookup(test_db, "你的优缺点是什么") is None

    def test_disabled_returns_none_and_skips_write(self, test_db, monkeypatch) -> None:
        monkeypatch.setattr(config, "QA_CACHE_ENABLED", False)
        qa_cache_service.store(test_db, "你的项目经历", make_response())
        assert qa_cache_service.lookup(test_db, "你的项目经历") is None
        assert qa_cache_service.cache_status()["items"] == 0


class TestStoreLifecycle:
    def test_clear_empties_cache(self, test_db) -> None:
        qa_cache_service.store(test_db, "你的职业规划", make_response())
        assert qa_cache_service.lookup(test_db, "你的职业规划") is not None
        qa_cache_service.clear()
        assert qa_cache_service.lookup(test_db, "你的职业规划") is None

    def test_lru_evicts_oldest_when_full(self, test_db, monkeypatch) -> None:
        monkeypatch.setattr(config, "QA_CACHE_MAX_ITEMS", 2)
        qa_cache_service.store(test_db, "问题一", make_response("一"))
        qa_cache_service.store(test_db, "问题二", make_response("二"))
        qa_cache_service.store(test_db, "问题三", make_response("三"))
        # 最旧（问题一）被淘汰，其余仍可精确命中
        assert qa_cache_service.lookup(test_db, "问题一") is None
        assert qa_cache_service.lookup(test_db, "问题二") is not None
        assert qa_cache_service.lookup(test_db, "问题三") is not None

    def test_store_overwrites_same_question(self, test_db) -> None:
        qa_cache_service.store(test_db, "你的优势是什么", make_response("旧答案"))
        qa_cache_service.store(test_db, "你的优势是什么", make_response("新答案"))
        hit = qa_cache_service.lookup(test_db, "你的优势是什么")
        assert hit is not None
        assert hit.answer == "新答案"


class TestAnswerQuestionCacheBranch:
    """answer_question 命中缓存分支：秒回、零 LLM、进度事件、审计日志。"""

    def test_cache_hit_short_circuits_with_zero_llm_calls(self, test_db, monkeypatch) -> None:
        from backend.app.services import rag_service

        cached = make_response("（缓存答案）技术栈是 Java。")
        monkeypatch.setattr(rag_service.qa_cache_service, "lookup", lambda db, q: cached)
        events: list[dict] = []

        response = rag_service.answer_question(
            test_db,
            "你的技术栈是什么",
            progress_reporter=events.append,
        )

        assert response.cached is True
        assert response.llm_call_count == 0
        assert response.answer == "（缓存答案）技术栈是 Java。"
        assert any(event.get("stage") == "cache" for event in events)

    def test_cache_hit_writes_qa_log(self, test_db, monkeypatch) -> None:
        from sqlalchemy import func, select

        from backend.app.models.document import QALog
        from backend.app.services import rag_service

        monkeypatch.setattr(
            rag_service.qa_cache_service,
            "lookup",
            lambda db, q: make_response("我是张三，河南大学本科。"),
        )
        rag_service.answer_question(test_db, "介绍一下你自己")
        count = test_db.scalar(select(func.count()).select_from(QALog))
        assert count == 1

    def test_session_questions_skip_cache_lookup(self, test_db, monkeypatch) -> None:
        from backend.app.services import rag_service

        called = {"lookup": 0}
        monkeypatch.setattr(
            rag_service.qa_cache_service,
            "lookup",
            lambda db, q: (called.__setitem__("lookup", called["lookup"] + 1) or None),
        )
        # 有 session 的追问链：不走缓存查询，直接进入完整链路（intent 会真实调用，
        # 但这里只断言 lookup 未被调用——answer_question 需要模型环境，用快速失败问题）
        monkeypatch.setattr(rag_service, "classify_and_resolve", lambda q, p: None)
        # 通过抛异常打断全链路，验证 lookup 未命中
        monkeypatch.setattr(rag_service, "_report_progress", lambda *a, **k: None)
        monkeypatch.setattr(rag_service, "recent_turns", lambda db, s, limit: [])
        try:
            with pytest.raises(Exception):
                rag_service.answer_question(test_db, "那超卖是怎么解决的", session_id="session-12345678")
        except TypeError:
            pass  # intent 返回 None 导致后续异常，说明未走缓存
        assert called["lookup"] == 0
