"""问答答案缓存：面试高频问题的相似语义复用（2026-08-12）。

两层判定，防"语义相似 ≠ 答案可复用"的张冠李戴：
1. 精确匹配：问题归一化（全角→半角、去标点空白；不去停用词——面试问题里的
   "怎么/为什么/哪些/多少"是关键差异词）后完全相等 → 直接命中（最安全）；
2. 语义匹配：问题 embedding 与缓存向量做余弦，top-1 ≥ QA_CACHE_SEMANTIC_THRESHOLD
   （默认 0.93，保守）→ 命中。同义改写通常 0.85-0.95，不同问题 < 0.8。

只缓存独立问题（无 session）的 answered+sufficient 答案（由 rag_service 保证条件）；
知识库变更（上传/删除/重建索引）时整体清空（失效挂点在文档服务中）。
缓存键隐含模型签名（INDEX_VERSION + embedding 模型 + LLM 模型），签名变化自然 miss。

存储：SQLite 表 qa_answer_cache（重启不丢）+ 进程内 numpy 向量（O(N) 余弦微秒级），
LRU 按 updated_at 淘汰（QA_CACHE_MAX_ITEMS）。
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from threading import RLock
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from backend.app.core import config
from backend.app.core.database import SessionLocal
from backend.app.models.document import QAAnswerCache
from backend.app.schemas.qa import QAResponse
from backend.app.services.embedding_service import embed_query


_VECTORS: dict[tuple[str, str], np.ndarray] = {}
_LOADED = False
_LOCK = RLock()


def normalize_question(question: str) -> str:
    """归一化问题：NFKC 全角→半角、小写、去掉所有非字词字符（空白与标点）。"""
    text = unicodedata.normalize("NFKC", question).strip().casefold()
    return re.sub(r"[^\w]+", "", text)


def model_signature() -> str:
    """模型签名：索引版本 + embedding 模型 + LLM 模型。变化即缓存自然失效。"""
    embedding_identity = config.EMBEDDING_MODEL_PATH or config.EMBEDDING_MODEL_NAME
    return f"{config.INDEX_VERSION}|{embedding_identity}|{config.LLM_MODEL}"


def lookup(db: Session, question: str) -> QAResponse | None:
    """查询缓存：先精确匹配，再高阈值语义匹配；均未命中返回 None。"""
    if not config.QA_CACHE_ENABLED:
        return None
    norm = normalize_question(question)
    if not norm:
        return None
    signature = model_signature()
    # 精确命中不应依赖 embedding：模型预热失败或临时不可用时，仍可安全复用同一个问题。
    exact = _load_answer(db, signature, norm)
    if exact is not None:
        _bump_hit(db, signature, norm)
        return exact

    _ensure_loaded(db, signature)
    with _LOCK:
        query_vector = _embed_vector(question)
        if query_vector is None:
            return None
        best: tuple[tuple[str, str] | None, float] = (None, 0.0)
        for key, vector in _VECTORS.items():
            if key[0] != signature:
                continue
            similarity = float(np.dot(query_vector, vector))
            if similarity > best[1]:
                best = (key, similarity)
        if best[0] is not None and best[1] >= config.QA_CACHE_SEMANTIC_THRESHOLD:
            _bump_hit(db, best[0][0], best[0][1])
            return _load_answer(db, best[0][0], best[0][1])
    return None


def store(db: Session, question: str, response: QAResponse) -> None:
    """写入缓存（upsert）；超限时按 updated_at 淘汰最旧条目。调用方保证缓存条件。"""
    if not config.QA_CACHE_ENABLED:
        return
    norm = normalize_question(question)
    if not norm:
        return
    signature = model_signature()
    vector = _embed_vector(question)
    _ensure_loaded(db, signature)
    with _LOCK:
        key = (signature, norm)
        row = db.scalar(
            select(QAAnswerCache).where(
                QAAnswerCache.model_signature == signature,
                QAAnswerCache.norm_question == norm,
            )
        )
        item_count = int(
            db.scalar(
                select(func.count()).select_from(QAAnswerCache).where(
                    QAAnswerCache.model_signature == signature
                )
            )
            or 0
        )
        if row is None and item_count >= config.QA_CACHE_MAX_ITEMS:
            _evict_oldest(db, signature)
        now = datetime.now(timezone.utc)
        embedding_json = json.dumps(vector.tolist(), ensure_ascii=False) if vector is not None else "[]"
        if row is None:
            db.add(
                QAAnswerCache(
                    model_signature=signature,
                    norm_question=norm,
                    question=question,
                    embedding_json=embedding_json,
                    answer_json=response.model_dump_json(),
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            row.question = question
            row.embedding_json = embedding_json
            row.answer_json = response.model_dump_json()
            row.updated_at = now
        db.commit()
        if vector is None:
            _VECTORS.pop(key, None)
        else:
            _VECTORS[key] = vector


def clear() -> None:
    """清空缓存：知识库变更（上传/删除/重建索引）时由文档服务调用。"""
    with _LOCK:
        _VECTORS.clear()
        _LOADED = False
    try:
        with SessionLocal() as db:
            db.execute(delete(QAAnswerCache))
            db.commit()
    except Exception:
        # 清空失败不阻塞业务：内存已清，下次查询走全链路
        pass


def cache_status() -> dict[str, Any]:
    """运行状态（后台诊断用）：条目数、命中统计。"""
    with _LOCK:
        return {
            "enabled": config.QA_CACHE_ENABLED,
            "items": len(_VECTORS),
            "max_items": config.QA_CACHE_MAX_ITEMS,
            "semantic_threshold": config.QA_CACHE_SEMANTIC_THRESHOLD,
        }


def _ensure_loaded(db: Session, signature: str) -> None:
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        rows = db.scalars(select(QAAnswerCache)).all()
        for row in rows:
            try:
                values = json.loads(row.embedding_json)
                vector = np.asarray(values, dtype=np.float64)
                if vector.size:
                    _VECTORS[(row.model_signature, row.norm_question)] = vector
            except (ValueError, TypeError):
                continue
        _LOADED = True


def _embed_vector(question: str) -> np.ndarray | None:
    try:
        return np.asarray(embed_query(question).dense, dtype=np.float64)
    except Exception:
        return None


def _load_answer(db: Session, signature: str, norm: str) -> QAResponse | None:
    row = db.scalar(
        select(QAAnswerCache).where(
            QAAnswerCache.model_signature == signature,
            QAAnswerCache.norm_question == norm,
        )
    )
    if row is None:
        return None
    try:
        return QAResponse.model_validate_json(row.answer_json)
    except ValueError:
        return None


def _bump_hit(db: Session, signature: str, norm: str) -> None:
    try:
        db.execute(
            update(QAAnswerCache)
            .where(
                QAAnswerCache.model_signature == signature,
                QAAnswerCache.norm_question == norm,
            )
            .values(
                hit_count=QAAnswerCache.hit_count + 1,
                updated_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
    except Exception:
        db.rollback()


def _evict_oldest(db: Session, signature: str) -> None:
    row = db.scalar(
        select(QAAnswerCache)
        .where(QAAnswerCache.model_signature == signature)
        .order_by(QAAnswerCache.updated_at.asc())
        .limit(1)
    )
    if row is None:
        return
    db.delete(row)
    db.commit()
    _VECTORS.pop((row.model_signature, row.norm_question), None)
