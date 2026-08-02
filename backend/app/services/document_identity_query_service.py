from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import Document
from backend.app.schemas.qa import QAResponse


# 简历场景意图词：面试官询问证书/材料的真实性与有效期。
# 不涉及银行场景的"废止/替代/版本状态"概念——个人证书不存在该语义。
_IDENTITY_TERMS = (
    "真实",
    "真假",
    "有效性",
    "有效吗",
    "有效期内",
    "过期",
    "是否有效",
    "靠不靠谱",
    "可信",
    "伪造",
    "造假",
)


def answer_document_identity_question(db: Session, question: str) -> QAResponse | None:
    """Answer explicit authenticity/validity questions from the document ledger.

    简历场景语义：确认知识库中是否存在对应材料、由谁颁发、是否注明有效期；
    系统不鉴定证书真伪（真伪以官方渠道核验为准），避免越权作保。
    """

    if not _is_identity_intent(question):
        return None
    title = _quoted_title(question)
    if not title:
        return None
    documents = list(db.scalars(select(Document).where(Document.status == "indexed")))
    ranked = sorted(
        (
            (_title_score(title, document), document)
            for document in documents
            if _title_score(title, document) > 0
        ),
        key=lambda item: (-item[0], item[1].document_id),
    )
    if not ranked:
        return QAResponse(
            answer=f"当前知识库中未找到与《{title}》对应的材料，无法核实其内容。",
            answer_mode="answered",
            generation_status="skipped",
        )
    top_score = ranked[0][0]
    top_documents = [document for score, document in ranked if score == top_score]
    if len(top_documents) != 1:
        return QAResponse(
            answer=f"《{title}》匹配到多份材料，当前无法唯一确定目标；请补充完整文件名。",
            answer_mode="answered",
            generation_status="skipped",
        )
    document = top_documents[0]
    issuer = (document.issuing_authority or "").strip()
    published = (document.publication_date or "").strip()
    expiry = (document.expiration_date or "").strip()
    provenance_parts: list[str] = []
    if issuer:
        provenance_parts.append(f"颁发机构为{issuer}")
    if published:
        provenance_parts.append(f"颁发时间为{published}")
    if expiry:
        provenance_parts.append(f"注明有效期为{expiry}")
    provenance = "，".join(provenance_parts) + "。" if provenance_parts else ""

    return QAResponse(
        answer=(
            f"《{title}》已收录在知识库材料中（来源文件：{document.filename}）。{provenance}"
            "系统按知识库材料作答，不对证书真伪作出鉴定——如需官方核验请通过颁发机构的正式渠道查询。"
        ),
        answer_mode="answered",
        generation_status="deterministic",
    )


def _quoted_title(question: str) -> str | None:
    match = re.search(r"《([^》]+)》", question)
    return match.group(1).strip() if match else None


def _is_identity_intent(question: str) -> bool:
    if any(term in question for term in _IDENTITY_TERMS):
        return True
    if "证书" not in question and "材料" not in question and "简历" not in question:
        return False
    return bool(
        re.search(r"(?:该|此|这份|证书|材料).{0,10}(?:真实性|有效性|有效)", question)
    )


def _title_score(title: str, document: Document) -> int:
    needle = _normalize(title)
    if not needle:
        return 0
    candidates = [document.filename, document.title or ""]
    best = 0
    for candidate in candidates:
        normalized = _normalize(candidate)
        if normalized == needle:
            best = max(best, 100)
        elif needle in normalized:
            best = max(best, 80 + min(19, len(needle)))
        elif normalized and normalized in needle:
            best = max(best, 60 + min(19, len(normalized)))
    return best


def _normalize(value: str) -> str:
    value = re.sub(r"^\d+_", "", value)
    value = re.sub(r"\.(?:docx?|pdf|xlsx?|csv|html?)$", "", value, flags=re.IGNORECASE)
    return re.sub(r"[\s_（）()《》【】\-—:：]+", "", value).casefold()
