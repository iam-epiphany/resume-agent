from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from backend.app.models.document import Document
from backend.app.services.document_types import ParsedDocument


# 简历场景的材料身份字段（7 项核心 + external_doc_id/source_type/attachment_url：
# external_doc_id 用于清单去重与覆盖更新，source_type 由上传/URL 导入路径引用，
# attachment_url 由 url-import 写入并被检索元数据读取，均保留）。
CORE_METADATA_FIELDS = (
    "external_doc_id",
    "title",
    "issuing_authority",
    "publication_date",
    "expiration_date",
    "document_number",
    "material_topic",
    "source_url",
    "attachment_url",
    "source_type",
    "metadata_status",
)

RETRIEVAL_METADATA_FIELDS = (
    "external_doc_id",
    "title",
    "issuing_authority",
    "publication_date",
    "expiration_date",
    "document_number",
    "material_topic",
    "source_url",
    "attachment_url",
    "source_type",
)

IDENTITY_METADATA_FIELDS = (
    "external_doc_id",
    "title",
    "issuing_authority",
    "publication_date",
    "expiration_date",
    "document_number",
    "material_topic",
    "source_url",
    "attachment_url",
    "source_type",
)

FIELD_ALIASES = {
    "doc_id": "external_doc_id",
    "official_doc_id": "external_doc_id",
    "source_title": "title",
    "document_title": "title",
    "authority": "issuing_authority",
    "issuer": "issuing_authority",
    "publish_date": "publication_date",
    "published_at": "publication_date",
    "expiry_date": "expiration_date",
    "invalid_date": "expiration_date",
    "file_number": "document_number",
    "topic": "material_topic",
    "category": "material_topic",
    "source_page_url": "source_url",
    "page_url": "source_url",
    "url": "source_url",
    "download_url": "attachment_url",
    "file_url": "attachment_url",
    "标题": "title",
    "颁发机构": "issuing_authority",
    "发证日期": "publication_date",
    "到期日期": "expiration_date",
    "编号": "document_number",
    "材料主题": "material_topic",
    "来源页面URL": "source_url",
    "来源URL": "source_url",
    "附件URL": "attachment_url",
}

SOURCE_PRIORITIES = {
    "user": 100,
    "manifest": 95,
    "official_url": 90,
    "document_body": 60,
    "parser": 50,
    "filename": 20,
    "legacy": 10,
}

KNOWN_AUTHORITIES = (
    "河南大学",
    "教育部",
    "人力资源和社会保障部",
    "工业和信息化部",
    "共青团中央",
    "河南省教育厅",
    "蓝桥杯大赛组委会",
)

TOPIC_RULES = (
    ("竞赛奖项", ("竞赛", "奖项", "获奖", "一等奖", "蓝桥杯")),
    ("荣誉奖励", ("荣誉", "奖学金", "三好", "优秀")),
    ("证书资格", ("证书", "软考", "资格", "认证")),
    ("教育背景", ("教育", "学校", "专业", "学位", "毕业")),
    ("技能专长", ("技能", "掌握", "熟悉", "精通")),
    ("求职意向", ("求职", "意向", "岗位", "城市")),
    ("项目经历", ("项目", "开发", "上线", "负责")),
)

MATERIAL_TOPICS = frozenset(
    {
        "项目经历",
        "技能专长",
        "教育背景",
        "竞赛奖项",
        "荣誉奖励",
        "证书资格",
        "求职意向",
        "个人特质",
        "自我介绍",
        "综合简历",
    }
)

# 文件名是个人知识库里最稳定的主题信号。正文经常同时提到项目、技能、证书和
# 奖项，不能再用“全文首次命中”决定唯一主题。
FILENAME_TOPIC_RULES = (
    ("项目经历", ("项目介绍", "项目经历", "项目经验")),
    ("技能专长", ("技能专长", "专业技能", "技术栈")),
    ("教育背景", ("教育背景", "教育经历", "课程成绩")),
    ("竞赛奖项", ("竞赛奖项", "竞赛经历", "获奖经历")),
    ("荣誉奖励", ("个人荣誉", "荣誉奖励", "奖学金")),
    ("证书资格", ("证书说明", "资格证书", "专业证书")),
    ("求职意向", ("求职意向", "求职动机", "职业规划")),
    ("个人特质", ("个人特质", "兴趣爱好", "性格特点")),
    ("自我介绍", ("自我介绍",)),
    ("综合简历", ("简历",)),
)

TOPIC_ALIASES = {
    "项目": "项目经历",
    "项目经验": "项目经历",
    "技能": "技能专长",
    "专业技能": "技能专长",
    "教育": "教育背景",
    "竞赛": "竞赛奖项",
    "奖项": "竞赛奖项",
    "荣誉": "荣誉奖励",
    "证书": "证书资格",
    "求职": "求职意向",
    "兴趣爱好": "个人特质",
    "简历": "综合简历",
}

class DocumentMetadataError(ValueError):
    pass


def normalize_metadata_input(
    value: Mapping[str, Any] | None,
    *,
    allow_clear: bool = False,
) -> dict[str, Any]:
    if not value:
        return {}
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = FIELD_ALIASES.get(str(raw_key).strip(), str(raw_key).strip())
        if key in {"filename", "original_name", "local_path", "path", "sha256", "size"}:
            normalized[key] = raw_value
            continue
        if raw_value in (None, "", []):
            if allow_clear:
                normalized[key] = None
            continue
        normalized[key] = _normalize_field_value(key, raw_value)
    return normalized


def apply_document_metadata(
    document: Document,
    metadata: Mapping[str, Any] | None,
    *,
    source: str,
    confidence: float,
    allow_clear: bool = False,
) -> dict[str, Any]:
    """Merge metadata without allowing low-trust inference to overwrite explicit values."""

    incoming = normalize_metadata_input(metadata, allow_clear=allow_clear)
    extension = _json_object(document.document_metadata)
    provenance = _json_object(document.metadata_provenance)
    priority = SOURCE_PRIORITIES.get(source, 0)
    now = datetime.now(timezone.utc).isoformat()

    for key, value in incoming.items():
        if key in {"filename", "original_name", "local_path", "path", "sha256", "size"}:
            continue
        previous = provenance.get(key) if isinstance(provenance.get(key), dict) else {}
        previous_priority = int(previous.get("priority") or 0)
        current_value = getattr(document, key, None) if key in CORE_METADATA_FIELDS else extension.get(key)
        if current_value not in (None, "", []) and previous_priority > priority:
            continue
        if key in CORE_METADATA_FIELDS:
            setattr(document, key, value)
        else:
            if value is None:
                extension.pop(key, None)
            else:
                extension[key] = value
        provenance[key] = {
            "source": "user_clear" if source == "user" and value is None else source,
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "priority": priority,
            "updated_at": now,
        }

    if source in {"user", "manifest", "official_url"} and incoming:
        document.metadata_status = "user_edited" if source == "user" else "structured"
        provenance["metadata_status"] = {
            "source": source,
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "priority": priority,
            "updated_at": now,
        }
    elif not document.metadata_status:
        document.metadata_status = "inferred"

    document.document_metadata = json.dumps(extension, ensure_ascii=False)
    document.metadata_provenance = json.dumps(provenance, ensure_ascii=False)
    return document_metadata_snapshot(document)


def infer_metadata_from_parsed(parsed: ParsedDocument, filename: str) -> dict[str, Any]:
    text = parsed.text[:12000]
    compact = " ".join(text.split())
    header_text = _header_text(parsed, fallback=text[:4000])
    footer_text = text[-3000:]
    filename_title = Path(filename).stem.split("_", maxsplit=1)[-1]
    title = _first_heading(parsed) or str(parsed.metadata.get("source_title") or filename_title)
    authority = _extract_labeled_authority(header_text, footer_text)
    document_number_match = re.search(
        r"(?:[\u4e00-\u9fffA-Za-z]{1,18})[〔\[]\d{4}[〕\]]\d{1,6}号",
        header_text,
    )
    publication_date = _extract_labeled_date(header_text + " " + footer_text, ("发布日期", "公布日期", "发布于"))
    expiration_date = _extract_labeled_date(compact, ("失效日期", "废止日期", "有效期至"))
    topic = infer_material_topic(
        text=compact,
        filename=filename,
        heading=_first_heading(parsed),
        header_text=header_text,
    )
    return {
        **dict(parsed.metadata or {}),
        "title": title,
        "issuing_authority": authority,
        "document_number": document_number_match.group(0) if document_number_match else None,
        "publication_date": publication_date,
        "expiration_date": expiration_date,
        "material_topic": topic,
        "source_type": "uploaded_file",
    }


def infer_material_topic(
    *,
    text: str,
    filename: str,
    heading: str | None = None,
    header_text: str | None = None,
) -> str | None:
    """Infer one stable resume-domain topic without letting incidental terms win."""

    explicit = _extract_labeled_material_topic(header_text or text[:2000])
    if explicit:
        return explicit

    filename_topic = _classify(Path(filename).stem, FILENAME_TOPIC_RULES)
    if filename_topic:
        return filename_topic

    heading_topic = _classify_by_score(heading or "", TOPIC_RULES)
    if heading_topic:
        return heading_topic
    return _classify_by_score(text, TOPIC_RULES)


def document_metadata_snapshot(document: Document, *, include_provenance: bool = True) -> dict[str, Any]:
    result = _json_object(document.document_metadata)
    for field_name in CORE_METADATA_FIELDS:
        value = getattr(document, field_name, None)
        if value not in (None, "", []):
            result[field_name] = value
    result.setdefault("source_title", document.title or Path(document.filename).stem)
    result.setdefault("source_filename", document.filename)
    result.setdefault("file_sha256", document.file_sha256)
    result["identity_review_status"] = document.identity_review_status or "unreviewed"
    result["identity_reviewed_at"] = (
        document.identity_reviewed_at.isoformat() if document.identity_reviewed_at else None
    )
    result["identity_reviewed_snapshot_hash"] = document.identity_reviewed_snapshot_hash
    result["identity_warnings"] = identity_metadata_warnings(document)
    if include_provenance:
        result["metadata_provenance"] = _json_object(document.metadata_provenance)
    return result


def retrieval_metadata_snapshot(document: Document) -> dict[str, Any]:
    extension = _json_object(document.document_metadata)
    result = {
        field_name: getattr(document, field_name, None)
        for field_name in RETRIEVAL_METADATA_FIELDS
        if getattr(document, field_name, None) not in (None, "", [])
    }
    result["source_title"] = document.title or str(extension.get("source_title") or Path(document.filename).stem)
    result["source_filename"] = document.filename
    result["file_sha256"] = document.file_sha256
    return result


def invalidate_identity_review(document: Document) -> None:
    document.identity_review_status = "unreviewed"
    document.identity_reviewed_at = None
    document.identity_reviewed_snapshot_hash = None


def confirm_document_identity(document: Document) -> str:
    validate_document_identity(document)
    reviewed_at = datetime.now(timezone.utc)
    snapshot_hash = identity_snapshot_hash(document)
    document.identity_review_status = "confirmed"
    document.identity_reviewed_at = reviewed_at
    document.identity_reviewed_snapshot_hash = snapshot_hash
    return snapshot_hash


def identity_snapshot_hash(document: Document) -> str:
    payload = {
        field_name: getattr(document, field_name, None)
        for field_name in IDENTITY_METADATA_FIELDS
    }
    payload["file_sha256"] = document.file_sha256
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def validate_document_identity(document: Document) -> None:
    if (
        document.publication_date
        and document.expiration_date
        and document.expiration_date < document.publication_date
    ):
        raise DocumentMetadataError("失效日期不能早于颁发日期")


def identity_metadata_warnings(document: Document) -> list[str]:
    warnings: list[str] = []
    if (
        document.publication_date
        and document.expiration_date
        and document.expiration_date < document.publication_date
    ):
        warnings.append("失效日期早于颁发日期，请人工核对")
    return warnings


def validate_metadata_urls(metadata: Mapping[str, Any]) -> None:
    for key in ("source_url", "attachment_url"):
        value = metadata.get(key)
        if not value:
            continue
        parsed = urlparse(str(value))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise DocumentMetadataError(f"{key} 必须是无凭据的 HTTP(S) URL")


def _normalize_field_value(key: str, value: Any) -> Any:
    if key in {"publication_date", "expiration_date"}:
        return _normalize_date(value)
    if key in {"source_url", "attachment_url"}:
        text = str(value).strip()
        validate_metadata_urls({key: text})
        return text
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip().replace("/", "-").replace("年", "-").replace("月", "-").replace("日", "")
    match = re.fullmatch(r"(20\d{2})-(\d{1,2})-(\d{1,2})", text)
    if not match:
        raise DocumentMetadataError(f"日期必须使用 YYYY-MM-DD：{value}")
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()


def _extract_date(text: str) -> str | None:
    match = re.search(r"(20\d{2})[年./-](\d{1,2})[月./-](\d{1,2})日?", text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


def _extract_labeled_date(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        if label == "自":
            pattern = r"自\s*((?:20\d{2})[年./-]\d{1,2}[月./-]\d{1,2}日?)\s*起(?:施行|实施|生效)"
        else:
            pattern = rf"{re.escape(label)}\s*[：:]?\s*((?:20\d{{2}})[年./-]\d{{1,2}}[月./-]\d{{1,2}}日?)"
        match = re.search(pattern, text)
        if match:
            return _extract_date(match.group(1))
    return None


def _extract_labeled_authority(header_text: str, footer_text: str) -> str | None:
    labeled = re.search(r"(?:颁发机构|发布机构|授予单位)\s*[：:]\s*([^\n；;]{2,80})", header_text)
    if labeled:
        candidate = labeled.group(1).strip()
        return next((item for item in KNOWN_AUTHORITIES if item in candidate), candidate)
    combined = f"{header_text}\n{footer_text}"
    return next((item for item in KNOWN_AUTHORITIES if item in combined), None)


def _header_text(parsed: ParsedDocument, *, fallback: str) -> str:
    values: list[str] = []
    for block in parsed.blocks[:20]:
        value = block.text.strip()
        if value:
            values.append(value)
        if sum(len(item) for item in values) >= 4000:
            break
    return "\n".join(values) or fallback


def _first_heading(parsed: ParsedDocument) -> str | None:
    for block in parsed.blocks:
        if block.block_type == "heading" and 2 <= len(block.text.strip()) <= 500:
            return block.text.strip()
    return None


def _classify(text: str, rules: tuple[tuple[str, tuple[str, ...]], ...]) -> str | None:
    return next((label for label, terms in rules if any(term in text for term in terms)), None)


def _classify_by_score(
    text: str,
    rules: tuple[tuple[str, tuple[str, ...]], ...],
) -> str | None:
    if not text:
        return None
    scores = {
        label: sum(min(text.count(term), 3) for term in terms)
        for label, terms in rules
    }
    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return None
    winners = [label for label, score in scores.items() if score == best_score]
    # 模糊正文宁可不分类，也不要制造一个会参与检索过滤的错误标签。
    return winners[0] if len(winners) == 1 else None


def _extract_labeled_material_topic(text: str) -> str | None:
    match = re.search(
        r"(?:材料主题|material[_ -]?topic)\s*[：:]\s*([^\n|>]{1,30})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw = match.group(1).strip().strip("`*_[]（）() ")
    if raw in MATERIAL_TOPICS:
        return raw
    return TOPIC_ALIASES.get(raw)


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}
