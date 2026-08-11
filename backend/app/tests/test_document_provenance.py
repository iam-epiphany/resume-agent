import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.core.database import Base
from backend.app.models.document import Document
from backend.app.services.document_manifest_service import (
    ManifestImportError,
    import_manifest_records,
    parse_manifest,
)
from backend.app.services.document_metadata_service import (
    apply_document_metadata,
    confirm_document_identity,
    document_metadata_snapshot,
    infer_metadata_from_parsed,
    invalidate_identity_review,
    validate_document_identity,
)
from backend.app.services.document_types import ParsedBlock, ParsedDocument
from backend.app.services.retrieval_service import filter_candidates_by_metadata
from backend.app.services.vector_store_service import VectorSearchResult
from backend.app.services.document_url_import_service import (
    DocumentUrlImportError,
    _validate_public_http_url,
)


def _document() -> Document:
    return Document(
        document_id="DOC-1",
        filename="项目介绍_外卖平台.md",
        filename_norm="项目介绍_外卖平台.md",
        file_type="md",
        size=12,
        file_sha256="a" * 64,
        storage_path="外卖平台.md",
        status="indexed",
        index_version="test",
        metadata_status="inferred",
        document_metadata="{}",
    )


def test_explicit_metadata_is_not_overwritten_by_parser_inference() -> None:
    document = _document()
    apply_document_metadata(
        document,
        {"title": "官方标题", "source_url": "https://edu.henu.edu.cn/material"},
        source="manifest",
        confidence=0.95,
    )
    apply_document_metadata(
        document,
        {"title": "文件名推断标题", "source_url": "https://example.com/inferred"},
        source="parser",
        confidence=0.5,
    )

    metadata = document_metadata_snapshot(document)
    assert document.title == "官方标题"
    assert document.source_url == "https://edu.henu.edu.cn/material"
    assert metadata["metadata_provenance"]["title"]["source"] == "manifest"


def test_parser_uses_labeled_dates_and_does_not_treat_first_body_date_as_publication() -> None:
    parsed = ParsedDocument(
        text=(
            "证书说明\n软考办发〔2026〕12号\n"
            "2024年1月31日的课程成绩仅用于举例。\n"
            "本证书自2026年7月1日起核发，有效期至2028年6月30日。\n"
            "人力资源和社会保障部"
        ),
        blocks=[
            ParsedBlock("证书说明", "heading", 0, level=1),
            ParsedBlock("软考办发〔2026〕12号", "paragraph", 1),
            ParsedBlock("2024年1月31日的课程成绩仅用于举例。", "paragraph", 2),
            ParsedBlock("本证书自2026年7月1日起核发，有效期至2028年6月30日。", "paragraph", 3),
            ParsedBlock("人力资源和社会保障部", "paragraph", 4),
        ],
    )

    metadata = infer_metadata_from_parsed(parsed, "证书说明.md")

    assert metadata["title"] == "证书说明"
    assert metadata["document_number"] == "软考办发〔2026〕12号"
    assert metadata["issuing_authority"] == "人力资源和社会保障部"
    assert metadata["publication_date"] is None
    assert metadata["expiration_date"] == "2028-06-30"


def test_material_topic_prefers_filename_over_incidental_body_terms() -> None:
    parsed = ParsedDocument(
        text=(
            "外卖平台项目\n"
            "项目使用证书认证、技能训练和优秀奖项作为示例，正文还会多次提到证书与奖项。\n"
            "我负责订单、支付和缓存模块的开发与上线。"
        ),
        blocks=[ParsedBlock("外卖平台项目", "heading", 0, level=1)],
    )

    metadata = infer_metadata_from_parsed(parsed, "项目介绍_外卖平台.md")

    assert metadata["material_topic"] == "项目经历"


def test_material_topic_accepts_concise_explicit_label() -> None:
    parsed = ParsedDocument(
        text=(
            "技能说明\n"
            "材料主题：技能专长\n"
            "正文提到蓝桥杯一等奖、证书和项目，但它们只是技能应用场景。"
        ),
        blocks=[
            ParsedBlock("技能说明", "heading", 0, level=1),
            ParsedBlock("材料主题：技能专长", "paragraph", 1),
        ],
    )

    metadata = infer_metadata_from_parsed(parsed, "能力说明.md")

    assert metadata["material_topic"] == "技能专长"


def test_manual_clear_is_unknown_and_confirmation_accepts_incomplete_identity() -> None:
    document = _document()
    apply_document_metadata(
        document,
        {"title": "待清空标题", "publication_date": "2026-01-01"},
        source="user",
        confidence=1.0,
        allow_clear=True,
    )
    apply_document_metadata(
        document,
        {"publication_date": None},
        source="user",
        confidence=1.0,
        allow_clear=True,
    )
    invalidate_identity_review(document)
    snapshot_hash = confirm_document_identity(document)

    metadata = document_metadata_snapshot(document)
    assert "publication_date" not in metadata
    assert metadata["metadata_provenance"]["publication_date"]["source"] == "user_clear"
    assert metadata["identity_review_status"] == "confirmed"
    assert metadata["identity_reviewed_snapshot_hash"] == snapshot_hash


def test_identity_rejects_expiration_before_publication_date() -> None:
    document = _document()
    document.publication_date = "2026-07-01"
    document.expiration_date = "2026-06-30"

    with pytest.raises(ValueError, match="失效日期不能早于颁发日期"):
        validate_document_identity(document)


def test_manifest_updates_matching_document_and_rejects_qa_data() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_document())
        db.commit()
        records = parse_manifest(
            json.dumps(
                {
                    "files": [
                        {
                            "filename": "项目介绍_外卖平台.md",
                            "sha256": "a" * 64,
                            "doc_id": "HENU-001",
                            "title": "外卖平台项目介绍",
                            "source_url": "https://edu.henu.edu.cn/project",
                            "match_status": "verified_manual",
                        }
                    ]
                },
                ensure_ascii=False,
            ).encode(),
            "manifest.json",
        )
        result = import_manifest_records(db, records)
        assert result[0].status == "updated"
        document = db.query(Document).one()
        assert document.external_doc_id == "HENU-001"
        assert document.source_url == "https://edu.henu.edu.cn/project"

    with pytest.raises(ManifestImportError, match="评测数据禁止"):
        parse_manifest(b'{"question":"q","answer":"a"}\n', "qa.jsonl")


def test_manifest_imports_source_metadata_without_review_gate() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_document())
        db.commit()
        result = import_manifest_records(
            db,
            [
                {
                    "filename": "项目介绍_外卖平台.md",
                    "sha256": "a" * 64,
                    "title": "未复核项目介绍",
                    "source_url": "https://edu.henu.edu.cn/project",
                    "attachment_url": "https://edu.henu.edu.cn/project.md",
                    "match_status": "needs_review",
                }
            ],
        )
        assert result[0].status == "updated"
        document = db.query(Document).one()
        assert document.title == "未复核项目介绍"
        assert document.source_url == "https://edu.henu.edu.cn/project"
        assert document.attachment_url == "https://edu.henu.edu.cn/project.md"
        metadata = document_metadata_snapshot(document)
        assert "provenance_status" not in metadata
        assert "official_match_status" not in metadata


def test_package_only_manifest_updates_custody_metadata() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_document())
        db.commit()
        result = import_manifest_records(
            db,
            [
                {
                    "filename": "项目介绍_外卖平台.md",
                    "sha256": "a" * 64,
                    "doc_id": "PKG-AAAAAAAAAAAAAAAA",
                    "title": "知识库包材料",
                    "source_type": "knowledge_package",
                    "provenance_status": "package_only",
                    "official_match_status": "package_only",
                    "contest_package_sha256": "b" * 64,
                    "match_status": "package_only",
                }
            ],
        )
        assert result[0].status == "updated"
        document = db.query(Document).one()
        metadata = document_metadata_snapshot(document)
        assert document.external_doc_id == "PKG-AAAAAAAAAAAAAAAA"
        assert document.source_type == "knowledge_package"
        assert document.source_url is None
        assert metadata["contest_package_sha256"] == "b" * 64


def test_package_manifest_does_not_clear_existing_optional_urls() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        document = _document()
        document.source_url = "https://edu.henu.edu.cn/verified"
        document.attachment_url = "https://edu.henu.edu.cn/verified.md"
        db.add(document)
        db.commit()
        result = import_manifest_records(
            db,
            [
                {
                    "filename": "项目介绍_外卖平台.md",
                    "sha256": "a" * 64,
                    "doc_id": "PKG-AAAAAAAAAAAAAAAA",
                    "source_type": "knowledge_package",
                    "match_status": "package_only",
                }
            ],
        )
        assert result[0].status == "updated"
        updated = db.query(Document).one()
        assert updated.source_url == "https://edu.henu.edu.cn/verified"
        assert updated.attachment_url == "https://edu.henu.edu.cn/verified.md"


def test_manifest_accepts_non_official_url_as_optional_metadata() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_document())
        db.commit()
        result = import_manifest_records(
            db,
            [
                {
                    "filename": "项目介绍_外卖平台.md",
                    "sha256": "a" * 64,
                    "title": "外部来源",
                    "source_url": "https://example.com/project",
                    "source_type": "external_page",
                    "match_status": "verified_manual",
                }
            ],
        )
        assert result[0].status == "updated"
        document = db.query(Document).one()
        assert document.source_url == "https://example.com/project"


def test_url_import_rejects_localhost() -> None:
    with pytest.raises(DocumentUrlImportError, match="非公网地址"):
        _validate_public_http_url("http://127.0.0.1/internal")


def test_metadata_filter_matches_authority_and_file_type() -> None:
    candidate = VectorSearchResult(
        chunk_id="C1",
        document_id="D1",
        filename="证书说明.md",
        section_title="软考证书",
        page_number=None,
        text="软考中级证书正文",
        embedding_text="正文",
        token_count=10,
        score=0.8,
        chunk_type="paragraph",
        section_number="1",
        metadata={
            "issuing_authority": "人力资源和社会保障部",
            "source_format": "md",
        },
    )
    assert filter_candidates_by_metadata(
        [candidate],
        {"issuing_authority": "人力资源和社会保障部"},
    ) == [candidate]
    assert filter_candidates_by_metadata([candidate], {"file_type": "pdf"}) == []
