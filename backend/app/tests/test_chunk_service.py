from backend.app.services.chunk_service import build_chunks_from_parsed
from backend.app.services.document_types import ParsedBlock, ParsedDocument


def test_cross_page_paragraph_blocks_stay_in_one_chunk() -> None:
    parsed = ParsedDocument(
        text="本材料适用于技能掌握情况统计中跨页延续的段落。",
        blocks=[
            ParsedBlock(
                text="本材料适用于技能掌握情况统计中跨页延续的",
                block_type="paragraph",
                order_index=1,
                page_number=7,
            ),
            ParsedBlock(
                text="段落，系统应保留完整语义连续性。",
                block_type="paragraph",
                order_index=2,
                page_number=8,
            ),
        ],
        metadata={"source_format": "pdf"},
    )

    chunks = build_chunks_from_parsed("DOC-TEST", parsed)

    assert len(chunks) == 1
    assert "本材料适用于技能掌握情况统计中跨页延续的" in chunks[0].text
    assert "段落，系统应保留完整语义连续性。" in chunks[0].text
    assert chunks[0].page_number == 7
