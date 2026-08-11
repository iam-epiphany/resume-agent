from dataclasses import dataclass
import re
from typing import Any

from backend.app.core.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    SEMANTIC_BREAK_THRESHOLD,
)
from backend.app.services.embedding_service import (
    EmbeddingServiceError,
    cosine_similarity,
    embed_for_semantic_split,
)
from backend.app.services.document_types import ParsedBlock, ParsedDocument


@dataclass
class ChunkDraft:
    chunk_id: str
    document_id: str
    text: str
    embedding_text: str
    token_count: int
    title: str | None
    section_title: str | None
    page_number: int | None
    chunk_type: str = "paragraph"
    section_path: list[str] | None = None
    section_number: str | None = None
    parent_section_number: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: dict[str, Any] | None = None


def build_chunks_from_parsed(
    document_id: str,
    parsed: ParsedDocument,
    source_file: str | None = None,
    inherited_metadata: dict[str, Any] | None = None,
) -> list[ChunkDraft]:
    """Build retrievable chunks from structured parser blocks."""

    source_format = _metadata_text(parsed.metadata.get("source_format"))
    groups = _group_blocks(parsed.blocks)
    chunks: list[ChunkDraft] = []
    for group in groups:
        for piece in _split_group(group, document_id=document_id):
            token_count = count_tokens(piece.text)
            piece_metadata = {
                key: value
                for key, value in (piece.metadata or {}).items()
                if value not in (None, "", [])
            }
            chunk_metadata = {**(inherited_metadata or {}), **piece_metadata}
            chunks.append(
                ChunkDraft(
                    chunk_id=f"{document_id}-CHUNK-{len(chunks) + 1:04d}",
                    document_id=document_id,
                    text=piece.text,
                    embedding_text=build_contextual_embedding_text(
                        text=piece.text,
                        source_file=source_file,
                        source_format=source_format,
                        section_title=group.section_title,
                        section_path=group.section_path,
                        section_number=group.section_number,
                        parent_section_number=group.parent_section_number,
                        page_number=group.page_number,
                        chunk_type=group.block_type,
                        chunk_metadata=chunk_metadata,
                    ),
                    token_count=token_count,
                    title=group.section_title,
                    section_title=group.section_title,
                    page_number=group.page_number,
                    chunk_type=group.block_type,
                    section_path=group.section_path,
                    section_number=group.section_number,
                    parent_section_number=group.parent_section_number,
                    metadata=chunk_metadata,
                )
            )
    for previous, current, next_chunk in zip([None, *chunks[:-1]], chunks, [*chunks[1:], None], strict=True):
        current.previous_chunk_id = previous.chunk_id if previous else None
        current.next_chunk_id = next_chunk.chunk_id if next_chunk else None
    return chunks


@dataclass
class _ChunkGroup:
    text: str
    section_title: str | None
    page_number: int | None
    block_type: str = "paragraph"
    section_path: list[str] | None = None
    section_number: str | None = None
    parent_section_number: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class _ChunkPiece:
    text: str
    metadata: dict[str, Any]


def _group_blocks(blocks: list[ParsedBlock]) -> list[_ChunkGroup]:
    groups: list[_ChunkGroup] = []
    buffer: list[str] = []
    current_section: str | None = None
    current_path: list[str] = []
    current_section_number: str | None = None
    current_parent_section_number: str | None = None
    current_page: int | None = None
    current_metadata: dict[str, Any] = {}
    has_body = False

    def flush() -> None:
        nonlocal buffer, has_body
        text = "\n\n".join(part for part in buffer if part.strip()).strip()
        if text and has_body:
            groups.append(
                _ChunkGroup(
                    text=text,
                    section_title=current_section,
                    page_number=current_page,
                    section_path=list(current_path),
                    section_number=current_section_number,
                    parent_section_number=current_parent_section_number,
                    metadata=dict(current_metadata),
                )
            )
        buffer = []
        has_body = False

    for block in blocks:
        if block.block_type == "heading":
            if has_body:
                flush()
            current_section = block.text
            current_section_number = _section_number(block.text)
            current_parent_section_number = _parent_section_number(current_section_number)
            current_path = _updated_section_path(current_path, block.text, block.level, current_section_number)
            buffer = [block.text]
            current_page = block.page_number
            current_metadata = dict(block.metadata or {})
            has_body = False
            continue

        if block.section_title and block.section_title != current_section and has_body:
            flush()

        if block.section_title:
            current_section = block.section_title
            current_section_number = _section_number(block.section_title)
            current_parent_section_number = _parent_section_number(current_section_number)
            if not current_path or current_path[-1] != block.section_title:
                current_path = _updated_section_path(
                    current_path,
                    block.section_title,
                    block.level,
                    current_section_number,
                )
        if block.metadata:
            current_metadata = {**current_metadata, **block.metadata}
        if block.page_number is not None and (current_page is None or not has_body):
            current_page = block.page_number

        buffer.append(block.text)
        has_body = True

    flush()
    return groups



def _section_number(section_title: str | None) -> str | None:
    if not section_title:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[\.、\s]", section_title)
    if match:
        return match.group(1)
    article = re.match(
        r"^\s*(第[零〇一二三四五六七八九十百千万两\d]+条(?:之[零〇一二三四五六七八九十百千万两\d]+)?)",
        section_title,
    )
    return article.group(1) if article else None


def _parent_section_number(section_number: str | None) -> str | None:
    if not section_number or "." not in section_number:
        return None
    return section_number.rsplit(".", maxsplit=1)[0]


def _updated_section_path(
    current_path: list[str],
    title: str,
    level: int | None,
    section_number: str | None,
) -> list[str]:
    if level is None and section_number:
        level = section_number.count(".") + 1
    if level is None:
        return [title]
    prefix = current_path[: max(level - 1, 0)]
    return [*prefix, title]


def _split_group(group: _ChunkGroup, *, document_id: str) -> list[_ChunkPiece]:
    cleaned = group.text.strip()
    if not cleaned:
        return []
    if count_tokens(cleaned) <= CHUNK_MAX_TOKENS:
        return [_ChunkPiece(text=cleaned, metadata=dict(group.metadata or {}))]

    paragraphs = [paragraph.strip() for paragraph in cleaned.split("\n\n") if paragraph.strip()]
    if len(paragraphs) >= 2:
        semantic_pieces = _split_by_semantic_breaks(paragraphs)
    else:
        semantic_pieces = [cleaned]

    pieces: list[str] = []
    for piece in semantic_pieces:
        pieces.extend(_split_by_token_limit(piece))
    return [
        _ChunkPiece(text=piece, metadata=dict(group.metadata or {}))
        for piece in _add_overlap([piece for piece in pieces if piece])
    ]


def _split_by_semantic_breaks(paragraphs: list[str]) -> list[str]:
    try:
        vectors = embed_for_semantic_split(paragraphs)
    except EmbeddingServiceError:
        # Parsing must remain available when the embedding model is temporarily
        # unavailable. Empty vectors keep the same token-aware grouping while
        # disabling only the optional semantic-distance break condition.
        vectors = []
    pieces: list[str] = []
    buffer: list[str] = []

    for index, paragraph in enumerate(paragraphs):
        if not buffer:
            buffer.append(paragraph)
            continue

        current_text = "\n\n".join(buffer)
        similarity = cosine_similarity(vectors[index - 1], vectors[index]) if index < len(vectors) else 1.0
        should_break = (
            count_tokens(current_text) >= CHUNK_TARGET_TOKENS
            or similarity < SEMANTIC_BREAK_THRESHOLD
            or count_tokens(f"{current_text}\n\n{paragraph}") > CHUNK_MAX_TOKENS
        )
        if should_break:
            pieces.append(current_text.strip())
            buffer = [paragraph]
        else:
            buffer.append(paragraph)

    if buffer:
        pieces.append("\n\n".join(buffer).strip())
    return pieces


def _split_by_token_limit(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if count_tokens(cleaned) <= CHUNK_MAX_TOKENS:
        return [cleaned]

    pieces: list[str] = []
    buffer = ""
    for unit in _semantic_units(cleaned):
        if count_tokens(unit) > CHUNK_MAX_TOKENS:
            if buffer.strip():
                pieces.append(buffer.strip())
                buffer = ""
            pieces.extend(_hard_split_token_units(unit))
            continue

        candidate = f"{buffer}{unit}" if buffer else unit.lstrip()
        if count_tokens(candidate) <= CHUNK_MAX_TOKENS:
            buffer = candidate
            continue

        if buffer.strip():
            pieces.append(buffer.strip())
        buffer = unit.lstrip()

    if buffer.strip():
        pieces.append(buffer.strip())
    return [piece for piece in pieces if piece]


def _add_overlap(pieces: list[str]) -> list[str]:
    if len(pieces) <= 1:
        return pieces

    overlapped = [pieces[0]]
    for previous, current in zip(pieces, pieces[1:]):
        tail = _semantic_overlap_tail(previous)
        candidate = f"{tail}\n\n{current}".strip() if tail else current
        if count_tokens(candidate) > CHUNK_MAX_TOKENS:
            candidate = current
        overlapped.append(candidate.strip())
    return overlapped


def count_tokens(text: str) -> int:
    return len(_token_units(text))


def _token_units(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+|[^\s]", text)


def _join_token_units(tokens: list[str]) -> str:
    text = ""
    previous_ascii = False
    for token in tokens:
        current_ascii = bool(re.fullmatch(r"[a-zA-Z0-9_]+", token))
        if text and previous_ascii and current_ascii:
            text += " "
        text += token
        previous_ascii = current_ascii
    return text


def build_contextual_embedding_text(
    *,
    text: str,
    source_file: str | None = None,
    source_format: str | None = None,
    section_title: str | None,
    section_path: list[str] | None = None,
    section_number: str | None = None,
    parent_section_number: str | None = None,
    page_number: int | None = None,
    chunk_type: str = "paragraph",
    chunk_metadata: dict[str, Any] | None = None,
) -> str:
    """Prepend deterministic retrieval context while keeping chunk text unchanged."""

    chunk_metadata = chunk_metadata or {}
    labels: list[str] = []
    if source_file:
        labels.append(f"来源文件：{source_file}")
    if source_format:
        labels.append(f"文档格式：{source_format}")
    if section_path:
        labels.append("章节路径：" + " > ".join(section_path))
    if section_title:
        labels.append(f"章节：{section_title}")
    if section_number:
        labels.append(f"条款号：{section_number}")
    if parent_section_number:
        labels.append(f"父条款号：{parent_section_number}")
    if page_number is not None:
        labels.append(f"页码：{page_number}")
    labels.append("内容类型：正文")
    labels = [label for label in labels if label]
    label_text = "\n".join(labels)
    return f"{label_text}\n\n{text}" if labels else text


def _metadata_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _semantic_units(text: str) -> list[str]:
    units: list[str] = []
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    for paragraph_index, paragraph in enumerate(paragraphs):
        prefix = "\n\n" if paragraph_index > 0 else ""
        if count_tokens(paragraph) <= CHUNK_TARGET_TOKENS:
            units.append(f"{prefix}{paragraph}")
            continue

        sentences = _split_sentences(paragraph)
        if not sentences:
            units.append(f"{prefix}{paragraph}")
            continue
        for sentence_index, sentence in enumerate(sentences):
            sentence_prefix = prefix if sentence_index == 0 else ""
            units.append(f"{sentence_prefix}{sentence}")
    return units


def _split_sentences(text: str) -> list[str]:
    parts = re.findall(r".+?(?:[。！？!?；;：:]|$)", text, flags=re.S)
    return [part.strip() for part in parts if part.strip()]


def _hard_split_token_units(text: str) -> list[str]:
    tokens = _token_units(text)
    pieces: list[str] = []
    start = 0
    step = max(CHUNK_MAX_TOKENS - CHUNK_OVERLAP_TOKENS, 1)
    while start < len(tokens):
        prefix = "" if start == 0 else "（承接上文）"
        limit = CHUNK_MAX_TOKENS - count_tokens(prefix)
        piece = f"{prefix}{_join_token_units(tokens[start : start + limit])}".strip()
        pieces.append(piece)
        start += step
    return pieces


def _semantic_overlap_tail(text: str) -> str:
    sentences = _split_sentences(text.replace("\n\n", ""))
    if not sentences:
        return ""

    selected: list[str] = []
    for sentence in reversed(sentences):
        candidate = "".join([sentence, *selected])
        if count_tokens(candidate) > CHUNK_OVERLAP_TOKENS:
            break
        selected.insert(0, sentence)
    return "".join(selected).strip()
