from dataclasses import dataclass
from pathlib import Path

from backend.app.services.document_parser import DocumentParseError, available_loader_names, parse_document_with_loader


@dataclass
class LoaderEvaluation:
    loader_name: str
    ok: bool
    block_count: int = 0
    heading_count: int = 0
    table_count: int = 0
    pages_with_text: int = 0
    text_preview: str = ""
    error: str | None = None


def evaluate_document_loaders(file_path: Path) -> list[LoaderEvaluation]:
    """Run all configured loaders for a file and return comparable extraction metrics."""

    evaluations: list[LoaderEvaluation] = []
    for loader_name in available_loader_names(file_path):
        try:
            parsed = parse_document_with_loader(file_path, loader_name=loader_name)
        except DocumentParseError as exc:
            evaluations.append(LoaderEvaluation(loader_name=loader_name, ok=False, error=str(exc)))
            continue

        page_numbers = {block.page_number for block in parsed.blocks if block.page_number is not None}
        evaluations.append(
            LoaderEvaluation(
                loader_name=loader_name,
                ok=True,
                block_count=len(parsed.blocks),
                heading_count=sum(1 for block in parsed.blocks if block.block_type == "heading"),
                table_count=sum(1 for block in parsed.blocks if block.block_type == "table"),
                pages_with_text=len(page_numbers),
                text_preview=parsed.text[:200],
            )
        )
    return evaluations
