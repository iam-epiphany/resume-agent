"""Audit ResumeMind Markdown and live SQLite metadata before deployment.

Examples:
    python scripts/audit_knowledge_base.py
    python scripts/audit_knowledge_base.py --strict --report kb-audit.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.document_metadata_service import (  # noqa: E402
    MATERIAL_TOPICS,
    infer_material_topic,
)


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    target: str
    message: str


TOPIC_PATTERN = re.compile(r"材料主题\s*[：:]\s*([^\n|>]{1,30})")
MOJIBAKE_MARKERS = ("锛", "銆", "鈥", "绠", "浣", "璇", "鐨", "鍙", "鏁")
SECRET_PATTERNS = (
    ("private_key", re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
    ("credential", re.compile(r"(?i)(?:api[_-]?key|secret|password)\s*[=:]\s*[^\s]{8,}")),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
)


def markdown_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.md") if path.is_file())


def read_utf8(path: Path) -> tuple[str | None, Finding | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except UnicodeDecodeError as exc:
        return None, Finding("error", "invalid_utf8", str(path), f"不是有效 UTF-8：{exc}")


def audit_file(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    relative = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    lines = text.splitlines()
    h1 = [line for line in lines if line.startswith("# ")]
    if len(h1) != 1:
        findings.append(Finding("error", "h1_count", relative, f"一级标题应恰好 1 个，当前为 {len(h1)} 个"))

    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    if "�" in text or marker_count >= 6:
        findings.append(Finding("error", "possible_mojibake", relative, "疑似存在乱码或错误转码"))

    topic_match = TOPIC_PATTERN.search("\n".join(lines[:20]))
    explicit_topic = topic_match.group(1).strip().strip("`*_[]（）() ") if topic_match else None
    if not explicit_topic:
        findings.append(Finding("warning", "missing_topic", relative, "正文开头缺少“材料主题”"))
    elif explicit_topic not in MATERIAL_TOPICS:
        findings.append(Finding("error", "invalid_topic", relative, f"未知材料主题：{explicit_topic}"))

    heading = h1[0][2:].strip() if h1 else None
    inferred = infer_material_topic(
        text=text,
        filename=path.name,
        heading=heading,
        header_text="\n".join(lines[:30]),
    )
    if explicit_topic and inferred and explicit_topic != inferred:
        findings.append(
            Finding("error", "topic_conflict", relative, f"显式主题 {explicit_topic} 与推断主题 {inferred} 不一致")
        )

    for code, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(Finding("error", code, relative, "检测到不应进入公开知识库的敏感内容"))
    return findings


def normalized_paragraphs(path: Path, text: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        raw = " ".join(line.strip() for line in paragraph.splitlines() if not line.lstrip().startswith(("#", "> 材料主题")))
        normalized = re.sub(r"[\s`*_>#\-—，。；：、（）()\[\]]+", "", raw).lower()
        if len(normalized) >= 80:
            results.append((normalized, raw[:100]))
    return results


def audit_duplicate_paragraphs(contents: dict[Path, str]) -> list[Finding]:
    owners: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, text in contents.items():
        for normalized, preview in normalized_paragraphs(path, text):
            owners[normalized].append((path, preview))

    findings: list[Finding] = []
    for matches in owners.values():
        distinct = list(dict.fromkeys(path for path, _ in matches))
        if len(distinct) < 2:
            continue
        targets = ", ".join(path.name for path in distinct)
        findings.append(
            Finding("warning", "duplicate_paragraph", targets, f"跨文档重复段落：{matches[0][1]}…")
        )
    return findings


def audit_database(db_path: Path, contents: dict[Path, str]) -> tuple[list[Finding], dict[str, int]]:
    if not db_path.is_file():
        return [Finding("warning", "database_missing", str(db_path), "未找到运行库，跳过线上元数据核对")], {}
    findings: list[Finding] = []
    stats: dict[str, int] = {}
    local_by_name = {path.name: text for path, text in contents.items()}
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        if "documents" not in tables:
            return [Finding("error", "documents_table_missing", str(db_path), "缺少 documents 表")], stats
        rows = list(
            connection.execute(
                "select filename, material_topic, status, chunk_count from documents order by filename"
            )
        )
        stats["documents"] = len(rows)
        stats["chunks"] = sum(int(row[3] or 0) for row in rows)
        for filename, current_topic, status, _chunk_count in rows:
            text = local_by_name.get(str(filename))
            if text is None:
                continue
            inferred = infer_material_topic(
                text=text,
                filename=str(filename),
                heading=next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), None),
                header_text="\n".join(text.splitlines()[:30]),
            )
            if inferred and current_topic != inferred:
                findings.append(
                    Finding(
                        "error",
                        "database_topic_drift",
                        str(filename),
                        f"运行库主题为 {current_topic or '空'}，源文档应为 {inferred}；重新上传该文档以刷新 SQLite 与 Qdrant payload",
                    )
                )
            if status != "indexed":
                findings.append(Finding("warning", "document_not_indexed", str(filename), f"当前状态：{status}"))

        names = {str(row[0]) for row in rows}
        if any(name.lower().endswith(".pdf") and "简历" in name for name in names) and "简历文字版.md" in names:
            findings.append(
                Finding(
                    "warning",
                    "duplicate_resume_sources",
                    "documents",
                    "简历 PDF 与规范化文字版同时索引，重复事实可能挤占 prompt；建议只保留文字版作为口径基线",
                )
            )
    return findings, stats


def audit_qdrant(storage: Path, stats: dict[str, int]) -> list[Finding]:
    if not storage.is_dir():
        return []
    collections_dir = storage / "collections"
    collections = [path for path in collections_dir.iterdir() if path.is_dir()] if collections_dir.is_dir() else []
    total_bytes = sum(path.stat().st_size for path in storage.rglob("*") if path.is_file())
    total_mb = total_bytes / 1024 / 1024
    findings: list[Finding] = []
    if len(collections) > 1:
        findings.append(
            Finding(
                "warning",
                "multiple_vector_collections",
                str(storage),
                "发现多个 collection：" + ", ".join(path.name for path in collections) + "；确认旧 collection 不再使用后再清理",
            )
        )
    chunks = stats.get("chunks", 0)
    if chunks and total_mb > max(512, chunks * 4):
        findings.append(
            Finding(
                "warning",
                "oversized_vector_storage",
                str(storage),
                f"{chunks} 个 chunk 对应约 {total_mb:.1f}MB 持久化数据，疑似残留历史段/WAL",
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 ResumeMind 知识库结构、主题和运行库漂移")
    parser.add_argument("--docs", type=Path, default=PROJECT_ROOT / "docs")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "app.db")
    parser.add_argument("--qdrant", type=Path, default=PROJECT_ROOT / "data" / "qdrant")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="存在 warning 时也返回非零")
    args = parser.parse_args()

    findings: list[Finding] = []
    contents: dict[Path, str] = {}
    for path in markdown_files(args.docs.resolve()):
        text, error = read_utf8(path)
        if error:
            findings.append(error)
        elif text is not None:
            contents[path] = text
            findings.extend(audit_file(path, text))
    findings.extend(audit_duplicate_paragraphs(contents))
    db_findings, stats = audit_database(args.db.resolve(), contents)
    findings.extend(db_findings)
    findings.extend(audit_qdrant(args.qdrant.resolve(), stats))

    errors = sum(item.level == "error" for item in findings)
    warnings = sum(item.level == "warning" for item in findings)
    print(f"知识库审计：files={len(contents)} errors={errors} warnings={warnings}")
    for item in findings:
        print(f"[{item.level.upper()}] {item.code} | {item.target} | {item.message}")

    if args.report:
        payload = {
            "files": len(contents),
            "errors": errors,
            "warnings": warnings,
            "database": stats,
            "findings": [asdict(item) for item in findings],
        }
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入：{args.report}")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
