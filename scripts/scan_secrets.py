from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)(?:deepseek_api_key|answer_generation_api_key)\s*[:=]\s*['\"]?[^\s'\"]{12,}"),
]
ALLOW = {".env"}
SKIP_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "dist-delivery",
    "data",
    "__pycache__",
}


def main() -> int:
    source_files, scan_mode = _source_files()
    findings: list[str] = []
    for path in source_files:
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.name in ALLOW or any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if (
                "${DEEPSEEK_API_KEY" in line
                or "$DeepSeekApiKey" in line
                or re.search(r"DEEPSEEK_API_KEY\s*=\s*$", line)
            ):
                continue
            if any(pattern.search(line) for pattern in PATTERNS):
                findings.append(f"{relative}:{line_number}")
    if findings:
        print("疑似密钥：")
        print("\n".join(findings))
        return 1
    print(f"source files secret scan passed ({scan_mode})")
    return 0


def _source_files() -> tuple[list[Path], str]:
    """Prefer Git's delivery view, but remain usable in source-only images."""

    if shutil.which("git") and (ROOT / ".git").exists():
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode == 0:
            return [ROOT / relative for relative in completed.stdout.splitlines()], "git"

    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
    ], "filesystem fallback"


if __name__ == "__main__":
    raise SystemExit(main())
