"""事实台账导入脚本：把 scripts/facts_seed.jsonl 幂等写入 fact_ledger 表。

用法（在项目根目录）：
    python scripts/seed_fact_ledger.py            # 从默认种子文件导入
    python scripts/seed_fact_ledger.py --seed 自定义.jsonl
    python scripts/seed_fact_ledger.py --dry-run  # 只打印条数，不写库

按 fact_id upsert：重复执行不会产生重复记录；种子文件是台账的事实源，
修改事实时先改种子再重跑本脚本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = PROJECT_ROOT / "scripts" / "facts_seed.jsonl"

VALID_STATUS = {"confirmed", "pending", "inferred", "conflict"}


def load_seed(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name}:{line_no} 不是合法 JSON：{exc}") from exc
        missing = [key for key in ("fact_id", "subject", "predicate", "value") if not record.get(key)]
        if missing:
            raise SystemExit(f"{path.name}:{line_no} 缺少必填字段：{', '.join(missing)}")
        status = str(record.get("status") or "confirmed")
        if status not in VALID_STATUS:
            raise SystemExit(
                f"{path.name}:{line_no} status={status!r} 非法，允许值：{sorted(VALID_STATUS)}"
            )
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="导入事实台账种子（按 fact_id 幂等 upsert）")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED, help="种子文件路径（JSONL）")
    parser.add_argument("--dry-run", action="store_true", help="只校验并打印条数，不写库")
    args = parser.parse_args()

    if not args.seed.exists():
        raise SystemExit(f"种子文件不存在：{args.seed}")

    records = load_seed(args.seed)
    print(f"种子文件：{args.seed.name}，共 {len(records)} 条")
    if args.dry_run:
        return 0

    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.app.core.database import SessionLocal, init_db
    from backend.app.services.fact_ledger_service import seed_fact_records

    init_db()
    with SessionLocal() as db:
        written = seed_fact_records(db, records)
    print(f"已写入 fact_ledger：{written} 条（按 fact_id upsert，重复执行幂等）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
