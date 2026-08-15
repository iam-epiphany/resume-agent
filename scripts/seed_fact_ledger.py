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
# 事实种子是部署者自己的个人数据，默认放 data/（不入 git）；
# 用 --seed 指向自己的种子文件，例如：
#   python scripts/seed_fact_ledger.py --seed data/facts_seed.jsonl
DEFAULT_SEED = PROJECT_ROOT / "data" / "facts_seed.jsonl"

# 2026-08-15 起优先写 evidence_status（explicit/inferred/conflict/missing）；
# 旧种子文件的 status（confirmed/pending/inferred/conflict）仍兼容，由
# seed_fact_records 映射。
VALID_EVIDENCE_STATUS = {"explicit", "inferred", "conflict", "missing"}
VALID_LEGACY_STATUS = {"confirmed", "pending", "inferred", "conflict"}


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
        if record.get("evidence_status"):
            evidence = str(record["evidence_status"])
            if evidence not in VALID_EVIDENCE_STATUS:
                raise SystemExit(
                    f"{path.name}:{line_no} evidence_status={evidence!r} 非法，"
                    f"允许值：{sorted(VALID_EVIDENCE_STATUS)}"
                )
        else:
            status = str(record.get("status") or "confirmed")
            if status not in VALID_LEGACY_STATUS:
                raise SystemExit(
                    f"{path.name}:{line_no} status={status!r} 非法，"
                    f"允许值：{sorted(VALID_LEGACY_STATUS)}（或改用 evidence_status）"
                )
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="导入事实台账种子（按 fact_id 幂等 upsert）")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED, help="种子文件路径（JSONL）")
    parser.add_argument("--dry-run", action="store_true", help="只校验并打印条数，不写库")
    args = parser.parse_args()

    if not args.seed.exists():
        raise SystemExit(
            f"种子文件不存在：{args.seed}
"
            "事实种子是部署者自己的个人数据（subject/predicate/value/evidence_status/source_file）。
"
            "参考格式：{\"fact_id\": \"edu_school\", \"subject\": \"学校名\", \"predicate\": \"学历角色\", "
            "\"value\": \"本科\", \"evidence_status\": \"explicit\", \"source_file\": \"教育背景.md\"}"
        )

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
