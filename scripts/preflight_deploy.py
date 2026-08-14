"""上线前只读检查：知识库边界、配置格式与 Python 依赖。

用法：
    python scripts/preflight_deploy.py
    python scripts/preflight_deploy.py --skip-dependency-check

脚本不会上传、删除或修改任何资料；适合在本机和云服务器启动容器前执行。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_knowledge_base() -> list[str]:
    errors: list[str] = []
    files = sorted(DOCS_ROOT.glob("*.md"))
    if not files:
        return ["docs/ 第一层未发现正式 Markdown 知识库文件"]
    for path in files:
        text = path.read_text(encoding="utf-8")
        if text.count("\n# ") + int(text.startswith("# ")) != 1:
            errors.append(f"{path.name}: 应且仅应有一个一级标题")
        if "> 材料主题：" not in text:
            errors.append(f"{path.name}: 缺少材料主题")
        if "[待确认]" in text or "[推断]" in text:
            errors.append(f"{path.name}: 含未确认或推断信息，不能作为已核实事实直接上线")
    nested = [path for path in DOCS_ROOT.rglob("*") if path.is_file() and path.parent != DOCS_ROOT]
    if nested:
        print(
            "提示：docs/ 存在子目录资料，默认上传不会收录："
            + "、".join(str(path.relative_to(DOCS_ROOT)) for path in nested[:5])
        )
    print(f"知识库：{len(files)} 个正式文件（docs/ 第一层）")
    return errors


def check_configuration() -> list[str]:
    try:
        from backend.app.core import config  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return [f"配置加载失败：{type(exc).__name__}: {exc}"]
    print("配置：格式校验通过（未输出任何密钥或访问码）")
    return []


def check_dependencies() -> list[str]:
    process = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode == 0:
        print("依赖：pip check 通过")
        return []
    detail = (process.stdout + process.stderr).strip().replace("\n", "；")
    return [f"依赖不兼容：{detail}"]


def main() -> int:
    parser = argparse.ArgumentParser(description="ResumeMind 部署前只读检查")
    parser.add_argument("--skip-dependency-check", action="store_true")
    args = parser.parse_args()

    errors = [*check_knowledge_base(), *check_configuration()]
    if not args.skip_dependency_check:
        errors.extend(check_dependencies())
    if errors:
        print("\n部署前检查未通过：")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\n部署前检查通过。下一步可执行 docker compose up -d，再用上传脚本的 --sync 重建知识库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
