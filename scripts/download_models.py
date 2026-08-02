"""直接从 HuggingFace 镜像下载模型到本地目录（绕过 huggingface_hub 的兼容问题）。

用法（本地或云服务器）:
    python scripts/download_models.py
    python scripts/download_models.py --endpoint https://hf-mirror.com
    python scripts/download_models.py --repo BAAI/bge-base-zh-v1.5 --repo BAAI/bge-reranker-base

下载完成后把 data/models 挂载进容器（docker-compose 已配置 :ro），
并将 .env 中 RESUME_OFFLINE_MODE 设为 true 即可离线运行。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOS = ["BAAI/bge-base-zh-v1.5", "BAAI/bge-reranker-base"]


def fetch_json(url: str, timeout: float = 60.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "ResumeMind/1.0 model-download"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, dest: Path, timeout: float = 300.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  已存在，跳过: {dest.name}")
        return
    request = urllib.request.Request(url, headers={"User-Agent": "ResumeMind/1.0 model-download"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with open(dest, "wb") as handle:
            written = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if total:
                    percent = written * 100 // total
                    print(f"\r  {dest.name}: {percent}% ({written // (1024 * 1024)}MB/{total // (1024 * 1024)}MB)", end="", flush=True)
    print(f"\r  {dest.name}: 完成 ({written // (1024 * 1024)}MB)")


def download_repo(repo: str, output_dir: Path, endpoint: str, timeout: float) -> None:
    target = output_dir / repo.split("/")[-1]
    print(f"\n=== 下载 {repo} → {target} ===")
    model_info = fetch_json(f"{endpoint}/api/models/{repo}", timeout=timeout)
    siblings = [entry["rfilename"] for entry in model_info.get("siblings", [])]
    if not siblings:
        raise RuntimeError(f"未从 {endpoint}/api/models/{repo} 获取到文件清单")
    print(f"文件清单: {len(siblings)} 个文件")
    for filename in siblings:
        url = f"{endpoint}/{repo}/resolve/main/{filename}"
        download_file(url, target / filename, timeout=timeout)
    print(f"完成: {repo}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载本地模型（绕过 huggingface_hub 兼容问题）")
    parser.add_argument("--repo", action="append", dest="repos", help="模型仓库（可多次指定）")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "models")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS
    print(f"下载端点: {args.endpoint}")
    print(f"输出目录: {args.output_dir}")
    for repo in repos:
        download_repo(repo, args.output_dir, args.endpoint, args.timeout)
    print("\n全部完成。将 .env 中 RESUME_OFFLINE_MODE 设为 true 后重启即可离线运行。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
