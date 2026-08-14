"""预下载 torch CPU wheel 到 wheels/，供 Docker 构建离线安装。

背景：download.pytorch.org 在本机网络环境不稳定，而 Docker 构建每次
都要下载 192MB 的 torch wheel，容易中途挂起。先在宿主机（网络更可控）
下载好 wheel，构建时自动走 wheels/ 离线安装（见 Dockerfile）。

用法：
    python scripts/download_torch_wheel.py

源（按顺序尝试，第一个成功的为止）：
    1. 阿里云镜像（国内，推荐）：https://mirrors.aliyun.com/pytorch-wheels/cpu/
    2. 官方源（兜底）：https://download.pytorch.org/whl/cpu/

wheel 文件名与 requirements-cpu.txt 的版本对应；下载后自动校验 zip 完整性。
"""
from __future__ import annotations

import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

WHEEL_NAME = "torch-2.12.1+cpu-cp313-cp313-manylinux_2_28_x86_64.whl"
SOURCES = [
    f"https://mirrors.aliyun.com/pytorch-wheels/cpu/{urllib.parse.quote(WHEEL_NAME)}",
    f"https://download.pytorch.org/whl/cpu/{urllib.parse.quote(WHEEL_NAME)}",
]
WHEELS_DIR = Path(__file__).resolve().parents[1] / "wheels"


def _download(url: str, target: Path) -> None:
    print(f"下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()


def main() -> int:
    WHEELS_DIR.mkdir(exist_ok=True)
    target = WHEELS_DIR / WHEEL_NAME
    if target.exists():
        print(f"已存在：{target}（如需重新下载请先删除）")
        return 0

    for url in SOURCES:
        try:
            _download(url, target)
            with zipfile.ZipFile(target) as z:
                print(f"校验通过：{len(z.namelist())} 个条目")
            print(f"完成：{target}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"失败（{exc}），尝试下一个源…")
            target.unlink(missing_ok=True)

    print("所有源均下载失败，请检查网络后重试。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
