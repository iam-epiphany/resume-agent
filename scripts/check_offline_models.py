from __future__ import annotations

import sys
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "data" / "models"
MODELS = {
    "BGE-Small-ZH": MODEL_ROOT / "bge-small-zh-v1.5",
    "BGE reranker": MODEL_ROOT / "bge-reranker-base",
}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "vocab.json",
}


def main() -> int:
    failed = False
    print("ResumeMind 离线模型检查")
    print(f"项目目录：{PROJECT_ROOT}")

    for display_name, model_dir in MODELS.items():
        ready, missing = check_model_directory(model_dir)
        if ready:
            print(f"[OK] {display_name} 模型目录可用：{relative_path(model_dir)}")
            continue

        failed = True
        print(f"[缺失] {display_name} 模型目录不可用：{relative_path(model_dir)}")
        for item in missing:
            print(f"  - {item}")
        print(f"  请将模型放置到：{relative_path(model_dir)}")
        print(f"  容器内路径应为：/app/data/models/{model_dir.name}")

    if failed:
        print("离线模型检查未通过。请先准备模型目录，再运行 docker compose up --build。")
        return 1

    print("离线模型检查通过。")
    return 0


def check_model_directory(model_dir: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not model_dir.is_dir():
        return False, ["模型目录不存在"]

    if not (model_dir / "config.json").is_file():
        missing.append("缺少 config.json")

    if not any((model_dir / name).is_file() for name in TOKENIZER_FILES):
        missing.append("缺少 tokenizer 相关文件")

    has_weights = any(model_dir.glob("*.safetensors")) or (model_dir / "pytorch_model.bin").is_file()
    has_weights = has_weights or any(model_dir.glob("*.bin"))
    if not has_weights:
        missing.append("缺少模型权重文件（*.safetensors 或 pytorch_model.bin）")

    return not missing, missing


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
