"""Create a reproducible CPU ONNX artifact for the local BGE reranker.

Examples:
    python scripts/export_reranker_onnx.py export --source data/models/bge-reranker-base --output data/models/bge-reranker-base-onnx-fp32
    python scripts/export_reranker_onnx.py quantize --source data/models/bge-reranker-base-onnx-fp32 --output data/models/bge-reranker-base-onnx-int8

The application only enables ``MODEL_BACKEND=onnx`` after the resulting artifact
has passed the parity benchmark.  This script never changes the source model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "models" / "bge-reranker-base"
DEFAULT_FP32_OUTPUT = PROJECT_ROOT / "data" / "models" / "bge-reranker-base-onnx-fp32"
DEFAULT_INT8_OUTPUT = PROJECT_ROOT / "data" / "models" / "bge-reranker-base-onnx-int8"
METADATA_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, default_source, default_output in (
        ("export", DEFAULT_SOURCE, DEFAULT_FP32_OUTPUT),
        ("quantize", DEFAULT_FP32_OUTPUT, DEFAULT_INT8_OUTPUT),
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--source", type=Path, default=default_source)
        subparser.add_argument("--output", type=Path, default=default_output)
        subparser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(f"Output directory is not empty: {path}. Use --overwrite to reuse it.")
    path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_metadata(source: Path, output: Path) -> None:
    for name in METADATA_FILES:
        source_file = source / name
        if source_file.is_file():
            shutil.copy2(source_file, output / name)


def write_manifest(output: Path, *, command: str, source: Path, model_path: Path) -> None:
    manifest = {
        "format": "resumemind-reranker-onnx-v1",
        "command": command,
        "source": str(source.resolve()),
        "model": model_path.name,
        "model_sha256": sha256(model_path),
    }
    source_weight = next(iter(_weight_files(source)), None)
    if source_weight is not None:
        manifest["source_weight"] = source_weight.name
        manifest["source_weight_sha256"] = sha256(source_weight)
    (output / "resumemind-onnx-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _weight_files(path: Path) -> Iterable[Path]:
    yield from path.glob("*.safetensors")
    yield from path.glob("*.bin")


def export(source: Path, output: Path, *, overwrite: bool) -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    if not source.is_dir():
        raise RuntimeError(f"Source model directory does not exist: {source}")
    prepare_output(output, overwrite=overwrite)
    print(f"Exporting {source} -> {output}")
    model = ORTModelForSequenceClassification.from_pretrained(
        source,
        export=True,
        local_files_only=True,
    )
    model.save_pretrained(output)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.save_pretrained(output)
    model_path = output / "model.onnx"
    if not model_path.is_file():
        raise RuntimeError(f"Export did not produce expected file: {model_path}")
    write_manifest(output, command="export", source=source, model_path=model_path)


def quantize(source: Path, output: Path, *, overwrite: bool) -> None:
    from optimum.onnxruntime import AutoQuantizationConfig, ORTQuantizer

    model_path = source / "model.onnx"
    if not model_path.is_file():
        raise RuntimeError(f"Expected an exported FP32 model at: {model_path}")
    prepare_output(output, overwrite=overwrite)
    print(f"Dynamically quantizing {source} -> {output}")
    quantizer = ORTQuantizer.from_pretrained(source, file_name="model.onnx")
    result_path = quantizer.quantize(
        quantization_config=AutoQuantizationConfig.avx2(
            is_static=False,
            # XLM-Reranker is sensitive to per-tensor weight quantization.
            # Per-channel dynamic INT8 keeps its ranking much closer to FP32.
            per_channel=True,
        ),
        save_dir=output,
        file_suffix="int8",
    )
    result_path = Path(result_path)
    if result_path.is_dir():
        result_path = output / "model_int8.onnx"
    target_model = output / "model.onnx"
    if result_path != target_model:
        if target_model.exists():
            target_model.unlink()
        result_path.replace(target_model)
    copy_metadata(source, output)
    write_manifest(output, command="quantize_dynamic_avx2", source=source, model_path=target_model)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise RuntimeError("Source and output must be different directories.")
    if args.command == "export":
        export(source, output, overwrite=args.overwrite)
    else:
        quantize(source, output, overwrite=args.overwrite)
    print(f"Done: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
