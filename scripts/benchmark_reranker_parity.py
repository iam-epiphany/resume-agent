"""Compare an ONNX reranker artifact against the original local PyTorch BGE model.

The report is a release gate, not a marketing benchmark: the application must
stay on ``MODEL_BACKEND=pytorch`` if top-k agreement or score calibration fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTORCH = PROJECT_ROOT / "data" / "models" / "bge-reranker-base"
DEFAULT_ONNX = PROJECT_ROOT / "data" / "models" / "bge-reranker-base-onnx-int8" / "model.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch", type=Path, default=DEFAULT_PYTORCH)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "scripts" / "eval_cases.jsonl")
    parser.add_argument("--docs", type=Path, default=PROJECT_ROOT / "docs")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "reranker-parity-report.json")
    return parser.parse_args()


def load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        question = str(payload.get("question") or "").strip()
        if question:
            questions.append(question)
    if not questions:
        raise RuntimeError(f"No questions found in {path}")
    return questions


def load_candidates(docs: Path, count: int) -> list[str]:
    chunks: list[str] = []
    for path in sorted(docs.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        for part in text.split("\n\n"):
            normalized = " ".join(part.split())
            if len(normalized) >= 80:
                chunks.append(f"来源：{path.name}\n{normalized}")
    if len(chunks) < count:
        raise RuntimeError(f"Only {len(chunks)} usable markdown chunks found under {docs}")
    # Deterministic coverage across the corpus, avoiding a single long document.
    step = max(1, len(chunks) // count)
    return [chunks[index] for index in range(0, len(chunks), step)][:count]


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def score_pytorch(model, tokenizer, pairs: list[tuple[str, str]], batch_size: int) -> np.ndarray:
    import torch

    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset : offset + batch_size]
            encoded = tokenizer(
                [item[0] for item in batch],
                [item[1] for item in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs.append(model(**encoded).logits.detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs), time.perf_counter() - started


def score_onnx(session, tokenizer, pairs: list[tuple[str, str]], batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    input_names = {item.name for item in session.get_inputs()}
    started = time.perf_counter()
    for offset in range(0, len(pairs), batch_size):
        batch = pairs[offset : offset + batch_size]
        encoded = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        logits = session.run(None, {key: value for key, value in encoded.items() if key in input_names})[0]
        outputs.append(np.asarray(logits).reshape(-1))
    return np.concatenate(outputs), time.perf_counter() - started


def top_indices(values: np.ndarray, top_k: int) -> set[int]:
    return set(np.argsort(values)[-top_k:].tolist())


def main() -> int:
    args = parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import onnxruntime as ort

    questions = load_questions(args.cases)
    candidates = load_candidates(args.docs, args.candidates)
    tokenizer = AutoTokenizer.from_pretrained(args.pytorch, local_files_only=True)
    pytorch = AutoModelForSequenceClassification.from_pretrained(args.pytorch, local_files_only=True)
    pytorch.eval()
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    per_question: list[dict[str, object]] = []
    all_pt: list[np.ndarray] = []
    all_onnx: list[np.ndarray] = []
    pt_seconds = 0.0
    onnx_seconds = 0.0
    for question in questions:
        pairs = [(question, candidate) for candidate in candidates]
        pt_logits, elapsed_pt = score_pytorch(pytorch, tokenizer, pairs, args.batch_size)
        onnx_logits, elapsed_onnx = score_onnx(session, tokenizer, pairs, args.batch_size)
        pt_scores = sigmoid(pt_logits)
        onnx_scores = sigmoid(onnx_logits)
        pt_top = top_indices(pt_scores, args.top_k)
        onnx_top = top_indices(onnx_scores, args.top_k)
        per_question.append(
            {
                "question": question,
                "top1_agrees": int(np.argmax(pt_scores)) == int(np.argmax(onnx_scores)),
                "top_k_jaccard": len(pt_top & onnx_top) / len(pt_top | onnx_top),
                "mean_abs_score_delta": float(np.mean(np.abs(pt_scores - onnx_scores))),
            }
        )
        all_pt.append(pt_scores)
        all_onnx.append(onnx_scores)
        pt_seconds += elapsed_pt
        onnx_seconds += elapsed_onnx

    pt = np.concatenate(all_pt)
    onnx = np.concatenate(all_onnx)
    top1_rate = float(np.mean([item["top1_agrees"] for item in per_question]))
    jaccard = float(np.mean([item["top_k_jaccard"] for item in per_question]))
    score_delta = float(np.mean(np.abs(pt - onnx)))
    report = {
        "artifact": str(args.onnx.resolve()),
        "questions": len(questions),
        "candidates_per_question": len(candidates),
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "metrics": {
            "top1_agreement": top1_rate,
            "mean_top_k_jaccard": jaccard,
            "mean_abs_score_delta": score_delta,
            "pytorch_seconds": pt_seconds,
            "onnx_seconds": onnx_seconds,
            "speedup": pt_seconds / onnx_seconds if onnx_seconds else None,
        },
        # Scores are used by evidence thresholds, so ranking agreement alone is insufficient.
        "gate": {
            "min_top1_agreement": 0.95,
            "min_mean_top_k_jaccard": 0.90,
            "max_mean_abs_score_delta": 0.03,
        },
        "per_question": per_question,
    }
    gate = report["gate"]
    report["passed"] = bool(
        top1_rate >= gate["min_top1_agreement"]
        and jaccard >= gate["min_mean_top_k_jaccard"]
        and score_delta <= gate["max_mean_abs_score_delta"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], **report["metrics"]}, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
