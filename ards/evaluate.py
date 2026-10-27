#!/usr/bin/env python3
"""Evaluation Pipeline Runner for ARDS Models.

Evaluates trained ARDS model checkpoints across:
1. 8 Public Zero-Shot Benchmark Tasks:
   arc_easy, arc_challenge, hellaswag, piqa, openbookqa, winogrande, boolq, lambada
2. 4 Bounded-Context Recall Tasks:
   needle_passkey, associative_recall, multi_needle, latest_value
3. Purity Gate: Strictly validates that evaluated models were trained with ARDS,
   refusing any model that employed cosine schedules.
4. Smoke Testing: Supports --limit 1 for fast pipeline verification.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repository root and src/ are in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from algebraic.benchmarking import (
    PUBLIC_TASKS,
    RECALL_PROTOCOL,
    ZERO_SHOT_PROTOCOL,
    Scorer,
    load_model,
    provenance,
    read_dataset,
    sha256_file,
    write_json,
)
from algebraic.recall_benchmarks import (
    TASKS as RECALL_TASKS,
    generated_metrics,
    summarize as summarize_recall,
    validate_example,
)
from ards.configs import validate_schedule_not_cosine


def validate_ards_checkpoint_purity(run_dir: Path, allow_cosine: bool = False) -> Dict[str, Any]:
    """Inspect checkpoint metadata to guarantee algebraic schedule purity unless allow_cosine is True."""
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata in checkpoint run: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    args = meta.get("args", {})
    schedule = args.get("schedule", "ards")

    # Strict purity gate: reject any cosine artifact when evaluating pure ARDS runs
    if not allow_cosine and "cos" in str(schedule).lower():
        raise ValueError(
            f"Refusing to evaluate checkpoint at {run_dir}: "
            f"Trained with forbidden schedule {schedule!r}. "
            f"ARDS evaluation requires algebraic schedule purity."
        )
    return meta


def evaluate_zero_shot(
    scorer: Scorer,
    benchmarks_dir: Path,
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute zero-shot evaluation on public benchmark tasks."""
    selected_tasks = tasks or list(PUBLIC_TASKS)
    results, predictions, datasets = {}, [], {}
    t0 = time.monotonic()

    for name in selected_tasks:
        rows, datasets[name] = read_dataset(benchmarks_dir, name, limit)
        truncated_before = scorer.truncated_pairs
        if name == "lambada":
            lps, hits = scorer.score([(r["context"], r["target"]) for r in rows])
            details = [
                {
                    "id": r.get("id", f"{name}/{i}"),
                    "task": name,
                    "correct": hit,
                    "target_mean_logprob": lp,
                }
                for i, (r, lp, hit) in enumerate(zip(rows, lps, hits))
            ]
        else:
            pairs = [(r["context"], choice) for r in rows for choice in r["choices"]]
            lps, _ = scorer.score(pairs)
            details, offset = [], 0
            for i, row in enumerate(rows):
                scores = lps[offset : offset + len(row["choices"])]
                predicted = int(np.argmax(scores))
                details.append({
                    "id": row.get("id", f"{name}/{i}"),
                    "task": name,
                    "answer": row["answer"],
                    "prediction": predicted,
                    "correct": predicted == row["answer"],
                    "choice_mean_logprobs": scores,
                })
                offset += len(scores)

        correct = sum(int(d["correct"]) for d in details)
        results[name] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else 0.0,
            "truncated_pairs": scorer.truncated_pairs - truncated_before,
        }
        predictions.extend(details)

    report = {
        "results": results,
        "dataset_metadata": {"datasets": datasets},
        "limited": limit is not None or any(d.get("limited", False) for d in datasets.values()),
        "limit": limit,
        "seconds": time.monotonic() - t0,
    }
    return report, predictions


def evaluate_recall(
    scorer: Scorer,
    recall_dir: Path,
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute greedy autoregressive retrieval evaluation on recall benchmark tasks."""
    selected_tasks = tasks or list(RECALL_TASKS)
    meta = json.loads((recall_dir / "metadata.json").read_text())
    if meta.get("protocol") != RECALL_PROTOCOL:
        raise ValueError(f"Unexpected recall protocol: {meta.get('protocol')}")

    results, predictions, datasets = {}, [], {}
    t0 = time.monotonic()

    for task in selected_tasks:
        rows, datasets[task] = read_dataset(recall_dir, task, limit)
        for row in rows:
            validate_example(row, scorer.enc, scorer.T)
        budgets = {r["max_new_tokens"] for r in rows}
        if budgets != {16}:
            raise ValueError("Recall protocol requires strictly a 16-token generation budget")

        generated = scorer.greedy([r["prompt_token_ids"] for r in rows], max_new_tokens=16)
        pairs = [(r["prompt_token_ids"], tokens) for r in rows for tokens in r["choice_token_ids"]]
        lps, hits = scorer.score_token_pairs(pairs, allow_truncation=False)

        details = []
        for i, (row, output) in enumerate(zip(rows, generated)):
            scores = lps[4 * i : 4 * i + 4]
            recognition = int(np.argmax(scores))
            details.append({
                key: row[key]
                for key in (
                    "id", "task", "trial", "condition",
                    "context_budget", "requested_depth", "prompt_tokens", "actual_depths"
                )
            })
            details[-1].update({
                **generated_metrics(output["text"], row["expected_codes"]),
                "generated_text": output["text"],
                "generated_token_ids": output["token_ids"],
                "expected_codes": row["expected_codes"],
                "target_prefix_match": hits[4 * i + row["answer"]],
                "target_mean_logprob": scores[row["answer"]],
                "recognition_prediction": recognition,
                "recognition_correct": recognition == row["answer"],
                "choice_mean_logprobs": scores,
                "evidence_truncated": False,
            })
        results[task] = summarize_recall(details)
        predictions.extend(details)

    report = {
        "results": results,
        "dataset_metadata": {**meta, "datasets": datasets},
        "limited": limit is not None or any(d.get("limited", False) for d in datasets.values()),
        "limit": limit,
        "seconds": time.monotonic() - t0,
    }
    return report, predictions


def run_ards_evaluation(
    run_dir: str | Path,
    benchmarks_dir: Optional[str | Path] = None,
    recall_dir: Optional[str | Path] = None,
    suite: str = "all",
    limit: Optional[int] = None,
    batch: int = 8,
    out_dir: Optional[str | Path] = None,
    allow_cosine: bool = False,
) -> Dict[str, Any]:
    """High-level runner evaluating an ARDS model run."""
    run_path = Path(run_dir)
    validate_ards_checkpoint_purity(run_path, allow_cosine=allow_cosine)

    bench_path = Path(benchmarks_dir or Path.home() / "data/evals")
    rec_path = Path(recall_dir or Path.home() / "data/recall_llama32k")
    target_out = Path(out_dir or run_path)
    target_out.mkdir(parents=True, exist_ok=True)

    cfg, params = load_model(run_path)
    scorer = Scorer(cfg, params, batch)

    summary: Dict[str, Any] = {"run": str(run_path), "status": "completed"}

    if suite in ("zero_shot", "all"):
        zs_report, zs_preds = evaluate_zero_shot(scorer, bench_path, limit=limit)
        zs_report.update(provenance(run_path, scorer, ZERO_SHOT_PROTOCOL))
        pred_file = target_out / f"{run_path.name}_zero_shot.predictions.jsonl"
        pred_file.write_text("".join(json.dumps(r) + "\n" for r in zs_preds))
        zs_report["predictions"] = str(pred_file)
        zs_report["predictions_sha256"] = sha256_file(pred_file)
        write_json(target_out / f"{run_path.name}_zero_shot.json", zs_report)
        summary["zero_shot"] = zs_report["results"]

    if suite in ("recall", "all"):
        rec_report, rec_preds = evaluate_recall(scorer, rec_path, limit=limit)
        rec_report.update(provenance(run_path, scorer, RECALL_PROTOCOL))
        pred_file = target_out / f"{run_path.name}_recall.predictions.jsonl"
        pred_file.write_text("".join(json.dumps(r) + "\n" for r in rec_preds))
        rec_report["predictions"] = str(pred_file)
        rec_report["predictions_sha256"] = sha256_file(pred_file)
        write_json(target_out / f"{run_path.name}_recall.json", rec_report)
        summary["recall"] = rec_report["results"]

    return summary


evaluate_ards_run = run_ards_evaluation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="path to completed run directory")
    p.add_argument("--benchmarks", type=Path, default=Path.home() / "data/evals")
    p.add_argument("--recall-benchmarks", type=Path, default=Path.home() / "data/recall_llama32k")
    p.add_argument("--suite", choices=("all", "zero_shot", "recall"), default="all")
    p.add_argument("--limit", type=int, default=None, help="limit rows for fast smoke testing")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--allow-cosine", action="store_true", help="permit checkpoints trained with cosine schedule")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    summary = run_ards_evaluation(
        run_dir=args.run,
        benchmarks_dir=args.benchmarks,
        recall_dir=args.recall_benchmarks,
        suite=args.suite,
        limit=args.limit,
        batch=args.batch,
        out_dir=args.out,
        allow_cosine=args.allow_cosine,
    )
    print(json.dumps(summary, indent=2))
    return 0


evaluate_main = main

if __name__ == "__main__":
    raise SystemExit(main())
