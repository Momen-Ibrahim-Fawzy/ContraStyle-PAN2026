#!/usr/bin/env python3
"""
Evaluation script: computes F1-macro for style change detection predictions.

Usage:
    python evaluate.py \
        --predictions OUTPUT_DIR \
        --ground-truth INPUT_DIR

OUTPUT_DIR should mirror the structure written by predict.py.
INPUT_DIR should be the data directory containing truth-problem-*.json files
(same structure as what was passed to predict.py).

The script reports F1-macro per difficulty level and overall.
"""
import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DIFFICULTIES


def load_solutions(directory: Path) -> dict:
    """Load solution-problem-*.json files. Returns {problem_id: [changes]}."""
    solutions = {}
    for p in sorted(directory.glob("solution-problem-*.json")):
        pid = p.stem.replace("solution-problem-", "")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        solutions[pid] = data.get("changes", [])
    return solutions


def load_truths(directory: Path) -> dict:
    """Load truth-problem-*.json files. Returns {problem_id: [changes]}."""
    truths = {}
    for p in sorted(directory.glob("truth-problem-*.json")):
        pid = p.stem.replace("truth-problem-", "")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        truths[pid] = data.get("changes", [])
    return truths


def compute_f1(truths: dict, solutions: dict) -> dict:
    """
    Compute F1-macro, precision and recall for matched problems.

    Returns dict with 'f1', 'precision', 'recall', 'n_problems', 'n_pairs'.
    """
    all_true, all_pred = [], []

    for pid, true_changes in truths.items():
        if pid not in solutions:
            # Missing prediction — treat as all zeros (no style change)
            all_true.extend(true_changes)
            all_pred.extend([0] * len(true_changes))
            continue

        pred_changes = solutions[pid]
        n = len(true_changes)
        # Pad or truncate predictions to match truth length
        pred_changes = (list(pred_changes) + [0] * n)[:n]

        all_true.extend(true_changes)
        all_pred.extend(pred_changes)

    if not all_true:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_problems": 0, "n_pairs": 0}

    f1   = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    prec = float(precision_score(all_true, all_pred, average="macro", zero_division=0))
    rec  = float(recall_score(all_true, all_pred, average="macro", zero_division=0))

    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "n_problems": len(truths),
        "n_pairs": len(all_true),
        "n_matched": sum(1 for pid in truths if pid in solutions),
    }


def evaluate_flat(pred_dir: Path, truth_dir: Path, label: str = "") -> dict:
    solutions = load_solutions(pred_dir)
    truths    = load_truths(truth_dir)

    if not truths:
        return {}

    metrics = compute_f1(truths, solutions)

    tag = f"[{label}]" if label else ""
    print(f"  {tag} F1-macro={metrics['f1']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
          f"| {metrics.get('n_matched',0)}/{metrics['n_problems']} problems "
          f"| {metrics['n_pairs']:,} pairs")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate style change detection predictions")
    parser.add_argument("--predictions",  "-p", required=True, type=Path,
                        help="Directory containing solution-problem-*.json files")
    parser.add_argument("--ground-truth", "-g", required=True, type=Path,
                        help="Directory containing truth-problem-*.json files")
    args = parser.parse_args()

    pred_dir  = Path(args.predictions)
    truth_dir = Path(args.ground_truth)

    print(f"Predictions:  {pred_dir}")
    print(f"Ground truth: {truth_dir}")
    print()

    all_results = {}

    # Check if structured with difficulty subdirectories
    has_subdirs_pred  = any((pred_dir / d).is_dir()  for d in DIFFICULTIES)
    has_subdirs_truth = any((truth_dir / d).is_dir() for d in DIFFICULTIES)

    if has_subdirs_pred and has_subdirs_truth:
        # Per-difficulty evaluation
        all_true_flat, all_pred_flat = [], []

        for difficulty in DIFFICULTIES:
            p_sub = pred_dir  / difficulty
            t_sub = truth_dir / difficulty
            if not p_sub.exists() or not t_sub.exists():
                continue
            metrics = evaluate_flat(p_sub, t_sub, label=difficulty)
            if metrics:
                all_results[difficulty] = metrics

        # Also check subdirectory structures with train/validation splits
        for difficulty in DIFFICULTIES:
            for split in ["validation", "train"]:
                p_sub = pred_dir  / difficulty
                t_sub = truth_dir / difficulty / split
                if not p_sub.exists() or not t_sub.exists():
                    continue
                evaluate_flat(p_sub, t_sub, label=f"{difficulty}/{split}")

    else:
        # Flat structure
        metrics = evaluate_flat(pred_dir, truth_dir)
        if metrics:
            all_results["flat"] = metrics

    # Overall summary
    if len(all_results) > 1:
        all_f1s = [m["f1"] for m in all_results.values() if "f1" in m]
        print(f"\n  OVERALL mean F1-macro: {sum(all_f1s) / len(all_f1s):.4f}")


if __name__ == "__main__":
    main()