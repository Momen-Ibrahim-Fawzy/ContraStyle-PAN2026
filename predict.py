#!/usr/bin/env python3
"""
Inference CLI for PAN 2026 Multi-Author Writing Style Analysis.

For each problem-X.txt in the input directory, produces
solution-problem-X.json with the predicted style-change array.

Usage (as specified by PAN task):
    python predict.py -i INPUT_DIR -o OUTPUT_DIR

The INPUT_DIR may be structured as:
  (a) Subdirectories per difficulty:
        INPUT_DIR/easy/problem-1.txt ...
        INPUT_DIR/medium/problem-1.txt ...
        INPUT_DIR/hard/problem-1.txt ...
  (b) Flat — all problem files in one directory:
        INPUT_DIR/problem-1.txt ...
      In this case the difficulty is inferred from the path or defaults
      to running all three models and using a merged fallback.

Output mirrors the input directory structure:
  OUTPUT_DIR/easy/solution-problem-1.json
  OUTPUT_DIR/medium/solution-problem-1.json
  OUTPUT_DIR/hard/solution-problem-1.json

TIRA command:
    python /app/predict.py --input $inputDataset --output $outputDir
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DIFFICULTIES, DEBERTA_DIR_TMPL, LGBM_PATH_TMPL, SSPC_PATH_TMPL,
    ENSEMBLE_PATH, WINDOW_SIZE,
)
from src.data import load_dataset, make_test_pair_records, load_problems_jsonl
from src.features import build_pair_feature_matrix
from src.ensemble import EnsembleConfig, predict_binary, aggregate_to_documents


# ─── Model loading (cached per difficulty to avoid reloading) ─────────────────

_deberta_cache: dict = {}
_lgbm_cache: dict    = {}
_sspc_cache: dict    = {}
_ensemble_cfg: EnsembleConfig = None


def _load_deberta(difficulty: str):
    if difficulty in _deberta_cache:
        return _deberta_cache[difficulty]
    model_dir = Path(DEBERTA_DIR_TMPL.format(difficulty=difficulty))
    if not model_dir.exists():
        print(f"  [WARN] DeBERTa model not found for '{difficulty}': {model_dir}")
        _deberta_cache[difficulty] = (None, None)
        return None, None
    from src.transformer_model import load_model
    model, tokenizer = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    _deberta_cache[difficulty] = (model, tokenizer)
    return model, tokenizer


def _load_lgbm(difficulty: str):
    if difficulty in _lgbm_cache:
        return _lgbm_cache[difficulty]
    lgbm_path = Path(LGBM_PATH_TMPL.format(difficulty=difficulty))
    if not lgbm_path.exists():
        print(f"  [WARN] LightGBM model not found for '{difficulty}': {lgbm_path}")
        _lgbm_cache[difficulty] = None
        return None
    from src.classical_models import StylemetricClassifier
    clf = StylemetricClassifier.load(lgbm_path)
    _lgbm_cache[difficulty] = clf
    return clf


def _load_sspc(difficulty: str):
    if difficulty in _sspc_cache:
        return _sspc_cache[difficulty]
    sspc_path = Path(SSPC_PATH_TMPL.format(difficulty=difficulty))
    if not sspc_path.exists():
        _sspc_cache[difficulty] = None
        return None
    from src.sspc_model import load_sspc
    model = load_sspc(sspc_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    _sspc_cache[difficulty] = model
    return model


def _load_ensemble() -> EnsembleConfig:
    global _ensemble_cfg
    if _ensemble_cfg is not None:
        return _ensemble_cfg
    if ENSEMBLE_PATH.exists():
        _ensemble_cfg = EnsembleConfig.load(ENSEMBLE_PATH)
        print(f"  Ensemble config loaded from {ENSEMBLE_PATH}")
    else:
        print(f"  [WARN] Ensemble config not found — using defaults")
        _ensemble_cfg = EnsembleConfig.default()
    return _ensemble_cfg


# ─── Input format helpers ─────────────────────────────────────────────────────

def _detect_format(input_dir: Path, difficulty: str) -> str:
    """Return 'prepared' if a JSONL file is present, else 'raw'."""
    if (input_dir / f"{difficulty}.jsonl").exists():
        return "prepared"
    if any(input_dir.glob("*.jsonl")):
        return "prepared"
    return "raw"


def _load_problems_for_predict(
    input_dir: Path,
    difficulty: str,
    input_format: str,
) -> list:
    """
    Load test problems from input_dir using the detected or specified format.

    raw format:      problem-*.txt files (standard TIRA / PAN layout)
    prepared format: {difficulty}.jsonl  (or any *.jsonl in the directory)
                     Each line: {"id": "X", "sentences": [...]}
                     'changes' key is ignored if present.
    """
    if input_format == "prepared":
        jsonl = input_dir / f"{difficulty}.jsonl"
        if not jsonl.exists():
            candidates = sorted(input_dir.glob("*.jsonl"))
            if not candidates:
                print(f"  [WARN] No JSONL file found in {input_dir}; falling back to raw.")
                return load_dataset(input_dir, include_truth=False, difficulty=difficulty)
            jsonl = candidates[0]
        print(f"  Loading prepared: {jsonl.name}")
        return load_problems_jsonl(jsonl, difficulty=difficulty)
    else:
        return load_dataset(input_dir, include_truth=False, difficulty=difficulty)


# ─── Prediction for one directory ────────────────────────────────────────────

def predict_directory(
    input_dir:     Path,
    output_dir:    Path,
    difficulty:    str,
    window_size:   int  = WINDOW_SIZE,
    input_format:  str  = "auto",
    symmetric_tta: bool = False,
) -> int:
    """
    Process all problem files in input_dir, write solutions to output_dir.

    input_format:  'auto' (detect), 'raw' (problem-*.txt), or 'prepared' (JSONL).
    symmetric_tta: if True, average DeBERTa predictions with left/right swapped.
    Returns number of problems processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_format == "auto":
        input_format = _detect_format(input_dir, difficulty)

    # Load problems (no truth in test mode)
    problems = _load_problems_for_predict(input_dir, difficulty, input_format)
    if not problems:
        print(f"  No problems found in {input_dir}")
        return 0

    print(f"  [{difficulty}] {len(problems)} problems")

    # Create pair records for inference
    records = make_test_pair_records(problems, difficulty=difficulty)
    if not records:
        # All single-sentence documents — output empty changes
        for prob in problems:
            _write_solution(output_dir, prob.id, [])
        return len(problems)

    # ── DeBERTa inference ────────────────────────────────────────────────
    model, tokenizer = _load_deberta(difficulty)
    if model is not None:
        from src.transformer_model import predict_proba
        deberta_probs = predict_proba(
            model, tokenizer, records,
            window_size=window_size,
            symmetric_tta=symmetric_tta,
        )
    else:
        deberta_probs = np.full(len(records), 0.5, dtype=np.float32)

    # ── LightGBM inference ───────────────────────────────────────────────
    clf = _load_lgbm(difficulty)
    if clf is not None:
        X = build_pair_feature_matrix(records, window_size=window_size)
        lgbm_probs = clf.predict_proba(X)
    else:
        lgbm_probs = np.full(len(records), 0.5, dtype=np.float32)

    # ── SSPC inference ───────────────────────────────────────────────────
    sspc_model = _load_sspc(difficulty)
    if sspc_model is not None:
        from src.sspc_model import predict_sspc_from_records
        sspc_probs = predict_sspc_from_records(sspc_model, records)
    else:
        sspc_probs = None

    # ── Ensemble + threshold ─────────────────────────────────────────────
    ens_cfg = _load_ensemble()
    binary_preds = predict_binary(
        deberta_probs, lgbm_probs, difficulty, ens_cfg,
        sspc_probs=sspc_probs, records=records,
    )

    # ── Reassemble per document ──────────────────────────────────────────
    doc_changes = aggregate_to_documents(records, binary_preds)

    # ── Write solutions ──────────────────────────────────────────────────
    for prob in problems:
        changes = doc_changes.get(prob.id, [])
        _write_solution(output_dir, prob.id, changes)

    return len(problems)


def _write_solution(output_dir: Path, problem_id: str, changes: list) -> None:
    out_path = output_dir / f"solution-problem-{problem_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"changes": [int(c) for c in changes]}, f)


# ─── Input structure detection ────────────────────────────────────────────────

def _detect_structure(input_dir: Path) -> dict:
    """
    Returns {difficulty: Path} for the datasets found in input_dir.

    Recognises layouts:
      1. Subdirectory (raw):      input_dir/{easy,medium,hard}/problem-*.txt
      2. Subdirectory (prepared): input_dir/{easy,medium,hard}/{diff}.jsonl
      3. train/ subdir:           input_dir/{easy,medium,hard}/train/problem-*.txt
      4. TIRA layout:             input_dir/input-data/{easy,medium,hard}/train/problem-*.txt
      5. Flat (raw):              input_dir/problem-*.txt
      6. Flat (prepared):         input_dir/{diff}.jsonl
    """
    datasets = {}

    # TIRA layout: input-data/{diff}/{split}/problem-*.txt
    input_data_dir = input_dir / "input-data"
    if input_data_dir.is_dir():
        for diff in DIFFICULTIES:
            diff_dir = input_data_dir / diff
            if not diff_dir.is_dir():
                continue
            # Check split subdirs first, then the diff dir itself
            for sub in ["train", "validation", "test", ""]:
                candidate = diff_dir / sub if sub else diff_dir
                if candidate.is_dir() and any(candidate.glob("problem-*.txt")):
                    datasets[diff] = candidate
                    break

    if not datasets:
        # Standard subdirectory layout (raw or prepared), with optional train/ subdir
        for diff in DIFFICULTIES:
            diff_dir = input_dir / diff
            if not diff_dir.is_dir():
                continue
            if any(diff_dir.glob("problem-*.txt")) or any(diff_dir.glob("*.jsonl")):
                datasets[diff] = diff_dir
                continue
            # Files nested in train/ or validation/
            for sub in ["train", "validation", "test"]:
                candidate = diff_dir / sub
                if candidate.is_dir() and any(candidate.glob("problem-*.txt")):
                    datasets[diff] = candidate
                    break

    if not datasets:
        # Flat layout — check for raw or prepared files
        has_raw      = any(input_dir.glob("problem-*.txt"))
        jsonl_diffs  = [d for d in DIFFICULTIES if (input_dir / f"{d}.jsonl").exists()]
        any_jsonl    = bool(jsonl_diffs) or bool(list(input_dir.glob("*.jsonl")))

        if jsonl_diffs:
            for d in jsonl_diffs:
                datasets[d] = input_dir
        elif any_jsonl or has_raw:
            # Try to detect difficulty from directory path
            name = str(input_dir).lower()
            detected = next((d for d in DIFFICULTIES if d in name), None)
            if detected:
                datasets[detected] = input_dir
            else:
                print(f"  [WARN] Cannot detect difficulty from path '{input_dir}'. "
                      f"Defaulting to 'hard' (no topic signal assumed).")
                datasets["hard"] = input_dir

    return datasets


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict style changes for PAN 2026 multi-author writing analysis"
    )
    parser.add_argument("-i", "--input",  required=True, type=Path,
                        help="Input directory containing problem-*.txt files")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Output directory for solution-problem-*.json files")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help="Context window size (default: 1)")
    parser.add_argument(
        "--input-format", choices=["auto", "raw", "prepared"], default="auto",
        help=(
            "Input format: 'raw' = problem-*.txt (standard PAN/TIRA layout), "
            "'prepared' = {difficulty}.jsonl produced by prepare_data.py, "
            "'auto' = detect automatically (default)."
        ),
    )
    parser.add_argument(
        "--tta", action="store_true",
        help=(
            "Symmetric Test-Time Augmentation: also score each sentence pair with "
            "left/right contexts swapped, then average with the original prediction. "
            "Style comparison is symmetric → averaging removes the direction bias. "
            "Typically gives +0.5–2%% F1 at the cost of 2× DeBERTa inference time."
        ),
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    import torch
    if torch.cuda.is_available():
        device_info = f"GPU — {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
    else:
        device_info = "CPU (no CUDA GPU detected — inference will be slow)"
    print(f"Device: {device_info}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    datasets = _detect_structure(input_dir)
    if not datasets:
        print(f"ERROR: No problem files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found datasets: {list(datasets.keys())}")

    total = 0
    for difficulty, data_path in sorted(datasets.items()):
        print(f"\nProcessing '{difficulty}'...")
        out_subdir = output_dir / difficulty if len(datasets) > 1 else output_dir
        n = predict_directory(
            data_path, out_subdir, difficulty,
            window_size=args.window_size,
            input_format=args.input_format,
            symmetric_tta=args.tta,
        )
        total += n
        print(f"  Written {n} solution files to {out_subdir}")

    print(f"\nDone. Total: {total} problems processed.")


if __name__ == "__main__":
    main()