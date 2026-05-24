#!/usr/bin/env python3
"""
Multi-year data preparation for PAN multi-author writing style analysis.

Loads training + validation data from PAN 2022-2026, deduplicates across
years using SHA-256 hashes, and saves the merged results as six JSONL files:

  data_prepared/
    train_easy.jsonl     train_medium.jsonl     train_hard.jsonl
    val_easy.jsonl       val_medium.jsonl       val_hard.jsonl

Each line in a JSONL file is one document:
  {"id": "...", "sentences": [...], "changes": [...], "difficulty": "..."}

train.py auto-detects these files and loads from them instead of re-reading
all individual problem-*.txt files from every year directory.

Usage:
    python prepare_data.py
    python prepare_data.py --difficulties easy medium hard
    python prepare_data.py --no-2022
    python prepare_data.py --output-dir /custom/path
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATA_2022, DATA_2023, DATA_2024, DATA_2025, DATA_2026,
    DIFFICULTIES, PREPARED_DIR, PREPARED_TRAIN_TMPL, PREPARED_VAL_TMPL,
)
from src.data import (
    load_2022, load_2023, load_2024, load_2025, load_2026,
    make_pair_records, save_problems_jsonl,
)


def _doc_hash(sentences: list) -> str:
    return hashlib.sha256("\n".join(sentences).encode("utf-8")).hexdigest()


def load_and_dedup(difficulty: str, split: str, args) -> tuple:
    """
    Load data from all selected years for one difficulty / split.
    Deduplicate by SHA-256 hash of full document text.
    Returns (problems, n_duplicates_removed).
    """
    all_problems = []
    seen_hashes  = set()
    n_dupes      = 0

    sources = []
    if not args.no_2026 and DATA_2026.exists():
        sources.append(("2026", lambda: load_2026(DATA_2026, difficulty, split)))
    if not args.no_2025 and DATA_2025.exists():
        sources.append(("2025", lambda: load_2025(DATA_2025, difficulty, split)))
    if not args.no_2024 and DATA_2024.exists():
        sources.append(("2024", lambda: load_2024(DATA_2024, difficulty, split)))
    if not args.no_2023 and DATA_2023.exists():
        sources.append(("2023", lambda: load_2023(DATA_2023, difficulty, split)))
    # 2022 has no validation split — contribute to train only
    if not args.no_2022 and DATA_2022.exists() and split == "train":
        sources.append(("2022", lambda: load_2022(DATA_2022, difficulty, "train")))

    for year, loader in sources:
        try:
            probs = loader()
        except Exception as e:
            print(f"  WARNING: Could not load {year}/{difficulty}/{split}: {e}")
            continue

        kept = 0
        for p in probs:
            h = _doc_hash(p.sentences)
            if h in seen_hashes:
                n_dupes += 1
                continue
            seen_hashes.add(h)
            # Prefix id with year so doc_ids stay unique across years.
            # smooth_per_document groups by doc_id — collisions would merge
            # sequences from different documents during ensemble calibration.
            p.id = f"{year}_{p.id}"
            all_problems.append(p)
            kept += 1

        print(f"  {year}/{difficulty}/{split}: {len(probs):,} loaded, "
              f"{kept:,} kept ({len(probs)-kept} dupes removed)")

    return all_problems, n_dupes


def _pair_stats(problems: list, difficulty: str) -> str:
    pairs = make_pair_records(problems, difficulty)
    if not pairs:
        return "0 docs / 0 pairs"
    labels = [p.label for p in pairs]
    pos    = sum(labels)
    # Count directly from problems — doc_ids are NOT unique across years so
    # len(set(rec.doc_id for rec in pairs)) would under-count.
    n_docs = sum(
        1 for p in problems
        if p.changes is not None
        and len(p.sentences) >= 2
        and len(p.changes) == len(p.sentences) - 1
    )
    return (f"{n_docs:,} docs / {len(labels):,} pairs / "
            f"{100*pos/max(len(labels),1):.1f}% positive")


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-year PAN data into JSONL files")
    parser.add_argument("--difficulties", nargs="+", default=DIFFICULTIES, choices=DIFFICULTIES)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (default: data_prepared/ next to solution/)")
    parser.add_argument("--no-2022", action="store_true", help="Exclude PAN 2022 data")
    parser.add_argument("--no-2023", action="store_true", help="Exclude PAN 2023 data")
    parser.add_argument("--no-2024", action="store_true", help="Exclude PAN 2024 data")
    parser.add_argument("--no-2025", action="store_true", help="Exclude PAN 2025 data")
    parser.add_argument("--no-2026", action="store_true", help="Exclude PAN 2026 data")
    args = parser.parse_args()

    out_dir = args.output_dir or PREPARED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override TMPL paths if custom output dir is given
    train_tmpl = str(out_dir / "train_{difficulty}.jsonl")
    val_tmpl   = str(out_dir / "val_{difficulty}.jsonl")

    print("=" * 60)
    print("PAN Multi-Author Writing Style Analysis — Data Preparation")
    print("=" * 60)
    print(f"Output directory: {out_dir}\n")

    grand_train = grand_val = grand_dupes = 0

    for difficulty in args.difficulties:
        print(f"\n{'-'*60}")
        print(f"Difficulty: {difficulty.upper()}")
        print("-"*60)

        print("Loading + deduplicating training data:")
        train_probs, train_dupes = load_and_dedup(difficulty, "train", args)

        print("Loading + deduplicating validation data:")
        val_probs, val_dupes = load_and_dedup(difficulty, "validation", args)

        # Save JSONL
        train_path = Path(train_tmpl.format(difficulty=difficulty))
        val_path   = Path(val_tmpl.format(difficulty=difficulty))

        n_train = save_problems_jsonl(train_probs, train_path)
        n_val   = save_problems_jsonl(val_probs,   val_path)

        print(f"\nSaved:")
        print(f"  {train_path.name}: {n_train:,} docs  ({_pair_stats(train_probs, difficulty)})")
        print(f"  {val_path.name}:   {n_val:,} docs  ({_pair_stats(val_probs, difficulty)})")
        print(f"  Duplicates removed: {train_dupes + val_dupes}")

        grand_train += n_train
        grand_val   += n_val
        grand_dupes += train_dupes + val_dupes

    print(f"\n{'='*60}")
    print(f"Grand total — train: {grand_train:,} docs | val: {grand_val:,} docs")
    print(f"Total duplicates removed: {grand_dupes}")
    print("="*60)
    print(f"\nPrepared data saved to: {out_dir}")
    print("Run train.py — it will auto-detect these files and load from them.")


if __name__ == "__main__":
    main()