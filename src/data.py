"""
Data loading utilities for PAN multi-author writing style analysis.

Each problem is a .txt file where each line is one sentence.
Each truth file is a JSON with keys 'authors' and 'changes'
(binary array of length len(sentences) - 1).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Problem:
    id: str
    sentences: list
    changes: Optional[list] = None   # len == len(sentences) - 1, or None for test
    authors: Optional[int] = None
    difficulty: Optional[str] = None


@dataclass
class PairRecord:
    """A single consecutive-sentence pair extracted from a problem."""
    doc_id: str
    pair_idx: int       # index i: predicting change between sent_i and sent_{i+1}
    sentences: list     # all sentences of the document (for window context)
    label: int          # 0 or 1; -1 when truth is not available
    difficulty: Optional[str] = None


def read_sentences(path: Path) -> list:
    """Read one sentence per line from a problem .txt file."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        content = f.read()
    # Strip each line to remove \r (Windows line endings) and leading/trailing
    # whitespace; then drop empty lines.  Preserves internal punctuation and
    # spacing because those are stylometric signals.
    return [s for s in (line.strip() for line in content.split("\n")) if s]


def read_truth(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(
    dir_path: Path,
    include_truth: bool = True,
    difficulty: Optional[str] = None,
) -> list:
    """
    Load all problems from a single directory (one difficulty / split).

    Returns list of Problem objects sorted by problem id numerically.
    """
    dir_path = Path(dir_path)
    problems = []

    for txt_path in dir_path.glob("problem-*.txt"):
        pid = txt_path.stem.replace("problem-", "")
        sentences = read_sentences(txt_path)
        if not sentences:
            continue

        changes = None
        authors = None
        if include_truth:
            truth_path = dir_path / f"truth-problem-{pid}.json"
            if truth_path.exists():
                truth = read_truth(truth_path)
                changes = truth.get("changes", [])
                authors = truth.get("authors")

        problems.append(Problem(
            id=pid,
            sentences=sentences,
            changes=changes,
            authors=authors,
            difficulty=difficulty,
        ))

    # Sort numerically (problem ids are integers)
    problems.sort(key=lambda p: int(p.id) if p.id.isdigit() else p.id)
    return problems


def make_pair_records(problems: list, difficulty: Optional[str] = None) -> list:
    """
    Explode a list of Problem objects into individual PairRecord objects.

    One record per consecutive sentence pair per document.
    Only includes problems that have ground truth (changes != None).
    """
    records = []
    for prob in problems:
        if prob.changes is None:
            continue
        if len(prob.sentences) < 2:
            continue
        n_pairs = len(prob.sentences) - 1
        if len(prob.changes) != n_pairs:
            continue  # malformed, skip

        diff = difficulty or prob.difficulty
        for i in range(n_pairs):
            records.append(PairRecord(
                doc_id=prob.id,
                pair_idx=i,
                sentences=prob.sentences,
                label=int(prob.changes[i]),
                difficulty=diff,
            ))
    return records


def make_test_pair_records(problems: list, difficulty: Optional[str] = None) -> list:
    """
    Explode test problems (no truth) into PairRecord objects.
    label is set to -1 (unknown).
    """
    records = []
    for prob in problems:
        if len(prob.sentences) < 2:
            continue
        diff = difficulty or prob.difficulty
        n_pairs = len(prob.sentences) - 1
        for i in range(n_pairs):
            records.append(PairRecord(
                doc_id=prob.id,
                pair_idx=i,
                sentences=prob.sentences,
                label=-1,
                difficulty=diff,
            ))
    return records


def get_pair_texts(record: "PairRecord", window_size: int = 1) -> tuple:
    """
    Build (left_text, right_text) for a pair record using window context.

    left_text  = sentences[max(0, i - window_size) : i + 1]  joined by '\\n'
    right_text = sentences[i + 1 : i + 1 + window_size + 1] joined by '\\n'
    """
    i = record.pair_idx
    sents = record.sentences
    n = len(sents)

    left_sents  = sents[max(0, i - window_size): i + 1]
    right_sents = sents[i + 1: min(n, i + 1 + window_size + 1)]

    left_text  = "\n".join(left_sents)
    right_text = "\n".join(right_sents)
    return left_text, right_text


# ─── Multi-year data loaders ──────────────────────────────────────────────────

def load_2026(data_root: Path, difficulty: str, split: str) -> list:
    """Load PAN 2026 data. split ∈ {'train', 'validation'}."""
    d = data_root / difficulty / split
    return load_dataset(d, include_truth=(split != "test"), difficulty=difficulty)


def load_2025(data_root: Path, difficulty: str, split: str) -> list:
    """Load PAN 2025 data. Same easy/medium/hard structure as 2026."""
    d = data_root / difficulty / split
    return load_dataset(d, include_truth=(split != "test"), difficulty=difficulty)


def load_2024(data_root: Path, difficulty: str, split: str) -> list:
    """Load PAN 2024 data. Same easy/medium/hard structure."""
    d = data_root / difficulty / split
    return load_dataset(d, include_truth=(split != "test"), difficulty=difficulty)


def load_2023(data_root: Path, difficulty: str, split: str) -> list:
    """
    Load PAN 2023 data. Uses dataset1/2/3 naming mapped to easy/medium/hard.
    split ∈ {'train', 'validation'}
    """
    dataset_map = {"easy": "1", "medium": "2", "hard": "3"}
    ds = dataset_map[difficulty]
    folder_name = f"pan23-multi-author-analysis-dataset{ds}"
    split_folder = f"{folder_name}-{split}"
    d = data_root / folder_name / split_folder
    return load_dataset(d, include_truth=True, difficulty=difficulty)


def load_2022(data_root: Path, difficulty: str, split: str) -> list:
    """
    Load PAN 2022 data. Uses dataset1/2/3 mapped to easy/medium/hard.
    Note: 2022 has paragraph-level changes (compatible format, just fewer pairs).
    """
    dataset_map = {"easy": "1", "medium": "2", "hard": "3"}
    ds = dataset_map[difficulty]
    d = data_root / f"dataset{ds}" / split
    if not d.exists():
        return []
    return load_dataset(d, include_truth=True, difficulty=difficulty)


def load_all_years_for_difficulty(
    difficulty: str,
    data_2026: Path,
    data_2025: Path,
    data_2024: Path,
    data_2023: Path,
    data_2022: Path,
    split: str = "train",
    include_2026: bool = True,
    include_2025: bool = True,
    include_2024: bool = True,
    include_2023: bool = True,
    include_2022: bool = True,
) -> list:
    """Combine training data from all available years for one difficulty level."""
    all_problems = []

    if include_2026 and data_2026.exists():
        probs = load_2026(data_2026, difficulty, split)
        all_problems.extend(probs)
        print(f"  2026/{difficulty}/{split}: {len(probs)} problems")

    if include_2025 and data_2025.exists():
        try:
            probs = load_2025(data_2025, difficulty, split)
            all_problems.extend(probs)
            print(f"  2025/{difficulty}/{split}: {len(probs)} problems")
        except Exception as e:
            print(f"  2025 load failed: {e}")

    if include_2024 and data_2024.exists():
        try:
            probs = load_2024(data_2024, difficulty, split)
            all_problems.extend(probs)
            print(f"  2024/{difficulty}/{split}: {len(probs)} problems")
        except Exception as e:
            print(f"  2024 load failed: {e}")

    if include_2023 and data_2023.exists():
        try:
            probs = load_2023(data_2023, difficulty, split)
            all_problems.extend(probs)
            print(f"  2023/{difficulty}/{split}: {len(probs)} problems")
        except Exception as e:
            print(f"  2023 load failed: {e}")

    if include_2022 and data_2022.exists():
        try:
            probs = load_2022(data_2022, difficulty, "train")
            all_problems.extend(probs)
            print(f"  2022/{difficulty}/train: {len(probs)} problems")
        except Exception as e:
            print(f"  2022 load failed: {e}")

    return all_problems


# ─── JSONL persistence (prepared data format) ────────────────────────────────

def save_problems_jsonl(problems: list, path: Path) -> int:
    """
    Write a list of Problem objects to a JSONL file (one document per line).

    Returns the number of records written.
    Used by prepare_data.py to persist the merged + deduplicated training data
    so that train.py can load it in a single pass instead of re-reading every
    individual problem-*.txt file from all years.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for prob in problems:
            obj: dict = {
                "id":         prob.id,
                "sentences":  prob.sentences,
                "difficulty": prob.difficulty,
            }
            if prob.changes is not None:
                obj["changes"] = prob.changes
            if prob.authors is not None:
                obj["authors"] = prob.authors
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(problems)


def load_problems_jsonl(path: Path, difficulty: Optional[str] = None) -> list:
    """
    Load Problem objects from a JSONL file written by save_problems_jsonl.

    Also accepts test-time JSONL (no 'changes' key) — those problems will have
    changes=None, suitable for make_test_pair_records().
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared data file not found: {path}")
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            problems.append(Problem(
                id=d["id"],
                sentences=d["sentences"],
                changes=d.get("changes"),
                authors=d.get("authors"),
                difficulty=difficulty or d.get("difficulty"),
            ))
    return problems


def group_records_by_doc(records: list) -> dict:
    """Group PairRecord list by doc_id for document-level post-processing."""
    groups = {}
    for r in records:
        groups.setdefault(r.doc_id, []).append(r)
    for doc_id in groups:
        groups[doc_id].sort(key=lambda r: r.pair_idx)
    return groups