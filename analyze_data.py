#!/usr/bin/env python3
"""
Data analysis for PAN 2026 multi-author writing style analysis.

Prints per-year / per-difficulty statistics to the console and saves a
comprehensive Markdown report to data_analysis_report.md.

Usage:
    python analyze_data.py
    python analyze_data.py --year 2026
    python analyze_data.py --difficulty hard
    python analyze_data.py --no-report      # console output only
    python analyze_data.py --no-deep        # skip token-budget + feature analysis (fast)
"""
import argparse
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATA_2026, DATA_2025, DATA_2024, DATA_2023, DATA_2022,
    DIFFICULTIES, MAX_LENGTH,
)
from src.data import load_2022, load_2023, load_2024, load_2025, load_2026, make_pair_records
from src.features import extract_sentence_features, FEATURE_NAMES_SENT, N_SENT_FEATURES

REPORT_PATH = Path(__file__).parent / "data_analysis_report.md"
TOKEN_BUDGETS = [128, 192, 256, 384, 512]


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PercDist:
    """Percentile distribution summary for a numerical array."""
    p10: float; p25: float; p50: float; p75: float
    p90: float; p95: float; p99: float
    max_val: float; std: float; mean: float


@dataclass
class YearStats:
    year: str
    n_docs: int
    n_pairs: int
    n_pos: int
    mean_doc_len: float
    std_doc_len: float
    min_doc_len: int = 0
    max_doc_len: int = 0
    median_doc_len: float = 0.0
    mean_word_len: float = 0.0
    std_word_len: float = 0.0
    doc_len_dist: Optional[PercDist] = None
    sent_len_dist: Optional[PercDist] = None

    @property
    def pos_rate(self) -> float:
        return self.n_pos / max(self.n_pairs, 1)


@dataclass
class TokenBudgetStats:
    """% of window-context pairs that fit within each token budget."""
    pct_fit: Dict[int, float] = field(default_factory=dict)
    mean_tokens: float = 0.0
    p50_tokens: float = 0.0
    p90_tokens: float = 0.0
    p95_tokens: float = 0.0
    p99_tokens: float = 0.0
    max_tokens: int = 0


@dataclass
class FeatureDiffStats:
    """Mean |feat_i - feat_j| for style-change vs no-change pairs."""
    name: str
    mean_no_change: float
    mean_change: float
    discrimination: float   # mean_change - mean_no_change; higher = more useful


@dataclass
class SplitCompStats:
    """PAN 2026 train vs validation comparison per difficulty."""
    difficulty: str
    train_n_docs: int
    train_n_pairs: int
    train_pos_rate: float
    val_n_docs: int
    val_n_pairs: int
    val_pos_rate: float
    train_mean_doc_len: float
    val_mean_doc_len: float
    train_mean_sent_len: float
    val_mean_sent_len: float


@dataclass
class DuplicateGroup:
    years: tuple
    count: int


@dataclass
class DifficultyReport:
    difficulty: str
    year_stats: List[YearStats] = field(default_factory=list)
    duplicates: List[DuplicateGroup] = field(default_factory=list)
    token_budget: Optional[TokenBudgetStats] = None
    feature_diffs: List[FeatureDiffStats] = field(default_factory=list)
    split_comp: Optional[SplitCompStats] = None
    hard_pos_rate: float = 0.0  # % change pairs that look stylistically like no-change
    hard_neg_rate: float = 0.0  # % no-change pairs that look stylistically like change


# ─── Utilities ────────────────────────────────────────────────────────────────

def _hash(prob) -> str:
    return hashlib.sha256("\n".join(prob.sentences).encode()).hexdigest()


def _perc_dist(arr) -> PercDist:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return PercDist(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    p = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    return PercDist(
        p10=float(p[0]), p25=float(p[1]), p50=float(p[2]),
        p75=float(p[3]), p90=float(p[4]), p95=float(p[5]), p99=float(p[6]),
        max_val=float(np.max(arr)), std=float(np.std(arr)), mean=float(np.mean(arr)),
    )


def _approx_tokens(text: str) -> int:
    """Approximate DeBERTa-v3 (SentencePiece) token count: words × 1.35."""
    return max(1, round(len(text.split()) * 1.35))


def _compute_token_budget(probs, window_size: int = 1) -> TokenBudgetStats:
    """
    For every (sent_i, sent_{i+1}) pair with ±window_size surrounding sentences,
    compute the approximate DeBERTa token count and return coverage statistics.

    Window input format:
        [CLS] sent_{i-w} \\n … \\n sent_i [SEP] sent_{i+1} \\n … \\n sent_{i+1+w} [SEP]
    """
    counts = []
    for prob in probs:
        sents = prob.sentences
        n = len(sents)
        for i in range(n - 1):
            left  = " \n ".join(sents[max(0, i - window_size) : i + 1])
            right = " \n ".join(sents[i + 1 : min(n, i + 1 + window_size + 1)])
            # +3 for [CLS], [SEP], [SEP]
            counts.append(_approx_tokens(left + " [SEP] " + right) + 3)

    if not counts:
        return TokenBudgetStats()

    arr = np.array(counts, dtype=np.int32)
    stats = TokenBudgetStats(
        mean_tokens=float(np.mean(arr)),
        p50_tokens=float(np.percentile(arr, 50)),
        p90_tokens=float(np.percentile(arr, 90)),
        p95_tokens=float(np.percentile(arr, 95)),
        p99_tokens=float(np.percentile(arr, 99)),
        max_tokens=int(np.max(arr)),
    )
    for budget in TOKEN_BUDGETS:
        stats.pct_fit[budget] = 100.0 * float(np.mean(arr <= budget))
    return stats


def _stylometric_comparison(
    pairs,
) -> Tuple[List[FeatureDiffStats], float, float]:
    """
    For each of 28 sentence-level style features, compute mean |feat_i − feat_j|
    separately for style-change pairs (label=1) and no-change pairs (label=0).

    Returns (feature_diffs sorted by discrimination, hard_pos_rate, hard_neg_rate).

    Hard positive: a change pair whose total feature delta is below the median
    total delta of no-change pairs — hard to distinguish from no-change.
    Hard negative: a no-change pair whose total delta exceeds the median total
    delta of change pairs — hard to distinguish from change.
    """
    fc_list, fn_list = [], []
    for rec in pairs:
        sent_a = rec.sentences[rec.pair_idx]
        sent_b = rec.sentences[rec.pair_idx + 1]
        fi = extract_sentence_features(sent_a)
        fj = extract_sentence_features(sent_b)
        delta = np.abs(fi - fj)
        if rec.label == 1:
            fc_list.append(delta)
        else:
            fn_list.append(delta)

    if not fc_list or not fn_list:
        return [], 0.0, 0.0

    fc = np.array(fc_list)    # (n_change,   N_SENT_FEATURES)
    fn = np.array(fn_list)    # (n_nochange, N_SENT_FEATURES)

    diffs = []
    for i, name in enumerate(FEATURE_NAMES_SENT):
        mc = float(np.mean(fc[:, i]))
        mn = float(np.mean(fn[:, i]))
        diffs.append(FeatureDiffStats(
            name=name, mean_no_change=mn, mean_change=mc, discrimination=mc - mn,
        ))
    diffs.sort(key=lambda x: -abs(x.discrimination))

    total_fc = fc.mean(axis=1)
    total_fn = fn.mean(axis=1)
    hard_pos_rate = float(np.mean(total_fc < np.median(total_fn)))
    hard_neg_rate = float(np.mean(total_fn > np.median(total_fc)))

    return diffs, hard_pos_rate, hard_neg_rate


def _split_comparison(train_probs, val_probs, difficulty: str) -> SplitCompStats:
    t_pairs = make_pair_records(train_probs, difficulty)
    v_pairs = make_pair_records(val_probs,   difficulty)
    t_doc  = [len(p.sentences) for p in train_probs]
    v_doc  = [len(p.sentences) for p in val_probs]
    t_sent = [len(s.split()) for p in train_probs for s in p.sentences]
    v_sent = [len(s.split()) for p in val_probs   for s in p.sentences]
    return SplitCompStats(
        difficulty=difficulty,
        train_n_docs=len(train_probs),
        train_n_pairs=len(t_pairs),
        train_pos_rate=sum(r.label for r in t_pairs) / max(len(t_pairs), 1),
        val_n_docs=len(val_probs),
        val_n_pairs=len(v_pairs),
        val_pos_rate=sum(r.label for r in v_pairs) / max(len(v_pairs), 1),
        train_mean_doc_len=float(np.mean(t_doc)) if t_doc else 0.0,
        val_mean_doc_len=float(np.mean(v_doc)) if v_doc else 0.0,
        train_mean_sent_len=float(np.mean(t_sent)) if t_sent else 0.0,
        val_mean_sent_len=float(np.mean(v_sent)) if v_sent else 0.0,
    )


# ─── Main analysis ────────────────────────────────────────────────────────────

def analyze_difficulty(
    difficulty: str,
    year_filter: Optional[str] = None,
    do_deep: bool = True,
) -> DifficultyReport:
    report = DifficultyReport(difficulty=difficulty)

    print(f"\n{'='*60}")
    print(f"Difficulty: {difficulty.upper()}")
    print("="*60)

    year_hashes: Dict[str, list] = {}
    probs_2026_all: list = []

    loaders = [
        ("2026", lambda: load_2026(DATA_2026, difficulty, "train") +
                         load_2026(DATA_2026, difficulty, "validation")),
        ("2025", lambda: load_2025(DATA_2025, difficulty, "train") +
                         load_2025(DATA_2025, difficulty, "validation")),
        ("2024", lambda: load_2024(DATA_2024, difficulty, "train") +
                         load_2024(DATA_2024, difficulty, "validation")),
        ("2023", lambda: load_2023(DATA_2023, difficulty, "train") +
                         load_2023(DATA_2023, difficulty, "validation")),
        ("2022", lambda: load_2022(DATA_2022, difficulty, "train")),
    ]

    for yr, loader_fn in loaders:
        if year_filter and yr != year_filter:
            continue
        try:
            probs = loader_fn()
        except Exception as e:
            print(f"  {yr}: ERROR — {e}")
            continue
        if not probs:
            continue

        if yr == "2026":
            probs_2026_all = probs

        pairs    = make_pair_records(probs, difficulty)
        labels   = [p.label for p in pairs]
        doc_lens = np.array([len(p.sentences) for p in probs], dtype=float)
        wrd_lens = np.array([len(s.split()) for p in probs for s in p.sentences], dtype=float)
        year_hashes[yr] = [_hash(p) for p in probs]

        stats = YearStats(
            year=yr,
            n_docs=len(probs),
            n_pairs=len(labels),
            n_pos=sum(labels),
            mean_doc_len=float(np.mean(doc_lens)),
            std_doc_len=float(np.std(doc_lens)),
            min_doc_len=int(np.min(doc_lens)),
            max_doc_len=int(np.max(doc_lens)),
            median_doc_len=float(np.median(doc_lens)),
            mean_word_len=float(np.mean(wrd_lens)),
            std_word_len=float(np.std(wrd_lens)),
            doc_len_dist=_perc_dist(doc_lens),
            sent_len_dist=_perc_dist(wrd_lens),
        )
        report.year_stats.append(stats)

        print(f"\n  {yr}:")
        print(f"    Docs:        {stats.n_docs:,}")
        print(f"    Pairs:       {stats.n_pairs:,}  ({100*stats.pos_rate:.1f}% positive)")
        print(f"    Sent/doc:    {stats.mean_doc_len:.1f} ± {stats.std_doc_len:.1f}"
              f"  [med={stats.median_doc_len:.0f}, max={stats.max_doc_len}]")
        print(f"    Words/sent:  {stats.mean_word_len:.1f} ± {stats.std_word_len:.1f}")

    # Cross-year duplicate analysis
    if len(year_hashes) > 1:
        print(f"\n  Cross-year duplicates:")
        hash_to_years: Dict[str, list] = {}
        for yr, hs in year_hashes.items():
            for h in hs:
                hash_to_years.setdefault(h, []).append(yr)

        dupes = {h: yrs for h, yrs in hash_to_years.items() if len(yrs) > 1}
        if dupes:
            groups = Counter(tuple(sorted(set(v))) for v in dupes.values())
            for grp, cnt in sorted(groups.items(), key=lambda x: -x[1]):
                report.duplicates.append(DuplicateGroup(years=grp, count=cnt))
                print(f"    {' & '.join(grp)}: {cnt} duplicate docs")
        else:
            print("    No cross-year duplicates detected")

    if not do_deep or year_filter or not probs_2026_all:
        return report

    # ── Token budget ─────────────────────────────────────────────────────────
    print(f"\n  Token budget (2026 pairs, approx — words × 1.35 + 3 special tokens):")
    try:
        report.token_budget = _compute_token_budget(probs_2026_all)
        tb = report.token_budget
        for budget in TOKEN_BUDGETS:
            pct = tb.pct_fit.get(budget, 0.0)
            marker = "  ← current MAX_LENGTH" if budget == MAX_LENGTH else ""
            print(f"    ≤{budget:3d} tokens: {pct:5.1f}%{marker}")
        print(f"    Tokens/pair — mean={tb.mean_tokens:.0f}  p50={tb.p50_tokens:.0f}"
              f"  p90={tb.p90_tokens:.0f}  p95={tb.p95_tokens:.0f}  p99={tb.p99_tokens:.0f}"
              f"  max={tb.max_tokens}")
    except Exception as exc:
        print(f"    [WARN] Token budget failed: {exc}")

    # ── Stylometric feature comparison ────────────────────────────────────────
    pairs_2026 = make_pair_records(probs_2026_all, difficulty)
    n_sample = min(len(pairs_2026), 5000)
    if n_sample > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs_2026), size=n_sample, replace=False)
        sample = [pairs_2026[i] for i in sorted(idx)]
        print(f"\n  Stylometric feature comparison ({n_sample:,} pairs from 2026):")
        try:
            diffs, hpos, hneg = _stylometric_comparison(sample)
            report.feature_diffs = diffs
            report.hard_pos_rate = hpos
            report.hard_neg_rate = hneg
            if diffs:
                print(f"    Top-5 most discriminative features (|delta| change vs no-change):")
                for fd in diffs[:5]:
                    print(f"      {fd.name:26s}: Δ={fd.discrimination:+.4f}"
                          f"  (change={fd.mean_change:.4f}  no-change={fd.mean_no_change:.4f})")
                print(f"    Hard positives (change pairs w/ no-change-like style): {100*hpos:.1f}%")
                print(f"    Hard negatives (no-change pairs w/ change-like style): {100*hneg:.1f}%")
        except Exception as exc:
            print(f"    [WARN] Stylometric comparison failed: {exc}")

    # ── Cross-split (train vs val) ────────────────────────────────────────────
    try:
        t2026 = load_2026(DATA_2026, difficulty, "train")
        v2026 = load_2026(DATA_2026, difficulty, "validation")
        if t2026 and v2026:
            sc = _split_comparison(t2026, v2026, difficulty)
            report.split_comp = sc
            print(f"\n  PAN 2026 train / val split:")
            print(f"    Train: {sc.train_n_docs:,} docs | {sc.train_n_pairs:,} pairs"
                  f" | {100*sc.train_pos_rate:.2f}% positive"
                  f" | {sc.train_mean_doc_len:.1f} sent/doc avg")
            print(f"    Val:   {sc.val_n_docs:,} docs | {sc.val_n_pairs:,} pairs"
                  f" | {100*sc.val_pos_rate:.2f}% positive"
                  f" | {sc.val_mean_doc_len:.1f} sent/doc avg")
            skew = abs(sc.train_pos_rate - sc.val_pos_rate) / max(sc.train_pos_rate, 1e-6)
            if skew > 0.10:
                print(f"    ⚠  Positive-rate skew {100*skew:.0f}%"
                      f" — threshold must be calibrated on val, not train")
    except Exception as exc:
        print(f"    [WARN] Split comparison failed: {exc}")

    return report


# ─── Markdown report builder ──────────────────────────────────────────────────

def _fmt_perc(d: PercDist, fmt: str = ".1f") -> str:
    """Format a PercDist row for a Markdown table."""
    f = fmt
    return (f"p10={d.p10:{f}} p25={d.p25:{f}} p50={d.p50:{f}} "
            f"p75={d.p75:{f}} p90={d.p90:{f}} p95={d.p95:{f}} p99={d.p99:{f}}"
            f" max={d.max_val:{f}} std={d.std:{f}}")


def _build_report(reports: List[DifficultyReport]) -> str:
    today = date.today().isoformat()
    md = []

    md.append("# PAN Multi-Author Writing Style Analysis — Data Analysis Report")
    md.append(f"\n*Generated: {today}*\n")
    md.append("---\n")

    # ── Executive summary ─────────────────────────────────────────────────────
    md.append("## Executive Summary\n")
    md.append(
        "This report covers all available PAN datasets (2022–2026) for the "
        "multi-author writing style analysis task.  \n"
        "**Task**: for each consecutive sentence pair in a document, predict whether "
        "a style change occurred (binary classification, label = 0 or 1).  \n"
        "**Metric**: F1-macro across all sentence pairs.\n"
    )

    md.append("### Class Imbalance at a Glance (PAN 2026)\n")
    md.append("| Difficulty | Positive rate | Modeling implication |")
    md.append("|------------|--------------|----------------------|")
    impl = {
        "easy":   "Moderate imbalance; topic shift is a usable signal",
        "medium": "**Critical** — threshold=0.5 almost always wrong; calibrate down to ~0.10–0.25",
        "hard":   "Severe imbalance; pure style signal; SCL + class weights essential",
    }
    for rpt in reports:
        s26 = next((s for s in rpt.year_stats if s.year == "2026"), None)
        rate = f"{100*s26.pos_rate:.1f}%" if s26 else "N/A"
        md.append(f"| {rpt.difficulty} | {rate} | {impl.get(rpt.difficulty, '')} |")
    md.append("")

    md.append("### Total Data Volume (all years combined, before dedup)\n")
    md.append("| Difficulty | Total docs | Total pairs | Combined positive% |")
    md.append("|------------|-----------|------------|-------------------|")
    for rpt in reports:
        tot_docs  = sum(s.n_docs  for s in rpt.year_stats)
        tot_pairs = sum(s.n_pairs for s in rpt.year_stats)
        tot_pos   = sum(s.n_pos   for s in rpt.year_stats)
        rate      = 100 * tot_pos / max(tot_pairs, 1)
        md.append(f"| {rpt.difficulty} | {tot_docs:,} | {tot_pairs:,} | {rate:.1f}% |")
    md.append("")

    # ── Per-difficulty sections ────────────────────────────────────────────────
    for rpt in reports:
        diff = rpt.difficulty
        md.append(f"\n---\n\n## Difficulty: {diff.upper()}\n")

        # Per-year stats table
        md.append("### Per-Year Statistics\n")
        md.append("| Year | Docs | Pairs | Positive% | Sent/doc mean±std | "
                  "min / median / max | Words/sent mean±std |")
        md.append("|------|------|-------|-----------|-------------------"
                  "|--------------------|---------------------|")
        for s in rpt.year_stats:
            md.append(
                f"| {s.year} | {s.n_docs:,} | {s.n_pairs:,} | {100*s.pos_rate:.1f}% "
                f"| {s.mean_doc_len:.1f} ± {s.std_doc_len:.1f} "
                f"| {s.min_doc_len} / {s.median_doc_len:.0f} / {s.max_doc_len} "
                f"| {s.mean_word_len:.1f} ± {s.std_word_len:.1f} |"
            )
        tot_docs  = sum(s.n_docs  for s in rpt.year_stats)
        tot_pairs = sum(s.n_pairs for s in rpt.year_stats)
        tot_pos   = sum(s.n_pos   for s in rpt.year_stats)
        md.append(f"\n**Combined (before dedup):** {tot_docs:,} docs | "
                  f"{tot_pairs:,} pairs | {100*tot_pos/max(tot_pairs,1):.1f}% positive\n")

        # Percentile distributions
        s26 = next((s for s in rpt.year_stats if s.year == "2026"), None)
        if s26 and s26.doc_len_dist:
            md.append("### Length Distributions (PAN 2026)\n")
            dl = s26.doc_len_dist
            sl = s26.sent_len_dist
            md.append("**Document length (sentences per document):**\n")
            md.append("| p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | std |")
            md.append("|-----|-----|-----|-----|-----|-----|-----|-----|-----|")
            md.append(
                f"| {dl.p10:.0f} | {dl.p25:.0f} | {dl.p50:.0f} | {dl.p75:.0f}"
                f" | {dl.p90:.0f} | {dl.p95:.0f} | {dl.p99:.0f}"
                f" | {dl.max_val:.0f} | {dl.std:.1f} |"
            )
            md.append("")
            if sl:
                md.append("**Sentence length (words per sentence):**\n")
                md.append("| p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | std |")
                md.append("|-----|-----|-----|-----|-----|-----|-----|-----|-----|")
                md.append(
                    f"| {sl.p10:.0f} | {sl.p25:.0f} | {sl.p50:.0f} | {sl.p75:.0f}"
                    f" | {sl.p90:.0f} | {sl.p95:.0f} | {sl.p99:.0f}"
                    f" | {sl.max_val:.0f} | {sl.std:.1f} |"
                )
            md.append("")

        # Token budget
        if rpt.token_budget and rpt.token_budget.pct_fit:
            tb = rpt.token_budget
            md.append("### Token Budget Analysis (PAN 2026, DeBERTa window context)\n")
            md.append(
                f"> Token counts are **approximate** (words × 1.35 + 3 special tokens). "
                f"Window format: `[CLS] sent_{{i-1}} \\n sent_i [SEP] sent_{{i+1}} \\n "
                f"sent_{{i+2}} [SEP]` (WINDOW_SIZE=1).\n"
            )
            md.append("| MAX_LENGTH | % pairs that fit | Note |")
            md.append("|-----------|-----------------|------|")
            for budget in TOKEN_BUDGETS:
                pct = tb.pct_fit.get(budget, 0.0)
                note = "← **current MAX_LENGTH**" if budget == MAX_LENGTH else ""
                if pct < 90.0 and not note:
                    note = "⚠ significant truncation"
                elif pct >= 99.0 and not note:
                    note = "safe headroom"
                md.append(f"| {budget} | {pct:.1f}% | {note} |")
            md.append("")
            md.append(
                f"Token statistics per pair: "
                f"mean={tb.mean_tokens:.0f} · p50={tb.p50_tokens:.0f} · "
                f"p90={tb.p90_tokens:.0f} · p95={tb.p95_tokens:.0f} · "
                f"p99={tb.p99_tokens:.0f} · max={tb.max_tokens}\n"
            )
            pct_current = tb.pct_fit.get(MAX_LENGTH, 0.0)
            if pct_current < 95.0:
                safe_budget = next(
                    (b for b in TOKEN_BUDGETS if tb.pct_fit.get(b, 0) >= 99.0), 512
                )
                md.append(
                    f"> **Action required**: current `MAX_LENGTH={MAX_LENGTH}` covers only "
                    f"{pct_current:.1f}% of pairs. Consider increasing to "
                    f"{safe_budget} to avoid truncation.\n"
                )
            else:
                md.append(
                    f"> Current `MAX_LENGTH={MAX_LENGTH}` covers {pct_current:.1f}% of pairs "
                    f"— adequate.\n"
                )

        # Stylometric feature comparison
        if rpt.feature_diffs:
            md.append("### Stylometric Feature Analysis (PAN 2026)\n")
            md.append(
                "Mean absolute per-feature difference `abs(feat(sent_i) − feat(sent_{i+1}))` "
                "for style-change (label=1) vs no-change (label=0) pairs. "
                "Higher **Discrimination Δ** means more useful for detecting style change.\n"
            )
            md.append("**Hard example rates:**\n")
            md.append(
                f"- Hard positives (style-change pairs that look stylistically similar "
                f"to no-change pairs): **{100*rpt.hard_pos_rate:.1f}%**\n"
                f"- Hard negatives (no-change pairs that look stylistically different "
                f"from other no-change pairs): **{100*rpt.hard_neg_rate:.1f}%**\n"
            )
            top10 = rpt.feature_diffs[:10]
            md.append("**Top-10 features by discrimination (sorted by |Δ|):**\n")
            md.append("| Feature | No-change mean δ | Change mean δ | Discrimination Δ |")
            md.append("|---------|-----------------|--------------|-----------------|")
            for fd in top10:
                direction = "↑" if fd.discrimination > 0 else "↓"
                md.append(
                    f"| `{fd.name}` | {fd.mean_no_change:.4f} | {fd.mean_change:.4f}"
                    f" | {fd.discrimination:+.4f} {direction} |"
                )
            md.append("")
            bottom5 = rpt.feature_diffs[-5:]
            md.append("**Least discriminative features (may be noise):**\n")
            md.append("| Feature | Discrimination Δ |")
            md.append("|---------|-----------------|")
            for fd in bottom5:
                md.append(f"| `{fd.name}` | {fd.discrimination:+.4f} |")
            md.append("")

        # Cross-split comparison
        if rpt.split_comp:
            sc = rpt.split_comp
            md.append("### Train / Validation Split Consistency (PAN 2026)\n")
            md.append("| Split | Docs | Pairs | Positive% | Mean sent/doc | Mean words/sent |")
            md.append("|-------|------|-------|-----------|--------------|----------------|")
            md.append(
                f"| train | {sc.train_n_docs:,} | {sc.train_n_pairs:,}"
                f" | {100*sc.train_pos_rate:.2f}%"
                f" | {sc.train_mean_doc_len:.1f}"
                f" | {sc.train_mean_sent_len:.1f} |"
            )
            md.append(
                f"| val   | {sc.val_n_docs:,} | {sc.val_n_pairs:,}"
                f" | {100*sc.val_pos_rate:.2f}%"
                f" | {sc.val_mean_doc_len:.1f}"
                f" | {sc.val_mean_sent_len:.1f} |"
            )
            skew = abs(sc.train_pos_rate - sc.val_pos_rate) / max(sc.train_pos_rate, 1e-6)
            if skew > 0.10:
                md.append(
                    f"\n> ⚠ Positive-rate skew: {100*skew:.0f}% between train and val. "
                    f"Threshold **must** be calibrated on validation set, not train.\n"
                )
            else:
                md.append(
                    f"\n> Split is balanced: train/val positive rates differ by only "
                    f"{100*skew:.1f}%. Threshold calibration on val is still recommended.\n"
                )

        # Key observations
        md.append("### Key Observations\n")
        s25 = next((s for s in rpt.year_stats if s.year == "2025"), None)
        s24 = next((s for s in rpt.year_stats if s.year == "2024"), None)
        s23 = next((s for s in rpt.year_stats if s.year == "2023"), None)
        s22 = next((s for s in rpt.year_stats if s.year == "2022"), None)
        if s26:
            md.append(
                f"- **PAN 2026** is the primary dataset: {s26.n_docs:,} documents, "
                f"avg {s26.mean_doc_len:.0f} sentences/doc "
                f"(std={s26.std_doc_len:.0f}, max={s26.max_doc_len}). "
                f"Long documents make the SSPC BiLSTM model especially valuable "
                f"because it captures full-document style consistency."
            )
        if s25:
            md.append(
                f"- **PAN 2025**: {s25.n_docs:,} docs, avg {s25.mean_doc_len:.0f} sent/doc, "
                f"{100*s25.pos_rate:.1f}% positive. "
                f"Similar format to 2026; safe to include as extra training data."
            )
        if s24 and s23:
            avg_pos = (s24.pos_rate + s23.pos_rate) / 2
            avg_len = (s24.mean_doc_len + s23.mean_doc_len) / 2
            md.append(
                f"- **PAN 2023/2024**: shorter documents (avg {avg_len:.1f} sent/doc), "
                f"higher positive rates (~{100*avg_pos:.0f}%). "
                f"May introduce distribution shift — monitor their effect on validation F1."
            )
        if s22:
            md.append(
                f"- **PAN 2022**: {s22.n_docs:,} docs (train-only), "
                f"{100*s22.pos_rate:.1f}% positive. "
                f"Paragraph-level granularity but format-compatible. "
                f"Provides useful extra examples, especially for minority classes."
            )
        if diff == "medium":
            rate_str = f"{100*s26.pos_rate:.1f}%" if s26 else "~4%"
            md.append(
                f"- **Extreme class imbalance**: only {rate_str} positive in PAN 2026. "
                f"Threshold 0.5 predicts nearly all zeros — calibration is the single most "
                f"impactful step. Target threshold range: 0.08–0.25."
            )
        if diff == "hard":
            md.append(
                f"- **No topic signal**: documents are strictly same-topic. "
                f"The model must rely on stylometric features (rhythm, vocabulary, "
                f"punctuation, function words). SCL training enforces topic-invariant "
                f"DeBERTa representations."
            )
        md.append("")

        # Cross-year duplicates
        md.append("### Cross-Year Duplicate Analysis\n")
        if rpt.duplicates:
            total_dupe = sum(d.count for d in rpt.duplicates)
            md.append(
                f"Found **{total_dupe} duplicate documents** across years "
                f"(removed during prepare_data.py):\n"
            )
            md.append("| Year group | Duplicate docs |")
            md.append("|-----------|---------------|")
            for d in sorted(rpt.duplicates, key=lambda x: -x.count):
                md.append(f"| {' & '.join(d.years)} | {d.count} |")
            md.append("")
            md.append(
                "> `prepare_data.py` uses SHA-256 hashing to remove these duplicates "
                "before training, ensuring no document appears twice regardless of year."
            )
        else:
            md.append(
                "No cross-year duplicates detected. "
                "All years contribute independent documents."
            )
        md.append("")

    # ── Training recommendations ──────────────────────────────────────────────
    md.append("\n---\n\n## Training Recommendations\n")

    # Compute data-justified class weights and thresholds from the reports
    rec_rows = [
        ("Data preparation",
         "Run `prepare_data.py` first — saves merged + deduped JSONL that `train.py` auto-loads"),
        ("Year inclusion",
         "Include all years (2022–2026); 2022 train-only; "
         "watch 2023/2024 for distribution shift on validation F1"),
        ("Class imbalance",
         "`scale_pos_weight` in LightGBM = (1−pos_rate)/pos_rate; "
         "class-weighted CE in DeBERTa; threshold grid-search on val set"),
    ]

    for rpt in reports:
        s26 = next((s for s in rpt.year_stats if s.year == "2026"), None)
        pos = s26.pos_rate if s26 else 0.314
        spw = (1 - pos) / max(pos, 1e-6)
        ce_w = round(spw, 1)
        if rpt.difficulty == "medium":
            thr_range = "0.05–0.30 (optimal typically 0.08–0.15)"
        elif rpt.difficulty == "hard":
            thr_range = "0.25–0.50 (optimal typically 0.30–0.45)"
        else:
            thr_range = "0.25–0.55 (optimal typically 0.35–0.45)"
        rec_rows.append((
            f"{rpt.difficulty} — class weight",
            f"LightGBM `scale_pos_weight ≈ {spw:.1f}`, CE weight ≈ {ce_w:.1f}x; "
            f"threshold search range {thr_range}",
        ))

    rec_rows += [
        ("SCL loss (hard)",
         "Use SCL alpha=0.3 — pushes style-change embeddings together regardless of topic"),
        ("Long documents (2026)",
         "SSPC BiLSTM processes full document context — critical when avg doc > 50 sentences"),
        ("Sequence smoothing",
         "Gaussian smoothing sigma=0.5–1.5 reduces isolated false positives; "
         "calibrate sigma on val per difficulty"),
        ("MAX_LENGTH",
         "Verify token budget table above — truncation at 256 may lose context for "
         "very long sentences; 384 is usually safe"),
        ("Evaluation",
         "Always compute F1-macro on the validation set with the calibrated threshold, "
         "not with threshold=0.5"),
    ]

    md.append("| Aspect | Recommendation |")
    md.append("|--------|----------------|")
    for aspect, rec in rec_rows:
        md.append(f"| {aspect} | {rec} |")
    md.append("")

    # ── Data format reference ─────────────────────────────────────────────────
    md.append("\n---\n\n## Prepared Data Format\n")
    md.append("After running `python prepare_data.py`, the following files are created:\n")
    md.append("```")
    md.append("data_prepared/")
    for d in ["easy", "medium", "hard"]:
        md.append(f"  train_{d}.jsonl      # merged + deduped training docs")
        md.append(f"  val_{d}.jsonl        # merged + deduped validation docs")
    md.append("```\n")
    md.append("Each line is one document:")
    md.append("```json")
    md.append('{"id": "42", "sentences": ["First sent.", "Second sent.", ...],')
    md.append(' "changes": [0, 1, 0, ...], "difficulty": "easy"}')
    md.append("```\n")
    md.append(
        "- `train.py` auto-detects these files and loads from them. Use `--force-raw` to bypass.\n"
        "- `predict.py` accepts JSONL input (`--input-format prepared` or auto-detect). "
        "Test data omits the `changes` key.\n"
    )

    md.append("\n---\n")
    md.append("*Report generated by `analyze_data.py`*")
    return "\n".join(md)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse PAN multi-author writing style data and save report"
    )
    parser.add_argument("--year",       help="Restrict to one year (2022–2026)")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, help="Restrict to one difficulty")
    parser.add_argument("--no-report",  action="store_true",
                        help="Print to console only; do not write data_analysis_report.md")
    parser.add_argument("--no-deep",    action="store_true",
                        help="Skip token-budget + feature analysis (fast mode)")
    args = parser.parse_args()

    diffs = [args.difficulty] if args.difficulty else DIFFICULTIES
    reports = []
    for diff in diffs:
        rpt = analyze_difficulty(diff, year_filter=args.year, do_deep=not args.no_deep)
        reports.append(rpt)

    if not args.no_report:
        md = _build_report(reports)
        REPORT_PATH.write_text(md, encoding="utf-8")
        print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()