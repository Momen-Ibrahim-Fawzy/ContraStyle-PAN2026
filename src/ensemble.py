"""
Ensemble and threshold calibration for the style change detection system.

Three models are combined:
  1. DeBERTa-v3 (SCL fine-tuned) — neural, strong pair-level cross-attention
  2. LightGBM on stylometric features — classical, fully topic-agnostic
  3. SSPC (BiLSTM over sentence embeddings) — document-level context

For each difficulty level we:
  a) Grid-search over three blend weights (w_deberta, w_lgbm, w_sspc)
     summing to 1.0, maximising F1-macro on the validation set.
  b) Grid-search over decision threshold t ∈ [0.05, 0.95].
  c) Optionally apply Gaussian sequence smoothing to the blended probabilities
     before thresholding (helps reduce isolated false positives in hard/medium).

Threshold calibration is the MOST critical step, especially for the medium
subset (positive rate ≈ 4.4%) where threshold=0.5 is almost always wrong.

Post-processing (sequence smoothing):
  A short Gaussian kernel is convolved with the blended probability sequence
  within each document. This exploits the block structure of authorship:
  true style changes tend to persist for several sentences, not flip single-
  pair predictions. Smoothing is most helpful for hard/medium and can be
  disabled for easy where topic-driven changes are sharp and correct.

  Smoothing sigma is calibrated per difficulty and is set to 0 to disable.
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

from .config import (
    DIFFICULTIES, DEFAULT_ENSEMBLE_WEIGHTS, DEFAULT_THRESHOLD,
    ENSEMBLE_PATH,
)


@dataclass
class DifficultyConfig:
    weight_deberta: float   # w1
    weight_lgbm:    float   # w2
    weight_sspc:    float   # w3  (0.0 when SSPC model not available)
    threshold:      float   # decision threshold P(change) → binary
    smooth_sigma:   float = 0.0  # Gaussian smoothing sigma (0 = disabled)


@dataclass
class EnsembleConfig:
    per_difficulty: Dict[str, DifficultyConfig] = field(default_factory=dict)

    def get(self, difficulty: str) -> DifficultyConfig:
        if difficulty in self.per_difficulty:
            return self.per_difficulty[difficulty]
        if "hard" in self.per_difficulty:
            return self.per_difficulty["hard"]
        return DifficultyConfig(
            weight_deberta=0.60, weight_lgbm=0.25, weight_sspc=0.15,
            threshold=0.35, smooth_sigma=0.0,
        )

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "EnsembleConfig":
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def default(cls) -> "EnsembleConfig":
        cfg = cls()
        for diff in DIFFICULTIES:
            w_d, w_l = DEFAULT_ENSEMBLE_WEIGHTS[diff]
            cfg.per_difficulty[diff] = DifficultyConfig(
                weight_deberta=w_d,
                weight_lgbm=w_l,
                weight_sspc=0.0,
                threshold=DEFAULT_THRESHOLD[diff],
                smooth_sigma=0.0,
            )
        return cfg


# ─── Blending ─────────────────────────────────────────────────────────────────

def _blend(
    deberta_probs: np.ndarray,
    lgbm_probs:    np.ndarray,
    sspc_probs:    Optional[np.ndarray],
    w_d: float,
    w_l: float,
    w_s: float,
) -> np.ndarray:
    blended = w_d * deberta_probs + w_l * lgbm_probs
    if sspc_probs is not None and w_s > 0:
        blended = blended + w_s * sspc_probs
    return blended


# ─── Sequence smoothing ───────────────────────────────────────────────────────

def smooth_sequence(probs: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to a probability sequence within a document.

    This exploits authorship block structure: if the model is uncertain
    about a single pair but the surrounding pairs are confidently no-change,
    smoothing reduces the spike to below threshold.

    sigma=0 → no smoothing (pass-through).
    sigma=1 → slight smoothing (~3 pairs).
    sigma=2 → moderate smoothing (~5 pairs).
    """
    if sigma <= 0 or len(probs) < 3:
        return probs
    return gaussian_filter1d(probs, sigma=sigma)


def smooth_per_document(
    records:     list,
    probs:       np.ndarray,
    sigma:       float,
) -> np.ndarray:
    """
    Apply document-level Gaussian smoothing to blended probabilities.

    Smoothing is applied within each document independently (never across
    document boundaries), which preserves the document-level context.
    """
    if sigma <= 0:
        return probs

    result = probs.copy()
    # Group indices by document
    from collections import defaultdict
    doc_indices = defaultdict(list)
    for i, rec in enumerate(records):
        doc_indices[rec.doc_id].append(i)

    for doc_id, idxs in doc_indices.items():
        # Sort by pair index
        idxs_sorted = sorted(idxs, key=lambda i: records[i].pair_idx)
        doc_probs   = result[[i for i in idxs_sorted]]
        smoothed    = smooth_sequence(doc_probs, sigma)
        for j, orig_idx in enumerate(idxs_sorted):
            result[orig_idx] = smoothed[j]

    return result


# ─── Isotonic probability calibration ────────────────────────────────────────

def _isotonic_calibrate(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Fit an isotonic regression on (probs, labels) and return calibrated probs.

    Isotonic regression finds a non-decreasing step function that minimises
    the squared error against the binary labels — essentially a non-parametric
    way to turn uncalibrated model scores into well-calibrated probabilities.

    Why this helps:
      - DeBERTa often outputs overconfident scores (near 0 or 1).
      - Isotonic regression re-maps the score distribution so P(score > t)
        closely matches the actual positive rate.
      - Better calibrated scores → more reliable weighted blending and threshold.
      - Especially important for medium (4.4% positive rate) where the raw
        DeBERTa threshold of 0.5 is far from optimal.

    Note: fitted on the same validation set used for calibration.
    This is slightly optimistic but standard practice for competition calibration.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(probs, labels)
    return ir.transform(probs).astype(np.float32)


# ─── Calibration ──────────────────────────────────────────────────────────────

def calibrate_difficulty(
    deberta_probs: np.ndarray,
    lgbm_probs:    np.ndarray,
    labels:        np.ndarray,
    difficulty:    str,
    sspc_probs:    Optional[np.ndarray] = None,
    records:       Optional[list]       = None,
    use_isotonic:  bool                 = True,
) -> DifficultyConfig:
    """
    Grid-search over (blend weights, threshold, smooth_sigma) to maximise
    F1-macro on the validation set for one difficulty level.

    use_isotonic=True: isotonic-calibrate each model's probabilities before
    blending.  This re-maps overconfident raw scores to well-calibrated
    probabilities, making weighted blending and threshold search more reliable.
    """
    # ── Per-model isotonic calibration ───────────────────────────────────────
    if use_isotonic:
        deberta_probs = _isotonic_calibrate(deberta_probs, labels)
        lgbm_probs    = _isotonic_calibrate(lgbm_probs,    labels)
        if sspc_probs is not None:
            sspc_probs = _isotonic_calibrate(sspc_probs, labels)
        print(f"  [{difficulty}] Isotonic calibration applied to each component.")

    # Weight grids
    w_deberta_grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    has_sspc = sspc_probs is not None

    if has_sspc:
        # With SSPC: search over all 3 weights
        weight_combos = [
            (wd, wl, ws)
            for wd in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7]
            for ws in [0.0, 0.1, 0.2, 0.3]
            for wl in [round(1.0 - wd - ws, 2)]
            if 0.0 <= wl <= 1.0
        ]
    else:
        # Without SSPC: just DeBERTa + LightGBM
        weight_combos = [
            (wd, round(1.0 - wd, 2), 0.0)
            for wd in w_deberta_grid
        ]

    # Threshold grid
    if difficulty == "medium":
        thr_grid = list(np.arange(0.04, 0.50, 0.02))
    else:
        thr_grid = list(np.arange(0.10, 0.65, 0.03))

    # Smoothing sigma grid
    if difficulty in ("hard", "medium"):
        sigma_grid = [0.0, 0.5, 1.0, 1.5]
    else:
        sigma_grid = [0.0]  # Easy: topic signals are sharp, don't smooth

    best_f1   = -1.0
    best_conf = DifficultyConfig(
        weight_deberta=DEFAULT_ENSEMBLE_WEIGHTS[difficulty][0],
        weight_lgbm=DEFAULT_ENSEMBLE_WEIGHTS[difficulty][1],
        weight_sspc=0.0,
        threshold=DEFAULT_THRESHOLD[difficulty],
        smooth_sigma=0.0,
    )

    for wd, wl, ws in weight_combos:
        blended = _blend(deberta_probs, lgbm_probs,
                         sspc_probs if has_sspc else None, wd, wl, ws)
        for sigma in sigma_grid:
            if sigma > 0 and records is not None:
                smoothed = smooth_per_document(records, blended, sigma)
            else:
                smoothed = blended

            for thr in thr_grid:
                preds = (smoothed >= thr).astype(int)
                f1    = float(f1_score(labels, preds, average="macro", zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_conf = DifficultyConfig(
                        weight_deberta=wd,
                        weight_lgbm=wl,
                        weight_sspc=ws,
                        threshold=thr,
                        smooth_sigma=sigma,
                    )

    print(
        f"  [{difficulty}] best F1={best_f1:.4f} "
        f"w_d={best_conf.weight_deberta:.2f} "
        f"w_l={best_conf.weight_lgbm:.2f} "
        f"w_s={best_conf.weight_sspc:.2f} "
        f"thr={best_conf.threshold:.3f} "
        f"sigma={best_conf.smooth_sigma}"
    )
    return best_conf


def calibrate_all(
    val_deberta: Dict[str, np.ndarray],
    val_lgbm:    Dict[str, np.ndarray],
    val_labels:  Dict[str, np.ndarray],
    val_sspc:    Optional[Dict[str, np.ndarray]] = None,
    val_records: Optional[Dict[str, list]] = None,
) -> EnsembleConfig:
    """
    Calibrate all difficulty levels.

    Parameters:
      val_deberta, val_lgbm, val_labels: {difficulty → array}
      val_sspc (optional): {difficulty → array}  SSPC probabilities
      val_records (optional): {difficulty → list[PairRecord]}  for smoothing
    """
    cfg = EnsembleConfig()
    for diff in DIFFICULTIES:
        if diff not in val_deberta or diff not in val_lgbm:
            print(f"  [{diff}] no validation data — using defaults")
            w_d, w_l = DEFAULT_ENSEMBLE_WEIGHTS[diff]
            cfg.per_difficulty[diff] = DifficultyConfig(
                weight_deberta=w_d, weight_lgbm=w_l, weight_sspc=0.0,
                threshold=DEFAULT_THRESHOLD[diff],
            )
            continue

        sspc_arr = val_sspc.get(diff) if val_sspc else None
        records  = val_records.get(diff) if val_records else None

        diff_cfg = calibrate_difficulty(
            val_deberta[diff],
            val_lgbm[diff],
            val_labels[diff],
            diff,
            sspc_probs=sspc_arr,
            records=records,
        )
        cfg.per_difficulty[diff] = diff_cfg
    return cfg


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_binary(
    deberta_probs: np.ndarray,
    lgbm_probs:    np.ndarray,
    difficulty:    str,
    config:        EnsembleConfig,
    sspc_probs:    Optional[np.ndarray] = None,
    records:       Optional[list]       = None,
) -> np.ndarray:
    """
    Combine model probabilities, apply smoothing, and threshold to binary.

    Returns int32 array of shape (N,).
    """
    diff_cfg = config.get(difficulty)
    blended  = _blend(
        deberta_probs, lgbm_probs, sspc_probs,
        diff_cfg.weight_deberta,
        diff_cfg.weight_lgbm,
        diff_cfg.weight_sspc,
    )

    if diff_cfg.smooth_sigma > 0 and records is not None:
        blended = smooth_per_document(records, blended, diff_cfg.smooth_sigma)

    return (blended >= diff_cfg.threshold).astype(np.int32)


def predict_proba_blended(
    deberta_probs: np.ndarray,
    lgbm_probs:    np.ndarray,
    difficulty:    str,
    config:        EnsembleConfig,
    sspc_probs:    Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return blended continuous probabilities (before thresholding)."""
    diff_cfg = config.get(difficulty)
    return _blend(
        deberta_probs, lgbm_probs, sspc_probs,
        diff_cfg.weight_deberta,
        diff_cfg.weight_lgbm,
        diff_cfg.weight_sspc,
    )


# ─── Document-level aggregation ───────────────────────────────────────────────

def aggregate_to_documents(
    records: list,
    preds:   np.ndarray,
) -> dict:
    """Reassemble per-pair predictions back into per-document changes arrays."""
    result = {}
    for rec, pred in zip(records, preds):
        result.setdefault(rec.doc_id, []).append(int(pred))
    return result