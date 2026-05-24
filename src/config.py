"""
Configuration for the PAN 2026 Multi-Author Writing Style Analysis system.

Task: Given a document written by multiple authors, predict for each pair of
consecutive sentences whether there is a style change (1) or not (0).

Evaluation: F1-macro across all sentence pairs.

Key data statistics (PAN 2026, from analyze_data.py — 2026-04-21):
  Per-difficulty positive rate (style-change pairs):
    easy:   ~30.5%  — documents span many topics; topic shift is a usable signal
    medium: ~4.4%   — same/similar topic; very class-imbalanced; style-only signal
    hard:   ~21.0%  — strictly same topic; pure style signal needed

  Average sentences per document (mean ± std, PAN 2026):
    easy:   53 ± 44  (median=34,  max=162)  → ~52 pairs/doc
    medium: 125 ± 27 (median=133, max=175)  → ~124 pairs/doc
    hard:   57 ± 44  (median=40,  max=169)  → ~56 pairs/doc

  Token budget (window context, approx words×1.35+3):
    easy:   p95=246 tok — MAX_LENGTH=256 covers 96.0%
    medium: p95=205 tok — MAX_LENGTH=256 covers 98.9%
    hard:   p95=230 tok at WS=1; ~320 tok at WS=2 — MAX_LENGTH=384 covers 97%+

Approach (3-level ensemble):
  1. DeBERTa-v3 pairwise classifier with window context + R-Drop regularization
     — base for easy (sufficient, fast); large for medium/hard (richer style signal).
     — R-Drop runs each batch twice with different dropout masks and adds a KL
       consistency loss, acting as a strong regularizer (Wu et al., 2021).
     — WINDOW_SIZE=2 for hard: 3 sentences per side gives more style evidence.
  2. LightGBM on handcrafted stylometric features + SBERT distance
  3. Per-difficulty threshold calibration on the validation set

Training data:
  All four years of available data are combined per difficulty:
    PAN 2026: 10,500 train + 2,250 val  (primary)
    PAN 2025:  4,200 train +   900 val
    PAN 2024:  4,200 train +   900 val
    PAN 2023:  4,200 train +   900 val
    PAN 2022:  ~7,000 train (paragraph-level but compatible format)
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"

DEBERTA_DIR_TMPL  = str(MODELS_DIR / "deberta_{difficulty}")
LGBM_PATH_TMPL    = str(MODELS_DIR / "lgbm_{difficulty}.pkl")
SSPC_PATH_TMPL    = str(MODELS_DIR / "sspc_{difficulty}.pt")
ENSEMBLE_PATH     = MODELS_DIR / "ensemble_config.pkl"

_DATA_BASE = ROOT_DIR.parent / "DATA"

DATA_2026 = _DATA_BASE / "19068843" / "mawsa26-pan-zenodo" / "mawsa26-pan-zenodo"
DATA_2025 = _DATA_BASE / "14891299" / "pan25-multi-author-analysis"
DATA_2024 = _DATA_BASE / "10677876" / "pan24-multi-author-analysis"
DATA_2023 = _DATA_BASE / "7729178"  / "pan23-multi-author-analysis" / "release"
DATA_2022 = _DATA_BASE / "6334245"  / "pan22"

DIFFICULTIES = ["easy", "medium", "hard"]

# ─── Transformer model selection ──────────────────────────────────────────────
# easy  → deberta-v3-base  (near-perfect; base is sufficient and fast)
# medium/hard → deberta-v3-large (richer representations for subtle style signal)
TRANSFORMER_MODEL_NAME = "microsoft/deberta-v3-base"   # fallback / easy

TRANSFORMER_MODEL_PER_DIFF = {
    "easy":   "microsoft/deberta-v3-base",
    "medium": "microsoft/deberta-v3-large",
    "hard":   "microsoft/deberta-v3-large",
}

# ─── Sequence length ──────────────────────────────────────────────────────────
# hard uses WINDOW_SIZE=2 → 3 sentences per side → needs more token budget
MAX_LENGTH = 256   # fallback / easy / medium

MAX_LENGTH_PER_DIFF = {
    "easy":   256,
    "medium": 256,
    "hard":   384,   # covers p97+ of hard pairs at WINDOW_SIZE=2
}

# ─── Window size (sentences of context on each side of the boundary) ──────────
# easy/medium: 1 → 2-sentence style sample each side
# hard: 2 → 3-sentence style sample each side (more evidence for pure style signal)
WINDOW_SIZE = 1    # fallback / easy / medium

WINDOW_SIZE_PER_DIFF = {
    "easy":   1,
    "medium": 1,
    "hard":   2,
}

# ─── Training hyperparameters ─────────────────────────────────────────────────
# Per-difficulty batch sizes tuned for a shared A100 80 GB (~40 GiB available).
# R-Drop doubles activation memory; FGM adds one more forward → peak ~3× normal.
# Keeping batch small avoids OOM on shared GPUs; accum compensates for effective
# batch size (all difficulties target effective batch = 128).
#
# Memory budget (shared A100, ~40 GiB free, base model):
#   easy   (base, b=32, accum=4, R-Drop+FGM): ~10–12 GB
#   medium (base, b=16, accum=8, R-Drop+FGM): ~8–10 GB
#   hard   (base, b=16, accum=8, R-Drop+FGM): ~8–10 GB
#
# If the GPU is unshared (full 80 GB), you can double these batch sizes.
TRAIN_BATCH_SIZE            = 32    # fallback
EVAL_BATCH_SIZE             = 256
GRADIENT_ACCUMULATION_STEPS = 4     # fallback; effective = 128

TRAIN_BATCH_SIZE_PER_DIFF = {
    "easy":   32,   # effective 32×4 = 128
    "medium": 16,   # effective 16×8 = 128
    "hard":   16,   # effective 16×8 = 128
}

GRAD_ACCUM_PER_DIFF = {
    "easy":   4,    # effective batch = 32×4 = 128
    "medium": 8,    # effective batch = 16×8 = 128
    "hard":   8,    # effective batch = 16×8 = 128
}

# Enable gradient checkpointing for large models (saves ~70% activation memory).
# Adds ~30-35% extra compute per epoch — worth it to fit in 20 GiB.
GRADIENT_CHECKPOINTING_PER_DIFF = {
    "easy":   False,   # base model fits fine without it
    "medium": True,    # large model requires it to stay under 20 GiB
    "hard":   True,
}

NUM_EPOCHS = 15    # fallback

NUM_EPOCHS_PER_DIFF = {
    "easy":   15,   # already near-perfect; early stopping triggers early
    "medium": 25,   # was hitting max at 15 and still improving
    "hard":   30,   # was still improving at epoch 15
}

# Large model needs a gentler LR; hard needs the most stable schedule
LEARNING_RATE = 2e-5   # fallback / easy

LEARNING_RATE_PER_DIFF = {
    "easy":   2e-5,
    "medium": 1.5e-5,   # deberta-v3-large
    "hard":   1e-5,     # deberta-v3-large + longer training
}

WARMUP_RATIO     = 0.06
WEIGHT_DECAY     = 0.01

FP16 = False
BF16 = True   # BF16 on A100; .float() in train_model ensures no dtype mismatch

LABEL_SMOOTHING  = 0.05

# DataLoader worker processes for faster data loading
DATALOADER_NUM_WORKERS = 4

# ─── FGM adversarial training ────────────────────────────────────────────────
# Fast Gradient Method (FGM, Miyato et al.): perturbs word embeddings in the
# gradient direction before the optimizer step, then runs a second forward+backward
# pass on the perturbed model. The adversarial gradients are added to the normal
# gradients before the single optimizer step → smoother loss landscape, better
# generalisation. Consistently adds +1–3% F1 in NLP competitions.
# Set FGM_EPS=0.0 to disable.
FGM_EPS = 0.5    # L2 norm of embedding perturbation (typical range: 0.3–1.0)

# ─── EMA (Exponential Moving Average of weights) ─────────────────────────────
# Keep a shadow copy of weights as a running EMA after every optimizer step.
# Use EMA weights for validation evaluation and for the saved best model.
# Consistently gives +0.5–2% F1 with zero extra training time cost.
# Set EMA_DECAY < 0 to disable.
EMA_DECAY = 0.9995   # higher = smoother averaging (typical: 0.999–0.9999)

# ─── LLRD (Layer-wise Learning Rate Decay) ────────────────────────────────────
# Apply a multiplicative LR decay from top encoder layers to bottom layers.
# Bottom layers (closer to raw text) get the lowest LR to preserve pre-trained
# representations; task-specific head layers get the full base LR.
# lr_layer_i = base_lr * LLRD_DECAY^(num_layers - i)
# Consistently adds +0.5–2% F1 on transformer fine-tuning tasks.
# Set LLRD_DECAY=1.0 to disable (uniform LR for all layers).
LLRD_DECAY = 0.9   # per-layer multiplier (typical: 0.8–0.95)

# ─── Focal Loss ───────────────────────────────────────────────────────────────
# Focal Loss down-weights well-classified easy examples so the model focuses on
# hard-to-classify pairs. Critical for medium difficulty (4.4% positive rate)
# where most negative pairs are trivially easy.
# FL(p_t) = -(1-p_t)^gamma * log(p_t)  ;  gamma=0 → standard CE
# Applied only to medium and hard (where class imbalance/hard examples dominate).
FOCAL_GAMMA = 2.0                              # focusing parameter
FOCAL_DIFFICULTIES = {"medium", "hard"}        # apply focal loss for these

# ─── R-Drop regularization ────────────────────────────────────────────────────
# R-Drop (Wu et al., 2021): each batch is forwarded twice with different dropout
# masks; the bidirectional KL divergence between the two output distributions is
# added as a consistency loss. Confirmed effective in PAN 2024 competition.
# Set RDROP_ALPHA=0.0 to disable.
RDROP_ALPHA = 0.3

# ─── Virtual Softmax ──────────────────────────────────────────────────────────
# Team baker (PAN 2024): adds a phantom 3rd class to the classifier head.
# With labels ∈ {0,1} and 3-class output, the softmax denominator gains an
# extra term → the model must push the real class logits higher to achieve
# the same probability → enforces stricter, more confident decision boundaries.
# At inference, we renormalise over the first 2 (real) classes only.
USE_VIRTUAL_SOFTMAX = True

# ─── Content-Agnostic Contrastive Loss ───────────────────────────────────────
# InfoNCE loss with in-batch hard negatives for hard/medium difficulties.
# For each "no-change" pair (anchor), hard negatives are the "style-change" pairs
# in the same batch that come from the same document — they share topic content
# but have different authors. This forces the model to rely on style, not topic.
# Set CACO_ALPHA=0.0 to disable; effective range 0.1–0.3.
CACO_ALPHA = 0.2   # weight of Content-Agnostic Contrastive loss

# ─── Class imbalance ──────────────────────────────────────────────────────────
POS_RATE = {
    "easy":   0.305,
    "medium": 0.044,
    "hard":   0.210,
}

LGBM_SCALE_POS_WEIGHT = {
    "easy":   (1.0 - 0.305) / 0.305,   # 2.3
    "medium": 1.0,
    "hard":   1.0,
}

# ─── LightGBM hyperparameters ─────────────────────────────────────────────────
LGBM_N_ESTIMATORS      = 500
LGBM_LEARNING_RATE     = 0.05
LGBM_NUM_LEAVES        = 63
LGBM_MIN_CHILD_SAMPLES = 20
LGBM_COLSAMPLE_BYTREE  = 0.8
LGBM_SUBSAMPLE         = 0.8

# ─── Ensemble ─────────────────────────────────────────────────────────────────
DEFAULT_ENSEMBLE_WEIGHTS = {
    "easy":   [0.70, 0.30],
    "medium": [0.65, 0.35],
    "hard":   [0.70, 0.30],
}

DEFAULT_THRESHOLD = {
    "easy":   0.40,
    "medium": 0.12,
    "hard":   0.35,
}

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR      = ROOT_DIR / "logs"
LOG_INTERVAL = 50

# ─── Prepared data ────────────────────────────────────────────────────────────
PREPARED_DIR        = ROOT_DIR / "data_prepared"
PREPARED_TRAIN_TMPL = str(PREPARED_DIR / "train_{difficulty}.jsonl")
PREPARED_VAL_TMPL   = str(PREPARED_DIR / "val_{difficulty}.jsonl")

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED = 42