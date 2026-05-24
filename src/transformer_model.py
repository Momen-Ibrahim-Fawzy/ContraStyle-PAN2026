"""
DeBERTa-v3 pairwise classifier for style change detection.

Architecture: SCL-DeBERTa + R-Drop
  — DeBERTa-v3-base (easy) or DeBERTa-v3-large (medium/hard) encodes a sentence
    pair with optional window context.
  — The [CLS] representation is fed into:
      (a) a binary classification head  → Cross-Entropy loss
      (b) an L2-normalised projection head → Supervised Contrastive loss
  — R-Drop: each batch is forwarded twice with different dropout masks; the
    bidirectional KL divergence between the two softmax distributions is added
    as a consistency regularisation loss (Wu et al., 2021).
  — The total loss = CE + SCL_ALPHA * SCL + RDROP_ALPHA * KL

Input format (with WINDOW_SIZE = 1):
  tokenizer(
      left_text  = "sent_{i-1}\\nsent_i",
      right_text = "sent_{i+1}\\nsent_{i+2}",
  )
  → [CLS] sent_{i-1} \\n sent_i [SEP] sent_{i+1} \\n sent_{i+2} [SEP]

With WINDOW_SIZE = 2 (hard difficulty):
  → [CLS] s_{i-2}\\ns_{i-1}\\ns_i [SEP] s_{i+1}\\ns_{i+2}\\ns_{i+3} [SEP]

References:
  — DeBERTa: He et al., ICLR 2021.
  — SCL: Khosla et al., NeurIPS 2020.
  — R-Drop: Wu et al., NeurIPS 2021.
"""
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .config import (
    TRANSFORMER_MODEL_NAME, MAX_LENGTH, MAX_LENGTH_PER_DIFF, WINDOW_SIZE,
    TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE_PER_DIFF,
    EVAL_BATCH_SIZE, LEARNING_RATE, LEARNING_RATE_PER_DIFF,
    NUM_EPOCHS, NUM_EPOCHS_PER_DIFF,
    GRADIENT_ACCUMULATION_STEPS, GRAD_ACCUM_PER_DIFF,
    GRADIENT_CHECKPOINTING_PER_DIFF,
    WARMUP_RATIO, WEIGHT_DECAY, FP16, BF16,
    LABEL_SMOOTHING, POS_RATE, SEED,
    LOG_INTERVAL, RDROP_ALPHA, DATALOADER_NUM_WORKERS,
    USE_VIRTUAL_SOFTMAX, CACO_ALPHA,
    FGM_EPS, EMA_DECAY, LLRD_DECAY, FOCAL_GAMMA, FOCAL_DIFFICULTIES,
)
from .data import get_pair_texts, PairRecord

SCL_ALPHA       = 0.3
SCL_TEMPERATURE = 0.1
SCL_PROJ_DIM    = 128


# ─── FGM (Fast Gradient Method) ───────────────────────────────────────────────

class FGM:
    """
    Fast Gradient Method adversarial training (Miyato et al., 2017).

    Perturbs the word-embedding layer in the gradient direction before the
    optimizer step, runs an adversarial forward+backward, then restores the
    original embeddings.  The adversarial gradients accumulate on top of the
    normal gradients → single optimizer step sees a combined update that is
    more robust to embedding perturbations.

    Only the embedding layer is attacked (not all weights) — this is the
    standard NLP competition approach (fast, effective, low overhead).
    """
    def __init__(self, model: nn.Module, eps: float = FGM_EPS):
        self.model  = model
        self.eps    = eps
        self.backup: dict = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None \
                    and 'word_embeddings' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()


# ─── EMA (Exponential Moving Average of weights) ──────────────────────────────

class EMA:
    """
    Maintains a shadow copy of model weights as a running EMA.

    Call .update() after every optimizer step.
    Call .apply_shadow() before evaluation / saving; .restore() after.
    Using EMA weights for inference consistently gives +0.5–2% F1 for free.
    """
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.model  = model
        self.decay  = decay
        self.shadow: dict = {}
        self.backup: dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.float()
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.data.dtype))

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with class weights and label smoothing.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Down-weights easy, well-classified examples so training focuses on hard
    pairs.  Critical for medium difficulty (4.4% positive rate).

    Implementation: compute standard CE per-sample, then multiply by the
    focal modulation factor (1 - exp(-CE))^gamma.  This is equivalent to
    the original formulation and compatible with label smoothing.
    """
    def __init__(self, gamma: float = FOCAL_GAMMA,
                 weight: torch.Tensor = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self._ce   = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing, reduction="none"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce     = self._ce(logits, targets)          # (B,)  per-sample CE
        pt     = torch.exp(-ce.detach())            # ≈ probability of correct class
        weight = (1.0 - pt) ** self.gamma
        return (weight * ce).mean()


# ─── LLRD optimizer ───────────────────────────────────────────────────────────

def _make_llrd_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    llrd_decay: float = LLRD_DECAY,
) -> torch.optim.AdamW:
    """
    Create AdamW with Layer-wise LR Decay (LLRD).

    Parameters closer to the input get a lower LR; the classification heads
    get the full base_lr.  This prevents catastrophic forgetting of pre-trained
    representations in early encoder layers.

    lr_layer_i = base_lr * llrd_decay^(num_layers - i)
    """
    import re
    no_decay_keywords = {"bias", "LayerNorm.weight", "layernorm.weight"}

    orig = getattr(model, "_orig_mod", model)

    # Detect number of encoder layers from config
    try:
        num_layers = orig.encoder.config.num_hidden_layers
    except AttributeError:
        num_layers = 12  # safe default for DeBERTa-v3-base

    def _lr_for(name: str) -> float:
        if "classifier" in name or "projector" in name:
            return base_lr
        m = re.search(r"layer\.(\d+)\.", name)
        if m:
            layer_idx = int(m.group(1))
            return base_lr * (llrd_decay ** (num_layers - layer_idx - 1))
        # Embeddings and pre-layer params get the lowest LR
        return base_lr * (llrd_decay ** num_layers)

    def _no_decay(name: str) -> bool:
        return any(kw in name for kw in no_decay_keywords)

    # Build per-parameter groups (group by (lr, no_decay) to merge duplicates)
    groups: dict = {}
    for name, param in orig.named_parameters():
        if not param.requires_grad:
            continue
        lr = round(_lr_for(name), 10)
        nd = _no_decay(name)
        key = (lr, nd)
        groups.setdefault(key, []).append(param)

    param_groups = [
        {"params": ps, "lr": lr, "weight_decay": 0.0 if nd else weight_decay}
        for (lr, nd), ps in groups.items()
    ]
    return torch.optim.AdamW(param_groups)


def _set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class StyleChangePairDataset(Dataset):
    def __init__(self, records: list, tokenizer, window_size: int = WINDOW_SIZE,
                 max_length: int = MAX_LENGTH):
        self.tokenizer   = tokenizer
        self.window_size = window_size
        self.max_length  = max_length
        self.items = []

        # Build a stable integer doc_id mapping for the CACO loss
        doc_id_map: dict = {}
        for rec in records:
            left_text, right_text = get_pair_texts(rec, window_size=window_size)
            if rec.doc_id not in doc_id_map:
                doc_id_map[rec.doc_id] = len(doc_id_map)
            self.items.append((left_text, right_text, rec.label, doc_id_map[rec.doc_id]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        left, right, label, doc_int_id = self.items[idx]
        enc = self.tokenizer(
            text=left,
            text_pair=right,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(0),
            "label":  torch.tensor(label,       dtype=torch.long),
            "doc_id": torch.tensor(doc_int_id,  dtype=torch.long),
        }


# ─── Model ────────────────────────────────────────────────────────────────────

class StyleChangeClassifier(nn.Module):
    """
    DeBERTa-v3 + classification head + SCL projection head.

    virtual_softmax=True adds a phantom 3rd class (Team baker, PAN 2024):
      larger softmax denominator → stricter decision boundaries.
      At inference, renormalise over first 2 (real) classes only.
    """

    def __init__(self, model_name: str = TRANSFORMER_MODEL_NAME,
                 virtual_softmax: bool = USE_VIRTUAL_SOFTMAX):
        super().__init__()
        self.encoder        = AutoModel.from_pretrained(model_name)
        hidden              = self.encoder.config.hidden_size
        self.dropout        = nn.Dropout(0.1)
        n_out               = 3 if virtual_softmax else 2
        self.classifier     = nn.Linear(hidden, n_out)
        self.virtual_softmax = virtual_softmax
        self.projector      = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, SCL_PROJ_DIM),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls     = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits  = self.classifier(cls)     # (B, 2) or (B, 3)
        proj    = F.normalize(self.projector(cls), dim=-1)
        return logits, proj


# ─── Supervised Contrastive Loss ─────────────────────────────────────────────

def supervised_contrastive_loss(
    features: torch.Tensor,
    labels:   torch.Tensor,
    temperature: float = SCL_TEMPERATURE,
) -> torch.Tensor:
    B      = features.shape[0]
    device = features.device
    sim    = torch.mm(features, features.T) / temperature
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim    = sim.masked_fill(self_mask, float("-inf"))
    log_probs = F.log_softmax(sim, dim=1)
    label_eq  = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask  = label_eq & ~self_mask
    n_pos     = pos_mask.sum(dim=1).float()
    valid     = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    pos_loss = -(log_probs.masked_fill(~pos_mask, 0.0)).sum(dim=1)
    pos_loss  = pos_loss[valid] / n_pos[valid]
    return pos_loss.mean()


# ─── R-Drop KL loss ───────────────────────────────────────────────────────────

def rdrop_kl_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two logit tensors (R-Drop)."""
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)
    kl1 = F.kl_div(F.log_softmax(logits1, dim=-1), p2, reduction="batchmean")
    kl2 = F.kl_div(F.log_softmax(logits2, dim=-1), p1, reduction="batchmean")
    return (kl1 + kl2) / 2.0


def content_agnostic_contrastive_loss(
    features:  torch.Tensor,   # (B, D) L2-normalised projections
    labels:    torch.Tensor,   # (B,)   0=no-change, 1=style-change
    doc_ids:   torch.Tensor,   # (B,)   integer document id per pair
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Content-Agnostic Contrastive Loss with in-batch hard negatives.

    For each "no-change" anchor (label=0), the hard negatives are "style-change"
    pairs (label=1) from the SAME DOCUMENT. These share the same topic/content
    as the anchor but have a different author — forcing the model to rely on
    writing style rather than topic overlap.

    Pairs without a hard negative in the batch are skipped.
    """
    device = features.device
    B      = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    sim = torch.mm(features, features.T) / temperature   # (B, B)
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_anchors  = 0

    for i in range(B):
        if labels[i] != 0:          # anchor must be a no-change pair
            continue
        doc = doc_ids[i]
        # Hard negatives: style-change pairs from the SAME document
        hard_neg_mask = (labels == 1) & (doc_ids == doc) & ~self_mask[i]
        if hard_neg_mask.sum() == 0:
            continue
        # Positives: other no-change pairs (any document)
        pos_mask = (labels == 0) & ~self_mask[i]
        if pos_mask.sum() == 0:
            continue

        # InfoNCE: anchor vs all non-self; treat hard negatives as the "easy to confuse" set
        log_probs = F.log_softmax(sim[i], dim=0)   # (B,)
        # Pull: maximise similarity to positives
        pos_loss = -log_probs[pos_mask].mean()
        total_loss = total_loss + pos_loss
        n_anchors += 1

    return total_loss / max(n_anchors, 1)


# ─── Training ─────────────────────────────────────────────────────────────────

def _adapt_heads_state(heads_state: dict, model) -> dict:
    """
    Adapt a saved heads state dict to the current model's classifier shape.

    Handles the 2→3 class expansion needed when Virtual Softmax is enabled but
    the checkpoint was saved without it (n_out=2 → n_out=3).
    """
    clf_state  = dict(heads_state["classifier"])  # copy
    ckpt_n_out = clf_state["weight"].shape[0]
    model_n_out = model.classifier.weight.shape[0]

    if ckpt_n_out != model_n_out:
        hidden   = clf_state["weight"].shape[1]
        new_w    = torch.zeros(model_n_out, hidden, dtype=clf_state["weight"].dtype)
        new_b    = torch.zeros(model_n_out,          dtype=clf_state["bias"].dtype)
        k = min(ckpt_n_out, model_n_out)
        new_w[:k] = clf_state["weight"][:k]
        new_b[:k] = clf_state["bias"][:k]
        # Extra phantom class: small random init so it doesn't dominate early
        if model_n_out > ckpt_n_out:
            nn.init.normal_(new_w[ckpt_n_out:], 0.0, 0.02)
        clf_state["weight"] = new_w
        clf_state["bias"]   = new_b
        print(f"  Classifier head expanded: {ckpt_n_out}→{model_n_out} classes")

    return {**heads_state, "classifier": clf_state}


def train_model(
    train_records:        list,
    val_records:          list,
    difficulty:           str,
    save_dir:             Path,
    model_name:           str   = TRANSFORMER_MODEL_NAME,
    window_size:          int   = WINDOW_SIZE,
    max_length:           int   = MAX_LENGTH,
    num_epochs:           int   = NUM_EPOCHS,
    learning_rate:        float = LEARNING_RATE,
    train_batch_size:     int   = TRAIN_BATCH_SIZE,
    grad_accum_steps:     int   = GRADIENT_ACCUMULATION_STEPS,
    grad_checkpointing:   bool  = False,
    use_scl:              bool  = True,
    use_rdrop:            bool  = True,
    use_fgm:              bool  = True,
    use_ema:              bool  = True,
    use_llrd:             bool  = True,
    use_focal:            bool  = True,
    resume:               bool  = False,
    warm_start:           bool  = False,
    logger=None,
) -> None:
    """
    Fine-tune DeBERTa for style change detection on one difficulty level.

    Saves the model (weights + tokenizer) to save_dir.

    resume=True     — continue an interrupted run from save_dir+'_latest',
                      restoring epoch counter and patience state.
    warm_start=True — use an existing checkpoint as weight initialisation
                      (best model in save_dir, else _latest), then train
                      from epoch 0 with the current config (all new losses).
                      Handles 2→3 class head expansion for Virtual Softmax.
                      resume and warm_start are mutually exclusive; warm_start
                      takes priority if both are set.
    use_fgm         — FGM adversarial training on word embeddings (+1–3% F1).
    use_ema         — EMA weight averaging; EMA weights used for eval/save.
    use_llrd        — Layer-wise LR decay; bottom layers get lower LR.
    use_focal       — Focal Loss for medium/hard (replaces weighted CE).
    """
    _set_seed(SEED)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = save_dir.parent / (save_dir.name + "_latest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff_batch = train_batch_size * grad_accum_steps
    print(f"  Device: {device}  model={model_name.split('/')[-1]}")
    print(f"  window={window_size}  max_len={max_length}  epochs={num_epochs}  lr={learning_rate}")
    print(f"  batch={train_batch_size}  accum={grad_accum_steps}  eff_batch={eff_batch}  "
          f"grad_ckpt={grad_checkpointing}  rdrop={use_rdrop}  virtual_softmax={USE_VIRTUAL_SOFTMAX}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    train_ds = StyleChangePairDataset(train_records, tokenizer, window_size, max_length)
    val_ds   = StyleChangePairDataset(val_records,   tokenizer, window_size, max_length)
    print(f"  Train pairs: {len(train_ds):,} | Val pairs: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=(DATALOADER_NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=(DATALOADER_NUM_WORKERS > 0),
    )

    model = StyleChangeClassifier(model_name).float().to(device)

    # Gradient checkpointing: recomputes activations during backward instead of
    # storing them → saves ~70% activation VRAM at ~30% extra compute cost.
    if grad_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED")

    # ── Optional: warm start / resume from checkpoint ────────────────────────
    start_epoch = 0
    best_f1     = 0.0
    no_improve  = 0

    if warm_start:
        # Find best checkpoint first, fall back to _latest
        warm_dir = None
        for candidate in [save_dir, latest_dir]:
            if candidate.exists() and (candidate / "config.json").exists():
                warm_dir = candidate
                break
        if warm_dir is not None:
            print(f"  Warm start: loading encoder from {warm_dir}")
            model.encoder = AutoModel.from_pretrained(warm_dir).to(device)
            heads_path = warm_dir / "heads.pt"
            if heads_path.exists():
                raw = torch.load(heads_path, map_location=device, weights_only=True)
                adapted = _adapt_heads_state(raw, model)
                model.classifier.load_state_dict(adapted["classifier"])
                model.projector.load_state_dict(adapted["projector"])
                model.dropout.load_state_dict(adapted["dropout"])
                ckpt_epoch = raw.get("epoch", "?")
                ckpt_f1    = raw.get("val_f1", raw.get("best_f1", 0.0))
                print(f"  Warm start: restored from epoch={ckpt_epoch}  f1={ckpt_f1:.4f}")
            print(f"  Warm start: training from epoch 0 for {num_epochs} epochs with new config")
        else:
            print(f"  Warm start: no checkpoint found at {save_dir}; training from scratch.")

    elif resume and latest_dir.exists() and (latest_dir / "config.json").exists():
        print(f"  Resuming from latest checkpoint: {latest_dir}")
        model.encoder = AutoModel.from_pretrained(latest_dir).to(device)
        heads_path = latest_dir / "heads.pt"
        if heads_path.exists():
            raw     = torch.load(heads_path, map_location=device, weights_only=True)
            adapted = _adapt_heads_state(raw, model)
            model.classifier.load_state_dict(adapted["classifier"])
            model.projector.load_state_dict(adapted["projector"])
            model.dropout.load_state_dict(adapted["dropout"])
            start_epoch = raw.get("epoch", 0)
            best_f1     = raw.get("best_f1", raw.get("val_f1", 0.0))
            no_improve  = raw.get("no_improve", 0)
        print(f"  Loaded: epoch={start_epoch}  best_f1={best_f1:.4f}  no_improve={no_improve}")
    elif resume:
        print(f"  --resume set but no checkpoint found at {latest_dir}; training from scratch.")

    remaining_epochs = num_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"  Already trained {start_epoch}/{num_epochs} epochs. Nothing to do.")
        return

    # ── torch.compile for ~20% speedup on A100 (PyTorch 2.x) ────────────────
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})")

    pos_rate    = POS_RATE.get(difficulty, 0.3)
    pos_weight  = (1.0 - pos_rate) / pos_rate

    # When Virtual Softmax is used the classifier has 3 outputs.
    # CrossEntropyLoss requires weight tensor length == n_classes.
    # The phantom 3rd class is never a true label, so its weight doesn't
    # matter for learning — we give it weight=1.0 (neutral).
    orig_model   = getattr(model, "_orig_mod", model)
    n_cls        = orig_model.classifier.weight.shape[0]   # 2 or 3
    w_base       = [1.0, pos_weight] + [1.0] * (n_cls - 2)
    class_weight = torch.tensor(w_base, dtype=torch.float32, device=device)

    # Focal Loss for imbalanced difficulties; standard CE for easy
    _use_focal = use_focal and difficulty in FOCAL_DIFFICULTIES
    if _use_focal:
        ce_loss_fn = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weight,
                               label_smoothing=LABEL_SMOOTHING)
        print(f"  Loss: FocalLoss(gamma={FOCAL_GAMMA}, n_cls={n_cls})")
    else:
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weight,
                                         label_smoothing=LABEL_SMOOTHING)
        print(f"  Loss: CrossEntropyLoss(n_cls={n_cls})")

    # Optimizer: LLRD or flat AdamW
    if use_llrd:
        optimizer = _make_llrd_optimizer(model, learning_rate, WEIGHT_DECAY)
        print(f"  Optimizer: AdamW + LLRD (decay={LLRD_DECAY})")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
        )
        print(f"  Optimizer: AdamW (flat LR)")

    # FGM and EMA setup
    _fgm = FGM(model, eps=FGM_EPS) if use_fgm and FGM_EPS > 0 else None
    _ema = EMA(model, decay=EMA_DECAY) if use_ema and EMA_DECAY > 0 else None
    if _fgm:
        print(f"  FGM: ENABLED (eps={FGM_EPS})")
    if _ema:
        print(f"  EMA: ENABLED (decay={EMA_DECAY})")
    steps_per_epoch = max(len(train_loader) // grad_accum_steps, 1)
    total_steps     = steps_per_epoch * remaining_epochs
    warmup_steps    = max(int(total_steps * WARMUP_RATIO), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_amp  = (FP16 or BF16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if BF16 else torch.float16
    scaler    = torch.amp.GradScaler("cuda") if (FP16 and device.type == "cuda") else None

    rdrop_enabled = use_rdrop and RDROP_ALPHA > 0.0
    global_step   = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        accum_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)
            doc_ids        = batch["doc_id"].to(device)

            caco_enabled = (CACO_ALPHA > 0.0)

            def _compute_loss(l, proj):
                """CE + SCL + (optional) CACO from a single forward pass."""
                ce   = ce_loss_fn(l, labels)
                scl  = (supervised_contrastive_loss(proj, labels)
                        if use_scl else torch.tensor(0.0, device=device))
                caco = (content_agnostic_contrastive_loss(proj, labels, doc_ids)
                        if caco_enabled else torch.tensor(0.0, device=device))
                return ce, scl, caco

            def _forward_loss():
                if rdrop_enabled:
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            l1, p1 = model(input_ids, attention_mask)
                            l2, p2 = model(input_ids, attention_mask)
                            ce1, scl1, caco1 = _compute_loss(l1, p1)
                            ce2, scl2, caco2 = _compute_loss(l2, p2)
                            ce   = (ce1 + ce2) / 2
                            scl  = (scl1 + scl2) / 2
                            caco = (caco1 + caco2) / 2
                            kl   = rdrop_kl_loss(l1, l2)
                    else:
                        l1, p1 = model(input_ids, attention_mask)
                        l2, p2 = model(input_ids, attention_mask)
                        ce1, scl1, caco1 = _compute_loss(l1, p1)
                        ce2, scl2, caco2 = _compute_loss(l2, p2)
                        ce   = (ce1 + ce2) / 2
                        scl  = (scl1 + scl2) / 2
                        caco = (caco1 + caco2) / 2
                        kl   = rdrop_kl_loss(l1, l2)
                    return (ce + SCL_ALPHA * scl + RDROP_ALPHA * kl + CACO_ALPHA * caco) \
                           / grad_accum_steps
                else:
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            logits, proj = model(input_ids, attention_mask)
                            ce, scl, caco = _compute_loss(logits, proj)
                    else:
                        logits, proj = model(input_ids, attention_mask)
                        ce, scl, caco = _compute_loss(logits, proj)
                    return (ce + SCL_ALPHA * scl + CACO_ALPHA * caco) \
                           / grad_accum_steps

            loss = _forward_loss()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                # ── FGM adversarial step ─────────────────────────────────────
                # Attack word embeddings → ONE simple CE forward on perturbed
                # model → accumulate adversarial gradients → restore embeddings.
                # Using a SINGLE forward (not R-Drop's two) halves FGM memory.
                # The adversarial gradient direction still improves generalisation.
                if _fgm is not None:
                    if scaler:
                        scaler.unscale_(optimizer)   # need real grads for attack
                    _fgm.attack()
                    # Simple single forward for adversarial gradient — no R-Drop
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            _adv_logits, _ = model(input_ids, attention_mask)
                            adv_loss = ce_loss_fn(_adv_logits, labels) / grad_accum_steps
                    else:
                        _adv_logits, _ = model(input_ids, attention_mask)
                        adv_loss = ce_loss_fn(_adv_logits, labels) / grad_accum_steps
                    if scaler:
                        scaler.scale(adv_loss).backward()
                    else:
                        adv_loss.backward()
                    _fgm.restore()

                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()

                # ── EMA update ───────────────────────────────────────────────
                if _ema is not None:
                    _ema.update()

                optimizer.zero_grad()
                global_step += 1

                step_loss  = accum_loss / grad_accum_steps
                epoch_loss += step_loss
                accum_loss  = 0.0

                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr_now:.2e}")

                if logger and global_step % LOG_INTERVAL == 0:
                    logger.log_step(global_step, step_loss, lr=lr_now)

        pbar.close()
        avg_loss = epoch_loss / max(steps_per_epoch, 1)

        # Evaluate with EMA weights when available (better generalisation)
        if _ema is not None:
            _ema.apply_shadow()
        val_f1, val_loss = evaluate_model(model, val_loader, device, ce_loss_fn)
        if _ema is not None:
            _ema.restore()
        epoch_mins = (time.time() - epoch_start) / 60
        gpu_mem = f"{torch.cuda.memory_reserved() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
        improved = "✓ best" if val_f1 > best_f1 else f"no improve {no_improve+1}/4"
        print(
            f"  Epoch {epoch+1}/{num_epochs} | "
            f"loss={avg_loss:.4f} | val_loss={val_loss:.4f} | val_F1={val_f1:.4f} | "
            f"best_F1={max(best_f1, val_f1):.4f} | {improved} | "
            f"time={epoch_mins:.1f}m | GPU={gpu_mem}"
        )

        if logger:
            logger.log_epoch(epoch + 1, avg_loss, val_f1,
                             val_loss=val_loss, best_f1=max(best_f1, val_f1))

        # Save latest checkpoint (including state for resume)
        latest_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(model, tokenizer, latest_dir, epoch + 1,
                         val_f1=val_f1, best_f1=max(best_f1, val_f1),
                         no_improve=no_improve + (0 if val_f1 > best_f1 else 1))

        if val_f1 > best_f1:
            best_f1    = val_f1
            no_improve = 0
            # Save EMA weights as the best model (already validated with EMA)
            if _ema is not None:
                _ema.apply_shadow()
            _save_checkpoint(model, tokenizer, save_dir)
            if _ema is not None:
                _ema.restore()
            print(f"    Saved best model (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= 4:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best validation F1: {best_f1:.4f}")
    if logger:
        logger.log_final(best_f1=best_f1, total_steps=global_step)


def _save_checkpoint(model, tokenizer, out_dir: Path,
                     epoch: int = 0, val_f1: float = 0.0,
                     best_f1: float = 0.0, no_improve: int = 0):
    """Save encoder weights, tokenizer, and head weights to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Unwrap torch.compile if needed
    encoder = getattr(model, "_orig_mod", model).encoder
    encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    classifier = getattr(model, "_orig_mod", model).classifier
    projector  = getattr(model, "_orig_mod", model).projector
    dropout    = getattr(model, "_orig_mod", model).dropout
    torch.save({
        "classifier": classifier.state_dict(),
        "projector":  projector.state_dict(),
        "dropout":    dropout.state_dict(),
        "epoch":      epoch,
        "val_f1":     val_f1,
        "best_f1":    best_f1,
        "no_improve": no_improve,
    }, out_dir / "heads.pt")


def evaluate_model(model, loader, device, loss_fn=None) -> tuple:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    vs_enabled = getattr(model, "virtual_softmax", False) or \
                 getattr(getattr(model, "_orig_mod", model), "virtual_softmax", False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="DeBERTa val eval", unit="batch",
                          dynamic_ncols=True, leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            logits, _ = model(input_ids, attention_mask, token_type_ids)

            if loss_fn is not None:
                total_loss += loss_fn(logits, labels).item()
                n_batches  += 1

            # Virtual Softmax: predict from real classes only
            real_logits = logits[:, :2] if vs_enabled else logits
            preds = real_logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    from sklearn.metrics import f1_score
    f1       = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    avg_loss = total_loss / max(n_batches, 1)
    return f1, avg_loss


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(model_dir: Path) -> tuple:
    """Load fine-tuned model and tokenizer from directory."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    model = StyleChangeClassifier.__new__(StyleChangeClassifier)
    nn.Module.__init__(model)
    model.encoder = AutoModel.from_pretrained(model_dir)
    hidden        = model.encoder.config.hidden_size
    model.dropout = nn.Dropout(0.1)

    heads_path = model_dir / "heads.pt"
    if heads_path.exists():
        state = torch.load(heads_path, map_location="cpu", weights_only=True)
        # Detect whether this was saved with Virtual Softmax (3 classes) or not (2)
        n_out = state["classifier"]["weight"].shape[0]
    else:
        n_out = 3 if USE_VIRTUAL_SOFTMAX else 2

    model.classifier     = nn.Linear(hidden, n_out)
    model.virtual_softmax = (n_out == 3)
    model.projector = nn.Sequential(
        nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, SCL_PROJ_DIM)
    )

    if heads_path.exists():
        model.classifier.load_state_dict(state["classifier"])
        model.projector.load_state_dict(state["projector"])
        model.dropout.load_state_dict(state["dropout"])

    model.eval()
    return model, tokenizer


def _predict_proba_loader(model, loader, device, vs_enabled: bool) -> np.ndarray:
    """Run inference on a DataLoader; return P(style_change) array."""
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            logits, _ = model(input_ids, attention_mask, token_type_ids)
            real_logits = logits[:, :2] if vs_enabled else logits
            p = torch.softmax(real_logits, dim=-1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())
    return np.array(probs, dtype=np.float32)


def predict_proba(
    model,
    tokenizer,
    records:       list,
    window_size:   int  = WINDOW_SIZE,
    max_length:    int  = MAX_LENGTH,
    batch_size:    int  = EVAL_BATCH_SIZE,
    symmetric_tta: bool = False,
) -> np.ndarray:
    """
    Return P(style_change) for each PairRecord as a float32 array.

    symmetric_tta=True: also score each pair with left/right sides SWAPPED
    (style comparison is symmetric), then average the two probability sets.
    Gives +0.5–2% F1 for free at inference time.
    """
    if not records:
        return np.array([], dtype=np.float32)

    device     = next(model.parameters()).device
    vs_enabled = getattr(model, "virtual_softmax", False) or \
                 getattr(getattr(model, "_orig_mod", model), "virtual_softmax", False)
    model.eval()

    dataset = StyleChangePairDataset(records, tokenizer, window_size, max_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs = _predict_proba_loader(model, tqdm(
        loader, desc="DeBERTa inference", unit="batch", dynamic_ncols=True
    ), device, vs_enabled)

    if symmetric_tta:
        # Build a mirror dataset: swap left_text ↔ right_text per record.
        # StyleChangePairDataset uses get_pair_texts which concatenates window
        # sentences; we patch by building mirrored records via a subclass.
        mirror_ds = _MirroredPairDataset(records, tokenizer, window_size, max_length)
        mirror_loader = DataLoader(mirror_ds, batch_size=batch_size,
                                   shuffle=False, num_workers=0)
        probs_mirror = _predict_proba_loader(model, tqdm(
            mirror_loader, desc="DeBERTa TTA (symmetric)", unit="batch",
            dynamic_ncols=True,
        ), device, vs_enabled)
        probs = (probs + probs_mirror) / 2.0

    return probs.astype(np.float32)


class _MirroredPairDataset(StyleChangePairDataset):
    """
    StyleChangePairDataset with left/right contexts swapped for symmetric TTA.
    Same items list as parent; __getitem__ tokenizes (right, left) instead of
    (left, right).  Averaging both gives a symmetric boundary score.
    """
    def __getitem__(self, idx):
        left, right, label, doc_int_id = self.items[idx]
        enc = self.tokenizer(
            text=right,         # ← swapped
            text_pair=left,     # ← swapped
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(0),
            "label":  torch.tensor(label,      dtype=torch.long),
            "doc_id": torch.tensor(doc_int_id, dtype=torch.long),
        }
