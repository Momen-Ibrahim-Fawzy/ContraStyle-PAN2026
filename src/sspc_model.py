"""
Sequential Sentence Pair Classifier (SSPC) for style change detection.

Architecture (from "better_call_claude" team, PAN 2025, arXiv:2508.00675):

  1. Sentence encoder: encodes EACH sentence INDEPENDENTLY using mean pooling
     over a frozen/fine-tuned transformer (SBERT or DeBERTa).
  2. BiLSTM: processes the full sequence of sentence embeddings for a document,
     so each position gets DOCUMENT-LEVEL contextual information.
  3. Pair classifier: for each adjacent pair (i, i+1), concatenate
     BiLSTM[i] and BiLSTM[i+1] → MLP → binary classification.

Why this is complementary to the pairwise DeBERTa model:
  — DeBERTa (pairwise): encodes sent_i + sent_{i+1} TOGETHER; cross-attention
    directly compares the two sides. Sees only a ±window context.
  — SSPC: encodes every sentence INDEPENDENTLY; BiLSTM propagates information
    across the ENTIRE document. Detects consistent author styles by comparing
    them globally (e.g. "this author always uses short sentences").

For the hard subset (same topic), document-level style consistency is the key
discriminating signal — SSPC directly exploits it.

Results from "better_call_claude" (PAN 2025):
  F1-macro: easy=0.923, medium=0.828, hard=0.724

We use a frozen SBERT encoder (all-mpnet-base-v2, 768-dim) for the sentence
representations. This is faster than fine-tuning per-sentence DeBERTa and
still captures strong style-aware embeddings.

References:
  — "better_call_claude" (arXiv:2508.00675)
  — "Pretrained Language Models for Sequential Sentence Classification"
    (Cohan et al., EMNLP 2019, arXiv:1909.04054)
"""
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import SEED

# ─── Hyperparameters ─────────────────────────────────────────────────────────
# ─── Encoder choice ──────────────────────────────────────────────────────────
# "sbert"          → all-mpnet-base-v2 (768-dim semantic model, default)
# "style_distance" → AnonymousSub/ScientificLayersOfAuthorship-2-mpnet-base
#                    (768-dim model trained specifically to capture writing
#                     STYLE while ignoring content — ideal for same-topic tasks)
SSPC_ENCODER = "style_distance"   # change to "sbert" to revert

SBERT_MODEL_NAME          = "all-mpnet-base-v2"
STYLE_DISTANCE_MODEL_NAME = "AnonymousSub/ScientificLayersOfAuthorship-2-mpnet-base"
SENTENCE_EMBED_DIM = 768
LSTM_HIDDEN        = 256
LSTM_LAYERS        = 2
LSTM_DROPOUT       = 0.3
MLP_HIDDEN         = 256
SSPC_BATCH_DOCS    = 16    # documents per batch (variable sentence count)
SSPC_LR            = 1e-3
SSPC_EPOCHS        = 70
SSPC_PATIENCE      = 10
MAX_SENT_PER_DOC   = 200   # truncate very long documents

# Per-difficulty overrides.
# Hard SSPC consistently overfits (val_f1 peaks at epoch 7, train_loss still falling):
#   → larger hidden dim for more capacity, higher dropout + weight decay for regularisation,
#     lower LR with cosine decay, and longer patience to escape local plateaus.
SSPC_LSTM_HIDDEN_PER_DIFF = {
    "easy":   256,
    "medium": 256,
    "hard":   512,   # more capacity for pure-style signal
}
SSPC_DROPOUT_PER_DIFF = {
    "easy":   0.3,
    "medium": 0.3,
    "hard":   0.5,   # strong regularisation to fight overfitting
}
SSPC_LR_PER_DIFF = {
    "easy":   1e-3,
    "medium": 1e-3,
    "hard":   5e-4,  # gentler LR; cosine schedule does the rest
}
SSPC_WEIGHT_DECAY_PER_DIFF = {
    "easy":   1e-4,
    "medium": 1e-4,
    "hard":   1e-3,  # stronger L2 for hard
}
SSPC_PATIENCE_PER_DIFF = {
    "easy":   10,
    "medium": 10,
    "hard":   15,   # harder task needs more patience
}


def _set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Sentence encoder ─────────────────────────────────────────────────────────

_encoder_cache: dict = {}

def _get_encoder(encoder_name: str = SSPC_ENCODER):
    """
    Lazily load and cache the sentence encoder.

    encoder_name:
      "sbert"          → all-mpnet-base-v2 (semantic similarity)
      "style_distance" → StyleDistance model (content-agnostic style)
                         Wegmann et al. 2022 — trained specifically to capture
                         writing style while ignoring topic content.
                         HuggingFace: AnonymousSub/ScientificLayersOfAuthorship-2-mpnet-base
    """
    global _encoder_cache
    if encoder_name in _encoder_cache:
        return _encoder_cache[encoder_name]

    from sentence_transformers import SentenceTransformer

    if encoder_name == "style_distance":
        model_id = STYLE_DISTANCE_MODEL_NAME
    else:
        model_id = SBERT_MODEL_NAME

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # avoid network calls if model is cached
    try:
        enc = SentenceTransformer(model_id)
        print(f"  SSPC encoder loaded: {model_id}")
    except Exception as e:
        print(f"  [WARN] Could not load {model_id} ({e}); falling back to {SBERT_MODEL_NAME}")
        enc = SentenceTransformer(SBERT_MODEL_NAME)
        print(f"  SSPC encoder loaded (fallback): {SBERT_MODEL_NAME}")

    _encoder_cache[encoder_name] = enc
    return enc

# Keep the old name for backward compatibility
def _get_sbert_encoder():
    return _get_encoder(SSPC_ENCODER)


def encode_sentences_batch(sentences: list, device=None,
                            encoder_name: str = SSPC_ENCODER) -> torch.Tensor:
    """
    Encode a list of sentences with the configured encoder (mean pooling).
    Returns (N, SENTENCE_EMBED_DIM) float32 tensor.
    """
    enc = _get_encoder(encoder_name)
    with torch.no_grad():
        embs = enc.encode(
            sentences,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
    if device is not None:
        embs = embs.to(device)
    return embs.float()


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SSPCDocDataset(Dataset):
    """
    Each item is one document: (sentence_embeddings, change_labels).

    sentence_embeddings: (N_sents, SENTENCE_EMBED_DIM)
    change_labels:       (N_sents - 1,)  int labels (0/1)
    """

    def __init__(self, problems: list, device=None, max_sents: int = MAX_SENT_PER_DOC):
        self.items = []
        sbert = _get_sbert_encoder()

        all_docs = [(p.sentences[:max_sents], p.changes[:max_sents - 1])
                    for p in problems
                    if p.changes is not None and len(p.sentences) >= 2]

        for sentences, changes in tqdm(all_docs, desc="SBERT encoding", unit="doc", dynamic_ncols=True):
            embs = encode_sentences_batch(sentences, device=device)  # (N, D)
            labels = torch.tensor(changes[:len(sentences) - 1], dtype=torch.long)
            self.items.append((embs, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class SSPCTestDocDataset(Dataset):
    """Dataset for inference (no labels)."""

    def __init__(self, problems: list, device=None, max_sents: int = MAX_SENT_PER_DOC):
        self.items   = []
        self.doc_ids = []
        for p in problems:
            if len(p.sentences) < 2:
                continue
            sents = p.sentences[:max_sents]
            embs  = encode_sentences_batch(sents, device=device)
            self.items.append(embs)
            self.doc_ids.append(p.id)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _collate_fn(batch):
    """
    Variable-length documents → padded tensors.
    batch: list of (embs, labels) tuples
    """
    embs_list, labels_list = zip(*batch)
    max_n = max(e.shape[0] for e in embs_list)

    padded_embs   = torch.zeros(len(batch), max_n, SENTENCE_EMBED_DIM)
    padded_labels = torch.full((len(batch), max_n - 1), fill_value=-1, dtype=torch.long)
    lengths       = []

    for i, (embs, labels) in enumerate(zip(embs_list, labels_list)):
        n = embs.shape[0]
        padded_embs[i, :n]       = embs
        padded_labels[i, :n - 1] = labels
        lengths.append(n)

    return padded_embs, padded_labels, torch.tensor(lengths, dtype=torch.long)


def _collate_test_fn(batch):
    """Collate for inference (no labels)."""
    max_n = max(e.shape[0] for e in batch)
    padded = torch.zeros(len(batch), max_n, SENTENCE_EMBED_DIM)
    lengths = []
    for i, embs in enumerate(batch):
        n = embs.shape[0]
        padded[i, :n] = embs
        lengths.append(n)
    return padded, torch.tensor(lengths, dtype=torch.long)


# ─── Model ────────────────────────────────────────────────────────────────────

class SSPCModel(nn.Module):
    """
    BiLSTM contextualiser + MLP classifier for document-level style change.

    Input:  (B, N, SENTENCE_EMBED_DIM)  batch of padded sentence embeddings
    Output: (B, N-1, 2)                 logits per adjacent pair
    """

    def __init__(
        self,
        embed_dim:    int   = SENTENCE_EMBED_DIM,
        lstm_hidden:  int   = LSTM_HIDDEN,
        lstm_layers:  int   = LSTM_LAYERS,
        mlp_hidden:   int   = MLP_HIDDEN,
        dropout:      float = LSTM_DROPOUT,
    ):
        # Store for load_sspc reconstruction
        self._lstm_hidden = lstm_hidden
        self._dropout     = dropout
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Pair classifier: concat(BiLSTM[i], BiLSTM[i+1]) → logits
        pair_dim = lstm_hidden * 4    # 2 (bidirectional) × 2 (both sentences)
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, N, D)
        lengths:    torch.Tensor,  # (B,)
    ) -> torch.Tensor:             # (B, N-1, 2)
        B, N, D = embeddings.shape

        # Pack padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        ctx, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=N
        )
        ctx = self.dropout(ctx)   # (B, N, 2*lstm_hidden)

        # Build pair representations: concat ctx[i] and ctx[i+1]
        left_ctx  = ctx[:, :-1, :]   # (B, N-1, 2H)
        right_ctx = ctx[:, 1:, :]    # (B, N-1, 2H)
        pair_repr = torch.cat([left_ctx, right_ctx], dim=-1)  # (B, N-1, 4H)

        logits = self.classifier(pair_repr)   # (B, N-1, 2)
        return logits


# ─── Training ─────────────────────────────────────────────────────────────────

def train_sspc(
    train_problems: list,
    val_problems:   list,
    difficulty:     str,
    save_path:      Path,
    logger=None,
) -> None:
    """
    Train the SSPC model for one difficulty level.

    Saves model weights to save_path (a .pt file).
    Per-difficulty settings control hidden size, dropout, LR, and patience.
    A cosine annealing LR schedule is applied for smoother convergence.
    """
    _set_seed(SEED)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm_hidden  = SSPC_LSTM_HIDDEN_PER_DIFF.get(difficulty, LSTM_HIDDEN)
    dropout      = SSPC_DROPOUT_PER_DIFF.get(difficulty, LSTM_DROPOUT)
    lr           = SSPC_LR_PER_DIFF.get(difficulty, SSPC_LR)
    weight_decay = SSPC_WEIGHT_DECAY_PER_DIFF.get(difficulty, 1e-4)
    patience     = SSPC_PATIENCE_PER_DIFF.get(difficulty, SSPC_PATIENCE)

    print(f"  SSPC device: {device}  hidden={lstm_hidden}  dropout={dropout}  "
          f"lr={lr}  wd={weight_decay}  patience={patience}")

    print("  Building train dataset (SBERT encoding)...")
    train_ds = SSPCDocDataset(train_problems, device=device)
    print("  Building val dataset (SBERT encoding)...")
    val_ds   = SSPCDocDataset(val_problems,   device=device)

    print(f"  Train docs: {len(train_ds)} | Val docs: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=SSPC_BATCH_DOCS, shuffle=True,
        collate_fn=_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=SSPC_BATCH_DOCS, shuffle=False,
        collate_fn=_collate_fn, num_workers=0,
    )

    from .config import POS_RATE
    pos_rate   = POS_RATE.get(difficulty, 0.3)
    pos_weight = torch.tensor([(1.0 - pos_rate) / pos_rate], device=device)

    model = SSPCModel(
        lstm_hidden=lstm_hidden,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Cosine annealing restarts every SSPC_EPOCHS steps → smooth LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SSPC_EPOCHS, eta_min=lr * 0.01
    )

    best_f1, no_improve = 0.0, 0

    for epoch in range(SSPC_EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"SSPC epoch {epoch+1}/{SSPC_EPOCHS}", unit="batch", dynamic_ncols=True)
        for embs, labels, lengths in pbar:
            embs    = embs.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)

            logits = model(embs, lengths)   # (B, N-1, 2)
            B, M, _ = logits.shape

            # Flatten, mask padding (label == -1)
            flat_logits = logits.view(-1, 2)
            flat_labels = labels.view(-1)
            mask = flat_labels >= 0

            flat_logits = flat_logits[mask]
            flat_labels = flat_labels[mask]

            # Class-weighted cross-entropy
            weights = torch.cat([torch.ones(1, device=device), pos_weight])
            loss = F.cross_entropy(flat_logits, flat_labels, weight=weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        avg_loss = total_loss / max(len(train_loader), 1)

        # ── Validate ───────────────────────────────────────────────────────
        val_f1 = _eval_sspc(model, val_loader, device)
        epoch_mins = (time.time() - epoch_start) / 60
        gpu_mem = f"{torch.cuda.memory_reserved() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if val_f1 > best_f1:
            improved = "✓ best"
        else:
            improved = f"no improve {no_improve+1}/{patience}"

        print(
            f"  SSPC epoch {epoch+1}/{SSPC_EPOCHS} | "
            f"loss={avg_loss:.4f} | val_F1={val_f1:.4f} | "
            f"best_F1={max(best_f1, val_f1):.4f} | {improved} | "
            f"lr={lr_now:.2e} | time={epoch_mins:.1f}m | GPU={gpu_mem}"
        )

        if logger:
            logger.log_epoch(
                epoch + 1, avg_loss, val_f1,
                best_f1=max(best_f1, val_f1),
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"    Saved (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  SSPC best F1: {best_f1:.4f}")
    if logger:
        logger.log_final(best_f1=best_f1)


def _eval_sspc(model, loader, device, threshold: float = 0.5) -> float:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for embs, labels, lengths in tqdm(loader, desc="SSPC val eval", unit="batch", dynamic_ncols=True, leave=False):
            embs    = embs.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)

            logits  = model(embs, lengths)
            probs   = torch.softmax(logits, dim=-1)[:, :, 1]  # (B, N-1)

            flat_probs  = probs.view(-1).cpu()
            flat_labels = labels.view(-1).cpu()
            mask        = flat_labels >= 0

            preds  = (flat_probs[mask] >= threshold).int().tolist()
            true_l = flat_labels[mask].int().tolist()

            all_preds.extend(preds)
            all_labels.extend(true_l)

    from sklearn.metrics import f1_score
    return float(f1_score(all_labels, all_preds, average="macro", zero_division=0))


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_sspc(model_path: Path, difficulty: str = "") -> SSPCModel:
    """Load SSPC model, reconstructing the right hidden size for the difficulty."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    # Infer lstm_hidden from the saved weight shape (lstm.weight_ih_l0 rows = 4*hidden)
    lstm_hidden = state["lstm.weight_ih_l0"].shape[0] // 4
    model = SSPCModel(lstm_hidden=lstm_hidden)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_sspc_proba(
    model:     SSPCModel,
    problems:  list,
    batch_size: int = SSPC_BATCH_DOCS,
) -> dict:
    """
    Run SSPC inference on a list of Problem objects.

    Returns {doc_id: np.ndarray of shape (N-1,)} with P(label=1) per pair.
    """
    device = next(model.parameters()).device
    dataset = SSPCTestDocDataset(problems, device=device)

    if len(dataset) == 0:
        return {}

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_test_fn, num_workers=0,
    )

    model.eval()
    results = {}
    idx = 0

    with torch.no_grad():
        for embs, lengths in tqdm(loader, desc="SSPC inference", unit="batch", dynamic_ncols=True):
            embs    = embs.to(device)
            lengths = lengths.to(device)

            logits = model(embs, lengths)          # (B, N-1, 2)
            probs  = torch.softmax(logits, dim=-1)[:, :, 1]  # (B, N-1)

            for b, n in enumerate(lengths.cpu().tolist()):
                doc_id = dataset.doc_ids[idx]
                pair_probs = probs[b, :n - 1].cpu().numpy().astype(np.float32)
                results[doc_id] = pair_probs
                idx += 1

    return results


def predict_sspc_from_records(
    model:   SSPCModel,
    records: list,
) -> np.ndarray:
    """
    Run SSPC inference and return a flat array aligned with a list of PairRecords.

    Groups records by document, runs SSPC per document, then re-aligns.
    """
    if not records:
        return np.array([], dtype=np.float32)

    # Rebuild minimal Problem-like objects per document
    from .data import Problem
    doc_map = {}
    for rec in records:
        if rec.doc_id not in doc_map:
            doc_map[rec.doc_id] = (rec.sentences, [])
        doc_map[rec.doc_id][1].append(rec.pair_idx)

    # Build fake Problem objects (no changes needed for inference)
    fake_problems = [
        Problem(id=doc_id, sentences=sents_and_idxs[0])
        for doc_id, sents_and_idxs in doc_map.items()
    ]

    proba_by_doc = predict_sspc_proba(model, fake_problems)

    # Re-align to the original records order
    result = np.zeros(len(records), dtype=np.float32)
    for i, rec in enumerate(records):
        doc_probs = proba_by_doc.get(rec.doc_id, None)
        if doc_probs is not None and rec.pair_idx < len(doc_probs):
            result[i] = float(doc_probs[rec.pair_idx])
        else:
            result[i] = 0.5

    return result