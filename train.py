#!/usr/bin/env python3
"""
Training script for PAN 2026 Multi-Author Writing Style Analysis.

Steps for each difficulty level (easy / medium / hard):
  1. Load and deduplicate training + validation data from all years (2022-2026).
  2. Train DeBERTa-v3-base pairwise classifier with SCL loss.
  3. Build stylometric + SBERT-distance pair features; train LightGBM.
  4. Train SSPC (BiLSTM over sentence embeddings) for document-level context.
  5. Run all three models on the validation set.
  6. Grid-search over (blend weights, threshold, smooth_sigma) for F1-macro.
  7. Save calibrated EnsembleConfig.

Usage:
    python train.py                             # train all three models
    python train.py --difficulties easy hard    # subset of difficulties
    python train.py --skip-transformer          # skip DeBERTa
    python train.py --skip-lgbm                 # skip LightGBM
    python train.py --skip-sspc                 # skip SSPC
    python train.py --no-scl                    # disable contrastive loss
    python train.py --no-2022 --no-2023         # only recent years
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATA_2022, DATA_2023, DATA_2024, DATA_2025, DATA_2026,
    DIFFICULTIES, MODELS_DIR,
    DEBERTA_DIR_TMPL, LGBM_PATH_TMPL, SSPC_PATH_TMPL, ENSEMBLE_PATH,
    PREPARED_TRAIN_TMPL, PREPARED_VAL_TMPL,
    WINDOW_SIZE, WINDOW_SIZE_PER_DIFF,
    MAX_LENGTH_PER_DIFF,
    TRANSFORMER_MODEL_PER_DIFF,
    NUM_EPOCHS_PER_DIFF,
    LEARNING_RATE_PER_DIFF,
    TRAIN_BATCH_SIZE_PER_DIFF, TRAIN_BATCH_SIZE,
    GRAD_ACCUM_PER_DIFF, GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING_PER_DIFF,
    LOG_DIR,
)
from src.training_logger import TrainingLogger
from src.data import (
    load_2022, load_2023, load_2024, load_2025, load_2026,
    make_pair_records, load_problems_jsonl,
)
from src.features import build_pair_feature_matrix
from src.classical_models import StylemetricClassifier
from src.ensemble import calibrate_all, EnsembleConfig, predict_binary


def parse_args():
    p = argparse.ArgumentParser(description="Train PAN 2026 style change detection models")
    p.add_argument("--difficulties", nargs="+", default=DIFFICULTIES, choices=DIFFICULTIES)
    p.add_argument("--skip-transformer", action="store_true",
                   help="Skip DeBERTa-SCL training")
    p.add_argument("--skip-lgbm", action="store_true",
                   help="Skip LightGBM training")
    p.add_argument("--skip-sspc", action="store_true",
                   help="Skip SSPC (BiLSTM) training")
    p.add_argument("--force-lgbm", action="store_true",
                   help="Force retrain LightGBM even if model file exists")
    p.add_argument("--force-sspc", action="store_true",
                   help="Force retrain SSPC even if model file exists")
    p.add_argument("--force-transformer", action="store_true",
                   help="Force retrain DeBERTa even if model directory exists")
    p.add_argument("--resume", action="store_true",
                   help="Resume DeBERTa training from the latest checkpoint (_latest dir)")
    p.add_argument("--warm-start", action="store_true",
                   help="Initialise DeBERTa from the best existing checkpoint, then train "
                        "from epoch 0 with the current config (R-Drop, CACO, Virtual Softmax, "
                        "more epochs). Handles 2→3 class head expansion automatically. "
                        "Forces deberta-v3-base for all difficulties so the checkpoint "
                        "architecture matches the saved weights.")
    p.add_argument("--no-scl", action="store_true",
                   help="Disable supervised contrastive loss for DeBERTa")
    p.add_argument("--no-rdrop", action="store_true",
                   help="Disable R-Drop regularization for DeBERTa")
    p.add_argument("--no-fgm", action="store_true",
                   help="Disable FGM adversarial training on word embeddings")
    p.add_argument("--no-ema", action="store_true",
                   help="Disable EMA weight averaging (use raw weights for eval/save)")
    p.add_argument("--no-llrd", action="store_true",
                   help="Disable layer-wise LR decay (use flat LR for all layers)")
    p.add_argument("--no-focal", action="store_true",
                   help="Disable Focal Loss; use weighted CE for all difficulties")
    p.add_argument("--no-2022", action="store_true")
    p.add_argument("--no-2023", action="store_true")
    p.add_argument("--no-2024", action="store_true")
    p.add_argument("--no-2025", action="store_true")
    p.add_argument("--no-2026", action="store_true")
    p.add_argument("--window-size", type=int, default=None,
                   help="Override per-difficulty WINDOW_SIZE for all difficulties")
    p.add_argument("--force-raw", action="store_true",
                   help="Ignore prepared JSONL files; reload from raw data directories")
    return p.parse_args()


def _hash_doc(prob) -> str:
    return hashlib.sha256("\n".join(prob.sentences).encode()).hexdigest()


def _load_and_dedup_raw(difficulty: str, split: str, args) -> list:
    """Load and SHA-256 deduplicate problems from all selected raw year directories."""
    all_problems, seen = [], set()

    loaders = []
    if not args.no_2026 and DATA_2026.exists():
        loaders.append(("2026", lambda: load_2026(DATA_2026, difficulty, split)))
    if not args.no_2025 and DATA_2025.exists():
        loaders.append(("2025", lambda: load_2025(DATA_2025, difficulty, split)))
    if not args.no_2024 and DATA_2024.exists():
        loaders.append(("2024", lambda: load_2024(DATA_2024, difficulty, split)))
    if not args.no_2023 and DATA_2023.exists():
        loaders.append(("2023", lambda: load_2023(DATA_2023, difficulty, split)))
    if not args.no_2022 and DATA_2022.exists() and split == "train":
        loaders.append(("2022", lambda: load_2022(DATA_2022, difficulty, "train")))

    for year, loader_fn in loaders:
        try:
            probs = loader_fn()
        except Exception as e:
            print(f"  [WARN] {year}/{difficulty}/{split}: {e}")
            continue
        n_before = len(all_problems)
        for p in probs:
            h = _hash_doc(p)
            if h not in seen:
                seen.add(h)
                p.id = f"{year}_{p.id}"  # keep doc_ids unique across years
                all_problems.append(p)
        print(f"  {year}/{difficulty}/{split}: +{len(all_problems)-n_before} (of {len(probs)})")

    return all_problems


def _load_problems(difficulty: str, split: str, args) -> list:
    """
    Load problems for one difficulty / split.

    Tries prepared JSONL first (fast single-file read).
    Falls back to raw year-by-year loading with SHA-256 dedup if:
      - --force-raw is set, or
      - the prepared file does not exist yet (run prepare_data.py first).
    """
    if not args.force_raw:
        tmpl = PREPARED_TRAIN_TMPL if split == "train" else PREPARED_VAL_TMPL
        prepared = Path(tmpl.format(difficulty=difficulty))
        if prepared.exists():
            print(f"  Using prepared: {prepared}")
            return load_problems_jsonl(prepared, difficulty=difficulty)
        print(f"  Prepared file not found ({prepared.name}); loading from raw data.")

    return _load_and_dedup_raw(difficulty, split, args)


def _check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("=" * 60)
        print("ERROR: No GPU detected. Training cannot start.")
        print("  DeBERTa and SSPC fine-tuning require a CUDA-capable GPU.")
        print("  Please run on a machine with at least one NVIDIA GPU.")
        print("=" * 60)
        sys.exit(1)
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU check passed: {n} device(s) — {name} ({mem_gb:.1f} GB)")


def main():
    args = parse_args()
    _check_gpu()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    val_deberta  = {}
    val_lgbm     = {}
    val_sspc     = {}
    val_labels   = {}
    val_records_map = {}

    for difficulty in args.difficulties:
        print(f"\n{'='*60}")
        print(f"DIFFICULTY: {difficulty.upper()}")
        print("="*60)

        # ── 1. Load data ─────────────────────────────────────────────────────
        print("Loading training data:")
        train_probs = _load_problems(difficulty, "train", args)
        print("Loading validation data:")
        val_probs   = _load_problems(difficulty, "validation", args)

        train_records = make_pair_records(train_probs, difficulty)
        val_records   = make_pair_records(val_probs,   difficulty)
        labels_val    = np.array([r.label for r in val_records], dtype=np.int32)

        n_pos = int(labels_val.sum())
        print(f"\nTrain pairs: {len(train_records):,} | Val pairs: {len(labels_val):,} "
              f"({100*n_pos/max(len(labels_val),1):.1f}% positive)")

        val_labels[difficulty]      = labels_val
        val_records_map[difficulty] = val_records

        deberta_dir = Path(DEBERTA_DIR_TMPL.format(difficulty=difficulty))
        lgbm_path   = Path(LGBM_PATH_TMPL.format(difficulty=difficulty))
        sspc_path   = Path(SSPC_PATH_TMPL.format(difficulty=difficulty))

        # ── Per-difficulty transformer settings ──────────────────────────────
        _preferred_model = TRANSFORMER_MODEL_PER_DIFF.get(difficulty, "microsoft/deberta-v3-base")
        # Graceful fallback: if deberta-v3-large is not cached locally, use base.
        # Run the download command in the mifawzy terminal first to unlock large:
        #   (mo) mifawzy$ python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/deberta-v3-large')"
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(_preferred_model, local_files_only=True)
            diff_model_name = _preferred_model
        except Exception:
            fallback = "microsoft/deberta-v3-base"
            if _preferred_model != fallback:
                print(f"  [WARN] {_preferred_model} not cached locally — falling back to {fallback}.")
                print(f"         Run: python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{_preferred_model}')\"")
            diff_model_name = fallback

        diff_window_size  = args.window_size if args.window_size is not None \
                            else WINDOW_SIZE_PER_DIFF.get(difficulty, WINDOW_SIZE)
        diff_max_length   = MAX_LENGTH_PER_DIFF.get(difficulty, 256)
        diff_num_epochs   = NUM_EPOCHS_PER_DIFF.get(difficulty, 15)
        diff_lr           = LEARNING_RATE_PER_DIFF.get(difficulty, 2e-5)
        diff_batch        = TRAIN_BATCH_SIZE_PER_DIFF.get(difficulty, TRAIN_BATCH_SIZE)
        diff_grad_accum   = GRAD_ACCUM_PER_DIFF.get(difficulty, GRADIENT_ACCUMULATION_STEPS)
        diff_grad_ckpt    = GRADIENT_CHECKPOINTING_PER_DIFF.get(difficulty, False)
        # If falling back to base, revert large-model settings
        if diff_model_name == "microsoft/deberta-v3-base":
            diff_lr         = 2e-5
            diff_max_length = min(diff_max_length, 256)
            diff_grad_ckpt  = False   # base doesn't need it

        # Warm start: the existing checkpoints were trained with deberta-v3-base.
        # Weights can't transfer to deberta-v3-large (different hidden size).
        # Force base for all difficulties when warm start is requested.
        if args.warm_start and diff_model_name != "microsoft/deberta-v3-base":
            print(f"  [warm-start] Overriding {diff_model_name.split('/')[-1]} → deberta-v3-base "
                  f"to match existing checkpoint architecture.")
            diff_model_name = "microsoft/deberta-v3-base"
            diff_lr         = 2e-5
            diff_max_length = min(diff_max_length, 256)
            diff_grad_ckpt  = False

        # ── 2. DeBERTa-SCL ───────────────────────────────────────────────────
        warm_starting   = args.warm_start
        deberta_trained = (deberta_dir / "config.json").exists() and not args.force_transformer
        resuming        = args.resume and not args.force_transformer and not warm_starting

        should_train = not args.skip_transformer and (
            not deberta_trained or resuming or
            (warm_starting and args.force_transformer)
        )

        if should_train:
            if warm_starting:
                action = "Warm-starting"
            elif resuming and deberta_trained:
                action = "Resuming"
            else:
                action = "Training"
            print(f"\n[1/3] {action} SCL-DeBERTa for '{difficulty}' "
                  f"({diff_model_name.split('/')[-1]}, "
                  f"window={diff_window_size}, max_len={diff_max_length}, "
                  f"epochs={diff_num_epochs}, lr={diff_lr})...")
        elif deberta_trained:
            print(f"\n[1/3] SCL-DeBERTa already trained for '{difficulty}' — skipping.")

        if should_train:
            from src.transformer_model import train_model
            deberta_logger = TrainingLogger(f"deberta_{difficulty}", LOG_DIR)
            deberta_logger.info("difficulty=%s  train=%d  val=%d",
                                difficulty, len(train_records), len(val_records))
            train_model(
                train_records=train_records,
                val_records=val_records,
                difficulty=difficulty,
                save_dir=deberta_dir,
                model_name=diff_model_name,
                window_size=diff_window_size,
                max_length=diff_max_length,
                num_epochs=diff_num_epochs,
                learning_rate=diff_lr,
                train_batch_size=diff_batch,
                grad_accum_steps=diff_grad_accum,
                grad_checkpointing=diff_grad_ckpt,
                use_scl=not args.no_scl,
                use_rdrop=not args.no_rdrop,
                use_fgm=not args.no_fgm,
                use_ema=not args.no_ema,
                use_llrd=not args.no_llrd,
                use_focal=not args.no_focal,
                resume=resuming,
                warm_start=warm_starting,
                logger=deberta_logger,
            )
            deberta_logger.close()

        if deberta_dir.exists():
            from src.transformer_model import load_model, predict_proba
            import torch
            model, tok = load_model(deberta_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            print("  DeBERTa val inference...")
            val_deberta[difficulty] = predict_proba(
                model, tok, val_records,
                window_size=diff_window_size,
                max_length=diff_max_length,
            )
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"  [SKIP] DeBERTa model not found for '{difficulty}'.")
            val_deberta[difficulty] = np.full(len(val_records), 0.5, dtype=np.float32)

        # ── 3. LightGBM ──────────────────────────────────────────────────────
        X_val = None
        lgbm_trained = lgbm_path.exists() and not args.force_lgbm
        if not args.skip_lgbm and not lgbm_trained:
            print(f"\n[2/3] Building stylometric features + LightGBM for '{difficulty}'...")
        elif lgbm_trained:
            print(f"\n[2/3] LightGBM already trained for '{difficulty}' — skipping.")
        if not args.skip_lgbm and not lgbm_trained:
            X_train = build_pair_feature_matrix(train_records, window_size=diff_window_size)
            y_train = np.array([r.label for r in train_records], dtype=np.int32)
            X_val   = build_pair_feature_matrix(val_records,   window_size=diff_window_size)
            print(f"  Features: {X_train.shape}")
            lgbm_logger = TrainingLogger(f"lgbm_{difficulty}", LOG_DIR)
            lgbm_logger.info("difficulty=%s  train=%d  val=%d  features=%d",
                             difficulty, len(X_train), len(X_val), X_train.shape[1])
            clf = StylemetricClassifier(difficulty)
            clf.fit(X_train, y_train, X_val=X_val, y_val=labels_val, logger=lgbm_logger)
            clf.save(lgbm_path)
            # Log final val F1 for LightGBM
            from sklearn.metrics import f1_score as _f1
            lgbm_val_preds = (clf.predict_proba(X_val) >= 0.5).astype(int)
            lgbm_f1 = float(_f1(labels_val, lgbm_val_preds, average="macro", zero_division=0))
            lgbm_logger.log_final(best_f1=lgbm_f1, n_estimators=clf.model.best_iteration_ or 0)
            lgbm_logger.close()
            print(f"  LightGBM saved → {lgbm_path}")

        if lgbm_path.exists():
            if X_val is None:
                X_val = build_pair_feature_matrix(val_records, window_size=diff_window_size)
            clf = StylemetricClassifier.load(lgbm_path)
            print("  LightGBM val inference...")
            val_lgbm[difficulty] = clf.predict_proba(X_val)
        else:
            print(f"  [SKIP] LightGBM model not found for '{difficulty}'.")
            val_lgbm[difficulty] = np.full(len(val_records), 0.5, dtype=np.float32)

        # ── 4. SSPC (BiLSTM) ─────────────────────────────────────────────────
        sspc_trained = sspc_path.exists() and not args.force_sspc
        if not args.skip_sspc and not sspc_trained:
            print(f"\n[3/3] Training SSPC (BiLSTM) for '{difficulty}'...")
        elif sspc_trained:
            print(f"\n[3/3] SSPC already trained for '{difficulty}' — skipping.")
        if not args.skip_sspc and not sspc_trained:
            from src.sspc_model import train_sspc
            sspc_logger = TrainingLogger(f"sspc_{difficulty}", LOG_DIR)
            sspc_logger.info("difficulty=%s  train_docs=%d  val_docs=%d",
                             difficulty, len(train_probs), len(val_probs))
            train_sspc(
                train_problems=train_probs,
                val_problems=val_probs,
                difficulty=difficulty,
                save_path=sspc_path,
                logger=sspc_logger,
            )
            sspc_logger.close()

        if sspc_path.exists():
            from src.sspc_model import load_sspc, predict_sspc_from_records
            import torch
            sspc_model = load_sspc(sspc_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sspc_model = sspc_model.to(device)
            print("  SSPC val inference...")
            val_sspc[difficulty] = predict_sspc_from_records(sspc_model, val_records)
            del sspc_model
        else:
            print(f"  [SKIP] SSPC model not found for '{difficulty}'.")

    # ── 5. Ensemble calibration ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Calibrating ensemble (grid-search: weights + threshold + smoothing)...")
    print("="*60)

    ens_cfg = calibrate_all(
        val_deberta,
        val_lgbm,
        val_labels,
        val_sspc=val_sspc if val_sspc else None,
        val_records=val_records_map,
    )
    ens_cfg.save(ENSEMBLE_PATH)
    print(f"\nEnsemble config saved → {ENSEMBLE_PATH}")

    # ── 6. Final validation summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final validation F1-macro (calibrated ensemble):")
    from sklearn.metrics import f1_score

    for diff in args.difficulties:
        if diff not in val_labels:
            continue
        sspc_arr = val_sspc.get(diff)
        preds = predict_binary(
            val_deberta[diff], val_lgbm[diff], diff, ens_cfg,
            sspc_probs=sspc_arr,
            records=val_records_map.get(diff),
        )
        f1 = f1_score(val_labels[diff], preds, average="macro", zero_division=0)
        print(f"  {diff:6s}: F1-macro = {f1:.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()