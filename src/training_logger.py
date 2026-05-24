"""
Training logger for PAN style change detection models.

Each TrainingLogger instance corresponds to one training run (one model ×
one difficulty level).  It writes to three sinks simultaneously:

  1. logs/history/{run_name}.jsonl  — machine-readable event stream used by
                                      dashboard.py to build charts.
  2. logs/tensorboard/{run_name}/   — TensorBoard event files for live
                                      monitoring with:
                                        tensorboard --logdir logs/tensorboard/
  3. logs/training.log              — human-readable append log shared by all
                                      runs (plain text, one line per event).

Events written by this class:

  type="step"        global_step, loss, [lr, ce_loss, scl_loss]
  type="epoch"       epoch, train_loss, val_f1, [val_loss, best_f1]
  type="lgbm_round"  round, [train_logloss, val_logloss]
  type="final"       arbitrary key=value pairs (best metrics summary)

Usage:
  logger = TrainingLogger("deberta_easy", LOG_DIR)
  logger.log_step(100, loss=0.42, lr=2e-5, ce_loss=0.35, scl_loss=0.07)
  logger.log_epoch(1, train_loss=0.45, val_f1=0.82, val_loss=0.41)
  logger.log_final(best_f1=0.85, best_epoch=3)
  logger.close()
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class TrainingLogger:

    def __init__(
        self,
        run_name:        str,
        log_dir:         Path,
        use_tensorboard: bool = True,
    ):
        self.run_name = run_name
        self.log_dir  = Path(log_dir)

        # ── Directories ────────────────────────────────────────────────────
        self.log_dir.mkdir(parents=True, exist_ok=True)
        history_dir = self.log_dir / "history"
        history_dir.mkdir(exist_ok=True)

        # ── JSON Lines history ─────────────────────────────────────────────
        self.jsonl_path  = history_dir / f"{run_name}.jsonl"
        self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

        # ── Python logger (console + shared text file) ────────────────────
        self._logger = self._build_logger(run_name)

        # ── TensorBoard ───────────────────────────────────────────────────
        self.tb: Optional[object] = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir  = self.log_dir / "tensorboard" / run_name
                self.tb = SummaryWriter(log_dir=str(tb_dir))
                self._logger.info("TensorBoard: %s", tb_dir)
                self._logger.info(
                    "  → view with: tensorboard --logdir %s",
                    self.log_dir / "tensorboard",
                )
            except ImportError:
                self._logger.warning(
                    "tensorboard not installed; TensorBoard logging disabled. "
                    "Install with: pip install tensorboard"
                )

        self._logger.info("Run started  ─── %s", run_name)
        self._logger.info("JSONL log: %s", self.jsonl_path)

    # ── Logger setup ──────────────────────────────────────────────────────────

    def _build_logger(self, run_name: str) -> logging.Logger:
        name   = f"train.{run_name}"
        logger = logging.getLogger(name)
        if logger.handlers:          # avoid duplicate handlers on re-import
            return logger
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        fmt = logging.Formatter(
            fmt="%(asctime)s  %(name)-34s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(
            self.log_dir / "training.log", mode="a", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        return logger

    # ── Internal write ────────────────────────────────────────────────────────

    def _write(self, event: dict) -> None:
        event.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        event["run"] = self.run_name
        self._jsonl_file.write(json.dumps(event) + "\n")
        self._jsonl_file.flush()

    # ── Public logging API ────────────────────────────────────────────────────

    def log_step(
        self,
        global_step: int,
        loss:        float,
        lr:          float = None,
        ce_loss:     float = None,
        scl_loss:    float = None,
    ) -> None:
        """Log one optimizer step (called every LOG_INTERVAL steps)."""
        event: dict = {
            "type": "step",
            "global_step": global_step,
            "loss": round(loss, 6),
        }
        if lr       is not None: event["lr"]       = float(lr)
        if ce_loss  is not None: event["ce_loss"]  = round(ce_loss, 6)
        if scl_loss is not None: event["scl_loss"] = round(scl_loss, 6)
        self._write(event)

        if self.tb:
            self.tb.add_scalar("Loss/train_step", loss, global_step)
            if lr       is not None: self.tb.add_scalar("LR",             lr,       global_step)
            if ce_loss  is not None: self.tb.add_scalar("Loss/ce_step",   ce_loss,  global_step)
            if scl_loss is not None: self.tb.add_scalar("Loss/scl_step",  scl_loss, global_step)

    def log_epoch(
        self,
        epoch:      int,
        train_loss: float,
        val_f1:     float,
        val_loss:   float = None,
        best_f1:    float = None,
    ) -> None:
        """Log end-of-epoch metrics."""
        event: dict = {
            "type":       "epoch",
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_f1":     round(val_f1,     6),
        }
        if val_loss is not None: event["val_loss"] = round(val_loss, 6)
        if best_f1  is not None: event["best_f1"]  = round(best_f1,  6)
        self._write(event)

        extras = ""
        if val_loss is not None: extras += f"  val_loss={val_loss:.4f}"
        if best_f1  is not None: extras += f"  best_f1={best_f1:.4f}"
        self._logger.info(
            "epoch=%d  train_loss=%.4f  val_f1=%.4f%s",
            epoch, train_loss, val_f1, extras,
        )

        if self.tb:
            self.tb.add_scalar("Loss/train_epoch", train_loss, epoch)
            self.tb.add_scalar("F1/val",           val_f1,     epoch)
            if val_loss is not None: self.tb.add_scalar("Loss/val_epoch", val_loss, epoch)
            if best_f1  is not None: self.tb.add_scalar("F1/best",        best_f1,  epoch)

    def log_lgbm_round(
        self,
        round_idx:    int,
        train_metric: Optional[float],
        val_metric:   Optional[float],
    ) -> None:
        """Log one LightGBM boosting round (called every LOG_INTERVAL rounds)."""
        event: dict = {"type": "lgbm_round", "round": round_idx}
        if train_metric is not None: event["train_logloss"] = round(train_metric, 6)
        if val_metric   is not None: event["val_logloss"]   = round(val_metric,   6)
        self._write(event)

        if self.tb:
            if train_metric is not None:
                self.tb.add_scalar("LightGBM/train_logloss", train_metric, round_idx)
            if val_metric is not None:
                self.tb.add_scalar("LightGBM/val_logloss",   val_metric,   round_idx)

    def log_final(self, **kwargs) -> None:
        """Log a summary of final/best metrics at the end of training."""
        event: dict = {"type": "final"}
        event.update({
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in kwargs.items()
        })
        self._write(event)
        summary = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in kwargs.items()
        )
        self._logger.info("FINAL  ─── %s", summary)
        if self.tb:
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(f"Final/{k}", float(v), 0)

    def info(self, msg: str, *args) -> None:
        """Write a plain info message to the text log."""
        self._logger.info(msg, *args)

    def close(self) -> None:
        self._jsonl_file.close()
        if self.tb:
            self.tb.flush()
            self.tb.close()
        self._logger.info("Run finished ─── %s\n", self.run_name)