"""
LightGBM classifier on handcrafted stylometric + SBERT-distance features.

This model is complementary to DeBERTa:
  — It captures explicit style signals (punctuation, word-length distributions,
    vocabulary richness, function-word patterns) that are fully topic-agnostic.
  — SBERT cosine distance is included as a feature to capture semantic/style
    distance that the handcrafted features might miss.
  — It runs on CPU in seconds, making it practical even without a GPU.

Class imbalance is handled via LightGBM's `scale_pos_weight` parameter,
which is set to (# negative pairs) / (# positive pairs) per difficulty.
"""
import pickle
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from tqdm import tqdm

from .config import (
    LGBM_SCALE_POS_WEIGHT, LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE,
    LGBM_NUM_LEAVES, LGBM_MIN_CHILD_SAMPLES, LGBM_COLSAMPLE_BYTREE,
    LGBM_SUBSAMPLE, SEED, LOG_INTERVAL,
)


def _make_lgbm_tqdm_callback(n_estimators: int):
    """Return a LightGBM callback that drives a tqdm progress bar over boosting rounds."""
    pbar = tqdm(total=n_estimators, desc="LightGBM", unit="round", dynamic_ncols=True)

    def _cb(env):
        val_loss = None
        for item in env.evaluation_result_list:
            data_name, _metric_name, value, _higher = item
            if "valid" in data_name.lower():
                val_loss = value
        postfix = {"val_auc": f"{val_loss:.4f}"} if val_loss is not None else {}
        pbar.set_postfix(**postfix)
        pbar.update(1)
        if env.iteration + 1 >= env.end_iteration:
            pbar.close()

    _cb.order = 0
    return _cb


def _make_lgbm_log_callback(logger, log_period: int = LOG_INTERVAL):
    """Return a LightGBM callback that writes logloss per round to TrainingLogger."""
    def _cb(env):
        if env.iteration % log_period != 0 and env.iteration != env.end_iteration - 1:
            return
        train_metric = val_metric = None
        for item in env.evaluation_result_list:
            # item format: (dataset_name, metric_name, value, is_higher_better)
            data_name, _metric_name, value, _higher = item
            if "train" in data_name.lower():
                train_metric = value
            else:
                val_metric = value
        logger.log_lgbm_round(env.iteration, train_metric, val_metric)
    _cb.order = 10
    return _cb


class StylemetricClassifier:
    """
    LightGBM trained on pairwise stylometric + SBERT-distance features.
    """

    def __init__(self, difficulty: str):
        self.difficulty = difficulty
        scale_pos = LGBM_SCALE_POS_WEIGHT.get(difficulty, 1.0)
        self.model = LGBMClassifier(
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            colsample_bytree=LGBM_COLSAMPLE_BYTREE,
            subsample=LGBM_SUBSAMPLE,
            scale_pos_weight=scale_pos,
            n_jobs=-1,
            random_state=SEED,
            verbose=-1,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            logger=None) -> "StylemetricClassifier":
        """
        Train the LightGBM model.

        X: (N, F) feature matrix from features.build_pair_feature_matrix()
        y: (N,)  binary labels
        X_val, y_val: optional validation for early stopping
        """
        if X_val is not None and y_val is not None:
            self.model.set_params(n_estimators=1000)
            import lightgbm as lgb
            callbacks = [
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=-1),
                _make_lgbm_tqdm_callback(1000),
            ]
            if logger is not None:
                callbacks.append(_make_lgbm_log_callback(logger))
            self.model.fit(
                X, y,
                eval_set=[(X, y), (X_val, y_val)],
                eval_names=["train", "valid"],
                eval_metric="auc",   # AUC is stable with extreme class imbalance
                callbacks=callbacks,
            )
        else:
            self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of style change (P(label=1)), shape (N,)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "difficulty": self.difficulty}, f)

    @classmethod
    def load(cls, path: Path) -> "StylemetricClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model      = state["model"]
        obj.difficulty = state["difficulty"]
        obj._is_fitted = True
        return obj

    def feature_importances(self, feature_names: list = None) -> dict:
        """Return feature importance dict (for inspection / debugging)."""
        if not self._is_fitted:
            return {}
        imps = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(imps))]
        return dict(sorted(zip(names, imps), key=lambda x: -x[1]))