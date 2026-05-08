"""
splitting.py - Stratified 5-fold split with an inner train/val carve-out.

Per fold: idx_test is the held-out fold; the remaining samples are split
80/20 into train/val (stratified). evaluate.run_evaluation() averages
metrics across folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


N_SPLITS = 5
VAL_FRAC = 0.2
SEED = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,  # kept for signature compatibility
    val_size: float = 0.15,   # kept for signature compatibility
    random_state: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return 5 stratified (train, val, test) folds."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    idx = np.arange(len(y))

    folds: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for trainval_idx, test_idx in skf.split(idx, y):
        idx_train, idx_val = train_test_split(
            trainval_idx,
            test_size=VAL_FRAC,
            random_state=random_state,
            stratify=y[trainval_idx],
        )
        folds.append((idx_train, idx_val, test_idx))
    return folds
