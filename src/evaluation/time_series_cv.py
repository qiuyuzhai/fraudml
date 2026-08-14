"""
Purged Time Series Cross-Validation splitter.

Standard K-Fold is invalid for time-series because future data
leaks into training.  PurgedTimeSeriesSplit uses an expanding
window with a purge gap to prevent leakage.
"""

from __future__ import annotations

from typing import Generator, List, Tuple

import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """Expanding window cross-validation with a purge gap.

    Assumes the input DataFrame is sorted by time (ascending).
    Each fold trains on an expanding set of past data and validates
    on the next chunk, with a configurable purge gap between
    train and validation to prevent information leakage.

    Parameters
    ----------
    n_splits : int
        Number of CV splits (default 5).
    purge_gap : int
        Number of rows to skip between training and validation
        sets to prevent leakage (default 0).
    min_train_size : int
        Minimum number of training rows in the first fold.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        (train_indices, val_indices) arrays of positional indices.

    Examples
    --------
    >>> splitter = PurgedTimeSeriesSplit(n_splits=3, purge_gap=100)
    >>> for train_idx, val_idx in splitter.split(df):
    ...     X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
    ...     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        min_train_size: int = 0,
    ) -> None:
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.min_train_size = min_train_size

    def split(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/val index splits.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (must be sorted by time).
        y : pd.Series, optional
            Target variable (unused, present for API compatibility).

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            (train_indices, val_indices).
        """
        n = len(df)
        total_folds = self.n_splits + 1

        fold_size = n // total_folds
        fold_boundaries = [i * fold_size for i in range(total_folds + 1)]
        fold_boundaries[-1] = n

        for i in range(1, self.n_splits + 1):
            train_end = fold_boundaries[i]
            val_start = fold_boundaries[i] + self.purge_gap
            val_end = fold_boundaries[i + 1]

            if val_start >= val_end:
                continue

            if train_end < self.min_train_size:
                continue

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            if len(val_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_n_splits(self, df: pd.DataFrame) -> int:
        """Return the actual number of valid splits for this dataset."""
        count = 0
        for _ in self.split(df):
            count += 1
        return count