"""
IV-based feature selector.

Filters features by Information Value threshold (default >= 0.02).
Implements fit/transform interface for pipeline integration.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ..scoring.iv import compute_iv_batch
from .base import SelectionBase


class IVSelector(SelectionBase):
    """Select features whose Information Value exceeds a threshold.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    target_col : str
        Binary target column name (e.g. 'isFraud').
    threshold : float
        Minimum IV to retain a feature.  Default 0.02 (weak predictor).
    n_bins : int
        Number of bins for WOE-based IV calculation.

    Attributes (set after ``fit()``):
    ----------
    iv_scores_ : pd.DataFrame
        IV values per feature.
    retained_features_ : list[str]
        Features retained after filtering.
    """

    def __init__(
        self,
        name: str = "IVSelector",
        target_col: str = "isFraud",
        threshold: float = 0.02,
        n_bins: int = 10,
    ) -> None:
        super().__init__(name=name)
        self._target_col = target_col
        self.threshold = threshold
        self.n_bins = n_bins
        self.iv_scores_: Optional[pd.DataFrame] = None
        self.retained_features_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "IVSelector":
        """Compute IV scores and select features above threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame with feature columns and target column.

        Returns
        -------
        self : IVSelector
        """
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self._target_col]

        self.iv_scores_ = compute_iv_batch(
            df, numeric_cols, self._target_col, n_bins=self.n_bins,
        )
        self.retained_features_ = self.iv_scores_[
            self.iv_scores_["iv"] >= self.threshold
        ]["feature"].tolist()

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retain only selected features.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with only the selected feature columns.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted. Call fit() first.")
        available = [c for c in self.retained_features_ if c in df.columns]
        return df[available].copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit then transform in one call."""
        self.fit(df)
        return self.transform(df)

    def summary(self) -> pd.DataFrame:
        """Return IV scores for all evaluated features."""
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")
        return self.iv_scores_.copy()

    def get_feature_metadata(self) -> dict:
        """Return metadata for drifted-feature monitoring."""
        return {
            "feature_names": self.retained_features_,
            "physical_meaning": f"IV-selected features (threshold={self.threshold})",
            "unit": "iv_score",
            "depends_on_target": True,
        }