"""
PSI-based feature monitor.

Compares feature distributions between training and current data.
Flags features with significant drift for retraining.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..scoring.psi import compute_psi_batch
from ..selection.base import SelectionBase


class PSIMonitor(SelectionBase):
    """Monitor feature drift using Population Stability Index.

    Unlike IVSelector / VIFFilter, PSIMonitor does **not** filter
    features — it only *flags* drifted ones.  :meth:`transform`
    passes the DataFrame through unchanged.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    threshold : float
        PSI threshold above which drift is flagged.
        Default 0.25 (moderate drift).
    n_bins : int
        Number of bins for PSI calculation.

    Attributes (set after ``fit()``):
    psi_scores_ : pd.DataFrame
        PSI values per feature comparing train vs. current.
    drifted_features_ : list[str]
        Features flagged for drift.
    """

    def __init__(
        self,
        name: str = "PSIMonitor",
        threshold: float = 0.25,
        n_bins: int = 10,
    ) -> None:
        super().__init__(name=name)
        self.threshold = threshold
        self.n_bins = n_bins
        self.psi_scores_: Optional[pd.DataFrame] = None
        self.drifted_features_: List[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> "PSIMonitor":
        """Compute PSI between training and current distributions.

        Parameters
        ----------
        train_df : pd.DataFrame
            Reference (training) DataFrame.
        current_df : pd.DataFrame
            Current DataFrame to compare.
        features : list[str], optional
            Subset of features to monitor.  Defaults to all numeric
            columns present in both DataFrames.

        Returns
        self : PSIMonitor
        """
        if features is None:
            common_cols = set(train_df.columns) & set(current_df.columns)
            features = [
                c for c in common_cols
                if pd.api.types.is_numeric_dtype(train_df[c])
            ]

        self.psi_scores_ = compute_psi_batch(
            train_df, current_df, features,
            n_bins=self.n_bins, threshold=self.threshold,
        )
        self.drifted_features_ = self.psi_scores_[
            self.psi_scores_["drifted"]
        ]["feature"].tolist()
        self.retained_features_ = self.psi_scores_["feature"].tolist()

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """PSIMonitor does not filter features.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (returned unchanged).

        Returns
        pd.DataFrame
            The same DataFrame (no filtering applied).
        """
        return df

    def get_drift_report(self) -> pd.DataFrame:
        """Return PSI scores for all monitored features."""
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")
        return self.psi_scores_.copy()

    def summary(self) -> str:
        """Return a human-readable drift summary."""
        if not self._fitted:
            return f"{self.name}: not fitted."
        total = len(self.psi_scores_)
        drifted = len(self.drifted_features_)
        lines = [
            f"PSI Monitor Summary",
            f"  Total features checked: {total}",
            f"  Drifted features:       {drifted}",
        ]
        if self.drifted_features_:
            lines.append(f"  Drifted feature list:   {self.drifted_features_}")
        return "\n".join(lines)

    def get_feature_metadata(self) -> dict:
        """Return metadata for drifted-feature monitoring."""
        feature_names = (
            self.psi_scores_["feature"].tolist()
            if self._fitted and self.psi_scores_ is not None
            else []
        )
        return {
            "feature_names": feature_names,
            "physical_meaning": f"PSI-monitored features (threshold={self.threshold})",
            "unit": "psi_score",
            "depends_on_target": False,
        }