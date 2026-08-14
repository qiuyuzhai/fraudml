"""
Drift detector for monitoring feature stability over time.

Combines PSI-based distribution drift detection with the
adversarial validation approach for comprehensive monitoring.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..scoring.psi import compute_psi_batch
from .adversarial import AdversarialValidation


class DriftDetector:
    """Comprehensive drift detection combining PSI and adversarial validation.

    Parameters
    ----------
    psi_threshold : float
        PSI threshold for flagging distribution drift.
    auc_threshold : float
        AUC threshold for adversarial validation shift detection.
    n_bins : int
        Number of bins for PSI calculation.

    Attributes (set after ``detect()``):
    ----------
    psi_report_ : pd.DataFrame
        PSI scores per feature.
    adversarial_report_ : dict
        Adversarial validation results.
    drifted_features_ : list[str]
        Features flagged for drift by either method.
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        auc_threshold: float = 0.7,
        n_bins: int = 10,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.auc_threshold = auc_threshold
        self.n_bins = n_bins
        self.psi_report_: Optional[pd.DataFrame] = None
        self.adversarial_report_: Optional[dict] = None
        self.drifted_features_: List[str] = []
        self._fitted: bool = False

    def detect(
        self,
        train_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """Run comprehensive drift detection.

        Parameters
        ----------
        train_df : pd.DataFrame
            Reference (training) DataFrame.
        current_df : pd.DataFrame
            Current DataFrame to compare.
        features : list[str], optional
            Subset of features to monitor.

        Returns
        -------
        dict
            Combined drift report.
        """
        if features is None:
            common_cols = set(train_df.columns) & set(current_df.columns)
            features = [
                c for c in common_cols
                if pd.api.types.is_numeric_dtype(train_df[c])
            ]

        self.psi_report_ = compute_psi_batch(
            train_df, current_df, features,
            n_bins=self.n_bins, threshold=self.psi_threshold,
        )
        psi_drifted = self.psi_report_[self.psi_report_["drifted"]]["feature"].tolist()

        adv = AdversarialValidation(auc_threshold=self.auc_threshold)
        self.adversarial_report_ = adv.evaluate(
            train_df[features].fillna(0),
            current_df[features].fillna(0),
            features=features,
            fast=False,
        )
        adv_drifted = adv.drifting_features_

        all_drifted = list(set(psi_drifted + adv_drifted))
        self.drifted_features_ = all_drifted

        self._fitted = True

        return {
            "psi_report": self.psi_report_,
            "adversarial_report": self.adversarial_report_,
            "psi_drifted": psi_drifted,
            "adversarial_drifted": adv_drifted,
            "all_drifted": all_drifted,
            "n_features": len(features),
            "n_drifted": len(all_drifted),
        }

    def summary(self) -> str:
        """Return a human-readable drift summary."""
        if not self._fitted:
            return "DriftDetector: not run. Call detect() first."

        n_total = len(self.psi_report_) if self.psi_report_ is not None else 0
        n_drifted = len(self.drifted_features_)
        auc = self.adversarial_report_.get("auc", "N/A") if self.adversarial_report_ else "N/A"

        lines = [
            f"Drift Detection Report",
            f"  Total features:     {n_total}",
            f"  PSI threshold:      {self.psi_threshold}",
            f"  AUC threshold:       {self.auc_threshold}",
            f"  Adversarial AUC:    {auc}",
            f"  Features drifted:   {n_drifted}",
        ]
        if self.drifted_features_:
            lines.append(f"  Drifted features:   {self.drifted_features_}")
        return "\n".join(lines)