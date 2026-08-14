"""
Adversarial validation for detecting train-test distribution shift.

Trains a classifier to distinguish between training and test sets.
High AUC indicates the two distributions are significantly different,
which suggests dataset shift that may degrade model performance.

idea: use a logistic regression to distinguish between train and test sets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance


class AdversarialValidation:
    """Detect distribution shift between train and test sets.

    Parameters
    ----------
    auc_threshold : float
        AUC above which distributions are considered significantly
        different.  Default 0.7.

    Attributes (set after ``evaluate()``):
    ----------
    auc_ : float
        AUC of the adversarial classifier.
    is_shifted_ : bool
        Whether the shift is significant (AUC > threshold).
    drifting_features_ : list[str]
        Features with highest permutation importance in the
        adversarial classifier — these are the drivers of drift.
    importance_df_ : pd.DataFrame
        Permutation importance for all features.
    """

    def __init__(self, auc_threshold: float = 0.7) -> None:
        self.auc_threshold = auc_threshold
        self.auc_: Optional[float] = None
        self.is_shifted_: Optional[bool] = None
        self.drifting_features_: List[str] = []
        self.importance_df_: Optional[pd.DataFrame] = None
        self._fitted: bool = False

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        top_k: int = 10,
        fast: bool = True,
    ) -> Dict[str, object]:
        """Run adversarial validation between train and test.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training DataFrame.
        test_df : pd.DataFrame
            Test / validation DataFrame.
        features : list[str], optional
            Subset of features to use.  Defaults to all common
            numeric columns.
        top_k : int
            Number of top drifting features to return.
        fast : bool
            If True, skip ``permutation_importance`` and use LR
            coefficients for ranking.  This is ~10x faster and
            recommended for the training pipeline.  Set False for
            detailed post-hoc analysis.

        Returns
        -------
        dict
            Keys: auc, is_shifted, drifting_features, importance_df.
        """
        if features is None:
            common_cols = set(train_df.columns) & set(test_df.columns)
            features = [
                c for c in common_cols
                if pd.api.types.is_numeric_dtype(train_df[c])
            ]

        train_feat = train_df[features]
        test_feat = test_df[features]

        # 降采样，避免内存问题
        max_samples = 50000
        if len(train_feat) > max_samples:
            train_feat = train_feat.sample(n=max_samples, random_state=42)
        if len(test_feat) > max_samples:
            test_feat = test_feat.sample(n=max_samples, random_state=42)

        combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
        combined = combined.fillna(0)

        labels = np.concatenate([
            np.zeros(len(train_feat)),
            np.ones(len(test_feat)),
        ])

        lr = LogisticRegression(
            max_iter=500 if fast else 2000,
            random_state=42,
            solver="saga", # 用于大样本的求解器
        )
        lr.fit(combined, labels)

        y_pred = lr.predict_proba(combined)[:, 1]
        self.auc_ = float(roc_auc_score(labels, y_pred))
        self.is_shifted_ = self.auc_ > self.auc_threshold

        if fast:
            importances = np.abs(lr.coef_[0])
            self.importance_df_ = pd.DataFrame({
                "feature": features,
                "importance_mean": importances,
                "importance_std": np.zeros(len(features)),
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        else:
            perm_imp = permutation_importance(
                lr, combined, labels,
                n_repeats=3, random_state=42, n_jobs=1,
            )
            self.importance_df_ = pd.DataFrame({
                "feature": features,
                "importance_mean": perm_imp.importances_mean,
                "importance_std": perm_imp.importances_std,
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        self.drifting_features_ = self.importance_df_.head(top_k)["feature"].tolist()

        self._fitted = True

        return {
            "auc": self.auc_,
            "is_shifted": self.is_shifted_,
            "drifting_features": self.drifting_features_,
            "importance_df": self.importance_df_,
        }

    def get_dropped_features(self, max_features: int = 5) -> List[str]:
        """Get top drifting features recommended for dropping.

        Parameters
        ----------
        max_features : int
            Maximum number of features to drop.

        Returns
        -------
        list[str]
            Features to consider dropping from the pipeline.
        """
        if not self._fitted:
            raise RuntimeError("Call evaluate() first.")
        return self.drifting_features_[:max_features]

    def summary(self) -> str:
        """Return a human-readable summary of adversarial validation."""
        if not self._fitted:
            return "AdversarialValidation: not evaluated."
        lines = [
            f"Adversarial Validation Results",
            f"  AUC:              {self.auc_:.4f}",
            f"  Threshold:        {self.auc_threshold}",
            f"  Distribution shift: {'YES' if self.is_shifted_ else 'NO'}",
            f"  Top drifting features: {self.drifting_features_}",
        ]
        return "\n".join(lines)