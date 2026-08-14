"""
VIF-based feature filter .

Iteratively removes features with high Variance Inflation Factor
(VIF > 10 by default) to mitigate multicollinearity 多重共线性问题.
Implements fit/transform interface.

Uses scikit-learn LinearRegression for VIF computation, avoiding
the statsmodels dependency.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import SelectionBase


def _compute_vif(X: np.ndarray, feature_idx: int) -> float:
    """Compute VIF for a single feature using OLS regression.

    VIF_i = 1 / (1 - R²_i) where R²_i comes from regressing
    feature i on all other features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    feature_idx : int
        Index of the target feature.

    Returns
    -------
    float
        VIF value.  Returns float('inf') if R² >= 1.
    """
    n_features = X.shape[1]
    if n_features <= 1:
        return 1.0

    y = X[:, feature_idx] # 目标特征
    other_cols = [j for j in range(n_features) if j != feature_idx]
    X_rest = X[:, other_cols]

    reg = LinearRegression()
    reg.fit(X_rest, y)
    r2 = reg.score(X_rest, y) # 计算R²值：其它特征能多大程度解释当前这个特征的波动

    if r2 >= 1.0:
        return float("inf")

    return float(1.0 / (1.0 - r2))


class VIFFilter(SelectionBase):
    """Remove features with VIF above a threshold.

    Iteratively computes VIF for all features and drops the one
    with the highest VIF until all remaining features are below
    the threshold or the maximum iteration count is reached.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    threshold : float
        VIF threshold.  Features with VIF > this are removed.
        Default 10 (common rule of thumb).
    max_iterations : int
        Maximum removal iterations to prevent infinite loops.

    Attributes (set after ``fit()``):
    ----------
    removed_features_ : list[str]
        Features removed due to high VIF.
    retained_features_ : list[str]
        Features kept after VIF filtering.
    vif_history_ : list[dict]
        Each iteration's VIF values before removal.
    """

    def __init__(
        self,
        name: str = "VIFFilter",
        threshold: float = 10.0,
        max_iterations: int = 50,
    ) -> None:
        super().__init__(name=name)
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.removed_features_: List[str] = []
        self.retained_features_: List[str] = []
        self.vif_history_: List[dict] = []

    def fit(
        self,
        df: pd.DataFrame,
        iv_scores: Optional[dict] = None,
    ) -> "VIFFilter":
        """Identify features to remove based on VIF.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame with numeric feature columns.
        iv_scores : dict, optional
            Mapping of feature → IV score.  When two features are
            highly collinear (VIF > threshold), the one with the
            **lower** IV is removed first.

        Returns
        -------
        self : VIFFilter
        """
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        current_features = numeric_cols.copy()

        for iteration in range(self.max_iterations):
            if len(current_features) <= 1:
                break

            subset = df[current_features].copy()
            subset = subset.fillna(0)
            X = subset.values.astype(float)

            vif_values = {}
            try:
                for i in range(len(current_features)):
                    vif = _compute_vif(X, i)
                    vif_values[current_features[i]] = vif
            except Exception:
                break

            self.vif_history_.append(vif_values)

            self._iv_scores = iv_scores or {}

            inf_features = [k for k, v in vif_values.items() if v == float("inf")]
            if inf_features:
                inf_features.sort(
                    key=lambda f: self._iv_scores.get(f, 0), reverse=False,
                )
                worst = inf_features[0]
                self.removed_features_.append(worst)
                current_features.remove(worst)
                continue

            finite_vifs = {k: v for k, v in vif_values.items() if v != float("inf")}
            if not finite_vifs:
                break

            over_threshold = {k: v for k, v in finite_vifs.items() if v > self.threshold}
            if not over_threshold:
                break

            if self._iv_scores:
                max_col = min(
                    over_threshold,
                    key=lambda k: (self._iv_scores.get(k, 0), -over_threshold[k]),
                )
            else:
                max_col = max(over_threshold, key=over_threshold.get)

            self.removed_features_.append(max_col)
            current_features.remove(max_col)

        self.retained_features_ = current_features
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retain only features that passed VIF filtering.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with retained feature columns only.
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
        """Return VIF removal history."""
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")
        rows = []
        for i, snapshot in enumerate(self.vif_history_):
            for col, vif_val in snapshot.items():
                rows.append({
                    "iteration": i,
                    "feature": col,
                    "vif": round(vif_val, 4), # 保留4位小数
                    "removed": col in self.removed_features_ and
                    (self.removed_features_.index(col) == i if col in self.removed_features_ else False),
                })
        return pd.DataFrame(rows)

    def get_feature_metadata(self) -> dict:
        """Return metadata for drifted-feature monitoring."""
        return {
            "feature_names": self.retained_features_,
            "physical_meaning": f"VIF-filtered features (threshold={self.threshold})",
            "unit": "vif_score",
            "depends_on_target": False,
        }