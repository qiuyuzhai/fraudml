"""
SHAPExplainer — SHAP-based model interpretability for tree models.

Provides:
- Global feature importance via SHAP mean absolute values
- Local explanation (per-sample SHAP values) for risk review
- Summary data export for downstream analysis
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class SHAPExplainer:
    """SHAP-based model interpretability.

    Works with tree-based models (LightGBM, XGBoost, CatBoost).
    Falls back gracefully when SHAP or matplotlib is not installed.

    Parameters
    ----------
    model : object
        Trained tree-based model with ``predict_proba``.
    feature_names : list of str
        Feature names corresponding to model inputs.
    max_features : int
        Maximum number of features to include in explanations.
    n_samples : int
        Number of background/summary samples to use for SHAP.
    """

    def __init__(
        self,
        model: object,
        feature_names: List[str],
        max_features: int = 20,
        n_samples: int = 100,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.max_features = max_features
        self.n_samples = n_samples

        self._explainer: Optional[object] = None
        self._shap_values_: Optional[np.ndarray] = None
        self._global_importance_: Optional[pd.DataFrame] = None
        self._fitted: bool = False

    def fit(
        self,
        X_background: pd.DataFrame,
    ) -> "SHAPExplainer":
        """Initialize SHAP explainer with background data.

        Parameters
        ----------
        X_background : pd.DataFrame
            Background dataset (typically training data) for SHAP
            expectation values.

        Returns
        -------
        self : SHAPExplainer
        """
        if not _HAS_SHAP:
            return self

        try:
            self._explainer = shap.TreeExplainer(
                self.model,
                data=X_background,
                feature_perturbation="interventional",
                model_output="probability",
            )
            self._fitted = True
        except Exception:
            try:
                self._explainer = shap.TreeExplainer(
                    self.model,
                    data=X_background,
                )
                self._fitted = True
            except Exception as e:
                self._fitted = False

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Compute SHAP values for a dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        np.ndarray
            SHAP values (n_samples, n_features) for the positive class.
        """
        if not self._fitted or self._explainer is None:
            return np.zeros((len(X), len(self.feature_names)))

        X_sample = X.head(self.n_samples).copy() if len(X) > self.n_samples else X.copy()

        try:
            shap_output = self._explainer.shap_values(X_sample)

            if isinstance(shap_output, list):
                shap_vals = shap_output[1]
            elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 3:
                shap_vals = shap_output[:, :, 1]
            else:
                shap_vals = shap_output

            self._shap_values_ = np.array(shap_vals)
        except Exception:
            self._shap_values_ = np.zeros((len(X_sample), len(self.feature_names)))

        return self._shap_values_

    def global_importance(
        self,
        X: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute global feature importance from SHAP values.

        Parameters
        ----------
        X : pd.DataFrame, optional
            If provided, computes SHAP values first. Otherwise uses
            cached values from a prior ``transform()`` call.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, mean_abs_shap, sorted
            by importance descending.
        """
        if X is not None:
            self.transform(X)

        if self._shap_values_ is None or len(self._shap_values_) == 0:
            importance_df = pd.DataFrame({
                "feature": self.feature_names,
                "mean_abs_shap": np.zeros(len(self.feature_names)),
            }).sort_values("mean_abs_shap", ascending=False)
            self._global_importance_ = importance_df
            return importance_df

        mean_abs = np.abs(self._shap_values_).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

        self._global_importance_ = importance_df
        return importance_df

    def local_explanation(
        self,
        X: pd.DataFrame,
        idx: int = 0,
    ) -> Dict[str, float]:
        """Explain a single prediction's feature contributions.

        Parameters
        ----------
        X : pd.DataFrame
            Full dataset (used for SHAP computation).
        idx : int
            Index of the sample to explain.

        Returns
        -------
        dict
            Feature → SHAP value mapping for the explained sample.
        """
        if self._shap_values_ is None:
            self.transform(X)

        if self._shap_values_ is None or len(self._shap_values_) <= idx:
            return {}

        row_shap = self._shap_values_[idx]
        explanation = {}
        for feat, val in zip(self.feature_names, row_shap):
            explanation[feat] = float(val)

        sorted_explanation = dict(
            sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        return sorted_explanation

    def summary_plot(
        self,
        X: Optional[pd.DataFrame] = None,
        max_display: int = 20,
        ax: Optional[object] = None,
    ) -> Optional[object]:
        """Create a SHAP summary plot (beeswarm).

        Parameters
        ----------
        X : pd.DataFrame, optional
            Dataset to plot. Uses cached values if not provided.
        max_display : int
            Max features to display.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.

        Returns
        -------
        matplotlib.axes.Axes or None
        """
        if not _HAS_MATPLOTLIB:
            return None

        if X is not None:
            self.transform(X)

        if self._shap_values_ is None:
            return None

        if ax is None:
            _, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))

        top_k = min(max_display, len(self.feature_names))
        importance = np.abs(self._shap_values_).mean(axis=0)
        top_idx = np.argsort(importance)[-top_k:]
        top_features = [self.feature_names[i] for i in top_idx]

        shap_vals_top = self._shap_values_[:, top_idx]

        y_pos = np.arange(top_k)
        for i, (feat, vals) in enumerate(zip(top_features, shap_vals_top.T)):
            ax.scatter(vals, np.full_like(vals, i, dtype=float),
                       alpha=0.5, s=10, c="steelblue")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("SHAP value (impact on fraud probability)")
        ax.set_title("SHAP Feature Importance")
        ax.grid(True, alpha=0.3, axis="x")
        return ax

    def export(
        self,
        X: pd.DataFrame,
        output_dir: str = "artifacts",
    ) -> Dict[str, str]:
        """Export SHAP outputs to files.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset for SHAP computation.
        output_dir : str
            Directory to save exports.

        Returns
        -------
        dict
            Paths to exported files.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        paths = {}

        importance_df = self.global_importance(X)
        imp_path = os.path.join(output_dir, "shap_importance.csv")
        importance_df.head(self.max_features).to_csv(imp_path, index=False)
        paths["importance"] = imp_path

        if self._shap_values_ is not None:
            shap_df = pd.DataFrame(
                self._shap_values_,
                columns=self.feature_names,
            )
            shap_path = os.path.join(output_dir, "shap_values.csv")
            shap_df.to_csv(shap_path, index=False)
            paths["shap_values"] = shap_path

        return paths

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = ["SHAP Interpretability Summary"]
        lines.append(f"  SHAP installed: {_HAS_SHAP}")
        lines.append(f"  Fitted:         {self._fitted}")
        lines.append(f"  Max features:   {self.max_features}")
        lines.append(f"  N samples:      {self.n_samples}")

        if self._global_importance_ is not None:
            lines.append("  Top features by |SHAP|:")
            for _, row in self._global_importance_.head(5).iterrows():
                lines.append(f"    {row['feature']}: {row['mean_abs_shap']:.6f}")

        return "\n".join(lines)