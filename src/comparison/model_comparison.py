"""
ModelComparator — Multi-model comparison framework for fraud detection.

Trains and evaluates multiple model architectures on the same data,
producing a unified comparison report across key metrics.

Supported models:
- lr: Logistic Regression (baseline)
- lightgbm: LightGBM tree-based model
- xgboost: XGBoost tree-based model (if available)
- random_forest: Random Forest (if available)
- catboost: CatBoost gradient boosting (if available)

Supported metrics:
- auc: ROC-AUC
- ks: Kolmogorov-Smirnov statistic
- brier: Brier score
- logloss: Logarithmic loss
- pr_auc: Precision-Recall AUC
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from sklearn.ensemble import RandomForestClassifier
    _HAS_RF = True
except ImportError:
    _HAS_RF = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


class ModelComparator:
    """Multi-model comparison framework.

    Parameters
    ----------
    model_types : list of str
        Model types to compare. Options: 'lr', 'lightgbm', 'xgboost',
        'random_forest', 'catboost'.
    metrics : list of str
        Metrics to compute. Options: 'auc', 'ks', 'brier', 'logloss',
        'pr_auc'.
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model_types: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        random_seed: int = 42,
    ) -> None:
        self.model_types = model_types or ["lr", "lightgbm"]
        self.metrics = metrics or ["auc", "ks", "brier", "logloss"]
        self.random_seed = random_seed

        self.models_: Dict[str, object] = {}
        self.results_: Optional[pd.DataFrame] = None
        self._fitted: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> "ModelComparator":
        """Train all models and evaluate on validation data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : np.ndarray
            Training labels.
        X_val : pd.DataFrame
            Validation features.
        y_val : np.ndarray
            Validation labels.

        Returns
        -------
        self : ModelComparator
        """
        records: List[Dict[str, float]] = []

        pos_weight = self._compute_pos_weight(y_train)

        for model_type in self.model_types:
            model = self._create_model(model_type, pos_weight)
            if model is None:
                continue

            try:
                model.fit(X_train, y_train)
                self.models_[model_type] = model

                y_prob = self._predict_proba(model, X_val)
                metrics = self._compute_metrics(y_val, y_prob)

                record = {"model": model_type}
                record.update(metrics)
                records.append(record)

            except Exception:
                pass

        if records:
            self.results_ = pd.DataFrame(records).sort_values(
                "auc", ascending=False
            )
        else:
            self.results_ = pd.DataFrame(columns=["model"] + self.metrics)

        self._fitted = True
        return self

    def _create_model(
        self, model_type: str, pos_weight: float
    ) -> Optional[object]:
        """Create a model instance by type."""
        if model_type == "lr":
            return Pipeline([
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=self.random_seed,
                        n_jobs=-1,
                    ),
                ),
            ])

        elif model_type == "lightgbm":
            if not _HAS_LIGHTGBM:
                return None
            return lgb.LGBMClassifier(
                is_unbalance=False,
                scale_pos_weight=pos_weight,
                random_state=self.random_seed,
                verbosity=-1,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
            )

        elif model_type == "xgboost":
            if not _HAS_XGBOOST:
                return None
            return xgb.XGBClassifier(
                scale_pos_weight=pos_weight,
                random_state=self.random_seed,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )

        elif model_type == "random_forest":
            if not _HAS_RF:
                return None
            return RandomForestClassifier(
                class_weight="balanced",
                random_state=self.random_seed,
                n_estimators=200,
                max_depth=10,
                n_jobs=-1,
            )

        elif model_type == "catboost":
            if not _HAS_CATBOOST:
                return None
            return CatBoostClassifier(
                auto_class_weights="Balanced",
                random_seed=self.random_seed,
                iterations=300,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                subsample=0.8,
                colsample_bylevel=0.8,
                verbose=0,
                allow_writing_files=False,
            )

        return None

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> float:
        neg = float((y == 0).sum())
        pos = float((y == 1).sum())
        return neg / max(pos, 1.0)

    @staticmethod
    def _predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
        """Extract positive-class probabilities."""
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    def _compute_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute selected evaluation metrics."""
        result: Dict[str, float] = {}

        if "auc" in self.metrics:
            result["auc"] = float(roc_auc_score(y_true, y_prob))

        if "ks" in self.metrics:
            result["ks"] = float(self._ks_score(y_true, y_prob))

        if "brier" in self.metrics:
            result["brier"] = float(brier_score_loss(y_true, y_prob))

        if "logloss" in self.metrics:
            result["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))

        if "pr_auc" in self.metrics:
            result["pr_auc"] = float(average_precision_score(y_true, y_prob))

        return result

    @staticmethod
    def _ks_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Kolmogorov-Smirnov statistic."""
        sorted_idx = np.argsort(y_prob)
        sorted_y = y_true[sorted_idx]
        n_pos = max((y_true == 1).sum(), 1)
        n_neg = max((y_true == 0).sum(), 1)
        cum_pos = np.cumsum(sorted_y == 1) / n_pos
        cum_neg = np.cumsum(sorted_y == 0) / n_neg
        return float(np.max(np.abs(cum_pos - cum_neg)))

    def get_results(self) -> pd.DataFrame:
        """Return comparison results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Model comparison results sorted by AUC descending.
        """
        if self.results_ is None:
            return pd.DataFrame()
        return self.results_.copy()

    def get_best_model(self, metric: str = "auc") -> Optional[object]:
        """Return the best model by a given metric.

        Parameters
        ----------
        metric : str
            Metric to use for selection. Default: 'auc'.

        Returns
        -------
        object or None
            Best model instance, or None if no models trained.
        """
        if self.results_ is None or len(self.results_) == 0:
            return None

        if metric not in self.results_.columns:
            metric = "auc"

        if metric in ("brier", "logloss"):
            best_idx = self.results_[metric].idxmin()
        else:
            best_idx = self.results_[metric].idxmax()

        best_model_name = self.results_.loc[best_idx, "model"]
        return self.models_.get(best_model_name)

    def export_results(
        self, output_path: str = "model_comparison.csv"
    ) -> str:
        """Export comparison results to CSV.

        Parameters
        ----------
        output_path : str
            Path to save CSV.

        Returns
        -------
        str
            Path to saved file.
        """
        if self.results_ is not None:
            self.results_.to_csv(output_path, index=False)
        return output_path

    def summary(self) -> str:
        """Return human-readable comparison summary."""
        lines = ["Model Comparison Summary"]
        lines.append(f"  Models compared: {len(self.models_)}")
        lines.append(f"  Metrics: {self.metrics}")

        if self.results_ is not None and len(self.results_) > 0:
            lines.append("\n  Results:")
            for _, row in self.results_.iterrows():
                model_name = row["model"]
                metric_parts = []
                for m in self.metrics:
                    if m in row.index:
                        metric_parts.append(f"{m}={row[m]:.4f}")
                lines.append(f"    {model_name}: {', '.join(metric_parts)}")

            best = self.get_best_model("auc")
            if best is not None:
                best_name = self.results_.loc[self.results_["auc"].idxmax(), "model"]
                lines.append(f"\n  Best model (AUC): {best_name}")

        return "\n".join(lines)