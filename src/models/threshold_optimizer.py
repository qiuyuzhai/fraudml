"""
Threshold optimizer for fraud detection.

Moves beyond accuracy-based evaluation to business cost optimization.
Finds the classification threshold that minimizes total expected cost
(TotalCost = FP_{count} * times cost_{fp} + FN_{count} * times cost_{fn}).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class ThresholdOptimizer:
    """Optimize classification threshold based on business cost.

    Parameters
    ----------
    cost_fp : float
        Cost of a false positive (e.g. manual review cost).
    cost_fn : float
        Cost of a false negative (e.g. fraud loss per missed case).
    min_threshold : float
        Minimum threshold to evaluate.
    max_threshold : float
        Maximum threshold to evaluate.
    n_steps : int
        Number of threshold steps in the grid search.

    Attributes (set after ``optimize()``):
    ----------
    best_threshold_ : float
        Optimal threshold that minimizes total cost.
    best_cost_ : float
        Total cost at the optimal threshold.
    best_precision_ : float
        Precision at the optimal threshold.
    best_recall_ : float
        Recall at the optimal threshold.
    best_ks_ : float
        KS statistic at the optimal threshold.
    cost_curve_ : pd.DataFrame
        Cost vs. threshold for all evaluated thresholds.
    """

    def __init__(
        self,
        cost_fp: float = 10.0,
        cost_fn: float = 500.0,
        min_threshold: float = 0.01,
        max_threshold: float = 0.99,
        n_steps: int = 99,
    ) -> None:
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.n_steps = n_steps
        self.best_threshold_: Optional[float] = None
        self.best_cost_: Optional[float] = None
        self.best_precision_: Optional[float] = None
        self.best_recall_: Optional[float] = None
        self.best_ks_: Optional[float] = None
        self.cost_curve_: Optional[pd.DataFrame] = None

    @staticmethod
    # TPR和FPR的差值最大值：ROC 曲线上离对角线y=x垂直距离最大的那个点
    def _compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """KS statistic — max |F_pos(prob) - F_neg(prob)|."""
        sorted_idx = np.argsort(y_prob)
        sorted_y = y_true[sorted_idx]
        n_pos = max((y_true == 1).sum(), 1)
        n_neg = max((y_true == 0).sum(), 1)
        cum_pos = np.cumsum(sorted_y == 1) / n_pos
        cum_neg = np.cumsum(sorted_y == 0) / n_neg
        return float(np.max(np.abs(cum_pos - cum_neg)))

    def optimize(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Find the threshold that minimizes total business cost.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels (0/1).
        y_pred_proba : np.ndarray
            Predicted probabilities for the positive class: 坏样本.

        Returns
        -------
        dict
            Dictionary with keys: best_threshold, best_cost,
            best_precision, best_recall, best_ks.
        """
        thresholds = np.linspace(self.min_threshold, self.max_threshold, self.n_steps)

        records = []
        best_cost = float("inf")
        best_threshold = self.min_threshold
        best_prec = 0.0
        best_rec = 0.0

        for t in thresholds:
            y_pred = (y_pred_proba >= t).astype(int)

            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            total_cost = fp * self.cost_fp + fn * self.cost_fn
            records.append({
                "threshold": round(t, 4),
                "cost": float(total_cost),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            })

            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = t
                if tp + fp > 0:
                    best_prec = float(tp / (tp + fp))
                else:
                    best_prec = 0.0
                if tp + fn > 0:
                    best_rec = float(tp / (tp + fn))
                else:
                    best_rec = 0.0

        best_ks = self._compute_ks(y_true, y_pred_proba)

        self.best_threshold_ = float(best_threshold)
        self.best_cost_ = float(best_cost)
        self.best_precision_ = best_prec
        self.best_recall_ = best_rec
        self.best_ks_ = best_ks
        self.cost_curve_ = pd.DataFrame(records)

        return {
            "best_threshold": self.best_threshold_,
            "best_cost": self.best_cost_,
            "best_precision": self.best_precision_,
            "best_recall": self.best_recall_,
            "best_ks": self.best_ks_,
        }

    def plot_cost_curve(self, ax: Optional[object] = None) -> object:
        """Plot total cost vs. threshold.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  Creates a new figure if not provided.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting.")
        if self.cost_curve_ is None:
            raise RuntimeError("Call optimize() first.")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            self.cost_curve_["threshold"],
            self.cost_curve_["cost"],
            linewidth=2,
            color="steelblue",
        )
        ax.axvline(
            self.best_threshold_,
            color="crimson",
            linestyle="--",
            label=f'Best threshold = {self.best_threshold_:.3f}',
        )
        ax.scatter(
            [self.best_threshold_],
            [self.best_cost_],
            color="crimson",
            zorder=5,
            s=80,
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Total Cost (FP * {:.0f} + FN * {:.0f})".format(
            self.cost_fp, self.cost_fn))
        ax.set_title("Cost Curve — Threshold Optimization")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def summary(self) -> str:
        """Return a human-readable summary of optimization results."""
        if self.best_threshold_ is None:
            return "ThresholdOptimizer: not fitted. Call optimize() first."
        return (
            f"Threshold Optimization Results\n"
            f"  Best threshold:  {self.best_threshold_:.4f}\n"
            f"  Minimum cost:    {self.best_cost_:.2f}\n"
            f"  Precision:        {self.best_precision_:.4f}\n"
            f"  Recall:           {self.best_recall_:.4f}\n"
            f"  KS:               {self.best_ks_:.4f}"
        )