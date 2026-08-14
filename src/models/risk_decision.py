"""
RiskDecisionEngine — Multi-level risk decision engine for fraud detection.

Replaces simple threshold classification with a three-tier risk system:
    - LOW:    Auto-pass (legitimate transaction)
    - MEDIUM: Manual review required
    - HIGH:   Auto-block (fraudulent transaction)

Cost formula:
    ExpectedCost = P(fraud) * cost_fn + P(legit) * cost_fp # 如果放过该样本,成本的公式
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class RiskDecisionEngine:
    """Multi-level risk decision engine.

    Parameters
    ----------
    cost_fp : float
        Cost of a false positive (manual review / customer complaint).
    cost_fn : float
        Cost of a false negative (fraud loss per missed case).
    medium_threshold : float # 初始中风险阈值
        Threshold below which transactions are classified as LOW risk.
    high_threshold : float # 初始高风险阈值
        Threshold above which transactions are classified as HIGH risk.
        Values between medium and high are classified as MEDIUM risk.
    min_threshold : float # 搜索下限
        Minimum threshold to evaluate during optimization.
    max_threshold : float # 搜索上限
        Maximum threshold to evaluate during optimization.
    n_steps : int # 搜索步数
        Number of threshold steps in the grid search.

    Attributes (set after ``fit()``):
    ----------
    optimal_medium_threshold_ : float
        Optimal medium threshold.
    optimal_high_threshold_ : float
        Optimal high threshold.
    total_cost_ : float
        Total cost at optimal thresholds.
    cost_curve_ : pd.DataFrame
        Cost vs. threshold combinations.
    """

    RISK_LOW = "low"
    RISK_MEDIUM = "medium"
    RISK_HIGH = "high"

    def __init__(
        self,
        cost_fp: float = 10.0,
        cost_fn: float = 500.0,
        medium_threshold: float = 0.3,
        high_threshold: float = 0.7,
        min_threshold: float = 0.01,
        max_threshold: float = 0.99,
        n_steps: int = 50,
    ) -> None:
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.n_steps = n_steps

        self.optimal_medium_threshold_: Optional[float] = None
        self.optimal_high_threshold_: Optional[float] = None
        self.total_cost_: Optional[float] = None
        self.cost_curve_: Optional[pd.DataFrame] = None
        self._fitted: bool = False

    def fit(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> "RiskDecisionEngine":
        """Learn optimal thresholds from validation data.
        
        用历史数据当沙盘，把所有可能的双阈值组合都演练一遍，谁算出来的总成本最低，谁就是最终的最优决策线


        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels (0/1).
        y_prob : np.ndarray
            Predicted probabilities for the positive class.

        Returns
        -------
        self : RiskDecisionEngine
        """
        thresholds = np.linspace(self.min_threshold, self.max_threshold, self.n_steps)

        best_cost = float("inf")
        best_low_th = self.medium_threshold
        best_high_th = self.high_threshold
        records: List[Dict[str, float]] = []

        for low_th in thresholds:
            for high_th in thresholds:
                if low_th >= high_th:
                    continue

                risk_levels = self._classify(y_prob, low_th, high_th)

                low_mask = risk_levels == self.RISK_LOW
                medium_mask = risk_levels == self.RISK_MEDIUM
                high_mask = risk_levels == self.RISK_HIGH

                fn_count = int(np.sum((low_mask | medium_mask) & (y_true == 1)))
                fp_count = int(np.sum(high_mask & (y_true == 0)))

                total_cost = fn_count * self.cost_fn + fp_count * self.cost_fp

                records.append({
                    "low_threshold": float(low_th),
                    "high_threshold": float(high_th),
                    "cost": float(total_cost),
                    "fn": fn_count,
                    "fp": fp_count,
                })

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_low_th = low_th
                    best_high_th = high_th

        self.optimal_medium_threshold_ = float(best_low_th)
        self.optimal_high_threshold_ = float(best_high_th)
        self.total_cost_ = float(best_cost)
        self.cost_curve_ = pd.DataFrame(records)
        self._fitted = True

        self.medium_threshold = best_low_th
        self.high_threshold = best_high_th

        return self

    def predict(
        self, y_prob: np.ndarray, thresholds: Optional[Tuple[float, float]] = None
    ) -> Dict[str, np.ndarray]:
        """Classify transactions into risk levels.

        Parameters
        ----------
        y_prob : np.ndarray
            Predicted probabilities for the positive class.
        thresholds : tuple of (float, float), optional
            (medium_threshold, high_threshold). If None, learned thresholds
            are used.

        Returns
        -------
        dict
            - risk_levels: array of 'low', 'medium', 'high'
            - recommended_actions: array of 'auto_pass', 'manual_review', 'auto_block'
            - confidence: array of confidence scores
        """
        if thresholds is None:
            if not self._fitted:
                raise RuntimeError("RiskDecisionEngine not fitted. Call fit() first.")
            low_th = self.optimal_medium_threshold_
            high_th = self.optimal_high_threshold_
        else:
            low_th, high_th = thresholds

        risk_levels = self._classify(y_prob, low_th, high_th)

        actions = np.where(
            risk_levels == self.RISK_LOW, "auto_pass",
            np.where(
                risk_levels == self.RISK_HIGH, "auto_block",
                "manual_review"
            )
        )

        confidence = np.where(
            risk_levels == self.RISK_LOW,
            1.0 - y_prob,
            np.where(
                risk_levels == self.RISK_HIGH,
                y_prob,
                1.0 - np.abs(y_prob - 0.5) * 2
            )
        )

        return {
            "risk_levels": risk_levels,
            "recommended_actions": actions,
            "confidence": confidence,
        }

    def _classify(
        self, y_prob: np.ndarray, low_th: float, high_th: float
    ) -> np.ndarray:
        """Classify probabilities into risk levels.

        Parameters
        ----------
        y_prob : np.ndarray
            Predicted probabilities.
        low_th : float
            Low threshold (below this → LOW risk).
        high_th : float
            High threshold (above this → HIGH risk).

        Returns
        -------
        np.ndarray
            Array of risk level strings.
        """
        levels = np.where(
            y_prob < low_th, self.RISK_LOW,
            np.where(y_prob > high_th, self.RISK_HIGH, self.RISK_MEDIUM)
        )
        return levels

    def get_thresholds(self) -> Tuple[float, float]:
        """Return (medium_threshold, high_threshold)."""
        return self.medium_threshold, self.high_threshold

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Risk Decision Engine Summary",
            f"  Medium threshold (LOW <):  {self.medium_threshold:.4f}",
            f"  High threshold (HIGH >):   {self.high_threshold:.4f}",
            f"  Cost FP:                   {self.cost_fp:.2f}",
            f"  Cost FN:                   {self.cost_fn:.2f}",
        ]
        if self.total_cost_ is not None:
            lines.append(f"  Min total cost:            {self.total_cost_:.2f}")
        return "\n".join(lines)

    def evaluate(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the risk decision engine on validation data.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_prob : np.ndarray
            Predicted probabilities.

        Returns
        -------
        dict
            Performance metrics.
        """
        results = self.predict(y_prob)
        risk_levels = results["risk_levels"]

        low_mask = risk_levels == self.RISK_LOW
        medium_mask = risk_levels == self.RISK_MEDIUM
        high_mask = risk_levels == self.RISK_HIGH

        n_low = int(low_mask.sum())
        n_medium = int(medium_mask.sum())
        n_high = int(high_mask.sum())

        fn = int(np.sum((low_mask | medium_mask) & (y_true == 1)))
        fp = int(np.sum(high_mask & (y_true == 0)))
        tp = int(np.sum(high_mask & (y_true == 1)))
        tn = int(np.sum((low_mask | medium_mask) & (y_true == 0)))

        total_cost = fn * self.cost_fn + fp * self.cost_fp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        return {
            "n_low": n_low,
            "n_medium": n_medium,
            "n_high": n_high,
            "fn": fn,
            "fp": fp,
            "tp": tp,
            "tn": tn,
            "total_cost": float(total_cost),
            "precision": float(precision),
            "recall": float(recall),
        }