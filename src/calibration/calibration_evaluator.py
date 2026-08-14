"""
CalibrationEvaluator — Assess model calibration quality.

Computes:
- Brier Score: mean squared error between predicted probabilities and outcomes
- Reliability Diagram: plot of predicted vs. actual probability by bin
- Expected Calibration Error (ECE): weighted average of calibration errors
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

try:
    from sklearn.calibration import calibration_curve
    _HAS_SK_CALIBRATION = True
except ImportError:
    _HAS_SK_CALIBRATION = False


class CalibrationEvaluator:
    """Evaluate model calibration quality.

    Parameters
    ----------
    n_bins : int
        Number of bins for reliability diagram. Default: 10.
    strategy : str
        Strategy for binning: 'uniform' or 'quantile'. Default: 'uniform'.
    """

    def __init__(
        self,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self._brier_score: Optional[float] = None
        self._ece: Optional[float] = None
        self._reliability_data: Optional[pd.DataFrame] = None

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_prob_calibrated: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute calibration metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_prob : np.ndarray
            Raw predicted probabilities.
        y_prob_calibrated : np.ndarray, optional
            Calibrated probabilities. If provided, metrics are reported
            for both raw and calibrated.

        Returns
        -------
        dict
            Dictionary with keys: brier_score_raw, ece_raw,
            brier_score_calibrated, ece_calibrated (if provided).
        """
        results: Dict[str, float] = {}

        brier_raw = self._brier_score(y_true, y_prob)
        ece_raw, rel_raw = self._ece(y_true, y_prob)

        results["brier_score_raw"] = brier_raw
        results["ece_raw"] = ece_raw
        self._brier_score = brier_raw
        self._ece = ece_raw
        self._reliability_data = rel_raw

        if y_prob_calibrated is not None:
            brier_cal = self._brier_score(y_true, y_prob_calibrated)
            ece_cal, rel_cal = self._ece(y_true, y_prob_calibrated)

            results["brier_score_calibrated"] = brier_cal
            results["ece_calibrated"] = ece_cal

        return results

    @staticmethod
    def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Brier score = mean((y_prob - y_true)^2)."""
        return float(np.mean((y_prob - y_true) ** 2))

    def _ece(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[float, pd.DataFrame]:
        """Expected Calibration Error.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_prob : np.ndarray
            Predicted probabilities.

        Returns
        -------
        tuple
            (ece_value, reliability_dataframe)
        """
        if _HAS_SK_CALIBRATION:
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=self.n_bins, strategy=self.strategy
            )
        else:
            prob_true, prob_pred = self._manual_calibration_curve(y_true, y_prob)

        bin_counts = np.zeros(len(prob_pred))
        for i, p in enumerate(y_prob):
            bin_idx = min(int(p * self.n_bins), self.n_bins - 1)
            bin_counts[bin_idx] += 1

        total = len(y_true)
        ece = 0.0
        for i in range(len(prob_pred)):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / total) * abs(prob_true[i] - prob_pred[i])

        rel_df = pd.DataFrame({
            "bin_mean_predicted": prob_pred,
            "bin_fraction_positives": prob_true,
            "bin_count": bin_counts,
        })

        return float(ece), rel_df

    def _manual_calibration_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback calibration curve computation."""
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        prob_pred = []
        prob_true = []

        for i in range(self.n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

            if mask.sum() > 0:
                prob_pred.append(y_prob[mask].mean())
                prob_true.append(y_true[mask].mean())
            else:
                prob_pred.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                prob_true.append(0.0)

        return np.array(prob_true), np.array(prob_pred)

    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob_raw: np.ndarray,
        y_prob_calibrated: Optional[np.ndarray] = None,
        ax: Optional[object] = None,
    ) -> object:
        """Plot reliability diagram.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_prob_raw : np.ndarray
            Raw predicted probabilities.
        y_prob_calibrated : np.ndarray, optional
            Calibrated probabilities.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting.")

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        if _HAS_SK_CALIBRATION:
            prob_true_raw, prob_pred_raw = calibration_curve(
                y_true, y_prob_raw, n_bins=self.n_bins, strategy=self.strategy
            )
            ax.plot(prob_pred_raw, prob_true_raw, "s-", color="steelblue",
                    label=f"Raw (Brier={self._brier_score(y_true, y_prob_raw):.4f})")

            if y_prob_calibrated is not None:
                prob_true_cal, prob_pred_cal = calibration_curve(
                    y_true, y_prob_calibrated, n_bins=self.n_bins, strategy=self.strategy
                )
                ax.plot(prob_pred_cal, prob_true_cal, "o-", color="crimson",
                        label=f"Calibrated (Brier={self._brier_score(y_true, y_prob_calibrated):.4f})")
        else:
            prob_true_raw, prob_pred_raw = self._manual_calibration_curve(y_true, y_prob_raw)
            ax.plot(prob_pred_raw, prob_true_raw, "s-", color="steelblue", label="Raw")

            if y_prob_calibrated is not None:
                prob_true_cal, prob_pred_cal = self._manual_calibration_curve(y_true, y_prob_calibrated)
                ax.plot(prob_pred_cal, prob_true_cal, "o-", color="crimson", label="Calibrated")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Reliability Diagram")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        return ax

    def summary(self) -> str:
        """Return human-readable calibration summary."""
        lines = ["Calibration Evaluation Summary"]
        if self._brier_score is not None:
            lines.append(f"  Brier Score:  {self._brier_score:.6f}")
        if self._ece is not None:
            lines.append(f"  ECE:          {self._ece:.6f}")
        return "\n".join(lines)