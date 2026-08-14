"""
IsotonicCalibrator — Isotonic regression calibration.

Non-parametric calibration that learns a monotonic mapping from
raw scores to calibrated probabilities. More flexible than Platt
scaling but requires more data. 容易过拟合,所以数据量一定要够
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .calibrator import Calibrator


class IsotonicCalibrator(Calibrator):
    """Isotonic regression calibration.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    out_of_bounds : str
        How to handle out-of-bounds predictions. One of 'clip', 'NaN',
        or 'raise'. Default: 'clip'.
    y_min : float or None
        Minimum value for isotonic regression output. Default: 0.
    y_max : float or None
        Maximum value for isotonic regression output. Default: 1.

    Attributes (set after ``fit()``):
    ----------
    iso_model_ : IsotonicRegression
        Fitted isotonic regression model.
    """

    def __init__(
        self,
        name: str = "IsotonicCalibrator",
        out_of_bounds: str = "clip",
        y_min: float = 0.0,
        y_max: float = 1.0,
    ) -> None:
        super().__init__(name=name)
        self.out_of_bounds = out_of_bounds
        self.y_min = y_min
        self.y_max = y_max
        self.iso_model_: Optional[IsotonicRegression] = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """Learn isotonic regression mapping.

        Parameters
        ----------
        y_prob : np.ndarray
            Raw predicted probabilities.
        y_true : np.ndarray
            Ground truth binary labels.

        Returns
        -------
        self : IsotonicCalibrator
        """
        self.iso_model_ = IsotonicRegression(
            out_of_bounds=self.out_of_bounds,
            y_min=self.y_min,
            y_max=self.y_max,
        )
        self.iso_model_.fit(y_prob, y_true)

        self._fitted = True
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities.

        Parameters
        ----------
        y_prob : np.ndarray
            Raw predicted probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probabilities in [0, 1].
        """
        if not self._fitted or self.iso_model_ is None:
            raise RuntimeError(f"{self.name}: not fitted.")

        calibrated = self.iso_model_.predict(y_prob)
        return np.clip(calibrated, 0.0, 1.0)

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "fitted": self._fitted,
            "out_of_bounds": self.out_of_bounds,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }