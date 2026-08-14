"""
PlattScalingCalibrator — Logistic regression calibration (Platt scaling).

Maps raw model scores to calibrated probabilities via logistic regression:
    P(calibrated) = 1 / (1 + exp(A * score + B))

where A and B are learned parameters.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .calibrator import Calibrator


class PlattScalingCalibrator(Calibrator):
    """Platt scaling calibration via logistic regression.

    Parameters
    ----------
    name : str
        Human-readable identifier.

    Attributes (set after ``fit()``):
    ----------
    lr_model_ : LogisticRegression
        Fitted logistic regression model.
    coef_ : float
        Learned slope (A).
    intercept_ : float
        Learned intercept (B).
    """

    def __init__(self, name: str = "PlattScalingCalibrator") -> None:
        super().__init__(name=name)
        self.lr_model_: Optional[LogisticRegression] = None
        self.coef_: Optional[float] = None
        self.intercept_: Optional[float] = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattScalingCalibrator":
        """Learn Platt scaling parameters via logistic regression.

        Parameters
        ----------
        y_prob : np.ndarray
            Raw predicted probabilities.
        y_true : np.ndarray
            Ground truth binary labels.

        Returns
        -------
        self : PlattScalingCalibrator
        """
        X = y_prob.reshape(-1, 1)
        y = y_true.astype(int)

        self.lr_model_ = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model_.fit(X, y)

        self.coef_ = float(self.lr_model_.coef_[0][0])
        self.intercept_ = float(self.lr_model_.intercept_[0])

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
        if not self._fitted or self.lr_model_ is None:
            raise RuntimeError(f"{self.name}: not fitted.")

        X = y_prob.reshape(-1, 1)
        calibrated = self.lr_model_.predict_proba(X)[:, 1]
        return np.clip(calibrated, 0.0, 1.0)

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "fitted": self._fitted,
            "coef": self.coef_,
            "intercept": self.intercept_,
        }