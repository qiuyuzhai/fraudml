"""
Calibrator — Abstract base class for probability calibration.

All calibration methods must inherit from Calibrator and implement
fit() / transform() to produce well-calibrated probabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class Calibrator(ABC):
    """Abstract base class for probability calibrators.

    Calibrators map raw model scores to calibrated probabilities
    that reflect the true likelihood of the positive class.

    Parameters
    ----------
    name : str
        Human-readable identifier.

    Attributes (set after ``fit()``):
    ----------
    _fitted : bool
        Whether the calibrator has been fitted.
    """

    def __init__(self, name: str = "Calibrator") -> None:
        self.name = name
        self._fitted: bool = False

    @abstractmethod
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "Calibrator":
        """Learn calibration mapping from (probability, label) pairs.

        Parameters
        ----------
        y_prob : np.ndarray
            Raw predicted probabilities (shape: (n_samples,)).
        y_true : np.ndarray
            Ground truth binary labels (shape: (n_samples,)).

        Returns
        -------
        self : Calibrator
        """
        ...

    @abstractmethod
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities.

        Parameters
        ----------
        y_prob : np.ndarray
            Raw predicted probabilities (shape: (n_samples,)).

        Returns
        -------
        np.ndarray
            Calibrated probabilities (shape: (n_samples,)).
        """
        ...

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Convenience method: fit then transform."""
        self.fit(y_prob, y_true)
        return self.transform(y_prob)

    def get_params(self) -> Dict[str, Any]:
        """Return learned parameters as a dict."""
        return {"name": self.name, "fitted": self._fitted}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._fitted})"