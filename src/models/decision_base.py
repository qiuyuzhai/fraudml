"""
Abstract base class for post-scoring decision components.

A :class:`ModelBase` produces probabilities; a :class:`DecisionBase`
turns those probabilities into business actions (auto-pass / manual-review
/ auto-block) given a cost structure. Two existing components inherit
this ABC:

* :class:`ThresholdOptimizer` — single-threshold cost minimization.
* :class:`RiskDecisionEngine` — three-tier (LOW / MEDIUM / HIGH) decision.

Both already expose ``fit`` / ``predict`` / ``summary`` methods; this ABC
formalizes the contract so future decision components (e.g. graph-based
rules, velocity rules) slot in uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class DecisionBase(ABC):
    """Abstract base class for post-scoring decision components.

    Subclasses take predicted probabilities and produce business-level
    decisions (risk levels, recommended actions, cost-minimizing
    thresholds).

    Attributes
    ----------
    _fitted : bool
        Whether :meth:`fit` has been called successfully.
    """

    def __init__(self) -> None:
        self._fitted: bool = False

    @abstractmethod
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray, **kwargs: Any) -> "DecisionBase":
        """Learn decision thresholds from (y_true, y_prob) on validation data.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels (0/1).
        y_prob : np.ndarray
            Predicted positive-class probabilities.
        **kwargs
            Backend-specific arguments.

        Returns
        -------
        self : DecisionBase
        """
        ...

    @abstractmethod
    def predict(self, y_prob: np.ndarray, **kwargs: Any) -> Dict[str, np.ndarray]:
        """Apply learned thresholds to probabilities.

        Parameters
        ----------
        y_prob : np.ndarray
            Predicted positive-class probabilities.
        **kwargs
            Backend-specific arguments (e.g. override thresholds).

        Returns
        -------
        dict
            Backend-specific output (e.g. ``risk_levels``, ``recommended_actions``).
        """
        ...

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable summary of the learned decision policy."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._fitted
