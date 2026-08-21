"""
Abstract base class for classification scoring models.

Mirrors the ABC pattern of :class:`FeatureBase` / :class:`SelectionBase` /
:class:`Calibrator`: every backend estimator (LightGBM / XGBoost / CatBoost)
must inherit from :class:`ModelBase` and implement a uniform
``fit / predict_proba / get_feature_importance / save / load`` contract.

Design principles
-----------------
* **Backend-agnostic contract** — train_pipeline works with any
  ``ModelBase`` subclass via the ``make_model(model_type, params)`` factory;
  switching backends is a one-line config change.
* **Sklearn-compatible** — ``predict_proba`` returns a 2D ``(n, 2)`` array
  so existing downstream code (``[:, 1]`` slicing in :class:`FraudPredictor`,
  :class:`SHAPExplainer`, calibrators) keeps working unchanged.
* **Transparent delegation** — backend-specific attributes (``booster_``
  for LightGBM, ``feature_importances_`` for sklearn) are forwarded via
  ``__getattr__`` so :class:`TreeAnalyzer` / :class:`SHAPExplainer` keep
  accessing the underlying estimator without API changes.
* **Serializable parameters only** — ``save()`` / ``load()`` use joblib,
  matching the :class:`FeatureBase` persistence pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd


class ModelBase(ABC):
    """Abstract base class for binary classification scoring models.

    Subclasses wrap a backend estimator and expose a uniform interface.
    The wrapper instance is what gets persisted as ``model.joblib`` —
    downstream code (``FraudPredictor``, ``SHAPExplainer``, ``TreeAnalyzer``)
    treats it as the model object.

    Parameters
    ----------
    name : str
        Human-readable identifier for logging / registry.

    Attributes
    ----------
    name : str
    _fitted : bool
        Whether :meth:`fit` has been called successfully.
    _model : Optional[Any]
        The underlying backend estimator (e.g. ``lgb.LGBMClassifier``).
        ``None`` until ``fit()`` constructs it.
    """

    def __init__(self, name: str = "ModelBase") -> None:
        self.name = name
        self._fitted: bool = False
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Core contract — fit / predict_proba / feature importance
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> "ModelBase":
        """Train the underlying estimator on (X, y).

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : np.ndarray
            Binary target labels (0/1).
        **kwargs
            Backend-specific fit arguments (e.g. ``eval_set`` for early
            stopping).

        Returns
        -------
        self : ModelBase
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities as a 2D ``(n, 2)`` array.

        Sklearn-compatible — downstream code slices with ``[:, 1]`` to
        extract the positive-class probability.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n, 2)``; column 0 is the negative-class probability,
            column 1 is the positive-class probability.
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return ``{feature_name: importance}`` for the fitted model.

        Returns
        -------
        dict
            Empty if the model is not fitted or the backend exposes no
            importance signal.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence — serialize the wrapper (not just the estimator)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist this wrapper (including the fitted estimator) via joblib."""
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "ModelBase":
        """Load a :class:`ModelBase` wrapper previously saved via :meth:`save`."""
        return joblib.load(Path(path))

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._fitted

    # ------------------------------------------------------------------
    # Transparent delegation — forward unknown attributes to the backend
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # ``__getattr__`` is only called when normal attribute lookup fails.
        # Forward backend-specific attributes (``booster_``, ``feature_importances_``,
        # ``feature_name_``, ``feature_names_in_``, ...) to the underlying
        # estimator so :class:`TreeAnalyzer` / :class:`SHAPExplainer` keep working.
        try:
            model = object.__getattribute__(self, "_model")
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        if model is not None:
            try:
                return getattr(model, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
