"""
CatBoost wrapper implementing :class:`ModelBase`.

Wraps :class:`catboost.CatBoostClassifier` so the train pipeline can switch
to CatBoost via ``model.type: catboost`` in the config.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import ModelBase

try:
    import catboost as cb

    _HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    _HAS_CATBOOST = False


class CatBoostModel(ModelBase):
    """Wrapper around :class:`catboost.CatBoostClassifier`.

    Parameters
    ----------
    **cb_params : Any
        Forwarded verbatim to :class:`catboost.CatBoostClassifier`.
        Common keys: ``iterations``, ``learning_rate``, ``depth``,
        ``l2_leaf_reg``, ``subsample``, ``colsample_bylevel``,
        ``scale_pos_weight``, ``random_state``, ``verbose``.
    """

    def __init__(self, **cb_params: Any) -> None:
        super().__init__(name="CatBoostModel")
        if not _HAS_CATBOOST:
            raise ImportError(
                "catboost is required for CatBoostModel. Install with `pip install catboost>=2.0.0`."
            )
        self._cb_params: Dict[str, Any] = cb_params
        self._model = cb.CatBoostClassifier(**cb_params)

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> "CatBoostModel":
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """2D ``(n, 2)`` array — column 1 is the positive class."""
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if not self._fitted or self._model is None:
            return {}
        try:
            importances = self._model.get_feature_importance()
        except Exception:
            return {}
        feature_names = (
            getattr(self._model, "feature_names_", None)
            or getattr(self._model, "feature_names_in_", None)
            or [f"f{i}" for i in range(len(importances))]
        )
        return {str(name): float(imp) for name, imp in zip(feature_names, importances)}
