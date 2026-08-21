"""
XGBoost wrapper implementing :class:`ModelBase`.

Wraps :class:`xgboost.XGBClassifier` so the train pipeline can switch to
XGBoost via ``model.type: xgboost`` in the config.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import ModelBase

try:
    import xgboost as xgb

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False


class XGBoostModel(ModelBase):
    """Wrapper around :class:`xgboost.XGBClassifier`.

    Parameters
    ----------
    **xgb_params : Any
        Forwarded verbatim to :class:`xgboost.XGBClassifier`.
        Common keys: ``n_estimators``, ``learning_rate``, ``max_depth``,
        ``min_child_weight``, ``subsample``, ``colsample_bytree``,
        ``reg_alpha``, ``reg_lambda``, ``scale_pos_weight``,
        ``random_state``.
    """

    def __init__(self, **xgb_params: Any) -> None:
        super().__init__(name="XGBoostModel")
        if not _HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for XGBoostModel. Install with `pip install xgboost>=1.6.0`."
            )
        self._xgb_params: Dict[str, Any] = xgb_params
        self._model = xgb.XGBClassifier(**xgb_params)

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> "XGBoostModel":
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
            importances = getattr(self._model, "feature_importances_", None)
        except AttributeError:
            importances = None
        if importances is None:
            return {}
        feature_names = getattr(self._model, "feature_names_in_", None) or [
            f"f{i}" for i in range(len(importances))
        ]
        return {str(name): float(imp) for name, imp in zip(feature_names, importances)}
