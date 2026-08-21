"""
LightGBM wrapper implementing :class:`ModelBase`.

Wraps :class:`lightgbm.LGBMClassifier` so the train pipeline can treat all
backends uniformly via :func:`make_model`. The wrapper preserves LightGBM-specific
attributes (``booster_``, ``feature_name_``, ``feature_importances_``) through
:class:`ModelBase.__getattr__` delegation, so :class:`TreeAnalyzer` and
:class:`SHAPExplainer` keep working without code changes.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import ModelBase

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - optional in test envs
    _HAS_LIGHTGBM = False


class LightGBMModel(ModelBase):
    """Wrapper around :class:`lightgbm.LGBMClassifier`.

    Parameters
    ----------
    **lgb_params : Any
        Forwarded verbatim to :class:`lightgbm.LGBMClassifier`.
        Common keys: ``n_estimators``, ``learning_rate``, ``num_leaves``,
        ``max_depth``, ``min_child_samples``, ``subsample``,
        ``colsample_bytree``, ``reg_alpha``, ``reg_lambda``,
        ``scale_pos_weight``, ``random_state``, ``verbosity``.
    """

    def __init__(self, **lgb_params: Any) -> None:
        super().__init__(name="LightGBMModel")
        if not _HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is required for LightGBMModel. Install with `pip install lightgbm>=4.0.0`."
            )
        self._lgb_params: Dict[str, Any] = lgb_params
        self._model = lgb.LGBMClassifier(**lgb_params)

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> "LightGBMModel":
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
        feature_names = getattr(self._model, "feature_name_", None) or getattr(
            self._model, "feature_names_in_", None
        ) or [f"f{i}" for i in range(len(importances))]
        return {str(name): float(imp) for name, imp in zip(feature_names, importances)}
