from .base import ModelBase
from .decision_base import DecisionBase
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel
from .threshold_optimizer import ThresholdOptimizer
from .risk_decision import RiskDecisionEngine


def make_model(model_type: str, params: dict | None = None) -> ModelBase:
    """Construct a :class:`ModelBase` wrapper by backend name.

    Parameters
    ----------
    model_type : str
        One of ``"lightgbm"`` / ``"xgboost"`` / ``"catboost"``
        (case-insensitive). Unknown values raise ``ValueError``.
    params : dict, optional
        Forwarded verbatim to the backend estimator constructor.

    Returns
    -------
    ModelBase
    """
    params = params or {}
    key = (model_type or "lightgbm").lower()
    if key == "lightgbm":
        return LightGBMModel(**params)
    if key == "xgboost":
        return XGBoostModel(**params)
    if key == "catboost":
        return CatBoostModel(**params)
    raise ValueError(f"Unknown model_type: {model_type!r}")


__all__ = [
    "ModelBase",
    "DecisionBase",
    "LightGBMModel",
    "XGBoostModel",
    "CatBoostModel",
    "ThresholdOptimizer",
    "RiskDecisionEngine",
    "make_model",
]
