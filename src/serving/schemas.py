"""
Pydantic schemas for the FastAPI serving layer.

``Transaction`` accepts the IEEE-CIS transaction schema. Because the raw
schema has 400+ columns, we use ``extra="allow"`` so any field present
in the payload is forwarded to the predictor verbatim. A handful of
commonly-typed fields are declared explicitly so OpenAPI / docs surface
them; everything else is captured in the ``extra`` mapping.

``ScoreResponse`` and ``ExplainResponse`` are the contract for ``/score``
and ``/explain`` respectively.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """A single fraud transaction (IEEE-CIS schema, subset typed)."""

    model_config = ConfigDict(extra="allow")

    TransactionID: Optional[int] = None
    TransactionDT: int = Field(..., description="Seconds since reference time.")
    TransactionAmt: float = Field(..., description="Transaction amount in USD.")
    ProductCD: Optional[str] = None
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None


class ScoreResponse(BaseModel):
    """Response from ``POST /score``."""

    transaction_id: Optional[int] = None
    probability: float
    risk_level: Optional[str] = None
    recommended_action: Optional[str] = None
    binary_prediction: Optional[int] = None
    features_degraded: bool
    model_version: Optional[str] = None
    decision_source: str = "model"
    matched_rule: Optional[str] = None


class ExplainResponse(BaseModel):
    """Response from ``POST /explain``."""

    transaction_id: Optional[int] = None
    probability: float
    shap_top_features: dict[str, float] = Field(default_factory=dict)
    model_version: Optional[str] = None


class ModelInfo(BaseModel):
    """Response from ``GET /model-info``."""

    model_type: str
    n_features: int
    selected_features: list[str] = Field(default_factory=list)
    has_calibrator: bool
    has_risk_engine: bool
    has_vif_filter: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    model_version: Optional[str] = None


class HealthResponse(BaseModel):
    """Response from ``GET /health`` (liveness)."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Response from ``GET /ready`` (readiness)."""

    status: str
    model_loaded: bool
    feature_store_ok: bool
    warnings: list[str] = Field(default_factory=list)
