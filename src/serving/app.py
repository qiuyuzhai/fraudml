"""FastAPI application exposing FraudPredictor for real-time scoring.

Lifespan loads the predictor once at startup (from a local artifact
directory or the MLflow Model Registry). Endpoints reuse the predictor
and the existing feature pipeline — no re-implementation.

Risk architecture (rules + model):
    交易 → [规则引擎] → 命中 → 直接 block/challenge（不走模型）
                      → 未命中 → ML 模型打分 → 三层分级

Endpoints:
- POST /score             : score a single transaction (rules + model)
- POST /explain           : SHAP-based local explanation
- GET  /health            : liveness probe
- GET  /ready             : readiness probe (model loaded + feature store ok)
- GET  /model-info        : pipeline metadata + model metrics
- GET  /rules             : list active pre-model rules
- POST /admin/blacklist   : add a card1 value to the blacklist
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.rules import RuleEngine, create_default_engine
from src.serving.config import settings
from src.serving.schemas import (
    ExplainResponse,
    HealthResponse,
    ModelInfo,
    ReadyResponse,
    ScoreResponse,
    Transaction,
)

if TYPE_CHECKING:
    from src.pipeline.predict import FraudPredictor

logger = logging.getLogger("fraudml.serving")

# Stateful feature classes that require historical context and therefore
# degrade when no online feature service is wired up. When any of these
# appear in the predictor's registry.execution_order, features_degraded
# is set to True on responses.
_STATEFUL_FEATURE_KINDS = ("HistoryFeature", "AggregationFeature")


class AppState:
    """Mutable application state populated at startup."""

    def __init__(self) -> None:
        self.predictor: Optional[FraudPredictor] = None
        self.model_version: Optional[str] = None
        self.features_degraded: bool = False
        self.feature_store_warnings: list[str] = []
        self.rule_engine: RuleEngine = create_default_engine()

    def reset(self) -> None:
        self.predictor = None
        self.model_version = None
        self.features_degraded = False
        self.feature_store_warnings = []
        self.rule_engine = create_default_engine()


_state = AppState()


def _load_predictor() -> "FraudPredictor":
    """Build a FraudPredictor from settings.

    Prefers MLflow Model Registry when ``MODEL_NAME`` is set; falls back
    to ``MODEL_ARTIFACT_DIR``. Raises if neither is configured.

    The import is deferred to call time so module import does not pull in
    the training pipeline's heavy dependencies (mlflow, lightgbm). The
    serving layer stays importable even when training-side packages are
    missing, and surfaces the failure as a /ready warning instead.
    """
    from src.pipeline.predict import FraudPredictor

    if settings.model_name:
        logger.info(
            "Loading predictor from MLflow registry: name=%s stage=%s",
            settings.model_name,
            settings.model_stage,
        )
        predictor = FraudPredictor.from_model_registry(
            name=settings.model_name,
            stage=settings.model_stage,
            tracking_uri=settings.mlflow_tracking_uri,
        )
        _state.model_version = f"{settings.model_name}:{settings.model_stage}"
        return predictor

    if settings.model_artifact_dir:
        logger.info(
            "Loading predictor from artifact dir: %s",
            settings.model_artifact_dir,
        )
        predictor = FraudPredictor.from_artifact_dir(settings.model_artifact_dir)
        _state.model_version = settings.model_artifact_dir
        return predictor

    raise RuntimeError(
        "No model source configured. Set MODEL_NAME (registry) or "
        "MODEL_ARTIFACT_DIR (local) in the environment."
    )


def _detect_degraded_mode(predictor: FraudPredictor) -> bool:
    """Return True if stateful history features are in the pipeline.

    Their presence means single-transaction requests lack the historical
    context they were trained on, so responses should flag degradation.
    """
    execution_order = getattr(predictor.registry, "execution_order", []) or []
    if not execution_order:
        return False
    instances = getattr(predictor.registry, "features", {}) or {}
    for name in execution_order:
        instance = instances.get(name)
        if instance is None:
            continue
        cls_name = type(instance).__name__
        if cls_name in _STATEFUL_FEATURE_KINDS:
            return True
    return False


def _check_feature_store() -> list[str]:
    """Verify active feature versions align with selected_features_.

    Returns a list of warning strings (empty when the store is not
    configured or everything aligns). Never raises — readiness should
    not crash on a stale store, only warn.
    """
    if not settings.feature_store_db:
        return []

    warnings: list[str] = []
    try:
        from src.feature_store import FeatureStore

        store = FeatureStore(settings.feature_store_db)
        registered = {f["name"] for f in store.registry.list_features()}
        selected = set(_state.predictor.selected_features_) if _state.predictor else set()

        missing = sorted(selected - registered)
        if missing:
            warnings.append(
                f"{len(missing)} selected feature(s) not registered in FeatureStore: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
    except Exception as exc:
        warnings.append(f"FeatureStore check skipped: {exc}")
    return warnings


def get_predictor() -> FraudPredictor:
    """Dependency accessor for the loaded predictor.

    Raises HTTPException 503 when the model is not loaded yet.
    """
    if _state.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /ready for readiness status.",
        )
    return _state.predictor


def _load_blacklist_from_train(blacklist_file: str = "data/blacklist.txt") -> set:
    """Load blacklisted card1 values from a file (one per line).

    If the file does not exist, returns an empty set. In production,
    this would be a Redis SET populated by the fraud-ops team.
    """
    p = Path(blacklist_file)
    if not p.exists():
        return set()
    values = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values.add(line)
    logger.info("Loaded %d blacklisted card1 values from %s", len(values), p)
    return values


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the FraudPredictor and rule engine at startup, clear at shutdown."""
    _state.reset()

    # Load blacklist into rule engine
    blacklist = _load_blacklist_from_train()
    for rule in _state.rule_engine.rules:
        if hasattr(rule, "_blacklist"):
            for v in blacklist:
                rule.add(v)

    try:
        _state.predictor = _load_predictor()
        _state.features_degraded = _detect_degraded_mode(_state.predictor)
        _state.feature_store_warnings = _check_feature_store()
        logger.info(
            "Predictor ready (version=%s, features_degraded=%s, warnings=%d)",
            _state.model_version,
            _state.features_degraded,
            len(_state.feature_store_warnings),
        )
    except Exception as exc:
        logger.error("Failed to load predictor at startup: %s", exc)
        _state.feature_store_warnings = [f"startup error: {exc}"]
    yield
    _state.reset()


app = FastAPI(
    title="FraudML Online Scoring API",
    description="Real-time fraud probability scoring built on FraudPredictor.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Liveness probe — process is up."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["health"])
def ready() -> ReadyResponse:
    """Readiness probe — model loaded + feature store aligned."""
    model_loaded = _state.predictor is not None
    feature_store_ok = not _state.feature_store_warnings
    status_str = "ready" if (model_loaded and feature_store_ok) else "not ready"
    return ReadyResponse(
        status=status_str,
        model_loaded=model_loaded,
        feature_store_ok=feature_store_ok,
        warnings=_state.feature_store_warnings,
    )


@app.get("/model-info", response_model=ModelInfo, tags=["meta"])
def model_info() -> ModelInfo:
    """Return metadata for the loaded pipeline."""
    predictor = get_predictor()
    info = predictor.get_model_info()
    return ModelInfo(
        model_type=info["model_type"],
        n_features=info["n_features"],
        selected_features=info["selected_features"],
        has_calibrator=info["has_calibrator"],
        has_risk_engine=info["has_risk_engine"],
        has_vif_filter=info["has_vif_filter"],
        metrics=info["metrics"],
        model_version=_state.model_version,
    )


@app.get("/rules", tags=["rules"])
def list_rules() -> dict:
    """List active pre-model rules and their configuration."""
    return {"rules": _state.rule_engine.list_rules()}


@app.post("/admin/blacklist", tags=["rules"])
def add_blacklist(card1: str) -> dict:
    """Add a card1 value to the blacklist (runtime, in-memory).

    In production this would write to Redis and propagate to all
    instances. Here it updates the in-memory set of the current process.
    """
    for rule in _state.rule_engine.rules:
        if hasattr(rule, "add"):
            rule.add(card1)
    return {"status": "added", "card1": card1}


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(transaction: Transaction) -> ScoreResponse:
    """Score a single transaction.

    Pipeline: rule engine → (hit?) block/challenge : ML model → three-tier risk.
    """
    payload = transaction.model_dump(exclude_none=False)
    tx_id = payload.get("TransactionID")

    # ---- Stage 1: pre-model rule engine ----
    rule_result = _state.rule_engine.evaluate(payload)
    if rule_result.action != "pass":
        if rule_result.action == "block":
            return ScoreResponse(
                transaction_id=tx_id,
                probability=1.0,
                risk_level="high",
                recommended_action="block_and_review",
                binary_prediction=1,
                features_degraded=False,
                model_version=None,
                decision_source="rule_engine",
                matched_rule=rule_result.matched_rule,
            )
        # challenge
        return ScoreResponse(
            transaction_id=tx_id,
            probability=0.5,
            risk_level="medium",
            recommended_action="challenge_step_up",
            binary_prediction=None,
            features_degraded=False,
            model_version=None,
            decision_source="rule_engine",
            matched_rule=rule_result.matched_rule,
        )

    # ---- Stage 2: ML model scoring ----
    predictor = get_predictor()
    df = pd.DataFrame([payload])

    result = predictor.predict(df, return_all=True)

    if isinstance(result, pd.DataFrame):
        row = result.iloc[0]
        probability = float(row.get("probability", 0.0))
        risk_level = row.get("risk_level")
        binary_prediction = row.get("binary_prediction")
    else:
        probability = float(result[0])
        risk_level = None
        binary_prediction = None

    recommended_action = _recommend_action(risk_level, probability)

    return ScoreResponse(
        transaction_id=tx_id,
        probability=probability,
        risk_level=str(risk_level) if risk_level is not None else None,
        recommended_action=recommended_action,
        binary_prediction=int(binary_prediction) if binary_prediction is not None else None,
        features_degraded=_state.features_degraded,
        model_version=_state.model_version,
        decision_source="model",
        matched_rule=None,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["scoring"])
def explain(transaction: Transaction) -> ExplainResponse:
    """Score + return top SHAP feature contributions for one transaction."""
    predictor = get_predictor()

    payload = transaction.model_dump(exclude_none=False)
    tx_id = payload.get("TransactionID")
    df = pd.DataFrame([payload])

    result = predictor.predict(df, return_all=True)
    if isinstance(result, pd.DataFrame):
        probability = float(result.iloc[0].get("probability", 0.0))
    else:
        probability = float(result[0])

    shap_top = _compute_shap_top(predictor, df)

    return ExplainResponse(
        transaction_id=tx_id,
        probability=probability,
        shap_top_features=shap_top,
        model_version=_state.model_version,
    )


def _recommend_action(risk_level: Optional[str], probability: float) -> str:
    """Map risk level / probability to a business action.

    The risk engine returns lowercase labels ('low'/'medium'/'high');
    compare case-insensitively so the action always aligns with the
    engine's verdict. When no risk engine is wired up, fall back to
    probability thresholds.
    """
    if risk_level:
        level = risk_level.lower()
        if level == "high":
            return "block_and_review"
        if level == "medium":
            return "challenge_step_up"
        if level == "low":
            return "allow"
    if probability >= 0.8:
        return "block_and_review"
    if probability >= 0.4:
        return "challenge_step_up"
    return "allow"


def _compute_shap_top(predictor: FraudPredictor, df: pd.DataFrame) -> dict[str, float]:
    """Best-effort SHAP explanation for the input row.

    Returns the top contributing features by absolute SHAP value. When
    SHAP is unavailable or the explainer fails, returns an empty dict
    rather than raising — explanation is a soft requirement.
    """
    try:
        from src.interpretability.shap_explainer import SHAPExplainer

        selected = predictor.selected_features_
        if not selected:
            return {}

        explainer = SHAPExplainer(
            model=predictor.model,
            feature_names=selected,
            max_features=10,
            n_samples=1,
        )
        explainer.fit(df.head(1))
        explanation = explainer.local_explanation(df, idx=0)
        top = dict(list(explanation.items())[:10])
        return {k: float(v) for k, v in top.items()}
    except Exception as exc:
        logger.debug("SHAP explanation skipped: %s", exc)
        return {}
