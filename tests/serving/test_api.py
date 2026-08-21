"""Tests for the FastAPI serving layer (Task 6).

These tests cover the no-model-loaded state — verifying that /health
stays up, /ready reports not-ready, and /score returns 503. End-to-end
scoring with a real FraudPredictor artifact is out of scope for the
unit suite (it requires a trained model on disk).
"""

from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.config import settings


def _stub_no_model(monkeypatch):
    """Force the serving settings to have no model source configured."""
    monkeypatch.setattr(settings, "model_name", None)
    monkeypatch.setattr(settings, "model_artifact_dir", None)
    monkeypatch.setattr(settings, "feature_store_db", None)


def test_health_returns_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_reports_not_ready_without_model(monkeypatch):
    """With no MODEL_NAME / MODEL_ARTIFACT_DIR set, /ready must report
    model_loaded=False and not crash."""
    _stub_no_model(monkeypatch)

    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is False
    assert body["status"] == "not ready"


def test_score_returns_503_without_model(monkeypatch):
    _stub_no_model(monkeypatch)

    client = TestClient(app)
    resp = client.post(
        "/score",
        json={"TransactionDT": 1000, "TransactionAmt": 50.0},
    )
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()


def test_score_rejects_missing_required_fields(monkeypatch):
    """TransactionDT and TransactionAmt are required — missing them
    yields a 422 from Pydantic validation, not a 503."""
    _stub_no_model(monkeypatch)

    client = TestClient(app)
    resp = client.post("/score", json={"TransactionID": 1})
    assert resp.status_code == 422


def test_model_info_returns_503_without_model(monkeypatch):
    _stub_no_model(monkeypatch)

    client = TestClient(app)
    resp = client.get("/model-info")
    assert resp.status_code == 503
