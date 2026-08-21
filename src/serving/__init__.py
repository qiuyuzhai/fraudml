"""FastAPI online serving layer for FraudML.

Loads a FraudPredictor at startup (from a local artifact directory or
the MLflow Model Registry) and exposes /score, /explain, /health, /ready,
/model-info endpoints.
"""

from src.serving.app import app, get_predictor

__all__ = ["app", "get_predictor"]
