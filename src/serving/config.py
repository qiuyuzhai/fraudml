"""
Settings for the FastAPI serving layer.

Reads from environment variables / a ``.env`` file via
:class:`pydantic_settings.BaseSettings`. No hardcoded paths — every
deployment can override via env.

The default model loading flow prefers the MLflow Model Registry (when
``MODEL_NAME`` + ``MODEL_STAGE`` are set) and falls back to the local
artifact directory (``MODEL_ARTIFACT_DIR``) when the registry is not
available.
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Serving-layer configuration loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Local artifact directory fallback
    model_artifact_dir: Optional[str] = None

    # MLflow Model Registry preferred path
    model_name: Optional[str] = None
    model_stage: str = "Production"
    mlflow_tracking_uri: Optional[str] = None

    # Feature Store DB (read at /ready for alignment check)
    feature_store_db: Optional[str] = None


settings = Settings()
