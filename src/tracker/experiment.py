"""
MLflow experiment tracking integration.

Provides a thin wrapper around MLflow for logging parameters,
metrics, and artifacts during model training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.lightgbm


class ExperimentTracker:
    """MLflow-based experiment tracker.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    tracking_uri : str, optional
        MLflow tracking URI.  Defaults to local ``./mlruns``.

    Attributes
    ----------
    run_id : str
        Current MLflow run ID.
    """

    def __init__(
        self,
        experiment_name: str = "IEEE_Fraud_Detection",
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id: Optional[str] = None

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None) -> "ExperimentTracker":
        """Start a new MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Name for this run.

        Returns
        -------
        self : ExperimentTracker
        """
        self._run = mlflow.start_run(run_name=run_name)
        self.run_id = self._run.info.run_id
        return self

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run.

        Parameters
        ----------
        params : dict
            Key-value parameter pairs.
        """
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Log metrics to the current run.

        Parameters
        ----------
        metrics : dict
            Key-value metric pairs.
        step : int
            Step index for the metric.
        """
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value), step=step)
            except Exception:
                pass

    def log_artifact(self, local_path: str | Path) -> None:
        """Log an artifact file or directory.

        Parameters
        ----------
        local_path : str or Path
            Path to the artifact to log.
        """
        try:
            mlflow.log_artifact(str(local_path))
        except Exception:
            pass

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    def register_model(
        self,
        name: str,
        artifact_dir: str | Path,
        stage: str = "Staging",
    ) -> Optional[str]:
        """Register the trained pipeline (artifact_dir) to the Model Registry.

        Packages the full artifact directory (cleaner / feature
        registry / selectors / model / calibrator / risk engine /
        metadata) as a pyfunc model so ``mlflow.pyfunc.load_model``
        returns a callable wrapping :class:`FraudPredictor`. Then calls
        :func:`mlflow.register_model` to create a new model version and
        transitions it to *stage*.

        Parameters
        ----------
        name : str
            Registered model name (created if it does not exist).
        artifact_dir : str | Path
            Path to the pipeline artifact directory (the one produced
            by :class:`ModelSerializer`).
        stage : str
            Initial stage for the new version (``"Staging"`` by
            default). Transitioning to ``"Production"`` is a separate
            manual step (``MlflowClient.transition_model_version_stage``).

        Returns
        -------
        str or None
            The registered version number on success, or ``None`` if
            registration failed (logged, not raised — the train pipeline
            should not abort on registry failure).
        """
        try:
            from mlflow.pyfunc import log_model
            from mlflow.tracking import MlflowClient

            from .model_pyfunc import FraudMLPyFunc

            artifacts = {"artifact_dir": str(Path(artifact_dir).resolve())}
            log_model(
                model=FraudMLPyFunc(),
                artifact_path="model",
                artifacts=artifacts,
            )
            model_uri = f"runs:/{self.run_id}/model"
            result = mlflow.register_model(model_uri=model_uri, name=name)

            if stage:
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=name,
                    version=int(result.version),
                    stage=stage,
                    archive_existing_versions=True,
                )
            return str(result.version)
        except Exception as e:
            print(f"[ExperimentTracker] register_model failed: {e}")
            return None