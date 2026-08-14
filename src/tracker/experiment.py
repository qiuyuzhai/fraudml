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