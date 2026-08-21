"""
MLflow pyfunc wrapper for :class:`FraudPredictor`.

Packages the entire artifact directory (cleaner / feature registry /
selectors / model / calibrator / risk engine / metadata) as a single
MLflow model so it can be versioned in the Model Registry and loaded
back as a callable via ``mlflow.pyfunc.load_model``.

``load_context`` reconstructs a :class:`FraudPredictor` from the logged
artifact_dir; ``predict`` delegates to ``FraudPredictor.predict``.
The underlying :class:`FraudPredictor` is also exposed via the
``predictor`` attribute for callers that need the full object (e.g.
batch scoring CLI, FastAPI service) rather than just a callable.
"""

from __future__ import annotations

from typing import Any, Optional

import mlflow.pyfunc
import pandas as pd


class FraudMLPyFunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper exposing a :class:`FraudPredictor`.

    The wrapped predictor is constructed lazily in :meth:`load_context`
    from the ``artifact_dir`` artifact recorded at log time.
    """

    def __init__(self) -> None:
        self._predictor: Optional[Any] = None

    def load_context(self, context) -> None:  # noqa: D401 - mlflow hook name
        """Load the FraudPredictor from the logged ``artifact_dir`` artifact."""
        from src.pipeline.predict import FraudPredictor

        artifact_dir = context.artifacts.get("artifact_dir")
        if artifact_dir is None:
            raise RuntimeError(
                "FraudMLPyFunc: 'artifact_dir' artifact missing; cannot load predictor."
            )
        self._predictor = FraudPredictor.from_artifact_dir(artifact_dir)

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Delegate to :meth:`FraudPredictor.predict` with ``return_all=True``."""
        if self._predictor is None:
            raise RuntimeError("FraudMLPyFunc: predictor not loaded. Call load_context first.")
        return self._predictor.predict(model_input, return_all=True)

    @property
    def predictor(self) -> Any:
        """Direct access to the wrapped :class:`FraudPredictor` instance."""
        if self._predictor is None:
            raise RuntimeError("FraudMLPyFunc: predictor not loaded yet.")
        return self._predictor
