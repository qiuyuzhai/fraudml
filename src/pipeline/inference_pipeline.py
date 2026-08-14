"""
FraudML Inference Pipeline — load a saved pipeline and score new data.

Usage::

    pipeline = InferencePipeline.load("artifacts/pipeline.pkl")
    scores = pipeline.predict(new_transactions_df)
    binary = pipeline.predict(new_transactions_df, threshold=0.7)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .train_pipeline import TrainPipeline


class InferencePipeline:
    """Load a serialized TrainPipeline and run inference.

    This is a thin wrapper around :class:`TrainPipeline.load` and
    :meth:`TrainPipeline.predict` for production scoring.

    Parameters
    ----------
    pipeline_path : str or Path
        Path to the serialized pipeline (.pkl file).
    """

    def __init__(self, pipeline_path: str | Path) -> None:
        self.pipeline_path = Path(pipeline_path)
        self._pipeline: Optional[TrainPipeline] = None

    def load(self) -> TrainPipeline:
        """Load the serialized pipeline.

        Returns
        -------
        TrainPipeline
            The loaded pipeline ready for prediction.
        """
        self._pipeline = TrainPipeline.load(self.pipeline_path)
        return self._pipeline

    def predict(
        self, df: pd.DataFrame, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Generate fraud probability scores or binary predictions.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame.
        threshold : float, optional
            Classification threshold for binary predictions.
            If not provided, returns probabilities.  If the pipeline
            has a learned optimal threshold, that value is used.

        Returns
        -------
        np.ndarray
            Fraud probability scores (0 to 1) or binary predictions
            (0 = legitimate, 1 = fraud).
        """
        if self._pipeline is None:
            self.load()
        return self._pipeline.predict(df, threshold=threshold)

    def predict_with_threshold(
        self, df: pd.DataFrame, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Generate binary fraud predictions using optimal threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame.
        threshold : float, optional
            Classification threshold.  Uses the pipeline's learned
            optimal threshold if not provided.

        Returns
        -------
        np.ndarray
            Binary predictions (0 = legitimate, 1 = fraud).
        """
        if self._pipeline is None:
            self.load()

        if threshold is None and self._pipeline.threshold_optimizer_ is not None:
            threshold = self._pipeline.threshold_optimizer_.best_threshold_
        elif threshold is None:
            threshold = 0.5

        return self._pipeline.predict(df, threshold=threshold)

    @property
    def selected_features(self) -> List[str]:
        """Return the list of features used by the model."""
        if self._pipeline is None:
            self.load()
        return self._pipeline.metadata_.get("selected_features", [])