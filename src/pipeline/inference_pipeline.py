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

    def trace_sample(
        self,
        df: pd.DataFrame,
        sample_index: int = 0,
        top_n_trees: int = 5,
    ) -> Dict:
        """Trace the decision path for a single incoming transaction.

        This is the main entry-point for online/near-line explainability:
        when a transaction is flagged (or approved), call this method to
        get a full breakdown of *why* the model reached that decision.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame (same schema as training input).
        sample_index : int
            Row index inside *df* to trace.
        top_n_trees : int
            Number of trees (sorted by leaf |value|) to detail in the
            step-by-step narrative.

        Returns
        -------
        dict with keys:
            - fraud_prob / calibrated_prob / raw_prob
            - risk_level, recommended_action, confidence
            - feature_contributions: ranked feature list
            - tree_traces: per-tree step-by-step paths
            - summary: human-readable narrative string

        Example
        -------
        >>> pipeline = InferencePipeline.load("artifacts/pipeline.pkl")
        >>> new_tx = pd.DataFrame([{...}])
        >>> result = pipeline.trace_sample(new_tx)
        >>> print(result["summary"])
        >>> print(f"Risk: {result['risk_level']} ({result['recommended_action']})")
        """
        if self._pipeline is None:
            self.load()
        return self._pipeline.trace_sample(df, sample_index=sample_index, top_n_trees=top_n_trees)

    def explain(
        self,
        df: pd.DataFrame,
        sample_index: int = 0,
    ) -> str:
        """Return a concise human-readable explanation string.

        Convenience wrapper around :meth:`trace_sample` that returns
        only the narrative, suitable for logging or UI display.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame.
        sample_index : int
            Row index to explain.

        Returns
        -------
        str
            Multi-line explanation text.
        """
        result = self.trace_sample(df, sample_index=sample_index)
        lines = [result.get("summary", "")]
        if result.get("risk_level"):
            lines.append("")
            lines.append(
                f"Final decision: {result['risk_level'].upper()} "
                f"→ {result['recommended_action']} "
                f"(confidence={result['confidence']:.2f})"
            )
        return "\n".join(lines)

    def score_and_explain(
        self,
        df: pd.DataFrame,
        sample_index: int = 0,
        threshold: Optional[float] = None,
    ) -> Dict:
        """Score a transaction and return both the decision and its explanation.

        Combines :meth:`predict` and :meth:`trace_sample` in one call,
        which is the typical pattern for real-time fraud screening where
        you need both the score and the reason.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame.
        sample_index : int
            Row index to score and explain.
        threshold : float, optional
            Classification threshold for the binary decision.

        Returns
        -------
        dict with keys:
            - probability / decision / risk_level / recommended_action
            - explanation: str (narrative)
            - trace: dict (full trace details)
        """
        if self._pipeline is None:
            self.load()

        row = df.iloc[[sample_index]]
        prob = float(self._pipeline.predict(row, threshold=None)[0])
        decision = int(self._pipeline.predict(row, threshold=threshold)[0])

        trace_result = self._pipeline.trace_sample(
            df, sample_index=sample_index
        )

        return {
            "probability": prob,
            "decision": decision,
            "risk_level": trace_result.get("risk_level"),
            "recommended_action": trace_result.get("recommended_action"),
            "confidence": trace_result.get("confidence"),
            "explanation": trace_result.get("summary", ""),
            "trace": trace_result,
        }