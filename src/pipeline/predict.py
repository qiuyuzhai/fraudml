"""
FraudPredictor — Standalone online inference module.

Loads only the stateful components needed for real-time fraud prediction,
decoupling inference from the training pipeline.

Usage::

    from src.pipeline.predict import FraudPredictor

    predictor = FraudPredictor.from_artifact_dir("artifacts/run_20260813_143052_a1b2c3")
    result = predictor.predict(transaction_df)

Designed for:
- Batch scoring (offline model scoring jobs)
- Real-time scoring (online fraud detection service)
- A/B testing (swap models without retraining)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.persistence import ModelSerializer


class FraudPredictor:
    """Standalone fraud prediction engine for online inference.

    Parameters
    ----------
    cleaner : DataCleaner
        Fitted data cleaner.
    registry : FeatureRegistry
        Fitted feature registry with loaded stateful components.
    iv_selector : IVSelector
        Fitted IV feature selector.
    vif_filter : VIFFilter or None
        Fitted VIF filter (or None for tree models).
    model : sklearn / lightgbm model
        Trained classification model.
    calibrator : Calibrator or None
        Fitted probability calibrator (or None).
    risk_engine : RiskDecisionEngine or None
        Fitted multi-level risk engine (or None).
    metadata : dict
        Pipeline metadata (config, feature lists, etc.).
    """

    def __init__(
        self,
        cleaner: Any,
        registry: Any,
        iv_selector: Any,
        vif_filter: Optional[Any],
        model: Any,
        calibrator: Optional[Any] = None,
        risk_engine: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cleaner = cleaner
        self.registry = registry
        self.iv_selector = iv_selector
        self.vif_filter = vif_filter
        self.model = model
        self.calibrator = calibrator
        self.risk_engine = risk_engine
        self.metadata = metadata or {}

        self.selected_features_: List[str] = self.metadata.get("selected_features", [])
        self.raw_columns_: List[str] = self.metadata.get("raw_columns", [])
        self._streaming_initialized: bool = False

        self.logger = logging.getLogger(f"FraudPredictor-{id(self)}")

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "FraudPredictor":
        """Load a predictor from a serialized artifact directory.

        Parameters
        ----------
        artifact_dir : str | Path
            Path to the artifact directory (e.g. ``artifacts/run_20260813_143052_a1b2c3``).

        Returns
        -------
        FraudPredictor
            Loaded predictor ready for inference.
        """
        serializer = ModelSerializer(artifact_dir)
        components = serializer.deserialize_for_inference()

        predictor = cls(
            cleaner=components["cleaner"],
            registry=components["registry"],
            iv_selector=components["iv_selector"],
            vif_filter=components.get("vif_filter"),
            model=components["model"],
            calibrator=components.get("calibrator"),
            risk_engine=components.get("risk_engine"),
            metadata=components.get("metadata", {}),
        )

        streaming_features = predictor.init_streaming()
        if streaming_features:
            predictor.logger.info(
                "Auto-initialized streaming for: %s", streaming_features
            )

        return predictor

    @classmethod
    def from_model_registry(
        cls,
        name: str,
        stage: str = "Production",
        tracking_uri: Optional[str] = None,
    ) -> "FraudPredictor":
        """Load a predictor from the MLflow Model Registry.

        Resolves the latest model version in *stage* for *name*,
        downloads the logged ``artifact_dir`` and reconstructs the
        predictor via :meth:`from_artifact_dir`. Falls back to
        :meth:`from_artifact_dir` semantics — no model is silently
        reconstructed from a bare estimator.

        Parameters
        ----------
        name : str
            Registered model name.
        stage : str
            Stage to resolve (``"Production"`` default).
        tracking_uri : str, optional
            MLflow tracking URI; forwarded to ``mlflow.set_tracking_uri``.

        Returns
        -------
        FraudPredictor
        """
        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        versions = client.get_latest_versions(name, stages=[stage])
        if not versions:
            raise ValueError(
                f"No model version found for name='{name}' stage='{stage}'."
            )
        version = versions[0]
        run_id = version.run_id

        # Download the "model" artifact path containing artifact_dir/
        dst_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model"
        )
        artifact_dir = Path(dst_path) / "artifact_dir"
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"artifact_dir not found in downloaded model artifacts at {dst_path}"
            )
        return cls.from_artifact_dir(artifact_dir)

    def predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        return_all: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Generate fraud predictions on new transaction data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame.  Must contain the columns
            expected by the pipeline (validated against raw_columns_
            when available).
        threshold : float, optional
            Classification threshold.  If None, returns probabilities.
            If the pipeline has a risk engine, uses risk-based classification.
        return_all : bool
            If True, returns a DataFrame with probabilities, risk levels,
            and key feature values.  Otherwise returns only the
            probability array.

        Returns
        -------
        np.ndarray or pd.DataFrame
            - If ``return_all=False``: probability array (n_samples,)
            - If ``return_all=True``: DataFrame with columns:
              probability, risk_level, binary_prediction,
              and top SHAP features (if available)
        """
        self._validate_input(df)

        df_clean = self.cleaner.transform(df)
        df_encoded = self.registry.transform_all(df_clean)

        for col in ["isFraud", "TransactionDT"]:
            if col in df_encoded.columns:
                df_encoded = df_encoded.drop(columns=[col])

        df_selected = self.iv_selector.transform(df_encoded)

        if self.vif_filter is not None:
            df_final = self.vif_filter.transform(df_selected)
        else:
            df_final = df_selected

        if self.selected_features_:
            for col in self.selected_features_:
                if col not in df_final.columns:
                    df_final[col] = 0.0
            df_final = df_final[self.selected_features_]

        y_prob_raw = self.model.predict_proba(df_final)[:, 1]

        if self.calibrator is not None and getattr(self.calibrator, "_fitted", False):
            y_prob = self.calibrator.transform(y_prob_raw)
        else:
            y_prob = y_prob_raw

        if threshold is None:
            if self.risk_engine is not None:
                risk_result = self.risk_engine.predict(y_prob)
                risk_levels = risk_result["risk_levels"]
                result = pd.DataFrame({
                    "probability": y_prob,
                    "risk_level": risk_levels,
                })
                self.logger.info(
                    "Predicted %d transactions: LOW=%d, MEDIUM=%d, HIGH=%d",
                    len(y_prob),
                    int((risk_levels == "LOW").sum()),
                    int((risk_levels == "MEDIUM").sum()),
                    int((risk_levels == "HIGH").sum()),
                )
                if return_all:
                    return result
                return y_prob
            return y_prob

        y_pred = (y_prob >= threshold).astype(int)
        if return_all:
            result = pd.DataFrame({
                "probability": y_prob,
                "binary_prediction": y_pred,
            })
            return result
        return y_prob

    def predict_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 10000,
        **kwargs: Any,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Generate predictions in batches (memory-efficient for large datasets).

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame.
        batch_size : int
            Number of rows per batch.
        **kwargs
            Passed to :meth:`predict`.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Concatenated predictions.
        """
        all_results = []
        n_total = len(df)

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = df.iloc[start:end]
            batch_result = self.predict(batch, **kwargs)
            if isinstance(batch_result, pd.DataFrame):
                all_results.append(batch_result)
            else:
                all_results.append(batch_result)

        if all_results and isinstance(all_results[0], pd.DataFrame):
            return pd.concat(all_results, axis=0).reset_index(drop=True)
        return np.concatenate(all_results)

    def init_streaming(self) -> List[str]:
        """Initialize streaming features for real-time incremental updates.

        Calls ``init_streaming_all()`` on the feature registry.
        Must be called once after loading the predictor and before
        using :meth:`predict_streaming`.

        Returns
        -------
        list[str]
            Names of features that were initialized for streaming.
        """
        if self._streaming_initialized:
            return self.registry.get_streaming_features()

        initialized = self.registry.init_streaming_all()
        self._streaming_initialized = True
        self.logger.info(
            "Streaming initialized for: %s", initialized
        )
        return initialized

    def predict_streaming(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        return_all: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Real-time prediction with incremental streaming updates.

        For each row in *df*, updates streaming features (e.g.
        SequenceFeature embeddings) before scoring, so that the
        model sees the most up-to-date behavioral state.

        Parameters
        ----------
        df : pd.DataFrame
            New transaction(s).  Must contain the columns expected
            by the pipeline.
        threshold : float, optional
            Classification threshold.
        return_all : bool
            If True, returns a DataFrame with probabilities and
            risk levels.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Prediction result (same format as :meth:`predict`).
        """
        if not self._streaming_initialized:
            self.init_streaming()

        self._validate_input(df)

        df_clean = self.cleaner.transform(df)

        df_updated = self.registry.update_streaming_all(df_clean)

        df_encoded = self.registry.transform_all(df_updated)

        for col in ["isFraud", "TransactionDT"]:
            if col in df_encoded.columns:
                df_encoded = df_encoded.drop(columns=[col])

        df_selected = self.iv_selector.transform(df_encoded)

        if self.vif_filter is not None:
            df_final = self.vif_filter.transform(df_selected)
        else:
            df_final = df_selected

        if self.selected_features_:
            for col in self.selected_features_:
                if col not in df_final.columns:
                    df_final[col] = 0.0
            df_final = df_final[self.selected_features_]

        y_prob_raw = self.model.predict_proba(df_final)[:, 1]

        if self.calibrator is not None and getattr(self.calibrator, "_fitted", False):
            y_prob = self.calibrator.transform(y_prob_raw)
        else:
            y_prob = y_prob_raw

        if threshold is None:
            if self.risk_engine is not None:
                risk_result = self.risk_engine.predict(y_prob)
                risk_levels = risk_result["risk_levels"]
                result = pd.DataFrame({
                    "probability": y_prob,
                    "risk_level": risk_levels,
                })
                if return_all:
                    return result
                return y_prob
            return y_prob

        y_pred = (y_prob >= threshold).astype(int)
        if return_all:
            result = pd.DataFrame({
                "probability": y_prob,
                "binary_prediction": y_pred,
            })
            return result
        return y_pred

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns."""
        if self.raw_columns_:
            missing = [c for c in self.raw_columns_ if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Input DataFrame missing {len(missing)} required "
                    f"column(s): {missing}"
                )

    def get_model_info(self) -> Dict[str, Any]:
        """Return summary information about the loaded model."""
        return {
            "model_type": self.metadata.get("model_type", "unknown"),
            "n_features": len(self.selected_features_),
            "selected_features": self.selected_features_,
            "has_calibrator": self.calibrator is not None,
            "has_risk_engine": self.risk_engine is not None,
            "has_vif_filter": self.vif_filter is not None,
            "metrics": self.metadata.get("metrics", {}),
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"FraudPredictor(model_type='{info['model_type']}', "
            f"n_features={info['n_features']}, "
            f"calibrated={info['has_calibrator']}, "
            f"risk_engine={info['has_risk_engine']})"
        )