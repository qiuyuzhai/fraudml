"""
ModelSerializer — Unified persistence for stateful pipeline components.

Replaces the monolithic pipeline.pkl with a structured directory layout:

    artifacts/{config}/
    ├── offline_features/          # Training snapshots (Hive-compatible)
    │   ├── train_features.parquet
    │   ├── val_features.parquet
    │   └── feature_catalog.json   # Feature metadata for Feast
    ├── online_artifacts/          # Online inference artifacts
    │   ├── stateful_components/   # Per-component joblib files
    │   │   ├── cleaner.joblib
    │   │   ├── CategoricalEncoder.joblib
    │   │   ├── TargetEncoderFeature.joblib
    │   │   ├── MissingPatternFeature.joblib
    │   │   ├── AggregationFeature.joblib
    │   │   ├── iv_selector.joblib
    │   │   └── vif_filter.joblib
    │   ├── model.joblib
    │   ├── calibrator.joblib      # (optional)
    │   ├── risk_engine.joblib     # (optional)
    │   └── metadata.json          # Pipeline config + feature lists
    ├── reports/                   # Analysis reports (IV/VIF/SHAP)
    └── config.yaml

This layout is designed for seamless migration to:
- Hive (offline_features → Hive tables)
- Feast (feature_catalog.json → Feast registry, online_artifacts → online store)
- Spark (FeatureBase abstraction → Spark implementation)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd


class ModelSerializer:
    """Unified persistence manager for FraudML pipelines.

    Handles split persistence of stateful components for online inference,
    while maintaining backward compatibility with the old monolithic .pkl.

    Parameters
    ----------
    artifact_dir : str | Path
        Root artifact directory (e.g. ``artifacts/run_20260813_143052_a1b2c3``).
    """

    ONLINE_STATEFUL_STEPS = [
        "cleaner",
        "CategoricalEncoder",
        "TargetEncoderFeature",
        "MissingPatternFeature",
        "AggregationFeature",
    ]

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.logger = logging.getLogger(f"ModelSerializer-{id(self)}")

        self.offline_dir = self.artifact_dir / "offline_features"
        self.online_dir = self.artifact_dir / "online_artifacts"
        self.stateful_dir = self.online_dir / "stateful_components"
        self.reports_dir = self.artifact_dir / "reports"

        self.stateful_dir.mkdir(parents=True, exist_ok=True)
        self.offline_dir.mkdir(parents=True, exist_ok=True)
        self.online_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def serialize_training_outputs(
        self,
        pipeline: Any,
        X_train: Optional[pd.DataFrame] = None,
        X_val: Optional[pd.DataFrame] = None,
    ) -> None:
        """Persist all training artifacts in the structured layout.

        Parameters
        ----------
        pipeline : TrainPipeline
            Fitted pipeline instance.
        X_train : pd.DataFrame, optional
            Final training feature DataFrame (for offline feature snapshot).
        X_val : pd.DataFrame, optional
            Final validation feature DataFrame (for offline feature snapshot).
        """
        self.logger.info("[Serialize] Saving pipeline artifacts to %s ...", self.artifact_dir)

        self._save_offline_features(X_train, X_val)
        self._save_stateful_components(pipeline)
        self._save_model_and_extras(pipeline)
        self._save_metadata(pipeline)

        self.logger.info("[Serialize] Done. Online artifacts ready for inference.")

    # 存最终的训练/验证特征快照
    def _save_offline_features(
        self,
        X_train: Optional[pd.DataFrame],
        X_val: Optional[pd.DataFrame],
    ) -> None:
        """Save offline feature snapshots (for Hive migration)."""
        try:
            if X_train is not None:
                X_train.to_parquet(
                    self.offline_dir / "train_features.parquet", index=False
                )
                self.logger.info("    Saved train_features.parquet (%d rows)", len(X_train))
            if X_val is not None:
                X_val.to_parquet(
                    self.offline_dir / "val_features.parquet", index=False
                )
                self.logger.info("    Saved val_features.parquet (%d rows)", len(X_val))
        except Exception as e:
            self.logger.warning("    Failed to save offline features: %s", e)

    # 核心状态组件
    def _save_stateful_components(self, pipeline: Any) -> None:
        """Save each stateful component as an independent joblib file.

        Only features with ``is_stateful == True`` are persisted.
        Stateless features (e.g. TimeFeature, EmailFeature) compute
        their output purely from input data and need no persistence.
        """
        saved = 0

        if pipeline.cleaner_ is not None:
            joblib.dump(pipeline.cleaner_, self.stateful_dir / "cleaner.joblib")
            saved += 1

        if pipeline.registry_ is not None:
            for name, feat in pipeline.registry_._instances.items():
                if hasattr(feat, "is_stateful") and feat.is_stateful:
                    feat.save(self.stateful_dir / f"{name}.joblib")
                    saved += 1

        if pipeline.iv_selector_ is not None:
            joblib.dump(pipeline.iv_selector_, self.stateful_dir / "iv_selector.joblib")
            saved += 1

        if pipeline.vif_filter_ is not None:
            joblib.dump(pipeline.vif_filter_, self.stateful_dir / "vif_filter.joblib")
            saved += 1

        self.logger.info("    Saved %d stateful components to %s", saved, self.stateful_dir)

    # 核心推理组件
    def _save_model_and_extras(self, pipeline: Any) -> None:
        """Save model, calibrator, risk engine as independent artifacts."""
        if pipeline.model_ is not None:
            joblib.dump(pipeline.model_, self.online_dir / "model.joblib")
            self.logger.info("    Saved model.joblib")

        if pipeline.calibrator_ is not None:
            joblib.dump(pipeline.calibrator_, self.online_dir / "calibrator.joblib")
            self.logger.info("    Saved calibrator.joblib")

        if pipeline.risk_engine_ is not None:
            joblib.dump(pipeline.risk_engine_, self.online_dir / "risk_engine.joblib")
            self.logger.info("    Saved risk_engine.joblib")

        if pipeline.threshold_optimizer_ is not None:
            joblib.dump(pipeline.threshold_optimizer_, self.online_dir / "threshold_optimizer.joblib")
            self.logger.info("    Saved threshold_optimizer.joblib")

        if pipeline.shap_explainer_ is not None:
            joblib.dump(pipeline.shap_explainer_, self.online_dir / "shap_explainer.joblib")
            self.logger.info("    Saved shap_explainer.joblib")

    # 元数据
    def _save_metadata(self, pipeline: Any) -> None:
        """Save pipeline metadata as JSON (for online inference)."""
        meta: Dict[str, Any] = {
            "config": pipeline.cfg,
            "selected_features": pipeline.selected_features_,
            "model_type": pipeline.metadata_.get("model_type", "unknown"),
            "random_seed": pipeline.random_seed,
            "raw_columns": pipeline.metadata_.get("raw_columns", []),
            "metrics": pipeline.metadata_.get("metrics", {}),
        }

        if pipeline.registry_ is not None:
            meta["execution_order"] = pipeline.registry_._execution_order

        if hasattr(pipeline, "_feature_catalog") and pipeline._feature_catalog:
            meta["feature_catalog"] = pipeline._feature_catalog.to_dict()

        if pipeline.calibrator_ is not None:
            meta["calibration"] = pipeline.metadata_.get("calibration", {})

        if pipeline.risk_engine_ is not None:
            meta["risk_decision"] = pipeline.metadata_.get("risk_decision", {})

        with open(self.online_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info("    Saved metadata.json")

    def deserialize_for_inference(self) -> Dict[str, Any]:
        """Load only the components needed for online inference.

        Returns
        -------
        dict with keys:
            - cleaner: DataCleaner instance
            - registry: FeatureRegistry instance (with loaded stateful features)
            - iv_selector: IVSelector instance
            - vif_filter: VIFFilter instance (or None)
            - model: Trained model
            - calibrator: Calibrator (or None)
            - risk_engine: RiskDecisionEngine (or None)
            - metadata: Full metadata dict
            - selected_features: List of selected feature names
        """
        self.logger.info("[Deserialize] Loading online artifacts from %s ...", self.online_dir)

        result: Dict[str, Any] = {}

        metadata_path = self.online_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.online_dir}. "
                "Run training first."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            result["metadata"] = json.load(f)

        result["selected_features"] = result["metadata"].get("selected_features", [])

        cleaner_path = self.stateful_dir / "cleaner.joblib"
        result["cleaner"] = joblib.load(cleaner_path) if cleaner_path.exists() else None

        from src.features import FeatureRegistry

        registry = FeatureRegistry()
        registry.auto_discover("src.features")

        execution_order = result["metadata"].get("execution_order", [])
        for class_name in execution_order:
            if class_name in registry._classes:
                cls = registry._classes[class_name]
                instance = cls(name=class_name)
                registry._instances[class_name] = instance
            else:
                self.logger.warning("    Feature class '%s' not discovered, skipping.", class_name)

        for name, feat in registry._instances.items():
            feat_path = self.stateful_dir / f"{name}.joblib"
            if feat_path.exists():
                feat.load(feat_path)

        registry._execution_order = execution_order
        result["registry"] = registry

        streaming_features = registry.init_streaming_all()
        if streaming_features:
            self.logger.info(
                "    Streaming initialized for: %s", streaming_features
            )

        iv_path = self.stateful_dir / "iv_selector.joblib"
        result["iv_selector"] = joblib.load(iv_path) if iv_path.exists() else None

        vif_path = self.stateful_dir / "vif_filter.joblib"
        result["vif_filter"] = joblib.load(vif_path) if vif_path.exists() else None

        model_path = self.online_dir / "model.joblib"
        result["model"] = joblib.load(model_path) if model_path.exists() else None

        cal_path = self.online_dir / "calibrator.joblib"
        result["calibrator"] = joblib.load(cal_path) if cal_path.exists() else None

        risk_path = self.online_dir / "risk_engine.joblib"
        result["risk_engine"] = joblib.load(risk_path) if risk_path.exists() else None

        self.logger.info("[Deserialize] Done. Loaded components: %s", list(result.keys()))
        return result

    def save_legacy(self, state: Dict[str, Any], path: str | Path) -> Path:
        """Backward-compatible save of the old monolithic pipeline.pkl.

        Parameters
        ----------
        state : dict
            Full pipeline state dict (same format as old save()).
        path : str | Path
            Output path.

        Returns
        -------
        Path
            Resolved save path.
        """
        path = Path(path)
        joblib.dump(state, path)
        self.logger.info("[Serialize] Legacy pipeline saved to %s", path)
        return path