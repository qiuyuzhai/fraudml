"""
FraudML Training Pipeline — end-to-end, serializable training flow.

Integrates all components:
    Load → Profile → Clean → Encode → Feature Generation (Registry)
    → Feature Selection (IV/VIF) → PSI Drift Monitor
    → Model Training (with Optuna + TimeSeriesCV)
    → Threshold Optimization → Save

The pipeline is configured via a Hydra YAML config dictionary.
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from src.data import DataProfiler, DataCleaner
from src.data.loader import make_loader
from src.features import FeatureRegistry, FeatureCatalog
from src.feature_store import FeatureStore
from src.selection import IVSelector, VIFFilter
from src.monitoring import PSIMonitor
from src.evaluation import PurgedTimeSeriesSplit
from src.models import ThresholdOptimizer, RiskDecisionEngine, ModelBase, make_model
from src.calibration import (
    Calibrator,
    PlattScalingCalibrator,
    IsotonicCalibrator,
    CalibrationEvaluator,
)
from src.interpretability import SHAPExplainer
from src.comparison import ModelComparator
from src.tracker import ExperimentTracker
from src.persistence import ModelSerializer


class TrainPipeline:
    """End-to-end training pipeline for fraud detection.

    Parameters
    ----------
    cfg : dict
        Hydra configuration dictionary.  Expected keys include:

        - ``data``: data loading settings
        - ``features``: feature engineering config path
        - ``selection``: feature selection parameters
        - ``model``: model hyperparameters
        - ``cv``: cross-validation settings
        - ``threshold``: threshold optimization settings
        - ``output``: output directory
        - ``mlflow``: MLflow tracking settings

    Attributes (set after ``fit()``):
    ----------
    cleaner_ : DataCleaner
        Fitted data cleaner.
    registry_ : FeatureRegistry
        Fitted feature registry.
    iv_selector_ : IVSelector
        Fitted IV selector.
    vif_filter_ : VIFFilter
        Fitted VIF filter.
    model_ : ModelBase
        Trained model.
    threshold_optimizer_ : ThresholdOptimizer
        Fitted threshold optimizer.
    metadata_ : dict
        Pipeline metadata (feature lists, config, etc.).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        config_name = cfg.get("name", "default")

        self._config_hash = self._compute_config_hash()
        self._run_id = self._generate_run_id()

        self.output_dir = Path(cfg.get("output", {}).get("dir", "artifacts")) / self._run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validate_config(cfg)
        self._setup_logging(cfg)

        self.random_seed = cfg.get("random_seed", 42)

        self.cleaner_: Optional[DataCleaner] = None
        self.registry_: Optional[FeatureRegistry] = None
        self._feature_catalog: Optional[FeatureCatalog] = None
        self._serializer = ModelSerializer(self.output_dir)
        self.iv_selector_: Optional[IVSelector] = None
        self.vif_filter_: Optional[VIFFilter] = None
        self.selected_features_: List[str] = []
        self.model_: Optional[ModelBase] = None
        self.calibrator_: Optional[Calibrator] = None
        self.calibrator_evaluator_: Optional[CalibrationEvaluator] = None
        self.threshold_optimizer_: Optional[ThresholdOptimizer] = None
        self.risk_engine_: Optional[RiskDecisionEngine] = None
        self.shap_explainer_: Optional[SHAPExplainer] = None
        self.tree_analyzer_: Optional[TreeAnalyzer] = None
        self.model_comparator_: Optional[ModelComparator] = None
        self.metadata_: Dict[str, Any] = {} # 数据总线

        self._mlflow = None
        mlflow_cfg = cfg.get("mlflow", {})
        if mlflow_cfg.get("enabled", False):
            run_name = mlflow_cfg.get("run_name") or self._run_id
            self._mlflow = ExperimentTracker(
                experiment_name=mlflow_cfg.get("experiment_name", "fraudml"),
                tracking_uri=mlflow_cfg.get("tracking_uri"),
            )
            self._mlflow.start_run(run_name=run_name)

        self._checkpoint_dir = self.output_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._save_config_hash()

        import json
        with open(self.output_dir / "config.yaml", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, default=str, ensure_ascii=False)

        self._save_run_info()

        self.logger.info("[Pipeline] Config: %s  →  %s  (run_id: %s, hash: %s)",
                         config_name, self.output_dir, self._run_id, self._config_hash[:6])

        
    def __enter__(self) -> "TrainPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()

    # 确保config.yaml一些字段存在
    @staticmethod
    def _validate_config(cfg: Dict[str, Any]) -> None:
        essential = ["data", "features", "model"]
        missing = [k for k in essential if k not in cfg]
        if missing:
            warnings.warn(
                f"Config missing essential keys: {missing}. "
                "Pipeline may fail during fit().",
                UserWarning,
                stacklevel=2,
            )

    def _setup_logging(self, cfg: Dict[str, Any]) -> None:
        log_cfg = cfg.get("logging", {})
        log_level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
        log_file = log_cfg.get("file")

        self.logger = logging.getLogger(f"TrainPipeline-{id(self)}")
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        if self.logger.handlers:
            return

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(log_level)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    def fit(self) -> "TrainPipeline":
        """Execute the full training pipeline.

        Sequence:
            1. Load data
            2. Time split
            3. Merge identity
            4. Profile (train only)
            5. Clean (fit on train, transform both)
            6. Encode + Feature Generation
            7. Feature Selection (IV → VIF) + PSI drift monitor
            8. Model Training with TimeSeriesCV
            9. Threshold Optimization

        Returns
        -------
        self : TrainPipeline
        """
        self.logger.info("=" * 60)
        self.logger.info("FraudML — TrainPipeline.fit()")
        self.logger.info("=" * 60)

        stage1_cache = self._checkpoint_dir / "stage1_clean.parquet"
        stage2_cache = self._checkpoint_dir / "stage2_features.parquet"
        stage2_registry = self._checkpoint_dir / "stage2_registry.pkl"
        stage2_meta = self._checkpoint_dir / "stage2_meta.pkl"

        cached_hash_file = self._checkpoint_dir / "config_hash.txt"
        force_refresh = self.cfg.get("checkpoint", {}).get("force_refresh", False)
        
        # Python对象用joblib/pickle序列化，parquet是列式存储
        if force_refresh:
            self.logger.info("[Checkpoint] force_refresh=True → ignoring all caches")
            skip_to = None
        elif stage2_cache.exists() and stage2_registry.exists() and stage2_meta.exists():
            if self._validate_checkpoint_hash(cached_hash_file):
                self.logger.info("[Checkpoint] Hash match (%s) → reusing Stage 2 cache", self._config_hash[:6])
                X_train_fe = pd.read_parquet(stage2_cache)
                stage2_data = joblib.load(stage2_meta)
                self.metadata_["X_train_fe"] = X_train_fe
                self.metadata_["X_val_fe"] = stage2_data["X_val_fe"]
                self.metadata_["y_train"] = stage2_data["y_train"]
                self.metadata_["y_val"] = stage2_data["y_val"]
                self.metadata_["val_df"] = stage2_data.get("val_df")
                self.registry_ = joblib.load(stage2_registry)
                self.cleaner_ = stage2_data.get("cleaner")
                skip_to = "feature_selection"
            else:
                self.logger.info("[Checkpoint] Hash mismatch → config changed → recalculating from Stage 1")
                skip_to = None
        elif stage1_cache.exists():
            if self._validate_checkpoint_hash(cached_hash_file):
                self.logger.info("[Checkpoint] Hash match (%s) → reusing Stage 1 cache", self._config_hash[:6])
                stage1_data = joblib.load(self._checkpoint_dir / "stage1_meta.pkl")
                self.metadata_.update(stage1_data)
                self.cleaner_ = joblib.load(self._checkpoint_dir / "stage1_cleaner.pkl")
                X_train_clean = pd.read_parquet(stage1_cache)
                X_val_clean = pd.read_parquet(self._checkpoint_dir / "stage1_clean_val.parquet")
                self.metadata_["X_train_clean"] = X_train_clean
                self.metadata_["X_val_clean"] = X_val_clean
                skip_to = "feature_engineering"
            else:
                self.logger.info("[Checkpoint] Hash mismatch → config changed → recalculating from scratch")
                skip_to = None
        else:
            skip_to = None



        if skip_to is None:
            self._step_load()
            self._step_split()
            self._step_merge_identity()
            self._step_profile()
            self._step_clean()
            self._save_stage1_cache()

        if skip_to != "feature_selection":
            self._step_encode_features()
            self._save_stage2_cache()

        self._step_feature_selection()
        self._step_train_model()
        self._step_calibrate()
        self._step_threshold_optimization()
        self._step_risk_decision()
        self._step_interpretability()
        self._step_tree_analysis()
        self._step_model_comparison()

        self.save()
        self._log_mlflow_params()

        self._export_feature_importance()

        if self.cfg.get("evaluate_on_finish", True):
            try:
                self.evaluate()
            except Exception as e:
                self.logger.warning("evaluate() failed: %s", e)

        self.logger.info("[Done] Pipeline fit() completed.")
        return self
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained pipeline on the validation set.

        Returns
        -------
        dict
            Evaluation metrics (AUC, KS, Precision, Recall).
        """
        if self.model_ is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # 从数据总线获取一堆操作过后剩余的验证集的特征们，可以直接给到模型的dataframe，和验证集标签
        val_features = self.metadata_.get("val_features")
        y_val = self.metadata_.get("y_val")

        if val_features is None or y_val is None:
            raise RuntimeError("No validation data available.")

        # 模型输出的0-1间的浮点数概率值
        # 先校正，然后拿这个数值去算出最佳阈值，然后把模型输出的概率都矫正后去和阈值作比较
        best_threshold = None
        if self.threshold_optimizer_ is not None:
            best_threshold = self.threshold_optimizer_.best_threshold_

        y_prob_raw = self.model_.predict_proba(val_features)[:, 1]
        if self.calibrator_ is not None and self.calibrator_._fitted:
            y_prob = self.calibrator_.transform(y_prob_raw)
        else:
            y_prob = y_prob_raw
        threshold = best_threshold if best_threshold is not None else 0.5
        y_pred = (y_prob >= threshold).astype(int)

        auc = float(roc_auc_score(y_val, y_prob))
        pr_auc = float(average_precision_score(y_val, y_prob))
        ks = self._compute_ks(y_val, y_prob)

        precision = float(precision_score(y_val, y_pred, zero_division=0))
        recall = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))

        n_total = len(y_val)
        k_pct = self.cfg.get("evaluation", {}).get("top_k_pct", 0.05)
        n_top = max(1, int(n_total * k_pct))
        top_idx = np.argsort(y_prob)[-n_top:]
        precision_at_top_k = float(y_val[top_idx].mean())

        self.metadata_["metrics"] = {
            "auc": auc,
            "pr_auc": pr_auc,
            "ks": ks,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            f"precision_at_top_{int(k_pct * 100)}pct": precision_at_top_k,
            "threshold": float(threshold),
        }

        self.logger.info("[Evaluate] AUC=%.4f, PR-AUC=%.4f, KS=%.4f", auc, pr_auc, ks)
        self.logger.info("           Precision=%.4f, Recall=%.4f, F1=%.4f, Threshold=%.4f",
                         precision, recall, f1, float(threshold))
        self.logger.info("           Precision@Top%d%%=%.4f", int(k_pct * 100), precision_at_top_k)

        if self._mlflow:
            self._mlflow.log_metrics({
                "auc": auc,
                "pr_auc": pr_auc,
                "ks": ks,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                f"precision_at_top_{int(k_pct * 100)}pct": precision_at_top_k,
            })

        return {
            "auc": auc,
            "pr_auc": pr_auc,
            "ks": ks,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            f"precision_at_top_{int(k_pct * 100)}pct": precision_at_top_k,
        }

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Serialize the pipeline using structured persistence.

        Creates a structured artifact layout for online inference
        while also writing a legacy pipeline.pkl for backward
        compatibility.

        Parameters
        ----------
        path : str or Path, optional
            Legacy .pkl output path.  Defaults to
            ``{output_dir}/pipeline.pkl``.

        Returns
        -------
        Path
            Resolved legacy save path.
        """
        if path is None:
            path = self.output_dir / "pipeline.pkl"
        path = Path(path)

        # 训练集/验证集最终特征
        X_train_final = self.metadata_.get("X_train_final")
        X_val_final = self.metadata_.get("val_features")

        # 给人看的特征说明书，用于合规性检查
        self._export_feature_catalog()

        # 把训练好的pipeline拆成独立的组件存到online_artifacts/目录下
        self._serializer.serialize_training_outputs(
            pipeline=self,
            X_train=X_train_final,
            X_val=X_val_final,
        )

        # 把训练好的所有组件打包成一个字典，用于全量依赖场景（如离线复训）
        # 是个兜底方案，用于在没有online_artifacts/目录的情况下，也能加载到所有组件
        state = {
            "cleaner": self.cleaner_,
            "registry": self.registry_,
            "feature_catalog": self._feature_catalog,
            "iv_selector": self.iv_selector_,
            "vif_filter": self.vif_filter_,
            "selected_features": self.selected_features_,
            "model": self.model_,
            "calibrator": self.calibrator_,
            "calibrator_evaluator": self.calibrator_evaluator_,
            "threshold_optimizer": self.threshold_optimizer_,
            "risk_engine": self.risk_engine_,
            "shap_explainer": self.shap_explainer_,
            "tree_analyzer": self.tree_analyzer_,
            "model_comparator": self.model_comparator_,
            "metadata": self.metadata_,
            "cfg": self.cfg,
            "random_seed": self.random_seed,
        }

        if self.registry_ is not None:
            state["execution_order"] = self.registry_._execution_order

        joblib.dump(state, path)
        self.logger.info("[Save] Legacy pipeline serialized to %s", path)
        self.logger.info("[Save] Structured artifacts in %s/online_artifacts/", self.output_dir)

        self._log_mlflow_artifacts()
        self._maybe_register_to_model_registry()

        return path

    def _maybe_register_to_model_registry(self) -> None:
        """Register the just-saved artifact_dir to MLflow Model Registry.

        Triggered at the end of :meth:`save` when
        ``cfg['mlflow']['registry']['enabled']`` is true. Default false
        to preserve the existing training flow; opt-in via config.
        """
        registry_cfg = (self.cfg.get("mlflow", {}) or {}).get("registry", {}) or {}
        if not registry_cfg.get("enabled", False):
            return
        if self._mlflow is None or not getattr(self._mlflow, "run_id", None):
            self.logger.warning(
                "[Registry] MLflow run not active; skipping model registration."
            )
            return
        model_name = registry_cfg.get("model_name", "fraudml")
        stage = registry_cfg.get("stage", "Staging")
        try:
            version = self._mlflow.register_model(
                name=model_name,
                artifact_dir=self.output_dir,
                stage=stage,
            )
            if version:
                self.logger.info(
                    "[Registry] Registered '%s' version %s (stage=%s)",
                    model_name, version, stage,
                )
                self.metadata_["registry_model_version"] = version
        except Exception as e:
            self.logger.warning("[Registry] Registration failed: %s", e)

    def predict(
        self, df: pd.DataFrame, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Generate predictions on new data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (must contain raw transaction columns).
        threshold : float, optional
            Classification threshold for binary predictions.
            If not provided, returns probabilities.  If the pipeline
            has a learned optimal threshold (from threshold
            optimisation), that value is used when ``threshold``
            is ``None``.

        Returns
        -------
        np.ndarray
            Predicted probabilities for the positive class, or
            binary predictions when *threshold* is provided or when
            a learned threshold is available.
        """
        if self.model_ is None:
            raise RuntimeError("Model not loaded or trained.")

        self._validate_input_columns(df)

        df_clean = self.cleaner_.transform(df)
        df_encoded = self.registry_.transform_all(df_clean)
        df_selected = self.iv_selector_.transform(df_encoded)
        if self.vif_filter_ is not None:
            df_final = self.vif_filter_.transform(df_selected)
        else:
            df_final = df_selected

        y_prob_raw = self.model_.predict_proba(df_final)[:, 1]
        if self.calibrator_ is not None and self.calibrator_._fitted:
            y_prob = self.calibrator_.transform(y_prob_raw)
        else:
            y_prob = y_prob_raw

        if threshold is None:
            if (
                self.threshold_optimizer_ is not None
                and self.threshold_optimizer_.best_threshold_ is not None
            ):
                threshold = self.threshold_optimizer_.best_threshold_
            else:
                return y_prob

        return (y_prob >= threshold).astype(int)

    def trace_sample(
        self,
        df: pd.DataFrame,
        sample_index: int = 0,
        top_n_trees: int = 5,
    ) -> Dict[str, Any]:
        """Trace the decision path for a single transaction.

        Useful for answering *"why was this transaction rejected?"*
        after a prediction has been made.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data (same columns as training input).
        sample_index : int
            Row index inside *df* to trace.
        top_n_trees : int
            Number of trees to include in the detailed trace.

        Returns
        -------
        dict
            - fraud_prob: final probability
            - log_odds: raw aggregated log-odds
            - feature_contributions: ranked list of influential features
            - summary: human-readable narrative
            - trace_data: full trace object (all trees, not just top N)
        """
        if self.model_ is None:
            raise RuntimeError("Model not loaded or trained.")
        if self.tree_analyzer_ is None:
            raise RuntimeError(
                "TreeAnalyzer not available. "
                "Re-run training with tree_analysis.enabled=true to enable "
                "decision-path tracing."
            )

        self._validate_input_columns(df)

        row = df.iloc[[sample_index]].copy()
        df_clean = self.cleaner_.transform(row)
        df_encoded = self.registry_.transform_all(df_clean)
        df_selected = self.iv_selector_.transform(df_encoded)
        if self.vif_filter_ is not None:
            df_final = self.vif_filter_.transform(df_selected)
        else:
            df_final = df_selected

        sample = df_final.iloc[0]
        result = self.tree_analyzer_.trace(sample, top_n_trees=top_n_trees)

        y_prob_raw = self.model_.predict_proba(df_final)[:, 1][0]
        if self.calibrator_ is not None and self.calibrator_._fitted:
            y_prob_cal = self.calibrator_.transform(np.array([y_prob_raw]))[0]
        else:
            y_prob_cal = float(y_prob_raw)

        result["raw_prob"] = float(y_prob_raw)
        result["calibrated_prob"] = float(y_prob_cal)

        if self.risk_engine_ is not None:
            risk_info = self.risk_engine_.predict(np.array([y_prob_cal]))
            result["risk_level"] = str(risk_info["risk_levels"][0])
            result["recommended_action"] = str(risk_info["recommended_actions"][0])
            result["confidence"] = float(risk_info["confidence"][0])

        return result

    @classmethod
    # 加载pipeline.pkl全量组件，用于离线复现训练效果
    def load(cls, path: str | Path) -> "TrainPipeline":
        """Load a previously saved pipeline.

        Prefers the structured online_artifacts/ layout when available,
        falling back to the legacy monolithic .pkl.

        Parameters
        ----------
        path : str or Path
            Path to the .pkl file OR the artifact directory.

        Returns
        -------
        TrainPipeline
            Loaded pipeline instance.
        """
        path = Path(path)

        if path.is_dir():
            artifact_dir = path
            structured_dir = artifact_dir / "online_artifacts"
            if structured_dir.exists():
                return cls._load_from_structured(artifact_dir)
            legacy_path = artifact_dir / "pipeline.pkl"
            if legacy_path.exists():
                path = legacy_path
            else:
                raise FileNotFoundError(
                    f"No structured artifacts or pipeline.pkl found in {artifact_dir}"
                )

        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        state = joblib.load(Path(path))

        pipeline = cls(state["cfg"])

        # --- 修复: 加载时重置路径，这样不管用户在哪个目录调用Load，都能指向原 artifact 目录 ---
        resolved_path = path if path.is_dir() else path.parent
        pipeline.output_dir = resolved_path
        pipeline._checkpoint_dir = resolved_path / "checkpoints"
        pipeline._serializer = ModelSerializer(resolved_path)
        pipeline._run_id = resolved_path.name
        # ---------------------------------------------------                

        pipeline.cleaner_ = state["cleaner"]
        pipeline.registry_ = state["registry"]
        pipeline._feature_catalog = state.get("feature_catalog")
        pipeline.iv_selector_ = state["iv_selector"]
        pipeline.vif_filter_ = state["vif_filter"]
        pipeline.selected_features_ = state.get("selected_features", [])
        pipeline.model_ = state["model"]
        pipeline.calibrator_ = state.get("calibrator")
        pipeline.calibrator_evaluator_ = state.get("calibrator_evaluator")
        pipeline.threshold_optimizer_ = state["threshold_optimizer"]
        pipeline.risk_engine_ = state.get("risk_engine")
        pipeline.shap_explainer_ = state.get("shap_explainer")
        pipeline.tree_analyzer_ = state.get("tree_analyzer")
        pipeline.model_comparator_ = state.get("model_comparator")
        pipeline.metadata_ = state["metadata"]
        pipeline.random_seed = state.get("random_seed", 42)

        if pipeline.registry_ is not None and "execution_order" in state:
            pipeline.registry_._execution_order = state["execution_order"]

        return pipeline

    # 加载online_artifacts/目录下的核心组件们，用于线上部署，实时预测，轻量快速
    @classmethod
    def _load_from_structured(cls, artifact_dir: Path) -> "TrainPipeline":
        """Load pipeline from the structured artifact layout.

        Parameters
        ----------
        artifact_dir : Path
            Root artifact directory.

        Returns
        -------
        TrainPipeline
            Loaded pipeline instance.
        """
        serializer = ModelSerializer(artifact_dir)
        components = serializer.deserialize_for_inference()
        metadata = components["metadata"]

        pipeline = cls(metadata["config"])

        # --- 修复: 加载时重置路径，指向原 artifact 目录 ---
        pipeline.output_dir = artifact_dir
        pipeline._checkpoint_dir = artifact_dir / "checkpoints"
        pipeline._serializer = ModelSerializer(artifact_dir)
        pipeline._run_id = artifact_dir.name
        # ---------------------------------------------------

        
        pipeline.cleaner_ = components["cleaner"]
        pipeline.registry_ = components["registry"]
        pipeline._feature_catalog = None
        pipeline.iv_selector_ = components["iv_selector"]
        pipeline.vif_filter_ = components.get("vif_filter")
        pipeline.selected_features_ = components.get("selected_features", [])
        pipeline.model_ = components["model"]
        pipeline.calibrator_ = components.get("calibrator")
        pipeline.risk_engine_ = components.get("risk_engine")

        pipeline.metadata_ = metadata
        pipeline.random_seed = metadata.get("random_seed", 42)

        pipeline.logger.info("[Load] Loaded from structured artifacts in %s", artifact_dir)
        return pipeline

    def _validate_input_columns(self, df: pd.DataFrame) -> None:
        raw_columns = self.metadata_.get("raw_columns")
        if raw_columns is None:
            return

        missing = [c for c in raw_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing {len(missing)} required "
                f"column(s): {missing}"
            )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    
    # 加载数据后，先把交易数据按照时间排序，再左并身份数据，防止时间穿越式数据泄露
    def _step_load(self) -> None:
        self.logger.info("[1] Loading data ...")
        try:
            data_cfg = self.cfg.get("data", {})
            data_dir = data_cfg.get("data_dir")
            engine = data_cfg.get("engine", "pandas")
            data_format = data_cfg.get("data_format", "csv")
            loader = make_loader(engine=engine, data_dir=data_dir, data_format=data_format)
            train_txn, train_id = loader.load_train()
            self.metadata_["train_id"] = train_id
            self.metadata_["raw_train"] = train_txn

            exclude = {"isFraud", "TransactionID", "TransactionDT"}
            self.metadata_["raw_columns"] = [
                c for c in train_txn.columns if c not in exclude
            ]

            self.logger.info("    Transactions: %s", train_txn.shape)
            self.logger.info("    Identity:     %s", train_id.shape)
        except Exception as e:
            self.logger.error("Step 'load' failed: %s", e)
            raise

    def _step_split(self) -> None:
        self.logger.info("[2] Time split ...")
        try:
            train_txn = self.metadata_["raw_train"]
            train_txn = train_txn.sort_values("TransactionDT").reset_index(drop=True)

            n_total = len(train_txn)
            val_ratio = self.cfg.get("data", {}).get("val_ratio", 0.2)
            n_val = int(n_total * val_ratio)

            train_df = train_txn.iloc[:-n_val].copy()
            val_df = train_txn.iloc[-n_val:].copy()

            self.metadata_["train_df"] = train_df
            self.metadata_["val_df"] = val_df

            self.logger.info(
                "    Train: %8d  fraud=%.4f",
                len(train_df), train_df["isFraud"].mean(),
            )
            self.logger.info(
                "    Val:   %8d  fraud=%.4f",
                len(val_df), val_df["isFraud"].mean(),
            )
        except Exception as e:
            self.logger.error("Step 'split' failed: %s", e)
            raise

    def _step_merge_identity(self) -> None:
        self.logger.info("[3] Merging Identity (post-split) ...")
        try:
            train_id = self.metadata_["train_id"]
            train_df = self.metadata_["train_df"]
            val_df = self.metadata_["val_df"]

            train_df = train_df.merge(train_id, on="TransactionID", how="left")
            val_df = val_df.merge(train_id, on="TransactionID", how="left")

            self.metadata_["train_df"] = train_df
            self.metadata_["val_df"] = val_df
        except Exception as e:
            self.logger.error("Step 'merge_identity' failed: %s", e)
            raise
    
    def _step_profile(self) -> None:
        self.logger.info("[4] Profiling training features ...")
        try:
            train_df = self.metadata_["train_df"]
            drop_cols = ["TransactionID", "isFraud", "TransactionDT"]
            X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])

            profiler = DataProfiler()
            profiler.run(X_train)
        except Exception as e:
            self.logger.error("Step 'profile' failed: %s", e)
            raise

    def _step_clean(self) -> None:
        self.logger.info("[5] Cleaning ...")
        try:
            train_df = self.metadata_["train_df"]
            val_df = self.metadata_["val_df"]

            drop_cols = ["TransactionID", "isFraud"] # 不能把 TransactionDT 丢掉，因为后续的特征工程需要用到它
            X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
            X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])

            y_train = train_df["isFraud"].values
            y_val = val_df["isFraud"].values

            self.cleaner_ = DataCleaner()
            self.cleaner_.fit(X_train)
            X_train_clean = self.cleaner_.transform(X_train)
            X_val_clean = self.cleaner_.transform(X_val)

            self.metadata_["X_train_clean"] = X_train_clean
            self.metadata_["X_val_clean"] = X_val_clean
            self.metadata_["y_train"] = y_train
            self.metadata_["y_val"] = y_val

            self.logger.info(
                "    Constant cols dropped: %d", len(self.cleaner_.constant_cols_)
            )
            self.logger.info("    Output features:        %d", X_train_clean.shape[1])
        except Exception as e:
            self.logger.error("Step 'clean' failed: %s", e)
            raise

    # 严格的 train-only fit 契约：fit 仅看训练数据，transform 应用到 val，避免跨集 groupby 聚合泄露
    def _step_encode_features(self) -> None:
        self.logger.info("[6] Feature engineering ...")
        try:
            X_train_clean = self.metadata_["X_train_clean"].copy()
            X_val_clean = self.metadata_["X_val_clean"].copy()

            sel_cfg = self.cfg.get("selection", {})
            target_col = sel_cfg.get("target_col", "isFraud")

            X_train_clean[target_col] = self.metadata_["y_train"] # 贴标签，索引对齐

            feature_cfg = self.cfg.get("features", {})
            config_path = feature_cfg.get("config_path", "config.yaml")

            self.registry_ = FeatureRegistry()
            self.registry_.auto_discover("src.features")

            # 仅在 cfg['features']['steps'] 缺失时回退到 config_path 文件解析
            # 路径查找限定在 configs/ 目录与包内 configs/，避免根 config.yaml 残留引用
            resolved_path = None
            for candidate in [
                Path("configs") / config_path,
                Path(__file__).resolve().parent.parent.parent / "configs" / config_path,
            ]:
                if candidate.exists():
                    resolved_path = candidate
                    break

            override_steps = feature_cfg.get("steps")
            if override_steps is not None:
                self.logger.info("    Using feature steps from config override")
                # Hydra yields ListConfig/DictConfig; the registry's
                # _configure_steps isinstance(step, dict) check needs
                # native containers, so resolve to plain Python first.
                override_steps = OmegaConf.to_container(override_steps, resolve=True)
                self.registry_._instances.clear()
                self.registry_._execution_order = []
                self.registry_._configure_steps(override_steps)
            elif resolved_path is not None:
                self.logger.info("    Using feature config: %s", resolved_path)
                self.registry_.configure(resolved_path)
            else:
                default_steps = [
                    "TargetEncoderFeature",
                    "CategoricalEncoder",
                    "TimeFeature",
                    "AmountFeature",
                    "DeviceFeature",
                    "EmailFeature",
                    "CardFeature",
                    "AddrFeature",
                    "HistoryFeature",
                    "MissingPatternFeature",
                    "AggregationFeature",
                    "CrossFeature",
                    "SequenceFeature",
                    "GraphFeature",
                ]
                self.registry_._configure_steps(default_steps)
            # --- Downcast to save memory ---
            for col in X_train_clean.select_dtypes(include="float64").columns:
                X_train_clean[col] = X_train_clean[col].astype(np.float32)
            for col in X_train_clean.select_dtypes(include="int64").columns:
                X_train_clean[col] = X_train_clean[col].astype(np.int32)
            for col in X_val_clean.select_dtypes(include="float64").columns:
                X_val_clean[col] = X_val_clean[col].astype(np.float32)
            for col in X_val_clean.select_dtypes(include="int64").columns:
                X_val_clean[col] = X_val_clean[col].astype(np.int32)

            # --- 防泄露：fit 仅在 train 上做，val 用 train 学到的状态 transform ---
            # HistoryFeature/AggregationFeature 等有状态特征依赖时间排序，单独排序各集合
            sort_col = "TransactionDT" if "TransactionDT" in X_train_clean.columns else None
            if sort_col is not None:
                X_train_sorted = X_train_clean.sort_values(sort_col).reset_index(drop=True)
                X_val_sorted = X_val_clean.sort_values(sort_col).reset_index(drop=True)
            else:
                X_train_sorted = X_train_clean.reset_index(drop=True)
                X_val_sorted = X_val_clean.reset_index(drop=True)

            # fit_transform_all = fit_all + transform_all；只对 train 做
            self.registry_.fit_all(X_train_sorted)
            X_train_fe = self.registry_.transform_all(X_train_sorted)
            # val 用 train 已 fit 的 registry 状态单独 transform，杜绝跨集 groupby 泄露
            X_val_fe = self.registry_.transform_all(X_val_sorted)

            streaming_features = self.registry_.init_streaming_all()
            if streaming_features:
                self.logger.info("    Streaming initialized for: %s", streaming_features)

            # 防御断言：特征工程后行数必须与输入一致（未丢行/加行）
            assert len(X_train_fe) == len(X_train_clean), (
                f"Train rows changed after feature engineering: "
                f"{len(X_train_clean)} → {len(X_train_fe)}"
            )
            assert len(X_val_fe) == len(X_val_clean), (
                f"Val rows changed after feature engineering: "
                f"{len(X_val_clean)} → {len(X_val_fe)}"
            )

            # 清理临时列（target 与时间列不进模型）
            for df in [X_train_fe, X_val_fe]:
                if target_col in df.columns:
                    df.drop(columns=[target_col], inplace=True)
                if "TransactionDT" in df.columns:
                    df.drop(columns=["TransactionDT"], inplace=True)

            # 确保两集列对齐（transform_all 应产出相同列集）
            if list(X_train_fe.columns) != list(X_val_fe.columns):
                # 补缺失列（val 未出现某 category 等），保证模型输入一致
                for col in X_train_fe.columns:
                    if col not in X_val_fe.columns:
                        X_val_fe[col] = 0.0
                X_val_fe = X_val_fe[X_train_fe.columns]

            self.metadata_["X_train_fe"] = X_train_fe
            self.metadata_["X_val_fe"] = X_val_fe

            self.logger.info("    Final features: %d", X_train_fe.shape[1])

            # Feature Store 注册（开关：cfg['feature_store']['enabled']，默认 true）
            self._register_feature_store(
                X_train_fe, self.metadata_["y_train"], target_col
            )
        except Exception as e:
            self.logger.error("Step 'encode_features' failed: %s", e)
            raise

    def _register_feature_store(
        self, X_train_fe: pd.DataFrame, y_train: pd.Series, target_col: str
    ) -> None:
        """Register engineered features into the Feature Store.

        Triggered at the end of :meth:`_step_encode_features` so the
        full engineered feature matrix is available for statistics.
        Skipped entirely when ``cfg['feature_store']['enabled']`` is
        false — zero impact on the existing pandas path.

        For each feature in ``registry_._instances``:
        * ``raw_columns`` lineage = ``feature.get_input_columns()`` (or
          fallback to intersection of upstream output + raw_columns).
        * ``schema_meta`` = ``feature.get_feature_metadata()``.
        * Statistics are recorded against the train feature matrix.
        """
        fs_cfg = self.cfg.get("feature_store", {})
        if not fs_cfg.get("enabled", True):
            self.logger.info("    [FeatureStore] Disabled by config, skipping.")
            return

        db_path = fs_cfg.get("db_path", "artifacts/feature_store.db")
        store = FeatureStore(db_path)
        raw_columns_known = set(self.metadata_.get("raw_columns", []))
        target_for_iv = target_col if target_col in X_train_fe.columns else None

        registered = 0
        for name, feat in self.registry_._instances.items():
            input_cols = feat.get_input_columns() if hasattr(feat, "get_input_columns") else []
            upstream_features = []  # cross-feature lineage not derivable from registry currently
            if not input_cols and raw_columns_known:
                # Fallback: any output column whose name matches a known raw column
                input_cols = [
                    c for c in feat.get_feature_metadata().get("feature_names", [])
                    if c in raw_columns_known
                ]
            try:
                store.registry.register(
                    name,
                    entity="transaction",
                    feature_type="derived",
                    description=feat.get_feature_metadata().get("physical_meaning", "") or name,
                    raw_columns=input_cols or None,
                    upstream_features=upstream_features or None,
                    schema_meta=feat.get_feature_metadata(),
                    run_id=getattr(self, "_mlflow_run_id", None),
                )
                # record_statistics expects columns matching the feature's outputs
                feat_outputs = feat.get_feature_metadata().get("feature_names", [])
                stats_df = X_train_fe if target_for_iv else X_train_fe
                if feat_outputs:
                    try:
                        store.registry.record_statistics(
                            name, stats_df, target=target_for_iv,
                            iv_bins=self.cfg.get("selection", {}).get("iv_bins", 10),
                        )
                    except Exception as stat_err:
                        self.logger.warning(
                            "    [FeatureStore] stats failed for '%s': %s", name, stat_err
                        )
                registered += 1
            except Exception as reg_err:
                self.logger.warning(
                    "    [FeatureStore] register '%s' failed: %s", name, reg_err
                )

        self.metadata_["feature_store_db"] = db_path
        self._feature_store = store
        self.logger.info(
            "    [FeatureStore] Registered %d features to %s", registered, db_path
        )

    def _step_feature_selection(self) -> None:
        self.logger.info("[7] Feature selection (IV + VIF) ...")
        try:
            X_train_fe = self.metadata_["X_train_fe"].copy()
            y_train = self.metadata_["y_train"]

            sel_cfg = self.cfg.get("selection", {})
            model_cfg = self.cfg.get("model", {})
            model_type = model_cfg.get("type", "lightgbm")

            target_col = sel_cfg.get("target_col", "isFraud")
            self.iv_selector_ = IVSelector(
                target_col=target_col,
                threshold=sel_cfg.get("iv_threshold", 0.02),
                n_bins=sel_cfg.get("iv_bins", 10),
            )

            if target_col not in X_train_fe.columns:
                X_train_fe[target_col] = y_train

            self.iv_selector_.fit(X_train_fe)
            X_train_fe = X_train_fe.drop(columns=[target_col]) # 必须删除目标列，否则模型100%过拟合标签

            X_train_sel = self.iv_selector_.transform(X_train_fe)
            X_val_sel = self.iv_selector_.transform(self.metadata_["X_val_fe"])

            self.logger.info(
                "    IV retained:  %d features", len(self.iv_selector_.retained_features_)
            )

            iv_score_map = dict(
                zip(
                    self.iv_selector_.iv_scores_["feature"],
                    self.iv_selector_.iv_scores_["iv"],
                )
            )

            tree_models = ("lightgbm", "xgboost", "catboost")
            if model_type in tree_models:
                self.logger.info("    Tree model detected — skipping VIF (tree models handle collinearity natively)")
                self.vif_filter_ = None # 树模型跳过VIF
                X_train_final = X_train_sel.copy()
                X_val_final = X_val_sel.copy()
                self.selected_features_ = list(X_train_final.columns)
            else:
                self.vif_filter_ = VIFFilter(
                    threshold=sel_cfg.get("vif_threshold", 10.0),
                    max_iterations=sel_cfg.get("vif_max_iter", 50),
                )
                self.vif_filter_.fit(X_train_sel, iv_scores=iv_score_map)

                X_train_final = self.vif_filter_.transform(X_train_sel)
                X_val_final = self.vif_filter_.transform(X_val_sel)
                self.selected_features_ = self.vif_filter_.retained_features_

                self.logger.info(
                    "    VIF retained: %d features", len(self.vif_filter_.retained_features_)
                )
                self.logger.info(
                    "    VIF removed:  %d features", len(self.vif_filter_.removed_features_)
                )
                self.logger.info(" IV retained: %d features — 请检查IV阈值是否过高", len(self.iv_selector_.retained_features_))

            self.metadata_["X_train_final"] = X_train_final
            self.metadata_["val_features"] = X_val_final
            self.metadata_["selected_features"] = self.selected_features_

            psi_monitor = PSIMonitor(threshold=sel_cfg.get("psi_threshold", 0.25))
            psi_monitor.fit(X_train_final, X_val_final, features=self.selected_features_)
            self.metadata_["psi_report"] = psi_monitor.get_drift_report()
            assert list(X_train_final.columns) == list(X_val_final.columns), "训练/验证特征列顺序不一致！"

            self._export_feature_selection_reports()
        except Exception as e:
            self.logger.error("Step 'feature_selection' failed: %s", e)
            raise

    def _export_feature_selection_reports(self) -> None:
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.iv_selector_ is not None and self.iv_selector_.iv_scores_ is not None:
                self.iv_selector_.iv_scores_.to_csv(
                    reports_dir / "iv_scores.csv", index=False
                )
                self.logger.info("    Exported IV scores to %s", reports_dir / "iv_scores.csv")
        except Exception as e:
            self.logger.warning("    Failed to export IV scores: %s", e)

        try:
            if self.vif_filter_ is not None:
                vif_summary = self.vif_filter_.summary()
                vif_summary.to_csv(reports_dir / "vif_history.csv", index=False)
                self.logger.info("    Exported VIF history to %s", reports_dir / "vif_history.csv")
        except Exception as e:
            self.logger.warning("    Failed to export VIF history: %s", e)

        try:
            psi_report = self.metadata_.get("psi_report")
            if psi_report is not None:
                psi_report.to_csv(reports_dir / "psi_drift.csv", index=False)
                self.logger.info("    Exported PSI drift report to %s", reports_dir / "psi_drift.csv")
        except Exception as e:
            self.logger.warning("    Failed to export PSI drift report: %s", e)

    def _export_feature_importance(self) -> None:
        if self.model_ is None:
            return

        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        model_cfg = self.cfg.get("model", {})
        model_type = model_cfg.get("type", "lightgbm")

        try:
            if model_type in ("lightgbm", "optuna_lgbm") and hasattr(self.model_, "feature_importances_"):
                importances = self.model_.feature_importances_
                features = self.selected_features_
                df_imp = pd.DataFrame({
                    "feature": features,
                    "importance": importances,
                }).sort_values("importance", ascending=False)
                df_imp.to_csv(reports_dir / "feature_importance.csv", index=False)
                self.logger.info("[Feature importance] Top 10:")
                for _, row in df_imp.head(10).iterrows():
                    self.logger.info("    %s: %.4f", row["feature"], row["importance"])

                self.metadata_["feature_importance"] = df_imp
        except Exception as e:
            self.logger.warning("    Failed to export feature importance: %s", e)

    def _step_train_model(self) -> None:
        self.logger.info("[9] Model training ...")
        try:
            X_train_final = self.metadata_["X_train_final"]
            y_train = self.metadata_["y_train"]

            model_cfg = self.cfg.get("model", {})
            model_type = model_cfg.get("type", "lightgbm")

            if model_type == "lightgbm":
                self.model_ = self._train_lightgbm(X_train_final, y_train, model_cfg)
            elif model_type == "optuna_lgbm":
                self.model_ = self._train_optuna_lgbm(X_train_final, y_train, model_cfg)
            else:
                self.model_ = self._train_lightgbm(X_train_final, y_train, model_cfg)

            self.metadata_["model_type"] = model_type
        except Exception as e:
            self.logger.error("Step 'train_model' failed: %s", e)
            raise

    def _step_calibrate(self) -> None:
        cal_cfg = self.cfg.get("calibration", {})
        enabled = cal_cfg.get("enabled", False)

        if not enabled or self.model_ is None:
            if not enabled:
                self.logger.info("[Calibrate] Skipped: calibration disabled in config.")
            return

        self.logger.info("[Calibrate] Model probability calibration ...")
        try:
            val_features = self.metadata_.get("val_features")
            y_val = self.metadata_.get("y_val")

            if val_features is None or y_val is None:
                self.logger.info("[Calibrate] Skipped: no validation data available.")
                return

            method = cal_cfg.get("method", "platt")
            y_prob_raw = self.model_.predict_proba(val_features)[:, 1]

            if method == "platt":
                self.calibrator_ = PlattScalingCalibrator()
            elif method == "isotonic":
                self.calibrator_ = IsotonicCalibrator()
            else:
                self.logger.warning("[Calibrate] Unknown method '%s', falling back to platt.", method)
                self.calibrator_ = PlattScalingCalibrator()

            self.calibrator_.fit(y_prob_raw, y_val)
            y_prob_calibrated = self.calibrator_.transform(y_prob_raw)

            self.calibrator_evaluator_ = CalibrationEvaluator()
            cal_metrics = self.calibrator_evaluator_.evaluate(
                y_val, y_prob_raw, y_prob_calibrated
            )

            self.metadata_["calibration"] = {
                "method": method,
                "enabled": True,
                "metrics": cal_metrics,
            }

            self.logger.info("[Calibrate] Completed. Method: %s", method)
            self.logger.info("           Raw   Brier: %.6f, ECE: %.6f",
                             cal_metrics["brier_score_raw"], cal_metrics["ece_raw"])
            self.logger.info("           Cal   Brier: %.6f, ECE: %.6f",
                             cal_metrics.get("brier_score_calibrated", 0),
                             cal_metrics.get("ece_calibrated", 0))

            if self._mlflow:
                self._mlflow.log_metrics({
                    "brier_score_raw": cal_metrics["brier_score_raw"],
                    "ece_raw": cal_metrics["ece_raw"],
                })
                if "brier_score_calibrated" in cal_metrics:
                    self._mlflow.log_metrics({
                        "brier_score_calibrated": cal_metrics["brier_score_calibrated"],
                        "ece_calibrated": cal_metrics["ece_calibrated"],
                    })
        except Exception as e:
            self.logger.warning("[Calibrate] Failed: %s. Continuing without calibration.", e)
            self.calibrator_ = None
            self.calibrator_evaluator_ = None

    def _train_lightgbm(
        self, X: pd.DataFrame, y: np.ndarray, model_cfg: Dict[str, Any]
    ) -> ModelBase:
        params = model_cfg.get("params", {})

        pos_weight_raw = params.get("scale_pos_weight", "auto")
        if isinstance(pos_weight_raw, str) and pos_weight_raw.lower() == "auto":
            neg = float((y == 0).sum())
            pos = float((y == 1).sum())

            cost_fp = None
            cost_fn = None
            risk_cfg = self.cfg.get("risk_decision", {})
            threshold_cfg = self.cfg.get("threshold", {})
            if risk_cfg.get("enabled", False):
                cost_fp = float(risk_cfg.get("cost_fp", 10.0))
                cost_fn = float(risk_cfg.get("cost_fn", 500.0))
            elif threshold_cfg:
                cost_fp = float(threshold_cfg.get("cost_fp", 10.0))
                cost_fn = float(threshold_cfg.get("cost_fn", 500.0))

            if cost_fp is not None and cost_fn is not None and cost_fn > cost_fp:
                pos_weight = cost_fn / max(cost_fp, 1e-6)
                self.logger.info(
                    "    scale_pos_weight: %.2f (cost-based, cost_fn=%.1f / cost_fp=%.1f, neg=%.0f, pos=%.0f)",
                    pos_weight, cost_fn, cost_fp, neg, pos
                )
            else:
                pos_weight = neg / max(pos, 1.0)
                self.logger.info(
                    "    scale_pos_weight: %.2f (class-ratio fallback, neg=%.0f, pos=%.0f)",
                    pos_weight, neg, pos
                )
        else:
            pos_weight = float(pos_weight_raw)
            self.logger.info(
                "    scale_pos_weight: %.2f (manual override)",
                pos_weight,
             )
            

        default_params = {
            "is_unbalance": False,
            "scale_pos_weight": pos_weight,
            "random_state": self.random_seed,
            "verbosity": -1,
            "n_estimators": params.get("n_estimators", 500),
            "learning_rate": params.get("learning_rate", 0.05),
            "num_leaves": params.get("num_leaves", 63),
            "max_depth": params.get("max_depth", -1),
            "min_child_samples": params.get("min_child_samples", 20),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 0.1),
        }

        cv_cfg = self.cfg.get("cv", {})
        use_cv = cv_cfg.get("enabled", False)

        if use_cv:
            return self._train_with_cv(X, y, default_params, cv_cfg)
        else:
            model = make_model("lightgbm", default_params)
            model.fit(X, y)
            self.logger.info(
                "    LightGBM trained with %d features", X.shape[1]
            )
            return model

    def _train_with_cv(
        self, X: pd.DataFrame, y: np.ndarray,
        params: Dict[str, Any], cv_cfg: Dict[str, Any],
    ) -> ModelBase:
        n_splits = cv_cfg.get("n_splits", 5)
        purge_gap = cv_cfg.get("purge_gap", 0)

        splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)

        cv_aucs = []
        best_model = None

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X)):
            X_tr = X.iloc[train_idx]
            y_tr = y[train_idx]
            X_va = X.iloc[val_idx]
            y_va = y[val_idx]

            fold_model = make_model("lightgbm", params)
            fold_model.fit(X_tr, y_tr)

            y_prob = fold_model.predict_proba(X_va)[:, 1]
            fold_auc = roc_auc_score(y_va, y_prob)
            cv_aucs.append(fold_auc)

            self.logger.info("    Fold %d: AUC=%.4f", fold_idx + 1, fold_auc)

        self.logger.info(
            "    CV Mean AUC: %.4f +/- %.4f", np.mean(cv_aucs), np.std(cv_aucs)
        )

        self.metadata_["cv_aucs"] = cv_aucs

        final_params = {k: v for k, v in params.items() if k != "verbosity"}
        final_params["verbosity"] = -1
        model = make_model("lightgbm", final_params)
        model.fit(X, y)
        return model

    def _train_optuna_lgbm(
        self, X: pd.DataFrame, y: np.ndarray, model_cfg: Dict[str, Any]
    ) -> ModelBase:
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            self.logger.warning("    Optuna not available. Falling back to LightGBM default.")
            return self._train_lightgbm(X, y, model_cfg)

        cv_cfg = self.cfg.get("cv", {})
        n_trials = model_cfg.get("n_trials", 20)

        random_seed = self.random_seed

        neg_total = float((y == 0).sum())
        pos_total = float((y == 1).sum())
        pos_weight = neg_total / max(pos_total, 1.0)

        def objective(trial):
            params = {
                "is_unbalance": False,
                "scale_pos_weight": pos_weight,
                "random_state": random_seed,
                "verbosity": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "max_depth": trial.suggest_int("max_depth", -1, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }

            splitter = PurgedTimeSeriesSplit(
                n_splits=cv_cfg.get("n_splits", 3),
                purge_gap=cv_cfg.get("purge_gap", 0),
            )

            fold_aucs = []
            for train_idx, val_idx in splitter.split(X):
                X_tr = X.iloc[train_idx]
                y_tr = y[train_idx]
                X_va = X.iloc[val_idx]
                y_va = y[val_idx]

                mdl = make_model("lightgbm", params)
                mdl.fit(X_tr, y_tr)
                y_prob = mdl.predict_proba(X_va)[:, 1]
                fold_aucs.append(roc_auc_score(y_va, y_prob))

            return float(np.mean(fold_aucs))

        sampler = TPESampler(seed=random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({
            "is_unbalance": False,
            "scale_pos_weight": pos_weight,
            "random_state": random_seed,
            "verbosity": -1,
        })

        self.logger.info("    Best trial: %.4f", study.best_trial.value)
        self.logger.info("    Best params: %s", best_params)

        self.metadata_["optuna_best"] = {
            "value": study.best_trial.value,
            "params": study.best_params,
        }

        model = make_model("lightgbm", best_params)
        model.fit(X, y)
        return model

    def _step_threshold_optimization(self) -> None:
        self.logger.info("[9] Threshold optimization ...")
        try:
            val_features = self.metadata_.get("val_features")
            y_val = self.metadata_.get("y_val")

            if val_features is None or y_val is None or self.model_ is None:
                self.logger.info("    Skipped: no validation data.")
                return

            y_prob_raw = self.model_.predict_proba(val_features)[:, 1]
            if self.calibrator_ is not None and self.calibrator_._fitted:
                y_prob = self.calibrator_.transform(y_prob_raw)
            else:
                y_prob = y_prob_raw

            th_cfg = self.cfg.get("threshold", {})
            self.threshold_optimizer_ = ThresholdOptimizer(
                cost_fp=th_cfg.get("cost_fp", 10.0),
                cost_fn=th_cfg.get("cost_fn", 500.0),
            )

            result = self.threshold_optimizer_.optimize(y_val, y_prob)

            self.logger.info("    Best threshold: %.4f", result["best_threshold"])
            self.logger.info("    Min cost:       %.2f", result["best_cost"])
            self.logger.info("    Precision:      %.4f", result["best_precision"])
            self.logger.info("    Recall:         %.4f", result["best_recall"])

            if self._mlflow:
                self._mlflow.log_metrics({
                    "best_threshold": result["best_threshold"],
                    "best_cost": result["best_cost"],
                    "precision_at_threshold": result["best_precision"],
                    "recall_at_threshold": result["best_recall"],
                })
        except Exception as e:
            self.logger.error("Step 'threshold_optimization' failed: %s", e)
            raise

    def _step_risk_decision(self) -> None:
        risk_cfg = self.cfg.get("risk_decision", {})
        enabled = risk_cfg.get("enabled", False)

        if not enabled:
            self.logger.info("[RiskDecision] Skipped: risk decision engine disabled.")
            return

        self.logger.info("[RiskDecision] Multi-level risk decision engine ...")
        try:
            val_features = self.metadata_.get("val_features")
            y_val = self.metadata_.get("y_val")

            if val_features is None or y_val is None or self.model_ is None:
                self.logger.info("[RiskDecision] Skipped: no validation data.")
                return

            y_prob_raw = self.model_.predict_proba(val_features)[:, 1]
            if self.calibrator_ is not None and self.calibrator_._fitted:
                y_prob = self.calibrator_.transform(y_prob_raw)
            else:
                y_prob = y_prob_raw

            self.risk_engine_ = RiskDecisionEngine(
                cost_fp=risk_cfg.get("cost_fp", 10.0),
                cost_fn=risk_cfg.get("cost_fn", 500.0),
                medium_threshold=risk_cfg.get("medium_threshold", 0.3),
                high_threshold=risk_cfg.get("high_threshold", 0.7),
            )

            self.risk_engine_.fit(y_val, y_prob)

            rd_result = self.risk_engine_.evaluate(y_val, y_prob)

            self.metadata_["risk_decision"] = {
                "medium_threshold": self.risk_engine_.optimal_medium_threshold_,
                "high_threshold": self.risk_engine_.optimal_high_threshold_,
                "metrics": rd_result,
            }

            self.logger.info("    Medium threshold: %.4f", self.risk_engine_.optimal_medium_threshold_)
            self.logger.info("    High threshold:    %.4f", self.risk_engine_.optimal_high_threshold_)
            self.logger.info("    LOW: %d, MEDIUM: %d, HIGH: %d",
                             rd_result["n_low"], rd_result["n_medium"], rd_result["n_high"])
            self.logger.info("    Total cost: %.2f", rd_result["total_cost"])

            if self._mlflow:
                self._mlflow.log_metrics({
                    "risk_medium_threshold": self.risk_engine_.optimal_medium_threshold_,
                    "risk_high_threshold": self.risk_engine_.optimal_high_threshold_,
                    "risk_total_cost": rd_result["total_cost"],
                    "risk_n_low": rd_result["n_low"],
                    "risk_n_medium": rd_result["n_medium"],
                    "risk_n_high": rd_result["n_high"],
                })
        except Exception as e:
            self.logger.warning("[RiskDecision] Failed: %s. Continuing without risk engine.", e)
            self.risk_engine_ = None

    def _step_interpretability(self) -> None:
        interp_cfg = self.cfg.get("interpretability", {})
        enabled = interp_cfg.get("enabled", False)

        if not enabled or self.model_ is None:
            if not enabled:
                self.logger.info("[Interpretability] Skipped: interpretability disabled.")
            return

        self.logger.info("[Interpretability] SHAP-based model interpretability ...")
        try:
            X_train_final = self.metadata_.get("X_train_final")
            val_features = self.metadata_.get("val_features")

            if X_train_final is None or val_features is None:
                self.logger.info("[Interpretability] Skipped: no data available.")
                return

            max_features = interp_cfg.get("max_features", 20)
            n_samples = interp_cfg.get("n_samples", 100)

            self.shap_explainer_ = SHAPExplainer(
                model=self.model_,
                feature_names=self.selected_features_,
                max_features=max_features,
                n_samples=n_samples,
            )

            self.shap_explainer_.fit(X_train_final)

            importance_df = self.shap_explainer_.global_importance(val_features)

            self.metadata_["shap_importance"] = importance_df

            self.logger.info("    SHAP Top %d features:", min(10, len(importance_df)))
            for _, row in importance_df.head(10).iterrows():
                self.logger.info("    %s: %.6f", row["feature"], row["mean_abs_shap"])

            reports_dir = self.output_dir / "reports"
            shap_paths = self.shap_explainer_.export(val_features, str(reports_dir))

            self.logger.info("    SHAP exported: %s", shap_paths)

            if self._mlflow:
                self._mlflow.log_metrics({
                    "shap_top_feature_importance": float(importance_df.iloc[0]["mean_abs_shap"]) if len(importance_df) > 0 else 0.0,
                })

        except Exception as e:
            self.logger.warning("[Interpretability] Failed: %s. Continuing without SHAP.", e)
            self.shap_explainer_ = None

    def _step_tree_analysis(self) -> None:
        tree_cfg = self.cfg.get("tree_analysis", {})
        enabled = tree_cfg.get("enabled", False)

        if not enabled or self.model_ is None:
            if not enabled:
                self.logger.info("[TreeAnalysis] Skipped: tree analysis disabled.")
            return

        self.logger.info("[TreeAnalysis] Structure-level tree model analysis ...")
        try:
            catalog = self.metadata_.get("feature_catalog")
            self.tree_analyzer_ = TreeAnalyzer(
                model=self.model_,
                feature_names=self.selected_features_,
                feature_catalog=catalog,
            )

            self.logger.info("    %s", self.tree_analyzer_.summary())

            rules_df = self.tree_analyzer_.extract_rules(
                max_rules_per_tree=tree_cfg.get("max_rules_per_tree", 5),
            )
            self.metadata_["tree_rules"] = rules_df

            depth_df = self.tree_analyzer_.analyze_depth()
            self.metadata_["feature_depth"] = depth_df

            inter_df = self.tree_analyzer_.mine_interactions(
                min_paths=tree_cfg.get("min_paths", 10),
            )
            self.metadata_["feature_interactions"] = inter_df

            reports_dir = self.output_dir / "reports"
            export_paths = self.tree_analyzer_.export(
                output_dir=reports_dir,
                run_prefix=self._run_id,
            )

            viz_paths = self.tree_analyzer_.visualize(
                output_dir=reports_dir / "tree_viz",
                run_prefix=self._run_id,
                top_k_gain=tree_cfg.get("top_k_gain", 20),
                top_k_depth=tree_cfg.get("top_k_depth", 15),
                top_k_inter=tree_cfg.get("top_k_inter", 20),
            )

            self.logger.info("    Tree analysis exported: %s", export_paths)
            if viz_paths:
                self.logger.info("    Tree visualizations: %s", viz_paths)
                self.metadata_["tree_viz_paths"] = {str(k): str(v) for k, v in viz_paths.items()}

            self._run_decision_trace_demos(reports_dir, tree_cfg)

            if self._mlflow:
                self._mlflow.log_metrics({
                    "tree_rules_extracted": float(len(rules_df)),
                    "tree_interactions_found": float(len(inter_df)),
                })
                for plot_name, plot_path in viz_paths.items():
                    try:
                        self._mlflow.log_artifact(str(plot_path))
                    except Exception:
                        pass

        except Exception as e:
            self.logger.warning("[TreeAnalysis] Failed: %s. Continuing.", e)
            self.tree_analyzer_ = None

    def _run_decision_trace_demos(
        self,
        reports_dir: Path,
        tree_cfg: Dict[str, Any],
    ) -> None:
        trace_cfg = tree_cfg.get("trace", {})
        enabled = trace_cfg.get("enabled", True)
        if not enabled or self.tree_analyzer_ is None or self.model_ is None:
            return

        val_features = self.metadata_.get("val_features")
        y_val = self.metadata_.get("y_val")
        if val_features is None:
            return

        n_demos = trace_cfg.get("n_samples", 3)
        self.logger.info("[Trace] Decision-path tracing on top %d high-risk validation samples ...", n_demos)

        try:
            y_prob_raw = self.model_.predict_proba(val_features)[:, 1]
            if self.calibrator_ is not None and self.calibrator_._fitted:
                y_prob = self.calibrator_.transform(y_prob_raw)
            else:
                y_prob = y_prob_raw

            top_idx = np.argsort(y_prob)[-n_demos:][::-1]

            trace_dir = reports_dir / "decision_traces"
            trace_dir.mkdir(parents=True, exist_ok=True)

            trace_records: List[Dict[str, Any]] = []
            for rank, idx in enumerate(top_idx):
                idx = int(idx)
                sample = val_features.iloc[idx]
                trace_result = self.tree_analyzer_.trace(
                    sample, top_n_trees=trace_cfg.get("top_n_trees", 5)
                )

                actual_label = int(y_val[idx]) if y_val is not None else -1
                self.logger.info(
                    "  #%d (idx=%d, fraud_prob=%.4f, y_true=%d):",
                    rank + 1, idx, trace_result["fraud_prob"], actual_label,
                )
                for line in trace_result["summary"].split("\n")[:20]:
                    self.logger.info("    %s", line)
                self.logger.info("    ...")

                trace_records.append({
                    "rank": rank + 1,
                    "row_index": idx,
                    "fraud_prob": trace_result["fraud_prob"],
                    "log_odds": trace_result["log_odds"],
                    "y_true": actual_label,
                    "top_features": [str(f) for f, _ in trace_result["feature_contributions"][:5]],
                    "n_trees_traced": len(trace_result.get("all_tree_traces", [])),
                })

                trace_file = trace_dir / f"trace_{rank+1}_idx{idx}.txt"
                with open(trace_file, "w", encoding="utf-8") as f:
                    f.write(trace_result["summary"])

                trace_plot_file = trace_dir / f"trace_{rank+1}_idx{idx}_tree0.png"
                ax = self.tree_analyzer_.plot_trace(sample, tree_idx=0)
                if ax is not None:
                    import matplotlib.pyplot as plt
                    fig = ax.figure
                    fig.savefig(str(trace_plot_file), dpi=150, bbox_inches="tight")
                    plt.close(fig)

            if trace_records:
                pd.DataFrame(trace_records).to_csv(
                    trace_dir / "trace_summary.csv", index=False
                )
                self.logger.info("[Trace] Saved %d trace demo files in %s",
                                 len(trace_records), trace_dir)
                self.metadata_["decision_trace_dir"] = str(trace_dir)

        except Exception as e:
            self.logger.warning("[Trace] Decision trace demo failed: %s", e)

    def _step_model_comparison(self) -> None:
        comp_cfg = self.cfg.get("model_comparison", {})
        enabled = comp_cfg.get("enabled", False)

        if not enabled:
            self.logger.info("[ModelComparison] Skipped: model comparison disabled.")
            return

        self.logger.info("[ModelComparison] Multi-model comparison ...")
        try:
            X_train_final = self.metadata_.get("X_train_final")
            y_train = self.metadata_.get("y_train")
            val_features = self.metadata_.get("val_features")
            y_val = self.metadata_.get("y_val")

            if X_train_final is None or val_features is None:
                self.logger.info("[ModelComparison] Skipped: no data available.")
                return

            model_types = comp_cfg.get("models", ["lr", "lightgbm"])
            metrics = comp_cfg.get("metrics", ["auc", "ks", "brier", "logloss"])

            self.model_comparator_ = ModelComparator(
                model_types=model_types,
                metrics=metrics,
                random_seed=self.random_seed,
            )

            self.model_comparator_.fit(X_train_final, y_train, val_features, y_val)

            results_df = self.model_comparator_.get_results()
            self.metadata_["model_comparison"] = results_df

            if len(results_df) > 0:
                self.logger.info("    Model comparison results:")
                for _, row in results_df.iterrows():
                    model_name = row["model"]
                    metric_parts = []
                    for m in metrics:
                        if m in row.index:
                            metric_parts.append(f"{m}={row[m]:.4f}")
                    self.logger.info("    %s: %s", model_name, ", ".join(metric_parts))

                best_model = self.model_comparator_.get_best_model("auc")
                if best_model is not None:
                    best_name = results_df.loc[results_df["auc"].idxmax(), "model"]
                    self.logger.info("    Best model (AUC): %s", best_name)

                reports_dir = self.output_dir / "reports"
                comp_path = str(reports_dir / "model_comparison.csv")
                self.model_comparator_.export_results(comp_path)
                self.logger.info("    Exported comparison to %s", comp_path)

                if self._mlflow and len(results_df) > 0:
                    for _, row in results_df.iterrows():
                        model_name = row["model"]
                        for m in metrics:
                            if m in row.index:
                                self._mlflow.log_metrics({
                                    f"comparison_{model_name}_{m}": float(row[m]),
                                })

        except Exception as e:
            self.logger.warning("[ModelComparison] Failed: %s. Continuing.", e)
            self.model_comparator_ = None

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def _log_mlflow_params(self) -> None:
        if self._mlflow is None:
            return

        flat_cfg = self._flatten_dict(self.cfg)
        self._mlflow.log_params(flat_cfg)

        self._mlflow.log_params({
            "n_features_selected": len(self.metadata_.get("selected_features", [])),
            "model_type": self.metadata_.get("model_type", "unknown"),
        })

    def _log_mlflow_artifacts(self) -> None:
        if self._mlflow is None:
            return

        pipeline_path = self.output_dir / "pipeline.pkl"
        if pipeline_path.exists():
            self._mlflow.log_artifact(pipeline_path)

        reports_dir = self.output_dir / "reports"
        if reports_dir.exists():
            self._mlflow.log_artifact(reports_dir)

        online_dir = self.output_dir / "online_artifacts"
        if online_dir.exists():
            self._mlflow.log_artifact(online_dir)

        offline_dir = self.output_dir / "offline_features"
        if offline_dir.exists():
            self._mlflow.log_artifact(offline_dir)

    def _export_feature_catalog(self) -> None:
        """Build and export the FeatureCatalog for Feast compatibility."""
        if self.registry_ is None:
            return

        try:
            catalog = FeatureCatalog(name=f"fraudml_{self.cfg.get('name', 'default')}")

            for feat_name, feat_instance in self.registry_._instances.items():
                catalog.register(feat_instance)

            self._feature_catalog = catalog

            catalog_path = self.output_dir / "offline_features" / "feature_catalog.json"
            catalog.export(catalog_path)

            # 双写过渡：把 catalog 快照也灌入 Feature Store（独立于 _register_feature_store
            # 的 per-version 注册，这里保证即使编码步骤被跳过、catalog 仍可桥接进 store）
            store = getattr(self, "_feature_store", None)
            if store is not None:
                try:
                    catalog.to_feature_store(store)
                except Exception as bridge_err:
                    self.logger.warning(
                        "[FeatureCatalog] to_feature_store bridge failed: %s", bridge_err
                    )

            self.logger.info("[FeatureCatalog] Exported %d entries, %d features to %s",
                             len(catalog._entries),
                             len(catalog.get_all_feature_names()),
                             catalog_path)
        except Exception as e:
            self.logger.warning("[FeatureCatalog] Failed to export: %s", e)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten a nested dict for MLflow param logging."""
        items: Dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(TrainPipeline._flatten_dict(v, new_key, sep))
            else:
                if isinstance(v, (str, int, float, bool)):
                    items[new_key] = str(v)
        return items

    @staticmethod
    def _compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """KS statistic."""
        sorted_idx = np.argsort(y_prob)
        sorted_y = y_true[sorted_idx]
        n_pos = max((y_true == 1).sum(), 1)
        n_neg = max((y_true == 0).sum(), 1)
        cum_pos = np.cumsum(sorted_y == 1) / n_pos
        cum_neg = np.cumsum(sorted_y == 0) / n_neg
        return float(np.max(np.abs(cum_pos - cum_neg)))
    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _compute_config_hash(self) -> str:
        import json
        cfg_str = json.dumps(self.cfg, sort_keys=True, default=str)
        return hashlib.md5(cfg_str.encode()).hexdigest()
    def _generate_run_id(self) -> str:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{ts}_{self._config_hash[:6]}"

    def _save_config_hash(self) -> None:
        hash_file = self._checkpoint_dir / "config_hash.txt"
        with open(hash_file, "w") as f:
            f.write(self._config_hash)

    def _validate_checkpoint_hash(self, hash_file: Path) -> bool:
        if not hash_file.exists():
            return False
        try:
            with open(hash_file) as f:
                cached_hash = f.read().strip()
            return cached_hash == self._config_hash
        except Exception:
            return False

    def _save_run_info(self) -> None:
        import json
        from datetime import datetime
        run_info = {
            "run_id": self._run_id,
            "config_hash": self._config_hash,
            "created_at": datetime.now().isoformat(),
            "random_seed": self.random_seed,
            "force_refresh": self.cfg.get("checkpoint", {}).get("force_refresh", False),
        }
        info_path = self.output_dir / "run_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)    

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_stage1_cache(self) -> None:
        self.metadata_["X_train_clean"].to_parquet(
            self._checkpoint_dir / "stage1_clean.parquet", index=False
        )
        self.metadata_["X_val_clean"].to_parquet(
            self._checkpoint_dir / "stage1_clean_val.parquet", index=False
        )
        stage1_meta = {
            "y_train": self.metadata_["y_train"],
            "y_val": self.metadata_["y_val"],
            "val_df": self.metadata_["val_df"],
        }
        joblib.dump(stage1_meta, self._checkpoint_dir / "stage1_meta.pkl")
        joblib.dump(self.cleaner_, self._checkpoint_dir / "stage1_cleaner.pkl")
        self.logger.info("[Checkpoint] Stage 1 cached.")

    def _save_stage2_cache(self) -> None:
        df = self.metadata_["X_train_fe"]
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(
            self._checkpoint_dir / "stage2_features.parquet", index=False
        )
        val_df = self.metadata_["X_val_fe"]
        val_df = val_df.loc[:, ~val_df.columns.duplicated()]
        stage2_data = {
            "X_val_fe": val_df,
            "y_train": self.metadata_["y_train"],
            "y_val": self.metadata_["y_val"],
            "val_df": self.metadata_.get("val_df"),
            "cleaner": self.cleaner_,
        }
        joblib.dump(stage2_data, self._checkpoint_dir / "stage2_meta.pkl")
        joblib.dump(self.registry_, self._checkpoint_dir / "stage2_registry.pkl")
        self.logger.info("[Checkpoint] Stage 2 cached.")