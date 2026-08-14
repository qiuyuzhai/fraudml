# FraudML

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/)
[![Framework](https://img.shields.io/badge/framework-lightgbm-orange.svg)](https://lightgbm.readthedocs.io/)
[![Tracking](https://img.shields.io/badge/tracking-mlflow-red.svg)](https://mlflow.org/)

**FraudML** is an end-to-end, production-grade fraud detection pipeline built on the IEEE-CIS Fraud Detection dataset (~590K transactions). It implements the complete industrial ML workflow — from feature engineering through model deployment — with a strong emphasis on anti-leakage design, time-series-aware validation, and business-centric optimization.

---

## Key Highlights

- **Time-Series Cross-Validation** — Custom `PurgedTimeSeriesSplit` with expanding windows and purge gaps to prevent temporal leakage
- **WOE / IV Feature Screening** — Automated Information Value calculation with Chi-Merge binning for feature selection
- **PSI Drift Monitoring** — Population Stability Index tracking to detect feature distribution shift between train and production
- **Adversarial Validation** — LogisticRegression-based distribution shift detection with permutation importance to identify drifting features
- **Optuna Hyperparameter Tuning** — Automated LightGBM hyperparameter optimization integrated with time-series CV
- **MLflow Experiment Tracking** — Full parameter, metric, and artifact logging for experiment reproducibility
- **SHAP Interpretability** — Feature importance and local explanation support for fraud case analysis
- **Threshold Optimization** — Business cost minimization (FP review cost vs. FN fraud loss) instead of accuracy-based decisions
- **Feature Catalog** — Centralized feature metadata export for Feast-compatible feature registration
- **Multi-Model Comparison** — Benchmark LR / LightGBM / XGBoost / Random Forest on AUC, KS, Brier, logloss
- **Risk Decision Engine** — Multi-level (LOW/MEDIUM/HIGH) risk classification replacing single-threshold decisions
- **Model Calibration** — Platt scaling / isotonic regression for accurate probability estimation
- **Serializable Pipeline** — Stateful components persisted as independent joblib files in a structured artifact layout (`offline_features/` + `online_artifacts/`), compatible with Feast/Hive migration.

---

## Architecture

```
                              ┌─────────────────────┐
                              │  Raw Transactions   │
                              │  + Identity Data    │
                              └─────────┬───────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │  DataLoader         │
                              │  (memory optimized) │
                              └─────────┬───────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │  Time Split        │
                              │  (no random split) │
                              └─────────┬───────────┘
                                        │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                     ┌──────────────┐      ┌──────────────┐
                     │ DataCleaner │      │  Profile     │
                     │ (Winsorize)  │      │  (train-only)│
                     └──────┬───────┘      └──────────────┘
                            │
                            ▼
                     ┌──────────────────────┐
                     │  FeatureRegistry     │
                     │  (12+ feature steps) │
                     │  • Time / Amount     │
                     │  • Device / Email    │
                     │  • Card / Addr       │
                     │  • History / Missing │
                     │  • Aggregation       │
                     │  • Cross features   │
                     └──────────┬───────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
          ┌───────────────────┐   ┌───────────────────┐
          │ Adversarial Valid │   │  Feature Selection │
          │ (shift detection) │   │  • IV >= 0.02     │
          │                   │   │  • VIF < 10      │
          └─────────┬─────────┘   └─────────┬─────────┘
                    │                        │
                    └──────────┬─────────────┘
                               ▼
                    ┌───────────────────────┐
                    │  Model Training       │
                    │  • LightGBM / Optuna  │
                    │  • TimeSeriesCV      │
                    └──────────┬────────────┘
                               ▼
                    ┌───────────────────────┐
                    │  Threshold Optimizer  │
                    │  (cost minimization)  │
                    └──────────┬────────────┘
                               ▼
                    ┌───────────────────────┐
                    │  Save Pipeline (.pkl) │
                    │  + MLflow Logging     │
                    └───────────────────────┘
```

---

## Project Structure

```
fraudml/
├── configs/                     # Hydra YAML configurations
│   ├── lgb_baseline.yaml        # Fast baseline config
│   └── lgb_full.yaml            # Production config (Optuna + MLflow)
├── src/
│   ├── data/                    # Data loading, profiling, cleaning
│   ├── features/                # Feature engineering (12+ transformers)
│   │   └── feature_catalog.py   # Feature metadata for Feast compatibility
│   ├── selection/               # Feature selection (IV / VIF / PSI)
│   ├── scoring/                 # Risk scoring (WOE / IV / Binning / PSI)
│   ├── evaluation/              # Time-series cross-validation
│   ├── models/                  # Threshold optimization + risk decision
│   ├── calibration/             # Probability calibration (Platt / Isotonic)
│   ├── interpretation/          # SHAP-based model interpretability
│   ├── comparison/              # Multi-model comparison framework
│   ├── monitoring/              # Drift detection + PSI monitor
│   ├── persistence/             # Structured artifact serialization
│   │   └── serializer.py        # Splits stateful/stateless components
│   ├── pipeline/                # TrainPipeline + FraudPredictor
│   │   ├── train_pipeline.py    # Training pipeline
│   │   └── predict.py           # Standalone online inference
│   └── train.py                 # Main entry point (Hydra-driven)
├── data/
│   └── raw/                     # IEEE-CIS raw CSV files
├── reports/                     # Generated reports
└── artifacts/                   # Structured artifacts + legacy .pkl
    └── {config}/
        ├── offline_features/    # Training snapshots (→ Hive)
        ├── online_artifacts/    # Inference artifacts (→ Feast)
        │   ├── stateful_components/
        │   ├── model.joblib
        │   └── metadata.json
        ├── reports/
        └── pipeline.pkl         # Legacy monolithic save
```

---

## Quick Start

### 1. Install

```powershell
pip install -r requirements.txt
```

Or install manually:

```powershell
pip install pandas numpy scikit-learn lightgbm hydra-core omegaconf pyyaml mlflow optuna shap matplotlib statsmodels scipy
```

### 2. Download Data

Download the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset and place the CSVs in `data/raw/`:

```
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### 3. Run Training

```powershell
# Default: full production pipeline with Optuna + MLflow
python src/train.py --config-name lgb_full

# Fast baseline (no Optuna, no CV)
python src/train.py --config-name lgb_baseline

# Override any parameter via Hydra
python src/train.py --config-name lgb_baseline "model.n_trials=50" "model.learning_rate=0.03"```

### 4. Load Saved Pipeline for Inference

```python
from src.pipeline import FraudPredictor

# From structured artifacts (recommended)
predictor = FraudPredictor.from_artifact_dir("artifacts/lgb_baseline")
result = predictor.predict(new_transactions_df, return_all=True)

# Or load from legacy .pkl
from src.pipeline.train_pipeline import TrainPipeline
pipeline = TrainPipeline.load("artifacts/lgb_baseline")
scores = pipeline.predict(new_transactions_df)
```

---

## Configuration

All settings are managed through Hydra YAML configs:

| Section | Key | Description |
|---------|-----|-------------|
| `data` | `val_ratio` | Validation set ratio (time-based) |
| `selection` | `iv_threshold` | Min IV to keep a feature |
| `selection` | `vif_threshold` | Max VIF before removal |
| `model` | `type` | `lightgbm` or `optuna_lgbm` |
| `cv` | `enabled` | Enable time-series cross-validation |
| `cv` | `purge_gap` | Rows to skip between train/val |
| `threshold` | `cost_fp` | False positive review cost |
| `threshold` | `cost_fn` | False negative fraud loss |
| `mlflow` | `enabled` | Enable MLflow tracking |

---

## Resume Highlights

- **Designed and implemented an end-to-end fraud detection pipeline** processing 590K+ transactions with 30+ engineered features, featuring strict anti-leakage architecture (time-based splitting, post-split identity merging, train-only preprocessing).
- **Built industrial ML infrastructure** including PurgedTimeSeriesSplit for temporal leakage prevention, IV/VIF/PSI-based automated feature selection, adversarial validation for distribution shift detection, and Optuna-driven LightGBM hyperparameter optimization.
- **Integrated production-grade tooling**: MLflow experiment tracking, serialized pipeline artifacts for one-click inference, and business-cost-based threshold optimization (balancing manual review cost vs. fraud loss) replacing traditional accuracy metrics.