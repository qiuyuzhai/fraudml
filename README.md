# FraudML — E-Commerce Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2)
![pytest](https://img.shields.io/badge/tests-32%20passed-brightgreen)

> IEEE-CIS 电商欺诈检测：**规则引擎 + ML 模型双层架构**，含数据泄漏修复、特征工程对比实验、成本优化三层风控决策、FastAPI 在线服务。

---

## Results

| Config | AUC | KS | PR-AUC | Prec@5% | Features | Notes |
|--------|-----|-----|--------|---------|----------|-------|
| **Hybrid (best)** | **0.8942** | **0.6265** | 0.5131 | 0.3812 | 1503 | Raw cols + Time + Amount features |
| Baseline (no FE) | 0.8939 | 0.6288 | 0.5153 | 0.3843 | 1494 | Raw columns only, no engineering |
| Default (strict IV) | 0.8348 | 0.5273 | 0.4384 | 0.3387 | 258 | Full FE + IV > 0.005 filter |
| Full config | 0.8237 | 0.5192 | 0.4360 | 0.3385 | 293 | All features + 5-model comparison |

**Key finding**: For tree-based models, 99% of predictive power comes from raw columns. Most handcrafted features (TargetEncoder, CrossFeature, HistoryFeature) add noise rather than signal. Only "transformations trees can't learn" (time periodicity, log scaling) provide marginal gains.

---

## Architecture

```
Transaction Request
       │
       ▼
┌───────────────────────────────────────┐
│  Layer 1: Rule Engine (millisecond)   │
│  · Blacklist match     → BLOCK        │   ← Hits directly block,
│  · Velocity check      → CHALLENGE    │   ← no model call needed
│  · Amount threshold    → CHALLENGE    │
└──────────────┬────────────────────────┘
               │ PASS (no rule hit)
               ▼
┌───────────────────────────────────────┐
│  Layer 2: ML Model (LightGBM)         │
│  · Feature pipeline (degraded mode)   │
│  · predict_proba → probability         │
│  · Three-tier risk: LOW / MED / HIGH  │
└──────────────┬────────────────────────┘
               │
               ▼
        allow / challenge_step_up / block_and_review
```

**Degraded mode**: Stateful features (HistoryFeature, AggregationFeature) require Redis for real-time historical context. When Redis is unavailable, the service flags `features_degraded=true` and scores with available stateless features only.

---

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Train (default config)
python -m src.train --config-name config_hybrid

# 3. Start online service
export MODEL_ARTIFACT_DIR=artifacts/run_YYYYMMDD_HHMMSS_xxxxxx
uvicorn src.serving.main:app --port 8000

# 4. Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"TransactionDT": 3459432, "TransactionAmt": 50, "card1": 17074}'

# 5. Batch scoring
fraudml-score --artifact-dir artifacts/run_xxx --data-source data/raw/train_transaction.parquet

# 6. Run tests
pytest
```

### Docker

```bash
# Training
docker compose --profile training up training

# Online API
docker compose --profile api up api

# MLflow tracking server
docker compose --profile mlflow up mlflow
```

---

## Key Design Decisions

### 1. Data Leakage Repair

Stateful features (TargetEncoder, HistoryFeature, AggregationFeature) must `fit()` on train only and `transform()` on validation. The initial run had train+val concatenated before fitting, leaking future statistics and labels:

| | AUC (leaked) | AUC (fixed) | Delta |
|---|---|---|---|
| Initial run | 0.9079 | 0.8939 | -0.014 |

The 1.4% AUC drop is itself proof that leakage existed.

### 2. Feature Engineering Evaluation

| Feature Type | Effect on LightGBM | Why |
|---|---|---|
| TimeFeature (hour/weekday) | +0.0003 | Trees can't extract periodicity from raw timestamp |
| AmountFeature (log/decimal) | +0.0001 | Log transform helps split points |
| TargetEncoder | -0.01 | Redundant for trees; leakage risk |
| CrossFeature | -0.005 | Trees already learn interactions |
| HistoryFeature | -0.01 | Sparse (74% of UIDs appear once) |

**Takeaway**: Don't blindly add features. For tree models, evaluate each feature's incremental value — most "standard" feature engineering techniques are designed for linear models and can hurt tree performance.

### 3. Rules + Model Two-Tier

Real fraud systems use deterministic rules before ML scoring:
- **Rules** (millisecond): blacklist, velocity, amount threshold — catch obvious fraud without model latency
- **Model** (100-200ms): LightGBM handles the "gray zone" where rules don't fire

The `/score` endpoint returns `decision_source: "rule_engine" | "model"` to make the decision path transparent.

### 4. Cost-Optimized Three-Tier Risk

Thresholds are not hardcoded — they're optimized on validation data using business cost weights:

```yaml
risk_decision:
  cost_fp: 10.0    # False positive cost (customer friction)
  cost_fn: 500.0   # False negative cost (fraud loss)
```

With a 50:1 cost ratio, the optimizer pushes thresholds low (medium=0.01, high=0.03), reflecting "better to over-block than miss fraud" — the correct trade-off for high-value fraud.

### 5. Online Degraded Mode

Stateful features need historical context (Redis sorted sets for transaction windows). Without Redis, the service still scores but sets `features_degraded=true`, so callers know the probability is from a partial feature set. This enables progressive deployment: ship the model first, add Redis later.

---

## Project Structure

```
fraudml/
├── configs/
│   ├── config.yaml              # Default (strict IV, minimal FE)
│   ├── config_hybrid.yaml       # Time+Amount features (best AUC)
│   ├── config_full.yaml         # All features + model comparison
│   └── config_baseline.yaml    # No feature engineering
├── data/
│   ├── raw/                     # IEEE-CIS parquet files
│   └── blacklist.txt            # High-fraud card1 values
├── src/
│   ├── data/                    # DataLoader, PolarsDataLoader, make_loader()
│   ├── features/                # FeatureBase ABC + 12 feature classes
│   ├── models/                  # ModelBase ABC + LightGBM/XGBoost/CatBoost wrappers
│   │   └── risk_decision.py     # Three-tier risk engine
│   ├── pipeline/                # TrainPipeline, FraudPredictor
│   ├── persistence/             # ModelSerializer (artifact save/load)
│   ├── tracker/                 # MLflow ExperimentTracker
│   ├── feature_store/           # SQLite-backed feature store (versioning + lineage)
│   ├── serving/                 # FastAPI app + config + schemas
│   ├── rules/                   # RuleEngine (Blacklist/Velocity/Amount)
│   ├── interpretability/        # SHAPExplainer
│   ├── batch_score.py           # Batch scoring CLI
│   └── train.py                 # Training entry point
├── tests/                       # 32 pytest tests
├── Dockerfile                   # Multi-stage build (builder + runtime)
├── docker-compose.yml           # training + mlflow + api services
├── pyproject.toml               # PEP 621 standard
└── README.md
```

---

## Configuration

| Config | Feature Engineering | IV Filter | AUC |
|--------|---------------------|-----------|-----|
| `config_baseline` | None (raw cols only) | No | 0.8939 |
| `config_hybrid` | Time + Amount | No | **0.8942** |
| `config` | Full FE (12 steps) | > 0.005 | 0.8348 |
| `config_full` | Full FE + model comparison | > 0.0 | 0.8237 |

```bash
# Switch config
python -m src.train --config-name config_hybrid
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/score` | Score a transaction (rules → model) |
| POST | `/explain` | Score + SHAP top features |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (model + feature store) |
| GET | `/model-info` | Model type, features, metrics |
| GET | `/rules` | List active pre-model rules |
| POST | `/admin/blacklist` | Add card1 to blacklist |

### Example Response

```json
{
  "transaction_id": 3459432,
  "probability": 0.1769,
  "risk_level": "low",
  "recommended_action": "allow",
  "decision_source": "model",
  "matched_rule": null,
  "features_degraded": true,
  "model_version": "run_20260820_232249"
}
```

---

## Testing

```bash
pytest -v
# 32 tests: features, feature_store, data, pipeline, serving
```

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/) — 590,540 transactions × 394 columns from Vesta's e-commerce platform. Train/validation split by time (80/20) to prevent temporal leakage.
