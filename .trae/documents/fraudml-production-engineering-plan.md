# FraudML 生产化工程能力建设方案

## Context

FraudML 已具备完整训练管线与 MLflow 跟踪，但推理侧与部署侧工程能力薄弱：仅有 `FraudPredictor` / `InferencePipeline` 两个 Python 类（[predict.py](file:///d:/fraudml/src/pipeline/predict.py)、[inference_pipeline.py](file:///d:/fraudml/src/pipeline/inference_pipeline.py)），无 HTTP 服务、无 MLflow Model Registry 集成（仅本地 joblib 加载）、无独立批量打分入口、不读 FeatureStore 做校验、配置全硬编码。

本方案在现有架构上补齐**十二项能力**：Feature Store、Docker 容器化、Polars 加速、Parquet 支持、项目结构标准化、FastAPI 在线服务、MLflow Model Registry 集成、批量打分 CLI、ModelBase/DecisionBase 抽象、数据泄露修复、测试目录建立、ModelSerializer 反射替代硬编码。

**项目定位**：简历项目。优先级逻辑为「面试官 5 分钟内能看懂 + 能讲出技术决策」，不是真正上生产。因此**不做** Redis 在线特征服务、在线漂移监控/告警、CI/CD、审计日志、A/B 测试——这些复杂度高且无真实流量佐证，简历讲不清楚。

**已确认的关键决策**（用户已选）：
1. **配置统一**：合并为单一 schema，以 `configs/config.yaml` 为主，废弃根 `config.yaml` 的双轨配置。
2. **Polars 集成**：提供 3 个独立 polars 替代模块 + `engine` 配置开关；FeatureBase 子类内部按 engine 委派，不改 FeatureRegistry 本身，Pandas 路径完全保留。
3. **Feature Store 集成**：`configs/config.yaml` 加 `feature_store.enabled`（默认 true）+ `db_path`；TrainPipeline.fit() 末尾按开关注册；模块可独立使用。
4. **FastAPI 先于 Redis**：在线服务先行（降级模式），历史聚合特征为 v2 路线图。理由：无 HTTP 层时在线特征服务无消费方，无法形成用户价值闭环。
5. **MLflow Model Registry 替换本地 joblib**：训练侧注册模型到 Registry，推理侧按 stage 加载，保留 `from_artifact_dir` 作为 fallback。
6. **批量打分 CLI**：`python -m src.batch_score` 一条龙，结果落 SQLite。
7. **ModelBase 抽象**：补 `src/models/base.py` + 三家 wrapper（LightGBM/XGBoost/CatBoost），统一 `fit/predict_proba/save/load/get_feature_importance` 接口，对齐 FeatureBase/SelectionBase/Calibrator 的 ABC 模式。
8. **数据泄露修复优先**：实读 [train_pipeline.py:882-909](file:///d:/fraudml/src/pipeline/train_pipeline.py#L882-L909) 确认存在 train/val concat 后聚合泄露（详见任务 10），作为 bug 修复优先于功能扩展。

**约束**：向后兼容（Pandas 路径不变）；Feature Store 可独立于 pipeline 使用；Docker 镜像 < 2GB；Polars 代码处理 Pandas 版本的所有边界情况；遵循现有代码风格（numpy docstring、`from __future__ import annotations`、类型注解）；FastAPI 复用 `FraudPredictor` 不重写特征流水线。

---

## 任务 1：Feature Store (`src/feature_store/`)

### 文件布局
```
src/feature_store/
├── __init__.py        # 导出 FeatureStore, FeatureRegistry, FeatureVersion, FeatureLineage
├── schema.sql         # SQLite DDL（包内资源，importlib.resources 读取）
├── backend.py         # SQLiteBackend：连接管理、建表迁移、CRUD 原语
├── lineage.py         # FeatureLineage：DAG 边构建、上下游查询、循环检测
├── versioning.py      # FeatureVersion + VersionManager：版本号、激活、回滚
├── registry.py        # FeatureRegistry：register/get_feature/list_features/get_lineage/archive
├── statistics.py      # 统计计算：missing_rate、distribution、IV（复用 src.scoring.iv.compute_iv）
└── store.py           # FeatureStore：门面类，组合上述组件
```

### SQLite Schema（`schema.sql`）
- `features`：name(PK)、entity、description、owner、feature_type、created_date、is_archived
- `feature_versions`：version_id(PK auto)、feature_name(FK)、version(per feature 单调)、created_date、is_active(同一 feature 仅一行=1)、schema_json、run_id；UNIQUE(feature_name, version)
- `lineage`：(version_id, source_type['raw_column'|'feature'], source_name) 三列复合主键，ON DELETE CASCADE；天然支持 DAG
- `statistics`：version_id(PK, FK)、missing_rate、iv_score、n_unique、mean、std、min_value、max_value、p50、p95、computed_at

### 类 API（核心签名）
```python
class FeatureRegistry:
    def __init__(self, db_path: str | Path = "artifacts/feature_store.db") -> None: ...
    def register(self, name, *, entity, feature_type, description="", owner="system",
                 raw_columns=None, upstream_features=None,
                 schema_meta=None, run_id=None) -> FeatureVersion: ...
    def get_feature(self, name, version: int | None = None) -> dict: ...   # None=active
    def list_features(self, entity=None, include_archived=False) -> list[dict]: ...
    def get_lineage(self, name, recursive=False) -> dict: ...
    def archive(self, name) -> None: ...
    def record_statistics(self, name, df, target=None, iv_bins=10) -> None: ...

class VersionManager:
    def next_version(self, feature_name) -> int: ...
    def activate(self, version_id) -> None: ...        # 旧 active 置 0
    def rollback(self, feature_name, to_version) -> FeatureVersion: ...
    def get_active(self, feature_name) -> FeatureVersion: ...
    def list_versions(self, feature_name) -> list[FeatureVersion]: ...

class FeatureLineage:
    def add_edges(self, version_id, sources: list[tuple[str,str]]) -> None: ...
    def get_upstream(self, feature_name, recursive=False) -> list[dict]: ...
    def get_downstream(self, feature_name) -> list[str]: ...
    def detect_cycle(self) -> bool: ...
```

`register` 流程：features 不存在则插入 → next_version → 插 feature_versions 并 activate（旧版本置 0）→ add_edges 写 raw_columns + upstream_features。
`rollback`：纯标记位翻转（activate 指定 version_id），安全可逆，不改数据。

### TrainPipeline 集成点
- **注册时机**：`_step_encode_features` 末尾（train_pipeline.py:914 附近，`X_train_fe` 写入 metadata 之后），此时全量 engineered features 已生成。
- **raw_columns lineage 获取**：在 `src/features/base.py` 新增可选方法 `get_input_columns(self) -> list[str]`（默认 `[]` 表示"全部上游输出"）。集成层回退策略：空列表时取上一步输出与 `metadata_['raw_columns']` 的交集。
- **统计**：`record_statistics` 内部 `df[col].isna().mean()` 算缺失率；`compute_iv(df, col, target, n_bins)` 算 IV（target 缺失则跳过 IV）。
- **开关**：`feature_store.enabled` 为 false 时完全不触发，零影响。
- **ModelSerializer 对接**：`_save_metadata` 追加 `meta["feature_store_db"]`；`save()` 把 db 复制到 `offline_features/` 供离线审计。

### 与 FeatureCatalog 关系：共存 + 桥接
- `FeatureCatalog`（无状态 JSON 导出）保留，不破坏现有 `metadata.json` schema 与 Feast 迁移路径。
- 在 `feature_catalog.py` 加 `to_feature_store(store) -> None` 桥接方法，把单次 catalog 快照灌入 store。
- `_export_feature_catalog` 改为「写 store + 仍导出 JSON」双写过渡。

### 独立使用示例
```python
from src.feature_store import FeatureStore
store = FeatureStore("artifacts/feature_store.db")
store.registry.register("amt_log", entity="transaction", feature_type="numeric",
    description="log1p(TransactionAmt)", raw_columns=["TransactionAmt"])
store.registry.record_statistics("amt_log", df, target="isFraud")
store.registry.get_feature("amt_log")      # 含 version+stats+lineage
store.registry.get_lineage("amt_log", recursive=True)
```

**涉及文件**：
- 新建 `src/feature_store/` 全部 8 个文件
- 修改 `src/features/base.py`（加 `get_input_columns`）
- 修改 `src/pipeline/train_pipeline.py`（`_step_encode_features` 末尾 + `save()` 钩子）
- 修改 `src/features/feature_catalog.py`（加 `to_feature_store`）
- 修改 `src/persistence/serializer.py`（`_save_metadata` 追加字段）

---

## 任务 2：Docker 容器化

### 新建文件
- **`Dockerfile`**（多阶段构建，目标 < 2GB）：
  - `builder` 阶段：`python:3.11-slim` 基础，安装 build-essential gcc，`pip install --user -r requirements.txt`（含 polars/pyarrow）
  - `runtime` 阶段：`python:3.11-slim`，`COPY --from=builder /root/.local /root/.local`，仅复制 src/ + configs/ + 入口，不复制 data/artifacts/mlruns/.venv
  - `ENTRYPOINT ["python", "src/train.py"]`，`HEALTHCHECK` 用 `python -c "import src"` 探活
- **`docker-compose.yml`**：
  - `training` 服务：build 上下文 `.`，挂载 `./data:/app/data`、`./artifacts:/app/artifacts`、`./mlruns:/app/mlruns`，env `MLFLOW_TRACKING_URI=http://mlflow:5000`
  - `mlflow` 服务：`ghcr.io/mlflow/mlflow:latest`，`ports 5000:5000`，`command mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns`
  - `postgres`（optional，profile: `postgres`）：`postgres:16-alpine`，volume 持久化，作为 MLflow backend 替代 sqlite（通过 compose profiles 按需启用）
- **`.dockerignore`**：排除 `.venv`、`__pycache__`、`data/raw/*.csv`、`artifacts/`、`mlruns/`、`outputs/`、`.git/`、`*.pkl`、`mlflow.db`

**注意**：torch/shap 较重，多阶段构建 + `--user` 安装 + slim 基础镜像控制体积。requirements.txt 加入 `polars>=0.20.0`、`pyarrow>=14.0.0`。

---

## 任务 3：Polars 迁移

### 新建 3 个独立 polars 模块
- **`src/data/loader_polars.py`**：`PolarsDataLoader`，用 `pl.scan_csv`（lazy）+ `collect()`，保留与 `DataLoader` 相同的 fit/transform/load_train/load_test 接口。处理 IEEE-CIS 的 id_* 转 category、数值 downcast（用 polars cast 表达式）。
- **`src/features/aggregation_polars.py`**：`AggregationFeaturePolars`，继承 `FeatureBase`，用 `pl.col().cum_sum().shift(1)` over group_by 实现 shift-protected 聚合，复刻 count/sum/mean/std 四个 stat 与列命名规则（`{key}_{col}_{stat}`）。
- **`src/features/time_feature_polars.py`**：`TimeFeaturePolars`，继承 `FeatureBase`，用 `pl.col().dt.hour()`、`pl.when().then()` 表达式替代 `np.select`，输出列名一致。

### engine 开关与委派
- `configs/config.yaml` 加 `data.engine: 'pandas' | 'polars'`（默认 pandas）。
- 现有 `DataLoader`（`src/data/loader.py`）增加工厂函数 `make_loader(engine, data_dir)`：polars 可用时按 engine 返回 `PolarsDataLoader`，否则 fallback 到 pandas `DataLoader`。TrainPipeline 的 `_step_load` 用此工厂。
- `AggregationFeature`/`TimeFeature` 的实际类不强制改；engine 切换在配置层选 `AggregationFeaturePolars`/`TimeFeaturePolars`（在 `features.steps` 中直接指定 polars 类名，auto_discover 会发现它们）。Pandas 类完全保留。

### Fallback 策略
- 所有 polars 模块文件顶部 `try: import polars as pl; _HAS_POLARS=True except ImportError: _HAS_POLARS=False`。
- `make_loader` 在 engine='polars' 但 polars 缺失时 `warnings.warn` 并回退 pandas。
- Polars 类的 `transform` 末尾统一 `pl_df.to_pandas()` 返回，保证下游（pandas 模型输入）一致。

### Benchmark
- 新建 `scripts/benchmark_polars.py`：对 IEEE-CIS train_transaction.csv 计时 pandas vs polars 的 load + aggregation + time transform，输出对比表。可选执行，非必须。

**涉及文件**：新建 3 个 polars 模块 + `src/data/loader.py` 加工厂函数 + `configs/config.yaml` 加 engine 字段。

---

## 任务 4：Parquet 支持

### 修改 `src/data/loader.py`
- `configs/config.yaml` 加 `data.data_format: 'csv' | 'parquet'`（默认 csv）。
- `DataLoader`（及 `PolarsDataLoader`）增加 `load_train`/`load_test` 内部按 `data_format` 选择 `.csv` 或 `.parquet`。
- 新增 `to_parquet(df, path)` / `read_parquet(path)` 便捷方法。
- **Parquet metadata 缓存**：首次读 parquet 时用 `pyarrow.parquet.read_metadata` 缓存 schema/row_count/dtype 到 `{data_dir}/.parquet_meta.json`，后续 load 先读缓存决定 dtype_map，跳过全表扫描做 fit。

### Checkpoint 复用
- train_pipeline 已用 parquet 做 checkpoint（stage1_clean.parquet 等），保持不变；DataLoader 的 parquet 支持与之对齐（都用 pyarrow 引擎）。

**涉及文件**：`src/data/loader.py`、`src/data/loader_polars.py`、`configs/config.yaml`。

---

## 任务 5：项目结构标准化

### 新建 `pyproject.toml`（PEP 621）
```toml
[project]
name = "fraudml"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [...requirements.txt 内容 + polars + pyarrow...]
[project.scripts]
fraudml-train = "src.train:main"
[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```
- 删除 `src/train.py` 与 `src/pipeline/train_pipeline.py` 中的 `sys.path.insert` hack（包安装后不再需要）。
- `requirements.txt` 保留（兼容性），新增 `requirements-dev.txt`（pytest 等）。

### `__init__.py` 标准化
- `src/__init__.py`：保留 `__version__`，增加顶层导出 `from .pipeline import TrainPipeline` 等（按需，避免循环导入则只导出 __version__ + 模块声明）。
- 各子模块 `__init__.py` 补全 `__all__`（calibration、scoring、persistence 当前缺 `__all__`）。

### 配置统一（合并单一 schema）
- **废弃根 `config.yaml`**：将其 `feature_steps` 内容迁移为 `configs/config.yaml` 的 `features.steps`（已存在该字段，仅需把根 config.yaml 的完整 step 列表合并进去，保留中文注释）。
- `configs/config.yaml` 新增段落：`data.engine`、`data.data_format`、`feature_store`。
- train_pipeline.py 的 `_step_encode_features` 简化路径查找：优先用 `cfg['features']['steps']`，回退到 `cfg['features']['config_path']`（保留以兼容外部引用），删除多路径回退中的根 `config.yaml` 引用。

**涉及文件**：新建 `pyproject.toml`、`requirements-dev.txt`；修改 `src/__init__.py`、各 `__init__.py`、`configs/config.yaml`、`src/pipeline/train_pipeline.py`、`src/train.py`（移除 sys.path hack）；删除根 `config.yaml`。

---

## 任务 6：FastAPI 在线服务

### 现状缺口
- 无 HTTP 服务层，仅有 `FraudPredictor` Python 类
- 配置全硬编码（artifact_dir 字符串传参），无 `.env`、无环境变量
- 无输入 schema 校验、无健康检查端点
- 无在线特征服务：`HistoryFeature`/`AggregationFeature` 单笔交易进来无历史上下文

### 文件布局
```
src/serving/
├── __init__.py
├── app.py            # FastAPI 应用 + lifespan（启动加载 FraudPredictor）
├── schemas.py        # Pydantic Transaction / ScoreResponse 模型
├── config.py         # Settings（pydantic-settings，从 .env 读取）
└── main.py           # uvicorn 入口
```
- `.env.example`：`MODEL_ARTIFACT_DIR=`、`MLFLOW_TRACKING_URI=`、`FEATURE_STORE_DB=`、`MODEL_NAME=`、`MODEL_STAGE=Production`

### 端点设计
- `POST /score`：接收 `Transaction`（Pydantic 模型，字段类型约束 IEEE-CIS schema）→ 返回 `{transaction_id, probability, risk_level, recommended_action, features_degraded, model_version}`
- `GET /health`：进程存活探针（liveness）
- `GET /ready`：模型已加载 + FeatureStore 可读（readiness，Docker healthcheck 用此）
- `POST /explain`：调 `FraudPredictor` → `trace_sample`，返回 SHAP 贡献 + 树路径叙事
- `GET /model-info`：返回 `get_model_info()`（模型类型、特征数、指标）

### 降级模式（核心架构决策）
- **无状态特征**（TimeFeature/AmountFeature/CategoricalEncoder/DeviceFeature 等）实时计算正确
- **有状态历史特征**（HistoryFeature/AggregationFeature）单笔交易无历史上下文 → 返回 0/NaN，模型仍能打分但偏保守
- 响应体统一加 `features_degraded: bool` 字段：当 `HistoryFeature`/`AggregationFeature` 在 `registry.execution_order` 中存在时置 true，否则 false
- **v2 路线图**：在线特征服务（Redis 存近期 N 小时交易）解决历史窗口问题，本任务不实现，仅在 README 注明

### 复用与集成
- **不重写特征流水线**：`app.py` lifespan 加载 `FraudPredictor.from_artifact_dir()` 或 `from_model_registry()`（任务 7 完成后），`/score` 内部 `predictor.predict(df)` 一行调用
- **FeatureStore 校验**：`/ready` 启动时读 `FEATURE_STORE_DB`，校验 active feature 版本与 `predictor.selected_features_` 对齐（缺失则告警但不阻断启动）
- **配置**：`pydantic-settings` 从 `.env` 读，无硬编码路径
- **docker-compose**：任务 2 的 compose 加 `api` 服务（build `.`，`ports 8000:8000`，`env_file .env`，依赖 `mlflow` 服务）

### 依赖
`requirements.txt` 追加：`fastapi>=0.110.0`、`uvicorn[standard]>=0.27.0`、`pydantic-settings>=2.0.0`

**涉及文件**：新建 `src/serving/` 5 个文件 + `.env.example`；修改 `configs/config.yaml`（加 `serving` 段）、`requirements.txt`、`docker-compose.yml`（任务 2 文件加 api 服务）。

---

## 任务 7：MLflow Model Registry 集成

### 现状缺口
- 训练侧：`ExperimentTracker` 只做 tracking（log_params/log_metrics/log_artifact），模型未注册到 Registry
- 推理侧：`FraudPredictor.from_artifact_dir` 仅读本地 `online_artifacts/model.joblib`，无版本管理、无 stage 切换、无回滚

### 训练侧集成
- `src/tracker/experiment.py` 新增方法 `register_model(name, artifact_path, stage="Staging")`：调用 `mlflow.<flavor>.log_model` + `mlflow.register_model`，把 `model.joblib`（连同 stateful_components）打包为 MLflow 模型注册到 Registry
- `TrainPipeline.save()` 末尾按 `cfg['mlflow']['registry']['enabled']` 触发注册（默认 false，避免破坏现有流程）
- `configs/config.yaml` 加 `mlflow.registry: {enabled, model_name, stage}`
- 模型打包：用 `mlflow.pyfunc` 自定义 PythonModel，封装 `FraudPredictor` 加载逻辑，使 `mlflow.<flavor>.load_model` 直接返回可调用对象

### 推理侧集成
- `FraudPredictor` 新增类方法 `from_model_registry(name, stage="Production", tracking_uri=None)`：`mlflow.<flavor>.load_model` 拉取指定 stage 模型，返回 `FraudPredictor` 实例
- 保留 `from_artifact_dir` 作为 fallback（Registry 不可用时）
- `src/serving/config.py` 优先用 `MODEL_NAME`+`MODEL_STAGE` 走 Registry，缺失时回退 `MODEL_ARTIFACT_DIR`

### Stage 切换叙事
- 训练注册时 stage=Staging，手动 `mlflow.<model>.transition_model_version_stage(name, version, "Production")` 提升到生产
- 推理侧 `MODEL_STAGE=Production` 自动拉生产版本
- 回滚：`transition_model_version_stage` 切回旧版本，无需改代码

**涉及文件**：修改 `src/tracker/experiment.py`（加 `register_model`）、`src/pipeline/train_pipeline.py`（save 末尾钩子）、`src/pipeline/predict.py`（加 `from_model_registry`）、`configs/config.yaml`（加 `mlflow.registry` 段）、`.env.example`。

---

## 任务 8：批量打分 CLI

### 现状缺口
- `FraudPredictor.predict_batch` 存在但无 CLI 入口，用户得自己写 Python
- 结果只返回 ndarray/DataFrame，不落库

### 新建 `src/batch_score.py`
- 支持 `python -m src.batch_score` 调用（`__main__` 块）
- argparse 参数：
  - `--artifact-dir` 或 `--model-name`+`--model-stage`（二选一，前者走本地 joblib，后者走 MLflow Registry）
  - `--input`（csv/parquet 路径，按扩展名自动选 reader）
  - `--output`（输出路径）
  - `--format`（`sqlite`|`csv`|`parquet`，默认 sqlite）
  - `--batch-size`（默认 10000，传给 `predict_batch`）
  - `--threshold`（可选，不传则用 risk_engine 或 0.5）
- 内部流程：加载 FraudPredictor → 读输入 → `predict_batch(return_all=True)` → 按 format 写出
- **SQLite 输出**（默认）：建表 `scores`，字段：`transaction_id, probability, risk_level, binary_prediction, model_version, scored_at`；支持 `--output existing.db` 追加
- **CSV/Parquet 输出**：直接 `to_csv`/`to_parquet`
- 日志：进度条（tqdm）+ 完成统计（总笔数、HIGH 占比、耗时）

### pyproject 注册
```toml
[project.scripts]
fraudml-train = "src.train:main"
fraudml-score = "src.batch_score:main"
```

### 简历叙事
"离线回填 + 在线推理双路径"：Cron 每日跑 `fraudml-score` 出昨日全量分数落库，业务侧查库做报表；FastAPI 处理实时交易。两种打分路径复用同一 `FraudPredictor`，保证一致性。

**涉及文件**：新建 `src/batch_score.py`；修改 `pyproject.toml`（加 console_scripts）；`requirements.txt` 加 `tqdm>=4.65.0`。

---

## 任务 9：ModelBase 抽象

### 现状缺口
- `src/models/` 仅有 `ThresholdOptimizer`、`RiskDecisionEngine`，**无 `ModelBase` 抽象基类**
- `lightgbm.LGBMClassifier` 在 [train_pipeline.py:30](file:///d:/fraudml/src/pipeline/train_pipeline.py#L30) 直接 import 实例化，无统一封装
- `requirements.txt` 含 `xgboost`/`catboost`/`lightgbm` 三家但实际只用 lightgbm，其余两家死代码
- 对比：`FeatureBase`/`SelectionBase`/`Calibrator` 三套 ABC 都已存在，唯独 Model 层缺位

### 新建 `src/models/base.py`
```python
class ModelBase(ABC):
    def __init__(self, name: str = "ModelBase") -> None:
        self.name = name
        self._fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ModelBase": ...
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]: ...
    def save(self, path: str | Path) -> None: ...   # joblib，同 FeatureBase 模式
    def load(self, path: str | Path) -> "ModelBase": ...
    @property
    def is_fitted(self) -> bool: ...
```

### 新建三家 wrapper
- `src/models/lightgbm_model.py`：`LightGBMModel(ModelBase)`，封装 `lgb.LGBMClassifier`，`predict_proba` 返回 `[:, 1]`，`get_feature_importance` 用 `feature_importances_`
- `src/models/xgboost_model.py`：`XGBoostModel(ModelBase)`，封装 `xgb.XGBClassifier`
- `src/models/catboost_model.py`：`CatBoostModel(ModelBase)`，封装 `cb.CatBoostClassifier`
- 三家统一 `scale_pos_weight='auto'` 处理（现有 train_pipeline 已有 auto 逻辑，迁移到 wrapper）

### 补 DecisionBase（任务 9 扩展）
- 新建 `src/models/decision_base.py` 定义 `DecisionBase(ABC)`，含 `fit/predict/summary` 抽象方法
- `ThresholdOptimizer`、`RiskDecisionEngine` 改继承 `DecisionBase`，统一接口
- 与 `ModelBase` 并列：ModelBase 管"打分模型"，DecisionBase 管"基于分数的决策组件"

### 工厂与配置
- `src/models/__init__.py`：导出 `ModelBase` + `DecisionBase` + 三家 wrapper + `make_model(model_type, params)` 工厂
- `configs/config.yaml` 的 `model.type` 已支持 `lightgbm|xgboost|catboost`，工厂按此返回对应 wrapper
- train_pipeline 的 `_step_train_model` 改用 `make_model()` 替代直接 import lgb

### 向后兼容
- `model_` 属性仍是 `ModelBase` 实例（而非裸 lgb），但 `predict_proba` 接口签名一致，下游 `evaluate()` / `FraudPredictor.predict` 无需改
- `pipeline.pkl` 序列化格式：lgb 原生对象 → ModelBase wrapper，旧 artifact 用 legacy load 路径回退（`_feature_catalog` 类似处理）

**涉及文件**：新建 `src/models/base.py`、`decision_base.py`、`lightgbm_model.py`、`xgboost_model.py`、`catboost_model.py`；修改 `src/models/__init__.py`、`src/models/threshold_optimizer.py`（继承 DecisionBase）、`src/models/risk_decision.py`（继承 DecisionBase）、`src/pipeline/train_pipeline.py`（`_step_train_model` 用工厂）、`configs/config.yaml`（model.params 注释补三家差异）。

---

## 任务 10：train/val concat 数据泄露修复

### 缺陷确认（实读 [train_pipeline.py:882-909](file:///d:/fraudml/src/pipeline/train_pipeline.py#L882-L909)）

**现有代码**：
```python
combined = pd.concat([X_train_clean, X_val_clean])       # L882
combined = combined.sort_values("TransactionDT").reset_index(drop=True)  # L883
combined_fe = self.registry_.fit_transform_all(combined)  # L885 在合并集上做特征工程
# ...
X_train_fe = combined_fe.iloc[:n_train].copy()            # L908
X_val_fe = combined_fe.iloc[n_train:].copy()              # L909
```

**两处泄露**：

1. **位置拆分错位（确定 bug）**：sort_values 后行顺序已变，`combined_fe.iloc[:n_train]` 取的是"排序后前 n_train 行"，里面混了 train+val（按 TransactionDT 交错），不再是 X_train。作者似乎假设 concat 后前 n_train 行还是 train——这是错的。

2. **groupby 跨集聚合（确定 bug）**：`AggregationFeature.transform` 的 `g.cumsum().shift(1)`（[aggregation_feature.py:124](file:///d:/fraudml/src/features/aggregation_feature.py#L124)）对合并全表做 groupby，train 行的历史聚合里包含 val 行的金额。`shift(1)` 只防"当前行泄露自身"，不防"val 行泄露进 train 行的历史窗口"。`HistoryFeature` 同理。

3. **现有防御断言失效**：[L892-900](file:///d:/fraudml/src/pipeline/train_pipeline.py#L892-L900) 检查"train slice 是否含 val TransactionID"——但因位置错位，这个断言**应该会触发**。代码要么没跑过 val 集，要么作者从未触发此路径。

### 修复方案
```python
# 正确做法：train 单独 fit，val 用 transform_all（FeatureBase 契约本意）
X_train_sorted = X_train_clean.sort_values("TransactionDT").reset_index(drop=True)
X_val_sorted = X_val_clean.sort_values("TransactionDT").reset_index(drop=True)

self.registry_.fit(X_train_sorted)                    # 仅 train
X_train_fe = self.registry_.transform_all(X_train_sorted)
X_val_fe = self.registry_.transform_all(X_val_sorted)  # val 用 train fit 的状态

# 删除 L882-885 的 concat + sort + fit_transform_all(combined)
# 删除 L892-900 的错位防御断言（不再需要）
# 删除 L908-909 的 iloc 位置拆分
```

### 设计一致性修复
现有 concat 做法的"理由"可能是让 `HistoryFeature`/`AggregationFeature` 看到"全部数据"。但这违背了 FeatureBase 的 fit/transform 契约——在线推理时单笔交易也只有"当前 registry 状态"，训练时却给完整 DataFrame，**训练/推理语义不一致**。修复后训练与推理走同一 `transform_all` 路径，语义对齐。

### 验证
- 修复前后跑同一 config，对比 `X_train_fe` 的 `card1_TransactionAmt_sum` 列：修复后 train 行的聚合值不应包含任何 val 行金额
- 跑 `python src/train.py` 全流程，AUC/PR-AUC 应有合理变化（泄露修复通常导致指标略降，但更真实）
- 断言 `combined_fe.iloc[:n_train]` 的 TransactionID 与 X_train 完全一致（位置正确性）

**涉及文件**：`src/pipeline/train_pipeline.py`（`_step_encode_features` 重写 L878-917）。

---

## 任务 11：测试目录建立 + pytest 化

### 现状缺口
- 无 `tests/` 目录，测试散落在各 feature 文件 `if __name__ == "__main__":` 块（如 [aggregation_feature.py:218-297](file:///d:/fraudml/src/features/aggregation_feature.py#L218-L297)、[time_feature.py:99-122](file:///d:/fraudml/src/features/time_feature.py#L99-L122)）
- 无 `conftest.py`、无 `pytest.ini`、无 `requirements-dev.txt`
- 简历"有测试覆盖"是基本盘，面试官常问"怎么验证的"

### 文件布局
```
tests/
├── __init__.py
├── conftest.py              # 共享 fixture：sample_df、tmp_artifact_dir、tmp_feature_store_db
├── features/
│   ├── __init__.py
│   ├── test_aggregation_feature.py    # 从 aggregation_feature.py __main__ 迁移
│   ├── test_time_feature.py            # 从 time_feature.py __main__ 迁移
│   ├── test_history_feature.py
│   └── test_encoding.py
├── data/
│   ├── test_loader.py
│   └── test_cleaner.py
├── feature_store/
│   ├── test_registry.py
│   ├── test_versioning.py
│   └── test_lineage.py
├── pipeline/
│   └── test_train_pipeline.py          # 任务 10 泄露修复的回归测试
└── serving/
    └── test_api.py                     # FastAPI 端点测试（任务 6）
```

### 迁移策略
- 把各 feature 文件 `__main__` 块里的 `test_*` 函数原样迁为 pytest 测试函数（断言不变，去掉 `if __name__ == "__main__":` 调用块）
- 源文件 `__main__` 块保留或删除：倾向删除（避免双份维护），README 注明用 `pytest tests/` 跑
- 新增任务 10 的回归测试：`tests/pipeline/test_train_pipeline.py` 用交错 TransactionDT 样本断言 train 行聚合值不含 val 金额（即之前验证脚本的内容，正式化）

### 配置
- 新建 `requirements-dev.txt`：`pytest>=7.4.0`、`pytest-cov>=4.1.0`
- `pyproject.toml` 加 `[tool.pytest.ini_options]` 段：`testpaths = ["tests"]`、`addopts = "-v --tb=short"`
- `conftest.py` 提供 `sample_df`、`tmp_artifact_dir`（tmp_path fixture）、`tmp_feature_store_db` fixture

### 覆盖范围
- 优先迁移已有 `__main__` 测试的模块（aggregation/time/encoding）
- 任务 1/10 完成后补 feature_store 与泄露修复的测试
- 服务层测试（test_api.py）在任务 6 后补
- **不追求全覆盖率**：简历项目有核心模块测试即可，标注关键路径覆盖

**涉及文件**：新建 `tests/` 全部文件 + `requirements-dev.txt`；修改 `pyproject.toml`（加 pytest 配置）；删除各 feature 文件 `__main__` 块（可选）。

---

## 任务 12：ModelSerializer 运行时反射替代硬编码 step 列表

### 现状缺口
- [serializer.py:56-62](file:///d:/fraudml/src/persistence/serializer.py#L56-L62) 的 `ONLINE_STATEFUL_STEPS` 是硬编码列表：`["cleaner", "CategoricalEncoder", "TargetEncoderFeature", "MissingPatternFeature", "AggregationFeature"]`
- 但 `_save_stateful_components`（[L139-143](file:///d:/fraudml/src/persistence/serializer.py#L139-L143)）实际已经用 `hasattr(feat, "is_stateful") and feat.is_stateful` 反射判断，常量是死代码
- 新增 stateful 特征（如任务 3 的 polars 变体）需检查是否要更新常量——容易遗漏

### 修复方案
- 删除 `ONLINE_STATEFUL_STEPS` 常量
- 确认 `_save_stateful_components` 完全靠 `feat.is_stateful` 反射（已经是）
- `deserialize_for_inference`（[L242-262](file:///d:/fraudml/src/persistence/serializer.py#L242-L262)）已按 `execution_order` 重建 + 检查 stateful_components 目录下文件存在性加载，不依赖常量——确认无引用后删常量
- grep 全项目确认 `ONLINE_STATEFUL_STEPS` 无其他引用点

### 价值
- 删死代码，强化"配置/反射驱动"叙事
- 新增 stateful 特征零配置自动持久化
- 工作量极小（删几行 + grep 验证）

**涉及文件**：`src/persistence/serializer.py`（删 `ONLINE_STATEFUL_STEPS` 常量）；全项目 grep 确认无引用。

---

## 实施顺序

按「bug 修复 → 基础设施 → 功能扩展 → 部署」分层：

**第一层：bug 修复（最优先，避免在错误基础上构建）**
1. **任务 10（泄露修复）**：train/val concat 跨集聚合泄露。先修，否则后续所有训练结果都基于污染数据，Feature Store 注册的统计/IV 也是错的。
2. **任务 12（反射替代硬编码）**：删 `ONLINE_STATEFUL_STEPS` 死代码。工作量极小，与任务 10 同层顺手清理。

**第二层：基础设施（后续功能依赖）**
3. **任务 5（项目结构）**：`pyproject.toml` + `__init__.py` 标准化 + 配置统一。为后续所有模块提供干净包结构。
4. **任务 9（ModelBase + DecisionBase）**：补 ABC + 三家 wrapper + DecisionBase。任务 7（Registry 加载模型）依赖统一的 ModelBase 接口。
5. **任务 1（Feature Store）**：SQLite 元数据库。任务 6（FastAPI ready 校验）依赖它。

**第三层：数据层增强（独立模块，可并行）**
6. **任务 4（Parquet）**：DataLoader 扩展。任务 3 的 polars loader 复用其 data_format 开关。
7. **任务 3（Polars）**：3 个 polars 模块 + engine 开关。依赖任务 4 的 data_format 字段。

**第四层：模型生命周期**
8. **任务 7（MLflow Registry）**：训练侧注册 + 推理侧 `from_model_registry`。依赖任务 9 的 ModelBase（打包为 pyfunc 需要统一接口）。

**第五层：服务层（消费前序所有能力）**
9. **任务 6（FastAPI）**：依赖 `FraudPredictor`（含任务 7 Registry 加载）+ FeatureStore（任务 1）做 ready 校验。
10. **任务 8（批量打分 CLI）**：依赖 `FraudPredictor`（含 Registry 加载）+ 任务 7 的双加载路径。

**第六层：测试（在功能稳定后补，避免随实现反复改）**
11. **任务 11（测试目录）**：迁移已有 `__main__` 测试为 pytest + 补任务 1/6/10 的回归测试。放此层是因为测试对象（Feature Store/FastAPI/泄露修复）需先实现。

**第七层：容器化**
12. **任务 2（Docker）**：依赖 requirements（含 polars/pyarrow/fastapi/pytest）与包结构稳定。最后做。

---

## 验证方案

1. **泄露修复（任务 10）**：修复前后跑同一 config，对比 `X_train_fe` 的 `card1_TransactionAmt_sum` 列——修复后 train 行聚合值不含任何 val 行金额；`combined_fe.iloc[:n_train]` 的 TransactionID 与 X_train 完全一致（位置正确性断言）；AUC/PR-AUC 应有合理变化（略降但更真实）。
2. **反射替代硬编码（任务 12）**：`grep -r "ONLINE_STATEFUL_STEPS" src/` 返回空；现有训练 + 推理全流程仍跑通，stateful 特征（TargetEncoder/Aggregation 等）仍被正确持久化与加载。
3. **包安装**：`pip install -e .` 成功；`python -c "import src; print(src.__version__)"` 输出 0.1.0；`python -c "from src.feature_store import FeatureStore"` 无错。
4. **ModelBase（任务 9）**：`from src.models import make_model; m = make_model("lightgbm", {"n_estimators":10}); m.fit(X, y); m.predict_proba(X)` 跑通；`make_model("xgboost")`、`make_model("catboost")` 同样；`get_feature_importance()` 返回 dict；`ThresholdOptimizer`/`RiskDecisionEngine` 是 `DecisionBase` 子类。
5. **Feature Store 独立**：跑「独立使用示例」脚本，验证 register→get_feature→record_statistics→get_lineage→archive→rollback 全链路，检查 SQLite 表数据正确。
6. **TrainPipeline 集成**：`python src/train.py --config-name config "feature_store.enabled=true"`，训练完成后查 `artifacts/feature_store.db` 有特征记录、lineage、statistics；`feature_store.enabled=false` 时行为与改造前一致（pandas 路径不变）。
7. **Polars**：`python src/train.py "data.engine=polars" "data.data_format=parquet"` 端到端跑通；`scripts/benchmark_polars.py` 输出 polars 加速比；polars 缺失时 fallback 到 pandas 不报错。
8. **Parquet**：DataLoader 读写 parquet 往返一致；`.parquet_meta.json` 缓存生效。
9. **Docker**：`docker build -t fraudml .` 成功且镜像 < 2GB（`docker images` 查看）；`docker compose up mlflow training` 启动正常；healthcheck 通过。
10. **MLflow Registry**：训练后 `mlflow.<model>.search_model_versions("name='fraudml'")` 能查到 Staging/Production 版本；`FraudPredictor.from_model_registry("fraudml", stage="Production")` 加载成功；`transition_model_version_stage` 切版本后推理结果随版本变化。
11. **FastAPI**：`uvicorn src.serving.main:app` 启动；`curl -X POST localhost:8000/score -d @sample_tx.json` 返回 probability+risk_level+features_degraded；`/ready` 在 FeatureStore 缺失时告警但不 500；`/explain` 返回 SHAP 贡献；`docker compose up api` 起来后端到端打分成功。
12. **批量打分 CLI**：`python -m src.batch_score --artifact-dir artifacts/run_xxx --input data/raw/test_transaction.csv --output outputs/scores.db` 跑通，SQLite `scores` 表有数据且字段完整；`--format csv` 输出文件可读；`--model-name fraudml --model-stage Production` 走 Registry 路径成功。
13. **测试套件（任务 11）**：`pytest tests/ -v` 全绿；`pytest --cov=src` 输出核心模块覆盖率（aggregation/time/encoding/loader/feature_store 关键路径覆盖）；任务 10 的回归测试在 `tests/pipeline/test_train_pipeline.py` 跑通。
14. **向后兼容**：不改任何配置跑现有 pandas+csv 训练（任务 10 修复除外），结果与改造前一致；`FraudPredictor.from_artifact_dir` 旧路径仍可用。
