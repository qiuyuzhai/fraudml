# Baseline 实验记录 (Refactored Pipeline)

## 实验 — 模块化重构 Baseline

- **日期**: 2026-08-06
- **数据**: `data/raw/train_transaction.csv` + `data/raw/train_identity.csv`
- **时间切分**: 按 TransactionDT 排序，最后 20% 作为验证集
- **Identity 合并**: 切分后分别合并，防止泄漏
- **预处理**: DataCleaner (Winsorization + 缺失值标志) + CategoricalEncoder
- **特征工程**: FeatureRegistry 驱动，config.yaml 配置执行顺序

### 指标

| 模型 | AUC | KS | Precision@Top5% |
|------|-----|----|-----------------|
| Logistic Regression | 0.8069 | 0.4845 | 0.2593 |
| LightGBM | 0.9044 | 0.6544 | 0.3788 |

### 流水线步骤

1. **Load** — DataLoader 内存优化 (dtype downcast)
2. **Profile** — DataProfiler 统计特征分布 (仅训练集)
3. **Clean** — DataCleaner 常量列删除 + Winsorization + 缺失值/剪辑标志
4. **Encode** — CategoricalEncoder 标签编码 (未见过类别 → -1)
5. **Train** — LogisticRegression + LightGBM(is_unbalance=True)

### 防泄漏保证

- Identity 表在时间切分后合并
- DataCleaner.fit() 仅在训练集上调用
- CategoricalEncoder.fit() 仅在训练集上调用
- Profile 仅在训练集上计算
