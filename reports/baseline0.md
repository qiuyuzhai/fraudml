# Baseline 实验记录

## 实验 1 — 初始 Baseline

- **日期**: 2026-08-04
- **数据**: `data/raw/train_transaction.csv` + `data/raw/train_identity.csv`
- **时间切分**: 按 TransactionDT 排序，最后 20% 作为验证集
- **Identity 合并**: 切分后分别合并，防止泄漏
- **特征**: 原始特征（无特征工程）
- **预处理**: 缺失值填充 + 标签编码

### 指标

| 模型 | AUC | KS | Precision@Top5% |
|------|-----|----|-----------------|
| Logistic Regression | 0.7742 | 0.4233 | 0.2430 |
| LightGBM | 0.9042 | 0.6472 | 0.3763 |

### 备注

- 时间切分模拟真实风控场景：用历史数据训练，用未来数据验证
- 切分后再合并 Identity 表，防止未来信息泄漏
- LightGBM 使用 is_unbalance=True 处理类别不平衡
