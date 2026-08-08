"""
FraudML — Refactored Baseline Pipeline

High-level flow:
    Load → Time Split → Merge Identity → Profile → Clean → Encode → Train → Evaluate

Usage: python src/train.py

Design notes
------------
* Time split: sort by TransactionDT, last 20% as validation (no train_test_split).
* Identity merge happens AFTER split to prevent leakage.
* Profile & Clean fit on training data only — no validation statistics
  leak into preprocessing.
* FeatureRegistry drives encoding via config.yaml; auto_discover finds
  all FeatureBase subclasses under src.features.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from src.data import DataLoader, DataProfiler, DataCleaner
from src.features import FeatureRegistry


def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS statistic — max |F_pos(prob) - F_neg(prob)|."""
    sorted_idx = np.argsort(y_prob)
    sorted_y = y_true[sorted_idx]

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    cum_pos = np.cumsum(sorted_y == 1) / n_pos
    cum_neg = np.cumsum(sorted_y == 0) / n_neg

    return float(np.max(np.abs(cum_pos - cum_neg)))


def compute_precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.05) -> float:
    """Precision@TopK% — fraction of positives in the top-k predicted samples."""
    n = len(y_true)
    top_n = max(int(n * k), 1)
    top_idx = np.argsort(y_prob)[-top_n:]
    return float(y_true[top_idx].mean())


def main() -> None:
    print("=" * 60)
    print(" FraudML — Refactored Baseline Pipeline")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────
    print("\n[1] Loading data with memory optimization ...")
    loader = DataLoader()
    train_txn, train_id = loader.load_train()
    print(f"    Transaction: {train_txn.shape}")
    print(f"    Identity:     {train_id.shape}")

    # ── 2. Time split (80% train / 20% val) ────────────────────
    print("\n[2] Time split — last 20% as validation ...")
    train_txn = train_txn.sort_values("TransactionDT").reset_index(drop=True)

    n_total = len(train_txn)
    n_val = int(n_total * 0.2)

    train_df = train_txn.iloc[:-n_val].copy()
    val_df = train_txn.iloc[-n_val:].copy()

    print(f"    Train: {len(train_df):>8}  fraud={train_df['isFraud'].mean():.4f}")
    print(f"    Val:   {len(val_df):>8}  fraud={val_df['isFraud'].mean():.4f}")

    # ── 3. Merge Identity AFTER split (no leakage) ─────────────
    print("\n[3] Merging Identity (post-split, no leakage) ...")
    train_df = train_df.merge(train_id, on="TransactionID", how="left")
    val_df = val_df.merge(train_id, on="TransactionID", how="left")

    # ── 4. Separate features / target ──────────────────────────
    drop_cols = ["TransactionID", "isFraud", "TransactionDT"]
    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values

    X_train = train_df.drop(columns=drop_cols)
    X_val = val_df.drop(columns=drop_cols)

    print(f"\n[4] Feature matrix: {X_train.shape[1]} columns")

    # ── 5. Profile (train only) ────────────────────────────────
    print("[5] Profiling training features ...")
    profiler = DataProfiler()
    profiler.run(X_train)
    print("    → reports/feature_profile.csv")

    # ── 6. Clean (fit on train, transform both) ────────────────
    print("[6] Cleaning (Winsorization + missing flags) ...")
    cleaner = DataCleaner()
    cleaner.fit(X_train)
    X_train_clean = cleaner.transform(X_train)
    X_val_clean = cleaner.transform(X_val)
    cleaner.save_summary()

    print(f"    Constant cols dropped: {len(cleaner.constant_cols_)}")
    print(f"    Numeric cols cleaned:  {len(cleaner.numeric_cols_)}")
    print(f"    Missing-value flags:    {len(cleaner.cols_with_missing_)}")
    print(f"    Output features:        {X_train_clean.shape[1]}")
    print("    → reports/cleaning_summary.csv")

    # ── 7. Feature engineering via Registry ────────────────────
    print("[7] Feature engineering via FeatureRegistry ...")
    registry = FeatureRegistry()
    discovered = registry.auto_discover("src.features")
    print(f"    Discovered: {[c.__name__ for c in discovered]}")

    configured = registry.configure("config.yaml")
    print(f"    Configured: {configured}")

    X_train_fe = registry.fit_transform_all(X_train_clean)
    X_val_fe = registry.transform_all(X_val_clean)
    registry.save_all("artifacts/features")
    print(f"    Final features: {X_train_fe.shape[1]}")

    # ── 8. Train Logistic Regression ───────────────────────────
    print("\n[8] Training Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_fe, y_train)
    y_prob_lr = lr.predict_proba(X_val_fe)[:, 1]

    auc_lr = roc_auc_score(y_val, y_prob_lr)
    ks_lr = compute_ks(y_val, y_prob_lr)
    p5_lr = compute_precision_at_top_k(y_val, y_prob_lr)

    # ── 9. Train LightGBM ──────────────────────────────────────
    print("[9] Training LightGBM (is_unbalance=True) ...")
    lgb_model = lgb.LGBMClassifier(is_unbalance=True, random_state=42, verbosity=-1)
    lgb_model.fit(X_train_fe, y_train)
    y_prob_lgb = lgb_model.predict_proba(X_val_fe)[:, 1]

    auc_lgb = roc_auc_score(y_val, y_prob_lgb)
    ks_lgb = compute_ks(y_val, y_prob_lgb)
    p5_lgb = compute_precision_at_top_k(y_val, y_prob_lgb)

    # ── 10. Results ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Validation Set Results")
    print("=" * 60)
    print(f"{'Model':<25} {'AUC':>10} {'KS':>10} {'P@5%':>10}")
    print("-" * 60)
    print(f"{'Logistic Regression':<25} {auc_lr:>10.4f} {ks_lr:>10.4f} {p5_lr:>10.4f}")
    print(f"{'LightGBM':<25} {auc_lgb:>10.4f} {ks_lgb:>10.4f} {p5_lgb:>10.4f}")
    print("=" * 60)

    # ── 11. Save report ─────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/baseline.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Baseline 实验记录 (Refactored Pipeline)\n\n")
        f.write("## 实验 — 模块化重构 Baseline\n\n")
        f.write(f"- **日期**: 2026-08-06\n")
        f.write(f"- **数据**: `data/raw/train_transaction.csv` + `data/raw/train_identity.csv`\n")
        f.write(f"- **时间切分**: 按 TransactionDT 排序，最后 20% 作为验证集\n")
        f.write(f"- **Identity 合并**: 切分后分别合并，防止泄漏\n")
        f.write(f"- **预处理**: DataCleaner (Winsorization + 缺失值标志) + CategoricalEncoder\n")
        f.write(f"- **特征工程**: FeatureRegistry 驱动，config.yaml 配置执行顺序\n\n")
        f.write("### 指标\n\n")
        f.write("| 模型 | AUC | KS | Precision@Top5% |\n")
        f.write("|------|-----|----|-----------------|\n")
        f.write(f"| Logistic Regression | {auc_lr:.4f} | {ks_lr:.4f} | {p5_lr:.4f} |\n")
        f.write(f"| LightGBM | {auc_lgb:.4f} | {ks_lgb:.4f} | {p5_lgb:.4f} |\n\n")
        f.write("### 流水线步骤\n\n")
        f.write("1. **Load** — DataLoader 内存优化 (dtype downcast)\n")
        f.write("2. **Profile** — DataProfiler 统计特征分布 (仅训练集)\n")
        f.write("3. **Clean** — DataCleaner 常量列删除 + Winsorization + 缺失值/剪辑标志\n")
        f.write("4. **Encode** — CategoricalEncoder 标签编码 (未见过类别 → -1)\n")
        f.write("5. **Train** — LogisticRegression + LightGBM(is_unbalance=True)\n\n")
        f.write("### 防泄漏保证\n\n")
        f.write("- Identity 表在时间切分后合并\n")
        f.write("- DataCleaner.fit() 仅在训练集上调用\n")
        f.write("- CategoricalEncoder.fit() 仅在训练集上调用\n")
        f.write("- Profile 仅在训练集上计算\n")

    print(f"\n[Done] Results saved to {report_path}")


if __name__ == "__main__":
    main()