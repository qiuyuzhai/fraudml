"""
baseline fraud detection training script.

使用方式:
    python src/train.py --data data/raw/sample.csv
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] 数据文件不存在: {path}")
        print("[INFO] 生成示例数据用于演示...")
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
            "feature_c": rng.binomial(1, 0.3, n),
            "label": rng.binomial(1, 0.1, n),
        })
        return df
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, target: str = "label"):
    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train(X_train, y_train, **kwargs):
    model = RandomForestClassifier(n_estimators=kwargs.get("n_estimators", 100), random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== 模型评估 ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))


def main():
    parser = argparse.ArgumentParser(description="Fraud detection baseline training")
    parser.add_argument("--data", default="data/raw/sample.csv", help="CSV 数据文件路径")
    parser.add_argument("--target", default="label", help="目标列名")
    parser.add_argument("--n-estimators", type=int, default=100, help="随机森林树数量")
    parser.add_argument("--output", default="src/model.pkl", help="模型保存路径")
    args = parser.parse_args()

    print(f"[INFO] 加载数据: {args.data}")
    df = load_data(args.data)
    print(f"[INFO] 数据形状: {df.shape}")

    X, y = preprocess(df, target=args.target)
    print(f"[INFO] 特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
    print(f"[INFO] 正样本比例: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] 训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    model = train(X_train, y_train, n_estimators=args.n_estimators)

    evaluate(model, X_test, y_test)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[INFO] 模型已保存: {args.output}")


if __name__ == "__main__":
    main()