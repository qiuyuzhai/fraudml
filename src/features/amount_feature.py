"""
AmountFeature — Transaction amount features

Precondition: Input dataframe MUST be pre-sorted by transaction_time.

Derives log-amount, integer-amount flag, and card1-group relative
amount stats.  Groupby computations use shift(1) to prevent
look-ahead leakage.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase


class AmountFeature(FeatureBase):
    """Transaction amount derived features.

    **Stateless** — fit() is a no-op.  No parameters learned.

    Features:
        - log_TransactionAmt: log1p-transformed amount
        - is_integer_amount: binary flag (amount == floor(amount))
        - card1_amount_ratio: amount / card1 historical mean (shift-protected)
        - card1_amount_delta: amount - previous tx amount (shift-protected)
    """

    @property
    def is_stateful(self) -> bool:
        return False

    def __init__(self, name: str = "AmountFeature") -> None:
        super().__init__(name=name)
        self._amount_col: str = "TransactionAmt"
        self._group_col: str = "card1"

    def fit(self, df: pd.DataFrame) -> "AmountFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        amt = df[self._amount_col].fillna(0).astype(float)

        # --- log transform ---
        df["log_TransactionAmt"] = np.log1p(amt)

        # --- integer flag ---
        df["is_integer_amount"] = (amt == np.floor(amt)).astype(np.int8)

        # --- card1 relative amount stats (shift-protected) ---
        g = df.groupby(self._group_col, sort=False)

        # Previous row's amount
        prev_amt = g[self._amount_col].shift(1)
        df["prev_amt"] = prev_amt # 把 groupby 输出的prev_amt先写回 df 列，再做减法，保证索引严格对齐，后面再删掉。
        df["is_first_trx"] = df["prev_amt"].isna().astype(np.int8) # 第一行保留NaN，不要fillna(0)
        df["card1_amount_delta"] = (df[self._amount_col] - df["prev_amt"]).fillna(0)

        # Cumulative mean up to previous row (for ratio)
        cum_sum_prev = g[self._amount_col].shift(1).cumsum().fillna(0)
        cum_count_prev = g.cumcount()  # number of prior rows in group
        # Avoid div-by-zero
        safe_count = cum_count_prev.replace(0, 1)
        hist_mean = cum_sum_prev / safe_count
        df["card1_amount_ratio"] = np.where(
            hist_mean > 0,
            amt / hist_mean,
            1.0,
        )
        # Ensure finite values
        df["card1_amount_ratio"] = df["card1_amount_ratio"].replace([np.inf, -np.inf], 0).fillna(0)

        # Drop intermediate columns
        df.drop(columns=["prev_amt"], inplace=True)
        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "AmountFeature",
            "layer": "generic",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "AmountFeature",
                    "description": "Instance name.",
                },
            ],
            "example": "- AmountFeature",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "log_TransactionAmt",
                "is_integer_amount",
                "is_first_trx",
                "card1_amount_delta",
                "card1_amount_ratio", 
            ],
            "physical_meaning": "Transaction amount derived features",
            "unit": "log_usd / flag / ratio / usd",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_amount_basic():
        df = pd.DataFrame({
            "card1": ["A", "A", "B"],
            "TransactionAmt": [10.5, 20.0, 5.0],
        })
        feat = AmountFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "log_TransactionAmt" in result.columns
        assert "is_integer_amount" in result.columns
        assert "card1_amount_ratio" in result.columns
        assert "card1_amount_delta" in result.columns

        # Row 0 (A): first row, no history → ratio=1.0, delta=0
        r0 = result.iloc[0]
        assert r0["card1_amount_delta"] == 0.0
        assert r0["is_integer_amount"] == 0  # 10.5 is not integer

        # Row 1 (A): prev=10.5, mean_prev=10.5, ratio=20/10.5≈1.9048
        r1 = result.iloc[1]
        assert abs(r1["card1_amount_ratio"] - 20.0 / 10.5) < 0.01
        assert r1["card1_amount_delta"] == 20.0 - 10.5

    def test_amount_leakage():
        df1 = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionAmt": [10.0, 20.0, 99999.0],
        })
        df2 = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionAmt": [10.0, 20.0, 30.0],
        })
        feat = AmountFeature()
        feat.fit(df1)
        r1 = feat.transform(df1)
        r2 = feat.transform(df2)

        
        cols_to_check = [
        "log_TransactionAmt",
        "is_integer_amount",
        "card1_amount_ratio",
        "card1_amount_delta",
        "is_first_trx"
       ]
        for col in cols_to_check:
           assert r1.iloc[0][col] == r2.iloc[0][col]
           assert r1.iloc[1][col] == r2.iloc[1][col]

    test_amount_basic()
    test_amount_leakage()
    print("All AmountFeature tests passed!")