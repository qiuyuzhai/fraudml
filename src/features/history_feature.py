"""
HistoryFeature — Time-series historical behavior features.

Anti-leakage design:
    Sort by TransactionDT before groupby. Use groupby + shift(1) so
    row-N only sees rows strictly before N. Current-row values NEVER
    leak into historical aggregates.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import FeatureBase


class HistoryFeature(FeatureBase):
    """Historical behavior features per card1 group.

    Features:
        - time_since_last_transaction: seconds since previous tx (same card1)
        - tx_count_last_{window_seconds}s: tx count within window_seconds before current row (same card1)
        - cumulative_spend: cumulative TransactionAmt up to previous row (same card1)

    All groupby computations use shift(1) to prevent look‑ahead leakage.

    **Stateless** — fit() is a no-op.  No parameters learned.

    Args:
        window_seconds: Time window in seconds for counting transactions. Default is 3600 (1 hour).
    """

    @property
    def is_stateful(self) -> bool:
        return False

    def __init__(self, name: str = "HistoryFeature", window_seconds: float = 3600.0) -> None:
        super().__init__(name=name)
        self._sorted_col: str = "TransactionDT"
        self._group_col: str = "card1"
        self._amount_col: str = "TransactionAmt"
        self._time_col: str = "TransactionDT"
        self._window_seconds = window_seconds
        self._count_col_name = f"tx_count_last_{int(self._window_seconds)}s"

    def fit(self, df: pd.DataFrame) -> "HistoryFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        df["_orig_pos"] = range(len(df)) # 用位置记录原始索引位置;明确记录「当前行在第几位」
        df = df.sort_values(by=[self._group_col, self._time_col]).reset_index(drop=True)

        g = df.groupby(self._group_col, sort=False)

        # ---------------- time_since_last_transaction ----------------
        shifted_dt = g[self._time_col].shift(1)
        df["time_since_last_transaction"] = (df[self._time_col] - shifted_dt).fillna(0)

        # ---------------- tx_count_last_{window_seconds}s ----------------
        count_arr = np.zeros(len(df), dtype=np.int32)
        for _, grp in df.groupby(self._group_col, sort=False):
            t = grp[self._time_col].values
            n = len(t)
            idx = grp.index
            for i in range(n):
                if i == 0:
                    cnt = 0
                else:
                    cnt = np.sum((t[i] - t[:i]) <= self._window_seconds)
                count_arr[idx[i]] = cnt
        df[self._count_col_name] = count_arr

        shifted_amt = g[self._amount_col].shift(1)
        df["cumulative_spend"] = shifted_amt.groupby(df[self._group_col]).cumsum().fillna(0)

        # 修改索引为原始索引，保持与目标列对应关系
        df = df.sort_values("_orig_pos").reset_index(drop=True) # 按记录的位置回去
        df.drop(columns=["_orig_pos"], inplace=True)


        df["time_since_last_transaction"] = df["time_since_last_transaction"].fillna(0)
        df["cumulative_spend"] = df["cumulative_spend"].fillna(0)

        # Restore the input row order so downstream positional slicing
        # (e.g. iloc-based train/val splits) still matches the original X/y pairing.
        # return df.sort_index()
        return df

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "time_since_last_transaction",
                self._count_col_name,
                "cumulative_spend",
            ],
            "physical_meaning": "Historical transaction behavior per card",
            "unit": "seconds / count / usd",
            "depends_on_target": False,
        }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def test_history_basic():
        df = pd.DataFrame({
            "card1": ["A", "A", "A", "B", "B"],
            "TransactionDT": [1000, 2000, 3000, 1500, 2500],
            "TransactionAmt": [10.0, 20.0, 30.0, 5.0, 15.0],
        })
        feat = HistoryFeature(window_seconds=3600)
        feat.fit(df)
        result = feat.transform(df)
        meta = feat.get_feature_metadata()
        count_col = meta["feature_names"][1]

        assert "time_since_last_transaction" in result.columns
        assert count_col in result.columns
        assert "cumulative_spend" in result.columns

        row0 = result.iloc[0]
        assert row0["time_since_last_transaction"] == 0.0
        assert row0[count_col] == 0
        assert row0["cumulative_spend"] == 0.0

        row1 = result.iloc[1]
        assert row1["time_since_last_transaction"] == 1000.0
        assert row1[count_col] == 1   

        row2 = result.iloc[2]
        assert row2["time_since_last_transaction"] == 1000.0
        assert row2["cumulative_spend"] == 30.0

    def test_history_leakage():
        """Inject extreme value on current row → assert output unchanged."""
        df1 = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionDT": [1000, 2000, 3000],
            "TransactionAmt": [10.0, 20.0, 999999.0],
        })
        df2 = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionDT": [1000, 2000, 3000],
            "TransactionAmt": [10.0, 20.0, 30.0],
        })
        feat = HistoryFeature()
        feat.fit(df1)
        r1 = feat.transform(df1)
        r2 = feat.transform(df2)

        assert r1.iloc[2]["cumulative_spend"] == 30.0
        assert r2.iloc[2]["cumulative_spend"] == 30.0

    def test_history_tx_count_window():
        df = pd.DataFrame({
            "card1": ["A", "A", "A", "A"],
            "TransactionDT": [0, 1000, 2000, 3500],
            "TransactionAmt": [5.0, 10.0, 15.0, 20.0],
        })
        feat = HistoryFeature(window_seconds=3600)
        feat.fit(df)
        result = feat.transform(df)
        meta = feat.get_feature_metadata()
        count_col = meta["feature_names"][1]

        assert result.iloc[0][count_col] == 0
        assert result.iloc[1][count_col] == 1
        assert result.iloc[2][count_col] == 2
        assert result.iloc[3][count_col] == 3

    def test_different_window():
        df = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionDT": [0, 2000, 5000],
            "TransactionAmt": [1, 2, 3]
        })
        feat_short = HistoryFeature(window_seconds=3000)
        feat_short.fit(df)
        res = feat_short.transform(df)
        col = feat_short.get_feature_metadata()["feature_names"][1]
        assert res.iloc[2][col] == 1

    test_history_basic()
    test_history_leakage()
    test_history_tx_count_window()
    test_different_window()
    print("All HistoryFeature tests passed!")