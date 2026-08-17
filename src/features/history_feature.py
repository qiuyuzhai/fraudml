"""
HistoryFeature — Time-series historical behavior features.

Anti-leakage design:
    Sort by TransactionDT before groupby. Use groupby + shift(1) so
    row-N only sees rows strictly before N. Current-row values NEVER
    leak into historical aggregates.

Supports multi-window rolling statistics via the ``windows`` parameter:
    - count, sum, mean, std per agg_col per window
    - Efficient implementation using np.searchsorted + cumulative sums
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import FeatureBase


class HistoryFeature(FeatureBase):
    """Historical behavior features per card1 group.

    Features (per window, per agg_col):
        - tx_count_last_{window}s: transaction count within window
        - {col}_sum_last_{window}s: rolling sum within window
        - {col}_mean_last_{window}s: rolling mean within window
        - {col}_std_last_{window}s: rolling std within window
    Plus base features:
        - time_since_last_transaction: seconds since previous tx
        - cumulative_spend: cumulative TransactionAmt up to previous row

    All groupby computations use shift(1) to prevent look‑ahead leakage.

    **Stateless** — fit() is a no-op. No parameters learned.

    Args:
        window_seconds: Deprecated, use ``windows`` instead.
        windows: List of time windows in seconds.  Default: [3600].
        stats: List of statistics to compute.
            Options: "count", "sum", "mean", "std".  Default: ["count"].
        agg_cols: Numerical columns for sum/mean/std.
            Default: ["TransactionAmt"].
    """

    @property
    def is_stateful(self) -> bool:
        return False

    @property
    def _col_suffix(self) -> str:
        if self.name == self.__class__.__name__:
            return ""
        return f"_{self.name}"

    def __init__(
        self,
        name: str = "HistoryFeature",
        window_seconds: float = 3600.0,
        windows: Optional[List[float]] = None,
        stats: Optional[List[str]] = None,
        agg_cols: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name=name)
        self._sorted_col: str = "TransactionDT"
        self._group_col: str = "card1"
        self._amount_col: str = "TransactionAmt"
        self._time_col: str = "TransactionDT"

        if windows is not None:
            self._windows: List[float] = list(windows)
        else:
            self._windows = [float(window_seconds)]

        self._stats: List[str] = stats if stats is not None else ["count"]
        self._agg_cols: List[str] = agg_cols if agg_cols is not None else [self._amount_col]

    def fit(self, df: pd.DataFrame) -> "HistoryFeature":
        self._agg_cols = [c for c in self._agg_cols if c in df.columns]
        self._fitted = True
        return self

    def _compute_group_stats(
        self,
        times: np.ndarray,
        window: float,
    ) -> np.ndarray:
        """For each position i, find the left boundary of [t[i]-window, t[i])
        using searchsorted.  Returns an array of left indices (excluding i).
        """
        n = len(times)
        if n == 0:
            return np.array([], dtype=np.int64)
        left = np.searchsorted(times, times - window, side="left")
        return left.astype(np.int64)

    def _window_stats_for_group(
        self,
        times: np.ndarray,
        values_by_col: Dict[str, np.ndarray],
        window: float,
        stats: List[str],
        agg_cols: List[str],
    ) -> Dict[str, np.ndarray]:
        """Compute count/sum/mean/std for one group, one window.

        Returns dict: stat_name -> values array aligned with *times*.
        """
        n = len(times)
        result: Dict[str, np.ndarray] = {}

        left = self._compute_group_stats(times, window)
        counts = np.arange(n, dtype=np.float64) - left.astype(np.float64)
        counts[0] = 0.0

        if "count" in stats:
            result["tx_count"] = counts.astype(np.int32)

        if "sum" in stats or "mean" in stats or "std" in stats:
            for col in agg_cols:
                if col not in values_by_col:
                    continue
                vals = values_by_col[col].astype(np.float64)
                cumsum = np.concatenate([[0.0], np.cumsum(vals)])
                sum_vals = cumsum[np.arange(n)] - cumsum[left]

                safe_counts = np.where(counts > 0, counts, 1.0)
                mean_vals = sum_vals / safe_counts
                mean_vals[counts == 0] = 0.0

                if "sum" in stats:
                    result[f"{col}_sum"] = sum_vals.astype(np.float32)
                if "mean" in stats:
                    result[f"{col}_mean"] = mean_vals.astype(np.float32)
                if "std" in stats:
                    cumsum_sq = np.concatenate([[0.0], np.cumsum(vals ** 2)])
                    sum_sq_vals = cumsum_sq[np.arange(n)] - cumsum_sq[left]
                    var_vals = sum_sq_vals / safe_counts - mean_vals ** 2
                    var_vals = np.maximum(var_vals, 0.0)
                    std_vals = np.sqrt(var_vals)
                    std_vals[counts == 0] = 0.0
                    result[f"{col}_std"] = std_vals.astype(np.float32)

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        suffix = self._col_suffix
        df = df.copy()
        df["_orig_pos"] = range(len(df))
        df = df.sort_values(by=[self._group_col, self._time_col]).reset_index(drop=True)

        g = df.groupby(self._group_col, sort=False)

        # --- time_since_last_transaction ---
        shifted_dt = g[self._time_col].shift(1)
        df[f"time_since_last_transaction{suffix}"] = (
            df[self._time_col] - shifted_dt
        ).fillna(0)

        # --- cumulative_spend ---
        shifted_amt = g[self._amount_col].shift(1)
        df[f"cumulative_spend{suffix}"] = (
            shifted_amt.groupby(df[self._group_col]).cumsum().fillna(0)
        )

        # --- per-window rolling stats ---
        for window in self._windows:
            window = float(window)
            col_suffix = f"_last_{int(window)}s{suffix}"

            for grp_name, grp in g:
                idx = grp.index
                times = grp[self._time_col].values.astype(np.float64)

                values_by_col: Dict[str, np.ndarray] = {}
                for col in self._agg_cols:
                    if col in df.columns:
                        values_by_col[col] = grp[col].fillna(0).values.astype(np.float64)

                stats_result = self._window_stats_for_group(
                    times, values_by_col, window, self._stats, self._agg_cols
                )

                for stat_name, vals in stats_result.items():
                    out_col = f"{stat_name}{col_suffix}"
                    if out_col not in df.columns:
                        df[out_col] = 0.0
                    df.loc[idx, out_col] = vals

        # --- restore original order ---
        df = df.sort_values("_orig_pos").reset_index(drop=True)
        df.drop(columns=["_orig_pos"], inplace=True)

        # fill NaN in new columns
        new_cols = [c for c in df.columns if c.endswith(suffix)]
        for c in new_cols:
            df[c] = df[c].fillna(0)

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        windows_example = [3600, 21600, 86400, 604800]
        return {
            "class_name": "HistoryFeature",
            "layer": "fraud-domain",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "HistoryFeature",
                    "description": "Instance name. Use unique name for multiple instances with different configs.",
                },
                {
                    "name": "windows",
                    "type": "list[float]",
                    "default": "[3600]",
                    "description": "Time windows in seconds. Examples: 600=10min, 3600=1h, 21600=6h, 86400=24h, 604800=7d.",
                },
                {
                    "name": "stats",
                    "type": "list[str]",
                    "default": '["count"]',
                    "description": "Statistics per window. Options: count, sum, mean, std.",
                },
                {
                    "name": "agg_cols",
                    "type": "list[str]",
                    "default": '["TransactionAmt"]',
                    "description": "Numerical columns for sum/mean/std computation.",
                },
            ],
            "example": """# Single window (backward compatible):
- HistoryFeature

# Multi-window with full stats (recommended):
- HistoryFeature:
    name: "HistoryFeature_multi"
    windows: [600, 3600, 21600, 86400, 604800]
    stats: ["count", "sum", "mean", "std"]
    agg_cols: ["TransactionAmt"]""",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        suffix = self._col_suffix
        feature_names: List[str] = [
            f"time_since_last_transaction{suffix}",
            f"cumulative_spend{suffix}",
        ]

        for window in self._windows:
            w = int(window)
            for stat in self._stats:
                if stat == "count":
                    feature_names.append(f"tx_count_last_{w}s{suffix}")
                else:
                    for col in self._agg_cols:
                        feature_names.append(f"{col}_{stat}_last_{w}s{suffix}")

        return {
            "feature_names": feature_names,
            "physical_meaning": "Historical transaction behavior per card",
            "unit": "seconds / count / usd / ratio",
            "depends_on_target": False,
        }


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
        count_col = meta["feature_names"][2]

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
        count_col = meta["feature_names"][2]

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
        col = feat_short.get_feature_metadata()["feature_names"][2]
        assert res.iloc[2][col] == 1

    def test_multi_window_stats():
        df = pd.DataFrame({
            "card1": ["A", "A", "A", "A", "A"],
            "TransactionDT": [0, 500, 1500, 3000, 5000],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        feat = HistoryFeature(
            windows=[1000, 3000],
            stats=["count", "sum", "mean"],
            agg_cols=["TransactionAmt"],
        )
        feat.fit(df)
        result = feat.transform(df)

        assert "tx_count_last_1000s" in result.columns
        assert "TransactionAmt_sum_last_1000s" in result.columns
        assert "TransactionAmt_mean_last_1000s" in result.columns
        assert "tx_count_last_3000s" in result.columns
        assert "TransactionAmt_sum_last_3000s" in result.columns
        assert "TransactionAmt_mean_last_3000s" in result.columns

        r0 = result.iloc[0]
        assert r0["tx_count_last_1000s"] == 0
        assert r0["TransactionAmt_sum_last_1000s"] == 0.0

        r1 = result.iloc[1]
        assert r1["tx_count_last_1000s"] == 1
        assert abs(r1["TransactionAmt_sum_last_1000s"] - 10.0) < 0.01
        assert abs(r1["TransactionAmt_mean_last_1000s"] - 10.0) < 0.01

        r2 = result.iloc[2]
        assert r2["tx_count_last_1000s"] == 1
        assert abs(r2["TransactionAmt_sum_last_1000s"] - 20.0) < 0.01

        r3 = result.iloc[3]
        assert r3["tx_count_last_3000s"] == 3
        assert abs(r3["TransactionAmt_sum_last_3000s"] - 60.0) < 0.01

    def test_multi_window_std():
        df = pd.DataFrame({
            "card1": ["A", "A", "A", "A"],
            "TransactionDT": [0, 1000, 2000, 3000],
            "TransactionAmt": [10.0, 10.0, 30.0, 30.0],
        })
        feat = HistoryFeature(
            windows=[3600],
            stats=["std"],
            agg_cols=["TransactionAmt"],
        )
        feat.fit(df)
        result = feat.transform(df)

        col = "TransactionAmt_std_last_3600s"
        assert col in result.columns

        r0 = result.iloc[0]
        assert r0[col] == 0.0

        r1 = result.iloc[1]
        assert r1[col] == 0.0

        r2 = result.iloc[2]
        expected_std = np.std([10.0, 10.0])
        assert abs(r2[col] - expected_std) < 0.01

        r3 = result.iloc[3]
        expected_std = np.std([10.0, 10.0, 30.0])
        assert abs(r3[col] - expected_std) < 0.01

    test_history_basic()
    test_history_leakage()
    test_history_tx_count_window()
    test_different_window()
    test_multi_window_stats()
    test_multi_window_std()
    print("All HistoryFeature tests passed!")