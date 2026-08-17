"""
AggregationFeature — Group-by aggregation factory.

Supports multiple group-by entity keys with count/sum/mean/std
aggregations on numerical columns.  All history aggregation uses
groupby + shift(1) to prevent look-ahead bias.

Group keys include:
    - card1 (single entity)
    - card1 + addr1 (composite entity)
    - card1 + device (via id_30, OS category)
    - email (via P_emaildomain)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import FeatureBase


# Default numeric columns to aggregate (set during fit)
_DEFAULT_AGG_COLS: List[str] = [
    "TransactionAmt",
    "dist1",
    "dist2",
]

_DEFAULT_STATS: List[str] = ["count", "sum", "mean", "std"]


class AggregationFeature(FeatureBase):
    """Group-by aggregation feature factory with shift-protection.

    Critical anti-leakage rule: all groupby aggregations use
    ``shift(1)`` so the current row never contributes to its own
    aggregated statistics.

    **Stateful** — learns which aggregation columns exist in
    training data during fit().

    Parameters
    ----------
    agg_cols : list of str, optional
        Numerical columns to aggregate.  If None, defaults are used.
    group_keys : list of str or list of tuple, optional
        Column(s) to group by.  Supports both single columns and
        composite keys (as tuples).  Defaults to:
        [('card1',), ('card1', 'addr1'), ('P_emaildomain',)]
    stats : list of str, optional
        Statistics to compute per group.  Default: count, sum, mean, std.
    """

    @property
    def is_stateful(self) -> bool:
        return True

    @property
    def _col_suffix(self) -> str:
        if self.name == self.__class__.__name__:
            return ""
        return f"_{self.name}"

    def __init__(
        self,
        name: str = "AggregationFeature",
        agg_cols: Optional[List[str]] = None,
        group_keys: Optional[List] = None,
        stats: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name=name)
        self._agg_cols = agg_cols or _DEFAULT_AGG_COLS
        self._group_keys = group_keys or [
            ("card1",),
            ("card1", "addr1"),
            ("P_emaildomain",),
        ]
        self._stats = stats or _DEFAULT_STATS

    def _get_state(self) -> Dict[str, Any]:
        return {
            "_agg_cols": self._agg_cols,
            "_group_keys": self._group_keys,
            "_stats": self._stats,
        }
    def fit(self, df: pd.DataFrame) -> "AggregationFeature":
        self._agg_cols = [c for c in self._agg_cols if c in df.columns]
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        suffix = self._col_suffix
        new_cols: Dict[str, pd.Series] = {}

        for key_spec in self._group_keys:
            if isinstance(key_spec, str):
                key_cols = [key_spec]
                key_name = key_spec
            else:
                key_cols = list(key_spec)
                key_name = "+".join(key_cols)

            if not all(c in df.columns for c in key_cols):
                continue

            g = df.groupby(key_cols, sort=False)
            key_arrays = [df[c] for c in key_cols]

            for col in self._agg_cols:
                if col not in df.columns:
                    continue

                for stat in self._stats:
                    col_name = f"{key_name}_{col}_{stat}{suffix}"

                    if stat == "count":
                        val = (g.cumcount() + 1).groupby(key_arrays).shift(1).fillna(0).astype(np.int32)
                        s = pd.Series(val.values, index=df.index, dtype=np.int32)
                        new_cols[col_name] = s

                    elif stat == "sum":
                        val = g[col].cumsum().groupby(key_arrays).shift(1).fillna(0).astype(np.float32)
                        s = pd.Series(val.values, index=df.index, dtype=np.float32)
                        new_cols[col_name] = s

                    elif stat == "mean":
                        cum_sum = g[col].cumsum()
                        cum_count = g.cumcount().add(1)
                        safe_div = cum_sum / cum_count.replace(0, 1)
                        safe_div = safe_div.where(cum_count != 0, 0.0)
                        val = safe_div.groupby(key_arrays).shift(1).fillna(0).astype(np.float32)
                        s = pd.Series(val.values, index=df.index, dtype=np.float32)
                        new_cols[col_name] = s

                    elif stat == "std":
                        expanded_std = g[col].expanding().std()
                        expanded_std = expanded_std.replace([np.inf, -np.inf], np.nan)
                        val = expanded_std.groupby(key_arrays).shift(1).fillna(0).astype(np.float32)
                        s = pd.Series(val.values, index=df.index, dtype=np.float32)
                        new_cols[col_name] = s

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            new_df = new_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            df = pd.concat([df, new_df], axis=1)

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "AggregationFeature",
            "layer": "fraud-domain",
            "is_stateful": True,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "AggregationFeature",
                    "description": "Instance name. Use unique name for different agg configs.",
                },
                {
                    "name": "agg_cols",
                    "type": "list[str]",
                    "default": ["TransactionAmt"],
                    "description": "Numerical columns to aggregate.",
                },
                {
                    "name": "group_keys",
                    "type": "list[list[str]]",
                    "default": [["card1"], ["card1", "addr1"], ["P_emaildomain"]],
                    "description": "Group-by keys. Each entry is a list of column names (supports composite keys).",
                },
                {
                    "name": "stats",
                    "type": "list[str]",
                    "default": ["count", "sum", "mean", "std"],
                    "description": "Statistics to compute. Options: count, sum, mean, std.",
                },
            ],
            "example": """# Single instance:
- AggregationFeature

# Multiple instances with different focus:
- AggregationFeature:
    name: "AggregationFeature_amt"
    agg_cols: ["TransactionAmt"]
    group_keys: [["card1"]]
    stats: ["count", "sum", "mean"]
- AggregationFeature:
    name: "AggregationFeature_dist"
    agg_cols: ["dist1", "dist2"]
    group_keys: [["card1", "addr1"]]
    stats: ["mean", "std"]""",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        suffix = self._col_suffix
        return {
            "feature_names": [
                f"{'+'.join(k)}_{c}_{s}{suffix}"
                for k in self._group_keys
                for c in self._agg_cols
                for s in self._stats
            ],
            "physical_meaning": "Group-level historical aggregates (shift-protected)",
            "unit": "count / usd / ratio",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_agg_basic():
        df = pd.DataFrame({
            "card1": ["A", "A", "A", "B", "B"],
            "addr1": [1, 1, 2, 3, 3],
            "TransactionAmt": [10.0, 20.0, 30.0, 5.0, 15.0],
            "dist1": [100, 200, 300, 400, 500],
        })
        feat = AggregationFeature(agg_cols=["TransactionAmt"], group_keys=[("card1",)], stats=["count", "sum"])
        feat.fit(df)
        result = feat.transform(df)

        agg_cols = [c for c in result.columns if "card1_TransactionAmt" in c]
        assert len(agg_cols) > 0

        # Row 0 (A): no prior → count=0, sum=0
        r0 = result.iloc[0]
        assert r0["card1_TransactionAmt_count"] == 0
        assert r0["card1_TransactionAmt_sum"] == 0

        # Row 1 (A): prior=10 → count=1, sum=10
        r1 = result.iloc[1]
        assert r1["card1_TransactionAmt_count"] == 1
        assert r1["card1_TransactionAmt_sum"] == 10

        # Row 2 (A): prior sum=10+20=30, count=2
        r2 = result.iloc[2]
        assert r2["card1_TransactionAmt_count"] == 2
        assert r2["card1_TransactionAmt_sum"] == 30

    def test_agg_leakage():
        df1 = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionAmt": [10.0, 20.0, 99999.0],
        })
        df2 = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionAmt": [10.0, 20.0, 30.0],
        })
        feat = AggregationFeature(agg_cols=["TransactionAmt"], group_keys=[("card1",)], stats=["sum"])
        feat.fit(df1)
        r1 = feat.transform(df1)
        r2 = feat.transform(df2)

        # Row 2: both should use sum of rows 0+1 = 30 (shift-protected)
        assert r1.iloc[2]["card1_TransactionAmt_sum"] == 30.0
        assert r2.iloc[2]["card1_TransactionAmt_sum"] == 30.0

    def test_agg_composite_key():
        df = pd.DataFrame({
            "card1": ["A", "A", "B", "B"],
            "addr1": [1, 1, 1, 2],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        })
        feat = AggregationFeature(
            agg_cols=["TransactionAmt"],
            group_keys=[("card1", "addr1")],
            stats=["count"],
        )
        feat.fit(df)
        result = feat.transform(df)

        col_name = "card1+addr1_TransactionAmt_count"
        assert col_name in result.columns

        # Row 0 (A,1): no prior → count=0
        assert result.iloc[0][col_name] == 0
        # Row 1 (A,1): prior count=1
        print(result.iloc[1][col_name])
        assert result.iloc[1][col_name] == 1
        
        # Row 2 (B,1): new group → count=0
        print(result.iloc[2][col_name])
        assert result.iloc[2][col_name] == 0
        
        

    test_agg_basic()
    test_agg_leakage()
    test_agg_composite_key()
    print("All AggregationFeature tests passed!")