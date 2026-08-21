"""
AggregationFeaturePolars — Polars-backed mirror of :class:`AggregationFeature`.

Replicates the same shift-protected count/sum/mean/std group-by
aggregations and identical output column naming (``{key}_{col}_{stat}``,
or ``{key1+key2}_{col}_{stat}`` for composite keys) so downstream
selection / model code is unaware of the backend.

Polars is imported lazily; if it is not installed, instantiation raises
:class:`ImportError`. The :func:`make_loader` factory and the registry's
auto-discovery both tolerate this — the polars class is simply not
instantiated when ``engine="pandas"``.

Output contract: :meth:`transform` returns a pandas DataFrame (via
``polars.DataFrame.to_pandas``) so the registry pipeline stays
homogeneous.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .base import FeatureBase

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    _HAS_POLARS = False


_DEFAULT_AGG_COLS: List[str] = ["TransactionAmt", "dist1", "dist2"]
_DEFAULT_STATS: List[str] = ["count", "sum", "mean", "std"]
_DEFAULT_GROUP_KEYS: List = [
    ("card1",),
    ("card1", "addr1"),
    ("P_emaildomain",),
]


class AggregationFeaturePolars(FeatureBase):
    """Polars-backed group-by aggregation with shift(1) leak protection.

    Parameters
    ----------
    name : str
    agg_cols : list of str, optional
        Numerical columns to aggregate. Defaults to
        ``["TransactionAmt", "dist1", "dist2"]``.
    group_keys : list of tuple, optional
        Group-by keys. Composite keys are tuples of column names.
    stats : list of str, optional
        Statistics: subset of ``["count", "sum", "mean", "std"]``.
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
        name: str = "AggregationFeaturePolars",
        agg_cols: Optional[List[str]] = None,
        group_keys: Optional[List] = None,
        stats: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name=name)
        if not _HAS_POLARS:
            raise ImportError(
                "polars is required for AggregationFeaturePolars. "
                "Install with `pip install polars>=0.20.0`."
            )
        self._agg_cols = agg_cols or _DEFAULT_AGG_COLS
        self._group_keys = group_keys or _DEFAULT_GROUP_KEYS
        self._stats = stats or _DEFAULT_STATS

    def _get_state(self) -> Dict[str, Any]:
        return {
            "_agg_cols": self._agg_cols,
            "_group_keys": self._group_keys,
            "_stats": self._stats,
        }

    def fit(self, df: pd.DataFrame) -> "AggregationFeaturePolars":
        self._agg_cols = [c for c in self._agg_cols if c in df.columns]
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")
        if not _HAS_POLARS:  # pragma: no cover - guarded at __init__
            raise ImportError("polars is required for AggregationFeaturePolars.transform")

        suffix = self._col_suffix
        pdf = df.copy()
        df_pl = pl.from_pandas(pdf)

        new_cols_expr: List[pl.Expr] = []
        new_col_names: List[str] = []

        for key_spec in self._group_keys:
            if isinstance(key_spec, str):
                key_cols = [key_spec]
                key_name = key_spec
            else:
                key_cols = list(key_spec)
                key_name = "+".join(key_cols)

            if not all(c in df_pl.columns for c in key_cols):
                continue

            # Sort by group keys + original order so cum_sum is monotone within group
            df_pl = df_pl.with_row_count("__row_order__").sort(key_cols + ["__row_order__"])

            for col in self._agg_cols:
                if col not in df_pl.columns:
                    continue

                for stat in self._stats:
                    col_name = f"{key_name}_{col}_{stat}{suffix}"

                    if stat == "count":
                        # cumulative count of group so far, then shift(1) to drop self
                        expr = (
                            pl.col(col)
                            .is_not_null()
                            .cast(pl.Int64)
                            .cum_sum()
                            .over(key_cols)
                            .shift(1)
                            .fill_null(0)
                            .alias(col_name)
                        )
                    elif stat == "sum":
                        expr = (
                            pl.col(col)
                            .fill_null(0)
                            .cum_sum()
                            .over(key_cols)
                            .shift(1)
                            .fill_null(0)
                            .alias(col_name)
                        )
                    elif stat == "mean":
                        cum_sum = pl.col(col).fill_null(0).cum_sum().over(key_cols)
                        cum_cnt = pl.col(col).is_not_null().cast(pl.Int64).cum_sum().over(key_cols)
                        safe_cnt = pl.when(cum_cnt == 0).then(1).otherwise(cum_cnt)
                        mean_val = cum_sum / safe_cnt
                        expr = mean_val.shift(1).over(key_cols).fill_null(0).alias(col_name)
                    elif stat == "std":
                        # expanding std over group; polars has .cum_* but no
                        # .cum_std, so fall back to per-group .std() computed
                        # against a cumulative window via a join-free approach:
                        # use pl.col(col).cum_* helpers below.
                        # Implement as: for each row, std of all prior rows in group.
                        # Polars doesn't have a direct expanding().std, so we
                        # approximate using the standard trick of rolling window
                        # the size of the group count — but that would be O(n²).
                        # For parity with the pandas implementation (which uses
                        # .expanding().std() then shift(1)), we compute it via
                        # a self-join on group + cumcount.
                        # Cheaper approach: maintain cum_sum, cum_sum_of_squares.
                        cum_sum_sq = (
                            (pl.col(col).fill_null(0) ** 2).cum_sum().over(key_cols)
                        )
                        cum_sum = pl.col(col).fill_null(0).cum_sum().over(key_cols)
                        cum_cnt = (
                            pl.col(col).is_not_null().cast(pl.Int64).cum_sum().over(key_cols)
                        )
                        # var = E[x²] - (E[x])²  (population variance)
                        safe_cnt = pl.when(cum_cnt == 0).then(1).otherwise(cum_cnt)
                        mean_sq = cum_sum_sq / safe_cnt
                        mean_lin = cum_sum / safe_cnt
                        var = (mean_sq - mean_lin ** 2).clip(lower_bound=0.0)
                        std_val = var.sqrt()
                        expr = std_val.shift(1).over(key_cols).fill_null(0).alias(col_name)
                    else:
                        continue

                    new_cols_expr.append(expr)
                    new_col_names.append(col_name)

        if new_cols_expr:
            df_pl = df_pl.with_columns(new_cols_expr)
            # Restore original row order before to_pandas
            df_pl = df_pl.sort("__row_order__").drop("__row_order__")
            extra = df_pl.select(new_col_names).to_pandas()
            # Align indexes (pdf was sorted or unsorted; concat on positional axis=1)
            extra = extra.reset_index(drop=True)
            pdf = pdf.reset_index(drop=True)
            pdf = pd.concat([pdf, extra], axis=1)
            pdf = pdf.replace([float("inf"), float("-inf")], 0).fillna(0)
        else:
            pdf = pdf.reset_index(drop=True)

        return pdf

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "AggregationFeaturePolars",
            "layer": "fraud-domain",
            "is_stateful": True,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "AggregationFeaturePolars",
                    "description": "Instance name. Use unique name for different agg configs.",
                },
                {
                    "name": "agg_cols",
                    "type": "list[str]",
                    "default": ["TransactionAmt", "dist1", "dist2"],
                    "description": "Numerical columns to aggregate.",
                },
                {
                    "name": "group_keys",
                    "type": "list[tuple]",
                    "default": [["card1"], ["card1", "addr1"], ["P_emaildomain"]],
                    "description": "Group-by keys. Composite keys as tuples.",
                },
                {
                    "name": "stats",
                    "type": "list[str]",
                    "default": ["count", "sum", "mean", "std"],
                    "description": "Statistics to compute. Options: count, sum, mean, std.",
                },
            ],
            "example": "- AggregationFeaturePolars",
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
            "physical_meaning": "Group-level historical aggregates (shift-protected, polars backend)",
            "unit": "count / usd / ratio",
            "depends_on_target": False,
        }

    def get_input_columns(self) -> list[str]:
        cols: list[str] = []
        for k in self._group_keys:
            cols.extend(list(k) if not isinstance(k, str) else [k])
        cols.extend(self._agg_cols)
        return list(dict.fromkeys(cols))
