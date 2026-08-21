"""
UIDFeature — 客户唯一标识 (UID) + 聚合特征.

IEEE-CIS Fraud Detection 竞赛冠军方案的 "magic feature"：
用 card1 + addr1 + D1 推断同一客户 (UID)，
然后计算每客户的历史聚合统计量。

防泄漏设计：
    fit() 只在 train 上统计每 UID 的聚合量（count/mean/std/min/max）。
    transform() 在 val/test 上查表映射，不计算新统计量。
    全局聚合不用 target，无目标泄漏。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import FeatureBase


class UIDFeature(FeatureBase):
    """客户唯一标识 + 聚合特征.

    用 ``uid_cols`` 的组合构造 UID（默认 card1+addr1+D1），
    然后为每笔交易附加该客户的历史聚合统计量。

    Parameters
    ----------
    uid_cols : list of str
        用于构造 UID 的列名。默认 ``["card1", "addr1", "D1"]``。
    agg_col : str
        做聚合的数值列。默认 ``"TransactionAmt"``。
    time_col : str
        时间列。默认 ``"TransactionDT"``。
    windows : list of float
        时间窗口（秒），用于计算 intra-batch 的滚动计数。
        默认 ``[3600, 86400]``（1 小时、1 天）。
    """

    @property
    def is_stateful(self) -> bool:
        return True

    def __init__(
        self,
        name: str = "UIDFeature",
        uid_cols: Optional[List[str]] = None,
        agg_col: str = "TransactionAmt",
        time_col: str = "TransactionDT",
        windows: Optional[List[float]] = None,
    ) -> None:
        super().__init__(name=name)
        self._uid_cols = uid_cols or ["card1", "addr1", "D1"]
        self._agg_col = agg_col
        self._time_col = time_col
        self._windows = windows or [3600, 86400]

        self._uid_stats: Dict[str, Dict[str, float]] = {}
        self._uid_last_time: Dict[str, float] = {}
        self._uid_freq: Dict[str, float] = {}
        self._n_train: int = 0

    # ------------------------------------------------------------------
    # UID construction
    # ------------------------------------------------------------------

    def _build_uid(self, df: pd.DataFrame) -> pd.Series:
        parts = []
        for col in self._uid_cols:
            if col in df.columns:
                parts.append(df[col].fillna(-1).round().astype("int64").astype(str))
            else:
                parts.append(pd.Series(["-1"] * len(df), index=df.index))
        result = parts[0].copy()
        for p in parts[1:]:
            result = result.str.cat(p, sep="_")
        return result

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "UIDFeature":
        uid = self._build_uid(df)
        self._n_train = len(df)

        tmp = df.assign(_uid=uid)
        grouped = tmp.groupby("_uid")

        agg = grouped[self._agg_col].agg(["count", "mean", "std", "min", "max"])
        self._uid_stats = agg.to_dict("index")

        self._uid_last_time = grouped[self._time_col].max().to_dict()

        self._uid_freq = (grouped.size() / self._n_train).to_dict()

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("UIDFeature not fitted")

        result = df.copy()
        uid = self._build_uid(result)

        # ---- global aggregations from train ----
        result["uid_txn_count"] = uid.map(
            lambda x: self._uid_stats.get(x, {}).get("count", 0)
        ).astype("float32")
        result["uid_amt_mean"] = uid.map(
            lambda x: self._uid_stats.get(x, {}).get("mean", np.nan)
        ).astype("float32")
        result["uid_amt_std"] = uid.map(
            lambda x: self._uid_stats.get(x, {}).get("std", np.nan)
        ).astype("float32")
        result["uid_amt_min"] = uid.map(
            lambda x: self._uid_stats.get(x, {}).get("min", np.nan)
        ).astype("float32")
        result["uid_amt_max"] = uid.map(
            lambda x: self._uid_stats.get(x, {}).get("max", np.nan)
        ).astype("float32")
        result["uid_freq"] = uid.map(
            lambda x: self._uid_freq.get(x, 0.0)
        ).astype("float32")
        result["uid_is_new"] = (~uid.isin(self._uid_stats)).astype("int8")

        # ---- time since last transaction (from train) ----
        min_dt = float(result[self._time_col].min())
        last_time = uid.map(lambda x: self._uid_last_time.get(x, min_dt))
        result["uid_time_since_last"] = (
            result[self._time_col].astype("float64") - last_time
        ).clip(lower=0).astype("float32")

        # ---- amount z-score (deviation from client's historical mean) ----
        amt_std = result["uid_amt_std"].replace(0, np.nan)
        result["uid_amt_zscore"] = (
            (result[self._agg_col].astype("float64") - result["uid_amt_mean"]) / amt_std
        ).fillna(0).astype("float32")

        # ---- intra-batch rolling features (within this DataFrame) ----
        result["_uid"] = uid
        sort_idx = result[self._time_col].sort_values().index
        sorted_df = result.loc[sort_idx]

        for w in self._windows:
            self._add_window_count(sorted_df, w)

        result = sorted_df.sort_index()
        result = result.drop(columns=["_uid"])
        return result

    def _add_window_count(self, df: pd.DataFrame, window: float) -> None:
        """Add ``uid_txn_count_{w}s`` — count of same-UID txns in past *window* seconds.

        Uses groupby + shift to avoid look-ahead.  Computed on the
        time-sorted DataFrame in-place.
        """
        col_name = f"uid_txn_count_{int(window)}s"
        grp = df.sort_values([self._time_col]).groupby("_uid")

        dt = df[self._time_col].values
        counts = np.zeros(len(df), dtype=np.float32)

        for _, idx in grp.groups.items():
            idx = np.asarray(idx)
            if len(idx) <= 1:
                continue
            t = dt[idx]
            for i in range(1, len(idx)):
                left = np.searchsorted(t[:i], t[i] - window, side="left")
                counts[idx[i]] = i - left

        df[col_name] = counts

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_feature_metadata(self) -> Dict[str, Any]:
        feats = [
            "uid_txn_count", "uid_amt_mean", "uid_amt_std",
            "uid_amt_min", "uid_amt_max", "uid_freq",
            "uid_is_new", "uid_time_since_last", "uid_amt_zscore",
        ]
        for w in self._windows:
            feats.append(f"uid_txn_count_{int(w)}s")
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "output_features": feats,
            "is_stateful": True,
            "uid_cols": self._uid_cols,
            "description": "Client UID (card1+addr1+D1) + historical aggregations",
        }
