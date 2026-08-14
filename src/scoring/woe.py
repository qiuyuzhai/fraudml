"""
Weight of Evidence (WOE) calculation module.

WOE quantifies the separation of good vs. bad (fraud) rates
across bins of a feature.  Used as a pre-processing step
for logistic regression and as a diagnostic tool.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_woe(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: Optional[List[float]] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute WOE and related statistics per bin.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing *feature* and *target* columns.
    feature : str
        Feature column name (numeric).
    target : str
        Binary target column name (0/1).
    bins : list[float], optional
        Pre-defined bin edges.  If None, equal-frequency bins are used.
    n_bins : int
        Number of bins when *bins* is not provided.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bin, count, bad, good, bad_rate,
        woe, iv, good_pct, bad_pct.
    """
    data = df[[feature, target]].copy()
    data = data.dropna()

    if bins is None:
        try:
            # 等频分箱：按样本数量等分
            data["bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")
        except ValueError:
            # 等宽分箱：按数值区间长度等分
            data["bin"] = pd.cut(data[feature], bins=n_bins, duplicates="drop")
    else:
        data["bin"] = pd.cut(data[feature], bins=bins, duplicates="drop")

    total_good = (data[target] == 0).sum()
    total_bad = (data[target] == 1).sum()

    grouped = data.groupby("bin", observed=False).agg(
        count=(target, "size"),
        bad=(target, "sum"),
    )
    grouped["good"] = grouped["count"] - grouped["bad"]
    grouped["bad_rate"] = grouped["bad"] / grouped["count"].replace(0, np.nan)

    grouped["good_pct"] = grouped["good"] / max(total_good, 1)
    grouped["bad_pct"] = grouped["bad"] / max(total_bad, 1)

    grouped["good_pct"] = grouped["good_pct"].replace(0, 1e-10)
    grouped["bad_pct"] = grouped["bad_pct"].replace(0, 1e-10)

    grouped["woe"] = np.log(grouped["bad_pct"] / grouped["good_pct"])
    grouped["iv_component"] = (grouped["bad_pct"] - grouped["good_pct"]) * grouped["woe"]

    grouped = grouped.reset_index()
    grouped["feature"] = feature

    return grouped[
        ["feature", "bin", "count", "bad", "good", "bad_rate",
         "good_pct", "bad_pct", "woe", "iv_component"]
    ]