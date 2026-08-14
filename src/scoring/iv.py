"""
Information Value (IV) calculation for feature screening.

IV measures the predictive power of a feature for a binary target.
Commonly used threshold: IV >= 0.02 indicates a useful feature.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .woe import compute_woe


def compute_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_bins: int = 10,
    bins: Optional[List[float]] = None,
) -> float:
    """Compute the Information Value of a single feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing *feature* and *target*.
    feature : str
        Numeric feature column name.
    target : str
        Binary target column name.
    n_bins : int
        Number of equal-frequency bins (ignored if *bins* is provided).
    bins : list[float], optional
        Pre-defined bin edges.

    Returns
    -------
    float
        IV value.  Returns 0.0 if computation fails.
    """
    try:
        woe_df = compute_woe(df, feature, target, bins=bins, n_bins=n_bins)
        return float(woe_df["iv_component"].sum())
    except Exception:
        return 0.0


def compute_iv_batch(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute IV for multiple features at once.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame.
    features : list[str]
        Feature column names to evaluate.
    target : str
        Binary target column name.
    n_bins : int
        Number of bins for WOE calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, iv, useful (IV >= 0.02).
    """
    results = []
    for feat in features:
        if feat not in df.columns:
            results.append({"feature": feat, "iv": 0.0, "useful": False})
            continue
        iv_val = compute_iv(df, feat, target, n_bins=n_bins)
        results.append({
            "feature": feat,
            "iv": round(iv_val, 6),
            "useful": iv_val >= 0.02,
        })
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("iv", ascending=False).reset_index(drop=True)
    return result_df