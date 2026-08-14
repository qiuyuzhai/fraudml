"""
Population Stability Index (PSI) for monitoring feature drift.

PSI quantifies the shift in the distribution of a feature between
two time periods (e.g. training vs. production).  A PSI > 0.25
indicates significant drift that may require model retraining.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: Optional[List[float]] = None,
    n_bins: int = 10,
) -> Tuple[float, pd.DataFrame]:
    """Compute Population Stability Index between two distributions.

    Parameters
    ----------
    expected : pd.Series
        Reference (training) distribution.
    actual : pd.Series
        Current (production / validation) distribution.
    bins : list[float], optional
        Pre-defined bin edges.  If None, equal-frequency bins from
        *expected* are used.
    n_bins : int
        Number of bins when *bins* is not provided.

    Returns
    -------
    tuple[float, pd.DataFrame]
        PSI value and a detail DataFrame with columns: bin,
        expected_pct, actual_pct, psi_component.
    """
    # np.histogram遇到NaN不会统计，会造成PSI错误
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0, pd.DataFrame(columns=["bin", "expected_pct", "actual_pct", "psi_component"])

    if bins is None:
        try:
            _, bin_edges = pd.qcut(expected, q=n_bins, duplicates="drop", retbins=True)
        except ValueError:
            _, bin_edges = pd.cut(expected, bins=n_bins, duplicates="drop", retbins=True)
    else:
        bin_edges = np.array(bins)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_components = (expected_pct - actual_pct) * np.log(expected_pct / actual_pct)
    psi = float(psi_components.sum())

    detail = pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "expected_pct": expected_pct,
        "actual_pct": actual_pct,
        "psi_component": psi_components, # 每个bin的PSI值
    })
    detail["bin"] = detail.apply(
        lambda r: f"[{r['bin_left']:.4f}, {r['bin_right']:.4f})", axis=1
    )

    return psi, detail[["bin", "expected_pct", "actual_pct", "psi_component"]]


def compute_psi_batch(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    features: List[str],
    n_bins: int = 10,
    threshold: float = 0.25,
) -> pd.DataFrame:
    """Compute PSI for multiple features at once.

    Parameters
    ----------
    expected_df : pd.DataFrame
        Reference DataFrame (e.g. training data).
    actual_df : pd.DataFrame
        Current DataFrame (e.g. production / validation data).
    features : list[str]
        Feature column names.
    n_bins : int
        Number of bins for histogram construction.
    threshold : float
        PSI threshold for flagging drift.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, psi, drifted (PSI > threshold).
    """
    results = []
    for feat in features:
        if feat not in expected_df.columns or feat not in actual_df.columns:
            results.append({"feature": feat, "psi": 0.0, "drifted": False})
            continue
        psi_val, _ = compute_psi(expected_df[feat], actual_df[feat], n_bins=n_bins)
        results.append({
            "feature": feat,
            "psi": round(psi_val, 6),
            "drifted": psi_val > threshold,
        })
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("psi", ascending=False).reset_index(drop=True)
    return result_df