"""
Statistics computation for the Feature Store.

Computes per-version distribution statistics (missing rate, mean, std,
quantiles, n_unique) and Information Value (IV) for engineered features,
then persists them to the ``statistics`` table.

IV calculation reuses :func:`src.scoring.iv.compute_iv` to keep the IV
definition consistent across the pipeline (selection / monitoring /
Feature Store). When the target is unavailable (e.g. inference-time
registration), IV is omitted.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


def compute_missing_rate(series: pd.Series) -> float:
    """Fraction of NaN / null values in *series*."""
    if len(series) == 0:
        return 1.0
    return float(series.isna().mean())


def compute_distribution(series: pd.Series) -> dict:
    """Return distribution stats for a numeric / categorical series."""
    s = series.dropna()
    if len(s) == 0:
        return {
            "n_unique": 0,
            "mean": None,
            "std": None,
            "min_value": None,
            "max_value": None,
            "p50": None,
            "p95": None,
        }
    n_unique = int(s.nunique())
    if pd.api.types.is_numeric_dtype(s):
        arr = s.astype(np.float64)
        return {
            "n_unique": n_unique,
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
            "min_value": float(arr.min()),
            "max_value": float(arr.max()),
            "p50": float(arr.quantile(0.5)),
            "p95": float(arr.quantile(0.95)),
        }
    # Categorical / object — no mean/std/min/max, but quantiles via rank
    return {
        "n_unique": n_unique,
        "mean": None,
        "std": None,
        "min_value": None,
        "max_value": None,
        "p50": None,
        "p95": None,
    }


def compute_iv_safe(
    df: pd.DataFrame, feature: str, target: Optional[str], n_bins: int = 10
) -> Optional[float]:
    """Compute IV for *feature* against *target*; return ``None`` if target missing."""
    if target is None or target not in df.columns or feature not in df.columns:
        return None
    try:
        from src.scoring.iv import compute_iv
    except ImportError:
        return None
    try:
        return float(compute_iv(df, feature, target, n_bins=n_bins))
    except Exception:
        return None


def compute_all_stats(
    df: pd.DataFrame, feature: str, target: Optional[str] = None, iv_bins: int = 10
) -> dict:
    """Compute the full statistics row for a feature column."""
    if feature not in df.columns:
        raise KeyError(f"Feature column '{feature}' not in DataFrame")
    series = df[feature]
    dist = compute_distribution(series)
    iv = compute_iv_safe(df, feature, target, n_bins=iv_bins)
    return {
        "missing_rate": compute_missing_rate(series),
        "iv_score": iv,
        "n_unique": dist["n_unique"],
        "mean": dist["mean"],
        "std": dist["std"],
        "min_value": dist["min_value"],
        "max_value": dist["max_value"],
        "p50": dist["p50"],
        "p95": dist["p95"],
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _is_nan_or_none(value) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False
