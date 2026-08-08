"""
Data profiling module for FraudML.

Provides DataProfiler for computing per-column statistics
that inform feature classification and engineering decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


class DataProfiler:
    """Computes per-column statistics and exports a feature profile CSV.

    The profile output serves as the foundation for deciding how to
    treat each column (category vs. numeric vs. high-cardinality code)
    in downstream feature engineering.

    Parameters
    ----------
    output_path : str or Path, optional
        Path to save the profile CSV. Defaults to
        ``../../reports/feature_profile.csv`` relative to this module.

    Attributes
    ----------
    output_path : Path
        Resolved output path.

    Examples
    --------
    >>> profiler = DataProfiler()
    >>> profile = profiler.run(df)
    """

    def __init__(self, output_path: Optional[str | Path] = None) -> None:
        if output_path is None:
            output_path = (
                Path(__file__).resolve().parent.parent.parent
                / "reports"
                / "feature_profile.csv"
            )
        self.output_path: Path = Path(output_path)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-column statistics and save to CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to profile.

        Returns
        -------
        pd.DataFrame
            Profile DataFrame with one row per column, containing
            all computed statistics.
        """
        profile = self._compute_profile(df)
        self._save(profile)
        return profile

    def _compute_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[Dict[str, float | int | str]] = []

        for col in df.columns:
            series = df[col]
            row = self._profile_column(series)
            rows.append(row)

        return pd.DataFrame(rows)

    def _profile_column(self, series: pd.Series) -> Dict[str, float | int | str]:
        """Compute statistics for a single column.

        Parameters
        ----------
        series : pd.Series
            The column to profile.

        Returns
        -------
        dict
            Dictionary with keys: dtype, missing_rate, unique_count,
            is_numeric, mean, std, min, max, q25, q50, q75.
            Non-numeric columns get NaN for numeric stats.
        """
        n_total = len(series)
        n_missing = int(series.isna().sum())
        missing_rate = n_missing / n_total if n_total > 0 else 1.0

        try:
            unique_count = int(series.nunique(dropna=False))
        except (TypeError, ValueError):
            unique_count = int(series.nunique(dropna=True))

        is_numeric = pd.api.types.is_numeric_dtype(series)

        row: Dict[str, float | int | str] = {
            "column": series.name,
            "dtype": str(series.dtype),
            "missing_rate": round(missing_rate, 4),
            "unique_count": unique_count,
            "is_numeric": is_numeric,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
        }


        if not is_numeric or missing_rate >= 1.0:
            return row

        clean = series.dropna()

        if len(clean) == 0:
            return row

        row["mean"] = float(clean.mean())
        row["std"] = float(clean.std())
        row["min"] = float(clean.min())
        row["max"] = float(clean.max())

        q = clean.quantile([0.25, 0.50, 0.75])
        row["q25"] = float(q.iloc[0])
        row["q50"] = float(q.iloc[1])
        row["q75"] = float(q.iloc[2])

        return row

    def _save(self, profile: pd.DataFrame) -> None:
        """Write profile DataFrame to CSV.

        Parameters
        ----------
        profile : pd.DataFrame
            Profile DataFrame to persist.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(self.output_path, index=False)

    def _repr_html_(self) -> str:
        return f"DataProfiler(output_path='{self.output_path}')"