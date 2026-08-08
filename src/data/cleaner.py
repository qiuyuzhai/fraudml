"""
Data cleaning module for FraudML.

Provides DataCleaner with strict fit/transform separation to prevent
data leakage in risk-modeling scenarios. Handles missing values, constant
columns, and extreme outliers via Winsorization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DataCleaner:
    """Cleans data with sklearn-style fit/transform pattern.

    Design principles for fraud detection (risk-modeling):

    1. **No leakage**: ``fit()`` MUST be called only on training data.
       No validation / test data participates in any statistical
       calculation (nunique, median, quantile).

    2. **Missing = signal**: Missing numeric values are not silently
       discarded.  A binary ``{col}_isna`` flag is generated so that
       the model can learn *why* a value was missing.

    3. **Winsorization ≠ deletion**: Extreme values are clipped to
       the 1% / 99% quantile thresholds rather than dropped.
       Auxiliary ``{col}_clip_low`` / ``{col}_clip_high`` flags
       preserve the signal that a sample was in the tail.

    Attributes (set after ``fit()``):
    ----------
    constant_cols_ : list[str]
        Columns with ``nunique(dropna=True) <= 1``, to be dropped.
    numeric_cols_ : list[str]
        All numeric columns present in training data.
    cols_with_missing_ : list[str]
        Subset of ``numeric_cols_`` that contained missing values.
    medians_ : dict[str, float]
        Median per numeric column (for imputation).
    quantile_thresholds_ : dict[str, tuple[float, float]]
        (q1, q99) per numeric column (for Winsorization).
    """

    def __init__(self) -> None:
        self.constant_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.cols_with_missing_: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.quantile_thresholds_: Dict[str, Tuple[float, float]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit — ONLY called on training data
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn cleaning parameters from **training data only**.

        Computes:
        - constant columns (nunique ≤ 1)
        - numeric columns with their median and 1%/99% quantiles

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame. Must never include validation / test data.

        Returns
        -------
        self : DataCleaner
        """
        self.constant_cols_ = self._find_constant_columns(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols_ = [c for c in numeric_cols if c not in self.constant_cols_]

        self.cols_with_missing_ = []
        self.medians_ = {}
        self.quantile_thresholds_ = {}

        for col in self.numeric_cols_:
            series = df[col]

            if series.isna().any():
                self.cols_with_missing_.append(col)

            clean = series.dropna()

            if len(clean) < 2:
                continue

            self.medians_[col] = float(clean.median())

            q1 = float(clean.quantile(0.01))
            q99 = float(clean.quantile(0.99))
            self.quantile_thresholds_[col] = (q1, q99)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform — applied to train / val / test sets
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned cleaning parameters.

        NO new statistics are computed inside this method.
        Everything comes from attributes set during ``fit()``.

        Steps applied:
        1. Drop constant columns.
        2. Generate ``{col}_isna`` flags for columns that had missing
           values in training, then fill NaN with stored medians.
        3. Generate ``{col}_clip_low`` / ``{col}_clip_high`` flags
           for every numeric column, then Winsorize clip.

        New columns are batch-inserted via ``pd.concat`` to avoid
        DataFrame fragmentation from per-column insertion.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to clean (training / validation / test).

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with new auxiliary flag columns.
        """
        if not self._fitted:
            raise RuntimeError("DataCleaner has not been fitted. Call fit() first.")

        # Step 1: drop constant columns identified during fit
        cols_to_drop = [c for c in self.constant_cols_ if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Step 2 & 3: process numeric columns — collect all new columns
        # in a dict, then batch-concat to avoid fragmented DataFrame warnings.
        new_data: Dict[str, pd.Series] = {}

        for col in self.numeric_cols_:
            if col not in df.columns:
                continue

            orig = df[col].copy()

            # Step 2a: generate isna flag for columns that had missing values
            if col in self.cols_with_missing_:
                new_data[f"{col}_isna"] = orig.isna().astype(int)

            # Step 3a: generate clip flags — based on ORIGINAL values
            q1, q99 = self.quantile_thresholds_[col]
            is_valid = orig.notna()
            new_data[f"{col}_clip_low"] = (is_valid & (orig < q1)).astype(int)
            new_data[f"{col}_clip_high"] = (is_valid & (orig > q99)).astype(int)

            # Step 2b: fill missing values with stored median
            if col in self.cols_with_missing_:
                median = self.medians_.get(col)
                if median is not None and not np.isnan(median):
                    filled = df[col].fillna(median)
                else:
                    filled = df[col]
            else:
                filled = df[col]

            # Step 3b: Winsorize — clip to [q1, q99]
            new_data[col] = filled.clip(lower=q1, upper=q99)

        if new_data:
            new_cols_df = pd.DataFrame(new_data, index=df.index)
            processed_num_cols = [c for c in self.numeric_cols_ if c in df.columns]
            df = df.drop(columns=processed_num_cols)
            df = pd.concat([df, new_cols_df], axis=1)

        return df

    # ------------------------------------------------------------------
    # Summary — export cleaning parameters for audit / review
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarizing all learned cleaning parameters.

        Each row represents one column with its cleaning action and
        associated statistics (median, q1, q99).  Useful for data-quality
        audits and downstream feature-engineering decisions.

        Returns
        -------
        pd.DataFrame
            Columns: ``column``, ``action`` (``"drop"`` / ``"keep"``),
            ``reason``, ``median``, ``q1``, ``q99``, ``has_missing``.

        Raises
        ------
        RuntimeError
            If the cleaner has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("DataCleaner has not been fitted. Call fit() first.")

        rows: List[Dict[str, object]] = []

        for col in self.constant_cols_:
            rows.append({
                "column": col,
                "action": "drop",
                "reason": "constant (nunique <= 1)",
                "median": np.nan,
                "q1": np.nan,
                "q99": np.nan,
                "has_missing": False,
            })

        for col in self.numeric_cols_:
            q1, q99 = self.quantile_thresholds_.get(col, (np.nan, np.nan))
            median = self.medians_.get(col, np.nan)
            rows.append({
                "column": col,
                "action": "keep",
                "reason": "",
                "median": median,
                "q1": q1,
                "q99": q99,
                "has_missing": col in self.cols_with_missing_,
            })

        return pd.DataFrame(rows)

    def save_summary(self, path: str | Path | None = None) -> Path:
        """Export cleaning summary to CSV for audit / review.

        Parameters
        ----------
        path : str | Path, optional
            Destination path.  Defaults to
            ``../../reports/cleaning_summary.csv`` relative to this module.

        Returns
        -------
        Path
            Resolved path of the written CSV file.
        """
        if path is None:
            path = (
                Path(__file__).resolve().parent.parent.parent
                / "reports"
                / "cleaning_summary.csv"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary_df = self.summary()
        summary_df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_constant_columns(df: pd.DataFrame) -> List[str]:
        """Identify columns with <= 1 unique non-null value.

        These carry no information for the model and should be dropped.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        list[str]
            Constant column names.
        """
        constant_cols: List[str] = []
        for col in df.columns:
            try:
                n_unique = df[col].nunique(dropna=True)
            except (TypeError, ValueError):
                continue
            if n_unique <= 1:
                constant_cols.append(col)
        return constant_cols

    def _repr_html_(self) -> str:
        return (
            f"DataCleaner("
            f"constant_cols={len(self.constant_cols_)}, "
            f"numeric_cols={len(self.numeric_cols_)}, "
            f"cols_with_missing={len(self.cols_with_missing_)}"
            f")"
        )