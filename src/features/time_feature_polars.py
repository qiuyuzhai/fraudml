"""
TimeFeaturePolars — Polars-backed mirror of :class:`TimeFeature`.

Outputs identical columns (``TransactionDT_hour``, ``TransactionDT_dow``,
``is_midnight_flag``, ``time_of_day``) so downstream selection / models
are unaware of the backend.

Polars is imported lazily; instantiation raises :class:`ImportError` if
polars is missing. :meth:`transform` returns a pandas DataFrame so the
pipeline stays homogeneous.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .base import FeatureBase

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    _HAS_POLARS = False


class TimeFeaturePolars(FeatureBase):
    """Extract time-of-day features from ``TransactionDT`` via polars.

    Stateless — :meth:`fit` is a no-op. Output columns:
    ``TransactionDT_hour``, ``TransactionDT_dow``, ``is_midnight_flag``,
    ``time_of_day`` (0=night, 1=morning, 2=afternoon, 3=evening, 4=late).
    """

    @property
    def is_stateful(self) -> bool:
        return False

    def __init__(self, name: str = "TimeFeaturePolars") -> None:
        super().__init__(name=name)
        if not _HAS_POLARS:
            raise ImportError(
                "polars is required for TimeFeaturePolars. "
                "Install with `pip install polars>=0.20.0`."
            )
        self._time_col: str = "TransactionDT"

    def fit(self, df: pd.DataFrame) -> "TimeFeaturePolars":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")
        if not _HAS_POLARS:  # pragma: no cover - guarded at __init__
            raise ImportError("polars is required for TimeFeaturePolars.transform")

        pdf = df.copy()
        if self._time_col not in pdf.columns:
            return pdf

        df_pl = pl.from_pandas(pdf)
        # Treat TransactionDT (seconds since epoch) asDatetime in UTC
        ts_expr = (pl.col(self._time_col).cast(pl.Int64) * 1_000_000).cast(pl.Datetime("us"))

        df_pl = df_pl.with_columns([
            ts_expr.dt.hour().alias("TransactionDT_hour"),
            ts_expr.dt.weekday().alias("TransactionDT_dow"),
        ])

        # is_midnight_flag: hour < 6
        df_pl = df_pl.with_columns(
            pl.when(pl.col("TransactionDT_hour") < 6)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_midnight_flag")
        )

        # time_of_day: 0=night(0-5), 1=morning(6-11), 2=afternoon(12-17),
        #              3=evening(18-21), 4=late_evening(22-23)
        h = pl.col("TransactionDT_hour")
        df_pl = df_pl.with_columns(
            pl.when((h >= 0) & (h < 6)).then(0)
            .when((h >= 6) & (h < 12)).then(1)
            .when((h >= 12) & (h < 18)).then(2)
            .when((h >= 18) & (h < 22)).then(3)
            .otherwise(4)
            .cast(pl.Int8)
            .alias("time_of_day")
        )

        df_pl = df_pl.with_columns(
            pl.col("TransactionDT_hour").cast(pl.Int8),
            pl.col("TransactionDT_dow").cast(pl.Int8),
        )

        return df_pl.to_pandas()

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "TimeFeaturePolars",
            "layer": "generic",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "TimeFeaturePolars",
                    "description": "Instance name.",
                },
            ],
            "example": "- TimeFeaturePolars",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "TransactionDT_hour",
                "TransactionDT_dow",
                "is_midnight_flag",
                "time_of_day",
            ],
            "physical_meaning": "Time-of-day / day-of-week metadata (polars backend)",
            "unit": "hour / flag / period",
            "depends_on_target": False,
        }

    def get_input_columns(self) -> list[str]:
        return [self._time_col]
