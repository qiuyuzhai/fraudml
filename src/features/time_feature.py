"""
TimeFeature — Time-of-day / day-of-week features .

Extracts time metadata purely from TransactionDT.  No cross-row
aggregation (time-delta features belong to HistoryFeature).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase


class TimeFeature(FeatureBase):
    """Extract time-of-day features from TransactionDT.

    **Stateless** — fit() is a no-op.  No parameters learned.

    Features:
        - TransactionDT_hour: hour of day (0-23)
        - TransactionDT_dow: day of week (0=Mon … 6=Sun)
        - is_midnight_flag: 1 if hour in [0, 6)
        - time_of_day: categorical period (0-5)
    """

    @property
    def is_stateful(self) -> bool:
        return False

    def __init__(self, name: str = "TimeFeature") -> None:
        super().__init__(name=name)
        self._time_col: str = "TransactionDT"

    def fit(self, df: pd.DataFrame) -> "TimeFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        ts = pd.to_datetime(df[self._time_col], unit="s", errors="coerce")

        df["TransactionDT_hour"] = ts.dt.hour.fillna(0).astype(np.int8)
        df["TransactionDT_dow"] = ts.dt.dayofweek.fillna(0).astype(np.int8)
        df["is_midnight_flag"] = (df["TransactionDT_hour"] < 6).astype(np.int8)

        # time_of_day: 0=night(0-5), 1=morning(6-11), 2=afternoon(12-17),
        #               3=evening(18-21), 4=late_evening(22-23)
        h = df["TransactionDT_hour"]
        df["time_of_day"] = np.select(
            [
                (h >= 0) & (h < 6),
                (h >= 6) & (h < 12),
                (h >= 12) & (h < 18),
                (h >= 18) & (h < 22),
            ],
            [0, 1, 2, 3],
            default=4,
        ).astype(np.int8)

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "TimeFeature",
            "layer": "generic",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "TimeFeature",
                    "description": "Instance name.",
                },
            ],
            "example": "- TimeFeature",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "TransactionDT_hour",
                "TransactionDT_dow",
                "is_midnight_flag",
                "time_of_day",
            ],
            "physical_meaning": "Time-of-day / day-of-week metadata",
            "unit": "hour / flag / period",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_time_basic():
        df = pd.DataFrame({
            "TransactionDT": [
                3600,       # 01:00 → hour=1, dow depends on epoch
                43200,      # 12:00
                75600,      # 21:00
            ],
        })
        feat = TimeFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "TransactionDT_hour" in result.columns
        assert "is_midnight_flag" in result.columns
        assert "time_of_day" in result.columns

        assert result.iloc[0]["TransactionDT_hour"] == 1
        assert result.iloc[0]["is_midnight_flag"] == 1  # hour 1 < 6
        assert result.iloc[1]["TransactionDT_hour"] == 12
        assert result.iloc[2]["TransactionDT_hour"] == 21

    test_time_basic()
    print("All TimeFeature tests passed!")