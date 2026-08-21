"""Tests for TimeFeature — migrated from src/features/time_feature.py __main__."""

import pandas as pd

from src.features.time_feature import TimeFeature


def test_time_basic():
    df = pd.DataFrame(
        {
            "TransactionDT": [
                3600,    # 01:00 → hour=1
                43200,   # 12:00
                75600,   # 21:00
            ],
        }
    )
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


def test_time_is_idempotent():
    """Transform twice on the same input yields identical hour column."""
    df = pd.DataFrame({"TransactionDT": [3600, 43200]})
    feat = TimeFeature()
    feat.fit(df)
    first = feat.transform(df)
    second = feat.transform(df)
    assert list(first["TransactionDT_hour"]) == list(second["TransactionDT_hour"])
