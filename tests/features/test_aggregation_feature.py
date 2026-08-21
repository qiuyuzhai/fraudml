"""Tests for AggregationFeature.

Migrated from src/features/aggregation_feature.py __main__ — preserves
the original assertions around shift-protected aggregation, composite
keys, and the train/val leakage guard.
"""

import pandas as pd

from src.features.aggregation_feature import AggregationFeature


def test_agg_basic():
    df = pd.DataFrame(
        {
            "card1": ["A", "A", "A", "B", "B"],
            "addr1": [1, 1, 2, 3, 3],
            "TransactionAmt": [10.0, 20.0, 30.0, 5.0, 15.0],
            "dist1": [100, 200, 300, 400, 500],
        }
    )
    feat = AggregationFeature(
        agg_cols=["TransactionAmt"],
        group_keys=[("card1",)],
        stats=["count", "sum"],
    )
    feat.fit(df)
    result = feat.transform(df)

    agg_cols = [c for c in result.columns if "card1_TransactionAmt" in c]
    assert len(agg_cols) > 0

    # Row 0 (A): no prior → count=0, sum=0
    r0 = result.iloc[0]
    assert r0["card1_TransactionAmt_count"] == 0
    assert r0["card1_TransactionAmt_sum"] == 0

    # Row 1 (A): prior=10 → count=1, sum=10
    r1 = result.iloc[1]
    assert r1["card1_TransactionAmt_count"] == 1
    assert r1["card1_TransactionAmt_sum"] == 10

    # Row 2 (A): prior sum=10+20=30, count=2
    r2 = result.iloc[2]
    assert r2["card1_TransactionAmt_count"] == 2
    assert r2["card1_TransactionAmt_sum"] == 30


def test_agg_leakage():
    """Shift protection: transform output must not include the current row's value."""
    df1 = pd.DataFrame(
        {
            "card1": ["A", "A", "A"],
            "TransactionAmt": [10.0, 20.0, 99999.0],
        }
    )
    df2 = pd.DataFrame(
        {
            "card1": ["A", "A", "A"],
            "TransactionAmt": [10.0, 20.0, 30.0],
        }
    )
    feat = AggregationFeature(
        agg_cols=["TransactionAmt"],
        group_keys=[("card1",)],
        stats=["sum"],
    )
    feat.fit(df1)
    r1 = feat.transform(df1)
    r2 = feat.transform(df2)

    # Row 2: both should use sum of rows 0+1 = 30 (shift-protected)
    assert r1.iloc[2]["card1_TransactionAmt_sum"] == 30.0
    assert r2.iloc[2]["card1_TransactionAmt_sum"] == 30.0


def test_agg_composite_key():
    df = pd.DataFrame(
        {
            "card1": ["A", "A", "B", "B"],
            "addr1": [1, 1, 1, 2],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        }
    )
    feat = AggregationFeature(
        agg_cols=["TransactionAmt"],
        group_keys=[("card1", "addr1")],
        stats=["count"],
    )
    feat.fit(df)
    result = feat.transform(df)

    col_name = "card1+addr1_TransactionAmt_count"
    assert col_name in result.columns

    # Row 0 (A,1): no prior → count=0
    assert result.iloc[0][col_name] == 0
    # Row 1 (A,1): prior count=1
    assert result.iloc[1][col_name] == 1
    # Row 2 (B,1): new group → count=0
    assert result.iloc[2][col_name] == 0
