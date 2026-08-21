"""Regression tests for the train/val data-leakage fix (Task 10).

The fix separated feature engineering so that fit runs only on the
training split and the validation split is transformed with the
state learned from training alone. These tests assert the invariant
directly against the stateful feature components the fix targets,
without spinning up the full TrainPipeline (which has heavy deps).
"""

import pandas as pd

from src.features.aggregation_feature import AggregationFeature


def _interleaved_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/val splits with interleaved TransactionDT and shared card1.

    Both splits share the card1 group 'A' so that any cross-split
    groupby leak would surface in the val aggregation values.
    """
    train = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [3600, 43200, 75600],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1": ["A", "A", "A"],
        }
    )
    val = pd.DataFrame(
        {
            "TransactionID": [4, 5],
            "TransactionDT": [86400, 90000],
            "TransactionAmt": [999.0, 1000.0],
            "card1": ["A", "A"],
        }
    )
    return train, val


def test_val_aggregation_does_not_include_train_amounts():
    """fit on train, transform val — val row 0 must NOT see train sums."""
    train, val = _interleaved_split()

    feat = AggregationFeature(
        agg_cols=["TransactionAmt"],
        group_keys=[("card1",)],
        stats=["sum", "count"],
    )
    feat.fit(train)

    val_result = feat.transform(val)

    # Val row 0 has card1='A'. If the registry leaked train state, the
    # sum column would carry train's 10+20+30=60. The fix resets state
    # per transform() call so the first val row sees count=0 / sum=0.
    assert val_result.iloc[0]["card1_TransactionAmt_count"] == 0
    assert val_result.iloc[0]["card1_TransactionAmt_sum"] == 0

    # Val row 1 sees only the prior val row (999), NOT any train amount.
    assert val_result.iloc[1]["card1_TransactionAmt_count"] == 1
    assert val_result.iloc[1]["card1_TransactionAmt_sum"] == 999.0


def test_train_aggregation_does_not_include_future_train_amounts():
    """Within train itself, shift protection still applies — row i sees
    only rows 0..i-1, never the current or later rows."""
    train, _ = _interleaved_split()

    feat = AggregationFeature(
        agg_cols=["TransactionAmt"],
        group_keys=[("card1",)],
        stats=["sum"],
    )
    feat.fit(train)
    result = feat.transform(train)

    # Row 0: no prior → 0
    assert result.iloc[0]["card1_TransactionAmt_sum"] == 0
    # Row 1: sees row 0 only → 10
    assert result.iloc[1]["card1_TransactionAmt_sum"] == 10
    # Row 2: sees rows 0+1 → 30, NOT 60 (would leak the current row)
    assert result.iloc[2]["card1_TransactionAmt_sum"] == 30


def test_val_row_count_preserved_after_separate_transform():
    """Row-count invariant the train pipeline asserts after FE."""
    train, val = _interleaved_split()
    feat = AggregationFeature(
        agg_cols=["TransactionAmt"],
        group_keys=[("card1",)],
        stats=["sum"],
    )
    feat.fit(train)

    val_result = feat.transform(val)
    assert len(val_result) == len(val)
