"""Tests for TargetEncoderFeature.

Migrated from src/features/encoding.py __main__ — covers basic
encoding, unseen-category fallback to global mean, the min_samples
shrinkage guard, and the train/transform no-leakage invariant.
"""

import pandas as pd

from src.features.encoding import TargetEncoderFeature


def test_target_encoder_basic():
    df = pd.DataFrame(
        {
            "card1": ["A", "A", "B", "B", "C"],
            "isFraud": [1, 0, 1, 1, 0],
        }
    )
    feat = TargetEncoderFeature(
        target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0
    )
    feat.fit(df)
    result = feat.transform(df)

    assert "card1_target_enc" in result.columns
    # A: 1 fraud out of 2 → 0.5
    assert abs(result.iloc[0]["card1_target_enc"] - 0.5) < 0.01
    # B: 2 fraud out of 2 → 1.0
    assert abs(result.iloc[2]["card1_target_enc"] - 1.0) < 0.01


def test_target_encoder_unseen_category():
    train = pd.DataFrame(
        {
            "card1": ["A", "A", "B"],
            "isFraud": [1, 0, 1],
        }
    )
    val = pd.DataFrame(
        {
            "card1": ["C", "A"],
            "isFraud": [0, 0],
        }
    )
    feat = TargetEncoderFeature(
        target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0
    )
    feat.fit(train)
    result = feat.transform(val)

    global_mean = train["isFraud"].mean()
    # 'C' is unseen → should fallback to global_mean
    assert abs(result.iloc[0]["card1_target_enc"] - global_mean) < 0.01
    # 'A' → 0.5 (stored from training)
    assert abs(result.iloc[1]["card1_target_enc"] - 0.5) < 0.01


def test_target_encoder_min_samples():
    train = pd.DataFrame(
        {
            "card1": ["A", "A", "B"],
            "isFraud": [1, 0, 1],
        }
    )
    feat = TargetEncoderFeature(
        target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=10
    )
    feat.fit(train)
    result = feat.transform(train)

    # 'A' has 2 samples < 10 → should get global_mean
    global_mean = train["isFraud"].mean()
    assert abs(result.iloc[0]["card1_target_enc"] - global_mean) < 0.01
    # 'B' has 1 sample < 10 → should get global_mean
    assert abs(result.iloc[2]["card1_target_enc"] - global_mean) < 0.01


def test_target_encoder_no_leakage():
    """transform() must not use the target column from the input df."""
    train = pd.DataFrame(
        {
            "card1": ["A", "A", "B"],
            "isFraud": [1, 0, 1],
        }
    )
    feat = TargetEncoderFeature(
        target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0
    )
    feat.fit(train)

    # Transform with different y — should NOT affect output
    val_diff_y = pd.DataFrame(
        {
            "card1": ["A", "A", "B"],
            "isFraud": [99, 99, 99],  # completely different target values
        }
    )
    result = feat.transform(val_diff_y)

    # Should still produce the stored encoding (0.5 for A, 1.0 for B)
    assert abs(result.iloc[0]["card1_target_enc"] - 0.5) < 0.01
    assert abs(result.iloc[2]["card1_target_enc"] - 1.0) < 0.01
