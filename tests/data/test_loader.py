"""Tests for the data loader helpers (Parquet/CSV read + cache)."""

from pathlib import Path

import pandas as pd

from src.data.loader import read_parquet, to_parquet


def test_to_parquet_then_read_roundtrip(tmp_path: Path):
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1": ["A", "B", "A"],
        }
    )
    out = to_parquet(df, tmp_path / "tx.parquet")
    assert out.exists()

    loaded = read_parquet(out)
    assert len(loaded) == 3
    assert list(loaded["TransactionID"]) == [1, 2, 3]
    assert list(loaded["TransactionAmt"]) == [10.0, 20.0, 30.0]


def test_to_parquet_preserves_dtypes(tmp_path: Path):
    df = pd.DataFrame(
        {
            "id": pd.array([1, 2, 3], dtype="int64"),
            "amt": pd.array([1.5, 2.5, 3.5], dtype="float64"),
        }
    )
    out = to_parquet(df, tmp_path / "dt.parquet")
    loaded = read_parquet(out)
    assert loaded["id"].dtype == "int64"
    assert loaded["amt"].dtype == "float64"
