"""Shared pytest fixtures for the FraudML test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal IEEE-CIS-shaped transaction DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [3600, 43200, 75600, 86400, 90000],
            "TransactionAmt": [10.0, 20.0, 30.0, 5.0, 15.0],
            "ProductCD": ["W", "H", "W", "C", "R"],
            "card1": ["A", "A", "A", "B", "B"],
            "card4": ["visa", "mastercard", "visa", "amex", "visa"],
            "addr1": [1, 1, 2, 3, 3],
            "isFraud": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def tmp_feature_store_db(tmp_path: Path) -> Path:
    """Path for a per-test SQLite feature store database."""
    return tmp_path / "feature_store.db"


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    """Empty per-test artifact directory."""
    d = tmp_path / "artifacts" / "run_test"
    d.mkdir(parents=True)
    return d
