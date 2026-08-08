"""
MissingPatternFeature — Missing-value pattern features .

Missing value patterns carry strong fraud signal:
fraudulent transactions often have different missing-data signatures
than legitimate ones.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase


class MissingPatternFeature(FeatureBase):
    """Generate missing-value pattern features.

    Stateless — fit() is a no-op.

    Outputs:
        - {col}_missing_flag: binary flag per input column
        - total_missing_count: per-row total missing count
        - id_cols_missing_count: count of missing values in id_* columns
    """

    def __init__(self, name: str = "MissingPatternFeature") -> None:
        super().__init__(name=name)

    def fit(self, df: pd.DataFrame) -> "MissingPatternFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()

        # --- Per-column missing flags ---
        for col in df.columns:
            df[f"{col}_missing_flag"] = df[col].isnull().astype(np.int8)

        # --- Total missing count per row ---
        df["total_missing_count"] = df.isnull().sum(axis=1).astype(np.int32)

        # --- id_cols missing count ---
        id_cols = [c for c in df.columns if c.startswith("id_")]
        if id_cols:
            df["id_cols_missing_count"] = df[id_cols].isnull().sum(axis=1).astype(np.int32)
        else:
            df["id_cols_missing_count"] = 0

        return df

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "{col}_missing_flag",
                "total_missing_count",
                "id_cols_missing_count",
            ],
            "physical_meaning": "Missing-value pattern per row",
            "unit": "flag / count",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_missing_basic():
        df = pd.DataFrame({
            "id_01": [1.0, np.nan, 3.0],
            "id_02": [np.nan, np.nan, 6.0],
            "V1": [7.0, 8.0, 9.0],
        })
        feat = MissingPatternFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "id_01_missing_flag" in result.columns
        assert "total_missing_count" in result.columns
        assert "id_cols_missing_count" in result.columns

        # Row 0: id_01 not null, id_02 null → total=1, id_missing=1
        assert result.iloc[0]["total_missing_count"] == 1
        assert result.iloc[0]["id_cols_missing_count"] == 1
        assert result.iloc[0]["id_01_missing_flag"] == 0
        assert result.iloc[0]["id_02_missing_flag"] == 1

        # Row 1: id_01 null, id_02 null → total=2, id_missing=2
        assert result.iloc[1]["total_missing_count"] == 2
        assert result.iloc[1]["id_cols_missing_count"] == 2

    def test_missing_all_nonnull():
        df = pd.DataFrame({
            "id_01": [1, 2, 3],
            "V1": [4, 5, 6],
        })
        feat = MissingPatternFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert result["total_missing_count"].sum() == 0
        assert result["id_cols_missing_count"].sum() == 0

    test_missing_basic()
    test_missing_all_nonnull()
    print("All MissingPatternFeature tests passed!")