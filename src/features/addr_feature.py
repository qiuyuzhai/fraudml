"""
AddrFeature — Address field features .

Missing flags for addr1/addr2 and combined addr1_addr2
composite categorical feature.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase


class AddrFeature(FeatureBase):
    """Address derived features.

    Stateless — fit() is a no-op.

    Features:
        - addr1_missing_flag: 1 if addr1 is null
        - addr2_missing_flag: 1 if addr2 is null
        - addr1_addr2: combined 'addr1@addr2' string
    """

    def __init__(self, name: str = "AddrFeature") -> None:
        super().__init__(name=name)

    @property
    def is_stateful(self) -> bool:
        return False

    def fit(self, df: pd.DataFrame) -> "AddrFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()

        for col in ["addr1", "addr2"]:
            if col in df.columns:
                df[f"{col}_missing_flag"] = df[col].isnull().astype(np.int8)
            else:
                df[f"{col}_missing_flag"] = 1

        if "addr1" in df.columns and "addr2" in df.columns:
            a1 = df["addr1"].fillna("NA").astype(str)
            a2 = df["addr2"].fillna("NA").astype(str)
            df["addr1_addr2"] = a1 + "@" + a2
        else:
            df["addr1_addr2"] = "NA@NA"

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "AddrFeature",
            "layer": "fraud-domain",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "AddrFeature",
                    "description": "Instance name.",
                },
            ],
            "example": "- AddrFeature",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "addr1_missing_flag",
                "addr2_missing_flag",
                "addr1_addr2",
            ],
            "physical_meaning": "Address metadata and composite identifier",
            "unit": "flag / string",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_addr_basic():
        df = pd.DataFrame({
            "addr1": [100, np.nan, 300], 
            "addr2": [np.nan, 200, 300],
        })
        feat = AddrFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "addr1_missing_flag" in result.columns
        assert "addr1_addr2" in result.columns

        assert result.iloc[0]["addr1_missing_flag"] == 0
        assert result.iloc[0]["addr2_missing_flag"] == 1
        assert result.iloc[1]["addr1_missing_flag"] == 1
        assert result.iloc[2]["addr1_addr2"] == "300.0@300.0"

    test_addr_basic()
    print("All AddrFeature tests passed!")