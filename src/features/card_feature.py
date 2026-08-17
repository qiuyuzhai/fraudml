"""
CardFeature — Bank card metadata features.

Missing flags for card2/card3/card5 and combined
card1_card2 composite categorical feature.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase


class CardFeature(FeatureBase):
    """Bank card derived features.

    Stateless — fit() is a no-op.

    Features:
        - card2_missing_flag: 1 if card2 is null
        - card3_missing_flag: 1 if card3 is null
        - card5_missing_flag: 1 if card5 is null

    Note: card1_card2 is intentionally omitted — CrossFeature
    already generates card1@card2 as part of its cross-pair set.
    """

    _CARD_COLS = ["card1", "card2", "card3", "card5"]

    def __init__(self, name: str = "CardFeature") -> None:
        super().__init__(name=name)

    @property
    def is_stateful(self) -> bool:
        return False
    
    def fit(self, df: pd.DataFrame) -> "CardFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()

        for col in ["card2", "card3", "card5"]:
            if col in df.columns:
                df[f"{col}_missing_flag"] = df[col].isnull().astype(np.int8)
            else:
                df[f"{col}_missing_flag"] = np.int8(1)

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "CardFeature",
            "layer": "fraud-domain",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "CardFeature",
                    "description": "Instance name.",
                },
            ],
            "example": "- CardFeature",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "card2_missing_flag",
                "card3_missing_flag",
                "card5_missing_flag",
            ],
            "physical_meaning": "Bank card metadata missing flags",
            "unit": "flag",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_card_basic():
        df = pd.DataFrame({
            "card1": [100, 200, 300],
            "card2": [1, np.nan, 3], 
            "card3": [np.nan, 2, 3],
            "card5": [5, 5, np.nan],
        })
        feat = CardFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "card2_missing_flag" in result.columns
        assert "card3_missing_flag" in result.columns
        assert "card5_missing_flag" in result.columns
        assert "card1_card2" not in result.columns

        assert result.iloc[0]["card2_missing_flag"] == 0
        assert result.iloc[1]["card2_missing_flag"] == 1
        assert result.iloc[2]["card3_missing_flag"] == 0

    test_card_basic()
    print("All CardFeature tests passed!")