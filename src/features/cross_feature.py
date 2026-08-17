"""
CrossFeature — Categorical cross features.

Creates combined category identifiers by concatenating pairs
of categorical columns (e.g. card1@addr1, device@emaildomain).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase


class CrossFeature(FeatureBase):
    """Categorical cross / concatenation features.

    Stateless — fit() is a no-op.

    Parameters
    ----------
    cross_pairs : list of (col_a, col_b, output_name) tuples
        Each tuple specifies two columns to concatenate and the
        output column name.  If None, defaults are used.
    """

    _DEFAULT_PAIRS = [
        ("card1", "addr1"),
        ("card1", "card2"),
        ("P_emaildomain", "addr1"),
    ]

    def __init__(
        self,
        name: str = "CrossFeature",
        cross_pairs: List[tuple] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._cross_pairs = cross_pairs or self._DEFAULT_PAIRS

    @property
    def is_stateful(self) -> bool:
        return False

    @property
    def _col_suffix(self) -> str:
        if self.name == self.__class__.__name__:
            return ""
        return f"_{self.name}"
    
    @staticmethod
    def _safe_cross_str(s: pd.Series) -> pd.Series:
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            s = s.astype("Int64")
        s = s.astype(str).replace("<NA>", "NA")
        s = s.replace("nan", "NA")
        return s

    def fit(self, df: pd.DataFrame) -> "CrossFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        suffix = self._col_suffix

        for col_a, col_b in self._cross_pairs:
            out_name = f"{col_a}@{col_b}{suffix}"
            if col_a not in df.columns or col_b not in df.columns:
                continue
            a = self._safe_cross_str(df[col_a])
            b = self._safe_cross_str(df[col_b])
            df[out_name] = a + "@" + b
        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "CrossFeature",
            "layer": "fraud-domain",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "CrossFeature",
                    "description": "Instance name. Use unique name for multiple cross pair groups.",
                },
                {
                    "name": "cross_pairs",
                    "type": "list[list[str]]",
                    "default": [["card1", "addr1"], ["card1", "card2"], ["P_emaildomain", "addr1"]],
                    "description": "List of column pairs to cross. Each pair is [col_a, col_b].",
                },
            ],
            "example": """# Single instance (default pairs):
- CrossFeature

# Multiple instances with different pair groups:
- CrossFeature:
    name: "CrossFeature_card_addr"
    cross_pairs:
      - ["card1", "addr1"]
      - ["card1", "card2"]
      - ["card1", "addr2"]
- CrossFeature:
    name: "CrossFeature_email_device"
    cross_pairs:
      - ["P_emaildomain", "DeviceType"]
      - ["P_emaildomain", "id_30"]""",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        suffix = self._col_suffix
        return {
            "feature_names": [f"{a}@{b}{suffix}" for a, b in self._cross_pairs],
            "physical_meaning": "Cross-categorical composite identifiers",
            "unit": "string",
            "depends_on_target": False,
        }

    

if __name__ == "__main__":
    def test_cross_basic():
        df = pd.DataFrame({
            "card1": [100, 200, 300],
            "addr1": [1, 2, 3],
            "card2": [10, 20, 30],
            "P_emaildomain": ["gmail.com", "yahoo.com", "hotmail.com"],
        })
        feat = CrossFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "card1@addr1" in result.columns
        assert "card1@card2" in result.columns
        assert "P_emaildomain@addr1" in result.columns

        assert result.iloc[0]["card1@addr1"] == "100@1"
        assert result.iloc[1]["card1@card2"] == "200@20"
        assert result.iloc[2]["P_emaildomain@addr1"] == "hotmail.com@3"

    def test_cross_missing_cols():
        df = pd.DataFrame({"card1": [1, 2], "addr1": [3, 4]})
        feat = CrossFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "card1@addr1" in result.columns
        assert "card1@card2" not in result.columns

    test_cross_basic()
    test_cross_missing_cols()
    print("All CrossFeature tests passed!")