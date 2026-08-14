"""
MissingPatternFeature — Missing-value pattern features .

Missing value patterns carry strong fraud signal:
fraudulent transactions often have different missing-data signatures
than legitimate ones.

Stateful since v2: fit() records which columns had partial missing
in training so that transform() produces a stable feature set even
when the live data's missing pattern differs.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase


class MissingPatternFeature(FeatureBase):
    """Generate missing-value pattern features.

    **Stateful** — fit() records columns with partial missing (0 < rate < 1)
    so transform() always produces the same flag columns regardless
    of the live data's missing pattern.

    Outputs:
        - {col}_missing_flag: binary flag per recorded column
        - total_missing_count: per-row total missing count (scoped to
          training-known columns)
        - id_cols_missing_count: count of missing values in id_*
          columns known during training
    """

    @property
    def is_stateful(self) -> bool:
        return True

    def __init__(self, name: str = "MissingPatternFeature") -> None:
        super().__init__(name=name)
        self._flag_cols: List[str] = []
        self._id_cols: List[str] = []
        self._all_input_cols: List[str] = []

    def _get_state(self) -> Dict[str, Any]:
        return {
            "_flag_cols": self._flag_cols,
            "_id_cols": self._id_cols,
            "_all_input_cols": self._all_input_cols,
        }
    
    def fit(self, df: pd.DataFrame) -> "MissingPatternFeature":
        """Record which columns had partial missing in training.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        self : MissingPatternFeature
        """
        self._all_input_cols = list(df.columns)

        self._flag_cols = []
        for col in df.columns:
            missing_rate = df[col].isnull().mean()
            if missing_rate > 0 and missing_rate < 1.0:
                self._flag_cols.append(col)

        self._id_cols = [c for c in df.columns if c.startswith("id_")]

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate flags for training-recorded columns only.

        Always produces the same set of flag columns regardless of
        the live data's missing pattern.  Columns that were not
        recorded during fit() are never flagged.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (training / validation / online).

        Returns
        -------
        pd.DataFrame
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()

        flag_dict = {}
        for col in self._flag_cols:
            flag_name = f"{col}_missing_flag"
            if col in df.columns:
                flag_dict[flag_name] = df[col].isnull().astype(np.int8)
            else:
                flag_dict[flag_name] = np.zeros(len(df), dtype=np.int8)

        if flag_dict:
            flag_df = pd.DataFrame(flag_dict, index=df.index)
            df = pd.concat([df, flag_df], axis=1)

        known_cols = [c for c in self._all_input_cols if c in df.columns]
        if known_cols:
            df["total_missing_count"] = df[known_cols].isnull().sum(axis=1).astype(np.int32)
        else:
            df["total_missing_count"] = 0

        known_id_cols = [c for c in self._id_cols if c in df.columns]
        if known_id_cols:
            df["id_cols_missing_count"] = df[known_id_cols].isnull().sum(axis=1).astype(np.int32)
        else:
            df["id_cols_missing_count"] = 0

        return df

    def get_feature_metadata(self) -> Dict[str, Any]:
        flag_names = [f"{c}_missing_flag" for c in self._flag_cols]
        return {
            "feature_names": flag_names + ["total_missing_count", "id_cols_missing_count"],
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
        assert "id_02_missing_flag" in result.columns
        assert "total_missing_count" in result.columns
        assert "id_cols_missing_count" in result.columns

        assert result.iloc[0]["total_missing_count"] == 1
        assert result.iloc[0]["id_cols_missing_count"] == 1
        assert result.iloc[0]["id_01_missing_flag"] == 0
        assert result.iloc[0]["id_02_missing_flag"] == 1

        assert result.iloc[1]["total_missing_count"] == 2
        assert result.iloc[1]["id_cols_missing_count"] == 2

    def test_missing_all_nonnull_in_training():
        df = pd.DataFrame({
            "id_01": [1, 2, 3],
            "V1": [4, 5, 6],
        })
        feat = MissingPatternFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "id_01_missing_flag" not in result.columns
        assert result["total_missing_count"].sum() == 0
        assert result["id_cols_missing_count"].sum() == 0

    def test_missing_disappears_in_inference():
        train = pd.DataFrame({
            "id_01": [1.0, np.nan, 3.0],
            "V1": [7.0, 8.0, 9.0],
        })
        feat = MissingPatternFeature()
        feat.fit(train)

        inference = pd.DataFrame({
            "id_01": [1.0, 2.0, 3.0],
            "V1": [7.0, 8.0, 9.0],
        })
        result = feat.transform(inference)

        assert "id_01_missing_flag" in result.columns
        assert result["id_01_missing_flag"].sum() == 0

    def test_missing_appears_in_inference():
        train = pd.DataFrame({
            "id_01": [1.0, 2.0, 3.0],
            "V1": [7.0, 8.0, 9.0],
        })
        feat = MissingPatternFeature()
        feat.fit(train)

        inference = pd.DataFrame({
            "id_01": [1.0, np.nan, 3.0],
            "V1": [7.0, 8.0, 9.0],
        })
        result = feat.transform(inference)

        assert "id_01_missing_flag" not in result.columns

    def test_missing_new_column_in_inference():
        train = pd.DataFrame({
            "id_01": [1.0, np.nan, 3.0],
            "V1": [7.0, 8.0, 9.0],
        })
        feat = MissingPatternFeature()
        feat.fit(train)

        inference = pd.DataFrame({
            "id_01": [1.0, 2.0, 3.0],
            "V1": [7.0, 8.0, 9.0],
            "new_col": [np.nan, np.nan, np.nan],
        })
        result = feat.transform(inference)

        assert "new_col_missing_flag" not in result.columns
        assert "id_01_missing_flag" in result.columns

    def test_missing_column_gone_in_inference():
        train = pd.DataFrame({
            "id_01": [1.0, np.nan, 3.0],
            "id_02": [np.nan, 2.0, 3.0],
            "V1": [7.0, 8.0, 9.0],
        })
        feat = MissingPatternFeature()
        feat.fit(train)

        inference = pd.DataFrame({
            "id_01": [1.0, 2.0, 3.0],
            "V1": [7.0, 8.0, 9.0],
        })
        result = feat.transform(inference)

        assert "id_01_missing_flag" in result.columns
        assert "id_02_missing_flag" not in result.columns
        assert result["id_01_missing_flag"].sum() == 0

    test_missing_basic()
    test_missing_all_nonnull_in_training()
    test_missing_disappears_in_inference()
    test_missing_appears_in_inference()
    test_missing_new_column_in_inference()
    test_missing_column_gone_in_inference()
    print("All MissingPatternFeature tests passed!")