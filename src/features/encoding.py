"""
Categorical encoding features for the FeatureRegistry pipeline.

Contains:
    - CategoricalEncoder: simple LabelEncoder for object/category columns.
    - TargetEncoderFeature: Bayesian target encoding for high-cardinality
      columns with strict fit/transform separation to prevent target leakage.

      用普通编码还是目标编码还是后续决定的
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .base import FeatureBase


class CategoricalEncoder(FeatureBase):
    """Label-encode categorical columns.

    Detects columns with ``object`` or ``category`` dtype and fits a
    separate :class:`sklearn.preprocessing.LabelEncoder` per column.
    Unseen categories (in validation / test) are mapped to ``-1``
    so the pipeline never fails on out-of-vocabulary values.

    **Stateful** — learns LabelEncoder mappings from training data.

    Parameters
    ----------
    name : str
        Feature name passed to :class:`FeatureBase`.

    Attributes (set after ``fit()``)
    --------------------------------
    cat_cols_ : list[str]
        Columns that were encoded.
    encoders_ : dict[str, LabelEncoder]
        Fitted encoders keyed by column name.
    """

    @property
    def is_stateful(self) -> bool:
        return True

    def __init__(self, name: str = "CategoricalEncoder") -> None:
        super().__init__(name=name)

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        """Learn label encoders from **training data only**.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        self : CategoricalEncoder
        """
        self.cat_cols_: List[str] = [
            c for c in df.columns
            if df[c].dtype == "object" or str(df[c].dtype) == "category"
        ]

        self.encoders_: Dict[str, LabelEncoder] = {}
        for col in self.cat_cols_:
            le = LabelEncoder()
            series = self._prep_series(df[col])
            le.fit(series)
            self.encoders_[col] = le

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding. Unseen categories → -1.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to encode (train / validation / test).

        Returns
        -------
        pd.DataFrame
            Encoded DataFrame with original column names replaced
            by integer codes.
        """
        df = df.copy()
        for col, le in self.encoders_.items():
            if col not in df.columns:
                continue
            series = self._prep_series(df[col])
            known = dict(zip(le.classes_, le.transform(le.classes_)))
            df[col] = series.map(known).fillna(-1).astype(np.int32)
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prep_series(series: pd.Series) -> pd.Series:
        """Fill NaN with 'missing' and convert to string.

        Handles ``Categorical`` dtype safely by adding 'missing' to
        the category set before calling :meth:`fillna`.
        """
        if str(series.dtype) == "category":
            series = series.cat.add_categories("missing")
        return series.fillna("missing").astype(str)

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": self.cat_cols_ if hasattr(self, "cat_cols_") else [],
            "physical_meaning": "Label-encoded categorical columns",
            "unit": "integer_code",
            "depends_on_target": False,
        }


class TargetEncoderFeature(FeatureBase):
    """Bayesian target encoding for high-cardinality columns.

    Encodes categorical values with their smoothed target mean:

        encoded = (count * category_mean + m * global_mean) / (count + m)

    where ``m`` is the smoothing factor (default 100).  Categories
    with fewer than ``min_samples`` (default 20) training samples
    fall back to the global mean to prevent overfitting on tiny
    groups.

    **Stateful** — learns category→smoothed_target_mean mappings
    from training data.

    Anti-target-leakage design:
    - ``fit()`` computes category→encoded_value mapping from
      **training data only** and stores it as a dict.
    - ``transform()`` purely looks up pre-computed values — it never
      touches ``y`` (the target column).  Unseen categories fall
      back to ``global_mean``.
    - This strict separation means validation / test targets cannot
      influence encoding.  Compare with naive mean encoding which
      would leak target information.

    Parameters
    ----------
    target_col : str
        Target column name (e.g. 'isFraud').
    encode_cols : list of str
        Columns to target-encode.  Default: ['card1', 'addr1', 'P_emaildomain'].
    smoothing : float
        Bayesian smoothing factor ``m``.  Larger → more regularization.
    min_samples : int
        Minimum category sample count; below this, use global mean.
    """

    @property
    def is_stateful(self) -> bool:
        return True

    def __init__(
        self,
        name: str = "TargetEncoderFeature",
        target_col: str = "isFraud",
        encode_cols: Optional[List[str]] = None,
        smoothing: float = 100.0,
        min_samples: int = 20,
    ) -> None:
        super().__init__(name=name)
        self._target_col = target_col
        self._encode_cols = encode_cols or ["card1", "addr1", "P_emaildomain"]
        self._smoothing = smoothing
        self._min_samples = min_samples

    def _get_state(self) -> Dict[str, Any]:
        return {
            "_target_col": self._target_col,
            "_encode_cols": self._encode_cols,
            "_smoothing": self._smoothing,
            "_min_samples": self._min_samples,
            "_global_mean": self._global_mean,
            "_mappings": self._mappings,
        }

    def fit(self, df: pd.DataFrame) -> "TargetEncoderFeature":
        """Learn target encoding from **training data only**.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.  Must contain the target column.
            Rows where target is NaN are ignored (validation/test rows
            concatenated for time-series alignment).

        Returns
        -------
        self : TargetEncoderFeature
        """
        y = df[self._target_col].astype(float)
        valid_mask = y.notna()
        y_valid = y[valid_mask]

        self._global_mean = float(y_valid.mean()) if len(y_valid) > 0 else 0.0

        self._mappings: Dict[str, Dict[str, float]] = {}

        for col in self._encode_cols:
            if col not in df.columns:
                continue

            series = df.loc[valid_mask, col].fillna("missing").astype(str)
            counts = series.value_counts()
            means = y_valid.groupby(series).mean()

            mapping: Dict[str, float] = {}
            for cat, cnt in counts.items():
                if cnt < self._min_samples:
                    mapping[cat] = self._global_mean
                else:
                    cat_mean = means.get(cat, self._global_mean)
                    smoothed = (cnt * cat_mean + self._smoothing * self._global_mean) / (cnt + self._smoothing)
                    mapping[cat] = smoothed

            self._mappings[col] = mapping

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-computed target encoding.

        **Does NOT touch target y** — no target leakage possible.
        Unseen categories fall back to ``global_mean``.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (train / validation / test).

        Returns
        -------
        pd.DataFrame
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()

        for col, mapping in self._mappings.items():
            if col not in df.columns:
                continue
            series = df[col].fillna("missing").astype(str)
            df[f"{col}_target_enc"] = series.map(mapping).fillna(self._global_mean).astype(np.float32)

        return df

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [f"{c}_target_enc" for c in self._encode_cols],
            "physical_meaning": "Bayesian smoothed target mean per category",
            "unit": "probability",
            "depends_on_target": True,
        }


if __name__ == "__main__":
    def test_target_encoder_basic():
        df = pd.DataFrame({
            "card1": ["A", "A", "B", "B", "C"],
            "isFraud": [1, 0, 1, 1, 0],
        })
        feat = TargetEncoderFeature(target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0)
        feat.fit(df)
        result = feat.transform(df)

        assert "card1_target_enc" in result.columns
        # A: 1 fraud out of 2 → 0.5
        assert abs(result.iloc[0]["card1_target_enc"] - 0.5) < 0.01
        # B: 2 fraud out of 2 → 1.0
        assert abs(result.iloc[2]["card1_target_enc"] - 1.0) < 0.01

    def test_target_encoder_unseen_category():
        train = pd.DataFrame({
            "card1": ["A", "A", "B"],
            "isFraud": [1, 0, 1],
        })
        val = pd.DataFrame({
            "card1": ["C", "A"],
            "isFraud": [0, 0],
        })
        feat = TargetEncoderFeature(target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0)
        feat.fit(train)
        result = feat.transform(val)

        global_mean = train["isFraud"].mean()
        # 'C' is unseen → should fallback to global_mean
        assert abs(result.iloc[0]["card1_target_enc"] - global_mean) < 0.01
        # 'A' → 0.5 (stored from training)
        assert abs(result.iloc[1]["card1_target_enc"] - 0.5) < 0.01

    def test_target_encoder_min_samples():
        train = pd.DataFrame({
            "card1": ["A", "A", "B"],
            "isFraud": [1, 0, 1],
        })
        feat = TargetEncoderFeature(target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=10)
        feat.fit(train)
        result = feat.transform(train)

        # 'A' has 2 samples < 10 → should get global_mean
        global_mean = train["isFraud"].mean()
        assert abs(result.iloc[0]["card1_target_enc"] - global_mean) < 0.01
        # 'B' has 1 sample < 10 → should get global_mean
        assert abs(result.iloc[2]["card1_target_enc"] - global_mean) < 0.01

    def test_target_encoder_no_leakage():
        """Verify transform() does not use y from input df."""
        train = pd.DataFrame({
            "card1": ["A", "A", "B"],
            "isFraud": [1, 0, 1],
        })
        feat = TargetEncoderFeature(target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0)
        feat.fit(train)

        # Transform with different y — should NOT affect output
        val_diff_y = pd.DataFrame({
            "card1": ["A", "A", "B"],
            "isFraud": [99, 99, 99],  # completely different target values
        })
        result = feat.transform(val_diff_y)

        # Should still produce the stored encoding (0.5 for A, 1.0 for B)
        assert abs(result.iloc[0]["card1_target_enc"] - 0.5) < 0.01
        assert abs(result.iloc[2]["card1_target_enc"] - 1.0) < 0.01

    test_target_encoder_basic()
    test_target_encoder_unseen_category()
    test_target_encoder_min_samples()
    test_target_encoder_no_leakage()
    print("All TargetEncoderFeature tests passed!")