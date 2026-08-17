"""
VelocityFeature — Transaction velocity and acceleration features.

Combines outputs from AmountFeature and HistoryFeature:
    - card1_amount_delta (from AmountFeature)
    - time_since_last_transaction (from HistoryFeature)

Derives:
    - velocity:  amount_delta / time_delta  (consumption speed)
    - acceleration: delta(velocity) / time_delta  (change in speed)

Must run AFTER AmountFeature and HistoryFeature in the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase


class VelocityFeature(FeatureBase):
    """Transaction velocity (consumption speed) and acceleration.

    **Stateless** — fit() is a no-op.  Reads columns produced by
    upstream features (AmountFeature, HistoryFeature).

    Features:
        - card1_velocity: card1_amount_delta / time_since_last_transaction
        - card1_acceleration: delta(velocity) / time_since_last_transaction
    """

    @property
    def is_stateful(self) -> bool:
        return False

    def __init__(self, name: str = "VelocityFeature") -> None:
        super().__init__(name=name)
        self._group_col: str = "card1"
        self._amount_delta_col: str = "card1_amount_delta"
        self._time_delta_col: str = "time_since_last_transaction"

    def fit(self, df: pd.DataFrame) -> "VelocityFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        g = df.groupby(self._group_col, sort=False)

        amt_delta = df[self._amount_delta_col].fillna(0).astype(np.float64)
        time_delta = df[self._time_delta_col].fillna(0).astype(np.float64)

        safe_time = time_delta.copy()
        safe_time[safe_time == 0] = np.inf

        velocity = amt_delta / safe_time
        velocity = velocity.replace([np.inf, -np.inf], 0.0).fillna(0)
        df["card1_velocity"] = velocity.astype(np.float32)

        prev_velocity = g["card1_velocity"].shift(1)
        delta_velocity = df["card1_velocity"] - prev_velocity.fillna(0)

        safe_time_acc = time_delta.copy()
        safe_time_acc[safe_time_acc == 0] = np.inf

        acceleration = delta_velocity / safe_time_acc
        acceleration = acceleration.replace([np.inf, -np.inf], 0.0).fillna(0)
        df["card1_acceleration"] = acceleration.astype(np.float32)

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "VelocityFeature",
            "layer": "fraud-domain",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "VelocityFeature",
                    "description": "Instance name.",
                },
            ],
            "example": """# Must run AFTER AmountFeature and HistoryFeature:
feature_steps:
  - AmountFeature
  - HistoryFeature
  - VelocityFeature""",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "card1_velocity",
                "card1_acceleration",
            ],
            "physical_meaning": "Consumption speed and its rate of change per card",
            "unit": "usd/s / usd/s^2",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_velocity_basic():
        df = pd.DataFrame({
            "card1": ["A", "A", "A", "B", "B"],
            "TransactionAmt": [10.0, 30.0, 20.0, 5.0, 15.0],
            "card1_amount_delta": [0.0, 20.0, -10.0, 0.0, 10.0],
            "time_since_last_transaction": [0.0, 1000.0, 500.0, 0.0, 2000.0],
        })
        feat = VelocityFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "card1_velocity" in result.columns
        assert "card1_acceleration" in result.columns

        r0 = result.iloc[0]
        assert r0["card1_velocity"] == 0.0
        assert r0["card1_acceleration"] == 0.0

        r1 = result.iloc[1]
        assert abs(r1["card1_velocity"] - 0.02) < 0.01

        r2 = result.iloc[2]
        expected_v = -10.0 / 500.0
        assert abs(r2["card1_velocity"] - expected_v) < 0.01

    def test_velocity_division_by_zero():
        df = pd.DataFrame({
            "card1": ["A", "A"],
            "TransactionAmt": [10.0, 20.0],
            "card1_amount_delta": [0.0, 10.0],
            "time_since_last_transaction": [0.0, 0.0],
        })
        feat = VelocityFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert result.iloc[0]["card1_velocity"] == 0.0
        assert result.iloc[1]["card1_velocity"] == 0.0
        assert result.iloc[0]["card1_acceleration"] == 0.0
        assert result.iloc[1]["card1_acceleration"] == 0.0

    def test_velocity_leakage():
        df1 = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionAmt": [10.0, 20.0, 99999.0],
            "card1_amount_delta": [0.0, 10.0, 99979.0],
            "time_since_last_transaction": [0.0, 1000.0, 2000.0],
        })
        df2 = pd.DataFrame({
            "card1": ["A", "A", "A"],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1_amount_delta": [0.0, 10.0, 10.0],
            "time_since_last_transaction": [0.0, 1000.0, 2000.0],
        })
        feat = VelocityFeature()
        feat.fit(df1)
        r1 = feat.transform(df1)
        r2 = feat.transform(df2)

        assert r1.iloc[0]["card1_velocity"] == r2.iloc[0]["card1_velocity"]
        assert abs(r1.iloc[1]["card1_velocity"] - r2.iloc[1]["card1_velocity"]) < 0.01

    test_velocity_basic()
    test_velocity_division_by_zero()
    test_velocity_leakage()
    print("All VelocityFeature tests passed!")