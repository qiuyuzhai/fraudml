"""Test FeatureBase + FeatureRegistry with a simple end-to-end pipeline.

Uses auto_discover to find sample feature classes inside src.features.
"""
import sys, os
sys.path.insert(0, r"d:\fraudml")
os.chdir(r"d:\fraudml")

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.features.base import FeatureBase
from src.features.registry import FeatureRegistry


# ── 1. Create test data ──
np.random.seed(42)
df = pd.DataFrame({
    "a": np.random.randn(100),
    "b": np.random.randint(0, 100, 100).astype(float),
    "c": np.where(np.random.rand(100) < 0.2, np.nan, np.random.randn(100)),
})
print(f"Input: {df.shape}")

# ── 2. Write config.yaml ──
config = {"feature_steps": ["DoubleFeature", "ThresholdFlagFeature"]}
Path("config.yaml").write_text(yaml.dump(config))
print(f"Config written: {config}")

# ── 3. Discover + configure + run ──
registry = FeatureRegistry()
discovered = registry.auto_discover("src.features")
print(f"Discovered: {[c.__name__ for c in discovered]}")

configured = registry.configure("config.yaml")
print(f"Configured: {configured}")

result = registry.fit_transform_all(df)
print(f"Output: {result.shape}")
print(f"Result columns: {list(result.columns)}")

# ── 4. Verify correctness ──
assert "a" in result.columns
assert "a_high" in result.columns
assert result["a"].iloc[0] == df["a"].iloc[0] * 2
assert result["a_high"].sum() > 0
print("Correctness checks passed.")

# ── 5. Save + load round-trip ──
registry.save_all("artifacts/features")

registry2 = FeatureRegistry()
registry2.auto_discover("src.features")
registry2.configure("config.yaml")
registry2.load_all("artifacts/features")

result2 = registry2.transform_all(df)
assert result.equals(result2), "FAIL: save/load round-trip differs!"
print("save/load round-trip OK.")

# ── 6. Verify isolation: val data doesn't affect fit ──
val_df = pd.DataFrame({"a": [1000.0, -1000.0], "b": [1.0, 2.0], "c": [0.5, 0.5]})
val_result = registry.transform_all(val_df)
print(f"Val transform output: {val_result.shape}")
print(f"  a_high for extreme val row 1000: {val_result['a_high'].iloc[0]}")
print("Val isolation check passed.")

print("\nALL CHECKS PASSED")