"""
FraudML — End-to-End Training Pipeline (Hydra-driven)

Usage (Windows PowerShell):
    python src/train.py --config-name config
    python src/train.py --config-name config "selection.iv_threshold=0.01"
    python src/train.py --config-name config "model.params.learning_rate=0.03"
    python src/train.py --config-name config "checkpoint.force_refresh=true"

Note: In PowerShell, Hydra override keys containing dots (e.g. model.n_trials)
MUST be wrapped in quotes to avoid being parsed as property access.

High-level flow:
    Load → Time Split → Merge Identity → Profile → Clean
    → Encode + Feature Generation → Feature Catalog
    → Feature Selection (IV/VIF/PSI) → Model Training
    → Calibration → Risk Decision → Threshold Optimization → Save
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pipeline.train_pipeline import TrainPipeline


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for FraudML training.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration loaded from configs/lgb_full.yaml
        (or whichever config_name is specified).
    """
    cfg_dict = dict(cfg)

    force_refresh = cfg_dict.get("checkpoint", {}).get("force_refresh", False)
    if force_refresh:
        print("[Force Refresh] Will ignore checkpoint hashes in current run directory.")

    pipeline = TrainPipeline(cfg_dict)

    pipeline.fit()
    pipeline.evaluate()
    pipeline.save()


if __name__ == "__main__":
    main()