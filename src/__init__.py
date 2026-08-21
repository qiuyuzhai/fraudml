"""
FraudML — End-to-end fraud detection pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = ["__version__", "TrainPipeline"]


if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查，运行时不触发
    from .pipeline.train_pipeline import TrainPipeline


def __getattr__(name: str):  # PEP 562 lazy attribute
    if name == "TrainPipeline":
        from .pipeline.train_pipeline import TrainPipeline

        return TrainPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
