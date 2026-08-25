"""Customer segmentation engine for value-based customer classification."""

from src.segmentation.customer_segmentation import (
    CustomerSegmentationEngine,
    SegmentationResult,
    create_default_segmentation_engine,
)

__all__ = [
    "CustomerSegmentationEngine",
    "SegmentationResult",
    "create_default_segmentation_engine",
]