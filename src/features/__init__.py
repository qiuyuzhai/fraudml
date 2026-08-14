"""
Feature engineering module for FraudML.

Provides the abstract FeatureBase contract and FeatureRegistry
for managing multiple feature-engineering steps with config-driven
execution order.

Concrete feature classes are auto-discovered by FeatureRegistry
via :meth:`FeatureRegistry.auto_discover`.
"""

from .base import FeatureBase
from .registry import FeatureRegistry
from .feature_catalog import FeatureCatalog
from .encoding import CategoricalEncoder, TargetEncoderFeature
from .history_feature import HistoryFeature
from .missing_pattern_feature import MissingPatternFeature
from .amount_feature import AmountFeature
from .time_feature import TimeFeature
from .device_feature import DeviceFeature
from .email_feature import EmailFeature
from .card_feature import CardFeature
from .addr_feature import AddrFeature
from .aggregation_feature import AggregationFeature
from .cross_feature import CrossFeature

__all__ = [
    "FeatureBase",
    "FeatureRegistry",
    "FeatureCatalog",
    "CategoricalEncoder",
    "TargetEncoderFeature",
    "HistoryFeature",
    "MissingPatternFeature",
    "AmountFeature",
    "TimeFeature",
    "DeviceFeature",
    "EmailFeature",
    "CardFeature",
    "AddrFeature",
    "AggregationFeature",
    "CrossFeature",
]