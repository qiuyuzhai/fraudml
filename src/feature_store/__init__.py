"""
FraudML Feature Store — SQLite-backed feature metadata / version / lineage.

Public API::

    from src.feature_store import FeatureStore

    store = FeatureStore("artifacts/feature_store.db")
    store.registry.register("amt_log", entity="transaction", ...)
    store.registry.get_feature("amt_log")
    store.registry.get_lineage("amt_log", recursive=True)
    store.versions.list_versions("amt_log")
    store.versions.rollback("amt_log", to_version=1)
"""

from .backend import SQLiteBackend
from .lineage import FeatureLineage
from .registry import FeatureRegistry
from .statistics import compute_all_stats, compute_distribution, compute_missing_rate
from .store import FeatureStore
from .versioning import FeatureVersion, VersionManager

__all__ = [
    "FeatureStore",
    "FeatureRegistry",
    "FeatureVersion",
    "VersionManager",
    "FeatureLineage",
    "SQLiteBackend",
    "compute_all_stats",
    "compute_distribution",
    "compute_missing_rate",
]
