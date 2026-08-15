"""
FeatureCatalog — Centralized feature metadata management.

所有特征的户口本;类似Feast的FeatureCatalog

Collects metadata from all feature engineering steps and exports
a unified catalog for Feast-compatible feature registration.

This is a critical bridge between the local pandas pipeline and
future migration to Spark/Hive/Feast:
- Feast uses feature metadata to build FeatureViews
- Hive uses feature names/types to define table schemas
- Spark uses feature metadata for schema inference

Usage::

    catalog = FeatureCatalog()
    for feat in registry._instances.values():
        catalog.register(feat)
    catalog.export("artifacts/run_20260813_143052_a1b2c3/offline_features/feature_catalog.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import FeatureBase


class FeatureCatalog:
    """Collects and exports feature metadata from all pipeline steps.

    Parameters
    ----------
    name : str
        Catalog identifier.
    """

    def __init__(self, name: str = "fraudml_catalog") -> None:
        self.name = name
        self._entries: List[Dict[str, Any]] = []

    def register(self, feature: FeatureBase, **kwargs: Any) -> None:
        """Register a single feature's metadata.

        Parameters
        ----------
        feature : FeatureBase
            A fitted feature engineering step.
        **kwargs
            Extra metadata (e.g. source config params).
        """
        metadata = feature.get_feature_metadata()

        entry: Dict[str, Any] = {
            "source": feature.name,
            "feature_names": metadata.get("feature_names", []),
            "physical_meaning": metadata.get("physical_meaning", ""),
            "unit": metadata.get("unit", ""),
            "depends_on_target": metadata.get("depends_on_target", False),
            "is_stateful": getattr(feature, "is_stateful", False),
            "fitted": getattr(feature, "_fitted", False),
        }
        entry.update(kwargs)
        self._entries.append(entry)

    def register_all(self, features: List[FeatureBase]) -> None:
        """Register multiple features at once.

        Parameters
        ----------
        features : list of FeatureBase
            Fitted feature instances.
        """
        for feat in features:
            self.register(feat)

    def get_all_feature_names(self) -> List[str]:
        """Return a flat list of all output feature names."""
        names: List[str] = []
        for entry in self._entries:
            names.extend(entry.get("feature_names", []))
        return names

    def get_stateful_features(self) -> List[Dict[str, Any]]:
        """Return entries for stateful features only."""
        return [e for e in self._entries if e.get("is_stateful", False)]

    def get_target_dependent_features(self) -> List[Dict[str, Any]]:
        """Return entries for features that depend on target y."""
        return [e for e in self._entries if e.get("depends_on_target", False)]

    def to_dataframe(self) -> pd.DataFrame:
        """Export catalog as a DataFrame."""
        rows: List[Dict[str, Any]] = []
        for entry in self._entries:
            for fname in entry.get("feature_names", []):
                row = {
                    "feature_name": fname,
                    "source": entry.get("source", ""),
                    "physical_meaning": entry.get("physical_meaning", ""),
                    "unit": entry.get("unit", ""),
                    "depends_on_target": entry.get("depends_on_target", False),
                    "is_stateful": entry.get("is_stateful", False),
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize catalog to a JSON-safe dict."""
        return {
            "catalog_name": self.name,
            "total_entries": len(self._entries),
            "entries": self._entries,
            "all_feature_names": self.get_all_feature_names(),
            "stateful_feature_sources": [
                e["source"] for e in self.get_stateful_features()
            ],
            "target_dependent_sources": [
                e["source"] for e in self.get_target_dependent_features()
            ],
        }

    def export(self, path: str | Path) -> Path:
        """Export catalog to JSON file (Feast-compatible format).

        Parameters
        ----------
        path : str | Path
            Destination file path.

        Returns
        -------
        Path
            Resolved save path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCatalog":
        """Load a previously exported catalog.

        Parameters
        ----------
        path : str | Path
            Path to the JSON file.

        Returns
        -------
        FeatureCatalog
            Loaded catalog instance.
        """
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)

        catalog = cls(name=data.get("catalog_name", "loaded_catalog"))
        catalog._entries = data.get("entries", [])
        return catalog

    def __repr__(self) -> str:
        total = len(self._entries)
        features = len(self.get_all_feature_names())
        stateful = len(self.get_stateful_features())
        return (
            f"FeatureCatalog(name='{self.name}', "
            f"entries={total}, features={features}, stateful_sources={stateful})"
        )