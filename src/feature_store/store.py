"""
FeatureStore — facade composing the Feature Store components.

Exposes :attr:`registry` (the high-level API) and the lower-level
:attr:`versions` / :attr:`lineage` / :attr:`backend` for advanced
queries. Use this as the single entry point from the train pipeline
and any external tooling.
"""

from __future__ import annotations

from pathlib import Path

from .backend import SQLiteBackend
from .lineage import FeatureLineage
from .registry import FeatureRegistry
from .versioning import VersionManager


class FeatureStore:
    """Facade over the SQLite-backed Feature Store.

    Parameters
    ----------
    db_path : str | Path
        SQLite database path. ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str | Path = "artifacts/feature_store.db") -> None:
        self.backend = SQLiteBackend(db_path)
        self.versions = VersionManager(self.backend)
        self.lineage = FeatureLineage(self.backend, self.versions)
        self.registry = FeatureRegistry.__new__(FeatureRegistry)
        # Reuse the same backend/versions/lineage instance triple
        self.registry.backend = self.backend
        self.registry.versions = self.versions
        self.registry.lineage = self.lineage

    def __repr__(self) -> str:
        return f"FeatureStore(db_path={self.backend.db_path!r})"
