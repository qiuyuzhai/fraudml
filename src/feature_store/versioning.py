"""
FeatureVersion + VersionManager — version lifecycle for the Feature Store.

Each feature may have multiple versions; only one row per feature is
``is_active=1`` at any time. ``VersionManager`` enforces this invariant
when activating / rolling back versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .backend import SQLiteBackend


@dataclass
class FeatureVersion:
    """A single version of a feature."""

    version_id: int
    feature_name: str
    version: int
    created_date: str
    is_active: bool
    schema_json: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "feature_name": self.feature_name,
            "version": self.version,
            "created_date": self.created_date,
            "is_active": self.is_active,
            "schema_json": self.schema_json,
            "run_id": self.run_id,
        }

    @classmethod
    def from_row(cls, row) -> "FeatureVersion":
        return cls(
            version_id=int(row["version_id"]),
            feature_name=str(row["feature_name"]),
            version=int(row["version"]),
            created_date=str(row["created_date"]),
            is_active=bool(row["is_active"]),
            schema_json=row["schema_json"],
            run_id=row["run_id"],
        )


class VersionManager:
    """Per-feature version numbering + activation / rollback lifecycle.

    Parameters
    ----------
    backend : SQLiteBackend
        Injected backend (shared with the parent :class:`FeatureStore`).
    """

    def __init__(self, backend: SQLiteBackend) -> None:
        self.backend = backend

    def next_version(self, feature_name: str) -> int:
        """Return the next available version number for *feature_name*."""
        row = self.backend.fetchone(
            "SELECT COALESCE(MAX(version), 0) AS max_v FROM feature_versions WHERE feature_name = ?",
            (feature_name,),
        )
        return int(row["max_v"] if row is not None else 0) + 1

    def create_version(
        self,
        feature_name: str,
        schema_json: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> FeatureVersion:
        """Insert a new version row (does NOT activate it; call :meth:`activate`)."""
        version = self.next_version(feature_name)
        created_date = datetime.now(timezone.utc).isoformat(timespec="seconds")
        version_id = self.backend.insert_and_get_id(
            "INSERT INTO feature_versions (feature_name, version, created_date, is_active, schema_json, run_id) "
            "VALUES (?, ?, ?, 0, ?, ?)",
            (feature_name, version, created_date, schema_json, run_id),
        )
        return FeatureVersion(
            version_id=version_id,
            feature_name=feature_name,
            version=version,
            created_date=created_date,
            is_active=False,
            schema_json=schema_json,
            run_id=run_id,
        )

    def activate(self, version_id: int) -> None:
        """Mark *version_id* as active; deactivate every other version of its feature.

        Implemented as two SQL updates inside one transaction (via
        :meth:`SQLiteBackend.connect`); safe to call repeatedly.
        """
        row = self.backend.fetchone(
            "SELECT feature_name FROM feature_versions WHERE version_id = ?",
            (version_id,),
        )
        if row is None:
            raise ValueError(f"version_id={version_id} not found")
        feature_name = str(row["feature_name"])

        with self.backend.connect() as conn:
            conn.execute(
                "UPDATE feature_versions SET is_active = 0 WHERE feature_name = ?",
                (feature_name,),
            )
            conn.execute(
                "UPDATE feature_versions SET is_active = 1 WHERE version_id = ?",
                (version_id,),
            )

    def rollback(self, feature_name: str, to_version: int) -> FeatureVersion:
        """Activate the specified historical version; pure flag flip, no data mutation."""
        row = self.backend.fetchone(
            "SELECT * FROM feature_versions WHERE feature_name = ? AND version = ?",
            (feature_name, to_version),
        )
        if row is None:
            raise ValueError(
                f"Feature '{feature_name}' has no version {to_version}"
            )
        version_id = int(row["version_id"])
        self.activate(version_id)
        return self.get_active(feature_name)

    def get_active(self, feature_name: str) -> Optional[FeatureVersion]:
        row = self.backend.fetchone(
            "SELECT * FROM feature_versions WHERE feature_name = ? AND is_active = 1",
            (feature_name,),
        )
        return FeatureVersion.from_row(row) if row is not None else None

    def list_versions(self, feature_name: str) -> list[FeatureVersion]:
        rows = self.backend.fetchall(
            "SELECT * FROM feature_versions WHERE feature_name = ? ORDER BY version ASC",
            (feature_name,),
        )
        return [FeatureVersion.from_row(r) for r in rows]
