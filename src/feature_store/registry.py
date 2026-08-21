"""
FeatureRegistry — high-level Feature Store API.

Composes :class:`SQLiteBackend`, :class:`VersionManager`,
:class:`FeatureLineage`, and the statistics helpers into the public
``register / get_feature / list_features / get_lineage / archive /
record_statistics`` interface documented in the project plan.

Typical workflow::

    store = FeatureStore("artifacts/feature_store.db")
    store.registry.register(
        "amt_log", entity="transaction", feature_type="numeric",
        description="log1p(TransactionAmt)",
        raw_columns=["TransactionAmt"],
    )
    store.registry.record_statistics("amt_log", df, target="isFraud")
    store.registry.get_feature("amt_log")           # active version + stats + lineage
    store.registry.get_lineage("amt_log", recursive=True)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from .backend import SQLiteBackend
from .lineage import FeatureLineage
from .statistics import compute_all_stats, _is_nan_or_none
from .versioning import FeatureVersion, VersionManager


class FeatureRegistry:
    """High-level Feature Store API.

    Parameters
    ----------
    db_path : str | Path
        SQLite database path. ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str | Path = "artifacts/feature_store.db") -> None:
        self.backend = SQLiteBackend(db_path)
        self.versions = VersionManager(self.backend)
        self.lineage = FeatureLineage(self.backend, self.versions)

    # ------------------------------------------------------------------
    # Register / archive
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        entity: str,
        feature_type: str = "numeric",
        description: str = "",
        owner: str = "system",
        raw_columns: Optional[Iterable[str]] = None,
        upstream_features: Optional[Iterable[str]] = None,
        schema_meta: Optional[dict] = None,
        run_id: Optional[str] = None,
    ) -> FeatureVersion:
        """Register (or re-version) a feature.

        Idempotent on the *feature* row (insert-or-ignore), always
        creates a new version row, activates it, and writes lineage edges
        from the new version to its raw-column + upstream-feature sources.
        """
        created_date = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self.backend.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO features "
                "(name, entity, description, owner, feature_type, created_date, is_archived) "
                "VALUES (?, ?, ?, ?, ?, ?, 0)",
                (name, entity, description, owner, feature_type, created_date),
            )
            # Refresh description / owner on subsequent registrations
            conn.execute(
                "UPDATE features SET description = ?, owner = ?, feature_type = ? WHERE name = ?",
                (description, owner, feature_type, name),
            )

        schema_json = json.dumps(schema_meta, default=str) if schema_meta else None
        version = self.versions.create_version(
            feature_name=name,
            schema_json=schema_json,
            run_id=run_id,
        )
        self.versions.activate(version.version_id)

        sources: list[tuple[str, str]] = []
        if raw_columns:
            sources.extend(("raw_column", col) for col in raw_columns)
        if upstream_features:
            sources.extend(("feature", feat) for feat in upstream_features)
        if sources:
            self.lineage.add_edges(version.version_id, sources)

        return version

    def archive(self, name: str) -> None:
        """Mark *name* as archived (soft-delete; data preserved for audit)."""
        self.backend.execute(
            "UPDATE features SET is_archived = 1 WHERE name = ?", (name,)
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_feature(self, name: str, version: Optional[int] = None) -> dict:
        """Return feature metadata + active (or specified) version + statistics.

        Parameters
        ----------
        name : str
            Feature name.
        version : int, optional
            Specific version to fetch. ``None`` (default) returns the
            active version.

        Raises
        ------
        KeyError
            If the feature or requested version does not exist.
        """
        feat_row = self.backend.fetchone(
            "SELECT * FROM features WHERE name = ?", (name,)
        )
        if feat_row is None:
            raise KeyError(f"Feature '{name}' not registered")

        if version is None:
            v = self.versions.get_active(name)
        else:
            row = self.backend.fetchone(
                "SELECT * FROM feature_versions WHERE feature_name = ? AND version = ?",
                (name, version),
            )
            v = FeatureVersion.from_row(row) if row is not None else None

        if v is None:
            raise KeyError(f"Feature '{name}' has no matching version")

        stats_row = self.backend.fetchone(
            "SELECT * FROM statistics WHERE version_id = ?", (v.version_id,)
        )

        return {
            "name": str(feat_row["name"]),
            "entity": str(feat_row["entity"]),
            "description": str(feat_row["description"]),
            "owner": str(feat_row["owner"]),
            "feature_type": str(feat_row["feature_type"]),
            "is_archived": bool(feat_row["is_archived"]),
            "version": v.to_dict(),
            "statistics": dict(stats_row) if stats_row is not None else None,
        }

    def list_features(
        self, entity: Optional[str] = None, include_archived: bool = False
    ) -> list[dict]:
        """List registered features (optionally filtered by entity)."""
        params: list[Any] = []
        sql = "SELECT * FROM features"
        where: list[str] = []
        if entity is not None:
            where.append("entity = ?")
            params.append(entity)
        if not include_archived:
            where.append("is_archived = 0")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY name ASC"

        rows = self.backend.fetchall(sql, tuple(params))
        result: list[dict] = []
        for r in rows:
            entry = {
                "name": str(r["name"]),
                "entity": str(r["entity"]),
                "description": str(r["description"]),
                "owner": str(r["owner"]),
                "feature_type": str(r["feature_type"]),
                "is_archived": bool(r["is_archived"]),
                "created_date": str(r["created_date"]),
            }
            active = self.versions.get_active(str(r["name"]))
            entry["active_version"] = active.version if active else None
            result.append(entry)
        return result

    def get_lineage(self, name: str, recursive: bool = False) -> dict:
        """Return upstream sources + downstream consumers for *name*."""
        return {
            "feature": name,
            "upstream": self.lineage.get_upstream(name, recursive=recursive),
            "downstream": self.lineage.get_downstream(name),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def record_statistics(
        self,
        name: str,
        df: pd.DataFrame,
        target: Optional[str] = None,
        iv_bins: int = 10,
    ) -> None:
        """Compute and persist distribution + IV stats for *name*'s active version.

        If the feature is multi-output (e.g. ``AggregationFeature`` emits
        ``card1_TransactionAmt_sum`` etc.), statistics are recorded per
        output column via separate rows. For single-output features,
        *name* itself is used as the column.

        Parameters
        ----------
        name : str
            Registered feature name (used to resolve the active version).
        df : pd.DataFrame
            DataFrame containing the feature's output column(s).
        target : str, optional
            Target column for IV computation. ``None`` skips IV.
        iv_bins : int
            Number of bins for IV computation (default 10).
        """
        active = self.versions.get_active(name)
        if active is None:
            raise KeyError(
                f"Feature '{name}' has no active version; call register() first"
            )

        # Try the exact column name first; if missing, attempt a per-output
        # breakdown by scanning columns that start with the feature name.
        if name in df.columns:
            columns = [name]
        else:
            columns = [c for c in df.columns if c.startswith(name)]
            if not columns:
                raise KeyError(
                    f"No output column(s) for feature '{name}' found in DataFrame"
                )

        rows_to_upsert = []
        for col in columns:
            stats = compute_all_stats(df, col, target=target, iv_bins=iv_bins)
            stats["version_id"] = active.version_id
            # Convert NaN/None to None for SQLite
            for key in (
                "iv_score", "mean", "std", "min_value", "max_value", "p50", "p95",
            ):
                if _is_nan_or_none(stats.get(key)):
                    stats[key] = None
            rows_to_upsert.append(stats)

        with self.backend.connect() as conn:
            for s in rows_to_upsert:
                conn.execute(
                    "INSERT OR REPLACE INTO statistics "
                    "(version_id, missing_rate, iv_score, n_unique, mean, std, "
                    "min_value, max_value, p50, p95, computed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        s["version_id"], s["missing_rate"], s["iv_score"],
                        s["n_unique"], s["mean"], s["std"],
                        s["min_value"], s["max_value"], s["p50"], s["p95"],
                        s["computed_at"],
                    ),
                )
