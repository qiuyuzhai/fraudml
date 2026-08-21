"""
FeatureLineage — DAG construction and queries for the Feature Store.

Lineage edges record ``version_id -> (source_type, source_name)`` tuples,
where ``source_type`` is either ``'raw_column'`` (a column from the raw
input DataFrame) or ``'feature'`` (an upstream engineered feature). The
composite primary key naturally prevents duplicate edges; ``ON DELETE
CASCADE`` ensures pruning a version prunes its edges.

Cycle detection uses a depth-first traversal over the feature→feature
edges (raw-column sources have no outgoing edges by construction, so they
cannot participate in a cycle).
"""

from __future__ import annotations

from typing import Iterable

from .backend import SQLiteBackend
from .versioning import VersionManager


class FeatureLineage:
    """Build / query the feature lineage DAG.

    Parameters
    ----------
    backend : SQLiteBackend
    version_manager : VersionManager
        Used to resolve ``feature_name -> active version_id`` for queries
        that take a feature name rather than a version id.
    """

    def __init__(
        self, backend: SQLiteBackend, version_manager: VersionManager
    ) -> None:
        self.backend = backend
        self.version_manager = version_manager

    # ------------------------------------------------------------------
    # Edge construction
    # ------------------------------------------------------------------

    def add_edges(
        self, version_id: int, sources: Iterable[tuple[str, str]]
    ) -> None:
        """Bulk-insert lineage edges for *version_id*.

        Parameters
        ----------
        version_id : int
            Target version (the consumer of the sources).
        sources : iterable of (source_type, source_name)
            ``source_type`` is ``'raw_column'`` or ``'feature'``.
        """
        rows = [(version_id, st, sn) for st, sn in sources]
        if not rows:
            return
        self.backend.execute_many(
            "INSERT OR IGNORE INTO lineage (version_id, source_type, source_name) "
            "VALUES (?, ?, ?)",
            rows,
        )

    # ------------------------------------------------------------------
    # Upstream / downstream queries
    # ------------------------------------------------------------------

    def get_upstream(
        self, feature_name: str, recursive: bool = False
    ) -> list[dict]:
        """Return upstream sources of *feature_name*'s active version.

        Non-recursive returns the immediate sources (raw columns +
        direct feature dependencies). Recursive walks the feature→feature
        edges to the roots (raw columns), deduplicating by
        ``(source_type, source_name)``.
        """
        active = self.version_manager.get_active(feature_name)
        if active is None:
            return []

        if not recursive:
            rows = self.backend.fetchall(
                "SELECT source_type, source_name FROM lineage WHERE version_id = ?",
                (active.version_id,),
            )
            return [
                {"source_type": r["source_type"], "source_name": r["source_name"]}
                for r in rows
            ]

        # Recursive DFS over feature->feature edges
        visited: set[tuple[str, str]] = set()
        result: list[dict] = []
        stack: list[str] = [feature_name]
        seen_features: set[str] = set()

        while stack:
            current = stack.pop()
            if current in seen_features:
                continue
            seen_features.add(current)

            current_active = self.version_manager.get_active(current)
            if current_active is None:
                continue

            rows = self.backend.fetchall(
                "SELECT source_type, source_name FROM lineage WHERE version_id = ?",
                (current_active.version_id,),
            )
            for r in rows:
                key = (r["source_type"], r["source_name"])
                if key in visited:
                    continue
                visited.add(key)
                result.append(
                    {"source_type": r["source_type"], "source_name": r["source_name"]}
                )
                if r["source_type"] == "feature":
                    stack.append(r["source_name"])

        return result

    def get_downstream(self, feature_name: str) -> list[str]:
        """Return feature names that directly depend on *feature_name*."""
        active = self.version_manager.get_active(feature_name)
        if active is None:
            return []

        rows = self.backend.fetchall(
            "SELECT DISTINCT fv.feature_name "
            "FROM lineage l JOIN feature_versions fv ON l.version_id = fv.version_id "
            "WHERE l.source_type = 'feature' AND l.source_name = ?",
            (feature_name,),
        )
        return [r["feature_name"] for r in rows]

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def detect_cycle(self) -> bool:
        """Return ``True`` if the feature→feature lineage graph contains a cycle."""
        edges = self.backend.fetchall(
            "SELECT DISTINCT fv.feature_name AS consumer, l.source_name AS producer "
            "FROM lineage l "
            "JOIN feature_versions fv ON l.version_id = fv.version_id "
            "WHERE l.source_type = 'feature'"
        )
        graph: dict[str, list[str]] = {}
        for r in edges:
            graph.setdefault(r["consumer"], []).append(r["producer"])
            graph.setdefault(r["producer"], [])

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in graph}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in graph.get(node, []):
                if color.get(neighbor, WHITE) == GRAY:
                    return True
                if color.get(neighbor, WHITE) == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        return any(color[n] == WHITE and dfs(n) for n in graph)
