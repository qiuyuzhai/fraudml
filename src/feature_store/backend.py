"""
SQLiteBackend — low-level SQLite access primitives for the Feature Store.

Owns the database connection, schema initialization, and CRUD helpers.
Higher-level classes (``VersionManager``, ``FeatureLineage``,
``FeatureRegistry``) compose this backend rather than touching sqlite3
directly, keeping the storage layer swappable (e.g. to Postgres later).

Schema is loaded from ``schema.sql`` shipped alongside this module via
``importlib.resources`` so the package works regardless of the current
working directory (important for Docker / installed-package contexts).

Connection strategy: one persistent ``sqlite3.Connection`` is kept open
for the backend's lifetime. This is required for ``":memory:"`` mode
(every new connection would otherwise yield a fresh empty database) and
also avoids per-call connection overhead for file-backed DBs. The
:meth:`connect` context manager wraps this single connection in
transaction semantics (commit on success, rollback on error).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

SchemaPath = Union[str, Path]


class SQLiteBackend:
    """Connection manager + raw SQL primitives for the Feature Store.

    Parameters
    ----------
    db_path : str | Path
        SQLite database path. ``":memory:"`` for an in-memory DB (used
        in tests). Parent directories are created automatically for
        filesystem paths.
    """

    def __init__(self, db_path: SchemaPath = "artifacts/feature_store.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # Persistent connection: required for ":memory:" (a new connection
        # would otherwise get a fresh empty in-memory DB).
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield the persistent connection with transaction semantics.

        Commits on clean exit, rolls back on exception. The connection
        itself is NOT closed (it is shared across calls so the schema
        persists for ``":memory:"`` mode).
        """
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        """Close the persistent connection (mainly for tests)."""
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------

    def _initialize_schema(self) -> None:
        """Apply ``schema.sql`` (idempotent — uses ``IF NOT EXISTS``)."""
        schema_sql = self._load_schema_sql()
        with self.connect() as conn:
            conn.executescript(schema_sql)

    @staticmethod
    def _load_schema_sql() -> str:
        """Load ``schema.sql`` from package resources."""
        try:
            return resources.files("src.feature_store").joinpath("schema.sql").read_text(
                encoding="utf-8"
            )
        except (ModuleNotFoundError, AttributeError, FileNotFoundError):
            # Fallback for source-tree execution without package install
            schema_path = Path(__file__).resolve().parent / "schema.sql"
            return schema_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # CRUD primitives
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return cur  # cursor detached after commit; only rowcount/lastrowid valid

    def execute_many(
        self, sql: str, params_seq: Sequence[Sequence[Any]]
    ) -> None:
        with self.connect() as conn:
            conn.executemany(sql, params_seq)

    def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchone()

    def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    def insert_and_get_id(
        self, sql: str, params: Sequence[Any] = ()
    ) -> int:
        """INSERT a row and return its AUTOINCREMENT id."""
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return int(cur.lastrowid)
