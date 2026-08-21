-- FraudML Feature Store — SQLite schema (v1)
--
-- Three core tables:
--   features         — one row per feature name (logical entity)
--   feature_versions — monotonic version per feature; only one row per
--                       feature may have is_active=1
--   lineage          — DAG edges (version_id -> source), composite PK,
--                       ON DELETE CASCADE so dropping a version prunes
--                       its edges automatically
--   statistics       — per-version distribution / IV snapshot

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS features (
    name            TEXT PRIMARY KEY,
    entity          TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    owner           TEXT NOT NULL DEFAULT 'system',
    feature_type    TEXT NOT NULL DEFAULT 'numeric',  -- numeric|categorical|binary|timestamp
    created_date    TEXT NOT NULL,                    -- ISO 8601
    is_archived     INTEGER NOT NULL DEFAULT 0        -- 0|1
);

CREATE TABLE IF NOT EXISTS feature_versions (
    version_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name    TEXT NOT NULL,
    version         INTEGER NOT NULL,
    created_date    TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 0,      -- 0|1; only one row per feature = 1
    schema_json     TEXT,                             -- JSON: output column schema / dtype
    run_id          TEXT,                             -- MLflow run id (optional)
    FOREIGN KEY (feature_name) REFERENCES features(name) ON DELETE CASCADE,
    UNIQUE (feature_name, version)
);

-- Per-feature only-one-active invariant: enforced by application layer
-- (VersionManager.activate flips the prior active row to 0 before
-- setting the new row to 1).

CREATE TABLE IF NOT EXISTS lineage (
    version_id      INTEGER NOT NULL,
    source_type     TEXT NOT NULL CHECK (source_type IN ('raw_column', 'feature')),
    source_name     TEXT NOT NULL,
    PRIMARY KEY (version_id, source_type, source_name),
    FOREIGN KEY (version_id) REFERENCES feature_versions(version_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS statistics (
    version_id      INTEGER PRIMARY KEY,
    missing_rate    REAL,
    iv_score        REAL,
    n_unique        INTEGER,
    mean            REAL,
    std             REAL,
    min_value       REAL,
    max_value       REAL,
    p50             REAL,
    p95             REAL,
    computed_at     TEXT NOT NULL,
    FOREIGN KEY (version_id) REFERENCES feature_versions(version_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_versions_feature_active
    ON feature_versions(feature_name, is_active);

CREATE INDEX IF NOT EXISTS idx_lineage_source
    ON lineage(source_type, source_name);
