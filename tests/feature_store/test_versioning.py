"""Tests for the VersionManager — version lifecycle and rollback.

VersionManager.create_version inserts a row into feature_versions which
has a foreign key to features(name). These tests register the feature
via the public FeatureStore.registry.register() API first (which creates
the parent row + an initial active version), then exercise VersionManager
on additional versions.
"""

import pytest

from src.feature_store import FeatureStore


def _seed_feature(store):
    """Register a feature and return its initial active version."""
    return store.registry.register("amt_log", entity="transaction")


def test_next_version_starts_at_two_after_seed(tmp_feature_store_db):
    """register() already created version 1, so the next free version is 2."""
    store = FeatureStore(tmp_feature_store_db)
    _seed_feature(store)
    assert store.versions.next_version("amt_log") == 2


def test_create_version_is_inactive_until_activated(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    _seed_feature(store)

    v = store.versions.create_version("amt_log")
    assert v.is_active is False
    # The seeded version (1) should still be active until we flip the new one.
    assert store.versions.get_active("amt_log").version == 1

    store.versions.activate(v.version_id)
    active = store.versions.get_active("amt_log")
    assert active is not None
    assert active.version_id == v.version_id
    assert active.is_active is True


def test_activate_deactivates_others(tmp_feature_store_db):
    """Activating one version deactivates all other versions of the feature."""
    store = FeatureStore(tmp_feature_store_db)
    _seed_feature(store)
    v2 = store.versions.create_version("amt_log")
    v3 = store.versions.create_version("amt_log")

    store.versions.activate(v2.version_id)
    assert store.versions.get_active("amt_log").version_id == v2.version_id

    store.versions.activate(v3.version_id)
    assert store.versions.get_active("amt_log").version_id == v3.version_id

    versions = store.versions.list_versions("amt_log")
    active = [v for v in versions if v.is_active]
    assert len(active) == 1


def test_rollback_activates_historical_version(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    _seed_feature(store)
    store.versions.create_version("amt_log")  # version 2
    store.versions.create_version("amt_log")  # version 3
    v3 = store.versions.get_active("amt_log")
    store.versions.activate(v3.version_id)

    rolled = store.versions.rollback("amt_log", to_version=1)
    assert rolled.version == 1
    assert rolled.is_active is True

    versions = store.versions.list_versions("amt_log")
    by_ver = {v.version: v for v in versions}
    assert by_ver[3].is_active is False
    assert by_ver[1].is_active is True


def test_rollback_unknown_version_raises(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    _seed_feature(store)
    with pytest.raises(ValueError):
        store.versions.rollback("amt_log", to_version=99)
