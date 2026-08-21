"""Tests for the FeatureStore high-level registry API."""

import pytest

from src.feature_store import FeatureStore


def test_register_creates_active_version(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    v = store.registry.register(
        "amt_log",
        entity="transaction",
        feature_type="numeric",
        description="log1p(TransactionAmt)",
        raw_columns=["TransactionAmt"],
    )

    assert v.version == 1
    assert v.feature_name == "amt_log"

    # The returned FeatureVersion is a snapshot taken before activate();
    # confirm the active state by reading back from the store.
    feat = store.registry.get_feature("amt_log")
    assert feat["name"] == "amt_log"
    assert feat["is_archived"] is False
    assert feat["version"]["version"] == 1
    assert feat["version"]["is_active"] is True


def test_re_register_creates_new_version_and_activates(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    v1 = store.registry.register("amt_log", entity="transaction")
    v2 = store.registry.register("amt_log", entity="transaction", description="v2")

    assert v1.version == 1
    assert v2.version == 2
    # Only one active version at a time
    assert store.registry.get_feature("amt_log")["version"]["version"] == 2


def test_list_features_excludes_archived_by_default(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    store.registry.register("amt_log", entity="transaction")
    store.registry.register("card4_enc", entity="transaction")
    store.registry.archive("card4_enc")

    listed = store.registry.list_features()
    names = {f["name"] for f in listed}
    assert names == {"amt_log"}

    all_listed = store.registry.list_features(include_archived=True)
    all_names = {f["name"] for f in all_listed}
    assert all_names == {"amt_log", "card4_enc"}


def test_get_feature_unknown_raises(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    with pytest.raises(KeyError):
        store.registry.get_feature("does_not_exist")
