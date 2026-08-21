"""Tests for FeatureLineage — DAG construction and traversal."""

from src.feature_store import FeatureStore


def test_immediate_upstream_returns_raw_columns(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    store.registry.register(
        "amt_log",
        entity="transaction",
        raw_columns=["TransactionAmt"],
    )

    upstream = store.lineage.get_upstream("amt_log")
    sources = {(s["source_type"], s["source_name"]) for s in upstream}
    assert ("raw_column", "TransactionAmt") in sources


def test_recursive_upstream_walks_feature_edges(tmp_feature_store_db):
    """amt_log depends on TransactionAmt; agg depends on amt_log.

    Recursive upstream of agg must include both amt_log and the raw
    column TransactionAmt.
    """
    store = FeatureStore(tmp_feature_store_db)
    store.registry.register(
        "amt_log",
        entity="transaction",
        raw_columns=["TransactionAmt"],
    )
    store.registry.register(
        "amt_agg",
        entity="transaction",
        upstream_features=["amt_log"],
    )

    upstream = store.lineage.get_upstream("amt_agg", recursive=True)
    source_names = {s["source_name"] for s in upstream}
    assert "amt_log" in source_names
    assert "TransactionAmt" in source_names


def test_downstream_returns_consumers(tmp_feature_store_db):
    """When amt_log is consumed by amt_agg, get_downstream('amt_log')
    should report amt_agg as a direct consumer."""
    store = FeatureStore(tmp_feature_store_db)
    store.registry.register(
        "amt_log",
        entity="transaction",
        raw_columns=["TransactionAmt"],
    )
    store.registry.register(
        "amt_agg",
        entity="transaction",
        upstream_features=["amt_log"],
    )

    downstream = store.lineage.get_downstream("amt_log")
    # get_downstream returns a list of feature-name strings.
    assert "amt_agg" in downstream


def test_no_lineage_for_isolated_feature(tmp_feature_store_db):
    store = FeatureStore(tmp_feature_store_db)
    store.registry.register("iso", entity="transaction")

    upstream = store.lineage.get_upstream("iso")
    downstream = store.lineage.get_downstream("iso")
    assert upstream == []
    assert downstream == []
