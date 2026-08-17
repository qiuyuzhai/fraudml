"""
GraphFeature — Entity-relationship graph features for fraud detection.

Builds an incrementally-constructed entity graph from transaction data
and extracts node-level statistics that capture relational fraud patterns
(e.g. gang fraud via shared card/device/email/address clusters).

Anti-leakage design:
    Graph construction follows temporal order (TransactionDT ascending).
    Each transaction's features are computed from the graph state
    *before* that transaction is added — current-row values never leak
    into its own entity statistics. 本行交易的数据绝不会泄露到自身的实体统计量当中

Entity coverage:
    - card1, card2, card3, card5 (card entities)
    - addr1, addr2 (address entities)
    - P_emaildomain, R_emaildomain (email entities)
    - DeviceType, id_30, id_31 (device entities)

Features produced:
    Per-entity column: fraud_count, first_seen_dt, time_since_first.
    Cross-entity (card1-centric): unique addr1/addr2/email/device counts.
    Connected-component: card1 component size via union-find.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .base import FeatureBase


_CARD_COLS = ["card1", "card2", "card3", "card5"]
_ADDR_COLS = ["addr1", "addr2"]
_EMAIL_COLS = ["P_emaildomain", "R_emaildomain"]
_DEVICE_COLS = ["DeviceType", "id_30", "id_31"]

_DEFAULT_ENTITY_COLS = _CARD_COLS + _ADDR_COLS + _EMAIL_COLS + _DEVICE_COLS


class _UnionFind:
    """Union-Find with path compression and union by rank.

    Tracks connected component sizes for the entity graph.
    Each node is identified by a string key ``"{type}:{value}"``
    to avoid collisions across entity types.
    """

    __slots__ = ("_parent", "_size")

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._size: Dict[str, int] = {}

    def _ensure(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1

    def find(self, x: str) -> str:
        self._ensure(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._size[rx] < self._size[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        self._size[rx] += self._size[ry]
        del self._size[ry]

    def get_size(self, x: str) -> int:
        return self._size[self.find(x)]

    def get_component_size(self, x: str) -> int:
        return self.get_size(x)


class GraphFeature(FeatureBase):
    """Graph-based entity relationship features.

    Stateful — learns which entity columns are present during ``fit()``.
    ``transform()`` performs streaming graph construction with
    shift-protection (no future leakage).

    Parameters
    ----------
    entity_cols : list of str, optional
        Entity columns to track. Defaults to all known entity columns.
    card1_anchor : str
        Primary anchor column for cross-entity features. Default ``"card1"``.
    """

    def __init__(
        self,
        name: str = "GraphFeature",
        entity_cols: Optional[List[str]] = None,
        card1_anchor: str = "card1",
    ) -> None:
        super().__init__(name=name)
        self._entity_cols = entity_cols or list(_DEFAULT_ENTITY_COLS)
        self._card1_anchor = card1_anchor
        self._active_entity_cols: List[str] = []
        self._has_target: bool = False

    @property
    def is_stateful(self) -> bool:
        return True

    @property
    def _col_suffix(self) -> str:
        if self.name == self.__class__.__name__:
            return ""
        return f"_{self.name}"

    def fit(self, df: pd.DataFrame) -> "GraphFeature":
        self._active_entity_cols = [c for c in self._entity_cols if c in df.columns]
        self._has_target = "isFraud" in df.columns
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        has_target = "isFraud" in df.columns
        suffix = self._col_suffix

        # 保存原始索引
        df["_gf_orig_idx"] = df.index
        df = df.sort_values("TransactionDT").reset_index(drop=True)

        n = len(df)
        dt_values = df["TransactionDT"].values.astype(np.float64)

        fraud_arr = np.zeros(n, dtype=np.int32)
        if has_target:
            fraud_arr = df["isFraud"].fillna(0).astype(np.int32).values

        feature_data: Dict[str, np.ndarray] = {}

        for col in self._active_entity_cols:
            self._process_entity_column(
                df[col].values, dt_values, fraud_arr, col, n, feature_data, suffix
            )

        self._process_cross_entity(
            df, dt_values, fraud_arr, n, feature_data, suffix
        )

        self._process_connected_components(
            df, dt_values, fraud_arr, n, feature_data, suffix
        )

        for name, arr in feature_data.items():
            df[name] = arr

        df = df.sort_values("_gf_orig_idx").reset_index(drop=True)
        df.drop(columns=["_gf_orig_idx"], inplace=True)

        self._feature_names = list(feature_data.keys())
        return df

    # ------------------------------------------------------------------
    # Per-entity-column streaming statistics
    # ------------------------------------------------------------------
    
    # 单实体统计特征
    def _process_entity_column(
        self,
        values: np.ndarray,
        dt_values: np.ndarray,
        fraud_arr: np.ndarray,
        col_name: str,
        n: int,
        feature_data: Dict[str, np.ndarray],
        suffix: str = "",
    ) -> None:
        fraud_count = np.zeros(n, dtype=np.int32)
        first_seen = np.full(n, np.nan, dtype=np.float64)
        time_since_first = np.full(n, np.nan, dtype=np.float64)

        stats: Dict[str, Dict[str, Any]] = {}

        for i in range(n):
            val = values[i]
            if pd.isna(val):
                fraud_count[i] = 0
                first_seen[i] = np.nan
                time_since_first[i] = np.nan
                continue

            key = str(val)
            s = stats.get(key)

            if s is None:
                fraud_count[i] = 0
                first_seen[i] = dt_values[i]
                time_since_first[i] = 0.0
                stats[key] = {
                    "fraud_count": int(fraud_arr[i]),
                    "first_dt": dt_values[i],
                }
            else:
                fraud_count[i] = s["fraud_count"]
                first_seen[i] = s["first_dt"]
                time_since_first[i] = dt_values[i] - s["first_dt"]
                s["fraud_count"] += int(fraud_arr[i])

        feature_data[f"{col_name}_fraud_count{suffix}"] = fraud_count
        feature_data[f"{col_name}_first_seen_dt{suffix}"] = first_seen
        feature_data[f"{col_name}_time_since_first{suffix}"] = time_since_first

    # ------------------------------------------------------------------
    # Cross-entity features (card1-centric)
    # ------------------------------------------------------------------
    
    # 跨实体聚合特征
    def _process_cross_entity(
        self,
        df: pd.DataFrame,
        dt_values: np.ndarray,
        fraud_arr: np.ndarray,
        n: int,
        feature_data: Dict[str, np.ndarray],
        suffix: str = "",
    ) -> None:
        card1_col = self._card1_anchor
        if card1_col not in df.columns:
            return

        addr1_u = np.zeros(n, dtype=np.int32)
        addr2_u = np.zeros(n, dtype=np.int32)
        email_u = np.zeros(n, dtype=np.int32)
        device_u = np.zeros(n, dtype=np.int32)
        total_unique = np.zeros(n, dtype=np.int32)

        card_stats: Dict[str, Dict[str, Set[str]]] = {}

        addr1_available = "addr1" in df.columns
        addr2_available = "addr2" in df.columns
        email_available = "P_emaildomain" in df.columns or "R_emaildomain" in df.columns
        device_available = any(c in df.columns for c in _DEVICE_COLS)

        has_addr1 = addr1_available
        has_addr2 = addr2_available
        has_email = "P_emaildomain" in df.columns
        has_device = any(c in df.columns for c in ["DeviceType", "id_30", "id_31"])

        for i in range(n):
            card1_val = df.at[i, card1_col]
            if pd.isna(card1_val):
                continue

            card_key = str(card1_val)
            entry = card_stats.get(card_key)

            if entry is None:
                addr1_u[i] = 0
                addr2_u[i] = 0
                email_u[i] = 0
                device_u[i] = 0
                total_unique[i] = 0

                entry = {
                    "addr1_set": set(),
                    "addr2_set": set(),
                    "email_set": set(),
                    "device_set": set(),
                }
                card_stats[card_key] = entry
            else:
                addr1_u[i] = len(entry["addr1_set"])
                addr2_u[i] = len(entry["addr2_set"])
                email_u[i] = len(entry["email_set"])
                device_u[i] = len(entry["device_set"])
                total_unique[i] = addr1_u[i] + addr2_u[i] + email_u[i] + device_u[i]

            if has_addr1:
                v = df.at[i, "addr1"]
                if not pd.isna(v):
                    entry["addr1_set"].add(str(v))
            if has_addr2:
                v = df.at[i, "addr2"]
                if not pd.isna(v):
                    entry["addr2_set"].add(str(v))
            if has_email:
                v = df.at[i, "P_emaildomain"]
                if not pd.isna(v):
                    entry["email_set"].add(str(v))
            if has_device:
                for dc in _DEVICE_COLS:
                    if dc in df.columns:
                        v = df.at[i, dc]
                        if not pd.isna(v):
                            entry["device_set"].add(f"{dc}:{v}")

        feature_data[f"card1_unique_addr1{suffix}"] = addr1_u
        feature_data[f"card1_unique_addr2{suffix}"] = addr2_u
        feature_data[f"card1_unique_email{suffix}"] = email_u
        feature_data[f"card1_unique_device{suffix}"] = device_u
        feature_data[f"card1_entity_diversity{suffix}"] = total_unique

    # ------------------------------------------------------------------
    # Connected component features (union-find across entity types)
    # ------------------------------------------------------------------
    
    # 连通分量特征（将所有共享实体的交易连通起来，看每个交易所属的连通分量大小是多少）
    def _process_connected_components(
        self,
        df: pd.DataFrame,
        dt_values: np.ndarray,
        fraud_arr: np.ndarray,
        n: int,
        feature_data: Dict[str, np.ndarray],
        suffix: str = "",
    ) -> None:
        uf = _UnionFind()

        card1_col = self._card1_anchor
        card1_available = card1_col in df.columns

        comp_size = np.zeros(n, dtype=np.int32)

        for i in range(n):
            entities_for_row: List[str] = []

            for col in self._active_entity_cols:
                if col == card1_col:
                    continue
                val = df.at[i, col]
                if not pd.isna(val):
                    entities_for_row.append(f"{col}:{val}")

            card1_val = df.at[i, card1_col] if card1_available else None
            card1_key = f"{card1_col}:{card1_val}" if card1_available and not pd.isna(card1_val) else None

            if card1_key is not None:
                comp_size[i] = uf.get_component_size(card1_key)

            if card1_key is not None and entities_for_row:
                for e in entities_for_row:
                    uf.union(card1_key, e)

            if card1_key is None and len(entities_for_row) >= 2:
                for j in range(1, len(entities_for_row)):
                    uf.union(entities_for_row[0], entities_for_row[j])

        for i in range(n):
            if comp_size[i] == 0 and card1_available:
                card1_val = df.at[i, card1_col]
                if not pd.isna(card1_val):
                    comp_size[i] = uf.get_component_size(f"{card1_col}:{card1_val}")

        feature_data[f"card1_component_size{suffix}"] = comp_size

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "GraphFeature",
            "layer": "business-domain",
            "is_stateful": True,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "GraphFeature",
                    "description": "Instance name.",
                },
                {
                    "name": "entity_cols",
                    "type": "list[str]",
                    "default": ["card1", "card2", "card3", "P_emaildomain", "addr1", "DeviceType", "id_30"],
                    "description": "Entity columns for graph construction. Faster union-find merges entities sharing same fraud label.",
                },
                {
                    "name": "card1_anchor",
                    "type": "str",
                    "default": "card1",
                    "description": "Anchor column for node_id generation. card1 has largest coverage.",
                },
            ],
            "example": "- GraphFeature\n\n# Or with custom entity columns:\n- GraphFeature:\n    entity_cols: [\"card1\", \"card2\", \"P_emaildomain\"]\n    card1_anchor: \"card1\"",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        suffix = self._col_suffix
        names = getattr(self, "_feature_names", [])
        if not names:
            names = []
            for col in self._entity_cols:
                names.extend([
                    f"{col}_fraud_count{suffix}",
                    f"{col}_first_seen_dt{suffix}",
                    f"{col}_time_since_first{suffix}",
                ])
            names.extend([
                f"card1_unique_addr1{suffix}",
                f"card1_unique_addr2{suffix}",
                f"card1_unique_email{suffix}",
                f"card1_unique_device{suffix}",
                f"card1_entity_diversity{suffix}",
                f"card1_component_size{suffix}",
            ])

        return {
            "feature_names": names,
            "physical_meaning": "Entity-relationship graph statistics (fraud co-occurrence, connected components, entity diversity)",
            "unit": "count / seconds / component_size",
            "depends_on_target": True,
        }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def test_graph_basic():
        df = pd.DataFrame({
            "TransactionDT": [1000, 2000, 3000, 4000, 5000],
            "card1": ["A", "A", "B", "B", "A"],
            "addr1": [1, 1, 2, 2, 1],
            "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "yahoo.com", "hotmail.com"],
            "DeviceType": ["mobile", "mobile", "desktop", "mobile", "mobile"],
            "isFraud": [0, 1, 0, 1, 0],
        })
        feat = GraphFeature(entity_cols=["card1", "addr1", "P_emaildomain", "DeviceType"])
        feat.fit(df)
        result = feat.transform(df)

        assert "card1_fraud_count" in result.columns
        assert "card1_unique_addr1" in result.columns
        assert "card1_component_size" in result.columns

        r0 = result.iloc[0]
        assert r0["card1_fraud_count"] == 0
        assert r0["card1_unique_addr1"] == 0
        assert r0["card1_component_size"] == 1

        r1 = result.iloc[1]
        assert r1["card1_fraud_count"] == 0
        assert r1["card1_unique_addr1"] == 1
        assert r1["card1_unique_email"] == 1
        assert r1["card1_unique_device"] == 1
        assert r1["card1_entity_diversity"] == 3

        r2 = result.iloc[2]
        assert r2["card1_fraud_count"] == 0
        assert r2["addr1_fraud_count"] == 0

        r4 = result.iloc[4]
        assert r4["card1_fraud_count"] == 1
        assert r4["card1_unique_addr1"] == 1
        assert r4["card1_unique_email"] == 1

    def test_graph_leakage():
        df = pd.DataFrame({
            "TransactionDT": [1000, 2000, 3000],
            "card1": ["A", "A", "A"],
            "addr1": [1, 1, 1],
            "isFraud": [0, 0, 1],
        })
        feat = GraphFeature(entity_cols=["card1", "addr1"])
        feat.fit(df)
        result = feat.transform(df)

        r0 = result.iloc[0]
        assert r0["card1_fraud_count"] == 0

        r1 = result.iloc[1]
        assert r1["card1_fraud_count"] == 0

        r2 = result.iloc[2]
        assert r2["card1_fraud_count"] == 0

    def test_graph_no_target():
        df = pd.DataFrame({
            "TransactionDT": [1000, 2000, 3000],
            "card1": ["A", "A", "B"],
            "addr1": [1, 2, 1],
        })
        feat = GraphFeature(entity_cols=["card1", "addr1"])
        feat.fit(df)
        result = feat.transform(df)

        assert "card1_fraud_count" in result.columns
        assert result.iloc[0]["card1_fraud_count"] == 0
        assert result.iloc[1]["card1_fraud_count"] == 0

    def test_graph_order_preserved():
        df = pd.DataFrame({
            "TransactionDT": [3000, 1000, 2000],
            "card1": ["B", "A", "A"],
            "addr1": [2, 1, 1],
            "isFraud": [0, 0, 0],
        })
        feat = GraphFeature(entity_cols=["card1", "addr1"])
        feat.fit(df)
        result = feat.transform(df)

        assert result.iloc[0]["card1"] == "B"
        assert result.iloc[0]["card1_fraud_count"] == 0
        assert result.iloc[1]["card1"] == "A"
        assert result.iloc[1]["card1_fraud_count"] == 0
        assert result.iloc[2]["card1"] == "A"
        assert result.iloc[2]["card1_fraud_count"] == 0

    def test_graph_cross_entity():
        df = pd.DataFrame({
            "TransactionDT": [1000, 2000, 3000, 4000],
            "card1": ["A", "A", "A", "B"],
            "addr1": [1, 2, 3, 4],
            "P_emaildomain": ["gmail.com", "yahoo.com", "hotmail.com", "gmail.com"],
            "DeviceType": ["mobile", "desktop", "mobile", "mobile"],
            "isFraud": [0, 0, 0, 0],
        })
        feat = GraphFeature(entity_cols=["card1", "addr1", "P_emaildomain", "DeviceType"])
        feat.fit(df)
        result = feat.transform(df)

        r0 = result.iloc[0]
        assert r0["card1_unique_addr1"] == 0
        assert r0["card1_unique_email"] == 0
        assert r0["card1_unique_device"] == 0
        assert r0["card1_entity_diversity"] == 0

        r1 = result.iloc[1]
        assert r1["card1_unique_addr1"] == 1
        assert r1["card1_unique_email"] == 1
        assert r1["card1_unique_device"] == 1

        r2 = result.iloc[2]
        assert r2["card1_unique_addr1"] == 2
        assert r2["card1_unique_email"] == 2
        assert r2["card1_unique_device"] == 2

    def test_graph_component():
        df = pd.DataFrame({
            "TransactionDT": [1000, 2000, 3000, 4000, 5000],
            "card1": ["A", "A", "B", "C", "C"],
            "addr1": [1, 2, 2, 3, 3],
            "isFraud": [0, 0, 0, 0, 0],
        })
        feat = GraphFeature(entity_cols=["card1", "addr1"])
        feat.fit(df)
        result = feat.transform(df)

        assert result.iloc[0]["card1_component_size"] == 1
        assert result.iloc[1]["card1_component_size"] == 2
        assert result.iloc[2]["card1_component_size"] == 1
        assert result.iloc[3]["card1_component_size"] == 1
        assert result.iloc[4]["card1_component_size"] == 2

    def test_graph_metadata():
        df = pd.DataFrame({
            "TransactionDT": [1000],
            "card1": ["A"],
            "addr1": [1],
            "isFraud": [0],
        })
        feat = GraphFeature(entity_cols=["card1", "addr1"])
        feat.fit(df)
        result = feat.transform(df)
        meta = feat.get_feature_metadata()
        assert "feature_names" in meta
        assert "physical_meaning" in meta
        assert meta["depends_on_target"] is True

    test_graph_basic()
    test_graph_leakage()
    test_graph_no_target()
    test_graph_order_preserved()
    test_graph_cross_entity()
    test_graph_component()
    test_graph_metadata()
    print("All GraphFeature tests passed!")