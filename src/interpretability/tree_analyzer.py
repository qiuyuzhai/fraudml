"""
TreeAnalyzer — Structure-level analysis of tree-based models.

Extracts actionable "discoveries" from trained tree models without
requiring SHAP or other heavy dependencies.  Covers three layers:

1. **Rule extraction** — every root-to-leaf path is converted into
   a human-readable decision rule with support count and confidence.
2. **Feature depth analysis** — identifies which features act at
   which tree depths (deep features participate in more interactions).
3. **Interaction mining** — path-level (not tree-level) feature
   co-occurrence reveals *actual* conditional combinations the model
   uses, with cross-source highlighting.

Designed for tree models that expose ``booster_.dump_model()``
(LightGBM, XGBoost, CatBoost via their native APIs).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TreeAnalyzer:
    """Structure-level analyzer for tree-based models.

    Parameters
    ----------
    model : object
        Trained tree model (e.g. ``LGBMClassifier``).  Must expose
        ``booster_`` with ``dump_model()``.
    feature_names : list of str
        Feature names corresponding to model columns.
    feature_catalog : dict, optional
        Feature catalog metadata (from ``metadata.json``) that maps
        each feature to its source module.  Used for cross-source
        interaction detection.
    """

    def __init__(
        self,
        model: object,
        feature_names: List[str],
        feature_catalog: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.feature_catalog = feature_catalog

        self._trees: List[Dict[str, Any]] = []
        self._tree_meta: Dict[str, Any] = {}

        self._rules_: Optional[pd.DataFrame] = None
        self._depth_report_: Optional[pd.DataFrame] = None
        self._interaction_report_: Optional[pd.DataFrame] = None
        self._tree_summary_: Optional[pd.DataFrame] = None

        self._parse()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        booster = self.model.booster_
        raw = booster.dump_model()
        self._trees = raw.get("tree_info", [])

        depths = [t.get("max_depth", 0) for t in self._trees]
        leaves = [t.get("num_leaves", 0) for t in self._trees]

        self._tree_meta = {
            "n_trees": len(self._trees),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": float(np.mean(depths)) if depths else 0.0,
            "min_depth": min(depths) if depths else 0,
            "avg_leaves": float(np.mean(leaves)) if leaves else 0.0,
            "max_leaves": max(leaves) if leaves else 0,
            "min_leaves": min(leaves) if leaves else 0,
        }

    # ------------------------------------------------------------------
    # Rule extraction
    # ------------------------------------------------------------------

    def extract_rules(
        self,
        max_rules_per_tree: int = 5,
        min_support: int = 0,
    ) -> pd.DataFrame:
        """Extract root-to-leaf decision rules from all trees.

        Each rule is a conjunction of split conditions leading to a
        leaf node, annotated with the leaf's fraud probability and
        an estimated support (number of training samples that would
        reach this leaf — proxied by leaf count statistics when
        available).

        Parameters
        ----------
        max_rules_per_tree : int
            Maximum rules to keep per tree (kept by leaf gain).
        min_support : int
            Minimum estimated support to keep a rule.

        Returns
        -------
        pd.DataFrame
            Columns: tree_id, rule, depth, leaf_value, fraud_prob,
            support, features_used, conditions.
        """
        all_rules: List[Dict[str, Any]] = []

        for tree_id, tree in enumerate(self._trees):
            split_feats = tree["split_feature"]
            thresholds = tree["threshold"]
            split_gains = tree.get("split_gain", [])
            left_children = tree["left_child"]
            right_children = tree["right_child"]
            leaf_values = tree["leaf_value"]
            leaf_counts = tree.get("leaf_count", [])
            n = len(split_feats)

            rules: List[Dict[str, Any]] = []
            self._traverse_tree(
                tree_id=tree_id,
                node_idx=0,
                depth=0,
                conditions=[],
                path_gain=0.0,
                split_feats=split_feats,
                thresholds=thresholds,
                split_gains=split_gains,
                left_children=left_children,
                right_children=right_children,
                leaf_values=leaf_values,
                leaf_counts=leaf_counts,
                n=n,
                out_rules=rules,
            )

            rules.sort(key=lambda r: r.get("leaf_gain", 0), reverse=True)
            for rule in rules[:max_rules_per_tree]:
                if rule.get("support", 0) >= min_support:
                    rule["tree_id"] = tree_id
                    all_rules.append(rule)

        df = pd.DataFrame(all_rules)
        if not df.empty:
            df = df.sort_values("leaf_gain", ascending=False).reset_index(drop=True)
        self._rules_ = df
        return df

    def _traverse_tree(
        self,
        tree_id: int,
        node_idx: int,
        depth: int,
        conditions: List[str],
        path_gain: float,
        split_feats: List[int],
        thresholds: List[float],
        split_gains: List[float],
        left_children: List[int],
        right_children: List[int],
        leaf_values: List[float],
        leaf_counts: List[int],
        n: int,
        out_rules: List[Dict[str, Any]],
    ) -> None:
        if node_idx >= n or node_idx < 0:
            return

        lc = left_children[node_idx]
        rc = right_children[node_idx]

        if lc < 0 and rc < 0:
            self._record_leaf(
                node_idx, depth, conditions, path_gain,
                leaf_values, leaf_counts, out_rules,
            )
            return

        fi = split_feats[node_idx]
        fn = self._feat_name(fi)
        th = thresholds[node_idx]

        node_gain = split_gains[node_idx] if node_idx < len(split_gains) else 0.0
        new_gain = path_gain + node_gain

        th_str = self._format_threshold(th)

        new_cond_le = conditions + [f"{fn} <= {th_str}"]
        new_cond_gt = conditions + [f"{fn} > {th_str}"]

        if lc >= 0:
            self._traverse_tree(
                tree_id, lc, depth + 1, new_cond_le, new_gain,
                split_feats, thresholds, split_gains,
                left_children, right_children,
                leaf_values, leaf_counts, n, out_rules,
            )

        if rc >= 0:
            self._traverse_tree(
                tree_id, rc, depth + 1, new_cond_gt, new_gain,
                split_feats, thresholds, split_gains,
                left_children, right_children,
                leaf_values, leaf_counts, n, out_rules,
            )

    def _record_leaf(
        self,
        node_idx: int,
        depth: int,
        conditions: List[str],
        path_gain: float,
        leaf_values: List[float],
        leaf_counts: List[int],
        out_rules: List[Dict[str, Any]],
    ) -> None:
        leaf_v = leaf_values[node_idx]
        prob = 1.0 / (1.0 + np.exp(-leaf_v)) if abs(leaf_v) < 50 else float(leaf_v > 0)

        support = int(leaf_counts[node_idx]) if node_idx < len(leaf_counts) else 0

        features = []
        for cond in conditions:
            fn = cond.split(" ")[0]
            features.append(fn)

        rule_str = " AND ".join(conditions) if conditions else "(root-only)"

        out_rules.append({
            "tree_id": -1,
            "rule": rule_str,
            "depth": depth,
            "leaf_value": float(leaf_v),
            "fraud_prob": float(prob),
            "support": support,
            "features_used": features,
            "conditions": conditions,
            "leaf_gain": float(path_gain),
        })

    # ------------------------------------------------------------------
    # Feature depth analysis
    # ------------------------------------------------------------------

    def analyze_depth(self) -> pd.DataFrame:
        """Analyze at which tree depths each feature is used.

        Deep features (used at large depths) participate in more
        conditional interactions and are generally more important
        for nuanced discrimination than shallow features.

        Returns
        -------
        pd.DataFrame
            Columns: feature, avg_depth, min_depth, max_depth,
            split_count, pct_trees_used, depth_distribution.
        """
        feature_depths: Dict[str, List[int]] = defaultdict(list)
        split_counts: Counter[str] = Counter()
        tree_usage: Counter[str] = Counter()

        for tree in self._trees:
            split_feats = tree["split_feature"]
            left_children = tree["left_child"]
            right_children = tree["right_child"]
            n = len(split_feats)
            used_in_tree = set()

            depth_map: Dict[int, int] = {}
            self._compute_depths(0, 0, left_children, right_children,
                                 n, depth_map)

            for node_i in range(n):
                lc = left_children[node_i]
                rc = right_children[node_i]
                if lc < 0 and rc < 0:
                    continue
                fi = split_feats[node_i]
                fn = self._feat_name(fi)
                d = depth_map.get(node_i, 0)
                feature_depths[fn].append(d)
                split_counts[fn] += 1
                used_in_tree.add(fn)

            for fn in used_in_tree:
                tree_usage[fn] += 1

        rows = []
        for fn, depths in feature_depths.items():
            depth_counts = Counter(depths)
            depth_dist = {str(k): v for k, v in sorted(depth_counts.items())}
            rows.append({
                "feature": fn,
                "avg_depth": float(np.mean(depths)),
                "min_depth": min(depths),
                "max_depth": max(depths),
                "split_count": split_counts[fn],
                "pct_trees_used": tree_usage[fn] / len(self._trees) * 100,
                "depth_distribution": str(depth_dist),
            })

        df = pd.DataFrame(rows).sort_values(
            "split_count", ascending=False
        ).reset_index(drop=True)
        self._depth_report_ = df
        return df

    def _compute_depths(
        self,
        node_idx: int,
        depth: int,
        left_children: List[int],
        right_children: List[int],
        n: int,
        depth_map: Dict[int, int],
    ) -> None:
        if node_idx >= n or node_idx < 0:
            return
        depth_map[node_idx] = depth
        lc = left_children[node_idx]
        rc = right_children[node_idx]
        if lc >= 0:
            self._compute_depths(lc, depth + 1, left_children, right_children, n, depth_map)
        if rc >= 0:
            self._compute_depths(rc, depth + 1, left_children, right_children, n, depth_map)

    # ------------------------------------------------------------------
    # Interaction mining (path-level)
    # ------------------------------------------------------------------

    def mine_interactions(
        self,
        min_paths: int = 10,
        top_k: int = 30,
    ) -> pd.DataFrame:
        """Mine feature interactions from root-to-leaf paths.

        Two features interact when they appear **on the same root-to-leaf
        path** (i.e. their conditions are AND-ed together for predictions).
        This is fundamentally different from tree-level co-occurrence,
        which counts any two features appearing anywhere in the same
        tree — even on mutually exclusive branches.

        Parameters
        ----------
        min_paths : int
            Minimum number of shared paths for a pair to be reported.
        top_k : int
            Maximum top pairs to return.

        Returns
        -------
        pd.DataFrame
            Columns: feature_1, feature_2, shared_paths,
            cross_source, source_1, source_2.
        """
        path_pair_counts: Counter[Tuple[str, str]] = Counter()

        for tree in self._trees:
            split_feats = tree["split_feature"]
            n = len(split_feats)
            path_features_list: List[List[str]] = []
            self._collect_path_features(
                0, split_feats, tree["left_child"], tree["right_child"],
                n, [], path_features_list,
            )

            for path_feats in path_features_list:
                sorted_feats = sorted(set(path_feats))
                for i in range(len(sorted_feats)):
                    for j in range(i + 1, len(sorted_feats)):
                        path_pair_counts[(sorted_feats[i], sorted_feats[j])] += 1

        rows = []
        for (f1, f2), cnt in path_pair_counts.most_common(top_k * 3):
            if cnt < min_paths:
                break
            s1 = self._feature_source(f1)
            s2 = self._feature_source(f2)
            rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "shared_paths": cnt,
                "cross_source": s1 != s2,
                "source_1": s1,
                "source_2": s2,
            })
            if len(rows) >= top_k:
                break

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("shared_paths", ascending=False).reset_index(drop=True)
        self._interaction_report_ = df
        return df

    def _collect_path_features(
        self,
        node_idx: int,
        split_feats: List[int],
        left_children: List[int],
        right_children: List[int],
        n: int,
        current_path: List[str],
        all_paths: List[List[str]],
    ) -> None:
        if node_idx >= n or node_idx < 0:
            return

        lc = left_children[node_idx]
        rc = right_children[node_idx]

        if lc < 0 and rc < 0:
            if current_path:
                all_paths.append(list(current_path))
            return

        fn = self._feat_name(split_feats[node_idx])

        if lc >= 0:
            self._collect_path_features(
                lc, split_feats, left_children, right_children, n,
                current_path + [fn], all_paths,
            )

        if rc >= 0:
            self._collect_path_features(
                rc, split_feats, left_children, right_children, n,
                current_path + [fn], all_paths,
            )

    # ------------------------------------------------------------------
    # Tree summary
    # ------------------------------------------------------------------

    def tree_summary(self) -> pd.DataFrame:
        """Per-tree structural summary.

        Returns
        -------
        pd.DataFrame
            One row per tree: tree_id, num_leaves, max_depth,
            avg_leaf_value, leaf_value_std, split_features_count.
        """
        rows = []
        for tree_id, tree in enumerate(self._trees):
            leaves = tree.get("num_leaves", 0)
            depth = tree.get("max_depth", 0)
            leaf_vals = tree.get("leaf_value", [])
            left_ch = tree["left_child"]
            right_ch = tree["right_child"]

            leaf_only_vals = [
                leaf_vals[i] for i in range(len(leaf_vals))
                if i < len(left_ch) and left_ch[i] < 0 and right_ch[i] < 0
            ]
            avg_lv = float(np.mean(leaf_only_vals)) if leaf_only_vals else 0.0
            std_lv = float(np.std(leaf_only_vals)) if leaf_only_vals else 0.0

            internal_feats = set()
            for i, fi in enumerate(tree["split_feature"]):
                if i < len(left_ch) and (left_ch[i] >= 0 or right_ch[i] >= 0):
                    internal_feats.add(fi)
            n_feats = len(internal_feats)

            rows.append({
                "tree_id": tree_id,
                "num_leaves": leaves,
                "max_depth": depth,
                "avg_leaf_value": avg_lv,
                "leaf_value_std": std_lv,
                "split_features_count": n_feats,
            })
        df = pd.DataFrame(rows)
        self._tree_summary_ = df
        return df

    # ------------------------------------------------------------------
    # Feature gain analysis
    # ------------------------------------------------------------------

    def analyze_gain(self, top_k: int = 30) -> pd.DataFrame:
        """Aggregate split gain per feature across all trees.

        Parameters
        ----------
        top_k : int
            Number of top features by total gain to return.

        Returns
        -------
        pd.DataFrame
            Columns: feature, total_gain, mean_gain, max_gain,
            split_count, pct_trees_with_gain.
        """
        gain_by_feat: Dict[str, List[float]] = defaultdict(list)
        trees_with_gain: Counter[str] = Counter()

        for tree in self._trees:
            split_feats = tree["split_feature"]
            left_children = tree["left_child"]
            right_children = tree["right_child"]
            gains = tree.get("split_gain", [])
            used_in_tree = set()

            for i, fi in enumerate(split_feats):
                lc = left_children[i]
                rc = right_children[i]
                if lc < 0 and rc < 0:
                    continue
                if i < len(gains):
                    fn = self._feat_name(fi)
                    gain_by_feat[fn].append(float(gains[i]))
                    used_in_tree.add(fn)

            for fn in used_in_tree:
                trees_with_gain[fn] += 1

        rows = []
        for fn, gains in gain_by_feat.items():
            rows.append({
                "feature": fn,
                "total_gain": float(np.sum(gains)),
                "mean_gain": float(np.mean(gains)),
                "max_gain": float(np.max(gains)),
                "split_count": len(gains),
                "pct_trees_with_gain": trees_with_gain[fn] / len(self._trees) * 100,
            })

        df = pd.DataFrame(rows).sort_values(
            "total_gain", ascending=False
        ).head(top_k).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        output_dir: str | Path,
        run_prefix: str = "",
    ) -> Dict[str, Path]:
        """Export all analysis results to CSV files.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save outputs.
        run_prefix : str
            Prefix prepended to every output filename (e.g. run_id).

        Returns
        -------
        dict
            Mapping of report name → file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{run_prefix}_" if run_prefix else ""

        paths: Dict[str, Path] = {}

        if self._rules_ is None:
            self.extract_rules()
        p = output_dir / f"{prefix}tree_rules.csv"
        self._rules_.to_csv(p, index=False)
        paths["rules"] = p

        if self._depth_report_ is None:
            self.analyze_depth()
        p = output_dir / f"{prefix}feature_depth.csv"
        self._depth_report_.to_csv(p, index=False)
        paths["depth"] = p

        if self._interaction_report_ is None:
            self.mine_interactions()
        p = output_dir / f"{prefix}feature_interactions.csv"
        self._interaction_report_.to_csv(p, index=False)
        paths["interactions"] = p

        if self._tree_summary_ is None:
            self.tree_summary()
        p = output_dir / f"{prefix}tree_summary.csv"
        self._tree_summary_.to_csv(p, index=False)
        paths["tree_summary"] = p

        gain_df = self.analyze_gain()
        p = output_dir / f"{prefix}feature_gain.csv"
        gain_df.to_csv(p, index=False)
        paths["gain"] = p

        return paths

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the tree analysis."""
        lines = [
            "TreeAnalyzer Summary",
            f"  Trees:            {self._tree_meta['n_trees']}",
            f"  Features:         {len(self.feature_names)}",
            f"  Avg depth:        {self._tree_meta['avg_depth']:.1f}",
            f"  Max depth:        {self._tree_meta['max_depth']}",
            f"  Avg leaves/tree:  {self._tree_meta['avg_leaves']:.1f}",
            f"  Max leaves/tree:  {self._tree_meta['max_leaves']}",
        ]

        gain_df = self.analyze_gain(top_k=10)
        if not gain_df.empty:
            lines.append("  Top 10 features by total gain:")
            for _, row in gain_df.iterrows():
                lines.append(
                    f"    {row['feature']}: {row['total_gain']:.0f} "
                    f"(mean={row['mean_gain']:.1f}, trees={row['pct_trees_with_gain']:.0f}%)"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _feat_name(self, fi: int) -> str:
        if 0 <= fi < len(self.feature_names):
            return self.feature_names[fi]
        return f"feat_{fi}"

    def _feature_source(self, fn: str) -> str:
        if self.feature_catalog is None:
            return "Unknown"
        for entry in self.feature_catalog.get("entries", []):
            if fn in entry.get("feature_names", []):
                return entry.get("source", "Unknown")
        return "Unknown"

    @staticmethod
    def _format_threshold(th: float) -> str:
        if abs(th) < 1e-10:
            return "MISSING"
        if th == int(th):
            return str(int(th))
        return f"{th:.4f}"