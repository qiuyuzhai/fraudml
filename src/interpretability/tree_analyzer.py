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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


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
    # Decision tracing — per-sample "why was this rejected"
    # ------------------------------------------------------------------

    def trace(
        self,
        sample: pd.Series,
        top_n_trees: int = 5,
    ) -> Dict[str, Any]:
        """Trace the exact decision path for a single prediction.

        Walks every tree from root to leaf using the sample's feature
        values, recording every split condition that was evaluated, and
        produces a human-readable "why was this transaction flagged".

        Parameters
        ----------
        sample : pd.Series
            A single transaction's feature values.  Index must match
            ``self.feature_names`` (i.e. the model's input columns).
        top_n_trees : int
            Number of trees (sorted by leaf absolute value descending)
            to include in the detailed step-by-step trace.

        Returns
        -------
        dict with keys:
            - fraud_prob: float — final fraud probability (sigmoid of sum)
            - log_odds: float — raw aggregated log-odds
            - feature_contributions: list of (feature, contribution) from
              each tree's leaf, sorted by abs(contribution)
            - tree_traces: list of per-tree step-by-step navigations
            - summary: str — human-readable narrative
        """
        if not self._trees:
            return {"fraud_prob": 0.0, "log_odds": 0.0,
                    "feature_contributions": [], "tree_traces": [],
                    "summary": "(no trees)"}

        tree_traces: List[Dict[str, Any]] = []
        leaf_contributions: List[Tuple[str, float]] = []
        total_log_odds = 0.0

        for tree_id, tree in enumerate(self._trees):
            split_feats = tree["split_feature"]
            thresholds = tree["threshold"]
            left_children = tree["left_child"]
            right_children = tree["right_child"]
            leaf_values = tree["leaf_value"]
            n = len(split_feats)

            node_idx = 0
            steps: List[Dict[str, Any]] = []
            reached_leaf = False

            while node_idx < n and node_idx >= 0:
                lc = left_children[node_idx]
                rc = right_children[node_idx]

                if lc < 0 and rc < 0:
                    leaf_v = float(leaf_values[node_idx]) if node_idx < len(leaf_values) else 0.0
                    total_log_odds += leaf_v
                    tree_traces.append({
                        "tree_id": tree_id,
                        "leaf_node": node_idx,
                        "leaf_value": leaf_v,
                        "steps": steps,
                    })
                    leaf_contributions.append((f"tree_{tree_id}", leaf_v))
                    reached_leaf = True
                    break

                fi = split_feats[node_idx]
                fn = self._feat_name(fi)
                th = float(thresholds[node_idx]) if node_idx < len(thresholds) else 0.0

                feat_val = None
                try:
                    feat_val = float(sample.get(fn, sample.get(fi, np.nan)))
                except (TypeError, ValueError):
                    feat_val = np.nan

                if np.isnan(feat_val):
                    feat_val = 0.0

                if feat_val <= th:
                    direction = "left"
                    target = lc
                else:
                    direction = "right"
                    target = rc

                steps.append({
                    "node": node_idx,
                    "feature": fn,
                    "threshold": th,
                    "actual_value": feat_val,
                    "direction": direction,
                    "condition": (
                        f"{fn} <= {self._format_threshold(th)}"
                        if direction == "left"
                        else f"{fn} > {self._format_threshold(th)}"
                    ),
                })

                node_idx = target

            if not reached_leaf:
                tree_traces.append({
                    "tree_id": tree_id,
                    "leaf_node": -1,
                    "leaf_value": 0.0,
                    "steps": steps,
                })

        fraud_prob = 1.0 / (1.0 + np.exp(-total_log_odds)) if abs(total_log_odds) < 50 else float(total_log_odds > 0)

        leaf_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        feature_contrib_map: Dict[str, float] = defaultdict(float)
        for _, lv in leaf_contributions:
            pass

        important_feats: Dict[str, float] = defaultdict(float)
        for tt in tree_traces:
            for step in tt["steps"]:
                fn = step["feature"]
                important_feats[fn] += abs(tt["leaf_value"])

        sorted_imp = sorted(important_feats.items(), key=lambda x: x[1], reverse=True)

        narrative_lines = [
            f"Fraud probability: {fraud_prob:.4f}",
            f"Total log-odds:    {total_log_odds:.4f}",
            f"Contributing trees: {len(tree_traces)}",
            "",
            "Most influential features (by |leaf_value| accumulated):",
        ]
        for feat, imp in sorted_imp[:10]:
            narrative_lines.append(f"  {feat}: cumulative_leaf_impact={imp:.4f}")

        top_tree_ids = [t["tree_id"] for t in tree_traces[:top_n_trees]]

        narrative_lines.append("")
        narrative_lines.append(f"--- Step-by-step for top {min(top_n_trees, len(tree_traces))} trees ---")
        for tt in tree_traces[:top_n_trees]:
            narrative_lines.append(
                f"Tree #{tt['tree_id']}: leaf={tt['leaf_node']} "
                f"(value={tt['leaf_value']:.4f}, prob={1/(1+np.exp(-tt['leaf_value'])) if abs(tt['leaf_value'])<50 else float(tt['leaf_value']>0):.4f})"
            )
            for step in tt["steps"]:
                flag = "***" if step["feature"] in [f for f, _ in sorted_imp[:5]] else "   "
                narrative_lines.append(
                    f"  {flag} {step['condition']}  (actual={step['actual_value']:.4f})  → {step['direction']}"
                )

        return {
            "fraud_prob": float(fraud_prob),
            "log_odds": float(total_log_odds),
            "feature_contributions": sorted_imp,
            "tree_traces": tree_traces[:top_n_trees],
            "all_tree_traces": tree_traces,
            "summary": "\n".join(narrative_lines),
        }

    def trace_summary(self, sample: pd.Series, top_n_trees: int = 3) -> str:
        """Convenience wrapper returning just the narrative string."""
        result = self.trace(sample, top_n_trees=top_n_trees)
        return result["summary"]

    def plot_trace(
        self,
        sample: pd.Series,
        tree_idx: int = 0,
        ax: Optional[object] = None,
    ) -> Optional[object]:
        """Visualize the decision path through a single tree for one sample.

        The traced path is highlighted in orange; non-taken branches are
        shown in light gray.  Leaf nodes are color-coded by fraud probability
        (red = high, green = low).

        Parameters
        ----------
        sample : pd.Series
            Transaction feature values.
        tree_idx : int
            Index of the tree to visualize.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.

        Returns
        -------
        matplotlib.axes.Axes or None
        """
        if not _HAS_MPL or not self._trees:
            return None

        if tree_idx >= len(self._trees) or tree_idx < 0:
            tree_idx = 0

        tree = self._trees[tree_idx]
        split_feats = tree["split_feature"]
        thresholds = tree["threshold"]
        left_children = tree["left_child"]
        right_children = tree["right_child"]
        leaf_values = tree["leaf_value"]
        n = len(split_feats)

        depth_map: Dict[int, int] = {}
        self._compute_depths(0, 0, left_children, right_children, n, depth_map)

        path_nodes: set = set()
        path_edges: set = set()
        node_idx = 0
        while node_idx < n and node_idx >= 0:
            path_nodes.add(node_idx)
            lc = left_children[node_idx]
            rc = right_children[node_idx]
            if lc < 0 and rc < 0:
                break
            fi = split_feats[node_idx]
            fn = self._feat_name(fi)
            th = float(thresholds[node_idx]) if node_idx < len(thresholds) else 0.0
            try:
                feat_val = float(sample.get(fn, sample.get(fi, 0)))
            except (TypeError, ValueError):
                feat_val = 0.0
            if feat_val <= th:
                target = lc
            else:
                target = rc
            if target >= 0:
                path_edges.add((node_idx, target))
            node_idx = target

        max_depth = max(depth_map.values()) if depth_map else 0
        n_levels = max_depth + 1

        fig_width = max(8, 2 ** max_depth * 1.8)
        fig_height = max(4, n_levels * 1.2)
        if ax is None:
            _, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-n_levels - 0.5, 0.5)
        ax.axis("off")

        positions: Dict[int, Tuple[float, float]] = {}
        for node_i, d in depth_map.items():
            leaves_below = self._count_leaves_below(node_i, left_children, right_children, n)
            total_leaves = self._count_leaves_below(0, left_children, right_children, n) or 1
            x = (leaves_below / total_leaves) * 2.0 - 1.0
            y = -d
            positions[node_i] = (x, y)

        for node_i in range(n):
            lc = left_children[node_i]
            rc = right_children[node_i]
            if lc >= 0 and node_i in positions and lc in positions:
                x1, y1 = positions[node_i]
                x2, y2 = positions[lc]
                is_on_path = (node_i, lc) in path_edges
                ax.plot([x1, x2], [y1, y2],
                        color="#e67e22" if is_on_path else "#d5dbdb",
                        linewidth=2.0 if is_on_path else 0.6,
                        alpha=0.9 if is_on_path else 0.4)
            if rc >= 0 and node_i in positions and rc in positions:
                x1, y1 = positions[node_i]
                x2, y2 = positions[rc]
                is_on_path = (node_i, rc) in path_edges
                ax.plot([x1, x2], [y1, y2],
                        color="#e67e22" if is_on_path else "#d5dbdb",
                        linewidth=2.0 if is_on_path else 0.6,
                        alpha=0.9 if is_on_path else 0.4)

        for node_i, (x, y) in positions.items():
            lc = left_children[node_i]
            rc = right_children[node_i]
            is_leaf = lc < 0 and rc < 0
            on_path = node_i in path_nodes

            if is_leaf:
                leaf_v = leaf_values[node_i] if node_i < len(leaf_values) else 0.0
                prob = 1.0 / (1.0 + np.exp(-leaf_v)) if abs(leaf_v) < 50 else float(leaf_v > 0)
                color = "#e74c3c" if prob > 0.5 else "#2ecc71"
                box = FancyBboxPatch(
                    (x - 0.12, y - 0.18), 0.24, 0.36,
                    boxstyle="round,pad=0.02",
                    facecolor=color if on_path else "#bdc3c7",
                    edgecolor="white",
                    alpha=0.9 if on_path else 0.5,
                )
                ax.add_patch(box)
                ax.text(x, y, f"p={prob:.2f}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold" if on_path else "normal")
            else:
                fi = split_feats[node_i] if node_i < len(split_feats) else -1
                fn = self._feat_name(fi)
                th = thresholds[node_i] if node_i < len(thresholds) else 0.0
                th_str = self._format_threshold(th)
                try:
                    fv = float(sample.get(fn, sample.get(fi, 0)))
                except (TypeError, ValueError):
                    fv = 0.0
                label = f"{fn}\n<= {th_str}\n(实际={fv:.2f})"
                box = FancyBboxPatch(
                    (x - 0.22, y - 0.28), 0.44, 0.56,
                    boxstyle="round,pad=0.02",
                    facecolor="#e67e22" if on_path else "#3498db",
                    edgecolor="white",
                    alpha=0.9 if on_path else 0.6,
                )
                ax.add_patch(box)
                ax.text(x, y, label, ha="center", va="center",
                        fontsize=6.5, color="white")

        ax.set_title(f"Decision Path — Tree #{tree_idx} (highlighted = taken path)", fontsize=10)
        fig = ax.figure
        fig.tight_layout()
        return ax

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
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        output_dir: str | Path,
        run_prefix: str = "",
        top_k_gain: int = 20,
        top_k_depth: int = 15,
        top_k_inter: int = 20,
        tree_idx: Optional[int] = None,
    ) -> Dict[str, Path]:
        """Generate all tree-model diagnostic plots and save as PNGs.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save figure PNGs.
        run_prefix : str
            Prefix prepended to every output filename.
        top_k_gain : int
            Number of top features (by total gain) to plot.
        top_k_depth : int
            Number of top features (by split count) for the depth plot.
        top_k_inter : int
            Number of top feature-pairs for the interaction heatmap.
        tree_idx : int, optional
            Index of the tree to visualize structurally.  When ``None``,
            the first tree with ``max_depth <= 5`` is chosen automatically
            (falls back to tree 0 if no shallow tree exists).

        Returns
        -------
        dict
            Mapping of plot name → saved PNG path.
        """
        if not _HAS_MPL:
            return {}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{run_prefix}_" if run_prefix else ""

        saved: Dict[str, Path] = {}

        p = self._plot_feature_gain(output_dir, prefix, top_k_gain)
        if p is not None:
            saved["feature_gain"] = p

        p = self._plot_feature_depth(output_dir, prefix, top_k_depth)
        if p is not None:
            saved["feature_depth"] = p

        p = self._plot_interactions(output_dir, prefix, top_k_inter)
        if p is not None:
            saved["interactions"] = p

        p = self._plot_tree_structure(output_dir, prefix, tree_idx)
        if p is not None:
            saved["tree_structure"] = p

        plt.close("all")
        return saved

    def _plot_feature_gain(
        self,
        output_dir: Path,
        prefix: str,
        top_k: int,
    ) -> Optional[Path]:
        gain_df = self.analyze_gain(top_k=top_k)
        if gain_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.35)))

        feats = gain_df["feature"].values[::-1]
        total = gain_df["total_gain"].values[::-1]
        pct = gain_df["pct_trees_with_gain"].values[::-1]

        y_pos = np.arange(len(feats))
        bars = ax.barh(y_pos, total, color="steelblue", alpha=0.85, edgecolor="white")

        for bar, pct_val in zip(bars, pct):
            x = bar.get_width()
            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"  {pct_val:.0f}%",
                va="center",
                fontsize=8,
                color="dimgray",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("Total Split Gain")
        ax.set_title(f"Top {len(feats)} Features by Total Gain")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"{prefix}feature_gain.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def _plot_feature_depth(
        self,
        output_dir: Path,
        prefix: str,
        top_k: int,
    ) -> Optional[Path]:
        depth_df = self.analyze_depth()
        if depth_df.empty:
            return None

        top_df = depth_df.head(top_k).copy()

        depth_dist_data: Dict[str, Dict[int, int]] = {}
        for _, row in top_df.iterrows():
            feat = row["feature"]
            raw = row["depth_distribution"]
            try:
                pairs = eval(raw)
                depth_dist_data[feat] = {int(k): v for k, v in pairs.items()}
            except Exception:
                depth_dist_data[feat] = {}

        all_depths = sorted(
            {d for dist in depth_dist_data.values() for d in dist.keys()}
        )
        if not all_depths:
            return None

        fig, ax = plt.subplots(figsize=(9, max(4, len(top_df) * 0.35)))

        feat_order = top_df["feature"].tolist()
        x = np.arange(len(feat_order))
        bottom = np.zeros(len(feat_order))
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_depths)))

        for d, color in zip(all_depths, cmap):
            values = np.array(
                [depth_dist_data.get(f, {}).get(d, 0) for f in feat_order],
                dtype=float,
            )
            ax.bar(x, values, bottom=bottom, label=f"depth {d}", color=color, edgecolor="white", linewidth=0.3)
            bottom += values

        ax.set_xticks(x)
        ax.set_xticklabels(feat_order, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Split Count")
        ax.set_title("Feature Split Count by Tree Depth")
        ax.legend(fontsize=8, loc="upper right", title="Depth")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"{prefix}feature_depth.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def _plot_interactions(
        self,
        output_dir: Path,
        prefix: str,
        top_k: int,
    ) -> Optional[Path]:
        inter_df = self.mine_interactions(top_k=top_k)
        if inter_df.empty:
            return None

        if _HAS_NX:
            return self._plot_interactions_network(inter_df, output_dir, prefix)
        return self._plot_interactions_heatmap(inter_df, output_dir, prefix)

    def _plot_interactions_network(
        self,
        inter_df: pd.DataFrame,
        output_dir: Path,
        prefix: str,
    ) -> Optional[Path]:
        G = nx.Graph()
        for _, row in inter_df.iterrows():
            G.add_edge(
                row["feature_1"],
                row["feature_2"],
                weight=row["shared_paths"],
                cross=row.get("cross_source", False),
            )

        if G.number_of_nodes() == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        pos = nx.spring_layout(G, k=1.2, seed=42, weight="weight")

        edges = G.edges(data=True)
        weights = [d["weight"] for _, _, d in edges]
        max_w = max(weights) if weights else 1.0
        edge_widths = [0.5 + 4.0 * (w / max_w) for w in weights]
        edge_colors = ["#e74c3c" if d.get("cross", False) else "#3498db" for _, _, d in edges]

        node_deg = dict(G.degree(weight="weight"))
        node_sizes = [80 + 200 * (node_deg[n] / max(node_deg.values()) if node_deg else 1) for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color="#f39c12", edgecolors="white")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#e74c3c", lw=2, label="Cross-source"),
            Line2D([0], [0], color="#3498db", lw=2, label="Same-source"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        ax.set_title("Feature Interaction Network\n(edge width = shared paths)")
        ax.axis("off")
        fig.tight_layout()

        out = output_dir / f"{prefix}feature_interactions.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def _plot_interactions_heatmap(
        self,
        inter_df: pd.DataFrame,
        output_dir: Path,
        prefix: str,
    ) -> Optional[Path]:
        features = sorted(
            set(inter_df["feature_1"].tolist() + inter_df["feature_2"].tolist())
        )
        if not features:
            return None

        matrix = pd.DataFrame(0, index=features, columns=features, dtype=float)
        for _, row in inter_df.iterrows():
            matrix.loc[row["feature_1"], row["feature_2"]] = row["shared_paths"]
            matrix.loc[row["feature_2"], row["feature_1"]] = row["shared_paths"]

        fig, ax = plt.subplots(figsize=(max(6, len(features) * 0.35), max(5, len(features) * 0.3)))
        im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=7)
        plt.colorbar(im, ax=ax, label="Shared Paths")
        ax.set_title("Feature Interaction Heatmap")
        fig.tight_layout()

        out = output_dir / f"{prefix}feature_interactions.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def _plot_tree_structure(
        self,
        output_dir: Path,
        prefix: str,
        tree_idx: Optional[int],
    ) -> Optional[Path]:
        if not self._trees:
            return None

        if tree_idx is None:
            tree_idx = 0
            for i, t in enumerate(self._trees):
                if t.get("max_depth", 99) <= 5:
                    tree_idx = i
                    break

        tree = self._trees[tree_idx]
        split_feats = tree["split_feature"]
        thresholds = tree["threshold"]
        left_children = tree["left_child"]
        right_children = tree["right_child"]
        leaf_values = tree["leaf_value"]
        n = len(split_feats)

        depth_map: Dict[int, int] = {}
        self._compute_depths(0, 0, left_children, right_children, n, depth_map)

        if not depth_map:
            return None

        max_depth = max(depth_map.values()) if depth_map else 0
        n_levels = max_depth + 1

        fig_width = max(10, 2 ** max_depth * 2.2)
        fig_height = max(4, n_levels * 1.3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-n_levels - 0.5, 0.5)
        ax.axis("off")

        positions: Dict[int, Tuple[float, float]] = {}
        for node_i, d in depth_map.items():
            leaves_below = self._count_leaves_below(node_i, left_children, right_children, n)
            total_leaves = self._count_leaves_below(0, left_children, right_children, n) or 1
            x = (leaves_below / total_leaves) * 2.0 - 1.0
            y = -d
            positions[node_i] = (x, y)

        for node_i in range(n):
            lc = left_children[node_i]
            rc = right_children[node_i]
            if lc >= 0 and node_i in positions and lc in positions:
                x1, y1 = positions[node_i]
                x2, y2 = positions[lc]
                ax.plot([x1, x2], [y1, y2], color="#7f8c8d", linewidth=0.8)
            if rc >= 0 and node_i in positions and rc in positions:
                x1, y1 = positions[node_i]
                x2, y2 = positions[rc]
                ax.plot([x1, x2], [y1, y2], color="#7f8c8d", linewidth=0.8)

        for node_i, (x, y) in positions.items():
            lc = left_children[node_i]
            rc = right_children[node_i]
            is_leaf = lc < 0 and rc < 0

            if is_leaf:
                leaf_v = leaf_values[node_i] if node_i < len(leaf_values) else 0.0
                prob = 1.0 / (1.0 + np.exp(-leaf_v)) if abs(leaf_v) < 50 else float(leaf_v > 0)
                color = "#e74c3c" if prob > 0.5 else "#2ecc71"
                label = f"p={prob:.2f}"
                box = FancyBboxPatch(
                    (x - 0.12, y - 0.18), 0.24, 0.36,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor="white",
                    alpha=0.85,
                )
                ax.add_patch(box)
                ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            else:
                fi = split_feats[node_i] if node_i < len(split_feats) else -1
                fn = self._feat_name(fi)
                th = thresholds[node_i] if node_i < len(thresholds) else 0.0
                th_str = self._format_threshold(th)
                label = f"{fn}\n<= {th_str}"
                box = FancyBboxPatch(
                    (x - 0.18, y - 0.22), 0.36, 0.44,
                    boxstyle="round,pad=0.02",
                    facecolor="#3498db", edgecolor="white",
                    alpha=0.9,
                )
                ax.add_patch(box)
                ax.text(x, y, label, ha="center", va="center", fontsize=7, color="white")

        ax.set_title(
            f"Tree #{tree_idx} (depth={max_depth}, n_leaves={sum(1 for i in range(n) if left_children[i] < 0 and right_children[i] < 0)})",
            fontsize=10,
        )
        fig.tight_layout()

        out = output_dir / f"{prefix}tree_structure.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    @staticmethod
    def _count_leaves_below(
        node_idx: int,
        left_children: List[int],
        right_children: List[int],
        n: int,
    ) -> int:
        if node_idx >= n or node_idx < 0:
            return 0
        lc = left_children[node_idx]
        rc = right_children[node_idx]
        if lc < 0 and rc < 0:
            return 1
        total = 0
        if lc >= 0:
            total += TreeAnalyzer._count_leaves_below(lc, left_children, right_children, n)
        if rc >= 0:
            total += TreeAnalyzer._count_leaves_below(rc, left_children, right_children, n)
        return total

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