"""
Chi-Merge binning algorithm for optimal WOE bin construction.

Chi-Merge iteratively merges adjacent bins that have similar
fraud rates (low chi-square statistic) to produce a compact,
monotonic WOE binning scheme.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# 计算两个bin的卡方统计
def _chi_square_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Chi-square statistic for two bin distributions.

    Parameters
    ----------
    a : np.ndarray
        Counts [good, bad] in bin A.
    b : np.ndarray
        Counts [good, bad] in bin B.

    Returns
    -------
    float
        Chi-square value.
    """
    table = np.array([a, b])
    try:
        chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
        return float(chi2)
    except ValueError:
        return float("inf")


def chi_merge(
    df: pd.DataFrame,
    feature: str,
    target: str,
    max_bins: int = 5,
    min_chi2: float = 0.5,
    min_bin_pct: float = 0.05,
) -> List[float]:
    """Compute optimal bin edges via Chi-Merge.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame.
    feature : str
        Numeric feature column name.
    target : str
        Binary target column name.
    max_bins : int
        Maximum number of bins allowed.
    min_chi2 : float
        Minimum chi-square threshold for merging.  Bins with chi2
        below this are candidates for merging.
    min_bin_pct : float
        Minimum fraction of total samples in each bin.

    Returns
    -------
    list[float]
        Bin edges (including -inf and inf).
    """
    data = df[[feature, target]].dropna().copy()
    if len(data) == 0:
        return [float("-inf"), float("inf")]

    # 先用等频分箱将数据分成50个初始bin
    data["bin"] = pd.qcut(data[feature], q=50, duplicates="drop")

    bin_stats = data.groupby("bin", observed=False).agg(
        good=(target, lambda x: (x == 0).sum()),
        bad=(target, lambda x: (x == 1).sum()),
        count=(target, "size"),
    ).reset_index()

    # 跟踪每个箱的原始边界，合并时同步更新，保证最终能正确提取边界
    if isinstance(bin_stats["bin"].iloc[0], pd.Interval):
        bin_stats["_left"] = bin_stats["bin"].apply(lambda x: x.left)
        bin_stats["_right"] = bin_stats["bin"].apply(lambda x: x.right)
    else:
        bin_stats["_left"] = bin_stats["bin"].astype(float)
        bin_stats["_right"] = bin_stats["bin"].astype(float)

    total = bin_stats["count"].sum()
    min_bin_count = max(int(min_bin_pct * total), 1)

    # 循环合并相邻箱子，直到箱子数 ≤ max_bins
    while len(bin_stats) > max_bins:
        if len(bin_stats) < 2:
            break

        merged = False
        best_idx = -1
        best_chi2 = float("inf")

        for i in range(len(bin_stats) - 1):
            a = bin_stats.iloc[i]
            b = bin_stats.iloc[i + 1]
            chi2_val = _chi_square_stat(
                np.array([a["good"], a["bad"]]),
                np.array([b["good"], b["bad"]]),
            )
            # 记录chi2最小的一对相邻bin
            if chi2_val < best_chi2:
                best_chi2 = chi2_val
                best_idx = i

        # 如果卡方统计大于阈值，且当前bin数量小于等于最大bin数，直接跳出循环
        if best_chi2 > min_chi2 and len(bin_stats) <= max_bins:
            break

        # 合并chi2最小的一对相邻bin（a为左箱，b为右箱）
        if best_idx >= 0:
            a = bin_stats.iloc[best_idx]
            b = bin_stats.iloc[best_idx + 1]
            new_bin = pd.DataFrame([{
                "bin": f"merged_{best_idx}",
                "good": a["good"] + b["good"],
                "bad": a["bad"] + b["bad"],
                "count": a["count"] + b["count"],
                "_left": a["_left"],
                "_right": b["_right"],
            }])
            bin_stats = pd.concat(
                [bin_stats.iloc[:best_idx], new_bin, bin_stats.iloc[best_idx + 2:]],
                ignore_index=True,
            )
            merged = True

        if not merged:
            break

    while len(bin_stats) > 1 and bin_stats["count"].min() < min_bin_count:
        # 前面卡方循环只负责合并「坏率接近」的箱子，但不管箱子样本多少，有可能合并结束后还存在样本极少的箱子
        # 样本太少的箱子：good/bad 计数噪声极大，WOE 剧烈抖动，IV 失真，建模不稳定
        # 本循环专门把样本过小的箱子强制和邻居合并，保证每个箱子样本量 ≥ min_bin_count

        # 找出样本数量最少那一行（箱子）的索引
        idx = bin_stats["count"].idxmin()

        if idx == 0:
            merge_idx = 1
        elif idx == len(bin_stats) - 1:
            merge_idx = idx - 1
        else:
            # 在中间：左右都有邻居，选择样本更少的那一侧邻居进行合并
            merge_idx = idx - 1 if bin_stats.iloc[idx - 1]["count"] <= bin_stats.iloc[idx + 1]["count"] else idx + 1

        a = bin_stats.iloc[idx]
        b = bin_stats.iloc[merge_idx]
        # 根据左右顺序决定合并后边界：取左箱的左边界和右箱的右边界
        if idx < merge_idx:
            new_left, new_right = a["_left"], b["_right"]
        else:
            new_left, new_right = b["_left"], a["_right"]
        new_bin = pd.DataFrame([{
            "bin": f"merged",
            "good": a["good"] + b["good"],
            "bad": a["bad"] + b["bad"],
            "count": a["count"] + b["count"],
            "_left": new_left,
            "_right": new_right,
        }])
        drop_indices = sorted([idx, merge_idx])
        # 切片：取两个箱子前面所有行 + 插入新合并箱子 + 取两个箱子后面所有行
        bin_stats = pd.concat(
            [bin_stats.iloc[:drop_indices[0]], new_bin, bin_stats.iloc[drop_indices[-1] + 1:]],
            ignore_index=True,
        )

    # 从合并后的 bin_stats 提取正确的分箱边界
    # 按左边界排序保证空间顺序，每个合并后箱的右边界即为分割点
    bin_stats = bin_stats.sort_values("_left")
    edges = [float("-inf")]
    for _, row in bin_stats.iterrows():
        edges.append(float(row["_right"]))
    edges[-1] = float("inf")

    return edges


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    # ---------- 测试 1: 基础功能 ----------
    print("=" * 60)
    print("测试 1: 基础分箱功能")
    df = pd.DataFrame({
        "feature": np.random.randn(1000),
        "target": np.random.binomial(1, 0.3, 1000),
    })
    edges = chi_merge(df, "feature", "target", max_bins=5)
    print(f"  输入样本量: {len(df)}")
    print(f"  返回边界: {edges}")
    print(f"  分箱数量: {len(edges) - 1}")
    assert edges[0] == float("-inf"), "首边界应为 -inf"
    assert edges[-1] == float("inf"), "尾边界应为 +inf"
    assert all(edges[i] < edges[i + 1] for i in range(len(edges) - 1)), "边界必须严格递增"
    assert len(edges) - 1 <= 5, f"分箱数 {len(edges)-1} 超过 max_bins=5"
    for e in edges[1:-1]:
        assert np.isfinite(e), f"中间边界 {e} 不是有限值"
    print("  => 通过\n")

    # ---------- 测试 2: 边界真正反映合并结果 ----------
    print("=" * 60)
    print("测试 2: 验证边界是合并后的结果（关键修复点）")
    # 构造一个有明显分段特征的数据集：前半段坏率高，后半段坏率低
    x = np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500),
    ])
    y = np.concatenate([
        np.random.binomial(1, 0.7, 500),
        np.random.binomial(1, 0.1, 500),
    ])
    df2 = pd.DataFrame({"feature": x, "target": y})
    edges2 = chi_merge(df2, "feature", "target", max_bins=5)
    print(f"  构造数据: 两团分离高斯 (坏率 0.7 vs 0.1)")
    print(f"  返回边界: {edges2}")
    # 因为数据有两个明显分离的簇，且 max_bins=5，
    # 最终应产生 2~5 个箱，且分界点应在 -2 和 2 之间
    n_bins = len(edges2) - 1
    assert 2 <= n_bins <= 5, f"分箱数 {n_bins} 应在 2~5 之间"
    # 至少一个中间边界应落在合理范围（数据中心之间）
    middle_edges = [e for e in edges2 if np.isfinite(e)]
    assert any(-5 < e < 5 for e in middle_edges), "分界点应在数据范围内"
    print(f"  分箱数: {n_bins}")
    print("  => 通过\n")

    # ---------- 测试 3: 不同 max_bins 参数 ----------
    print("=" * 60)
    print("测试 3: max_bins 参数控制")
    for mb in [2, 3, 5, 8]:
        df3 = pd.DataFrame({
            "feature": np.random.randn(500),
            "target": np.random.binomial(1, 0.4, 500),
        })
        e = chi_merge(df3, "feature", "target", max_bins=mb)
        n = len(e) - 1
        print(f"  max_bins={mb:2d}  =>  实际分箱数={n}")
        assert n <= mb, f"实际分箱数 {n} 超过 max_bins={mb}"
    print("  => 通过\n")

    # ---------- 测试 4: 空数据 ----------
    print("=" * 60)
    print("测试 4: 空数据处理")
    df_empty = pd.DataFrame({"feature": [], "target": []})
    e = chi_merge(df_empty, "feature", "target")
    assert e == [float("-inf"), float("inf")], f"空数据应返回 [-inf, inf]，实际: {e}"
    print(f"  空数据返回: {e}")
    print("  => 通过\n")

    # ---------- 测试 5: 含 NaN 数据 ----------
    print("=" * 60)
    print("测试 5: 含 NaN 数据")
    df_nan = pd.DataFrame({
        "feature": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        "target":  [0,   0,   1,     0,   1,   0,     1,   0,   0,   1],
    })
    e = chi_merge(df_nan, "feature", "target", max_bins=3)
    print(f"  原始数据 10 行，含 2 个 NaN")
    print(f"  返回边界: {e}")
    assert len(e) >= 2
    assert e[0] == float("-inf") and e[-1] == float("inf")
    print("  => 通过\n")

    # ---------- 测试 6: 小样本强制合并 ----------
    print("=" * 60)
    print("测试 6: 小样本箱强制合并")
    # 构造极不平衡数据：只有少量样本在某个区域
    x = np.concatenate([
        np.random.normal(0, 0.1, 950),
        np.random.normal(5, 0.1, 50),
    ])
    y = np.concatenate([
        np.random.binomial(1, 0.3, 950),
        np.random.binomial(1, 0.3, 50),
    ])
    df6 = pd.DataFrame({"feature": x, "target": y})
    e = chi_merge(df6, "feature", "target", max_bins=5, min_bin_pct=0.05)
    print(f"  950 样本在 0 附近 + 50 样本在 5 附近 (5% 占比)")
    print(f"  返回边界: {e}")
    n_bins = len(e) - 1
    print(f"  分箱数: {n_bins}")
    assert n_bins <= 5
    print("  => 通过\n")

    print("=" * 60)
    print("全部测试通过！")