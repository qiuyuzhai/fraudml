import sys
sys.path.insert(0, "d:/fraudml")
import numpy as np
import pandas as pd
from src.scoring.binning import chi_merge

np.random.seed(42)

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

print("=" * 60)
print("测试 2: 验证边界反映合并结果")
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
n_bins = len(edges2) - 1
assert 2 <= n_bins <= 5, f"分箱数 {n_bins} 应在 2~5 之间"
middle_edges = [e for e in edges2 if np.isfinite(e)]
assert any(-5 < e < 5 for e in middle_edges), "分界点应在数据范围内"
print(f"  分箱数: {n_bins}")
print("  => 通过\n")

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

print("=" * 60)
print("测试 4: 空数据处理")
df_empty = pd.DataFrame({"feature": [], "target": []})
e = chi_merge(df_empty, "feature", "target")
assert e == [float("-inf"), float("inf")], f"空数据应返回 [-inf, inf]，实际: {e}"
print(f"  空数据返回: {e}")
print("  => 通过\n")

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

print("=" * 60)
print("测试 6: 小样本箱强制合并")
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