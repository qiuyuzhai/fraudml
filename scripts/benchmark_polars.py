"""
Benchmark pandas vs polars loader / aggregation / time transform on
IEEE-CIS train_transaction.csv (or .parquet).

Run::

    python scripts/benchmark_polars.py
    python scripts/benchmark_polars.py --data-dir data/raw --format csv

Optional script — not part of the core pipeline. Falls back gracefully
if polars is not installed (reports only the pandas numbers).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def _time_it(fn, label: str) -> float:
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    print(f"  {label:35s}  {elapsed * 1000:8.1f} ms")
    return elapsed


def bench_pandas(data_dir: Path, data_format: str = "csv") -> dict:
    from src.data.loader import DataLoader
    from src.features.aggregation_feature import AggregationFeature
    from src.features.time_feature import TimeFeature

    print(f"\n[pandas] data_dir={data_dir}, format={data_format}")
    loader = DataLoader(data_dir=data_dir, data_format=data_format)
    t_load = _time_it(lambda: loader.load_train(), "pandas load_train")
    train_txn, _ = loader.load_train()
    agg = AggregationFeature(agg_cols=["TransactionAmt"], group_keys=[("card1",)])
    t_agg_fit = _time_it(lambda: agg.fit(train_txn), "pandas agg.fit")
    t_agg_trans = _time_it(lambda: agg.transform(train_txn), "pandas agg.transform")
    tf = TimeFeature()
    tf.fit(train_txn)
    t_tf_trans = _time_it(lambda: tf.transform(train_txn), "pandas time.transform")
    return {
        "load_train": t_load,
        "agg_fit": t_agg_fit,
        "agg_transform": t_agg_trans,
        "time_transform": t_tf_trans,
    }


def bench_polars(data_dir: Path, data_format: str = "csv") -> dict:
    try:
        from src.data.loader_polars import PolarsDataLoader
        from src.features.aggregation_polars import AggregationFeaturePolars
        from src.features.time_feature_polars import TimeFeaturePolars
    except ImportError as e:
        print(f"\n[polars] SKIPPED — polars not installed: {e}")
        return {}
    print(f"\n[polars] data_dir={data_dir}, format={data_format}")
    loader = PolarsDataLoader(data_dir=data_dir, data_format=data_format)
    t_load = _time_it(lambda: loader.load_train(), "polars load_train")
    train_txn, _ = loader.load_train()
    agg = AggregationFeaturePolars(agg_cols=["TransactionAmt"], group_keys=[("card1",)])
    t_agg_fit = _time_it(lambda: agg.fit(train_txn), "polars agg.fit")
    t_agg_trans = _time_it(lambda: agg.transform(train_txn), "polars agg.transform")
    tf = TimeFeaturePolars()
    tf.fit(train_txn)
    t_tf_trans = _time_it(lambda: tf.transform(train_txn), "polars time.transform")
    return {
        "load_train": t_load,
        "agg_fit": t_agg_fit,
        "agg_transform": t_agg_trans,
        "time_transform": t_tf_trans,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark pandas vs polars backends.")
    ap.add_argument("--data-dir", default="data/raw", help="Raw data directory.")
    ap.add_argument("--format", default="csv", choices=["csv", "parquet"])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"data_dir {data_dir} does not exist; nothing to benchmark.")
        return

    pd_stats = bench_pandas(data_dir, args.format)
    pl_stats = bench_polars(data_dir, args.format)

    if pl_stats:
        print("\n=== Speedup (pandas ms / polars ms) ===")
        for k in pd_stats:
            if pl_stats.get(k, 0) > 0:
                print(f"  {k:20s}  {pd_stats[k] / pl_stats[k]:.2f}x")


if __name__ == "__main__":
    main()
