"""Batch scoring CLI — offline scoring of a transaction dataset.

Loads a FraudPredictor (from a local artifact directory or the MLflow
Model Registry), scores an input CSV/Parquet file in batches, and
writes results to SQLite (default), CSV, or Parquet.

Usage::

    # Local artifact directory
    python -m src.batch_score --artifact-dir artifacts/run_xxx \
        --input transactions.parquet --output scores.db

    # MLflow Model Registry
    fraudml-score --model-name fraudml --model-stage Production \
        --input transactions.csv --output scores.db

The scoring path mirrors the FastAPI /score endpoint — both reuse
FraudPredictor.predict — so offline backfills and online scores stay
consistent.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("fraudml.batch_score")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraudml-score",
        description="Batch-score transactions with a FraudPredictor.",
    )

    source = parser.add_argument_group("model source (mutually exclusive)")
    source_exclusive = source.add_mutually_exclusive_group(required=True)
    source_exclusive.add_argument(
        "--artifact-dir",
        help="Path to a local artifact directory (from train save()).",
    )
    source_exclusive.add_argument(
        "--model-name",
        help="MLflow Model Registry name. Use with --model-stage.",
    )

    parser.add_argument(
        "--model-stage",
        default="Production",
        help="MLflow stage to resolve (default: Production).",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI (optional).",
    )

    parser.add_argument("--input", required=True, help="Input CSV/Parquet path.")
    parser.add_argument("--output", required=True, help="Output path.")
    parser.add_argument(
        "--format",
        choices=("sqlite", "csv", "parquet"),
        default="sqlite",
        help="Output format (default: sqlite).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Rows per batch (default: 10000).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. If omitted, uses risk_engine or 0.5.",
    )
    parser.add_argument(
        "--id-column",
        default="TransactionID",
        help="Column to forward as transaction_id (default: TransactionID).",
    )

    return parser


def _load_predictor(args: argparse.Namespace) -> "FraudPredictor":
    from src.pipeline.predict import FraudPredictor

    if args.model_name:
        logger.info(
            "Loading from MLflow registry: name=%s stage=%s",
            args.model_name,
            args.model_stage,
        )
        return FraudPredictor.from_model_registry(
            name=args.model_name,
            stage=args.model_stage,
            tracking_uri=args.mlflow_tracking_uri,
        )

    logger.info("Loading from artifact dir: %s", args.artifact_dir)
    return FraudPredictor.from_artifact_dir(args.artifact_dir)


def _read_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(p)
    raise ValueError(
        f"Unsupported input format '{suffix}'. Use .csv or .parquet."
    )


def _score(
    predictor: "FraudPredictor",
    df: pd.DataFrame,
    batch_size: int,
    threshold: Optional[float],
) -> pd.DataFrame:
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    n = len(df)
    if n == 0:
        logger.warning("Input has 0 rows; nothing to score.")
        return pd.DataFrame(columns=[
            "transaction_id", "probability", "risk_level",
            "binary_prediction", "model_version", "scored_at",
        ])

    batches = list(range(0, n, batch_size))
    iterator = tqdm(batches, desc="Scoring", unit="batch") if use_tqdm else batches

    parts: list[pd.DataFrame] = []
    for start in iterator:
        end = min(start + batch_size, n)
        batch = df.iloc[start:end]
        result = predictor.predict_batch(
            batch,
            batch_size=batch_size,
            threshold=threshold,
            return_all=True,
        )
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame({"probability": result})
        parts.append(result)

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def _enrich(
    df: pd.DataFrame,
    result: pd.DataFrame,
    id_column: str,
    model_version: str,
) -> pd.DataFrame:
    if id_column in df.columns:
        result.insert(0, "transaction_id", df[id_column].values)
    elif "transaction_id" not in result.columns:
        result["transaction_id"] = None

    if "risk_level" not in result.columns:
        result["risk_level"] = None
    if "binary_prediction" not in result.columns:
        result["binary_prediction"] = None

    result["model_version"] = model_version
    result["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cols = [
        "transaction_id", "probability", "risk_level",
        "binary_prediction", "model_version", "scored_at",
    ]
    for c in cols:
        if c not in result.columns:
            result[c] = None
    return result[cols]


def _write_sqlite(df: pd.DataFrame, output: str) -> None:
    from sqlite3 import connect

    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with connect(p) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scores ("
            "  transaction_id,"
            "  probability REAL,"
            "  risk_level TEXT,"
            "  binary_prediction INTEGER,"
            "  model_version TEXT,"
            "  scored_at TEXT"
            ")"
        )
        df.to_sql("scores", conn, if_exists="append", index=False)
    logger.info("Wrote %d rows to SQLite: %s", len(df), p)


def _write_flat(df: pd.DataFrame, output: str, fmt: str) -> None:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)
    logger.info("Wrote %d rows to %s: %s", len(df), fmt.upper(), p)


def _log_summary(df: pd.DataFrame, elapsed: float) -> None:
    n = len(df)
    high = int((df["risk_level"] == "HIGH").sum()) if "risk_level" in df else 0
    logger.info(
        "Done: %d rows, HIGH=%d (%.2f%%), elapsed=%.1fs (%.0f rows/s)",
        n,
        high,
        100.0 * high / n if n else 0.0,
        elapsed,
        n / elapsed if elapsed > 0 else 0.0,
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _build_parser().parse_args(argv)

    try:
        predictor = _load_predictor(args)
    except Exception as exc:
        logger.error("Failed to load predictor: %s", exc)
        return 2

    model_version = args.model_name or args.artifact_dir

    try:
        df = _read_input(args.input)
    except Exception as exc:
        logger.error("Failed to read input: %s", exc)
        return 2

    logger.info("Loaded %d rows from %s", len(df), args.input)

    start = time.time()
    try:
        result = _score(predictor, df, args.batch_size, args.threshold)
    except Exception as exc:
        logger.error("Scoring failed: %s", exc)
        return 3
    elapsed = time.time() - start

    out = _enrich(df, result, args.id_column, model_version)
    _log_summary(out, elapsed)

    try:
        if args.format == "sqlite":
            _write_sqlite(out, args.output)
        else:
            _write_flat(out, args.output, args.format)
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
