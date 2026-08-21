"""
PolarsDataLoader — Polars-backed loader with the same interface as :class:`DataLoader`.

Mirrors ``fit / transform / load_train / load_test`` so callers can swap
backends via ``make_loader(engine="polars")`` without further changes.
The transform method returns a pandas DataFrame at the end (via
``DataFrame.to_pandas()``) so downstream code (cleaner, feature registry,
model) keeps consuming pandas.

Polars is imported lazily so this module is import-safe even when polars
is not installed; :func:`make_loader` handles the fallback to the pandas
loader in that case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .loader import DataLoader, _convert_id_cols_to_category, _downcast_numeric

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:  # pragma: no cover - guarded by make_loader fallback
    _HAS_POLARS = False


class PolarsDataLoader(DataLoader):
    """Polars-backed loader returning pandas DataFrames.

    Reads CSV/Parquet via polars' lazy API (``scan_csv`` / ``scan_parquet``
    + ``collect``) for better throughput on large files, then converts
    to pandas before returning so the rest of the pipeline is unchanged.

    Parameters
    ----------
    data_dir : str | Path, optional
    data_format : ``"csv"`` | ``"parquet"``
    """

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        data_format: str = "csv",
    ) -> None:
        if not _HAS_POLARS:
            raise ImportError(
                "polars is required for PolarsDataLoader. Install with `pip install polars>=0.20.0`."
            )
        super().__init__(data_dir=data_dir, data_format=data_format)

    def _read_file(self, name: str) -> pd.DataFrame:
        """Read ``<name>.csv`` or ``<name>.parquet`` via polars; return pandas."""
        if self.data_format == "parquet":
            path = self.data_dir / f"{name}.parquet"
            lf = pl.scan_parquet(path)
        else:
            path = self.data_dir / f"{name}.csv"
            # scan_csv needs explicit schema inference hints for id_* → str
            lf = pl.scan_csv(path, try_parse_dates=False)
        df_pl: pl.DataFrame = lf.collect()
        # Convert id_* to categorical (polars Categorical) before pandas handoff
        id_cols = [
            c for c in df_pl.columns
            if c.startswith("id_") and c != "TransactionID"
        ]
        if id_cols:
            df_pl = df_pl.with_columns([
                pl.col(c).cast(pl.Utf8).cast(pl.Categorical).alias(c) for c in id_cols
            ])
        df = df_pl.to_pandas()
        return df

    def load_train(self):
        """Load training transaction + identity DataFrames (pandas output)."""
        train_transaction = self._read_file("train_transaction")
        train_identity = self._read_file("train_identity")

        # Reuse pandas-side downcast / id_ conversion for parity with DataLoader
        train_transaction = _downcast_numeric(train_transaction)
        train_identity = _downcast_numeric(train_identity)
        train_identity = _convert_id_cols_to_category(train_identity)

        self.fit(train_transaction)
        return train_transaction, train_identity

    def load_test(self):
        """Load test transaction + identity DataFrames (pandas output)."""
        test_transaction = self._read_file("test_transaction")
        test_identity = self._read_file("test_identity")

        test_transaction = self.transform(test_transaction)
        test_identity = self.transform(test_identity)
        return test_transaction, test_identity
