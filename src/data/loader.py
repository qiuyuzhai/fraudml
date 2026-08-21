"""
Data loading module for FraudML.

Provides :class:`DataLoader` for loading IEEE-CIS Fraud Detection data
with automatic memory optimization through dtype downcasting, plus
Parquet support with a metadata cache for fast re-load.

:func:`make_loader` is the factory entry point — pass ``engine="polars"``
to opt into the Polars backend (falls back to pandas with a warning if
polars is not installed).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to the smallest safe dtype.

    - float64 -> float32  (always safe, minor precision loss)
    - int64  -> int8/16/32 or uint8/16/32 based on value range

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns to downcast.

    Returns
    -------
    pd.DataFrame
        DataFrame with downcasted numeric dtypes.
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_max <= np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_max <= np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
            else:
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
    return df


def _convert_id_cols_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns starting with ``id_`` (except ``TransactionID``) to category dtype."""
    id_cols = [
        col for col in df.columns if col.startswith("id_") and col != "TransactionID"
    ]
    for col in id_cols:
        df[col] = df[col].astype("category")
    return df


def to_parquet(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write *df* to *path* as Parquet (pyarrow engine).

    Convenience function mirroring :meth:`DataLoader.to_parquet`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read a Parquet file to a pandas DataFrame (pyarrow engine)."""
    return pd.read_parquet(Path(path), engine="pyarrow")


class DataLoader:
    """Loads IEEE-CIS Fraud Detection data with automatic memory optimization.

    Implements a fit/transform interface so that dtype decisions made on
    the training set are consistently applied to the test set.

    Parameters
    ----------
    data_dir : str | Path, optional
        Path to the raw data directory. Defaults to ``data/raw`` relative
        to the project root.
    data_format : ``"csv"`` | ``"parquet"``, optional
        File format to read. Defaults to ``"csv"`` (the IEEE-CIS native
        format). When set to ``"parquet"``, looks for
        ``train_transaction.parquet`` / ``train_identity.parquet`` etc.
        and uses a metadata cache to skip the full scan on subsequent loads.

    Attributes
    ----------
    data_dir : Path
    _dtype_map : dict
        Mapping of column names to target dtypes, populated after ``fit()``.
    _fitted : bool
    """

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        data_format: str = "csv",
    ) -> None:
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
        self.data_dir: Path = Path(data_dir)
        self.data_format: str = data_format.lower()
        self._dtype_map: Dict[str, str] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Parquet metadata cache
    # ------------------------------------------------------------------

    @property
    def _parquet_meta_path(self) -> Path:
        return self.data_dir / ".parquet_meta.json"

    def _read_parquet_cached(self, name: str) -> pd.DataFrame:
        """Read ``<name>.parquet``, using cached metadata when available.

        The cache (``{data_dir}/.parquet_meta.json``) stores per-file
        schema (column → dtype) and row_count. On a cache hit we load the
        Parquet with the cached dtype map applied (skipping pandas'
        full-scan dtype inference for fit). On a miss we read normally,
        then populate the cache.
        """
        path = self.data_dir / f"{name}.parquet"
        cache = self._load_parquet_meta_cache()
        file_meta = cache.get(name) if cache else None

        if file_meta and "dtypes" in file_meta:
            dtype_map = {k: v for k, v in file_meta["dtypes"].items()}
            df = pd.read_parquet(path, engine="pyarrow")
            for col, dt in dtype_map.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dt)
                    except (ValueError, TypeError):
                        pass
        else:
            df = pd.read_parquet(path, engine="pyarrow")
            # Populate the cache
            self._update_parquet_meta_cache(name, df)

        return df

    def _load_parquet_meta_cache(self) -> Dict[str, Any]:
        if not self._parquet_meta_path.exists():
            return {}
        try:
            with open(self._parquet_meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _update_parquet_meta_cache(self, name: str, df: pd.DataFrame) -> None:
        cache = self._load_parquet_meta_cache()
        cache[name] = {
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
            "row_count": int(len(df)),
            "columns": list(df.columns),
        }
        try:
            with open(self._parquet_meta_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DataLoader":
        """Store target dtypes from a training DataFrame."""
        for col in df.columns:
            if col.startswith("id_") and col != "TransactionID":
                self._dtype_map[col] = "category"
            elif pd.api.types.is_float_dtype(df[col]):
                self._dtype_map[col] = "float32"
            elif pd.api.types.is_integer_dtype(df[col]):
                c_min, c_max = df[col].min(), df[col].max()
                if c_min >= 0:
                    if c_max <= np.iinfo(np.uint8).max:
                        self._dtype_map[col] = "uint8"
                    elif c_max <= np.iinfo(np.uint16).max:
                        self._dtype_map[col] = "uint16"
                    elif c_max <= np.iinfo(np.uint32).max:
                        self._dtype_map[col] = "uint32"
                    else:
                        self._dtype_map[col] = "uint64"
                else:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        self._dtype_map[col] = "int8"
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        self._dtype_map[col] = "int16"
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        self._dtype_map[col] = "int32"
                    else:
                        self._dtype_map[col] = "int64"

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored dtype optimizations to a DataFrame."""
        if self._fitted and self._dtype_map:
            for col, target_dtype in self._dtype_map.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(target_dtype)
                    except (ValueError, TypeError):
                        pass
        else:
            df = _downcast_numeric(df)
            df = _convert_id_cols_to_category(df)
        return df

    # ------------------------------------------------------------------
    # Loaders — dispatch on data_format
    # ------------------------------------------------------------------

    def _read_file(self, name: str) -> pd.DataFrame:
        """Read ``<name>.csv`` or ``<name>.parquet`` per ``data_format``."""
        if self.data_format == "parquet":
            return self._read_parquet_cached(name)
        return pd.read_csv(self.data_dir / f"{name}.csv")

    def load_train(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and optimize training transaction and identity data."""
        train_transaction = self._read_file("train_transaction")
        train_identity = self._read_file("train_identity")

        train_transaction = _downcast_numeric(train_transaction)
        train_identity = _downcast_numeric(train_identity)
        train_identity = _convert_id_cols_to_category(train_identity)

        self.fit(train_transaction)

        return train_transaction, train_identity

    def load_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and optimize test transaction and identity data."""
        test_transaction = self._read_file("test_transaction")
        test_identity = self._read_file("test_identity")

        test_transaction = self.transform(test_transaction)
        test_identity = self.transform(test_identity)

        return test_transaction, test_identity

    # ------------------------------------------------------------------
    # Parquet convenience methods (Task 4)
    # ------------------------------------------------------------------

    @staticmethod
    def to_parquet(df: pd.DataFrame, path: Union[str, Path]) -> Path:
        """Write *df* to Parquet (module-level :func:`to_parquet` alias)."""
        return to_parquet(df, path)

    @staticmethod
    def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
        """Read a Parquet file (module-level :func:`read_parquet` alias)."""
        return read_parquet(path)


def make_loader(
    engine: str = "pandas",
    data_dir: Union[str, Path, None] = None,
    data_format: str = "csv",
) -> Union[DataLoader, "PolarsDataLoader"]:
    """Factory returning a loader by engine.

    Parameters
    ----------
    engine : ``"pandas"`` | ``"polars"``
        Backend selection. ``"polars"`` returns a :class:`PolarsDataLoader`
        if polars is importable; otherwise falls back to :class:`DataLoader`
        with a warning.
    data_dir : str | Path, optional
        Path to the raw data directory.
    data_format : ``"csv"`` | ``"parquet"``
        Forwarded to the underlying loader's ``data_format`` argument.

    Returns
    -------
    DataLoader | PolarsDataLoader
    """
    engine_lower = (engine or "pandas").lower()
    if engine_lower == "polars":
        try:
            from .loader_polars import PolarsDataLoader  # local import; polars optional
        except ImportError:
            warnings.warn(
                "Polars not installed; falling back to pandas DataLoader.",
                stacklevel=2,
            )
            return DataLoader(data_dir=data_dir, data_format=data_format)
        return PolarsDataLoader(data_dir=data_dir, data_format=data_format)
    return DataLoader(data_dir=data_dir, data_format=data_format)
