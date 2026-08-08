"""
Data loading module for FraudML.

Provides DataLoader for loading IEEE-CIS Fraud Detection data
with automatic memory optimization through dtype downcasting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to the smallest safe dtype.

    - float64 → float32  (always safe, minor precision loss)
    - int64  → int8/16/32 or uint8/16/32 based on value range

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
    """Convert columns starting with ``id_`` (except ``TransactionID``) to category dtype.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing ``id_*`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``id_*`` columns cast to ``category``.
    """
    id_cols = [
        col for col in df.columns if col.startswith("id_") and col != "TransactionID"
    ]
    for col in id_cols:
        df[col] = df[col].astype("category")
    return df


class DataLoader:
    """Loads IEEE-CIS Fraud Detection data with automatic memory optimization.

    Implements a fit/transform interface so that dtype decisions made on
    the training set are consistently applied to the test set.

    Parameters
    ----------
    data_dir : str or Path, optional
        Path to the raw data directory. Defaults to ``../../data/raw/``
        relative to the loader module.

    Attributes
    ----------
    data_dir : Path
        Resolved path to the data directory.
    _dtype_map : dict
        Mapping of column names to target dtypes, populated after ``fit()``.
    _fitted : bool
        Whether ``fit()`` has been called.

    Examples
    --------
    >>> loader = DataLoader()
    >>> train_txn, train_id = loader.load_train()
    >>> test_txn, test_id = loader.load_test()
    """

    def __init__(self, data_dir: Union[str, Path, None] = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
        self.data_dir: Path = Path(data_dir)
        self._dtype_map: Dict[str, str] = {}
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DataLoader":
        """Store target dtypes from a training DataFrame.

        Although the optimization rules are deterministic (downcast floats,
        shrink ints by range, convert ``id_*`` to category), calling
        ``fit()`` records the mapping so that subsequent ``transform()``
        calls on different DataFrames follow the same decisions.

        Parameters
        ----------
        df : pd.DataFrame
            The training DataFrame to learn dtypes from.

        Returns
        -------
        self : DataLoader
        """
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
        """Apply stored dtype optimizations to a DataFrame.

        If ``fit()`` has not been called, this falls back to the same
        deterministic rules (downcast + id_ category conversion).

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Memory-optimized DataFrame.
        """
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

    def load_train(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and optimize training transaction and identity data.

        Reads ``train_transaction.csv`` and ``train_identity.csv`` from
        ``self.data_dir``, applies memory optimization, and calls
        ``fit()`` on the transaction DataFrame so that subsequent
        ``load_test()`` calls use consistent dtype decisions.

        Returns
        -------
        train_transaction : pd.DataFrame
            Training transaction data (memory-optimized).
        train_identity : pd.DataFrame
            Training identity data (memory-optimized).
        """
        train_transaction = pd.read_csv(self.data_dir / "train_transaction.csv")
        train_identity = pd.read_csv(self.data_dir / "train_identity.csv")

        train_transaction = _downcast_numeric(train_transaction)
        train_identity = _downcast_numeric(train_identity)
        train_identity = _convert_id_cols_to_category(train_identity)

        self.fit(train_transaction)

        return train_transaction, train_identity

    def load_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and optimize test transaction and identity data.

        Reads ``test_transaction.csv`` and ``test_identity.csv`` from
        ``self.data_dir``, then applies the same dtype decisions learned
        during ``load_train()``.

        Returns
        -------
        test_transaction : pd.DataFrame
            Test transaction data (memory-optimized).
        test_identity : pd.DataFrame
            Test identity data (memory-optimized).
        """
        test_transaction = pd.read_csv(self.data_dir / "test_transaction.csv")
        test_identity = pd.read_csv(self.data_dir / "test_identity.csv")

        test_transaction = self.transform(test_transaction)
        test_identity = self.transform(test_identity)

        return test_transaction, test_identity