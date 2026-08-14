"""
Abstract base class for feature selection / filtering steps.

For *removing* features from a feature matrix.  

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class SelectionBase(ABC):
    """Abstract base class for feature selection / filtering steps.

    Parameters
    ----------
    name : str
        Human-readable identifier.  Used by ``__repr__``.

    Attributes
    ----------
    name : str
    _fitted : bool
        Whether :meth:`fit` has been called successfully.
    retained_features_ : list[str]
        Feature column names that survive the selection step.
        Set after :meth:`fit`.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._fitted: bool = False
        self.retained_features_: List[str] = []

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply selection.  Returns a DataFrame with *fewer* columns.

        Subclasses must also set :attr:`retained_features_` after
        :meth:`fit` so that the pipeline can query it uniformly.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (training / validation / test).

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the retained feature columns.
        """
        ...

    @abstractmethod
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return metadata describing the retained features.

        Returns
        -------
        dict with keys:
            - feature_names: list of retained feature column names
            - physical_meaning: human-readable description
            - unit: unit of measurement (e.g. 'count', 'probability', 'score')
            - depends_on_target: bool, whether target y was used
        """
        ...

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._fitted})"