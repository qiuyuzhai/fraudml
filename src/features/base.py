"""
Abstract base class for feature engineering transformers.

Every feature-engineering step (imputation, encoding, scaling, ...)
must inherit from :class:`FeatureBase` and implement the contract.

Design principles
-----------------
* **Fit / Transform separation** — ``fit()`` learns parameters from
  training data only; ``transform()`` applies them to any dataset
  (train / val / test) without computing new statistics.
* **Pure pandas contract** — both methods accept and return
  ``pd.DataFrame``.  No IO, no side-effects.
* **Serializable parameters only** — ``save()`` / ``load()``
  persist learned parameters, never raw data.  This keeps the
  transformer portable and future-compatible with systems like
  Feast (serving can reconstruct features from stored params).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


class FeatureBase(ABC):
    """Abstract base class for all feature engineering transformers.

    Subclasses **must** implement :meth:`fit` and :meth:`transform`.
    They **may** override :meth:`save` and :meth:`load` if the default
    joblib-based serialization is insufficient.

    Parameters
    ----------
    name : str
        Human-readable identifier for this feature step.  Used by
        :class:`FeatureRegistry` for ordering and logging.

    Attributes
    ----------
    name : str
    _fitted : bool
        Whether :meth:`fit` has been called successfully.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Core contract — fit / transform
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "FeatureBase":
        """Learn parameters from **training data only**.

        Must NOT read from / write to disk.  Must NOT modify the
        input DataFrame in-place (operate on a copy if needed).

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        self : FeatureBase
        """
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned parameters.  No new statistics computed.

        Must NOT read from / write to disk.  Returns a new
        DataFrame (or a view).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (training / validation / test).

        Returns
        -------
        pd.DataFrame
        """
        ...

    # ------------------------------------------------------------------
    # Persistence — serialize learned parameters only
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist learned parameters via joblib.

        Subclasses that store learned parameters as instance
        attributes can rely on this default implementation.
        Override if custom serialization is needed.

        Parameters
        ----------
        path : str | Path
            Destination file path.

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted. Cannot save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._get_state()
        joblib.dump(state, path)

    def load(self, path: str | Path) -> "FeatureBase":
        """Restore learned parameters from disk.

        Parameters
        ----------
        path : str | Path
            Path to a file produced by :meth:`save`.

        Returns
        -------
        self : FeatureBase
        """
        state = joblib.load(Path(path))
        self._set_state(state)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # State helpers — override if subclass has non-serializable attrs
    # ------------------------------------------------------------------

    def _get_state(self) -> Dict[str, Any]:
        """Return a JSON / joblib-safe dict of learned parameters.

        Default implementation picks every public attribute whose
        name does not start with ``_`` and whose value is not a
        callable.  Override for finer control.
        """
        state: Dict[str, Any] = {}
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                continue
            if callable(val):
                continue
            state[key] = val
        return state

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Restore learned parameters from a state dict."""
        for key, val in state.items():
            setattr(self, key, val)

    # ------------------------------------------------------------------
    # Metadata — for drift monitoring & interpretability
    # ------------------------------------------------------------------

    @property
    def _col_suffix(self) -> str:
        if self.name == self.__class__.__name__:
            return ""
        return f"_{self.name}"

    @property
    def is_stateful(self) -> bool:
        """Whether this feature learns parameters from training data.

        Stateful features MUST be persisted for online inference.
        Stateless features are pure transforms that can be recreated
        without training data.

        Override in subclasses that learn parameters (e.g. encoders,
        target encoding, missing pattern detection).
        """
        return False

    @property
    def is_streaming(self) -> bool:
        """Whether this feature supports real-time incremental updates.

        Streaming features can update their internal state (e.g.
        sequence embeddings) one transaction at a time via
        :meth:`init_streaming` and :meth:`update_stream`.

        Override in subclasses (e.g. SequenceFeature).
        """
        return False

    def init_streaming(self) -> "FeatureBase":
        """Initialize streaming inference state after :meth:`fit`.

        Default is a no-op.  Subclasses that support streaming
        (``is_streaming == True``) should override to build any
        runtime-only components (e.g. LSTMCell stacks).
        """
        return self

    def update_stream(self, row: pd.DataFrame) -> pd.DataFrame:
        """Incrementally update streaming state with new data.

        Default is a no-op — returns *row* unchanged.  Subclasses
        that support streaming should override to update internal
        embeddings / statistics.

        Parameters
        ----------
        row : pd.DataFrame
            New transaction row(s) to absorb into the streaming state.

        Returns
        -------
        pd.DataFrame
            Transformed row(s) with updated features.
        """
        return row

    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for this feature.

        Returns a dict describing all constructor parameters with
        their types, defaults, and descriptions.  Override in
        subclasses to document feature-specific parameters.

        Returns
        -------
        dict with keys:
            - class_name: feature class name
            - layer: architecture layer (generic / fraud-domain / business-domain)
            - is_stateful: whether the feature learns from training data
            - parameters: list of param dicts with name, type, default, description
            - example: example YAML snippet
        """
        return {
            "class_name": self.__class__.__name__,
            "layer": "unknown",
            "is_stateful": self.is_stateful,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": self.__class__.__name__,
                    "description": "Instance name. Leave as class name for single instance; set uniquely for multiple instances.",
                },
            ],
            "example": f"- {self.__class__.__name__}",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        """Return metadata dict for feature drift monitoring.

        Returns
        -------
        dict with keys:
            - feature_names: list of output column names
            - physical_meaning: human-readable description
            - unit: unit of measurement (e.g. 'count', 'usd', 'flag')
            - depends_on_target: bool, whether target y is used
        """
        return {
            "feature_names": [],
            "physical_meaning": "Override in subclass",
            "unit": "",
            "depends_on_target": False,
        }

    def get_input_columns(self) -> list[str]:
        """Return the raw upstream column names this feature reads from.

        Used by the Feature Store to construct lineage edges (which raw
        columns feed into which engineered feature). Default ``[]``
        means "all upstream outputs"; subclasses that operate on a
        well-defined column set (e.g. ``AggregationFeature`` aggregating
        ``TransactionAmt`` grouped by ``card1``) should override to
        list those columns explicitly.
        """
        return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._fitted})"