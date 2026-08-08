"""
Feature registry with auto-discovery and config-driven execution.

The :class:`FeatureRegistry` manages a collection of
:class:`~src.features.base.FeatureBase` transformers:

* **Auto-discovery** — scans a package for all
  :class:`~src.features.base.FeatureBase` subclasses.
* **Config-driven ordering** — reads a YAML config (``feature_steps``
  list) to determine which features run and in what order.
* **Batch helpers** — ``fit_all``, ``transform_all``,
  ``save_all``, ``load_all`` operate on the full pipeline.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type

import pandas as pd
import yaml

from .base import FeatureBase


class FeatureRegistry:
    """Registry for managing feature-engineering transformers.

    Typical workflow::

        registry = FeatureRegistry()
        registry.auto_discover("src.features")
        registry.configure("config.yaml")
        registry.fit_all(train_df)
        result = registry.transform_all(train_df)
        registry.save_all("artifacts/features")

    Parameters
    ----------
    config_path : str | Path, optional
        Path to a YAML file with a ``feature_steps`` key.
        Can also be set later via :meth:`configure`.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._classes: Dict[str, Type[FeatureBase]] = {}
        self._instances: Dict[str, FeatureBase] = {}
        self._execution_order: List[str] = []

        if config_path is not None:
            self.configure(config_path)

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    def auto_discover(self, package: str) -> List[Type[FeatureBase]]:
        """Find all :class:`FeatureBase` subclasses in *package*.

        Uses :func:`pkgutil.walk_packages` to scan every submodule
        (including nested packages) under *package* on the filesystem.
        Discovers **classes only** — instantiation is driven by
        the config file (:meth:`configure`).

        Parameters
        ----------
        package : str
            Dotted module path, e.g. ``"src.features"``.

        Returns
        -------
        list[type[FeatureBase]]
            Discovered classes.
        """
        pkg = importlib.import_module(package)
        classes: Dict[str, Type[FeatureBase]] = {}

        for _, mod_name, _ in pkgutil.walk_packages(
            pkg.__path__, prefix=package + "."
        ):
            mod = importlib.import_module(mod_name)
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, FeatureBase) and obj is not FeatureBase:
                    classes[obj.__name__] = obj

        self._classes = classes
        return list(classes.values())

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, path: str | Path) -> List[str]:
        """Read a YAML config and instantiate features in order.

        The config file must contain a top-level key
        ``feature_steps`` — a list of class names (strings)
        in the order they should execute.

        Example YAML::

            feature_steps:
              - DataCleaner
              - SomeEncoder
              - SomeScaler

        Parameters
        ----------
        path : str | Path
            Path to the YAML config file.

        Returns
        -------
        list[str]
            Ordered list of feature names (== class names).

        Raises
        ------
        ValueError
            If a named feature class has not been discovered.
        """
        path = Path(path)
        with open(path) as fh:
            config = yaml.safe_load(fh)

        steps: List[str] = config.get("feature_steps", [])

        self._instances.clear()
        self._execution_order = []

        for class_name in steps:
            if class_name not in self._classes:
                raise ValueError(
                    f"Unknown feature '{class_name}'. "
                    f"Run auto_discover() first. "
                    f"Discovered: {list(self._classes.keys())}"
                )
            cls = self._classes[class_name]
            instance = cls(name=class_name)
            self._instances[class_name] = instance
            self._execution_order.append(class_name)

        return list(self._execution_order)

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def fit_all(self, df: pd.DataFrame) -> "FeatureRegistry":
        """Fit every registered feature in execution order.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        self : FeatureRegistry
        """
        for name in self._execution_order:
            feat = self._instances[name]
            feat.fit(df)
        return self

    def transform_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply every fitted feature in execution order.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (training / validation / test).

        Returns
        -------
        pd.DataFrame
        """
        for name in self._execution_order:
            feat = self._instances[name]
            df = feat.transform(df)
        return df

    def fit_transform_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        self.fit_all(df)
        return self.transform_all(df)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate full feature set in one call.

        Combines auto_discover + configure + fit_transform_all.
        Requires that ``config.yaml`` has been set (via constructor
        or :meth:`configure`).

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        if not self._classes:
            raise RuntimeError(
                "No feature classes discovered. "
                "Call auto_discover() first or pass config_path to constructor."
            )
        if not self._execution_order:
            raise RuntimeError(
                "No features configured. Call configure() first."
            )
        return self.fit_transform_all(df)

    # ------------------------------------------------------------------
    # Batch persistence
    # ------------------------------------------------------------------

    def save_all(self, base_path: str | Path) -> None:
        """Save every fitted feature's parameters to disk.

        Each feature is saved as ``{base_path}/{name}.joblib``.

        Parameters
        ----------
        base_path : str | Path
            Directory under which per-feature files are created.

        Raises
        ------
        RuntimeError
            If any feature has not been fitted.
        """
        base_path = Path(base_path)
        for name, feat in self._instances.items():
            feat.save(base_path / f"{name}.joblib")

    def load_all(self, base_path: str | Path) -> "FeatureRegistry":
        """Restore every feature's parameters from disk.

        Parameters
        ----------
        base_path : str | Path
            Directory containing ``{name}.joblib`` files.

        Returns
        -------
        self : FeatureRegistry
        """
        base_path = Path(base_path)
        for name, feat in self._instances.items():
            feat.load(base_path / f"{name}.joblib")
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def features(self) -> Dict[str, FeatureBase]:
        """Dict of name → instance."""
        return self._instances

    @property
    def execution_order(self) -> List[str]:
        """Ordered list of feature names."""
        return list(self._execution_order)

    def __repr__(self) -> str:
        discovered = list(self._classes.keys())
        configured = self._execution_order
        return (
            f"FeatureRegistry("
            f"discovered={discovered}, "
            f"configured={configured}"
            f")"
        )