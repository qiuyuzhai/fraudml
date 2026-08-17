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
from typing import Any, Dict, List, Optional, Type

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
        ``feature_steps`` — a list specifying feature classes
        in the order they should execute.

        Two formats are supported:

        1. Simple string (backward compatible)::

            feature_steps:
              - CategoricalEncoder
              - TimeFeature

        2. Dict with parameters (same class can be instantiated
           multiple times with different arguments)::

            feature_steps:
              - CategoricalEncoder
              - TargetEncoderFeature:
                  smoothing: 50.0
                  min_samples: 10
              - HistoryFeature:
                  name: "HistoryFeature_1h"
                  window_seconds: 3600
              - HistoryFeature:
                  name: "HistoryFeature_1day"
                  window_seconds: 86400

        Parameters
        ----------
        path : str | Path
            Path to the YAML config file.

        Returns
        -------
        list[str]
            Ordered list of instance names.

        Raises
        ------
        ValueError
            If a named feature class has not been discovered or
            the step format is invalid.
        """
        path = Path(path)
        with open(path) as fh:
            config = yaml.safe_load(fh)

        steps: List = config.get("feature_steps", [])

        self._instances.clear()
        self._execution_order = []

        for step in steps:
            if isinstance(step, str):
                cls_name = step
                instance_name = step
                params: Dict[str, Any] = {}
            elif isinstance(step, dict):
                cls_name = list(step.keys())[0]
                params = step[cls_name] or {}
                instance_name = params.pop("name", cls_name)
            else:
                raise ValueError(f"Invalid step format: {step}")

            if cls_name not in self._classes:
                raise ValueError(
                    f"Unknown feature '{cls_name}'. "
                    f"Run auto_discover() first. "
                    f"Discovered: {list(self._classes.keys())}"
                )
            cls = self._classes[cls_name]
            instance = cls(name=instance_name, **params)
            self._instances[instance_name] = instance
            self._execution_order.append(instance_name)

        return list(self._execution_order)

    # ------------------------------------------------------------------
    # Step configuration (programmatic, for config overrides)
    # ------------------------------------------------------------------

    def _configure_steps(self, steps: List) -> List[str]:
        """Instantiate features from a step list (strings or dicts).

        Same logic as :meth:`configure` but accepts a list directly
        instead of reading from a YAML file.

        Parameters
        ----------
        steps : list
            Step list.  Each element is either a plain string
            (class name) or a ``{ClassName: {params}}`` dict.

        Returns
        -------
        list[str]
            Ordered list of instance names.
        """
        for step in steps:
            if isinstance(step, str):
                cls_name = step
                instance_name = step
                params: Dict[str, Any] = {}
            elif isinstance(step, dict):
                cls_name = list(step.keys())[0]
                params = step[cls_name] or {}
                instance_name = params.pop("name", cls_name)
            else:
                raise ValueError(f"Invalid step format: {step}")

            if cls_name not in self._classes:
                raise ValueError(
                    f"Unknown feature '{cls_name}'. "
                    f"Run auto_discover() first. "
                    f"Discovered: {list(self._classes.keys())}"
                )
            cls = self._classes[cls_name]
            instance = cls(name=instance_name, **params)
            self._instances[instance_name] = instance
            self._execution_order.append(instance_name)

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
    # Introspection / schema discovery
    # ------------------------------------------------------------------

    def get_class_schema(self, class_name: str) -> Dict[str, Any]:
        """Get the config schema for a discovered feature class.

        Instantiates the class with default arguments (no side-effects)
        and calls its :meth:`~src.features.base.FeatureBase.get_config_schema`.

        Parameters
        ----------
        class_name : str
            Class name (e.g. ``"HistoryFeature"``).

        Returns
        -------
        dict
            Schema dict from the feature class, or empty dict if the
            class has not been discovered.
        """
        if class_name not in self._classes:
            return {}
        cls = self._classes[class_name]
        try:
            dummy = cls()
            return dummy.get_config_schema()
        except TypeError:
            return cls.get_config_schema(cls.__name__)

    def print_all_config_schemas(self) -> str:
        """Print a human-readable summary of every discovered feature's config schema.

        Returns
        -------
        str
            Formatted string that is also printed to stdout.
        """
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  FEATURE CONFIGURATION REFERENCE")
        lines.append("=" * 72)
        lines.append("")
        lines.append("  All discovered feature classes and their configurable parameters.")
        lines.append("  Use these to compose your config.yaml without reading source code.")
        lines.append("")

        layer_order = ["generic", "fraud-domain", "business-domain"]
        layer_labels = {
            "generic": "Layer 1 - Generic (stateless / lightweight)",
            "fraud-domain": "Layer 2 - Fraud-Domain (entity-aware)",
            "business-domain": "Layer 3 - Business-Domain (graph-based, heavy)",
        }

        for layer in layer_order:
            features_in_layer = []
            for cls_name in sorted(self._classes.keys()):
                schema = self.get_class_schema(cls_name)
                if schema.get("layer") == layer:
                    features_in_layer.append((cls_name, schema))

            if not features_in_layer:
                continue

            lines.append(f"{'─' * 72}")
            lines.append(f"  {layer_labels.get(layer, layer)}")
            lines.append(f"{'─' * 72}")
            lines.append("")

            for cls_name, schema in features_in_layer:
                lines.append(f"  [{cls_name}]")
                lines.append(f"    Stateful: {'YES (learns from train data)' if schema.get('is_stateful') else 'NO (stateless)'}")
                lines.append("")

                params = schema.get("parameters", [])
                if params:
                    lines.append("    Parameters:")
                    for p in params:
                        lines.append(f"      {p['name']} ({p['type']})")
                        default_repr = repr(p['default']) if not isinstance(p['default'], str) else p['default']
                        lines.append(f"        Default: {default_repr}")
                        lines.append(f"        {p['description']}")
                        lines.append("")

                example = schema.get("example", "")
                if example:
                    lines.append("    Example:")
                    for el in example.split("\n"):
                        lines.append(f"      {el}")
                    lines.append("")

        lines.append("=" * 72)
        text = "\n".join(lines)
        print(text)
        return text

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