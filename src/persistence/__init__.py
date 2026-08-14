"""
Persistence module for structured artifact serialization.

Provides ModelSerializer for split persistence of stateful components,
enabling seamless migration to Feast / Hive / Spark.
"""

from .serializer import ModelSerializer

__all__ = ["ModelSerializer"]