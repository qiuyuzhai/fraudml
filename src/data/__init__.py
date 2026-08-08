"""
FraudML data module.

Provides DataLoader for loading IEEE-CIS data with memory optimization,
DataProfiler for computing per-column statistics, and DataCleaner for
leakage-prevention cleaning with Winsorization.
"""

from .loader import DataLoader
from .profile import DataProfiler
from .cleaner import DataCleaner

__all__ = ["DataLoader", "DataProfiler", "DataCleaner"]