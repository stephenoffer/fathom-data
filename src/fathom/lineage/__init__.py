"""Lineage acquisition. SQL parsing today; execution-plan listeners next."""

from .sql import Extraction, extract

__all__ = ["Extraction", "extract"]
