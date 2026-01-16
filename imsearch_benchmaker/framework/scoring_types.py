"""
clip_types.py

Types for similarity scoring inputs/outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SimilarityInput:
    """
    Input for similarity scoring between a query and an image.
    """
    query: str
    image_url: str
    query_id: Optional[str] = None
    image_id: Optional[str] = None


@dataclass(frozen=True)
class SimilarityResult:
    """
    Result of similarity scoring.
    """
    query: str
    image_url: str
    score: float
    query_id: Optional[str] = None
    image_id: Optional[str] = None
