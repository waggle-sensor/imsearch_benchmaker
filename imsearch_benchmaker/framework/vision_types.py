"""
vision_types.py

Types for vision annotation inputs/outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VisionImage:
    """
    Input image metadata for vision annotation.
    """

    image_id: str
    image_url: str
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionAnnotation:
    """
    Parsed vision annotation output.
    """

    image_id: str
    fields: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    confidence: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

