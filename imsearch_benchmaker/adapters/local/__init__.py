"""
Local adapters for similarity scoring.
"""

from .clip import CLIP
from .config import CLIPConfig

__all__ = [
    "CLIP",
    "CLIPConfig",
]

