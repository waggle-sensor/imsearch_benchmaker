"""
Local adapters for similarity scoring.
"""

from .clip import CLIP
from .config import CLIPConfig
from ...framework.scoring import SimilarityAdapterRegistry

SimilarityAdapterRegistry.register("local_clip", CLIP, config_class=CLIPConfig)

__all__ = [
    "CLIP",
    "CLIPConfig",
]

