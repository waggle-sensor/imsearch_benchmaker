"""
config.py

Local CLIP adapter configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...framework.config import SimilarityConfig


@dataclass(frozen=True)
class CLIPConfig(SimilarityConfig):
    """
    Configuration for local CLIP adapter.
    """
    
    device: Optional[str] = "auto"
    torch_dtype: Optional[str] = None
    use_safetensors: Optional[bool] = True

