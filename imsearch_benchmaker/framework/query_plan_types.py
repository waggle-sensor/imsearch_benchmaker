"""
query_plan_types.py

Types for query plan generation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

from .config import BenchmarkConfig

@dataclass(frozen=True)
class ImageRec:
    """
    Image record containing image metadata for query planning.
    
    Attributes:
        image_id: Unique identifier for the image.
        facets: Dictionary mapping facet names (taxonomy/boolean columns) to their values.
        tags: Set of tag strings associated with the image.
    """
    image_id: str
    facets: Dict[str, Any]
    tags: Set[str]

class QueryPlanStrategy(ABC):
    """
    Abstract base class for query plan generation strategies.
    
    Query plan strategies define how to select candidate images for each query based on
    seed images and annotations. Different strategies can implement different selection
    criteria (e.g., tag overlap, facet matching, diversity constraints).
    """
    
    @abstractmethod
    def build(self, annotations: Dict[str, ImageRec], seeds_path: Path, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """
        Build query plan rows from annotations and seed images.
        
        Args:
            annotations: Dictionary mapping image IDs to ImageRec objects.
            seeds_path: Path to seeds JSONL file containing query seed image IDs.
            config: BenchmarkConfig instance with query planning parameters.
            
        Returns:
            List of dictionaries, each representing a query plan row with query_id,
            seed_image_ids, and candidate_image_ids.
        """