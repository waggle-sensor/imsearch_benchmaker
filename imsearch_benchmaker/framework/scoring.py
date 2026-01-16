"""
clip.py

Base framework for similarity scoring adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Type, Any, Optional
import requests
from io import BytesIO
from PIL import Image
from .scoring_types import SimilarityInput, SimilarityResult
from .config import SimilarityConfig


class Similarity(ABC):
    """
    Abstract base class for similarity scoring adapters.
    """

    @abstractmethod
    def score(self, query: str, image_url: str) -> float:
        """
        Calculate similarity score for a query-image pair.
        
        Args:
            query: The text query.
            image_url: The URL of the image.
            
        Returns:
            A float representing the similarity score.
        """
        pass

    @abstractmethod
    def score_batch(self, inputs: Iterable[SimilarityInput]) -> List[SimilarityResult]:
        """
        Calculate similarity scores for multiple query-image pairs.
        
        Args:
            inputs: Iterable of SimilarityInput objects.
            
        Returns:
            List of SimilarityResult objects.
        """
        pass

    def get_name(self) -> str:
        """
        Identifier for this adapter implementation.
        """
        return self.__class__.__name__

    def _load_image_from_url(self, image_url: str) -> Image.Image:
            """
            Load an image from a URL.
            
            Args:
                image_url: URL of the image.
                
            Returns:
                PIL Image object.
                
            Raises:
                Exception: If the image cannot be loaded.
            """
            try:
                response = requests.get(image_url, timeout=30, stream=True)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return image
            except Exception as e:
                raise Exception(f"Failed to load image from URL {image_url}: {e}") from e


class SimilarityAdapterRegistry:
    """
    Registry for similarity adapters and their config classes.
    """

    _adapters: Dict[str, Type[Similarity]] = {}
    _config_classes: Dict[str, Type] = {}  # Maps adapter name to config class

    @classmethod
    def register(cls, name: str, adapter_class: Type[Similarity], config_class: Type[SimilarityConfig]) -> None:
        if not issubclass(adapter_class, Similarity):
            raise ValueError(f"[{cls.__name__}] Adapter class must be a subclass of Similarity")
        cls._adapters[name] = adapter_class
        if not issubclass(config_class, SimilarityConfig):
            raise ValueError(f"[{cls.__name__}] Config class must be a subclass of SimilarityConfig")
        cls._config_classes[name] = config_class

    @classmethod
    def get(cls, name: str, **kwargs) -> Similarity:
        if name not in cls._adapters:
            raise ValueError(f"[{cls.__name__}] Adapter '{name}' not found. Available adapters: {list(cls._adapters.keys())}")
        return cls._adapters[name](**kwargs)

    @classmethod
    def get_config_class(cls, name: str) -> Optional[Type]:
        """
        Get the config class for a given adapter name.
        
        Args:
            name: Name of the adapter
            
        Returns:
            Config class if registered, None otherwise
        """
        return cls._config_classes.get(name)

    @classmethod
    def list_adapters(cls) -> List[str]:
        return list(cls._adapters.keys())
