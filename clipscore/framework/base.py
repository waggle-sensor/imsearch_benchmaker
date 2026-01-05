"""
base.py

Base class for service-agnostic CLIPScore calculation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Type
import requests
from io import BytesIO
from PIL import Image

class CLIPAdapter(ABC):
    """
    Abstract base class for CLIP API adapters.
    Subclass this to implement support for different CLIP API services.
    """
    
    @abstractmethod
    def score(self, query: str, image_url: str) -> float:
        """
        Calculate CLIPScore for a query-image pair.
        
        Args:
            query: The text query.
            image_url: The URL of the image.
            
        Returns:
            A float representing the CLIPScore (typically between 0 and 1, or -1 to 1).
            
        Raises:
            Exception: If the API call fails.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name/identifier of this adapter.
        
        Returns:
            A string identifier for this adapter.
        """
        pass

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

class CLIPAdapterRegistry:
    """
    Registry for CLIP adapters. Allows easy registration and retrieval of adapters.
    """
    
    _adapters: Dict[str, Type[CLIPAdapter]] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[CLIPAdapter]) -> None:
        """
        Register a CLIP adapter class.
        
        Args:
            name: The name/identifier for this adapter.
            adapter_class: The adapter class (must be a subclass of CLIPAdapter).
        """
        if not issubclass(adapter_class, CLIPAdapter):
            raise ValueError(f"Adapter class must be a subclass of CLIPAdapter")
        cls._adapters[name] = adapter_class
    
    @classmethod
    def get(cls, name: str, **kwargs) -> CLIPAdapter:
        """
        Get an instance of a registered adapter.
        
        Args:
            name: The name of the adapter.
            **kwargs: Arguments to pass to the adapter constructor.
            
        Returns:
            An instance of the adapter.
            
        Raises:
            ValueError: If the adapter is not registered.
        """
        if name not in cls._adapters:
            raise ValueError(f"Adapter '{name}' not found. Available adapters: {list(cls._adapters.keys())}")
        return cls._adapters[name](**kwargs)
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered adapter names."""
        return list(cls._adapters.keys())