"""
vision.py

Base framework for vision annotation adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Type, Any, Optional

from .vision_types import VisionAnnotation, VisionImage
from .config import BenchmarkConfig, VisionConfig

class Vision(ABC):
    """
    Abstract base class for vision annotation adapters.
    """

    batch_endpoint: str = "/v1/responses"

    def __init__(self, config: BenchmarkConfig, client: Any = None) -> None:
        self.config = config
        self.client = client

    @abstractmethod
    def build_request(self, image: VisionImage) -> Dict[str, object]:
        """
        Build a provider-specific request body for a single image.
        """

    @abstractmethod
    def parse_response(self, response_body: Dict[str, object], image: VisionImage) -> VisionAnnotation:
        """
        Parse a provider response body into a VisionAnnotation.
        """

    @abstractmethod
    def submit(self, images: Iterable[VisionImage], **kwargs: Any) -> object:
        """
        Submit images to the provider and return a provider-specific reference.
        """

    def get_client(self, config: Any = None) -> Any:
        """
        Get or create a client for this adapter.
        Returns None if the adapter doesn't need a client.
        Override in subclasses if client creation is needed.
        """
        return None

    def wait_for_batch(self, batch_ref: Any) -> None:
        """
        Wait for a batch to complete.
        Override in subclasses to implement provider-specific waiting logic.
        """
        pass

    def download_batch_results(
        self, batch_ref: Any, output_path: Path, error_path: Optional[Path] = None
    ) -> None:
        """
        Download batch results to output_path and optionally errors to error_path.
        Override in subclasses to implement provider-specific download logic.
        """
        pass

    def build_batch_lines(self, images: Iterable[VisionImage]) -> Iterable[Dict[str, object]]:
        """
        Build JSONL lines for batch execution.
        """
        for image in images:
            yield {
                "custom_id": f"{self.config.vision_config.stage}::{image.image_id}",
                "method": "POST",
                "url": self.batch_endpoint,
                "body": self.build_request(image),
            }

    def get_name(self) -> str:
        """
        Identifier for this adapter implementation.
        """
        return self.__class__.__name__

    def list_batches(self, active_only: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List batches for this adapter.
        
        Args:
            active_only: If True, only return active batches
            limit: Maximum number of batches to return
            
        Returns:
            List of batch dictionaries (format depends on adapter implementation)
        """
        return []

class VisionAdapterRegistry:
    """
    Registry for vision adapters and their config classes.
    """

    _adapters: Dict[str, Type[Vision]] = {}
    _config_classes: Dict[str, Type] = {}  # Maps adapter name to config class

    @classmethod
    def register(cls, name: str, adapter_class: Type[Vision], config_class: Type[VisionConfig]) -> None:
        if not issubclass(adapter_class, Vision):
            raise ValueError(f"[{cls.__name__}] Adapter class must be a subclass of Vision")
        cls._adapters[name] = adapter_class
        if not issubclass(config_class, VisionConfig):
            raise ValueError(f"[{cls.__name__}] Config class must be a subclass of VisionConfig")
        cls._config_classes[name] = config_class

    @classmethod
    def get(cls, name: str, **kwargs) -> Vision:
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

