"""
vision.py

Base framework for vision annotation adapters.

This module provides the abstract base class `Vision` that all vision annotation
adapters must implement, along with a registry system for dynamic adapter lookup
and instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Type, Any, Optional
import re

from .vision_types import VisionAnnotation, VisionImage
from .config import BenchmarkConfig, VisionConfig
from .cost import CostSummary

class Vision(ABC):
    """
    Abstract base class for vision annotation adapters.
    
    This class defines the interface that all vision annotation adapters must implement.
    Adapters are responsible for converting images to provider-specific request formats,
    submitting batches, waiting for completion, downloading results, and parsing responses.
    
    Attributes:
        batch_endpoint: Endpoint path for batch API requests (default: "/v1/responses").
        config: BenchmarkConfig instance containing adapter and benchmark settings.
        client: Provider-specific client instance (e.g., OpenAI client).
    """

    batch_endpoint: str = "/v1/responses"

    def __init__(self, config: BenchmarkConfig, client: Any = None) -> None:
        """
        Initialize a vision adapter.
        
        Args:
            config: BenchmarkConfig instance with vision adapter settings.
            client: Optional provider-specific client instance.
        """
        self.config = config
        self.client = client

    @abstractmethod
    def build_request(self, image: VisionImage) -> Dict[str, object]:
        """
        Build a provider-specific request body for a single image.
        
        Args:
            image: VisionImage object containing image_id, image_url, and metadata.
            
        Returns:
            Dictionary representing the provider-specific request body.
        """

    @abstractmethod
    def parse_response(self, response_body: Dict[str, object], image: VisionImage) -> VisionAnnotation:
        """
        Parse a provider response body into a VisionAnnotation.
        
        Args:
            response_body: Dictionary containing the provider's response.
            image: Original VisionImage that was submitted.
            
        Returns:
            VisionAnnotation object with parsed results.
        """

    @abstractmethod
    def submit(self, images: Iterable[VisionImage], **kwargs: Any) -> object:
        """
        Submit images to the provider and return a provider-specific reference.
        
        Args:
            images: Iterable of VisionImage objects to submit.
            **kwargs: Additional adapter-specific arguments.
            
        Returns:
            Provider-specific batch reference (e.g., BatchRefs for OpenAI).
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
        
        Converts VisionImage objects into batch API request format with custom_id,
        method, url, and body fields.
        
        Args:
            images: Iterable of VisionImage objects to convert.
            
        Yields:
            Dictionary representing a single batch request line.
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
        Get the identifier for this adapter implementation.
        
        Returns:
            Class name of the adapter (e.g., "OpenAIVision").
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

    @abstractmethod
    def extract_usage_from_response(self, response_body: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract usage/token information from a provider response body.
        
        This method should extract token usage data from the response structure
        in a provider-specific format.
        
        Args:
            response_body: Response body dictionary from the provider API.
            
        Returns:
            Dictionary with keys: input_tokens, cached_tokens (optional), output_tokens, 
            image_input_tokens, image_output_tokens. All values default to 0 if usage data is not found.
            cached_tokens is optional and only needed if the provider supports prompt caching.
        """
        pass

    @abstractmethod
    def calculate_actual_costs(
        self,
        batch_output_jsonl: Path,
        num_items: Optional[int] = None,
    ) -> CostSummary:
        """
        Calculate actual costs from batch output JSONL file.
        
        Extracts usage data from each successful response and calculates total costs
        based on pricing from config.
        
        Args:
            batch_output_jsonl: Path to batch output JSONL file.
            num_items: Number of items processed. If None, will be counted from successful responses.
            
        Returns:
            CostSummary object with calculated costs.
        """
        pass

    @staticmethod
    def interpolate_prompt(prompt: str, metadata_dict: Dict[str, Any]) -> str:
        """Interpolate metadata placeholders in prompt text.
        If {metadata.column_name} is in the prompt but metadata is missing/None, replace with 'None (no label)'.
        """
        if not prompt:
            return prompt

        # Regex to find all {metadata.column_name} patterns
        pattern = r"\{metadata\.([a-zA-Z0-9_]+)\}"
        result = prompt

        # Search for all placeholders
        placeholders = re.findall(pattern, prompt)
        for key in set(placeholders):
            value = metadata_dict.get(key, None)
            if value is None:
                str_value = "None (no label)"
            else:
                str_value = str(value)
            result = result.replace(f"{{metadata.{key}}}", str_value)
        return result

class VisionAdapterRegistry:
    """
    Registry for vision adapters and their config classes.
    
    This registry allows adapters to be registered by name and dynamically
    instantiated. It also maintains a mapping between adapter names and their
    corresponding config classes for automatic config class selection.
    """

    _adapters: Dict[str, Type[Vision]] = {}
    _config_classes: Dict[str, Type] = {}  # Maps adapter name to config class

    @classmethod
    def register(cls, name: str, adapter_class: Type[Vision], config_class: Type[VisionConfig]) -> None:
        """
        Register a vision adapter and its config class.
        
        Args:
            name: Adapter name (e.g., "openai").
            adapter_class: Vision adapter class to register.
            config_class: Corresponding VisionConfig subclass.
            
        Raises:
            ValueError: If adapter_class is not a Vision subclass or config_class is not a VisionConfig subclass.
        """
        if not issubclass(adapter_class, Vision):
            raise ValueError(f"[{cls.__name__}] Adapter class must be a subclass of Vision")
        cls._adapters[name] = adapter_class
        if not issubclass(config_class, VisionConfig):
            raise ValueError(f"[{cls.__name__}] Config class must be a subclass of VisionConfig")
        cls._config_classes[name] = config_class

    @classmethod
    def get(cls, name: str, **kwargs) -> Vision:
        """
        Get an instance of a registered vision adapter.
        
        Args:
            name: Adapter name to instantiate.
            **kwargs: Arguments to pass to the adapter constructor (typically config and client).
            
        Returns:
            Instance of the requested vision adapter.
            
        Raises:
            ValueError: If the adapter name is not registered.
        """
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
        """
        List all registered adapter names.
        
        Returns:
            List of registered adapter names.
        """
        return list(cls._adapters.keys())

