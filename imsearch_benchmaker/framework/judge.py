"""
judge.py

Base framework for judge adapters.

This module provides the abstract base class `Judge` that all relevance judgment
adapters must implement, along with a registry system for dynamic adapter lookup
and instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Type, Any, Optional

from .judge_types import JudgeQuery, JudgeResult
from .config import BenchmarkConfig, JudgeConfig
from .cost import CostSummary


class Judge(ABC):
    """
    Abstract base class for query+relevance judge adapters.
    
    This class defines the interface that all relevance judgment adapters must implement.
    Adapters are responsible for converting queries to provider-specific request formats,
    submitting batches, waiting for completion, downloading results, and parsing responses.
    
    Attributes:
        batch_endpoint: Endpoint path for batch API requests (default: "/v1/responses").
        config: BenchmarkConfig instance containing adapter and benchmark settings.
        client: Provider-specific client instance (e.g., OpenAI client).
    """

    batch_endpoint: str = "/v1/responses"

    def __init__(self, config: BenchmarkConfig, client: Any = None) -> None:
        """
        Initialize a judge adapter.
        
        Args:
            config: BenchmarkConfig instance with judge adapter settings.
            client: Optional provider-specific client instance.
        """
        self.config = config
        self.client = client

    @abstractmethod
    def build_request(self, query: JudgeQuery) -> Dict[str, object]:
        """
        Build a provider-specific request body for a query.
        
        Args:
            query: JudgeQuery object containing query_id, query_text, candidate images, and metadata.
            
        Returns:
            Dictionary representing the provider-specific request body.
        """

    @abstractmethod
    def parse_response(self, response_body: Dict[str, object], query: JudgeQuery) -> JudgeResult:
        """
        Parse a provider response body into a JudgeResult.
        
        Args:
            response_body: Dictionary containing the provider's response.
            query: Original JudgeQuery that was submitted.
            
        Returns:
            JudgeResult object with parsed relevance scores.
        """

    @abstractmethod
    def submit(self, queries: Iterable[JudgeQuery], **kwargs: Any) -> object:
        """
        Submit queries to the provider and return a provider-specific reference.
        
        Args:
            queries: Iterable of JudgeQuery objects to submit.
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

    def build_batch_lines(self, queries: Iterable[JudgeQuery]) -> Iterable[Dict[str, object]]:
        """
        Build JSONL lines for batch execution.
        
        Converts JudgeQuery objects into batch API request format with custom_id,
        method, url, and body fields.
        
        Args:
            queries: Iterable of JudgeQuery objects to convert.
            
        Yields:
            Dictionary representing a single batch request line.
        """
        for query in queries:
            yield {
                "custom_id": f"{self.config.judge_config.stage}::{query.query_id}",
                "method": "POST",
                "url": self.batch_endpoint,
                "body": self.build_request(query),
            }

    def get_name(self) -> str:
        """
        Get the identifier for this adapter implementation.
        
        Returns:
            Class name of the adapter (e.g., "OpenAIJudge").
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

class JudgeAdapterRegistry:
    """
    Registry for judge adapters and their config classes.
    
    This registry allows adapters to be registered by name and dynamically
    instantiated. It also maintains a mapping between adapter names and their
    corresponding config classes for automatic config class selection.
    """

    _adapters: Dict[str, Type[Judge]] = {}
    _config_classes: Dict[str, Type] = {}  # Maps adapter name to config class

    @classmethod
    def register(cls, name: str, adapter_class: Type[Judge], config_class: Type[JudgeConfig]) -> None:
        """
        Register a judge adapter and its config class.
        
        Args:
            name: Adapter name (e.g., "openai").
            adapter_class: Judge adapter class to register.
            config_class: Corresponding JudgeConfig subclass.
            
        Raises:
            ValueError: If adapter_class is not a Judge subclass or config_class is not a JudgeConfig subclass.
        """
        if not issubclass(adapter_class, Judge):
            raise ValueError("Adapter class must be a subclass of Judge")
        cls._adapters[name] = adapter_class
        if not issubclass(config_class, JudgeConfig):
            raise ValueError("Config class must be a subclass of JudgeConfig")
        cls._config_classes[name] = config_class

    @classmethod
    def get(cls, name: str, **kwargs) -> Judge:
        """
        Get an instance of a registered judge adapter.
        
        Args:
            name: Adapter name to instantiate.
            **kwargs: Arguments to pass to the adapter constructor (typically config and client).
            
        Returns:
            Instance of the requested judge adapter.
            
        Raises:
            ValueError: If the adapter name is not registered.
        """
        if name not in cls._adapters:
            raise ValueError(f"Adapter '{name}' not found. Available adapters: {list(cls._adapters.keys())}")
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

