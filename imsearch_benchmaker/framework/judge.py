"""
judge.py

Base framework for judge adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Type, Any, Optional

from .judge_types import JudgeQuery, JudgeResult
from .config import BenchmarkConfig, JudgeConfig


class Judge(ABC):
    """
    Abstract base class for query+relevance judge adapters.
    """

    batch_endpoint: str = "/v1/responses"

    def __init__(self, config: BenchmarkConfig, client: Any = None) -> None:
        self.config = config
        self.client = client

    @abstractmethod
    def build_request(self, query: JudgeQuery) -> Dict[str, object]:
        """
        Build a provider-specific request body for a query.
        """

    @abstractmethod
    def parse_response(self, response_body: Dict[str, object], query: JudgeQuery) -> JudgeResult:
        """
        Parse a provider response body into a JudgeResult.
        """

    @abstractmethod
    def submit(self, queries: Iterable[JudgeQuery], **kwargs: Any) -> object:
        """
        Submit queries to the provider and return a provider-specific reference.
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
        """
        for query in queries:
            yield {
                "custom_id": f"{self.config.judge_config.stage}::{query.query_id}",
                "method": "POST",
                "url": self.batch_endpoint,
                "body": self.build_request(query),
            }

    def get_name(self) -> str:
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

class JudgeAdapterRegistry:
    """
    Registry for judge adapters and their config classes.
    """

    _adapters: Dict[str, Type[Judge]] = {}
    _config_classes: Dict[str, Type] = {}  # Maps adapter name to config class

    @classmethod
    def register(cls, name: str, adapter_class: Type[Judge], config_class: Type[JudgeConfig]) -> None:
        if not issubclass(adapter_class, Judge):
            raise ValueError("Adapter class must be a subclass of Judge")
        cls._adapters[name] = adapter_class
        if not issubclass(config_class, JudgeConfig):
            raise ValueError("Config class must be a subclass of JudgeConfig")
        cls._config_classes[name] = config_class

    @classmethod
    def get(cls, name: str, **kwargs) -> Judge:
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
        return list(cls._adapters.keys())

