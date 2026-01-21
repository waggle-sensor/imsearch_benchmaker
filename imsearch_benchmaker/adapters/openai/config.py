"""
config.py

OpenAI-specific VisionConfig and JudgeConfig extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os

from ...framework.config import JudgeConfig, VisionConfig


@dataclass(frozen=True)
class OpenAIConfig:
    """
    OpenAI-specific config class.
    
    Attributes:
        _openai_api_key: OpenAI API key. If not provided, uses environment variable OPENAI_API_KEY.
        completion_window: Completion window for the batch. If not provided, uses default of 24 hours.
        price_per_million_input_tokens: Price per million uncached input text tokens. Must be set for cost calculation.
        price_per_million_cached_input_tokens: Price per million cached input text tokens (optional, for prompt caching).
        price_per_million_output_tokens: Price per million output text tokens. Must be set for cost calculation.
    """
    _openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    completion_window: Optional[str] = None
    price_per_million_input_tokens: Optional[float] = None
    price_per_million_cached_input_tokens: Optional[float] = None
    price_per_million_output_tokens: Optional[float] = None
    
    def get_effective_pricing(self) -> dict[str, float]:
        """
        Get effective pricing from config. All required pricing values must be explicitly set.
        
        Returns:
            Dictionary with keys: input_tokens, output_tokens, cached_input_tokens (if set).
            
        Raises:
            ValueError: If required pricing values are not set.
        """
        if self.price_per_million_input_tokens is None:
            raise ValueError(
                "price_per_million_input_tokens must be set in config for cost calculation. "
                "Set it in your config file or via the OpenAI config class."
            )
        if self.price_per_million_output_tokens is None:
            raise ValueError(
                "price_per_million_output_tokens must be set in config for cost calculation. "
                "Set it in your config file or via the OpenAI config class."
            )
        result = {
            "input_tokens": self.price_per_million_input_tokens,
            "output_tokens": self.price_per_million_output_tokens,
        }
        # Add cached input pricing if set (optional)
        if self.price_per_million_cached_input_tokens is not None:
            result["cached_input_tokens"] = self.price_per_million_cached_input_tokens
        return result

@dataclass(frozen=True)
class OpenAIVisionConfig(VisionConfig, OpenAIConfig):
    """
    OpenAI-specific vision config class.
    
    Attributes:
        price_per_million_image_input_tokens: Price per million image input tokens. Must be set for cost calculation.
        price_per_million_image_output_tokens: Price per million image output tokens. Must be set for cost calculation.
    """
    price_per_million_image_input_tokens: Optional[float] = None
    price_per_million_image_output_tokens: Optional[float] = None
    
    def get_effective_pricing(self) -> dict[str, float]:
        """
        Get effective pricing from config. All pricing values must be explicitly set.
        
        Returns:
            Dictionary with keys: input_tokens, output_tokens, image_input_tokens, image_output_tokens
            
        Raises:
            ValueError: If required pricing values are not set.
        """
        # Get base pricing (input/output tokens) - this will raise if not set
        base_pricing = super().get_effective_pricing()
        
        if self.price_per_million_image_input_tokens is None:
            raise ValueError(
                "price_per_million_image_input_tokens must be set in config for cost calculation. "
                "Set it in your config file or via the OpenAIVisionConfig class."
            )
        if self.price_per_million_image_output_tokens is None:
            raise ValueError(
                "price_per_million_image_output_tokens must be set in config for cost calculation. "
                "Set it in your config file or via the OpenAIVisionConfig class."
            )
        
        return {
            "input_tokens": base_pricing["input_tokens"],
            "output_tokens": base_pricing["output_tokens"],
            "image_input_tokens": self.price_per_million_image_input_tokens,
            "image_output_tokens": self.price_per_million_image_output_tokens,
        }

@dataclass(frozen=True)
class OpenAIJudgeConfig(JudgeConfig, OpenAIConfig):
    """
    OpenAI-specific judge config class.
    """
    # Add any additional openai judge-specific settings here