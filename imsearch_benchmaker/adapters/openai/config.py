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
    """
    _openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    completion_window: Optional[str] = None

@dataclass(frozen=True)
class OpenAIVisionConfig(VisionConfig, OpenAIConfig):
    """
    OpenAI-specific vision config class.
    """
    # Add any additional openai vision-specific settings here

@dataclass(frozen=True)
class OpenAIJudgeConfig(JudgeConfig, OpenAIConfig):
    """
    OpenAI-specific judge config class.
    """
    # Add any additional openai judge-specific settings here