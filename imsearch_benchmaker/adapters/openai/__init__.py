"""
OpenAI adapters for vision and judge stages.
"""

from .vision import OpenAIVision
from .judge import OpenAIJudge
from .config import OpenAIVisionConfig, OpenAIJudgeConfig
from ...framework.judge import JudgeAdapterRegistry
from ...framework.vision import VisionAdapterRegistry

JudgeAdapterRegistry.register("openai", OpenAIJudge, config_class=OpenAIJudgeConfig)
VisionAdapterRegistry.register("openai", OpenAIVision, config_class=OpenAIVisionConfig)

__all__ = [
    "OpenAIVision",
    "OpenAIJudge",
    "OpenAIVisionConfig",
    "OpenAIJudgeConfig",
]

