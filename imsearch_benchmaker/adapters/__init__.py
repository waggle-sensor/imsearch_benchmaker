"""
Adapters for specific providers.
"""
from ..framework.judge import JudgeAdapterRegistry
from ..framework.vision import VisionAdapterRegistry
from .openai import OpenAIVision, OpenAIJudge, OpenAIVisionConfig, OpenAIJudgeConfig
from ..framework.scoring import SimilarityAdapterRegistry
from .local import CLIP, CLIPConfig

#openai adapters
JudgeAdapterRegistry.register("openai", OpenAIJudge, config_class=OpenAIJudgeConfig)
VisionAdapterRegistry.register("openai", OpenAIVision, config_class=OpenAIVisionConfig)

#local adapters
SimilarityAdapterRegistry.register("local_clip", CLIP, config_class=CLIPConfig)