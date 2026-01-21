"""
Framework base types for vision annotation, judging, and CLIP scoring.
"""

from .vision import Vision, VisionAdapterRegistry
from .judge import Judge, JudgeAdapterRegistry
from .scoring import Similarity, SimilarityAdapterRegistry
from .preprocess import build_images_jsonl, build_seeds_jsonl
from .query_plan import build_query_plan, load_annotations, TagOverlapQueryPlan
from .postprocess import generate_dataset_summary, calculate_similarity_score, huggingface
from .config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG, VisionConfig, JudgeConfig, SimilarityConfig
from .pipeline import (
    run_preprocess,
    run_vision,
    run_query_plan,
    run_judge,
    build_cli_parser,
    main as pipeline_main,
)

__all__ = [
    "Vision",
    "VisionAdapterRegistry",
    "Judge",
    "JudgeAdapterRegistry",
    "Similarity",
    "SimilarityAdapterRegistry",
    "build_images_jsonl",
    "build_seeds_jsonl",
    "build_query_plan",
    "load_annotations",
    "TagOverlapQueryPlan",
    "generate_dataset_summary",
    "calculate_similarity_score",
    "huggingface",
    "BenchmarkConfig",
    "DEFAULT_BENCHMARK_CONFIG",
    "VisionConfig",
    "JudgeConfig",
    "SimilarityConfig",
    "run_preprocess",
    "run_vision",
    "run_query_plan",
    "run_judge",
    "build_cli_parser",
    "pipeline_main",
]

