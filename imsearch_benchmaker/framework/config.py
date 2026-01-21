"""
config.py

Benchmark configuration shared across framework utilities.

This module provides the core configuration classes for the imsearch_benchmaker framework,
including BenchmarkConfig which centralizes all benchmark settings, column names, file paths,
and adapter configurations. Supports loading from TOML or JSON files.
"""

from __future__ import annotations

import csv
import io
import os
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
import tomllib

# TypeVar for generic return types in class methods
T = TypeVar("T", bound="BenchmarkConfig")

# Track if adapters have been discovered to avoid repeated imports
_ADAPTERS_DISCOVERED = False

def _discover_adapters() -> None:
    """
    Automatically discover and import all adapter modules.
    
    This function scans the adapters directory and imports each adapter package's
    __init__.py, which triggers adapter registration. This allows new adapters to
    be added without modifying framework code.
    
    The function uses a global flag to ensure adapters are only discovered once,
    even if called multiple times.
    """
    global _ADAPTERS_DISCOVERED
    if _ADAPTERS_DISCOVERED:
        return
    
    try:
        import imsearch_benchmaker.adapters as adapters_pkg
        import importlib
        import pkgutil
        
        adapters_path = Path(adapters_pkg.__file__).parent
        
        # Discover all subpackages in adapters directory
        for finder, name, ispkg in pkgutil.iter_modules([str(adapters_path)]):
            if ispkg and not name.startswith("_"):
                try:
                    # Import the adapter package's __init__.py which contains registration code
                    importlib.import_module(f"imsearch_benchmaker.adapters.{name}")
                except ImportError:
                    # Silently skip adapters that can't be imported (missing dependencies, etc.)
                    pass
                except Exception:
                    # Silently skip adapters that fail to import for any reason
                    pass
        
        _ADAPTERS_DISCOVERED = True
    except Exception:
        # If adapter discovery fails, continue anyway
        _ADAPTERS_DISCOVERED = True


@dataclass(frozen=True)
class VisionConfig:
    """
    Configuration for vision annotation adapters.
    
    This class holds all configuration parameters specific to vision annotation,
    including model settings, prompts, batch processing limits, and adapter selection.
    
    Attributes:
        adapter: Name of the vision adapter to use (e.g., "openai"). If None, adapter must be specified elsewhere.
        model: Model identifier for the vision service (e.g., "gpt-4o").
        system_prompt: System prompt template for vision annotation requests.
        user_prompt: User prompt template for vision annotation requests.
        max_output_tokens: Maximum number of output tokens allowed in vision responses.
        reasoning_effort: Reasoning effort level (e.g., "low", "medium", "high") for models that support it.
        image_detail: Image detail level for processing (e.g., "low", "high").
        max_images_per_batch: Maximum number of images to include in a single batch submission.
        max_concurrent_batches: Maximum number of batches to submit concurrently.
        stage: Stage identifier for batch metadata (default: "vision").
    """
    
    adapter: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    max_output_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    image_detail: Optional[str] = None
    max_images_per_batch: Optional[int] = None
    max_concurrent_batches: Optional[int] = None
    stage: Optional[str] = "vision"


@dataclass(frozen=True)
class JudgeConfig:
    """
    Configuration for judge adapters.
    
    This class holds all configuration parameters specific to relevance judgment,
    including model settings, prompts, batch processing limits, and adapter selection.
    
    Attributes:
        adapter: Name of the judge adapter to use (e.g., "openai"). If None, adapter must be specified elsewhere.
        model: Model identifier for the judge service (e.g., "gpt-4o").
        system_prompt: System prompt template for judge requests.
        user_prompt: User prompt template for judge requests.
        max_output_tokens: Maximum number of output tokens allowed in judge responses.
        reasoning_effort: Reasoning effort level (e.g., "low", "medium", "high") for models that support it.
        max_queries_per_batch: Maximum number of queries to include in a single batch submission.
        max_candidates: Maximum number of candidate images to include per query in judge requests.
        max_concurrent_batches: Maximum number of batches to submit concurrently.
        stage: Stage identifier for batch metadata (default: "judge").
    """
    
    adapter: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    max_output_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    max_queries_per_batch: Optional[int] = None
    max_candidates: Optional[int] = None
    max_concurrent_batches: Optional[int] = None
    stage: Optional[str] = "judge"


@dataclass(frozen=True)
class SimilarityConfig:
    """
    Configuration for similarity scoring adapters.
    
    This class holds all configuration parameters specific to similarity scoring,
    such as CLIP-based image-text similarity computation.
    
    Attributes:
        adapter: Name of the similarity adapter to use (e.g., "local_clip"). If None, adapter must be specified elsewhere.
        model: Model identifier for similarity computation (e.g., "openai/clip-vit-base-patch32").
        col_name: Column name to use for similarity scores in the output dataset (default: "similarity_score").
    """
    
    adapter: Optional[str] = None
    model: Optional[str] = None
    col_name: Optional[str] = "similarity_score"


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Central configuration class for image search benchmark creation.
    
    This class defines all column names, file paths, adapter configurations, and benchmark
    metadata. Sensitive fields (API keys, tokens) should start with an underscore (_) and
    will be excluded from exports. Any variable starting with "column_" or "columns_" is
    considered a column in the final benchmark dataset.
    
    Attributes:
        benchmark_name: Name of the benchmark dataset.
        benchmark_description: Description of the benchmark dataset.
        benchmark_author: Author name for the benchmark.
        benchmark_author_email: Author email address.
        benchmark_author_affiliation: Author affiliation or institution.
        benchmark_author_orcid: Author ORCID identifier.
        benchmark_author_github: Author GitHub username.
        column_image: Column name for image file paths in the dataset.
        column_image_id: Column name for unique image identifiers.
        column_mime_type: Column name for image MIME types.
        column_license: Column name for image licenses.
        column_doi: Column name for Digital Object Identifiers (DOIs) of the image(s).
        column_query: Column name for query text.
        column_query_id: Column name for unique query identifiers.
        column_relevance: Column name for relevance labels (boolean: 0 or 1).
        column_tags: Column name for image tags (list of strings).
        column_summary: Column name for image summaries.
        column_confidence: Column name for confidence scores for taxonomy columns and boolean columns.
        columns_taxonomy: Dictionary of enum values mapping taxonomy column names to their allowed values.
        columns_boolean: List of boolean column names in the dataset.
        image_base_url: Base URL for constructing image URLs from relative paths.
        image_url_temp_column: Column name for temporary image URLs used for retrieving images in the pipeline.
        image_root_dir: Input directory containing image files for preprocessing.
        meta_json: Path to metadata JSON file for preprocessing.
        images_jsonl: Path to input images JSONL file.
        seeds_jsonl: Path to input seeds JSONL file.
        annotations_jsonl: Path to output annotations JSONL file (vision step output).
        query_plan_jsonl: Path to output query plan JSONL file.
        qrels_jsonl: Path to output qrels JSONL file (judge step output).
        qrels_with_score_jsonl: Path to output qrels with similarity scores JSONL file.
        summary_output_dir: Output directory for dataset summary visualizations.
        hf_dataset_dir: Output directory for Hugging Face dataset format.
        _hf_token: Hugging Face API token (sensitive, excluded from exports).
        _hf_repo_id: Hugging Face repository ID for dataset upload.
        _hf_private: Whether the Hugging Face repository should be private.
        controlled_tag_vocab: Controlled vocabulary list for tagging by the vision adapter.
        vision_config: Vision adapter configuration.
        judge_config: Judge adapter configuration.
        similarity_config: Similarity scoring adapter configuration.
        query_plan_num_seeds: Number of seed images to use in query planning.
        query_plan_seed_image_ids_column: Column name for seed image IDs in query plan.
        query_plan_candidate_image_ids_column: Column name for candidate image IDs in query plan.
        query_plan_core_facets: List of core facet names for query planning.
        query_plan_off_facets: List of off-facet names for query planning.
        query_plan_diversity_facets: List of diversity facet names for query planning.
        query_plan_neg_total: Total number of negative examples per query.
        query_plan_neg_hard: Number of hard negative examples per query.
        query_plan_neg_nearmiss: Number of near-miss negative examples per query.
        query_plan_neg_easy: Number of easy negative examples per query.
        query_plan_random_seed: Random seed for reproducible query planning.
    """

    # Benchmark metadata
    benchmark_name: str = "Default benchmark"
    benchmark_description: str = "A default benchmark configuration"
    benchmark_author: Optional[str] = None
    benchmark_author_email: Optional[str] = None
    benchmark_author_affiliation: Optional[str] = None
    benchmark_author_orcid: Optional[str] = None
    benchmark_author_github: Optional[str] = None
    
    # Dataset column names
    column_image: str = "image"
    column_image_id: str = "image_id"
    column_mime_type: str = "mime_type"
    column_license: str = "license"
    column_doi: str = "doi"
    column_query: str = "query_text"
    column_query_id: str = "query_id"
    column_relevance: str = "relevance_label"
    column_tags: Optional[str] = "tags"
    column_summary: Optional[str] = "summary"
    column_confidence: Optional[str] = "confidence"
    columns_taxonomy: Dict[str, List[str]] = field(default_factory=dict)
    columns_boolean: List[str] = field(default_factory=list)

    # File paths
    image_root_dir: Optional[str] = None
    meta_json: Optional[str] = None
    images_jsonl: Optional[str] = None
    seeds_jsonl: Optional[str] = None
    annotations_jsonl: Optional[str] = None
    query_plan_jsonl: Optional[str] = None
    qrels_jsonl: Optional[str] = None
    qrels_with_score_jsonl: Optional[str] = None
    summary_output_dir: Optional[str] = None
    hf_dataset_dir: Optional[str] = None

    # Hugging Face configuration (sensitive fields use _ prefix)
    _hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    _hf_repo_id: Optional[str] = None
    _hf_private: Optional[bool] = None

    # Controlled tag vocabulary
    controlled_tag_vocab: List[str] = field(default_factory=list)

    # Adapter configurations
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    judge_config: JudgeConfig = field(default_factory=JudgeConfig)
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)

    # Query planning configuration
    query_plan_num_seeds: Optional[int] = None
    query_plan_seed_image_ids_column: str = "seed_image_ids"
    query_plan_candidate_image_ids_column: str = "candidate_image_ids"
    query_plan_core_facets: List[str] = field(default_factory=list)
    query_plan_off_facets: List[str] = field(default_factory=list)
    query_plan_diversity_facets: List[str] = field(default_factory=list)
    query_plan_neg_total: Optional[int] = None
    query_plan_neg_hard: Optional[int] = None
    query_plan_neg_nearmiss: Optional[int] = None
    query_plan_neg_easy: Optional[int] = None
    query_plan_random_seed: Optional[int] = None

    # image URL configuration
    image_base_url: Optional[str] = None 
    image_url_temp_column: str = "image_url"

    def required_qrels_columns(self) -> List[str]:
        """
        Get the list of required column names for qrels output.
        
        Returns:
            List of required column names: query, query_id, relevance, and image_id.
        """
        return [self.column_query, self.column_query_id, self.column_relevance, self.column_image_id]
    
    def get_columns(self) -> List[str]:
        """
        Get all column configuration field names.
        
        Returns:
            List of field names that start with 'column_' or 'columns_' (excluding private fields).
        """
        return [field for field in dir(self) if (field.startswith('column_') or field.startswith('columns_')) and not field.startswith('_')]

    def to_csv(self) -> str:
        """
        Convert config to CSV string format.
        Excludes any fields starting with underscore (_).
        Handles nested configs by flattening them with prefixes.
        
        Returns:
            CSV string with columns: Config Variable, Value, Type
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["Config Variable", "Value", "Type"])
        
        # Write data rows, excluding fields starting with _
        for key, value in sorted(self.__dict__.items()):
            if key.startswith("_"):
                continue
            
            # Handle nested configs
            if isinstance(value, VisionConfig):
                for vkey, vval in sorted(asdict(value).items()):
                    if not vkey.startswith("_"):
                        writer.writerow([f"vision_config.{vkey}", str(vval), type(vval).__name__])
            elif isinstance(value, JudgeConfig):
                for jkey, jval in sorted(asdict(value).items()):
                    if not jkey.startswith("_"):
                        writer.writerow([f"judge_config.{jkey}", str(jval), type(jval).__name__])
            elif isinstance(value, SimilarityConfig):
                for skey, sval in sorted(asdict(value).items()):
                    if not skey.startswith("_"):
                        writer.writerow([f"similarity_config.{skey}", str(sval), type(sval).__name__])
            # Handle subclasses of VisionConfig/JudgeConfig/SimilarityConfig
            elif hasattr(value, "__class__") and (VisionConfig in value.__class__.__mro__ or JudgeConfig in value.__class__.__mro__ or SimilarityConfig in value.__class__.__mro__):
                if VisionConfig in value.__class__.__mro__:
                    prefix = "vision_config"
                elif JudgeConfig in value.__class__.__mro__:
                    prefix = "judge_config"
                else:
                    prefix = "similarity_config"
                for vkey, vval in sorted(asdict(value).items()):
                    if not vkey.startswith("_"):
                        writer.writerow([f"{prefix}.{vkey}", str(vval), type(vval).__name__])
            else:
                writer.writerow([key, str(value), type(value).__name__])
        
        return output.getvalue()

    @classmethod
    def from_file(cls: type[T], config_path: Path) -> T:
        """
        Load BenchmarkConfig from a TOML or JSON file.
        Tries TOML first, falls back to JSON.
        
        Args:
            config_path: Path to TOML or JSON config file
            
        Returns:
            BenchmarkConfig instance
            
        Raises:
            ValueError: If file format is not supported or file cannot be parsed
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        suffix = config_path.suffix.lower()
        
        # Try TOML first
        if suffix in (".toml", ".tml"):
            return cls.from_toml(config_path)
        
        # Fall back to JSON
        if suffix == ".json":
            return cls.from_json(config_path)        
        # Fall back to JSON
        return cls.from_json(config_path)

    @classmethod
    def from_toml(cls: type[T], toml_path: Path) -> T:
        """
        Load BenchmarkConfig from a TOML file.
        
        Args:
            toml_path: Path to TOML config file
            
        Returns:
            BenchmarkConfig instance
        """
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        
        return cls._from_dict(data)

    @classmethod
    def from_json(cls: type[T], json_path: Path) -> T:
        """
        Load BenchmarkConfig from a JSON file.
        
        Args:
            json_path: Path to JSON config file
            
        Returns:
            BenchmarkConfig instance
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """
        Create BenchmarkConfig from a dictionary.
        Handles nested vision_config, judge_config, and similarity_config structures.
        Automatically detects and uses adapter-specific config classes.
        
        Args:
            data: Dictionary of config values
            
        Returns:
            BenchmarkConfig instance
        """
        # Handle nested configs
        vision_config_data = data.pop("vision_config", {})
        judge_config_data = data.pop("judge_config", {})
        similarity_config_data = data.pop("similarity_config", {})
        
        # Determine which config classes to use based on adapter field
        vision_adapter = vision_config_data.get("adapter")
        judge_adapter = judge_config_data.get("adapter")
        similarity_adapter = similarity_config_data.get("adapter")
        
        vision_config_class = cls._get_vision_config_class(vision_adapter)
        judge_config_class = cls._get_judge_config_class(judge_adapter)
        similarity_config_class = cls._get_similarity_config_class(similarity_adapter)
        
        # Create nested config objects with appropriate classes
        vision_config = vision_config_class(**vision_config_data) if vision_config_data else vision_config_class()
        judge_config = judge_config_class(**judge_config_data) if judge_config_data else judge_config_class()
        similarity_config = similarity_config_class(**similarity_config_data) if similarity_config_data else similarity_config_class()
        
        # Filter remaining fields to only include valid field names
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        # Add nested configs
        filtered_data["vision_config"] = vision_config
        filtered_data["judge_config"] = judge_config
        filtered_data["similarity_config"] = similarity_config
        
        return cls(**filtered_data)
    
    @classmethod
    def _get_vision_config_class(cls, adapter_name: str) -> type[VisionConfig]:
        """
        Get the appropriate VisionConfig subclass for the given adapter.
        Uses the VisionAdapterRegistry to look up the config class.
        
        Args:
            adapter_name: Name of the adapter (e.g., "openai")
            
        Returns:
            VisionConfig subclass or VisionConfig if no specific class found
        """
        # Import here to avoid circular import
        from .vision import VisionAdapterRegistry
        
        # Discover and import all adapters to ensure they register themselves
        _discover_adapters()
        
        config_class = VisionAdapterRegistry.get_config_class(adapter_name)
        if config_class is None:
            raise ValueError(f"[{cls.__name__}] VisionConfig class not found for adapter '{adapter_name}'. Available adapters: {list(VisionAdapterRegistry.list_adapters())}")
        return config_class
    
    @classmethod
    def _get_judge_config_class(cls, adapter_name: str) -> type[JudgeConfig]:
        """
        Get the appropriate JudgeConfig subclass for the given adapter.
        Uses the JudgeAdapterRegistry to look up the config class.
        
        Args:
            adapter_name: Name of the adapter (e.g., "openai")
            
        Returns:
            JudgeConfig subclass or JudgeConfig if no specific class found
        """
        # Import here to avoid circular import
        from .judge import JudgeAdapterRegistry
        
        # Discover and import all adapters to ensure they register themselves
        _discover_adapters()
        
        config_class = JudgeAdapterRegistry.get_config_class(adapter_name)
        if config_class is None:
            raise ValueError(f"[{cls.__name__}] JudgeConfig class not found for adapter '{adapter_name}'. Available adapters: {list(JudgeAdapterRegistry.list_adapters())}")
        return config_class
    
    @classmethod
    def _get_similarity_config_class(cls, adapter_name: Optional[str]) -> type[SimilarityConfig]:
        """
        Get the appropriate SimilarityConfig subclass for the given adapter.
        Uses the SimilarityAdapterRegistry to look up the config class.
        
        Args:
            adapter_name: Name of the adapter (e.g., "local_clip")
            
        Returns:
            SimilarityConfig subclass or SimilarityConfig if no specific class found
        """
        # Import here to avoid circular import
        from .scoring import SimilarityAdapterRegistry
        
        # Discover and import all adapters to ensure they register themselves
        _discover_adapters()
        
        if adapter_name:
            config_class = SimilarityAdapterRegistry.get_config_class(adapter_name)
            if config_class is not None:
                return config_class
        return SimilarityConfig

    def to_toml(self, toml_path: Path) -> None:
        """
        Save BenchmarkConfig to a TOML file.
        Sensitive fields (starting with _) are excluded.
        
        Args:
            toml_path: Path to save TOML config file
            
        Raises:
            ValueError: If TOML writing support is not available
        """
        try:
            import tomli_w  # type: ignore
        except ImportError:
            raise ValueError(
                "TOML writing support not available. Install with: pip install tomli-w"
            )
        
        data = self._to_dict()
        
        toml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(toml_path, "wb") as f:
            tomli_w.dump(data, f)  # type: ignore

    def to_json(self, json_path: Path) -> None:
        """
        Save BenchmarkConfig to a JSON file.
        Sensitive fields (starting with _) are excluded.
        
        Args:
            json_path: Path to save JSON config file
        """
        data = self._to_dict()
        
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def _to_dict(self) -> dict[str, Any]:
        """
        Convert config to dictionary, excluding sensitive fields.
        Handles nested configs properly.
        
        Returns:
            Dictionary of config values (excluding fields starting with _)
        """
        data = {}
        for key, value in asdict(self).items():
            if not key.startswith("_"):  # Exclude sensitive fields
                # Convert nested dataclasses to dicts
                # Check if value is a VisionConfig, JudgeConfig, or SimilarityConfig (including subclasses)
                if hasattr(value, "__class__") and (
                    VisionConfig in value.__class__.__mro__ or JudgeConfig in value.__class__.__mro__ or SimilarityConfig in value.__class__.__mro__
                ):
                    # Convert nested config, excluding sensitive fields
                    nested_dict = {}
                    for nkey, nval in asdict(value).items():
                        if not nkey.startswith("_"):
                            nested_dict[nkey] = nval
                    data[key] = nested_dict
                else:
                    data[key] = value
        return data


DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()

