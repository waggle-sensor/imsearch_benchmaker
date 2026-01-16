"""
config.py

Benchmark configuration shared across framework utilities.
"""

from __future__ import annotations

import csv
import io
import os
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

# Try to import TOML support (tomllib in Python 3.11+, tomli for older versions)
try:
    import tomllib  # Python 3.11+
    HAS_TOML = True
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python
        HAS_TOML = True
    except ImportError:
        HAS_TOML = False

# TypeVar for generic return types in class methods
T = TypeVar("T", bound="BenchmarkConfig")


@dataclass(frozen=True)
class VisionConfig:
    """
    Configuration for vision annotation adapters.
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


@dataclass(frozen=True)
class JudgeConfig:
    """
    Configuration for judge adapters.
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


@dataclass(frozen=True)
class SimilarityConfig:
    """
    Configuration for similarity scoring adapters.
    """
    
    adapter: Optional[str] = None
    model: Optional[str] = None
    col_name: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Defines column names and categorical metadata for a benchmark.
    Sensitive keys should start with an underscore (_).
    """

    image_column: str = "image"
    image_id_column: str = "image_id"
    image_url_column: str = "image_url"
    mime_type_column: str = "mime_type"
    license_column: str = "license"
    doi_column: str = "doi"
    query_column: str = "query_text"
    query_id_column: str = "query_id"
    relevance_column: str = "relevance_label"
    seed_query_id_column: str = "query_id"
    seed_image_ids_column: str = "seed_image_ids"
    candidate_image_ids_column: str = "candidate_image_ids"

    tags_column: Optional[str] = "tags"
    summary_column: Optional[str] = "summary"
    confidence_column: Optional[str] = "confidence"

    taxonomy_columns: Dict[str, List[str]] = field(default_factory=dict)
    boolean_columns: List[str] = field(default_factory=list)
    metadata_columns: List[str] = field(default_factory=list) #TODO: do I need this?

    num_seeds: Optional[int] = None
    image_base_url: Optional[str] = None

    query_plan_neg_total: Optional[int] = None
    query_plan_neg_hard: Optional[int] = None
    query_plan_neg_nearmiss: Optional[int] = None
    query_plan_neg_easy: Optional[int] = None
    query_plan_random_seed: Optional[int] = None

    # File paths
    image_root_dir: Optional[str] = None  # Input directory for preprocessing
    meta_json: Optional[str] = None  # Metadata JSON file for preprocessing
    images_jsonl: Optional[str] = None  # Input images JSONL file
    seeds_jsonl: Optional[str] = None  # Input seeds JSONL file
    annotations_jsonl: Optional[str] = None  # Output annotations JSONL file
    query_plan_jsonl: Optional[str] = None  # Output query plan JSONL file
    qrels_jsonl: Optional[str] = None  # Output qrels JSONL file
    qrels_with_score_jsonl: Optional[str] = None  # Output qrels with similarity score JSONL file
    summary_output_dir: Optional[str] = None  # Output directory for dataset summary
    hf_dataset_dir: Optional[str] = None  # Output directory for Hugging Face dataset

    _hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    _hf_repo_id: Optional[str] = None
    _hf_private: Optional[bool] = None

    controlled_tag_vocab: List[str] = field(default_factory=list)

    vision_config: VisionConfig = field(default_factory=VisionConfig)
    judge_config: JudgeConfig = field(default_factory=JudgeConfig)
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)

    query_plan_core_facets: List[str] = field(default_factory=list)
    query_plan_off_facets: List[str] = field(default_factory=list)
    query_plan_diversity_facets: List[str] = field(default_factory=list)

    def required_qrels_columns(self) -> List[str]:
        return [self.query_column, self.query_id_column, self.relevance_column, self.image_id_column]

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
        if suffix in (".toml", ".tml") and HAS_TOML:
            return cls.from_toml(config_path)
        elif suffix in (".toml", ".tml") and not HAS_TOML:
            raise ValueError(
                "TOML file provided but tomllib/tomli not available. "
                "Install with: pip install tomli"
            )
        
        # Fall back to JSON
        if suffix == ".json":
            return cls.from_json(config_path)
        
        # Try to detect format by content
        try:
            if HAS_TOML:
                return cls.from_toml(config_path)
        except Exception:
            pass
        
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
        if not HAS_TOML:
            raise ValueError("TOML support not available. Install with: pip install tomli")
        
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

