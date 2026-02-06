"""
vision.py

OpenAI adapter for vision annotation using the Responses API.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Iterable, List
from pathlib import Path
import tempfile
import re

from .config import OpenAIVisionConfig
from .client import get_openai_client
from ...framework.vision import Vision
from ...framework.config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG
from ...framework.vision_types import VisionAnnotation, VisionImage
from ...framework.io import write_jsonl
from ...framework.cost import CostSummary
from ._responses import extract_json_from_response_body, extract_usage_from_response, calculate_actual_costs
from .batch import submit_batch, submit_batch_shards, shard_batch_jsonl, list_batches
import logging

logger = logging.getLogger(__name__)


class OpenAIVision(Vision):
    """
    OpenAI vision adapter for batch-friendly JSON schema outputs.
    """

    @staticmethod
    def build_json_schema(config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Build JSON schema for vision annotation based on config.
        
        Args:
            config: BenchmarkConfig instance
            
        Returns:
            JSON schema dictionary
        """
        properties: Dict[str, Any] = {}
        required: list[str] = []
        confidence_properties: Dict[str, Any] = {}
        confidence_required: list[str] = []
        
        # Always include summary
        properties["summary"] = {"type": "string"}
        required.append("summary")
        
        # Add taxonomy columns
        # These are categorical fields with enum values
        for col, values in config.columns_taxonomy.items():
            properties[col] = {"type": "string", "enum": values}
            required.append(col)
        
        # Add boolean columns
        # Each column in columns_boolean should be a separate property with boolean type
        for col in config.columns_boolean:
            properties[col] = {"type": "boolean"}
            required.append(col)
        
        # Add tags if controlled_tag_vocab is provided
        # Use column_tags as the key
        if config.vision_config.controlled_tag_vocab and config.column_tags:
            properties[config.column_tags] = {
                "type": "array",
                "minItems": config.vision_config.min_tags,
                "maxItems": config.vision_config.max_tags,
                "items": {"type": "string", "enum": config.vision_config.controlled_tag_vocab},
            }
            required.append(config.column_tags)
        
        # Add confidence object if we have taxonomy columns
        if config.columns_taxonomy:
            for col, values in config.columns_taxonomy.items():
                confidence_properties[col] = {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                }
                confidence_required.append(col)
        
        # Add confidence object if we have boolean columns
        if config.columns_boolean:
            for col in config.columns_boolean:
                confidence_properties[col] = {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                }
                confidence_required.append(col)

        # Add confidence object if we have confidence properties
        if confidence_properties:
            properties["confidence"] = {
                "type": "object",
                "additionalProperties": False,
                "properties": confidence_properties,
                "required": confidence_required,
            }
            required.append("confidence")

        schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }
        
        return schema

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        image_detail: str = "low",
        max_output_tokens: int = 4000,
        reasoning_effort: str = "low",
        metadata: Optional[Dict[str, str]] = None,
        config: Optional[BenchmarkConfig] = None,
        client: Any = None,
    ) -> None:
        config = config or DEFAULT_BENCHMARK_CONFIG
        
        # Initialize client if not provided
        if client is None:
            client = get_openai_client(openai_config=config.vision_config)
        
        # set self.config and self.client
        self.config = config
        self.client = client
        
        if not isinstance(config.vision_config, OpenAIVisionConfig):
            raise ValueError("OpenAI Vision adapter requires OpenAIVisionConfig in config.vision_config.")
        
        vision_cfg = config.vision_config
        self.model = model or vision_cfg.model
        if not self.model:
            raise ValueError("OpenAI Vision adapter requires a model name.")
        self.system_prompt = system_prompt or vision_cfg.system_prompt
        self.user_prompt = user_prompt or vision_cfg.user_prompt
        if not self.system_prompt or not self.user_prompt:
            raise ValueError("OpenAI Vision adapter requires system_prompt and user_prompt.")
        # Build schema automatically if not provided
        self.json_schema = json_schema or self.build_json_schema(config)
        self.image_detail = image_detail or vision_cfg.image_detail or "low"
        self.max_output_tokens = max_output_tokens or vision_cfg.max_output_tokens or 4000
        self.reasoning_effort = reasoning_effort or vision_cfg.reasoning_effort or "low"
        self.metadata = metadata or {}


    def build_request(self, image: VisionImage) -> Dict[str, object]:
        # Interpolate metadata into prompts using template placeholders
        # Support {metadata.column_name} format
        metadata_dict = image.metadata or {}
        
        def interpolate_prompt(prompt: str) -> str:
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
        
        system_prompt_interpolated = interpolate_prompt(self.system_prompt or "")
        user_prompt_interpolated = interpolate_prompt(self.user_prompt or "")
        
        body: Dict[str, object] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt_interpolated}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt_interpolated},
                        {
                            "type": "input_image",
                            "image_url": image.image_url,
                            "detail": self.image_detail,
                        },
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "imsearch_vision_annotation",
                    "strict": True,
                    "schema": self.json_schema,
                }
            },
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "metadata": {
                "image_id": image.image_id,
                **self.metadata,
                **(image.metadata or {}),
            },
        }
        return body

    def parse_response(self, response_body: Dict[str, object], image: VisionImage) -> VisionAnnotation:
        """
        Parse OpenAI response body into VisionAnnotation object.
        
        Extracts the JSON response and constructs a VisionAnnotation with:
        - Fields: All fields from the parsed response (excluding tags and confidence)
        - Tags: List of tags from the column_tags field
        - Confidence: Dictionary of confidence scores for taxonomy and boolean columns.
        
        Args:
            response_body: Response body dictionary from OpenAI API.
            image: VisionImage object for this annotation.
            
        Returns:
            VisionAnnotation object with parsed fields, tags, and confidence.
        """
        parsed = extract_json_from_response_body(response_body)
        fields = dict(parsed)
        
        # Extract tags using column_tags key
        tags = []
        if self.config.column_tags and self.config.column_tags in fields:
            tag_value = fields.pop(self.config.column_tags)
            if isinstance(tag_value, list):
                tags = tag_value
            elif isinstance(tag_value, str):
                tags = [tag_value]
        
        # Extract confidence - only keep taxonomy and boolean columns
        confidence = {}
        if "confidence" in fields:
            conf_dict = fields.pop("confidence")
            if isinstance(conf_dict, dict):
                for col in conf_dict.keys():
                    if col in self.config.columns_taxonomy or col in self.config.columns_boolean:
                        confidence[col] = conf_dict[col]
        
        return VisionAnnotation(
            image_id=image.image_id,
            fields=fields,
            tags=tags,
            confidence=confidence,
            metadata=image.metadata,
        )

    def submit(
        self,
        images: Iterable[VisionImage],
        out_jsonl: Optional[Path] = None,
        completion_window: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        max_items_per_shard: Optional[int] = None,
        shard_prefix: str = "vision_shard",
    ) -> object:
        if completion_window is None:
            if isinstance(self.config.vision_config, OpenAIVisionConfig):
                completion_window = self.config.vision_config.completion_window
        if max_items_per_shard is None:
            max_items_per_shard = self.config.vision_config.max_images_per_batch
        batch_lines = list(self.build_batch_lines(images))
        if out_jsonl is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
                out_jsonl = Path(tmp.name)
        write_jsonl(out_jsonl, batch_lines)

        # Ensure stage and purpose metadata are set
        if metadata is None:
            metadata = {}
        metadata = dict(metadata)  # Make a copy to avoid mutating the original
        metadata.setdefault("stage", self.config.vision_config.stage)
        metadata.setdefault("purpose", self.config.benchmark_name)
        
        if max_items_per_shard and len(batch_lines) > max_items_per_shard:
            shard_dir = out_jsonl.parent / f"{out_jsonl.stem}_shards"
            shard_paths = shard_batch_jsonl(out_jsonl, shard_dir, max_items_per_shard, shard_prefix)
            return submit_batch_shards(
                self.client,
                shard_paths,
                completion_window=completion_window,
                metadata=metadata,
            )

        return submit_batch(
            self.client,
            out_jsonl,
            completion_window=completion_window,
            metadata=metadata,
        )

    def get_client(self, config: Any = None) -> Any:
        """Get OpenAI client."""
        return self.client

    def wait_for_batch(self, batch_ref: Any) -> None:
        """Wait for OpenAI batch(es) to complete."""
        from .batch import wait_for_batches
        from ...framework.io import BatchRefs
        
        if isinstance(batch_ref, BatchRefs):
            batch_refs = [batch_ref]
        elif isinstance(batch_ref, list):
            batch_refs = batch_ref
        else:
            raise ValueError("Invalid batch reference type")
        
        # Wait for all batches in parallel using the shared wait_for_batches function
        wait_for_batches(self.client, batch_refs)

    def download_batch_results(
        self, batch_ref: Any, output_path: Path, error_path: Optional[Path] = None
    ) -> None:
        """Download OpenAI batch results."""
        from ...framework.io import BatchRefs
        from .client import download_file
        from ...framework.io import read_jsonl, write_jsonl
        
        batch_refs = [batch_ref] if isinstance(batch_ref, BatchRefs) else batch_ref
        if not isinstance(batch_refs, list):
            batch_refs = [batch_refs]
        
        all_output_rows = []
        all_error_rows = []
        
        for ref in batch_refs:
            if not isinstance(ref, BatchRefs):
                continue
            b = self.client.batches.retrieve(ref.batch_id)
            if b.output_file_id:
                temp_output = output_path.parent / f"temp_output_{ref.batch_id}.jsonl"
                download_file(self.client, b.output_file_id, temp_output)
                for row in read_jsonl(temp_output):
                    # Check if response has status "incomplete" - treat as error
                    response = row.get("response", {})
                    body = response.get("body", {}) if isinstance(response, dict) else {}
                    if isinstance(body, dict) and body.get("status") == "incomplete":
                        all_error_rows.append(row)
                    else:
                        all_output_rows.append(row)
                temp_output.unlink()
            if b.error_file_id and error_path:
                temp_error = error_path.parent / f"temp_error_{ref.batch_id}.jsonl"
                download_file(self.client, b.error_file_id, temp_error)
                for row in read_jsonl(temp_error):
                    all_error_rows.append(row)
                temp_error.unlink()
        
        write_jsonl(output_path, all_output_rows)
        if error_path and all_error_rows:
            write_jsonl(error_path, all_error_rows)

    def list_batches(self, active_only: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List OpenAI batches for vision stage.
        
        Args:
            active_only: If True, only return active batches
            limit: Maximum number of batches to return
            
        Returns:
            List of batch dictionaries with id, status, endpoint, created_at, metadata, and request_counts
            Only returns batches with stage="vision" in metadata
        """
        return list_batches(self.client, active_only=active_only, limit=limit, stage=self.config.vision_config.stage)

    def extract_usage_from_response(self, response_body: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract usage/token information from an OpenAI response body.
        
        This method delegates to the shared OpenAI implementation, which extracts
        cached tokens if available (for prompt caching discounts).
        
        Args:
            response_body: Response body dictionary from OpenAI API.
            
        Returns:
            Dictionary with keys: input_tokens, cached_tokens, output_tokens, 
            image_input_tokens, image_output_tokens.
        """
        return extract_usage_from_response(response_body)

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
        # Get pricing from config
        if not isinstance(self.config.vision_config, OpenAIVisionConfig):
            raise ValueError(
                "[COST] Vision config is not OpenAIVisionConfig. "
                "Cost calculation requires OpenAI-specific config with pricing information."
            )
        
        try:
            pricing = self.config.vision_config.get_effective_pricing()
        except ValueError as e:
            logger.error(f"[COST] {e}")
            raise
        
        return calculate_actual_costs(
            batch_output_jsonl=batch_output_jsonl,
            extract_usage_fn=self.extract_usage_from_response,
            pricing=pricing,
            phase="vision",
            num_items=num_items,
        )

    def get_name(self) -> str:
        return "openai_vision"
