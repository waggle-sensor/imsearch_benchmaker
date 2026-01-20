"""
judge.py

OpenAI adapter for query generation and relevance judgments.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Iterable
from pathlib import Path
import tempfile
from .config import OpenAIJudgeConfig
from ...framework.judge import Judge, JudgeAdapterRegistry
from ...framework.config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG
from ...framework.judge_types import JudgeJudgment, JudgeQuery, JudgeResult
from ...framework.io import write_jsonl
from ._responses import extract_json_from_response_body
from .batch import submit_batch, submit_batch_shards, shard_batch_jsonl
from .client import get_openai_client


class OpenAIJudge(Judge):
    """
    OpenAI judge adapter for batch-friendly JSON schema outputs.
    """

    @staticmethod
    def build_json_schema(config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Build JSON schema for judge based on config.
        
        Args:
            config: BenchmarkConfig instance
            
        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                config.column_query: {"type": "string"},
                "judgments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            config.column_image_id: {"type": "string"},
                            config.column_relevance: {"type": "integer", "enum": [0, 1]},
                        },
                        "required": [config.column_image_id, config.column_relevance],
                    },
                },
            },
            "required": [config.column_query, "judgments"],
        }

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        max_output_tokens: int = 8000,
        reasoning_effort: str = "medium",
        metadata: Optional[Dict[str, str]] = None,
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        if not isinstance(self.config.judge_config, OpenAIJudgeConfig):
            raise ValueError("OpenAI Judge adapter requires OpenAIJudgeConfig in config.judge_config.")
        if not self.model:
            raise ValueError("OpenAI Judge adapter requires a model name.")
        if not self.system_prompt or not self.user_prompt:
            raise ValueError("OpenAI Judge adapter requires system_prompt and user_prompt.")
        config = config or DEFAULT_BENCHMARK_CONFIG
        self.config = config
        judge_cfg = config.judge_config
        self.model = model or judge_cfg.model
        self.system_prompt = system_prompt or judge_cfg.system_prompt
        self.user_prompt = user_prompt or judge_cfg.user_prompt
        # Build schema automatically if not provided
        self.json_schema = json_schema or self.build_json_schema(config)
        self.max_output_tokens = max_output_tokens or judge_cfg.max_output_tokens or 8000
        self.reasoning_effort = reasoning_effort or judge_cfg.reasoning_effort or "medium"
        self.metadata = metadata or {}

    def build_request(self, query: JudgeQuery) -> Dict[str, object]:
        payload = {
            "query_id": query.query_id,
            "seed_images": query.seed_images,
            "candidates": query.candidate_images,
        }
        return {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": self.user_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"DATA (JSON):\n{json.dumps(payload, ensure_ascii=False)}"}
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "imsearch_query_and_judgments",
                    "strict": True,
                    "schema": self.json_schema,
                }
            },
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "metadata": {
                "query_id": query.query_id,
                **self.metadata,
            },
        }

    def parse_response(self, response_body: Dict[str, object], query: JudgeQuery) -> JudgeResult:
        parsed = extract_json_from_response_body(response_body)
        judgments = []
        for item in parsed.get("judgments", []) or []:
            judgments.append(
                JudgeJudgment(
                    image_id=item["image_id"],
                    relevance_label=int(item["relevance_label"]),
                )
            )
        return JudgeResult(
            query_id=query.query_id,
            query_text=parsed.get("query_text", ""),
            judgments=judgments,
        )

    def submit(
        self,
        client: Any,
        queries: Iterable[JudgeQuery],
        out_jsonl: Optional[Path] = None,
        completion_window: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        max_items_per_shard: Optional[int] = None,
        shard_prefix: str = "judge_shard",
        max_concurrent: int = None,
    ) -> object:
        if completion_window is None:
            completion_window = self.config.judge_config.completion_window
        if max_items_per_shard is None:
            max_items_per_shard = self.config.judge_config.max_queries_per_batch
        if max_concurrent is None:
            config_max_concurrent = self.config.judge_config.max_concurrent_batches
            if config_max_concurrent:
                max_concurrent = config_max_concurrent
        if max_concurrent is None:
            max_concurrent = 1
        batch_lines = list(self.build_batch_lines(queries))
        if out_jsonl is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
                out_jsonl = Path(tmp.name)
        write_jsonl(out_jsonl, batch_lines)

        if max_items_per_shard and len(batch_lines) > max_items_per_shard:
            shard_dir = out_jsonl.parent / f"{out_jsonl.stem}_shards"
            shard_paths = shard_batch_jsonl(out_jsonl, shard_dir, max_items_per_shard, shard_prefix)
            return submit_batch_shards(
                client,
                shard_paths,
                completion_window=completion_window,
                metadata=metadata,
                max_concurrent=max_concurrent,
            )

        return submit_batch(
            client,
            out_jsonl,
            completion_window=completion_window,
            metadata=metadata,
        )

    def get_client(self) -> Any:
        """Get OpenAI client."""
        return get_openai_client(openai_config=self.config.judge_config)

    def wait_for_batch(self, client: Any, batch_ref: Any) -> None:
        """Wait for OpenAI batch to complete."""
        from .batch import BatchRefs, wait_for_batch
        if isinstance(batch_ref, BatchRefs):
            wait_for_batch(client, batch_ref.batch_id)
        elif isinstance(batch_ref, list):
            for ref in batch_ref:
                if isinstance(ref, BatchRefs):
                    wait_for_batch(client, ref.batch_id)

    def download_batch_results(
        self, client: Any, batch_ref: Any, output_path: Path, error_path: Optional[Path] = None
    ) -> None:
        """Download OpenAI batch results."""
        from .batch import BatchRefs
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
            b = client.batches.retrieve(ref.batch_id)
            if b.output_file_id:
                temp_output = output_path.parent / f"temp_output_{ref.batch_id}.jsonl"
                download_file(client, b.output_file_id, temp_output)
                for row in read_jsonl(temp_output):
                    all_output_rows.append(row)
                temp_output.unlink()
            if b.error_file_id and error_path:
                temp_error = error_path.parent / f"temp_error_{ref.batch_id}.jsonl"
                download_file(client, b.error_file_id, temp_error)
                for row in read_jsonl(temp_error):
                    all_error_rows.append(row)
                temp_error.unlink()
        
        write_jsonl(output_path, all_output_rows)
        if error_path and all_error_rows:
            write_jsonl(error_path, all_error_rows)

    def get_name(self) -> str:
        return "openai_judge"


JudgeAdapterRegistry.register("openai", OpenAIJudge, config_class=OpenAIJudgeConfig)

