"""
_responses.py

Helpers for parsing OpenAI Responses API outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from ...framework.cost import CostSummary
from ...framework.io import read_jsonl

logger = logging.getLogger(__name__)


def extract_json_from_response_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract JSON output from a Responses API response body.
    
    Args:
        body: Response body dictionary from OpenAI API.
        
    Returns:
        Parsed JSON dictionary from the response output.
        
    Raises:
        RuntimeError: If JSON output cannot be located in the response body.
    """
    output_text = body.get("output_text")
    if output_text:
        return json.loads(output_text.strip())

    for item in body.get("output", []) or []:
        content_list = item.get("content", [])
        if not content_list and isinstance(item, dict):
            if item.get("type") == "output_text":
                text = item.get("text") or item.get("output_text")
                if text:
                    return json.loads(text.strip())

        for content in content_list:
            if content.get("type") == "output_text":
                text = content.get("text") or content.get("output_text")
                if text:
                    return json.loads(text.strip())

    raise RuntimeError(f"Could not locate JSON output in response body. Keys: {list(body.keys())}")


def extract_usage_from_response(response_body: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract usage/token information from an OpenAI response body.
    
    This function looks for usage data in various locations within the response
    structure, as OpenAI may include it in different formats. Also extracts cached
    tokens if available (for prompt caching discounts).
    
    Args:
        response_body: Response body dictionary from OpenAI API. This may be the
                       full response object or just the body portion.
        
    Returns:
        Dictionary with keys: input_tokens, cached_tokens, output_tokens, image_input_tokens, image_output_tokens.
        All values default to 0 if usage data is not found.
    """
    usage = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "image_input_tokens": 0,
        "image_output_tokens": 0,
    }
    
    def extract_from_usage_data(usage_data: Dict[str, Any]) -> None:
        """Helper to extract usage from a usage dict."""
        # Try input_tokens first, then prompt_tokens (OpenAI uses both)
        input_tokens = usage_data.get("input_tokens") or usage_data.get("prompt_tokens", 0) or 0
        usage["input_tokens"] = input_tokens
        usage["output_tokens"] = usage_data.get("output_tokens") or usage_data.get("completion_tokens", 0) or 0
        usage["image_input_tokens"] = usage_data.get("image_input_tokens", 0) or 0
        usage["image_output_tokens"] = usage_data.get("image_output_tokens", 0) or 0
        
        # Extract cached tokens if available (from prompt_tokens_details)
        prompt_tokens_details = usage_data.get("prompt_tokens_details")
        if isinstance(prompt_tokens_details, dict):
            usage["cached_tokens"] = prompt_tokens_details.get("cached_tokens", 0) or 0
    
    # Check for usage field directly in response body
    if "usage" in response_body:
        usage_data = response_body["usage"]
        if isinstance(usage_data, dict):
            extract_from_usage_data(usage_data)
            return usage
    
    # Check for usage in nested response object
    if "response" in response_body and isinstance(response_body["response"], dict):
        response_obj = response_body["response"]
        if "usage" in response_obj:
            usage_data = response_obj["usage"]
            if isinstance(usage_data, dict):
                extract_from_usage_data(usage_data)
                return usage
    
    # Check for usage in response.body (if response_body is the full response)
    if "body" in response_body and isinstance(response_body["body"], dict):
        body = response_body["body"]
        if "usage" in body:
            usage_data = body["usage"]
            if isinstance(usage_data, dict):
                extract_from_usage_data(usage_data)
                return usage
    
    return usage


def calculate_actual_costs(
    batch_output_jsonl: Path,
    extract_usage_fn: Callable[[Dict[str, Any]], Dict[str, int]],
    pricing: Dict[str, float],
    phase: str,
    num_items: Optional[int] = None,
) -> CostSummary:
    """
    Calculate actual costs from batch output JSONL file.
    
    Extracts usage data from each successful response and calculates total costs
    based on pricing.
    
    Args:
        batch_output_jsonl: Path to batch output JSONL file.
        extract_usage_fn: Function to extract usage from a response body.
        pricing: Dictionary with pricing per million tokens (keys: input_tokens, cached_input_tokens (optional),
                output_tokens, image_input_tokens, image_output_tokens).
        phase: Phase name ("vision" or "judge").
        num_items: Number of items processed. If None, will be counted from successful responses.
        
    Returns:
        CostSummary object with calculated costs.
    """
    if not batch_output_jsonl.exists():
        logger.warning(f"[COST] Batch output file not found: {batch_output_jsonl}")
        return CostSummary(phase=phase)
    
    # Aggregate usage from all responses
    total_input_tokens = 0
    total_cached_tokens = 0
    total_output_tokens = 0
    total_image_input_tokens = 0
    total_image_output_tokens = 0
    successful_count = 0
    
    for row in read_jsonl(batch_output_jsonl):
        # Skip failed requests
        if row.get("error"):
            continue
        
        # Extract usage from response
        response_body = row.get("response", {})
        if not response_body:
            continue
        
        usage = extract_usage_fn(response_body)
        total_input_tokens += usage["input_tokens"]
        total_cached_tokens += usage.get("cached_tokens", 0)
        total_output_tokens += usage["output_tokens"]
        total_image_input_tokens += usage["image_input_tokens"]
        total_image_output_tokens += usage["image_output_tokens"]
        successful_count += 1
    
    # Use provided num_items or count from successful responses
    if num_items is None:
        num_items = successful_count
    
    # Calculate costs
    # If cached_input_tokens pricing is provided, use it; otherwise treat all input tokens as uncached
    cached_input_price = pricing.get("cached_input_tokens")
    if cached_input_price is not None and total_cached_tokens > 0:
        # Separate pricing for cached vs uncached input tokens
        total_uncached_input_tokens = total_input_tokens - total_cached_tokens
        cost_uncached_input = (total_uncached_input_tokens / 1_000_000) * pricing["input_tokens"]
        cost_cached_input = (total_cached_tokens / 1_000_000) * cached_input_price
        cost_input = cost_uncached_input + cost_cached_input
    else:
        # No cached pricing or no cached tokens - use regular input pricing for all
        cost_input = (total_input_tokens / 1_000_000) * pricing["input_tokens"]
    
    cost_output = (total_output_tokens / 1_000_000) * pricing["output_tokens"]
    cost_image_input = (total_image_input_tokens / 1_000_000) * pricing.get("image_input_tokens", 0.0)
    cost_image_output = (total_image_output_tokens / 1_000_000) * pricing.get("image_output_tokens", 0.0)
    
    total_cost = cost_input + cost_output + cost_image_input + cost_image_output
    
    # Calculate per-item and per-token costs
    cost_per_item = total_cost / num_items if num_items > 0 else 0.0
    total_tokens = total_input_tokens + total_output_tokens + total_image_input_tokens + total_image_output_tokens
    cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0.0
    
    return CostSummary(
        phase=phase,
        total_input_tokens=total_input_tokens,
        total_cached_tokens=total_cached_tokens,
        total_output_tokens=total_output_tokens,
        total_image_input_tokens=total_image_input_tokens,
        total_image_output_tokens=total_image_output_tokens,
        total_cost=total_cost,
        num_items=num_items,
        cost_per_item=cost_per_item,
        cost_per_token=cost_per_token,
    )

