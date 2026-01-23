"""
cost.py

Utilities for calculating and summarizing costs incurred during vision and judge processing stages.

Provides functions for determining actual usage-based costs, including breakdowns by item and by token type.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CostSummary:
    """
    Cost summary for a single phase (vision or judge).
    
    Attributes:
        phase: Phase name ("vision" or "judge").
        total_input_tokens: Total input text tokens used (includes both cached and uncached).
        total_cached_tokens: Total cached input text tokens used (subset of total_input_tokens).
        total_output_tokens: Total output text tokens used.
        total_image_input_tokens: Total image input tokens used (vision only).
        total_image_output_tokens: Total image output tokens used (vision only).
        total_cost: Total cost in USD.
        num_items: Number of items processed (images for vision, queries for judge).
        cost_per_item: Average cost per item.
        cost_per_token: Average cost per token (all types combined).
    """
    phase: str
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    total_image_input_tokens: int = 0
    total_image_output_tokens: int = 0
    total_cost: float = 0.0
    num_items: int = 0
    cost_per_item: float = 0.0
    cost_per_token: float = 0.0


def aggregate_cost_summaries(summaries: List[CostSummary]) -> CostSummary:
    """
    Aggregate multiple cost summaries into a single total summary.
    
    Args:
        summaries: List of CostSummary objects to aggregate.
        
    Returns:
        CostSummary object with aggregated totals.
    """
    if not summaries:
        return CostSummary(phase="total")
    
    total = CostSummary(phase="total")
    for summary in summaries:
        total.total_input_tokens += summary.total_input_tokens
        total.total_cached_tokens += summary.total_cached_tokens
        total.total_output_tokens += summary.total_output_tokens
        total.total_image_input_tokens += summary.total_image_input_tokens
        total.total_image_output_tokens += summary.total_image_output_tokens
        total.total_cost += summary.total_cost
        total.num_items += summary.num_items
    
    # Recalculate per-item and per-token costs
    total.cost_per_item = total.total_cost / total.num_items if total.num_items > 0 else 0.0
    total_tokens = total.total_input_tokens + total.total_output_tokens + total.total_image_input_tokens + total.total_image_output_tokens
    total.cost_per_token = total.total_cost / total_tokens if total_tokens > 0 else 0.0
    
    return total


def write_cost_summary_csv(summaries: List[CostSummary], output_path: Path) -> None:
    """
    Write cost summaries to a CSV file.
    
    Args:
        summaries: List of CostSummary objects to write.
        output_path: Path to write the CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Phase",
            "Total Input Tokens",
            "Total Cached Tokens",
            "Total Uncached Input Tokens",
            "Total Output Tokens",
            "Total Image Input Tokens",
            "Total Image Output Tokens",
            "Total Cost (USD)",
            "Num Items",
            "Cost Per Item (USD)",
            "Cost Per Token (USD)",
        ])
        
        # Write data rows
        for summary in summaries:
            total_uncached_input = summary.total_input_tokens - summary.total_cached_tokens
            writer.writerow([
                summary.phase,
                summary.total_input_tokens,
                summary.total_cached_tokens,
                total_uncached_input,
                summary.total_output_tokens,
                summary.total_image_input_tokens,
                summary.total_image_output_tokens,
                round(summary.total_cost, 2),
                summary.num_items,
                f"{summary.cost_per_item:.6f}",
                f"{summary.cost_per_token:.8f}",
            ])
    
    logger.info(f"Cost summary written to {output_path}")

