"""
judge_types.py

Types for query generation and relevance judgments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class JudgeQuery:
    """
    Input payload for a judge adapter.
    """

    query_id: str
    seed_images: List[Dict[str, Any]] = field(default_factory=list)
    candidate_images: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class JudgeJudgment:
    image_id: str
    relevance_label: int


@dataclass(frozen=True)
class JudgeResult:
    query_id: str
    query_text: str
    judgments: List[JudgeJudgment] = field(default_factory=list)

