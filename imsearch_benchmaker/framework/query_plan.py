"""
query_plan.py

Framework utilities for building query plans from annotations + seeds.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .io import read_jsonl, write_jsonl
from .config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def majority_vote(values: List[Any], fallback: str = "unknown") -> Any:
    counts = Counter([v for v in values if v is not None])
    if not counts:
        return fallback
    most = counts.most_common()
    top_count = most[0][1]
    top_vals = sorted([v for v, n in most if n == top_count], key=lambda v: str(v))
    return top_vals[0] if top_vals else fallback


@dataclass(frozen=True)
class ImageRec:
    image_id: str
    facets: Dict[str, Any]
    tags: Set[str]


def load_annotations(path: Path, config: Optional[BenchmarkConfig] = None) -> Dict[str, ImageRec]:
    config = config or DEFAULT_BENCHMARK_CONFIG
    out: Dict[str, ImageRec] = {}
    for ann in read_jsonl(path):
        iid = ann[config.column_image_id]
        tags = set(ann.get(config.column_tags) or []) if config.column_tags else set()
        facets = {facet: ann.get(facet) for facet in config.columns_taxonomy}
        out[iid] = ImageRec(
            image_id=iid,
            facets=facets,
            tags=tags,
        )
    return out


def derive_query_profile(seed_recs: List[ImageRec], facet_keys: List[str]) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    for key in facet_keys:
        values = [rec.facets.get(key) for rec in seed_recs]
        if all(isinstance(v, bool) for v in values if v is not None):
            true_count = sum(1 for v in values if v is True)
            profile[key] = bool(round(true_count / max(1, len(values))))
        else:
            profile[key] = majority_vote(values)
    profile["seed_tags"] = set().union(*[r.tags for r in seed_recs]) if seed_recs else set()
    return profile


def one_facet_off(img: ImageRec, prof: Dict[str, Any], keys: List[str], off_key: str) -> bool:
    for k in keys:
        if k == off_key:
            continue
        if img.facets.get(k) != prof.get(k):
            return False
    return img.facets.get(off_key) != prof.get(off_key)


def score_candidates_by_tags(candidates: List[ImageRec], seed_tags: Set[str]) -> List[Tuple[float, ImageRec]]:
    scored = [(jaccard(c.tags, seed_tags), c) for c in candidates]
    scored.sort(key=lambda x: (-x[0], x[1].image_id))
    return scored


def diversity_key(img: ImageRec, diversity_facets: List[str]) -> Tuple[str, ...]:
    return tuple(str(img.facets.get(k, "")) for k in diversity_facets)


def pick_with_diversity(
    scored: List[Tuple[float, ImageRec]],
    k: int,
    already: Set[str],
    diversity_facets: List[str],
    max_per_divkey: int = 5,
) -> List[str]:
    picked: List[str] = []
    counts = Counter()

    for _, img in scored:
        if len(picked) >= k:
            break
        if img.image_id in already:
            continue
        dk = diversity_key(img, diversity_facets)
        if diversity_facets and counts[dk] >= max_per_divkey:
            continue
        picked.append(img.image_id)
        already.add(img.image_id)
        if diversity_facets:
            counts[dk] += 1

    return picked


class QueryPlanStrategy(ABC):
    @abstractmethod
    def build(self, annotations: Dict[str, ImageRec], seeds_path: Path, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """
        Build query plan rows from annotations + seeds.
        """


class TagOverlapQueryPlan(QueryPlanStrategy):
    def __init__(
        self,
        neg_total: Optional[int] = None,
        neg_hard: Optional[int] = None,
        neg_nearmiss: Optional[int] = None,
        neg_easy: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.neg_total = neg_total
        self.neg_hard = neg_hard
        self.neg_nearmiss = neg_nearmiss
        self.neg_easy = neg_easy
        self.random_seed = random_seed

    def build(self, annotations: Dict[str, ImageRec], seeds_path: Path, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        neg_total = self.neg_total if self.neg_total is not None else config.query_plan_neg_total
        neg_hard = self.neg_hard if self.neg_hard is not None else config.query_plan_neg_hard
        neg_nearmiss = self.neg_nearmiss if self.neg_nearmiss is not None else config.query_plan_neg_nearmiss
        neg_easy = self.neg_easy if self.neg_easy is not None else config.query_plan_neg_easy
        random_seed = self.random_seed if self.random_seed is not None else config.query_plan_random_seed

        if None in (neg_total, neg_hard, neg_nearmiss, neg_easy):
            raise ValueError("neg_total, neg_hard, neg_nearmiss, neg_easy must be set on strategy or config")
        if neg_hard + neg_nearmiss + neg_easy != neg_total:
            raise ValueError("hard + nearmiss + easy must equal total")

        rng = random.Random(random_seed or 42)
        all_ids = list(annotations.keys())

        core_keys = config.query_plan_core_facets
        off_keys = config.query_plan_off_facets or core_keys
        diversity_facets = config.query_plan_diversity_facets

        idx_core = defaultdict(list)
        if core_keys:
            for rec in annotations.values():
                idx_key = tuple(rec.facets.get(k) for k in core_keys)
                idx_core[idx_key].append(rec)

        rows_out: List[Dict[str, Any]] = []
        for q in read_jsonl(seeds_path):
            query_id = q[config.column_query_id]
            seed_ids: List[str] = list(q[config.query_plan_seed_image_ids_column])
            seed_recs = [annotations[sid] for sid in seed_ids]

            prof = derive_query_profile(seed_recs, core_keys)
            seed_tags: Set[str] = prof["seed_tags"]
            already: Set[str] = set(seed_ids)

            core_bucket = []
            if core_keys:
                core_bucket = idx_core[tuple(prof.get(k) for k in core_keys)]
            hard_scored = score_candidates_by_tags(core_bucket, seed_tags)
            hard_ids = pick_with_diversity(hard_scored, neg_hard, already, diversity_facets, max_per_divkey=6)

            if len(hard_ids) < neg_hard and core_keys:
                relax_sets: List[List[ImageRec]] = []
                for i in range(len(core_keys) - 1, 0, -1):
                    keys = core_keys[:i]
                    relax_sets.append([r for r in annotations.values() if all(r.facets.get(k) == prof.get(k) for k in keys)])

                for rs in relax_sets:
                    if len(hard_ids) >= neg_hard:
                        break
                    rs_scored = score_candidates_by_tags(rs, seed_tags)
                    hard_ids.extend(
                        pick_with_diversity(rs_scored, neg_hard - len(hard_ids), already, diversity_facets, max_per_divkey=6)
                    )

            near_candidates: List[ImageRec] = []
            if core_keys:
                for off in off_keys:
                    base = [r for r in annotations.values() if one_facet_off(r, prof, core_keys, off_key=off)]
                    near_candidates.extend(base)

            near_map = {r.image_id: r for r in near_candidates}
            near_scored = score_candidates_by_tags(list(near_map.values()), seed_tags)
            near_ids = pick_with_diversity(near_scored, neg_nearmiss, already, diversity_facets, max_per_divkey=5)

            if len(near_ids) < neg_nearmiss and core_keys:
                relax = [r for r in annotations.values() if r.facets.get(core_keys[0]) == prof.get(core_keys[0])]
                relax_scored = score_candidates_by_tags(relax, seed_tags)
                near_ids.extend(
                    pick_with_diversity(relax_scored, neg_nearmiss - len(near_ids), already, diversity_facets, max_per_divkey=5)
                )

            easy_pool = []
            for r in annotations.values():
                if r.image_id in already:
                    continue
                diff = 0
                for k in core_keys:
                    diff += 1 if r.facets.get(k) != prof.get(k) else 0
                if core_keys and diff >= max(2, len(core_keys) // 2) and jaccard(r.tags, seed_tags) <= 0.10:
                    easy_pool.append(r)

            rng.shuffle(easy_pool)
            easy_ids = []
            for r in easy_pool:
                if len(easy_ids) >= neg_easy:
                    break
                if r.image_id in already:
                    continue
                easy_ids.append(r.image_id)
                already.add(r.image_id)

            if len(easy_ids) < neg_easy:
                fallback = [iid for iid in all_ids if iid not in already]
                rng.shuffle(fallback)
                for iid in fallback:
                    if len(easy_ids) >= neg_easy:
                        break
                    easy_ids.append(iid)
                    already.add(iid)

            negatives = hard_ids[:neg_hard] + near_ids[:neg_nearmiss] + easy_ids[:neg_easy]
            if len(negatives) != neg_total:
                missing = neg_total - len(negatives)
                fallback = [iid for iid in all_ids if iid not in already]
                rng.shuffle(fallback)
                negatives.extend(fallback[:missing])

            candidate_ids = list(seed_ids) + negatives
            rows_out.append({
                config.column_query_id: query_id,
                config.query_plan_seed_image_ids_column: seed_ids,
                config.query_plan_candidate_image_ids_column: candidate_ids,
            })

        return rows_out


def build_query_plan(
    annotations: Dict[str, ImageRec],
    seeds_path: Path,
    strategy: QueryPlanStrategy,
    out_path: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
) -> List[Dict[str, Any]]:
    config = config or DEFAULT_BENCHMARK_CONFIG
    rows = strategy.build(annotations, seeds_path, config)
    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows
