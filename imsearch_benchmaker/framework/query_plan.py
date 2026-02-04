"""
query_plan.py

Framework utilities for building query plans from annotations + seeds.
"""
from __future__ import annotations
import random
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from .io import read_jsonl, write_jsonl
from .config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG
from .query_plan_types import ImageRec, QueryPlanStrategy

logger = logging.getLogger(__name__)

def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Calculate Jaccard similarity coefficient between two sets.
    
    The Jaccard coefficient is the size of the intersection divided by the size of the union
    of the two sets. Returns 0.0 if both sets are empty.
    
    Args:
        a: First set of strings.
        b: Second set of strings.
        
    Returns:
        Jaccard similarity coefficient between 0.0 and 1.0.
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def majority_vote(values: List[Any], fallback: str = "unknown") -> Any:
    """
    Return the most common value from a list, with deterministic tie-breaking.
    
    Counts occurrences of each non-None value and returns the most common one.
    In case of ties, returns the lexicographically first value. Returns the fallback
    if all values are None or the list is empty.
    
    Args:
        values: List of values to find the majority of.
        fallback: Default value to return if no valid values are found.
        
    Returns:
        The most common value, or fallback if no valid values exist.
    """
    counts = Counter([v for v in values if v is not None])
    if not counts:
        return fallback
    most = counts.most_common()
    top_count = most[0][1]
    top_vals = sorted([v for v, n in most if n == top_count], key=lambda v: str(v))
    return top_vals[0] if top_vals else fallback

def load_annotations(path: Path, config: Optional[BenchmarkConfig] = None) -> Dict[str, ImageRec]:
    """
    Load annotations from JSONL file and convert to ImageRec objects.
    
    Reads annotation records from a JSONL file and extracts image IDs, tags, and facet
    values (taxonomy and boolean columns) into ImageRec objects for query planning.
    
    Args:
        path: Path to annotations JSONL file.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        
    Returns:
        Dictionary mapping image IDs to ImageRec objects.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    out: Dict[str, ImageRec] = {}
    for ann in read_jsonl(path):
        iid = ann[config.column_image_id]
        tags = set(ann.get(config.column_tags) or []) if config.column_tags else set()
        facets = {facet: ann.get(facet) for facet in (list(config.columns_taxonomy) + list(config.columns_boolean))}
        out[iid] = ImageRec(
            image_id=iid,
            facets=facets,
            tags=tags,
        )
    return out

def derive_query_profile(seed_recs: List[ImageRec], facet_keys: List[str]) -> Dict[str, Any]:
    """
    Derive a query profile from seed image records.
    
    Creates a profile representing the common characteristics of seed images by:
    - For boolean facets: Computing the majority boolean value (True if >= 50% are True)
    - For other facets: Using majority vote to determine the most common value
    - Aggregating all tags from seed images into a union set
    
    Args:
        seed_recs: List of ImageRec objects representing seed images for a query.
        facet_keys: List of facet column names to include in the profile.
        
    Returns:
        Dictionary with:
        - Keys from facet_keys mapped to their profile values
        - "seed_tags" key containing the union of all tags from seed images
    """
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
    """
    Check if an image matches the profile on all facets except one.
    
    Returns True if the image matches the profile on all keys except off_key,
    and differs from the profile on off_key. Used to find near-miss negative examples.
    
    Args:
        img: ImageRec to check against the profile.
        prof: Query profile dictionary with facet values.
        keys: List of facet keys to check.
        off_key: The facet key that is allowed to differ from the profile.
        
    Returns:
        True if image matches profile on all keys except off_key, False otherwise.
    """
    for k in keys:
        if k == off_key:
            continue
        if img.facets.get(k) != prof.get(k):
            return False
    return img.facets.get(off_key) != prof.get(off_key)

def score_candidates_by_tags(candidates: List[ImageRec], seed_tags: Set[str]) -> List[Tuple[float, ImageRec]]:
    """
    Score candidate images by tag overlap with seed tags.
    
    Calculates Jaccard similarity between each candidate's tags and the seed tags,
    then sorts by similarity (descending) with image_id as tie-breaker.
    
    Args:
        candidates: List of ImageRec objects to score.
        seed_tags: Set of tags from seed images for the query.
        
    Returns:
        List of (score, ImageRec) tuples sorted by score (descending), then by image_id.
    """
    scored = [(jaccard(c.tags, seed_tags), c) for c in candidates]
    scored.sort(key=lambda x: (-x[0], x[1].image_id))
    return scored

def diversity_key(img: ImageRec, diversity_facets: List[str]) -> Tuple[str, ...]:
    """
    Generate a diversity key for an image based on specified facets.
    
    Creates a tuple of facet values that can be used to group images for diversity
    sampling. Images with the same diversity key are considered similar for diversity purposes.
    
    Args:
        img: ImageRec to generate a key for.
        diversity_facets: List of facet column names to include in the diversity key.
        
    Returns:
        Tuple of string representations of facet values.
    """
    return tuple(str(img.facets.get(k, "")) for k in diversity_facets)

def pick_with_diversity(
    scored: List[Tuple[float, ImageRec]],
    k: int,
    already: Set[str],
    diversity_facets: List[str],
    max_per_divkey: int = 5,
) -> List[str]:
    """
    Select top-k images from scored candidates while ensuring diversity.
    
    Iterates through pre-scored candidates and selects images, ensuring that no more
    than max_per_divkey images share the same diversity key. Skips images already
    in the 'already' set and updates it with selected images.
    
    Args:
        scored: List of (score, ImageRec) tuples, pre-sorted by score (descending).
        k: Number of images to select.
        already: Set of image IDs to exclude (modified in-place).
        diversity_facets: List of facet names to use for diversity grouping.
        max_per_divkey: Maximum number of images to select per diversity key.
        
    Returns:
        List of selected image IDs.
    """
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

class TagOverlapQueryPlan(QueryPlanStrategy):
    """
    Query plan strategy based on tag overlap and facet matching.
    
    Selects negative candidate images for each query using a three-tier approach:
    - Hard negatives: Images matching core facets with high tag overlap (difficult to distinguish)
    - Near-miss negatives: Images matching most facets but differing on one (similar but wrong)
    - Easy negatives: Images with low tag overlap and different core facets (clearly different)
    
    Uses diversity constraints to ensure variety in selected candidates.
    """
    
    def __init__(
        self,
        neg_total: Optional[int] = None,
        neg_hard: Optional[int] = None,
        neg_nearmiss: Optional[int] = None,
        neg_easy: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize TagOverlapQueryPlan strategy.
        
        Args:
            neg_total: Total number of negative candidates per query.
            neg_hard: Number of hard negative candidates (matching core facets, high tag overlap).
            neg_nearmiss: Number of near-miss negative candidates (one facet off).
            neg_easy: Number of easy negative candidates (low tag overlap, different facets).
            random_seed: Random seed for reproducible candidate selection.
        """
        self.neg_total = neg_total
        self.neg_hard = neg_hard
        self.neg_nearmiss = neg_nearmiss
        self.neg_easy = neg_easy
        self.random_seed = random_seed

    def build(self, annotations: Dict[str, ImageRec], seeds_path: Path, config: BenchmarkConfig) -> List[Dict[str, Any]]:
        """
        Build query plan using tag overlap and facet matching strategy.
        
        For each query defined in seeds_path:
        1. Derives a profile from seed images (common facets and tags)
        2. Selects hard negatives: images matching core facets with high tag overlap
        3. Selects near-miss negatives: images matching most facets but differing on one
        4. Selects easy negatives: images with low tag overlap and different core facets
        5. Combines seed images and negatives into candidate list
        
        Uses diversity constraints to limit selections per diversity key and ensures
        no image is selected multiple times for the same query.
        
        Args:
            annotations: Dictionary mapping image IDs to ImageRec objects.
            seeds_path: Path to seeds JSONL file containing query definitions.
            config: BenchmarkConfig instance with query planning parameters.
            
        Returns:
            List of dictionaries, each with:
            - query_id: Query identifier
            - seed_image_ids: List of seed image IDs for the query
            - candidate_image_ids: List of candidate image IDs (seeds + negatives)
            
        Raises:
            ValueError: If negative counts are not properly configured or don't sum correctly.
        """
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
        failed_seeds: List[Tuple[str, str]] = []  # (query_id, seed_id)
        succeeded_seeds: List[Tuple[str, str]] = []  # (query_id, seed_id)
        total_queries = 0
        
        for q in read_jsonl(seeds_path):
            total_queries += 1
            query_id = q[config.column_query_id]
            seed_ids: List[str] = list(q[config.query_plan_seed_image_ids_column])
            
            # Filter out seed IDs that don't have annotations (failed annotations)
            valid_seed_ids = []
            for sid in seed_ids:
                if sid in annotations:
                    valid_seed_ids.append(sid)
                    succeeded_seeds.append((query_id, sid))
                else:
                    failed_seeds.append((query_id, sid))
                    logger.warning(f"[QUERY_PLAN] Skipping seed {sid} for query {query_id}: annotation not found (likely failed during vision phase)")
            
            # Skip this query if all seeds failed
            if not valid_seed_ids:
                logger.error(f"[QUERY_PLAN] Skipping query {query_id}: all {len(seed_ids)} seed(s) failed annotation")
                continue
            
            # If some seeds failed but we have at least one valid seed, continue with valid ones
            if len(valid_seed_ids) < len(seed_ids):
                logger.warning(f"[QUERY_PLAN] Query {query_id}: {len(valid_seed_ids)}/{len(seed_ids)} seed(s) available, proceeding with available seeds")
            
            seed_recs = [annotations[sid] for sid in valid_seed_ids]

            prof = derive_query_profile(seed_recs, core_keys)
            seed_tags: Set[str] = prof["seed_tags"]
            already: Set[str] = set(valid_seed_ids)

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

            candidate_ids = list(valid_seed_ids) + negatives
            rows_out.append({
                config.column_query_id: query_id,
                config.query_plan_seed_image_ids_column: valid_seed_ids,
                config.query_plan_candidate_image_ids_column: candidate_ids,
            })

        # Log summary of failed and succeeded seeds        
        logger.info(f"[QUERY_PLAN] Failed seeds: {len(failed_seeds)}")
        logger.info(f"[QUERY_PLAN] Succeeded seeds: {len(succeeded_seeds)}")
        logger.info(f"[QUERY_PLAN] Built {len(rows_out)} query plan(s) from {total_queries} seed query(s)")

        return rows_out

def build_query_plan(
    annotations: Dict[str, ImageRec],
    seeds_path: Path,
    strategy: QueryPlanStrategy,
    out_path: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Build query plan using the specified strategy and optionally write to file.
    
    Executes the query plan strategy to generate candidate image selections for each
    query defined in the seeds file. Optionally writes the results to a JSONL file.
    
    Args:
        annotations: Dictionary mapping image IDs to ImageRec objects from annotations.
        seeds_path: Path to seeds JSONL file containing query definitions with seed image IDs.
        strategy: QueryPlanStrategy instance to use for candidate selection.
        out_path: Optional path to write the query plan JSONL file. If None, results are not written.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        
    Returns:
        List of dictionaries, each representing a query plan row with query_id,
        seed_image_ids, and candidate_image_ids.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    rows = strategy.build(annotations, seeds_path, config)
    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows
