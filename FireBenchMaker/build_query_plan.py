#!/usr/bin/env python3
"""
build_query_plan.py

Builds query_plan.jsonl for FireBench:
  {query_id, seed_image_ids, candidate_image_ids}

Candidate pool = seed images + similar images:
  - n hard negatives: same-facet, high tag overlap
  - n near-miss negatives: one-facet-off, high tag overlap
  - n easy negatives: low tag overlap

Inputs:
  --annotations annotations.jsonl  (from parse-vision; must include 'tags' + facets)
  --seeds seeds.jsonl              (lines: {"query_id":"...", "seed_image_ids":[...]} )
Output:
  --out query_plan.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
from config import NEG_TOTAL, NEG_HARD, NEG_NEARMISS, NEG_EASY, RANDOM_SEED
import logging

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)


# -----------------------------
# Assertions
# -----------------------------
assert NEG_HARD + NEG_NEARMISS + NEG_EASY == NEG_TOTAL, "hard+nearmiss+easy must equal total"

# -----------------------------
# Helpers
# -----------------------------
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Read a JSONL file and yield each line as a dictionary.
    Args:
        path: The path to the JSONL file.
    Returns:
        An iterable of dictionaries, one for each line in the file.
    """
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    Args:
        path: The path to the JSONL file.
        rows: An iterable of dictionaries to write to the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Calculate the Jaccard similarity between two sets of strings.
    Args:
        a: A set of strings.
        b: A set of strings.
    Returns:
        The Jaccard similarity between the two sets.
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    # If the union is empty, return 0.0 to avoid division by zero
    return inter / union if union else 0.0


def majority_vote(values: List[str], fallback: str = "unknown") -> str:
    """
    Calculate the majority vote from a list of strings.
    Args:
        values: A list of strings.
        fallback: The fallback value if the list is empty.
    Returns:
        The majority vote from the list.
    """
    c = Counter([v for v in values if v is not None])
    if not c:
        return fallback
    # If tie, take most common deterministically by sort
    most = c.most_common()
    top_count = most[0][1]
    top_vals = sorted([v for v, n in most if n == top_count])
    return top_vals[0] if top_vals else fallback


@dataclass(frozen=True)
class ImageRec:
    """
    A record of an image.
    Args:
        image_id: The ID of the image.
        viewpoint: The viewpoint of the image.
        plume_stage: The plume stage of the image.
        flame_visible: Whether the image shows a flame.
        lighting: The lighting of the image.
        confounder_type: The confounder type of the image.
        environment_type: The environment type of the image.
        tags: The tags of the image.
    """
    image_id: str
    viewpoint: str
    plume_stage: str
    flame_visible: bool
    lighting: str
    confounder_type: str
    environment_type: str
    tags: Set[str]


def load_annotations(path: Path) -> Dict[str, ImageRec]:
    """
    Load the annotations from a JSONL file.
    Args:
        path: The path to the JSONL file.
    Returns:
        A dictionary of image IDs to ImageRec objects.
    """
    out: Dict[str, ImageRec] = {}
    for a in read_jsonl(path):
        iid = a["image_id"]
        tags = set(a.get("tags") or [])
        out[iid] = ImageRec(
            image_id=iid,
            viewpoint=a.get("viewpoint", "unknown"),
            plume_stage=a.get("plume_stage", "unknown"),
            flame_visible=bool(a.get("flame_visible", False)),
            lighting=a.get("lighting", "unknown"),
            confounder_type=a.get("confounder_type", "unknown"),
            environment_type=a.get("environment_type", "unknown"),
            tags=tags,
        )
    return out


def derive_query_profile(seed_recs: List[ImageRec]) -> Dict[str, Any]:
    """
    Derive the query profile from a list of seed records.
    Args:
        seed_recs: A list of seed records.
    Returns:
        A dictionary of the query profile.
    """
    # Majority vote facets across seeds (if seeds disagree, we pick a stable winner)
    return {
        "viewpoint": majority_vote([r.viewpoint for r in seed_recs]),
        "plume_stage": majority_vote([r.plume_stage for r in seed_recs]),
        "flame_visible": bool(round(sum(1 if r.flame_visible else 0 for r in seed_recs) / max(1, len(seed_recs)))),
        "lighting": majority_vote([r.lighting for r in seed_recs]),
        "confounder_type": majority_vote([r.confounder_type for r in seed_recs]),
        "environment_type": majority_vote([r.environment_type for r in seed_recs]),
        # Tag union is best for retrieval-like behavior
        "seed_tags": set().union(*[r.tags for r in seed_recs]) if seed_recs else set(),
    }


def exact_facet_match(img: ImageRec, prof: Dict[str, Any], keys: List[str]) -> bool:
    """
    Check if an image matches the query profile exactly.
    Args:
        img: The image record.
        prof: The query profile.
        keys: The keys to check.
    Returns:
        True if the image matches the query profile exactly, False otherwise.
    """
    for k in keys:
        if getattr(img, k) != prof[k]:
            return False
    return True


def one_facet_off(img: ImageRec, prof: Dict[str, Any], keys: List[str], off_key: str) -> bool:
    """
    Check if an image matches the query profile with one facet off.
    Args:
        img: The image record.
        prof: The query profile.
        keys: The keys to check.
        off_key: The key to turn off.
    Returns:
        True if the image matches the query profile with one facet off, False otherwise.
    """
    # match all keys except off_key; require off_key to differ
    for k in keys:
        if k == off_key:
            continue
        if getattr(img, k) != prof[k]:
            return False
    return getattr(img, off_key) != prof[off_key]


def score_candidates_by_tags(
    candidates: List[ImageRec],
    seed_tags: Set[str],
) -> List[Tuple[float, ImageRec]]:
    """
    Score candidates by tags.
    Args:
        candidates: A list of image records.
        seed_tags: The tags of the seed images.
    Returns:
        A list of tuples, each containing a score and an image record.
    """
    scored = []
    for c in candidates:
        scored.append((jaccard(c.tags, seed_tags), c))
    # Sort descending by score; stable by image_id for determinism
    scored.sort(key=lambda x: (-x[0], x[1].image_id))
    return scored


def diversity_key(img: ImageRec) -> Tuple[str, str, str]:
    """
    Create a diversity key for an image.
    Args:
        img: The image record.
    Returns:
        A tuple of the environment type, confounder type, and lighting.
    """
    # Simple diversity key: env + confounder + lighting
    return (img.environment_type, img.confounder_type, img.lighting)


def pick_with_diversity(
    scored: List[Tuple[float, ImageRec]],
    k: int,
    already: Set[str],
    max_per_divkey: int = 5,
) -> List[str]:
    """
    Pick a list of image IDs with diversity.
    Args:
        scored: A list of tuples, each containing a score and an image record.
        k: The number of image IDs to pick.
        already: A set of image IDs that have already been picked.
        max_per_divkey: The maximum number of image IDs to pick per diversity key.
    Returns:
        A list of image IDs.
    """
    picked: List[str] = []
    counts = Counter()

    for s, img in scored:
        if len(picked) >= k:
            break
        if img.image_id in already:
            continue
        dk = diversity_key(img)
        if counts[dk] >= max_per_divkey:
            continue
        picked.append(img.image_id)
        already.add(img.image_id)
        counts[dk] += 1

    return picked


def build_query_plan(
    annotations: Dict[str, ImageRec],
    seeds_path: Path,
) -> List[Dict[str, Any]]:
    """
    Build a query plan.
    Args:
        annotations: A dictionary of image IDs to image records.
        seeds_path: The path to the seeds file.
    Returns:
        A list of dictionaries, each containing a query ID, seed image IDs, and candidate image IDs.
    """
    rng = random.Random(RANDOM_SEED)
    all_ids = list(annotations.keys())

    # Pre-index by core facets for speed
    # Key: (viewpoint, plume_stage, flame_visible, lighting)
    idx_core = defaultdict(list)
    for rec in annotations.values():
        idx_core[(rec.viewpoint, rec.plume_stage, rec.flame_visible, rec.lighting)].append(rec)

    rows_out: List[Dict[str, Any]] = []

    # Define which facets are “core” for hard negatives
    CORE_KEYS = ["viewpoint", "plume_stage", "flame_visible", "lighting"]
    # Near-miss facets to flip (we intentionally pick one off)
    OFF_KEYS = ["plume_stage", "flame_visible", "confounder_type", "lighting"]

    for q in read_jsonl(seeds_path):
        query_id = q["query_id"]
        seed_ids: List[str] = list(q["seed_image_ids"])
        seed_recs = [annotations[sid] for sid in seed_ids]

        prof = derive_query_profile(seed_recs)
        seed_tags: Set[str] = prof["seed_tags"]

        already: Set[str] = set(seed_ids)

        # --- Hard negatives: same core facets, high tag overlap ---
        core_bucket = idx_core[(prof["viewpoint"], prof["plume_stage"], prof["flame_visible"], prof["lighting"])]
        hard_scored = score_candidates_by_tags(core_bucket, seed_tags)
        hard_ids = pick_with_diversity(hard_scored, NEG_HARD, already, max_per_divkey=6)

        # If insufficient, relax progressively by dropping one core key at a time
        if len(hard_ids) < NEG_HARD:
            # Relax order: drop lighting, then flame_visible, then plume_stage
            relax_sets: List[List[ImageRec]] = []
            relax_sets.append([r for r in annotations.values() if r.viewpoint == prof["viewpoint"]
                               and r.plume_stage == prof["plume_stage"] and r.flame_visible == prof["flame_visible"]])
            relax_sets.append([r for r in annotations.values() if r.viewpoint == prof["viewpoint"]
                               and r.plume_stage == prof["plume_stage"]])
            relax_sets.append([r for r in annotations.values() if r.viewpoint == prof["viewpoint"]])

            for rs in relax_sets:
                if len(hard_ids) >= NEG_HARD:
                    break
                rs_scored = score_candidates_by_tags(rs, seed_tags)
                hard_ids.extend(pick_with_diversity(rs_scored, NEG_HARD - len(hard_ids), already, max_per_divkey=6))

        # --- Near-miss negatives: one-facet-off, high tag overlap ---
        near_candidates: List[ImageRec] = []
        # Make a single pooled list of one-off candidates across OFF_KEYS
        for off in OFF_KEYS:
            keys = CORE_KEYS.copy()
            # For confounder_type, use a slightly different “all-but” match:
            if off == "confounder_type":
                # Match viewpoint+plume_stage+lighting; keep flame_visible too
                base = [r for r in annotations.values()
                        if r.viewpoint == prof["viewpoint"]
                        and r.plume_stage == prof["plume_stage"]
                        and r.flame_visible == prof["flame_visible"]
                        and r.lighting == prof["lighting"]
                        and r.confounder_type != prof["confounder_type"]]
                near_candidates.extend(base)
            else:
                base = [r for r in annotations.values()
                        if one_facet_off(r, prof, CORE_KEYS, off_key=off)]
                near_candidates.extend(base)

        # Dedup near candidates, score by tag overlap
        near_map = {r.image_id: r for r in near_candidates}
        near_scored = score_candidates_by_tags(list(near_map.values()), seed_tags)
        near_ids = pick_with_diversity(near_scored, NEG_NEARMISS, already, max_per_divkey=5)

        # If insufficient, relax: just “same viewpoint + high tag overlap”
        if len(near_ids) < NEG_NEARMISS:
            relax = [r for r in annotations.values() if r.viewpoint == prof["viewpoint"]]
            relax_scored = score_candidates_by_tags(relax, seed_tags)
            near_ids.extend(pick_with_diversity(relax_scored, NEG_NEARMISS - len(near_ids), already, max_per_divkey=5))

        # --- Easy negatives: low tag overlap, different facets ---
        # Build a pool that differs on at least two core facets and has low Jaccard with seed tags
        easy_pool = []
        for r in annotations.values():
            if r.image_id in already:
                continue
            diff = 0
            diff += 1 if r.viewpoint != prof["viewpoint"] else 0
            diff += 1 if r.plume_stage != prof["plume_stage"] else 0
            diff += 1 if r.flame_visible != prof["flame_visible"] else 0
            diff += 1 if r.lighting != prof["lighting"] else 0
            if diff >= 2 and jaccard(r.tags, seed_tags) <= 0.10:
                easy_pool.append(r)

        rng.shuffle(easy_pool)
        easy_ids = []
        for r in easy_pool:
            if len(easy_ids) >= NEG_EASY:
                break
            if r.image_id in already:
                continue
            easy_ids.append(r.image_id)
            already.add(r.image_id)

        # If still short, fill from anywhere not already used
        if len(easy_ids) < NEG_EASY:
            fallback = [iid for iid in all_ids if iid not in already]
            rng.shuffle(fallback)
            for iid in fallback:
                if len(easy_ids) >= NEG_EASY:
                    break
                easy_ids.append(iid)
                already.add(iid)

        negatives = hard_ids[:NEG_HARD] + near_ids[:NEG_NEARMISS] + easy_ids[:NEG_EASY]
        if len(negatives) != NEG_TOTAL:
            # As a last resort, fill randomly
            missing = NEG_TOTAL - len(negatives)
            fallback = [iid for iid in all_ids if iid not in already]
            rng.shuffle(fallback)
            negatives.extend(fallback[:missing])

        candidate_ids = list(seed_ids) + negatives

        rows_out.append({
            "query_id": query_id,
            "seed_image_ids": seed_ids,
            "candidate_image_ids": candidate_ids,
        })

    return rows_out


def main() -> None:
    """
    Main function.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="annotations.jsonl from parse-vision (must include tags)")
    ap.add_argument("--seeds", required=True, help="seeds.jsonl (query_id + seed_image_ids)")
    ap.add_argument("--out", required=True, help="query_plan.jsonl output")
    args = ap.parse_args()

    # Load the annotations
    annotations = load_annotations(Path(args.annotations))
    # Build the query plan
    rows = build_query_plan(annotations, Path(args.seeds))
    # Write the query plan to a JSONL file
    write_jsonl(Path(args.out), rows)
    # Log the number of queries written
    logger.info(f"Wrote {len(rows)} queries to {args.out}")


if __name__ == "__main__":
    """
    Main function.
    """
    main()
