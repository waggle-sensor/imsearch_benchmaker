"""
cli.py

CLI entry point for imsearch_benchmaker.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG
from .preprocess import build_images_jsonl, build_seeds_jsonl, check_image_urls
from .query_plan import TagOverlapQueryPlan, build_query_plan, load_annotations
from .postprocess import calculate_similarity_score, generate_dataset_summary, huggingface
from .scoring import SimilarityAdapterRegistry
from .cost import (
    aggregate_cost_summaries,
    write_cost_summary_csv
)
from .io import (
    read_jsonl,
    write_jsonl,
    extract_failed_ids,
    save_batch_id,
    load_batch_id,
    format_batch_id,
    BatchRefs,
)
from .vision import Vision, VisionAdapterRegistry
from .judge import Judge, JudgeAdapterRegistry
from .vision_types import VisionImage, VisionAnnotation
from .judge_types import JudgeQuery, JudgeResult
from typing import Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Programmatic API
# -----------------------------


def run_preprocess(
    input_dir: Optional[Path] = None,
    out_images_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    out_seeds_jsonl: Optional[Path] = None,
    meta_json: Optional[Path] = None,
    default_license: Optional[str] = None,
    default_doi: Optional[str] = None,
    follow_symlinks: bool = False,
    limit: int = 0,
    num_seeds: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Preprocess images directory into images.jsonl and optionally seeds.jsonl.
    
    Args:
        input_dir: Input directory for images. If None, uses config.image_root_dir.
        out_images_jsonl: Output images JSONL path. If None, uses config.images_jsonl.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        out_seeds_jsonl: Output seeds JSONL path. If None, uses config.seeds_jsonl.
        meta_json: Metadata JSON file path. If None, uses config.meta_json.
        default_license: Default license string.
        default_doi: Default DOI string.
        follow_symlinks: Whether to follow symlinks.
        limit: Limit number of images to process (0 = no limit).
        num_seeds: Number of seeds. If None, uses config.num_seeds.
    
    Returns:
        List of image rows written to images.jsonl.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    # Get paths from config if not provided
    input_dir = Path(input_dir) if input_dir else (Path(config.image_root_dir) if config.image_root_dir else None)
    out_images_jsonl = Path(out_images_jsonl) if out_images_jsonl else (Path(config.images_jsonl) if config.images_jsonl else None)
    out_seeds_jsonl = Path(out_seeds_jsonl) if out_seeds_jsonl else (Path(config.seeds_jsonl) if config.seeds_jsonl else None)
    meta_json = Path(meta_json) if meta_json else (Path(config.meta_json) if config.meta_json else None)
    
    if input_dir is None:
        raise ValueError("input_dir must be provided or set in config.image_root_dir")
    if out_images_jsonl is None:
        raise ValueError("out_images_jsonl must be provided or set in config.images_jsonl")
    
    rows = build_images_jsonl(
        input_dir=input_dir,
        out_jsonl=out_images_jsonl,
        image_base_url=config.image_base_url,
        meta_json=meta_json,
        default_license=default_license,
        default_doi=default_doi,
        follow_symlinks=follow_symlinks,
        limit=limit,
    )
    
    if out_seeds_jsonl:
        num_seeds = num_seeds or config.query_plan_num_seeds
        build_seeds_jsonl(
            rows=rows,
            out_seeds_jsonl=out_seeds_jsonl,
            num_seeds=num_seeds,
            seed_prefix="query_",
        )
    
    return rows

def run_vision(
    images_jsonl: Optional[Path] = None,
    out_annotations_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    vision_adapter: Optional[Vision] = None,
    adapter_name: Optional[str] = None,
    batch_output_jsonl: Optional[Path] = None,
    batch_error_jsonl: Optional[Path] = None,
    wait_for_completion: bool = True,
) -> List[VisionAnnotation]:
    """
    Run vision annotation pipeline: build batch, submit, wait, download, parse.
    This function orchestrates the granular step-by-step functions.
    
    Args:
        images_jsonl: Input images JSONL path. If None, uses config.images_jsonl.
        out_annotations_jsonl: Output annotations JSONL path. If None, uses config.annotations_jsonl.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        vision_adapter: Optional pre-instantiated adapter. If None, will be created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.vision_config.adapter.
        batch_output_jsonl: Optional batch output JSONL path.
        batch_error_jsonl: Optional batch error JSONL path.
        wait_for_completion: If True, wait for batch completion and download results.
    
    Returns:
        List of VisionAnnotation objects.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG

    logger.info("=" * 80)
    logger.info("Starting vision pipeline")
    logger.info("=" * 80)
    
    # Step 1: Make batch input
    logger.info("\n" + "=" * 80)
    logger.info("Step a: Making vision batch input")
    logger.info("=" * 80)
    batch_input_jsonl = run_vision_make(
        images_jsonl=images_jsonl,
        config=config,
        vision_adapter=vision_adapter,
        adapter_name=adapter_name,
    )
    
    # Step 2: Submit batch
    logger.info("\n" + "=" * 80)
    logger.info("Step b: Submitting vision batch")
    logger.info("=" * 80)
    batch_id_file = batch_input_jsonl.parent / ".vision_batch_id"
    run_vision_submit(
        batch_input_jsonl=batch_input_jsonl,
        batch_id_file=batch_id_file,
        config=config,
        vision_adapter=vision_adapter,
        adapter_name=adapter_name,
    )
    
    if wait_for_completion:
        # Step 3: Wait for completion
        logger.info("\n" + "=" * 80)
        logger.info("Step c: Waiting for vision batch completion")
        logger.info("=" * 80)
        run_vision_wait(
            batch_id_file=batch_id_file,
            config=config,
            vision_adapter=vision_adapter,
            adapter_name=adapter_name,
        )
        
        # Step 4: Download results
        logger.info("\n" + "=" * 80)
        logger.info("Step d: Downloading vision batch results")
        logger.info("=" * 80)
        run_vision_download(
            batch_id_file=batch_id_file,
            batch_output_jsonl=batch_output_jsonl,
            batch_error_jsonl=batch_error_jsonl,
            config=config,
            vision_adapter=vision_adapter,
            adapter_name=adapter_name,
        )
    
    # Step 5: Parse results
    logger.info("\n" + "=" * 80)
    logger.info("Step e: Parsing vision batch results")
    logger.info("=" * 80)
    annotations = run_vision_parse(
        batch_output_jsonl=batch_output_jsonl,
        images_jsonl=images_jsonl,
        out_annotations_jsonl=out_annotations_jsonl,
        config=config,
        vision_adapter=vision_adapter,
        adapter_name=adapter_name,
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Vision pipeline complete!")
    logger.info("=" * 80)
    
    return annotations

def run_query_plan(
    annotations_jsonl: Optional[Path] = None,
    seeds_jsonl: Optional[Path] = None,
    out_query_plan_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    neg_total: Optional[int] = None,
    neg_hard: Optional[int] = None,
    neg_nearmiss: Optional[int] = None,
    neg_easy: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build query plan from annotations and seeds.
    
    Args:
        annotations_jsonl: Input annotations JSONL path. If None, uses config.annotations_jsonl.
        seeds_jsonl: Input seeds JSONL path. If None, uses config.seeds_jsonl.
        out_query_plan_jsonl: Output query plan JSONL path. If None, uses config.query_plan_jsonl.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        neg_total: Total negatives. If None, uses config.query_plan_neg_total.
        neg_hard: Hard negatives. If None, uses config.query_plan_neg_hard.
        neg_nearmiss: Nearmiss negatives. If None, uses config.query_plan_neg_nearmiss.
        neg_easy: Easy negatives. If None, uses config.query_plan_neg_easy.
        random_seed: Random seed. If None, uses config.query_plan_random_seed.
    
    Returns:
        List of query plan rows.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    # Get paths from config if not provided
    annotations_jsonl = Path(annotations_jsonl) if annotations_jsonl else (Path(config.annotations_jsonl) if config.annotations_jsonl else None)
    seeds_jsonl = Path(seeds_jsonl) if seeds_jsonl else (Path(config.seeds_jsonl) if config.seeds_jsonl else None)
    out_query_plan_jsonl = Path(out_query_plan_jsonl) if out_query_plan_jsonl else (Path(config.query_plan_jsonl) if config.query_plan_jsonl else None)
    
    if annotations_jsonl is None:
        raise ValueError("annotations_jsonl must be provided or set in config.annotations_jsonl")
    if seeds_jsonl is None:
        raise ValueError("seeds_jsonl must be provided or set in config.seeds_jsonl")
    if out_query_plan_jsonl is None:
        raise ValueError("out_query_plan_jsonl must be provided or set in config.query_plan_jsonl")
    
    annotations = load_annotations(annotations_jsonl, config)
    strategy = TagOverlapQueryPlan(
        neg_total=neg_total or config.query_plan_neg_total,
        neg_hard=neg_hard or config.query_plan_neg_hard,
        neg_nearmiss=neg_nearmiss or config.query_plan_neg_nearmiss,
        neg_easy=neg_easy or config.query_plan_neg_easy,
        random_seed=random_seed or config.query_plan_random_seed,
    )
    return build_query_plan(annotations, seeds_jsonl, strategy, out_query_plan_jsonl, config)

def run_judge(
    query_plan_jsonl: Optional[Path] = None,
    annotations_jsonl: Optional[Path] = None,
    out_qrels_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    judge_adapter: Optional[Judge] = None,
    adapter_name: Optional[str] = None,
    batch_output_jsonl: Optional[Path] = None,
    batch_error_jsonl: Optional[Path] = None,
    wait_for_completion: bool = True,
) -> List[JudgeResult]:
    """
    Run judge pipeline: build batch, submit, wait, download, parse.
    This function orchestrates the granular step-by-step functions.
    
    Args:
        query_plan_jsonl: Input query plan JSONL path. If None, uses config.query_plan_jsonl.
        annotations_jsonl: Input annotations JSONL path. If None, uses config.annotations_jsonl.
        out_qrels_jsonl: Output qrels JSONL path. If None, uses config.qrels_jsonl.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        judge_adapter: Optional pre-instantiated adapter. If None, will be created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.judge_config.adapter.
        batch_output_jsonl: Optional batch output JSONL path.
        batch_error_jsonl: Optional batch error JSONL path.
        wait_for_completion: If True, wait for batch completion and download results.
    
    Returns:
        List of JudgeResult objects.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG

    logger.info("=" * 80)
    logger.info("Starting judge pipeline")
    logger.info("=" * 80)
    
    # Step 1: Make batch input
    logger.info("\n" + "=" * 80)
    logger.info("Step a: Making judge batch input")
    logger.info("=" * 80)
    batch_input_jsonl = run_judge_make(
        query_plan_jsonl=query_plan_jsonl,
        annotations_jsonl=annotations_jsonl,
        config=config,
        judge_adapter=judge_adapter,
        adapter_name=adapter_name,
    )
    
    # Step 2: Submit batch
    logger.info("\n" + "=" * 80)
    logger.info("Step b: Submitting judge batch")
    logger.info("=" * 80)
    batch_id_file = batch_input_jsonl.parent / ".judge_batch_id"
    run_judge_submit(
        batch_input_jsonl=batch_input_jsonl,
        batch_id_file=batch_id_file,
        config=config,
        judge_adapter=judge_adapter,
        adapter_name=adapter_name,
    )
    
    if wait_for_completion:
        # Step 3: Wait for completion
        logger.info("\n" + "=" * 80)
        logger.info("Step c: Waiting for judge batch completion")
        logger.info("=" * 80)
        run_judge_wait(
            batch_id_file=batch_id_file,
            config=config,
            judge_adapter=judge_adapter,
            adapter_name=adapter_name,
        )
        
        # Step 4: Download results
        logger.info("\n" + "=" * 80)
        logger.info("Step d: Downloading judge batch results")
        logger.info("=" * 80)
        run_judge_download(
            batch_id_file=batch_id_file,
            batch_output_jsonl=batch_output_jsonl,
            batch_error_jsonl=batch_error_jsonl,
            config=config,
            judge_adapter=judge_adapter,
            adapter_name=adapter_name,
        )
    
    # Step 5: Parse results
    logger.info("\n" + "=" * 80)
    logger.info("Step e: Parsing judge batch results")
    logger.info("=" * 80)
    results = run_judge_parse(
        batch_output_jsonl=batch_output_jsonl,
        query_plan_jsonl=query_plan_jsonl,
        annotations_jsonl=annotations_jsonl,
        out_qrels_jsonl=out_qrels_jsonl,
        config=config,
        judge_adapter=judge_adapter,
        adapter_name=adapter_name,
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Judge pipeline complete!")
    logger.info("=" * 80)
    
    return results

def run_all(
    input_dir: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    vision_adapter: Optional[Vision] = None,
    judge_adapter: Optional[Judge] = None,
    vision_adapter_name: Optional[str] = None,
    judge_adapter_name: Optional[str] = None,
    wait_for_completion: bool = True,
    skip_preprocess: bool = False,
    skip_vision: bool = False,
    skip_query_plan: bool = False,
    skip_judge: bool = False,
    skip_similarity: bool = False,
    skip_postprocess: bool = False,
    skip_huggingface: bool = False,
) -> None:
    """
    Run all steps of the pipeline: preprocess, vision, query_plan, judge.
    This function orchestrates the complete benchmark creation pipeline.
    
    Args:
        input_dir: Input directory for images. If None, uses config.image_root_dir.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        vision_adapter: Optional pre-instantiated vision adapter.
        judge_adapter: Optional pre-instantiated judge adapter.
        vision_adapter_name: Vision adapter name to use. If None, uses config.vision_config.adapter.
        judge_adapter_name: Judge adapter name to use. If None, uses config.judge_config.adapter.
        wait_for_completion: If True, wait for batch completion in vision and judge steps.
        skip_preprocess: If True, skip the preprocess step.
        skip_vision: If True, skip the vision step.
        skip_query_plan: If True, skip the query plan step.
        skip_judge: If True, skip the judge step.
        skip_similarity: If True, skip the similarity score step.
        skip_postprocess: If True, skip the postprocess step.
        skip_huggingface: If True, skip the huggingface upload step.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    logger.info("=" * 80)
    logger.info("Starting complete benchmark pipeline")
    logger.info("=" * 80)
    
    # Step 1: Preprocess
    if not skip_preprocess:
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Preprocessing images")
        logger.info("=" * 80)
        run_preprocess(
            input_dir=input_dir,
            config=config,
        )
        logger.info("✓ Preprocessing complete")
    else:
        logger.info("[PIPELINE]Skipping preprocess step")
    
    # Step 2: Vision
    if not skip_vision:
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Vision annotation")
        logger.info("=" * 80)
        run_vision(
            config=config,
            vision_adapter=vision_adapter,
            adapter_name=vision_adapter_name,
            wait_for_completion=wait_for_completion,
        )
        logger.info("✓ Vision annotation complete")
    else:
        logger.info("[PIPELINE] Skipping vision step")
    
    # Step 3: Query Plan
    if not skip_query_plan:
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Building query plan")
        logger.info("=" * 80)
        run_query_plan(
            config=config,
        )
        logger.info("✓ Query plan complete")
    else:
        logger.info("[PIPELINE] Skipping query plan step")
    
    # Step 4: Judge
    if not skip_judge:
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Judge relevance")
        logger.info("=" * 80)
        run_judge(
            config=config,
            judge_adapter=judge_adapter,
            adapter_name=judge_adapter_name,
            wait_for_completion=wait_for_completion,
        )
        logger.info("✓ Judge relevance complete")
    else:
        logger.info("[PIPELINE] Skipping judge step") 
    
    # Step 5: Similarity Score
    if not skip_similarity:
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Calculating similarity score")
        logger.info("=" * 80)
        # Get adapter name from config
        adapter_name = config.similarity_config.adapter
        if not adapter_name:
            logger.warning("[PIPELINE] No similarity adapter available, skipping similarity score calculation")
            skip_similarity = True
        
        if not skip_similarity:
            # Get column name from config
            col_name = config.similarity_config.col_name or "similarity_score"
            
            calculate_similarity_score(
                qrels_path=None,  # Use config.qrels_jsonl
                output_path=None,  # Use config.qrels_with_score_jsonl
                col_name=col_name,
                adapter_name=adapter_name,
                images_jsonl_path=None,  # Use config.images_jsonl
                config=config,
            )
            logger.info("✓ Similarity score complete")
    else:
        logger.info("[PIPELINE] Skipping similarity score step")
    
    # Step 6: Dataset Summary
    if not skip_postprocess:
        logger.info("\n" + "=" * 80)
        logger.info("Step 6: Generating dataset summary")
        logger.info("=" * 80)
        generate_dataset_summary(
            qrels_path=None,  # Use config.qrels_with_score_jsonl or config.qrels_jsonl
            output_dir=None,  # Use config.summary_output_dir
            images_jsonl_path=None,  # Use config.images_jsonl
            config=config,
        )
        logger.info("✓ Dataset summary complete")
    else:
        logger.info("[PIPELINE] Skipping dataset summary step")

    # Step 7: Hugging Face Upload
    if not skip_huggingface:
        logger.info("\n" + "=" * 80)
        logger.info("Step 7: Uploading dataset to Hugging Face")
        logger.info("=" * 80)
        huggingface(
            qrels_path=None,  # Use config.qrels_with_score_jsonl or config.qrels_jsonl
            output_dir=None,  # Use config.hf_dataset_dir
            images_jsonl_path=None,  # Use config.images_jsonl
            image_root_dir=None,  # Use config.image_root_dir
            progress_interval=100,
            repo_id=None,  # Use config._hf_repo_id
            token=None,  # Use config._hf_token
            private=None,  # Use config._hf_private
            config=config,
        )
        logger.info("✓ Hugging Face upload complete")
    else:
        logger.info("[PIPELINE] Skipping huggingface upload step")
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline complete!")
    logger.info("=" * 80)

# -----------------------------
# Granular Step-by-Step Functions
# -----------------------------


def run_vision_make(
    images_jsonl: Optional[Path] = None,
    out_batch_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    vision_adapter: Optional[Vision] = None,
    adapter_name: Optional[str] = None,
) -> Path:
    """
    Create vision batch input JSONL file from images.
    
    Reads images from images.jsonl, converts them to VisionImage objects with metadata,
    and writes batch input lines using the vision adapter's build_batch_lines method.
    
    Args:
        images_jsonl: Input images JSONL path. If None, uses config.images_jsonl.
        out_batch_jsonl: Output batch input JSONL path. If None, auto-generated.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        vision_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.vision_config.adapter.
        
    Returns:
        Path to the created batch input JSONL file.
        
    Raises:
        ValueError: If images_jsonl is not provided and not in config.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    images_jsonl = Path(images_jsonl) if images_jsonl else (Path(config.images_jsonl) if config.images_jsonl else None)
    if images_jsonl is None:
        raise ValueError("images_jsonl must be provided or set in config.images_jsonl")
    
    if adapter_name is None:
        adapter_name = config.vision_config.adapter
    if vision_adapter is None:
        vision_adapter = VisionAdapterRegistry.get(adapter_name, config=config)
    
    images = []
    for row in read_jsonl(images_jsonl):
        metadata = {}
        if config.column_license in row:
            metadata[config.column_license] = row[config.column_license]
        if config.column_doi in row:
            metadata[config.column_doi] = row[config.column_doi]
        images.append(VisionImage(
            image_id=row[config.column_image_id],
            image_url=row[config.image_url_temp_column],
            metadata=metadata,
        ))
    
    out_batch_jsonl = out_batch_jsonl or Path(config.annotations_jsonl).parent / "vision_batch_input.jsonl" if config.annotations_jsonl else Path("vision_batch_input.jsonl")
    out_batch_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    # Build batch lines and write
    batch_lines = list(vision_adapter.build_batch_lines(images))
    write_jsonl(out_batch_jsonl, batch_lines)
    
    file_size_mb = out_batch_jsonl.stat().st_size / (1024 * 1024)
    logger.info(f"[VISION] Created vision batch input: {out_batch_jsonl} ({file_size_mb:.2f} MB, {len(batch_lines)} requests)")
    
    return out_batch_jsonl


def run_vision_submit(
    batch_input_jsonl: Optional[Path] = None,
    batch_id_file: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    vision_adapter: Optional[Vision] = None,
    adapter_name: Optional[str] = None,
) -> str:
    """
    Submit vision batch to the vision adapter and save batch ID(s) to file.
    
    Reads batch input JSONL, submits it using the vision adapter, and saves
    the batch ID(s) to a file for later retrieval.
    
    Args:
        batch_input_jsonl: Path to batch input JSONL file. If None, auto-generated from config.
        batch_id_file: Path to save batch ID(s). If None, auto-generated.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        vision_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.vision_config.adapter.
        
    Returns:
        Comma-separated string of batch ID(s).
        
    Raises:
        ValueError: If batch_input_jsonl is not provided and not in config.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if batch_input_jsonl is None:
        if config.annotations_jsonl:
            batch_input_jsonl = Path(config.annotations_jsonl).parent / "vision_batch_input.jsonl"
        else:
            raise ValueError("batch_input_jsonl must be provided or config.annotations_jsonl must be set")
    batch_input_jsonl = Path(batch_input_jsonl)
    
    if adapter_name is None:
        adapter_name = config.vision_config.adapter
    if vision_adapter is None:
        vision_adapter = VisionAdapterRegistry.get(adapter_name, config=config)
    
    # Load images to submit
    images = []
    for row in read_jsonl(batch_input_jsonl):
        # Extract from batch line format
        custom_id = row.get("custom_id", "")
        if custom_id.startswith(f"{config.vision_config.stage}::"):
            image_id = custom_id.split(f"{config.vision_config.stage}::", 1)[1]
            # Reconstruct VisionImage from batch line
            content = row.get("body", {}).get("input", [])
            image_url = None
            for item in content:
                if isinstance(item, dict) and item.get("content"):
                    for c in item["content"]:
                        if isinstance(c, dict) and c.get("type") == "input_image":
                            image_url = c.get("image_url")
                            break
            if image_url:
                images.append(VisionImage(image_id=image_id, image_url=image_url, metadata={}))
    
    batch_ref = vision_adapter.submit(
        images=images,
        out_jsonl=batch_input_jsonl,
    )
    
    batch_id_str = format_batch_id(batch_ref)
    if batch_id_file is None:
        batch_id_file = batch_input_jsonl.parent / ".vision_batch_id"
    save_batch_id(batch_id_str, batch_id_file)
    
    logger.info(f"[VISION] Submitted vision batch: {batch_id_str}")
    return batch_id_str


def run_vision_wait(
    batch_id: Optional[str] = None,
    batch_id_file: Optional[Path] = None,
    poll_s: int = 60,
    config: Optional[BenchmarkConfig] = None,
    vision_adapter: Optional[Vision] = None,
    adapter_name: Optional[str] = None,
) -> None:
    """
    Wait for vision batch(es) to complete.
    
    Loads batch ID(s) from file or uses provided batch_id, then waits for
    all batches to complete using the vision adapter's wait_for_batch method.
    
    Args:
        batch_id: Comma-separated batch ID(s). If None, loaded from batch_id_file.
        batch_id_file: Path to batch ID file. If None, auto-generated from config.
        poll_s: Polling interval in seconds (used by adapter if supported).
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        vision_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.vision_config.adapter.
        
    Raises:
        ValueError: If batch_id and batch_id_file are both None.
        FileNotFoundError: If batch_id_file is specified but doesn't exist.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if batch_id is None:
        if batch_id_file is None:
            if config.annotations_jsonl:
                batch_id_file = Path(config.annotations_jsonl).parent / ".vision_batch_id"
            else:
                raise ValueError("batch_id or batch_id_file must be provided")
        batch_id = load_batch_id(batch_id_file)
    
    if adapter_name is None:
        adapter_name = config.vision_config.adapter
    if vision_adapter is None:
        vision_adapter = VisionAdapterRegistry.get(adapter_name, config=config)
    
    # Parse comma-separated batch IDs
    batch_ids = [bid.strip() for bid in batch_id.split(",") if bid.strip()]
    
    
    # Create BatchRefs for each batch ID
    batch_refs = [BatchRefs(input_file_id="", batch_id=bid) for bid in batch_ids]
    batch_ref = batch_refs[0] if len(batch_refs) == 1 else batch_refs
    
    # Wait for batches using adapter method
    vision_adapter.wait_for_batch(batch_ref)
    
    logger.info(f"[VISION] All vision batches completed: {', '.join(batch_ids)}")


def run_vision_download(
    batch_id: Optional[str] = None,
    batch_id_file: Optional[Path] = None,
    batch_output_jsonl: Optional[Path] = None,
    batch_error_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    vision_adapter: Optional[Vision] = None,
    adapter_name: Optional[str] = None,
) -> None:
    """
    Download vision batch results from the provider.
    
    Loads batch ID(s) from file or uses provided batch_id, then downloads
    batch results and errors using the vision adapter's download_batch_results method.
    
    Args:
        batch_id: Comma-separated batch ID(s). If None, loaded from batch_id_file.
        batch_id_file: Path to batch ID file. If None, auto-generated from config.
        batch_output_jsonl: Path to save batch output. If None, auto-generated.
        batch_error_jsonl: Path to save batch errors. If None, auto-generated.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        vision_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.vision_config.adapter.
        
    Raises:
        ValueError: If batch_id and batch_id_file are both None.
        FileNotFoundError: If batch_id_file is specified but doesn't exist.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if batch_id is None:
        if batch_id_file is None:
            if config.annotations_jsonl:
                batch_id_file = Path(config.annotations_jsonl).parent / ".vision_batch_id"
            else:
                raise ValueError("batch_id or batch_id_file must be provided")
        batch_id = load_batch_id(batch_id_file)
    
    if adapter_name is None:
        adapter_name = config.vision_config.adapter
    if vision_adapter is None:
        vision_adapter = VisionAdapterRegistry.get(adapter_name, config=config)
    
    if batch_output_jsonl is None:
        if config.annotations_jsonl:
            batch_output_jsonl = Path(config.annotations_jsonl).parent / "vision_batch_output.jsonl"
        else:
            raise ValueError("batch_output_jsonl must be provided or config.annotations_jsonl must be set")
    if batch_error_jsonl is None:
        batch_error_jsonl = Path(batch_output_jsonl).parent / "vision_batch_error.jsonl"
    
    # Parse comma-separated batch IDs and create BatchRefs
    batch_ids = [bid.strip() for bid in batch_id.split(",") if bid.strip()]
    
    batch_refs = [BatchRefs(input_file_id="", batch_id=bid) for bid in batch_ids]
    batch_ref = batch_refs[0] if len(batch_refs) == 1 else batch_refs
    
    vision_adapter.download_batch_results(
        batch_ref=batch_ref,
        output_path=batch_output_jsonl,
        error_path=batch_error_jsonl,
    )
    
    logger.info(f"[VISION] Downloaded vision batch results to {batch_output_jsonl}")


def run_vision_parse(
    batch_output_jsonl: Optional[Path] = None,
    images_jsonl: Optional[Path] = None,
    out_annotations_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    vision_adapter: Optional[Vision] = None,
    adapter_name: Optional[str] = None,
) -> List[VisionAnnotation]:
    """
    Parse vision batch output into annotations JSONL format.
    
    Reads batch output JSONL, parses each response using the vision adapter's
    parse_response method, and writes the results to annotations JSONL. Failed
    requests are logged and written to a separate failed JSONL file.
    
    Args:
        batch_output_jsonl: Path to batch output JSONL. If None, auto-generated from config.
        images_jsonl: Path to input images JSONL. If None, uses config.images_jsonl.
        out_annotations_jsonl: Path to output annotations JSONL. If None, uses config.annotations_jsonl.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        vision_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.vision_config.adapter.
        
    Returns:
        List of VisionAnnotation objects successfully parsed.
        
    Raises:
        ValueError: If required paths are not provided and not in config.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if batch_output_jsonl is None:
        if config.annotations_jsonl:
            batch_output_jsonl = Path(config.annotations_jsonl).parent / "vision_batch_output.jsonl"
        else:
            raise ValueError("batch_output_jsonl must be provided or config.annotations_jsonl must be set")
    if images_jsonl is None:
        images_jsonl = Path(config.images_jsonl) if config.images_jsonl else None
    if images_jsonl is None:
        raise ValueError("images_jsonl must be provided or set in config.images_jsonl")
    if out_annotations_jsonl is None:
        out_annotations_jsonl = Path(config.annotations_jsonl) if config.annotations_jsonl else None
    if out_annotations_jsonl is None:
        raise ValueError("out_annotations_jsonl must be provided or set in config.annotations_jsonl")
    
    if adapter_name is None:
        adapter_name = config.vision_config.adapter
    if vision_adapter is None:
        vision_adapter = VisionAdapterRegistry.get(adapter_name, config=config)
    
    # Load images
    images = []
    for row in read_jsonl(images_jsonl):
        metadata = {}
        if config.column_license in row:
            metadata[config.column_license] = row[config.column_license]
        if config.column_doi in row:
            metadata[config.column_doi] = row[config.column_doi]
        images.append(VisionImage(
            image_id=row[config.column_image_id],
            image_url=row[config.image_url_temp_column],
            metadata=metadata,
        ))
    
    # Parse results
    annotations = []
    failed_rows = []
    if Path(batch_output_jsonl).exists():
        for row in read_jsonl(batch_output_jsonl):
            custom_id = row.get("custom_id", "")
            if not custom_id.startswith(f"{config.vision_config.stage}::"):
                continue
            image_id = custom_id.split(f"{config.vision_config.stage}::", 1)[1]
            error = row.get("error")
            if error:
                logger.warning(f"[VISION] Vision request failed for {image_id}: {error}")
                failed_rows.append({
                    "image_id": image_id,
                    "custom_id": custom_id,
                    "error": error,
                })
                continue
            
            body = row.get("response", {}).get("body", {})
            image = next((img for img in images if img.image_id == image_id), None)
            if image:
                try:
                    ann = vision_adapter.parse_response(body, image)
                    annotations.append(ann)
                except Exception as e:
                    logger.error(f"[VISION] Failed to parse vision result for {image_id}: {e}")
                    failed_rows.append({
                        "image_id": image_id,
                        "custom_id": custom_id,
                        "error": {"message": str(e), "type": "parse_error"},
                    })
    
    # Write failed rows
    if failed_rows:
        failed_path = Path(out_annotations_jsonl).parent / f"{Path(out_annotations_jsonl).stem}_failed.jsonl"
        write_jsonl(failed_path, failed_rows)
        logger.warning(f"[VISION] Wrote {len(failed_rows)} failed vision requests to {failed_path}")
    
    # Write annotations JSONL
    rows_out = []
    for ann in annotations:
        row = {
            config.column_image_id: ann.image_id,
            **ann.fields,
        }
        if ann.tags:
            row[config.column_tags or "tags"] = ann.tags
        if ann.confidence:
            row[config.column_confidence or "confidence"] = ann.confidence
        if ann.metadata:
            row.update(ann.metadata)
        rows_out.append(row)
    
    write_jsonl(out_annotations_jsonl, rows_out)
    logger.info(f"[VISION] Parsed {len(annotations)} successful vision annotations, {len(failed_rows)} failed")
    
    # Calculate actual costs from batch output
    try:
        actual_costs = vision_adapter.calculate_actual_costs(
            batch_output_jsonl=batch_output_jsonl,
            num_items=len(annotations),
        )
        logger.info(f"[COST] Vision actual costs: ${actual_costs.total_cost:.2f} for {actual_costs.num_items} images "
                   f"(${actual_costs.cost_per_item:.4f} per image, ${actual_costs.cost_per_token:.8f} per token)")
        # Store cost summary for later aggregation (we'll save it in run_cost_summary)
        # For now, just log it
    except Exception as e:
        logger.warning(f"[COST] Failed to calculate vision costs: {e}")
    
    return annotations


# Similar granular functions for judge
def run_judge_make(
    query_plan_jsonl: Optional[Path] = None,
    annotations_jsonl: Optional[Path] = None,
    out_batch_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    judge_adapter: Optional[Judge] = None,
    adapter_name: Optional[str] = None,
) -> Path:
    """
    Create judge batch input JSONL file from query plan and annotations.
    
    Loads query plan and annotations, converts them to JudgeQuery objects with
    metadata, and writes batch input lines using the judge adapter's build_batch_lines method.
    
    Args:
        query_plan_jsonl: Input query plan JSONL path. If None, uses config.query_plan_jsonl.
        annotations_jsonl: Input annotations JSONL path. If None, uses config.annotations_jsonl.
        out_batch_jsonl: Output batch input JSONL path. If None, auto-generated.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        judge_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.judge_config.adapter.
        
    Returns:
        Path to the created batch input JSONL file.
        
    Raises:
        ValueError: If query_plan_jsonl or annotations_jsonl are not provided and not in config.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if query_plan_jsonl is None:
        query_plan_jsonl = Path(config.query_plan_jsonl) if config.query_plan_jsonl else None
    if query_plan_jsonl is None:
        raise ValueError("query_plan_jsonl must be provided or set in config.query_plan_jsonl")
    if annotations_jsonl is None:
        annotations_jsonl = Path(config.annotations_jsonl) if config.annotations_jsonl else None
    if annotations_jsonl is None:
        raise ValueError("annotations_jsonl must be provided or set in config.annotations_jsonl")
    
    if adapter_name is None:
        adapter_name = config.judge_config.adapter
    if judge_adapter is None:
        judge_adapter = JudgeAdapterRegistry.get(adapter_name, config=config)
    
    # Load query plan and annotations
    ann_map = {}
    for row in read_jsonl(annotations_jsonl):
        iid = row.get(config.column_image_id)
        if iid:
            ann_map[iid] = row
    
    queries = []
    for row in read_jsonl(query_plan_jsonl):
        query_id = row.get(config.column_query_id)
        seed_ids = row.get(config.query_plan_seed_image_ids_column, [])
        cand_ids = row.get(config.query_plan_candidate_image_ids_column, [])
        
        seed_images = []
        for sid in seed_ids:
            if sid in ann_map:
                seed_images.append(ann_map[sid])
        
        candidate_images = []
        for cid in cand_ids:
            if cid in ann_map:
                candidate_images.append(ann_map[cid])
        
        queries.append(JudgeQuery(
            query_id=query_id,
            seed_images=seed_images,
            candidate_images=candidate_images,
        ))
    
    if out_batch_jsonl is None:
        if config.qrels_jsonl:
            out_batch_jsonl = Path(config.qrels_jsonl).parent / "judge_batch_input.jsonl"
        else:
            out_batch_jsonl = Path("judge_batch_input.jsonl")
    out_batch_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    # Build batch lines and write
    batch_lines = list(judge_adapter.build_batch_lines(queries))
    write_jsonl(out_batch_jsonl, batch_lines)
    
    file_size_mb = out_batch_jsonl.stat().st_size / (1024 * 1024)
    logger.info(f"[JUDGE] Created judge batch input: {out_batch_jsonl} ({file_size_mb:.2f} MB, {len(batch_lines)} requests)")
    
    return out_batch_jsonl


def run_judge_submit(
    batch_input_jsonl: Optional[Path] = None,
    batch_id_file: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    judge_adapter: Optional[Judge] = None,
    adapter_name: Optional[str] = None,
) -> str:
    """
    Submit judge batch to the judge adapter and save batch ID(s) to file.
    
    Reads batch input JSONL, submits it using the judge adapter, and saves
    the batch ID(s) to a file for later retrieval.
    
    Args:
        batch_input_jsonl: Path to batch input JSONL file. If None, auto-generated from config.
        batch_id_file: Path to save batch ID(s). If None, auto-generated.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        judge_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.judge_config.adapter.
        
    Returns:
        Comma-separated string of batch ID(s).
        
    Raises:
        ValueError: If batch_input_jsonl is not provided and not in config.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if batch_input_jsonl is None:
        if config.qrels_jsonl:
            batch_input_jsonl = Path(config.qrels_jsonl).parent / "judge_batch_input.jsonl"
        else:
            raise ValueError("batch_input_jsonl must be provided or config.qrels_jsonl must be set")
    batch_input_jsonl = Path(batch_input_jsonl)
    
    if adapter_name is None:
        adapter_name = config.judge_config.adapter
    if judge_adapter is None:
        judge_adapter = JudgeAdapterRegistry.get(adapter_name, config=config)
    
    # Load queries to submit (reconstruct from batch input)
    # For simplicity, we'll reload from query_plan and annotations
    if config.query_plan_jsonl and config.annotations_jsonl:
        query_plan_jsonl = Path(config.query_plan_jsonl)
        annotations_jsonl = Path(config.annotations_jsonl)
        
        ann_map = {}
        for row in read_jsonl(annotations_jsonl):
            iid = row.get(config.column_image_id)
            if iid:
                ann_map[iid] = row
        
        queries = []
        for row in read_jsonl(query_plan_jsonl):
            query_id = row.get(config.column_query_id)
            seed_ids = row.get(config.query_plan_seed_image_ids_column, [])
            cand_ids = row.get(config.query_plan_candidate_image_ids_column, [])
            
            seed_images = []
            for sid in seed_ids:
                if sid in ann_map:
                    seed_images.append(ann_map[sid])
            
            candidate_images = []
            for cid in cand_ids:
                if cid in ann_map:
                    candidate_images.append(ann_map[cid])
            
            queries.append(JudgeQuery(
                query_id=query_id,
                seed_images=seed_images,
                candidate_images=candidate_images,
            ))
    else:
        raise ValueError("config.query_plan_jsonl and config.annotations_jsonl must be set to submit judge batch")
    
    batch_ref = judge_adapter.submit(
        queries=queries,
        out_jsonl=batch_input_jsonl,
    )
    
    batch_id_str = format_batch_id(batch_ref)
    if batch_id_file is None:
        batch_id_file = batch_input_jsonl.parent / ".judge_batch_id"
    save_batch_id(batch_id_str, batch_id_file)
    
    logger.info(f"[JUDGE] Submitted judge batch: {batch_id_str}")
    return batch_id_str


def run_judge_wait(
    batch_id: Optional[str] = None,
    batch_id_file: Optional[Path] = None,
    poll_s: int = 60,
    config: Optional[BenchmarkConfig] = None,
    judge_adapter: Optional[Judge] = None,
    adapter_name: Optional[str] = None,
) -> None:
    """
    Wait for judge batch(es) to complete.
    
    Loads batch ID(s) from file or uses provided batch_id, then waits for
    all batches to complete using the judge adapter's wait_for_batch method.
    
    Args:
        batch_id: Comma-separated batch ID(s). If None, loaded from batch_id_file.
        batch_id_file: Path to batch ID file. If None, auto-generated from config.
        poll_s: Polling interval in seconds (used by adapter if supported).
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        judge_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.judge_config.adapter.
        
    Raises:
        ValueError: If batch_id and batch_id_file are both None.
        FileNotFoundError: If batch_id_file is specified but doesn't exist.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if batch_id is None:
        if batch_id_file is None:
            if config.qrels_jsonl:
                batch_id_file = Path(config.qrels_jsonl).parent / ".judge_batch_id"
            else:
                raise ValueError("batch_id or batch_id_file must be provided")
        batch_id = load_batch_id(batch_id_file)
    
    if adapter_name is None:
        adapter_name = config.judge_config.adapter
    if judge_adapter is None:
        judge_adapter = JudgeAdapterRegistry.get(adapter_name, config=config)
    
    # Parse comma-separated batch IDs
    batch_ids = [bid.strip() for bid in batch_id.split(",") if bid.strip()]
    
    
    # Create BatchRefs for each batch ID
    batch_refs = [BatchRefs(input_file_id="", batch_id=bid) for bid in batch_ids]
    batch_ref = batch_refs[0] if len(batch_refs) == 1 else batch_refs
    
    # Wait for batches using adapter method
    judge_adapter.wait_for_batch(batch_ref)
    
    logger.info(f"[JUDGE] All judge batches completed: {', '.join(batch_ids)}")


def run_judge_download(
    batch_id: Optional[str] = None,
    batch_id_file: Optional[Path] = None,
    batch_output_jsonl: Optional[Path] = None,
    batch_error_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    judge_adapter: Optional[Judge] = None,
    adapter_name: Optional[str] = None,
) -> None:
    """
    Download judge batch results from the provider.
    
    Loads batch ID(s) from file or uses provided batch_id, then downloads
    batch results and errors using the judge adapter's download_batch_results method.
    
    Args:
        batch_id: Comma-separated batch ID(s). If None, loaded from batch_id_file.
        batch_id_file: Path to batch ID file. If None, auto-generated from config.
        batch_output_jsonl: Path to save batch output. If None, auto-generated.
        batch_error_jsonl: Path to save batch errors. If None, auto-generated.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        judge_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.judge_config.adapter.
        
    Raises:
        ValueError: If batch_id and batch_id_file are both None.
        FileNotFoundError: If batch_id_file is specified but doesn't exist.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if batch_id is None:
        if batch_id_file is None:
            if config.qrels_jsonl:
                batch_id_file = Path(config.qrels_jsonl).parent / ".judge_batch_id"
            else:
                raise ValueError("batch_id or batch_id_file must be provided")
        batch_id = load_batch_id(batch_id_file)
    
    if adapter_name is None:
        adapter_name = config.judge_config.adapter
    if judge_adapter is None:
        judge_adapter = JudgeAdapterRegistry.get(adapter_name, config=config)
    
    if batch_output_jsonl is None:
        if config.qrels_jsonl:
            batch_output_jsonl = Path(config.qrels_jsonl).parent / "judge_batch_output.jsonl"
        else:
            raise ValueError("batch_output_jsonl must be provided or config.qrels_jsonl must be set")
    if batch_error_jsonl is None:
        batch_error_jsonl = Path(batch_output_jsonl).parent / "judge_batch_error.jsonl"
    
    # Parse comma-separated batch IDs and create BatchRefs
    batch_ids = [bid.strip() for bid in batch_id.split(",") if bid.strip()]
    
    batch_refs = [BatchRefs(input_file_id="", batch_id=bid) for bid in batch_ids]
    batch_ref = batch_refs[0] if len(batch_refs) == 1 else batch_refs
    
    judge_adapter.download_batch_results(
        batch_ref=batch_ref,
        output_path=batch_output_jsonl,
        error_path=batch_error_jsonl,
    )
    
    logger.info(f"[JUDGE] Downloaded judge batch results to {batch_output_jsonl}")


def run_judge_parse(
    batch_output_jsonl: Optional[Path] = None,
    query_plan_jsonl: Optional[Path] = None,
    annotations_jsonl: Optional[Path] = None,
    out_qrels_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    judge_adapter: Optional[Judge] = None,
    adapter_name: Optional[str] = None,
) -> List[JudgeResult]:
    """
    Parse judge batch output into qrels JSONL format.
    
    Reads batch output JSONL, parses each response using the judge adapter's
    parse_response method, and writes the results to qrels JSONL. Failed
    requests are logged and written to a separate failed JSONL file. Metadata
    from annotations is automatically added to the qrels output.
    
    Args:
        batch_output_jsonl: Path to batch output JSONL. If None, auto-generated from config.
        query_plan_jsonl: Path to input query plan JSONL. If None, uses config.query_plan_jsonl.
        annotations_jsonl: Path to input annotations JSONL. If None, uses config.annotations_jsonl.
        out_qrels_jsonl: Path to output qrels JSONL. If None, uses config.qrels_jsonl.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        judge_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.judge_config.adapter.
        
    Returns:
        List of JudgeResult objects successfully parsed.
        
    Raises:
        ValueError: If required paths are not provided and not in config.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if batch_output_jsonl is None:
        if config.qrels_jsonl:
            batch_output_jsonl = Path(config.qrels_jsonl).parent / "judge_batch_output.jsonl"
        else:
            raise ValueError("batch_output_jsonl must be provided or config.qrels_jsonl must be set")
    if query_plan_jsonl is None:
        query_plan_jsonl = Path(config.query_plan_jsonl) if config.query_plan_jsonl else None
    if query_plan_jsonl is None:
        raise ValueError("query_plan_jsonl must be provided or set in config.query_plan_jsonl")
    if annotations_jsonl is None:
        annotations_jsonl = Path(config.annotations_jsonl) if config.annotations_jsonl else None
    if annotations_jsonl is None:
        raise ValueError("annotations_jsonl must be provided or set in config.annotations_jsonl")
    if out_qrels_jsonl is None:
        out_qrels_jsonl = Path(config.qrels_jsonl) if config.qrels_jsonl else None
    if out_qrels_jsonl is None:
        raise ValueError("out_qrels_jsonl must be provided or set in config.qrels_jsonl")
    
    if adapter_name is None:
        adapter_name = config.judge_config.adapter
    if judge_adapter is None:
        judge_adapter = JudgeAdapterRegistry.get(adapter_name, config=config)
    
    # Load annotations and query plan
    ann_map = {}
    for row in read_jsonl(annotations_jsonl):
        iid = row.get(config.column_image_id)
        if iid:
            ann_map[iid] = row
    
    queries = []
    for row in read_jsonl(query_plan_jsonl):
        query_id = row.get(config.column_query_id)
        seed_ids = row.get(config.query_plan_seed_image_ids_column, [])
        cand_ids = row.get(config.query_plan_candidate_image_ids_column, [])
        
        seed_images = []
        for sid in seed_ids:
            if sid in ann_map:
                seed_images.append(ann_map[sid])
        
        candidate_images = []
        for cid in cand_ids:
            if cid in ann_map:
                candidate_images.append(ann_map[cid])
        
        queries.append(JudgeQuery(
            query_id=query_id,
            seed_images=seed_images,
            candidate_images=candidate_images,
        ))
    
    # Parse results
    results = []
    failed_rows = []
    if Path(batch_output_jsonl).exists():
        for row in read_jsonl(batch_output_jsonl):
            custom_id = row.get("custom_id", "")
            if not custom_id.startswith(f"{config.judge_config.stage}::"):
                continue
            query_id = custom_id.split(f"{config.judge_config.stage}::", 1)[1]
            error = row.get("error")
            if error:
                logger.warning(f"[JUDGE] Judge request failed for {query_id}: {error}")
                failed_rows.append({
                    "query_id": query_id,
                    "custom_id": custom_id,
                    "error": error,
                })
                continue
            
            body = row.get("response", {}).get("body", {})
            query = next((q for q in queries if q.query_id == query_id), None)
            if query:
                try:
                    result = judge_adapter.parse_response(body, query)
                    results.append(result)
                except Exception as e:
                    logger.error(f"[JUDGE] Failed to parse judge result for {query_id}: {e}")
                    failed_rows.append({
                        "query_id": query_id,
                        "custom_id": custom_id,
                        "error": {"message": str(e), "type": "parse_error"},
                    })
    
    # Write failed rows
    if failed_rows:
        failed_path = Path(out_qrels_jsonl).parent / f"{Path(out_qrels_jsonl).stem}_failed.jsonl"
        write_jsonl(failed_path, failed_rows)
        logger.warning(f"[JUDGE] Wrote {len(failed_rows)} failed judge requests to {failed_path}")
    
    # Write qrels JSONL
    rows_out = []
    for result in results:
        for judgment in result.judgments:
            row = {
                config.column_query_id: result.query_id,
                config.column_query: result.query_text,
                config.column_image_id: judgment.image_id,
                config.column_relevance: judgment.relevance_label,
            }
            # Add metadata from annotations
            if judgment.image_id in ann_map:
                ann = ann_map[judgment.image_id]
                column_fields = config.get_columns()
                
                for field_name in column_fields:
                    field_value = getattr(config, field_name)
                    
                    if isinstance(field_value, str) and field_value in ann:
                        row[field_value] = ann[field_value]
                    elif isinstance(field_value, list):
                        for col_name in field_value:
                            if col_name and col_name in ann:
                                row[col_name] = ann[col_name]
                    elif isinstance(field_value, dict):
                        for col_name in field_value.keys():
                            if col_name and col_name in ann:
                                row[col_name] = ann[col_name]
            rows_out.append(row)
    
    write_jsonl(out_qrels_jsonl, rows_out)
    logger.info(f"[JUDGE] Parsed {len(results)} successful judge results, {len(failed_rows)} failed")
    
    # Calculate actual costs from batch output
    try:
        actual_costs = judge_adapter.calculate_actual_costs(
            batch_output_jsonl=batch_output_jsonl,
            num_items=len(results),
        )
        logger.info(f"[COST] Judge actual costs: ${actual_costs.total_cost:.2f} for {actual_costs.num_items} queries "
                   f"(${actual_costs.cost_per_item:.4f} per query, ${actual_costs.cost_per_token:.8f} per token)")
        # Store cost summary for later aggregation (we'll save it in run_cost_summary)
        # For now, just log it
    except Exception as e:
        logger.warning(f"[COST] Failed to calculate judge costs: {e}")
    
    return results


# -----------------------------
# Retry/Resubmit Functions
# -----------------------------


def run_vision_retry(
    error_jsonl: Optional[Path] = None,
    images_jsonl: Optional[Path] = None,
    out_batch_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    vision_adapter: Optional[Vision] = None,
    adapter_name: Optional[str] = None,
    submit: bool = False,
) -> Path:
    """
    Create retry batch for failed vision requests.
    
    Extracts failed image IDs from the error JSONL file, loads the corresponding
    images from images.jsonl, and creates a new batch input file for resubmission.
    Optionally submits the retry batch automatically.
    
    Args:
        error_jsonl: Path to error JSONL file. If None, auto-generated from config.
        images_jsonl: Path to input images JSONL. If None, uses config.images_jsonl.
        out_batch_jsonl: Path to output retry batch JSONL. If None, auto-generated.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        vision_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.vision_config.adapter.
        submit: If True, automatically submit the retry batch after creating it.
        
    Returns:
        Path to the created retry batch JSONL file.
        
    Raises:
        ValueError: If error_jsonl or images_jsonl are not provided and not in config.
        FileNotFoundError: If error_jsonl does not exist.
        RuntimeError: If retry batch file size exceeds 200MB limit.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if error_jsonl is None:
        if config.annotations_jsonl:
            error_jsonl = Path(config.annotations_jsonl).parent / "vision_batch_error.jsonl"
        else:
            raise ValueError("error_jsonl must be provided or config.annotations_jsonl must be set")
    error_jsonl = Path(error_jsonl)
    
    if not error_jsonl.exists():
        raise FileNotFoundError(f"Error file not found: {error_jsonl}")
    
    if images_jsonl is None:
        images_jsonl = Path(config.images_jsonl) if config.images_jsonl else None
    if images_jsonl is None:
        raise ValueError("images_jsonl must be provided or set in config.images_jsonl")
    
    # Extract failed image IDs
    failed_ids = extract_failed_ids(error_jsonl, config.vision_config.stage)
    if not failed_ids:
        logger.warning(f"[VISION] No failed vision requests found in {error_jsonl}")
        if out_batch_jsonl is None:
            out_batch_jsonl = error_jsonl.parent / "vision_batch_input.jsonl.retry"
        write_jsonl(out_batch_jsonl, [])
        logger.info(f"[VISION] Created empty retry batch file: {out_batch_jsonl}")
        return out_batch_jsonl
    
    logger.info(f"[VISION] Found {len(failed_ids)} failed image IDs to re-submit")
    
    if adapter_name is None:
        adapter_name = config.vision_config.adapter
    if vision_adapter is None:
        vision_adapter = VisionAdapterRegistry.get(adapter_name, config=config)
    
    # Load only failed images
    images = []
    for row in read_jsonl(images_jsonl):
        image_id = row.get(config.column_image_id)
        if image_id in failed_ids:
            metadata = {}
            if config.column_license in row:
                metadata[config.column_license] = row[config.column_license]
            if config.column_doi in row:
                metadata[config.column_doi] = row[config.column_doi]
            images.append(VisionImage(
                image_id=image_id,
                image_url=row[config.image_url_temp_column],
                metadata=metadata,
            ))
    
    if not images:
        logger.warning(f"[VISION] None of the failed image IDs were found in {images_jsonl}")
        if out_batch_jsonl is None:
            out_batch_jsonl = error_jsonl.parent / "vision_batch_input.jsonl.retry"
        write_jsonl(out_batch_jsonl, [])
        return out_batch_jsonl
    
    if out_batch_jsonl is None:
        out_batch_jsonl = error_jsonl.parent / "vision_batch_input.jsonl.retry"
    out_batch_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    # Build batch lines and write
    batch_lines = list(vision_adapter.build_batch_lines(images))
    write_jsonl(out_batch_jsonl, batch_lines)
    
    file_size_mb = out_batch_jsonl.stat().st_size / (1024 * 1024)
    if file_size_mb > 200:
        raise RuntimeError(
            f"Retry batch file size ({file_size_mb:.2f} MB) exceeds 200MB limit. "
            f"Consider splitting into multiple batches."
        )
    elif file_size_mb > 150:
        logger.warning(f"[VISION] Retry batch file size ({file_size_mb:.2f} MB) is approaching 200MB limit")
    
    logger.info(f"[VISION] Created vision retry batch with {len(batch_lines)} requests: {out_batch_jsonl} ({file_size_mb:.2f} MB)")
    
    if submit:
        batch_ref = vision_adapter.submit(
            images=images,
            out_jsonl=out_batch_jsonl,
        )
        
        batch_id_str = format_batch_id(batch_ref)
        batch_id_file = out_batch_jsonl.parent / ".vision_retry_batch_id"
        save_batch_id(batch_id_str, batch_id_file)
        logger.info(f"[VISION] Submitted vision retry batch: {batch_id_str}")
    
    return out_batch_jsonl


def run_judge_retry(
    error_jsonl: Optional[Path] = None,
    query_plan_jsonl: Optional[Path] = None,
    annotations_jsonl: Optional[Path] = None,
    out_batch_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    judge_adapter: Optional[Judge] = None,
    adapter_name: Optional[str] = None,
    submit: bool = False,
) -> Path:
    """
    Create retry batch for failed judge requests.
    
    Extracts failed query IDs from the error JSONL file, loads the corresponding
    queries from query_plan.jsonl and annotations.jsonl, and creates a new batch
    input file for resubmission. Optionally submits the retry batch automatically.
    
    Args:
        error_jsonl: Path to error JSONL file. If None, auto-generated from config.
        query_plan_jsonl: Path to input query plan JSONL. If None, uses config.query_plan_jsonl.
        annotations_jsonl: Path to input annotations JSONL. If None, uses config.annotations_jsonl.
        out_batch_jsonl: Path to output retry batch JSONL. If None, auto-generated.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        judge_adapter: Optional pre-instantiated adapter. If None, created from adapter_name or config.
        adapter_name: Adapter name to use. If None, uses config.judge_config.adapter.
        submit: If True, automatically submit the retry batch after creating it.
        
    Returns:
        Path to the created retry batch JSONL file.
        
    Raises:
        ValueError: If required paths are not provided and not in config.
        FileNotFoundError: If error_jsonl does not exist.
        RuntimeError: If retry batch file size exceeds 200MB limit.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    if error_jsonl is None:
        if config.qrels_jsonl:
            error_jsonl = Path(config.qrels_jsonl).parent / "judge_batch_error.jsonl"
        else:
            raise ValueError("error_jsonl must be provided or config.qrels_jsonl must be set")
    error_jsonl = Path(error_jsonl)
    
    if not error_jsonl.exists():
        raise FileNotFoundError(f"Error file not found: {error_jsonl}")
    
    if query_plan_jsonl is None:
        query_plan_jsonl = Path(config.query_plan_jsonl) if config.query_plan_jsonl else None
    if query_plan_jsonl is None:
        raise ValueError("query_plan_jsonl must be provided or set in config.query_plan_jsonl")
    if annotations_jsonl is None:
        annotations_jsonl = Path(config.annotations_jsonl) if config.annotations_jsonl else None
    if annotations_jsonl is None:
        raise ValueError("annotations_jsonl must be provided or set in config.annotations_jsonl")
    
    # Extract failed query IDs
    failed_ids = extract_failed_ids(error_jsonl, config.judge_config.stage)
    if not failed_ids:
        logger.warning(f"[JUDGE] No failed judge requests found in {error_jsonl}")
        if out_batch_jsonl is None:
            out_batch_jsonl = error_jsonl.parent / "judge_batch_input.jsonl.retry"
        write_jsonl(out_batch_jsonl, [])
        logger.info(f"[JUDGE] Created empty retry batch file: {out_batch_jsonl}")
        return out_batch_jsonl
    
    logger.info(f"Found {len(failed_ids)} failed query IDs to re-submit")
    
    if adapter_name is None:
        adapter_name = config.judge_config.adapter
    if judge_adapter is None:
        judge_adapter = JudgeAdapterRegistry.get(adapter_name, config=config)
    
    # Load annotations
    ann_map = {}
    for row in read_jsonl(annotations_jsonl):
        iid = row.get(config.column_image_id)
        if iid:
            ann_map[iid] = row
    
    # Load only failed queries
    queries = []
    for row in read_jsonl(query_plan_jsonl):
        query_id = row.get(config.column_query_id)
        if query_id in failed_ids:
            seed_ids = row.get(config.query_plan_seed_image_ids_column, [])
            cand_ids = row.get(config.query_plan_candidate_image_ids_column, [])
            
            seed_images = []
            for sid in seed_ids:
                if sid in ann_map:
                    seed_images.append(ann_map[sid])
            
            candidate_images = []
            for cid in cand_ids:
                if cid in ann_map:
                    candidate_images.append(ann_map[cid])
            
            queries.append(JudgeQuery(
                query_id=query_id,
                seed_images=seed_images,
                candidate_images=candidate_images,
            ))
    
    if not queries:
        logger.warning(f"None of the failed query IDs were found in {query_plan_jsonl}")
        if out_batch_jsonl is None:
            out_batch_jsonl = error_jsonl.parent / "judge_batch_input.jsonl.retry"
        write_jsonl(out_batch_jsonl, [])
        return out_batch_jsonl
    
    if out_batch_jsonl is None:
        out_batch_jsonl = error_jsonl.parent / "judge_batch_input.jsonl.retry"
    out_batch_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    # Build batch lines and write
    batch_lines = list(judge_adapter.build_batch_lines(queries))
    write_jsonl(out_batch_jsonl, batch_lines)
    
    file_size_mb = out_batch_jsonl.stat().st_size / (1024 * 1024)
    if file_size_mb > 200:
        raise RuntimeError(
            f"Retry batch file size ({file_size_mb:.2f} MB) exceeds 200MB limit. "
            f"Consider splitting into multiple batches."
        )
    elif file_size_mb > 150:
        logger.warning(f"[JUDGE] Retry batch file size ({file_size_mb:.2f} MB) is approaching 200MB limit")
    
    logger.info(f"[JUDGE] Created judge retry batch with {len(batch_lines)} requests: {out_batch_jsonl} ({file_size_mb:.2f} MB)")
    
    if submit:
        batch_ref = judge_adapter.submit(
            queries=queries,
            out_jsonl=out_batch_jsonl,
        )
        
        batch_id_str = format_batch_id(batch_ref)
        batch_id_file = out_batch_jsonl.parent / ".judge_retry_batch_id"
        save_batch_id(batch_id_str, batch_id_file)
        logger.info(f"[JUDGE] Submitted judge retry batch: {batch_id_str}")
    
    return out_batch_jsonl


# -----------------------------
# List Batches Function
# -----------------------------


def run_cost_summary(
    vision_batch_output_jsonl: Optional[Path] = None,
    judge_batch_output_jsonl: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
) -> None:
    """
    Calculate and write cost summary CSV for vision and judge phases.
    
    Reads batch output files for vision and judge phases, calculates actual costs
    from usage data, aggregates them, and writes a CSV summary to the summary output directory.
    
    Args:
        vision_batch_output_jsonl: Path to vision batch output JSONL. If None, auto-generated from config.
        judge_batch_output_jsonl: Path to judge batch output JSONL. If None, auto-generated from config.
        output_dir: Directory to save cost summary CSV. If None, uses config.summary_output_dir.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        
    Raises:
        ValueError: If output_dir is not provided and not in config.
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(config.summary_output_dir) if config.summary_output_dir else None
    if output_dir is None:
        raise ValueError("output_dir must be provided or set in config.summary_output_dir")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine batch output file paths
    if vision_batch_output_jsonl is None:
        if config.annotations_jsonl:
            vision_batch_output_jsonl = Path(config.annotations_jsonl).parent / "vision_batch_output.jsonl"
        else:
            vision_batch_output_jsonl = None
    
    if judge_batch_output_jsonl is None:
        if config.qrels_jsonl:
            judge_batch_output_jsonl = Path(config.qrels_jsonl).parent / "judge_batch_output.jsonl"
        else:
            judge_batch_output_jsonl = None
    
    summaries = []
    
    # Calculate vision costs
    if vision_batch_output_jsonl and Path(vision_batch_output_jsonl).exists():
        try:
            # Get vision adapter
            vision_adapter_name = config.vision_config.adapter
            if vision_adapter_name:
                vision_adapter = VisionAdapterRegistry.get(vision_adapter_name, config=config)
                vision_summary = vision_adapter.calculate_actual_costs(
                    batch_output_jsonl=vision_batch_output_jsonl,
                )
                if vision_summary.num_items > 0:
                    summaries.append(vision_summary)
                    logger.info(f"[COST] Vision costs: ${vision_summary.total_cost:.2f} for {vision_summary.num_items} images")
            else:
                logger.warning("[COST] No vision adapter configured")
        except Exception as e:
            logger.warning(f"[COST] Failed to calculate vision costs: {e}")
    else:
        logger.warning(f"[COST] Vision batch output file not found: {vision_batch_output_jsonl}")
    
    # Calculate judge costs
    if judge_batch_output_jsonl and Path(judge_batch_output_jsonl).exists():
        try:
            # Get judge adapter
            judge_adapter_name = config.judge_config.adapter
            if judge_adapter_name:
                judge_adapter = JudgeAdapterRegistry.get(judge_adapter_name, config=config)
                judge_summary = judge_adapter.calculate_actual_costs(
                    batch_output_jsonl=judge_batch_output_jsonl,
                )
                if judge_summary.num_items > 0:
                    summaries.append(judge_summary)
                    logger.info(f"[COST] Judge costs: ${judge_summary.total_cost:.2f} for {judge_summary.num_items} queries")
            else:
                logger.warning("[COST] No judge adapter configured")
        except Exception as e:
            logger.warning(f"[COST] Failed to calculate judge costs: {e}")
    else:
        logger.warning(f"[COST] Judge batch output file not found: {judge_batch_output_jsonl}")
    
    # Aggregate and write CSV
    if summaries:
        total_summary = aggregate_cost_summaries(summaries)
        summaries.append(total_summary)
        
        csv_path = output_dir / "cost_summary.csv"
        write_cost_summary_csv(summaries, csv_path)
        logger.info(f"[COST] Cost summary written to {csv_path}")
        logger.info(f"[COST] Total cost: ${total_summary.total_cost:.2f}")
    else:
        logger.warning("[COST] No cost summaries to write (no batch output files found or all had 0 items)")


def run_list_batches(
    active_only: bool = False,
    limit: int = 50,
    config: Optional[BenchmarkConfig] = None,
    adapter_name: Optional[str] = None,
) -> None:
    """
    List batches for both vision and judge adapters.
    
    Args:
        active_only: If True, only return active batches
        limit: Maximum number of batches to return per adapter
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        adapter_name: Optional adapter name to use (overrides config)
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    # List vision batches
    vision_batches = []
    try:
        vision_adapter_name = adapter_name or config.vision_config.adapter
        if vision_adapter_name:
            vision_adapter = VisionAdapterRegistry.get(vision_adapter_name, config=config)
            # If client is stored in adapter, no need to get it separately
            vision_batches = vision_adapter.list_batches(active_only=active_only, limit=limit)
        else:
            logger.warning("[LIST] No vision adapter name provided")
    except Exception as e:
        logger.warning(f"[LIST] Failed to list vision batches: {e}")
    
    # List judge batches
    judge_batches = []
    try:
        judge_adapter_name = adapter_name or config.judge_config.adapter
        if judge_adapter_name:
            judge_adapter = JudgeAdapterRegistry.get(judge_adapter_name, config=config)
            # If client is stored in adapter, no need to get it separately
            judge_batches = judge_adapter.list_batches(active_only=active_only, limit=limit)
        else:
            logger.warning("[LIST] No judge adapter name provided")
    except Exception as e:
        logger.warning(f"[LIST] Failed to list judge batches: {e}")
    
    # Print all batches
    print("\n========== VISION BATCHES ==========")
    for batch in vision_batches:
        print(json.dumps(batch, indent=2))

    print("\n========== JUDGE BATCHES ==========")
    for batch in judge_batches:
        print(json.dumps(batch, indent=2))


# -----------------------------
# CLI Interface
# -----------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Image Search Benchmark Maker Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline command")
    
    config_help = (
        "Path to TOML or JSON config file. TOML is preferred (supports comments). "
        "If not provided, uses DEFAULT_BENCHMARK_CONFIG. "
        "Sensitive fields (starting with _) should be set via environment variables."
    )
    config_default = Path(os.getenv("IMSEARCH_BENCHMAKER_CONFIG_PATH")) if os.getenv("IMSEARCH_BENCHMAKER_CONFIG_PATH") else None
    
    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess images directory")
    preprocess_parser.add_argument("--input-dir", type=Path, help="Images directory (or use config.image_root_dir)")
    preprocess_parser.add_argument("--out-images-jsonl", type=Path, help="Output images.jsonl (or use config.images_jsonl)")
    preprocess_parser.add_argument("--out-seeds-jsonl", type=Path, help="Optional output seeds.jsonl (or use config.seeds_jsonl)")
    preprocess_parser.add_argument("--meta-json", type=Path, help="Metadata JSON file (or use config.meta_json)")
    preprocess_parser.add_argument("--license", help="Default license")
    preprocess_parser.add_argument("--doi", help="Default DOI")
    preprocess_parser.add_argument("--num-seeds", type=int, help="Number of seeds")
    preprocess_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # Check image URLs
    check_urls_parser = subparsers.add_parser("check-urls", help="Check if all image URLs in images.jsonl are reachable")
    check_urls_parser.add_argument("--images-jsonl", type=Path, help="Input images.jsonl (or use config.images_jsonl)")
    check_urls_parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds (default: 10)")
    check_urls_parser.add_argument("--max-workers", type=int, default=10, help="Maximum number of concurrent requests (default: 10)")
    check_urls_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # Vision
    vision_parser = subparsers.add_parser("vision", help="Run vision annotation pipeline")
    vision_parser.add_argument("--images-jsonl", type=Path, help="Input images.jsonl (or use config.images_jsonl)")
    vision_parser.add_argument("--out-annotations-jsonl", type=Path, help="Output annotations.jsonl (or use config.annotations_jsonl)")
    vision_parser.add_argument("--adapter", help="Vision adapter name (overrides config)")
    vision_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # Query plan
    plan_parser = subparsers.add_parser("plan", help="Build query plan")
    plan_parser.add_argument("--annotations-jsonl", type=Path, help="Input annotations.jsonl (or use config.annotations_jsonl)")
    plan_parser.add_argument("--seeds-jsonl", type=Path, help="Input seeds.jsonl (or use config.seeds_jsonl)")
    plan_parser.add_argument("--out-query-plan-jsonl", type=Path, help="Output query_plan.jsonl (or use config.query_plan_jsonl)")
    plan_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # Judge
    judge_parser = subparsers.add_parser("judge", help="Run judge pipeline")
    judge_parser.add_argument("--query-plan-jsonl", type=Path, help="Input query_plan.jsonl (or use config.query_plan_jsonl)")
    judge_parser.add_argument("--annotations-jsonl", type=Path, help="Input annotations.jsonl (or use config.annotations_jsonl)")
    judge_parser.add_argument("--out-qrels-jsonl", type=Path, help="Output qrels.jsonl (or use config.qrels_jsonl)")
    judge_parser.add_argument("--adapter", help="Judge adapter name (overrides config)")
    judge_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # Postprocess
    postprocess_parser = subparsers.add_parser("postprocess", help="Postprocess qrels")
    postprocess_subparsers = postprocess_parser.add_subparsers(dest="postprocess_cmd", required=True)
    
    similarity_parser = postprocess_subparsers.add_parser("similarity", help="Calculate similarity score")
    similarity_parser.add_argument("--qrels-jsonl", type=Path, help="Input qrels.jsonl (or use config.qrels_jsonl)")
    similarity_parser.add_argument("--output-jsonl", type=Path, help="Output qrels with score (or use config.qrels_with_score_jsonl)")
    similarity_parser.add_argument("--images-jsonl", type=Path, help="Input images.jsonl (or use config.images_jsonl)")
    similarity_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    summary_parser = postprocess_subparsers.add_parser("summary", help="Generate dataset summary")
    summary_parser.add_argument("--qrels-jsonl", type=Path, help="Input qrels.jsonl (or use config.qrels_with_score_jsonl or config.qrels_jsonl)")
    summary_parser.add_argument("--output-dir", type=Path, help="Output directory (or use config.summary_output_dir)")
    summary_parser.add_argument("--images-jsonl", type=Path, help="Optional images.jsonl (or use config.images_jsonl)")
    summary_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    huggingface_parser = postprocess_subparsers.add_parser("upload", help="Upload dataset to Hugging Face")
    huggingface_parser.add_argument("--qrels-jsonl", type=Path, help="Input qrels.jsonl (or use config.qrels_with_score_jsonl or config.qrels_jsonl)")
    huggingface_parser.add_argument("--output-dir", type=Path, help="Output directory (or use config.hf_dataset_dir)")
    huggingface_parser.add_argument("--images-jsonl", type=Path, help="Optional images.jsonl (or use config.images_jsonl, required if not using --image-root-dir)")
    huggingface_parser.add_argument("--image-root-dir", type=Path, help="Optional local image root directory (or use config.image_root_dir, alternative to --images-jsonl)")
    huggingface_parser.add_argument("--progress-interval", type=int, default=100, help="Progress update interval")
    huggingface_parser.add_argument("--repo-id", help="Hugging Face repo ID (overrides config)")
    huggingface_parser.add_argument("--token", help="Hugging Face token (overrides config)")
    huggingface_parser.add_argument("--private", action="store_true", help="Make repository private (overrides config). If not set, uses config value.")
    huggingface_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # Clean
    clean_parser = subparsers.add_parser("clean", help="Remove intermediate and output files")
    clean_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    clean_parser.add_argument("--intermediate-only", action="store_true", help="Only remove intermediate files (keep final outputs)")
    clean_parser.add_argument("--include-batch-files", action="store_true", default=True, help="Remove batch input/output/error files (default: True)")
    clean_parser.add_argument("--include-final-outputs", action="store_true", help="Also remove final outputs (qrels_with_score_jsonl, summary, hf_dataset)")
    
    # Granular Vision Commands
    vision_make_parser = subparsers.add_parser("vision-make", help="Create vision batch input JSONL")
    vision_make_parser.add_argument("--images-jsonl", type=Path, help="Input images.jsonl (or use config.images_jsonl)")
    vision_make_parser.add_argument("--out-batch-jsonl", type=Path, help="Output batch input JSONL")
    vision_make_parser.add_argument("--adapter", help="Vision adapter name (overrides config)")
    vision_make_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    vision_submit_parser = subparsers.add_parser("vision-submit", help="Submit vision batch")
    vision_submit_parser.add_argument("--batch-input-jsonl", type=Path, help="Batch input JSONL (or auto-detect from config)")
    vision_submit_parser.add_argument("--batch-id-file", type=Path, help="File to save batch ID(s)")
    vision_submit_parser.add_argument("--adapter", help="Vision adapter name (overrides config)")
    vision_submit_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    vision_wait_parser = subparsers.add_parser("vision-wait", help="Wait for vision batch(es) to complete")
    vision_wait_parser.add_argument("--batch-id", help="Batch ID(s), comma-separated (or use --batch-id-file)")
    vision_wait_parser.add_argument("--batch-id-file", type=Path, help="File containing batch ID(s)")
    vision_wait_parser.add_argument("--poll-s", type=int, default=60, help="Poll interval in seconds")
    vision_wait_parser.add_argument("--adapter", help="Vision adapter name (overrides config)")
    vision_wait_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    vision_download_parser = subparsers.add_parser("vision-download", help="Download vision batch results")
    vision_download_parser.add_argument("--batch-id", help="Batch ID(s), comma-separated (or use --batch-id-file)")
    vision_download_parser.add_argument("--batch-id-file", type=Path, help="File containing batch ID(s)")
    vision_download_parser.add_argument("--batch-output-jsonl", type=Path, help="Output batch output JSONL")
    vision_download_parser.add_argument("--batch-error-jsonl", type=Path, help="Output batch error JSONL")
    vision_download_parser.add_argument("--adapter", help="Vision adapter name (overrides config)")
    vision_download_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    vision_parse_parser = subparsers.add_parser("vision-parse", help="Parse vision batch output to annotations JSONL")
    vision_parse_parser.add_argument("--batch-output-jsonl", type=Path, help="Input batch output JSONL")
    vision_parse_parser.add_argument("--images-jsonl", type=Path, help="Input images.jsonl (or use config.images_jsonl)")
    vision_parse_parser.add_argument("--out-annotations-jsonl", type=Path, help="Output annotations.jsonl (or use config.annotations_jsonl)")
    vision_parse_parser.add_argument("--adapter", help="Vision adapter name (overrides config)")
    vision_parse_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # Granular Judge Commands
    judge_make_parser = subparsers.add_parser("judge-make", help="Create judge batch input JSONL")
    judge_make_parser.add_argument("--query-plan-jsonl", type=Path, help="Input query_plan.jsonl (or use config.query_plan_jsonl)")
    judge_make_parser.add_argument("--annotations-jsonl", type=Path, help="Input annotations.jsonl (or use config.annotations_jsonl)")
    judge_make_parser.add_argument("--out-batch-jsonl", type=Path, help="Output batch input JSONL")
    judge_make_parser.add_argument("--adapter", help="Judge adapter name (overrides config)")
    judge_make_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    judge_submit_parser = subparsers.add_parser("judge-submit", help="Submit judge batch")
    judge_submit_parser.add_argument("--batch-input-jsonl", type=Path, help="Batch input JSONL (or auto-detect from config)")
    judge_submit_parser.add_argument("--batch-id-file", type=Path, help="File to save batch ID(s)")
    judge_submit_parser.add_argument("--adapter", help="Judge adapter name (overrides config)")
    judge_submit_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    judge_wait_parser = subparsers.add_parser("judge-wait", help="Wait for judge batch(es) to complete")
    judge_wait_parser.add_argument("--batch-id", help="Batch ID(s), comma-separated (or use --batch-id-file)")
    judge_wait_parser.add_argument("--batch-id-file", type=Path, help="File containing batch ID(s)")
    judge_wait_parser.add_argument("--poll-s", type=int, default=60, help="Poll interval in seconds")
    judge_wait_parser.add_argument("--adapter", help="Judge adapter name (overrides config)")
    judge_wait_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    judge_download_parser = subparsers.add_parser("judge-download", help="Download judge batch results")
    judge_download_parser.add_argument("--batch-id", help="Batch ID(s), comma-separated (or use --batch-id-file)")
    judge_download_parser.add_argument("--batch-id-file", type=Path, help="File containing batch ID(s)")
    judge_download_parser.add_argument("--batch-output-jsonl", type=Path, help="Output batch output JSONL")
    judge_download_parser.add_argument("--batch-error-jsonl", type=Path, help="Output batch error JSONL")
    judge_download_parser.add_argument("--adapter", help="Judge adapter name (overrides config)")
    judge_download_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    judge_parse_parser = subparsers.add_parser("judge-parse", help="Parse judge batch output to qrels JSONL")
    judge_parse_parser.add_argument("--batch-output-jsonl", type=Path, help="Input batch output JSONL")
    judge_parse_parser.add_argument("--query-plan-jsonl", type=Path, help="Input query_plan.jsonl (or use config.query_plan_jsonl)")
    judge_parse_parser.add_argument("--annotations-jsonl", type=Path, help="Input annotations.jsonl (or use config.annotations_jsonl)")
    judge_parse_parser.add_argument("--out-qrels-jsonl", type=Path, help="Output qrels.jsonl (or use config.qrels_jsonl)")
    judge_parse_parser.add_argument("--adapter", help="Judge adapter name (overrides config)")
    judge_parse_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # Retry Commands
    vision_retry_parser = subparsers.add_parser("vision-retry", help="Create retry batch for failed vision requests")
    vision_retry_parser.add_argument("--error-jsonl", type=Path, help="Error JSONL file (or auto-detect from config)")
    vision_retry_parser.add_argument("--images-jsonl", type=Path, help="Input images.jsonl (or use config.images_jsonl)")
    vision_retry_parser.add_argument("--out-batch-jsonl", type=Path, help="Output retry batch JSONL")
    vision_retry_parser.add_argument("--submit", action="store_true", help="Automatically submit the retry batch")
    vision_retry_parser.add_argument("--adapter", help="Vision adapter name (overrides config)")
    vision_retry_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    judge_retry_parser = subparsers.add_parser("judge-retry", help="Create retry batch for failed judge requests")
    judge_retry_parser.add_argument("--error-jsonl", type=Path, help="Error JSONL file (or auto-detect from config)")
    judge_retry_parser.add_argument("--query-plan-jsonl", type=Path, help="Input query_plan.jsonl (or use config.query_plan_jsonl)")
    judge_retry_parser.add_argument("--annotations-jsonl", type=Path, help="Input annotations.jsonl (or use config.annotations_jsonl)")
    judge_retry_parser.add_argument("--out-batch-jsonl", type=Path, help="Output retry batch JSONL")
    judge_retry_parser.add_argument("--submit", action="store_true", help="Automatically submit the retry batch")
    judge_retry_parser.add_argument("--adapter", help="Judge adapter name (overrides config)")
    judge_retry_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # List Batches
    list_batches_parser = subparsers.add_parser("list-batches", help="List OpenAI batches")
    list_batches_parser.add_argument("--active-only", action="store_true", help="Only show active batches")
    list_batches_parser.add_argument("--limit", type=int, default=50, help="Maximum number of batches to return per adapter")
    list_batches_parser.add_argument("--adapter", help="Adapter name (overrides config)")
    list_batches_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    # All (full pipeline)
    all_parser = subparsers.add_parser("all", help="Run complete pipeline: preprocess -> vision -> plan -> judge -> similarity -> summary -> upload")
    all_parser.add_argument("--input-dir", type=Path, help="Images directory (or use config.image_root_dir)")
    all_parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocess step")
    all_parser.add_argument("--skip-vision", action="store_true", help="Skip vision step")
    all_parser.add_argument("--skip-query-plan", action="store_true", help="Skip query plan step")
    all_parser.add_argument("--skip-judge", action="store_true", help="Skip judge step")
    all_parser.add_argument("--skip-similarity", action="store_true", help="Skip similarity score step")
    all_parser.add_argument("--skip-postprocess", action="store_true", help="Skip postprocess summary step")
    all_parser.add_argument("--skip-huggingface", action="store_true", help="Skip Hugging Face upload step")
    all_parser.add_argument("--no-wait", action="store_true", help="Don't wait for batch completion (submit only)")
    all_parser.add_argument("--config", type=Path, help=config_help, default=config_default)
    
    return parser

def run_clean(
    config: Optional[BenchmarkConfig] = None,
    intermediate_only: bool = False,
    include_batch_files: bool = True,
    include_final_outputs: bool = False,
) -> None:
    """
    Remove intermediate and output files created during the benchmark pipeline.
    
    Args:
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        intermediate_only: If True, only remove intermediate files (keep final outputs).
        include_batch_files: If True, remove batch input/output/error files.
        include_final_outputs: If True, also remove final outputs (qrels_with_score_jsonl, summary, hf_dataset).
    """
    config = config or DEFAULT_BENCHMARK_CONFIG
    
    removed_files = []
    removed_dirs = []
    
    # Intermediate files to remove
    intermediate_paths = []
    if config.annotations_jsonl:
        intermediate_paths.append(Path(config.annotations_jsonl))
    if config.query_plan_jsonl:
        intermediate_paths.append(Path(config.query_plan_jsonl))
    if config.qrels_jsonl:
        intermediate_paths.append(Path(config.qrels_jsonl))
    
    # Final outputs (only if include_final_outputs is True)
    final_paths = []
    if include_final_outputs and not intermediate_only:
        if config.qrels_with_score_jsonl:
            final_paths.append(Path(config.qrels_with_score_jsonl))
        if config.summary_output_dir:
            final_paths.append(Path(config.summary_output_dir))
        if config.hf_dataset_dir:
            final_paths.append(Path(config.hf_dataset_dir))
    
    # Remove intermediate files
    for path in intermediate_paths:
        if path and path.exists():
            if path.is_file():
                path.unlink()
                removed_files.append(str(path))
            elif path.is_dir():
                shutil.rmtree(path)
                removed_dirs.append(str(path))
    
    # Remove final outputs if requested
    for path in final_paths:
        if path and path.exists():
            if path.is_file():
                path.unlink()
                removed_files.append(str(path))
            elif path.is_dir():
                shutil.rmtree(path)
                removed_dirs.append(str(path))
    
    # Remove batch files if requested
    if include_batch_files:
        # Vision batch files (in annotations_jsonl parent directory)
        if config.annotations_jsonl:
            annotations_dir = Path(config.annotations_jsonl).parent
            batch_patterns = [
                "vision_batch_input.jsonl",
                "vision_batch_output.jsonl",
                "vision_batch_error.jsonl",
                "vision_batch_input.jsonl.retry",
                "vision_batch_output.jsonl.retry",
                "vision_batch_error.jsonl.retry",
            ]
            for pattern in batch_patterns:
                batch_file = annotations_dir / pattern
                if batch_file.exists():
                    batch_file.unlink()
                    removed_files.append(str(batch_file))
        
        # Judge batch files (in qrels_jsonl parent directory)
        if config.qrels_jsonl:
            qrels_dir = Path(config.qrels_jsonl).parent
            batch_patterns = [
                "judge_batch_input.jsonl",
                "judge_batch_output.jsonl",
                "judge_batch_error.jsonl",
                "judge_batch_input.jsonl.retry",
                "judge_batch_output.jsonl.retry",
                "judge_batch_error.jsonl.retry",
            ]
            for pattern in batch_patterns:
                batch_file = qrels_dir / pattern
                if batch_file.exists():
                    batch_file.unlink()
                    removed_files.append(str(batch_file))
        
        # Batch ID files (common locations)
        common_dirs = []
        if config.annotations_jsonl:
            common_dirs.append(Path(config.annotations_jsonl).parent)
        if config.qrels_jsonl:
            common_dirs.append(Path(config.qrels_jsonl).parent)
        if config.query_plan_jsonl:
            common_dirs.append(Path(config.query_plan_jsonl).parent)
        
        batch_id_patterns = [
            ".vision_batch_id",
            ".vision_retry_batch_id",
            ".judge_batch_id",
            ".judge_retry_batch_id",
        ]
        
        for common_dir in set(common_dirs):  # Use set to avoid duplicates
            for pattern in batch_id_patterns:
                batch_id_file = common_dir / pattern
                if batch_id_file.exists():
                    batch_id_file.unlink()
                    removed_files.append(str(batch_id_file))
    
    # Report results
    if removed_files or removed_dirs:
        if removed_files:
            logger.info(f"Removed {len(removed_files)} file(s):")
            for f in removed_files:
                logger.info(f"  - {f}")
        if removed_dirs:
            logger.info(f"Removed {len(removed_dirs)} directory/directories):")
            for d in removed_dirs:
                logger.info(f"  - {d}")
        logger.info("✅ Cleanup complete!")
    else:
        logger.info("✅ No files to clean (or paths not set in config)")

def main() -> None:
    """CLI entry point."""
    parser = build_cli_parser()
    args = parser.parse_args()
    
    # Load config from file if provided, otherwise use default
    config = DEFAULT_BENCHMARK_CONFIG
    if hasattr(args, "config") and args.config:
        try:
            # from_file() automatically detects format and loads config
            config = BenchmarkConfig.from_file(args.config)
        except Exception as e:
            logger.error(f"Failed to load config from {args.config}: {e}")
            raise
    
    if args.command == "preprocess":
        run_preprocess(
            input_dir=getattr(args, "input_dir", None),
            out_images_jsonl=getattr(args, "out_images_jsonl", None),
            config=config,
            out_seeds_jsonl=getattr(args, "out_seeds_jsonl", None),
            meta_json=getattr(args, "meta_json", None),
            default_license=getattr(args, "license", None),
            default_doi=getattr(args, "doi", None),
            num_seeds=getattr(args, "num_seeds", None),
        )
        output_path = getattr(args, "out_images_jsonl", None) or config.images_jsonl
        logger.info(f"✅ Preprocess complete -> {output_path}")
    
    elif args.command == "check-urls":
        result = check_image_urls(
            images_jsonl=getattr(args, "images_jsonl", None),
            config=config,
            timeout=getattr(args, "timeout", 10),
            max_workers=getattr(args, "max_workers", 10),
        )
        logger.info("=" * 80)
        logger.info("Image URL Check Results")
        logger.info("=" * 80)
        logger.info(f"Total images: {result['total_count']}")
        logger.info(f"Successful: {result['success_count']}")
        logger.info(f"Failed: {result['failed_count']}")
        if result['failed_image_ids']:
            logger.info(f"\nFailed image IDs ({len(result['failed_image_ids'])}):")
            for image_id in result['failed_image_ids']:
                logger.info(f"  - {image_id}")
        logger.info("=" * 80)
    
    elif args.command == "vision":
        adapter_name = getattr(args, "adapter", None) or config.vision_config.adapter
        run_vision(
            images_jsonl=getattr(args, "images_jsonl", None),
            out_annotations_jsonl=getattr(args, "out_annotations_jsonl", None),
            config=config,
            adapter_name=adapter_name,
        )
        output_path = getattr(args, "out_annotations_jsonl", None) or config.annotations_jsonl
        logger.info(f"✅ Vision complete -> {output_path}")
    
    elif args.command == "plan":
        run_query_plan(
            annotations_jsonl=getattr(args, "annotations_jsonl", None),
            seeds_jsonl=getattr(args, "seeds_jsonl", None),
            out_query_plan_jsonl=getattr(args, "out_query_plan_jsonl", None),
            config=config,
        )
        output_path = getattr(args, "out_query_plan_jsonl", None) or config.query_plan_jsonl
        logger.info(f"✅ Query plan complete -> {output_path}")
    
    elif args.command == "judge":
        adapter_name = getattr(args, "adapter", None) or config.judge_config.adapter
        run_judge(
            query_plan_jsonl=getattr(args, "query_plan_jsonl", None),
            annotations_jsonl=getattr(args, "annotations_jsonl", None),
            out_qrels_jsonl=getattr(args, "out_qrels_jsonl", None),
            config=config,
            adapter_name=adapter_name,
        )
        output_path = getattr(args, "out_qrels_jsonl", None) or config.qrels_jsonl
        logger.info(f"✅ Judge complete -> {output_path}")
    
    elif args.command == "postprocess":
        if args.postprocess_cmd == "similarity":
            # Get adapter name from config or use default
            adapter_name = config.similarity_config.adapter
            if not adapter_name:
                available_adapters = SimilarityAdapterRegistry.list_adapters()
                logger.warning(
                    f"No similarity adapter specified in config. Available adapters: {', '.join(available_adapters)}. "
                    f"Using first available adapter: {available_adapters[0] if available_adapters else 'none'}"
                )
                adapter_name = available_adapters[0] if available_adapters else None
            
            # Get column name from config
            col_name = config.similarity_config.col_name or "similarity_score"
            
            calculate_similarity_score(
                qrels_path=getattr(args, "qrels_jsonl", None),
                output_path=getattr(args, "output_jsonl", None),
                col_name=col_name,
                adapter_name=adapter_name,
                images_jsonl_path=getattr(args, "images_jsonl", None),
                config=config,
            )
            output_path = getattr(args, "output_jsonl", None) or config.qrels_with_score_jsonl
            logger.info(f"✅ Similarity score calculation complete -> {output_path}")
        elif args.postprocess_cmd == "summary":
            generate_dataset_summary(
                qrels_path=getattr(args, "qrels_jsonl", None),
                output_dir=getattr(args, "output_dir", None),
                images_jsonl_path=getattr(args, "images_jsonl", None),
                config=config,
            )
            output_path = getattr(args, "output_dir", None) or config.summary_output_dir
            logger.info(f"✅ Summary complete -> {output_path}")
        elif args.postprocess_cmd == "upload":
            # Handle private flag: if --private is set, use True; otherwise None (will use config)
            private_value = args.private if args.private else None
            
            huggingface(
                qrels_path=getattr(args, "qrels_jsonl", None),
                output_dir=getattr(args, "output_dir", None),
                images_jsonl_path=getattr(args, "images_jsonl", None),
                image_root_dir=getattr(args, "image_root_dir", None),
                progress_interval=getattr(args, "progress_interval", 100),
                repo_id=getattr(args, "repo_id", None),
                token=getattr(args, "token", None),
                private=private_value,
                config=config,
            )
            output_path = getattr(args, "output_dir", None) or config.hf_dataset_dir
            logger.info(f"✅ Hugging Face upload complete -> {output_path}")
    
    elif args.command == "clean":
        run_clean(
            config=config,
            intermediate_only=getattr(args, "intermediate_only", False),
            include_batch_files=getattr(args, "include_batch_files", True),
            include_final_outputs=getattr(args, "include_final_outputs", False),
        )
        logger.info("✅ Clean complete")
    
    # Granular Vision Commands
    elif args.command == "vision-make":
        adapter_name = getattr(args, "adapter", None) or config.vision_config.adapter
        out_path = run_vision_make(
            images_jsonl=getattr(args, "images_jsonl", None),
            out_batch_jsonl=getattr(args, "out_batch_jsonl", None),
            config=config,
            adapter_name=adapter_name,
        )
        logger.info(f"✅ Vision batch created -> {out_path}")
    
    elif args.command == "vision-submit":
        adapter_name = getattr(args, "adapter", None) or config.vision_config.adapter
        batch_id = run_vision_submit(
            batch_input_jsonl=getattr(args, "batch_input_jsonl", None),
            batch_id_file=getattr(args, "batch_id_file", None),
            config=config,
            adapter_name=adapter_name,
        )
        logger.info(f"✅ Vision batch submitted: {batch_id}")
    
    elif args.command == "vision-wait":
        adapter_name = getattr(args, "adapter", None) or config.vision_config.adapter
        run_vision_wait(
            batch_id=getattr(args, "batch_id", None),
            batch_id_file=getattr(args, "batch_id_file", None),
            poll_s=getattr(args, "poll_s", 60),
            config=config,
            adapter_name=adapter_name,
        )
        logger.info("✅ Vision batch(es) completed")
    
    elif args.command == "vision-download":
        adapter_name = getattr(args, "adapter", None) or config.vision_config.adapter
        run_vision_download(
            batch_id=getattr(args, "batch_id", None),
            batch_id_file=getattr(args, "batch_id_file", None),
            batch_output_jsonl=getattr(args, "batch_output_jsonl", None),
            batch_error_jsonl=getattr(args, "batch_error_jsonl", None),
            config=config,
            adapter_name=adapter_name,
        )
        logger.info("✅ Vision batch results downloaded")
    
    elif args.command == "vision-parse":
        adapter_name = getattr(args, "adapter", None) or config.vision_config.adapter
        run_vision_parse(
            batch_output_jsonl=getattr(args, "batch_output_jsonl", None),
            images_jsonl=getattr(args, "images_jsonl", None),
            out_annotations_jsonl=getattr(args, "out_annotations_jsonl", None),
            config=config,
            adapter_name=adapter_name,
        )
        output_path = getattr(args, "out_annotations_jsonl", None) or config.annotations_jsonl
        logger.info(f"✅ Vision parsing complete -> {output_path}")
    
    # Granular Judge Commands
    elif args.command == "judge-make":
        adapter_name = getattr(args, "adapter", None) or config.judge_config.adapter
        out_path = run_judge_make(
            query_plan_jsonl=getattr(args, "query_plan_jsonl", None),
            annotations_jsonl=getattr(args, "annotations_jsonl", None),
            out_batch_jsonl=getattr(args, "out_batch_jsonl", None),
            config=config,
            adapter_name=adapter_name,
        )
        logger.info(f"✅ Judge batch created -> {out_path}")
    
    elif args.command == "judge-submit":
        adapter_name = getattr(args, "adapter", None) or config.judge_config.adapter
        batch_id = run_judge_submit(
            batch_input_jsonl=getattr(args, "batch_input_jsonl", None),
            batch_id_file=getattr(args, "batch_id_file", None),
            config=config,
            adapter_name=adapter_name,
        )
        logger.info(f"✅ Judge batch submitted: {batch_id}")
    
    elif args.command == "judge-wait":
        adapter_name = getattr(args, "adapter", None) or config.judge_config.adapter
        run_judge_wait(
            batch_id=getattr(args, "batch_id", None),
            batch_id_file=getattr(args, "batch_id_file", None),
            poll_s=getattr(args, "poll_s", 60),
            config=config,
            adapter_name=adapter_name,
        )
        logger.info("✅ Judge batch(es) completed")
    
    elif args.command == "judge-download":
        adapter_name = getattr(args, "adapter", None) or config.judge_config.adapter
        run_judge_download(
            batch_id=getattr(args, "batch_id", None),
            batch_id_file=getattr(args, "batch_id_file", None),
            batch_output_jsonl=getattr(args, "batch_output_jsonl", None),
            batch_error_jsonl=getattr(args, "batch_error_jsonl", None),
            config=config,
            adapter_name=adapter_name,
        )
        logger.info("✅ Judge batch results downloaded")
    
    elif args.command == "judge-parse":
        adapter_name = getattr(args, "adapter", None) or config.judge_config.adapter
        run_judge_parse(
            batch_output_jsonl=getattr(args, "batch_output_jsonl", None),
            query_plan_jsonl=getattr(args, "query_plan_jsonl", None),
            annotations_jsonl=getattr(args, "annotations_jsonl", None),
            out_qrels_jsonl=getattr(args, "out_qrels_jsonl", None),
            config=config,
            adapter_name=adapter_name,
        )
        output_path = getattr(args, "out_qrels_jsonl", None) or config.qrels_jsonl
        logger.info(f"✅ Judge parsing complete -> {output_path}")
    
    # Retry Commands
    elif args.command == "vision-retry":
        adapter_name = getattr(args, "adapter", None) or config.vision_config.adapter
        out_path = run_vision_retry(
            error_jsonl=getattr(args, "error_jsonl", None),
            images_jsonl=getattr(args, "images_jsonl", None),
            out_batch_jsonl=getattr(args, "out_batch_jsonl", None),
            config=config,
            adapter_name=adapter_name,
            submit=getattr(args, "submit", False),
        )
        logger.info(f"✅ Vision retry batch created -> {out_path}")
        if getattr(args, "submit", False):
            logger.info("✅ Vision retry batch submitted")
    
    elif args.command == "judge-retry":
        adapter_name = getattr(args, "adapter", None) or config.judge_config.adapter
        out_path = run_judge_retry(
            error_jsonl=getattr(args, "error_jsonl", None),
            query_plan_jsonl=getattr(args, "query_plan_jsonl", None),
            annotations_jsonl=getattr(args, "annotations_jsonl", None),
            out_batch_jsonl=getattr(args, "out_batch_jsonl", None),
            config=config,
            adapter_name=adapter_name,
            submit=getattr(args, "submit", False),
        )
        logger.info(f"✅ Judge retry batch created -> {out_path}")
        if getattr(args, "submit", False):
            logger.info("[JUDGE] ✅ Judge retry batch submitted")
    
    # List Batches
    elif args.command == "list-batches":
        run_list_batches(
            active_only=getattr(args, "active_only", False),
            limit=getattr(args, "limit", 50),
            config=config,
            adapter_name=getattr(args, "adapter", None),
        )
    
    # All (full pipeline)
    elif args.command == "all":
        run_all(
            input_dir=getattr(args, "input_dir", None),
            config=config,
            wait_for_completion=not getattr(args, "no_wait", False),
            skip_preprocess=getattr(args, "skip_preprocess", False),
            skip_vision=getattr(args, "skip_vision", False),
            skip_query_plan=getattr(args, "skip_query_plan", False),
            skip_judge=getattr(args, "skip_judge", False),
            skip_similarity=getattr(args, "skip_similarity", False),
            skip_postprocess=getattr(args, "skip_postprocess", False),
            skip_huggingface=getattr(args, "skip_huggingface", False),
        )
        logger.info("✅ Full pipeline complete!")


if __name__ == "__main__":
    main()

