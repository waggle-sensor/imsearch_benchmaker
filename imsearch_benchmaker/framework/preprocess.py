"""
preprocess.py

Framework utilities for building images.jsonl and seeds.jsonl.

This module provides functions to preprocess image directories into JSONL format,
extracting image metadata, constructing image URLs, and creating seed image lists
for query planning.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

from .io import read_jsonl, write_jsonl
from .config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG

logger = logging.getLogger(__name__)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"
}


@dataclass(frozen=True)
class Meta:
    """
    Image metadata container for license, DOI, and dataset name information.
    
    Attributes:
        license: License string for the image (e.g., "CC BY 4.0").
        doi: Digital Object Identifier or source identifier.
        dataset_name: Name of the original dataset the image came from.
    """
    license: str
    doi: str
    dataset_name: str


def posix_relpath(path: Path, root: Path) -> str:
    """
    Get POSIX-style relative path from root to path.
    
    Args:
        path: Target file path.
        root: Root directory path.
        
    Returns:
        POSIX-style relative path string.
    """
    return path.relative_to(root).as_posix()


def is_image_file(path: Path) -> bool:
    """
    Check if a path is an image file based on extension.
    
    Args:
        path: File path to check.
        
    Returns:
        True if the file has an image extension, False otherwise.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def remove_macos_metadata_files(root: Path) -> int:
    """
    Remove macOS metadata files (.DS_Store and ._* files) from a directory tree.
    
    macOS creates these files automatically and they're not needed for image processing.
    
    Args:
        root: Root directory to search for and remove metadata files.
        
    Returns:
        Number of metadata files removed.
    """
    logging.info("Removing macOS metadata files (.DS_Store and ._* files)...")
    removed_count = 0
    
    # Collect all files first to get accurate progress
    all_files = [p for p in root.rglob("*") if p.is_file()]
    
    with tqdm(total=len(all_files), desc="Scanning for metadata files", unit="file") as pbar:
        for path in all_files:
            pbar.update(1)
            
            # Remove .DS_Store files
            if path.name == ".DS_Store":
                try:
                    path.unlink()
                    removed_count += 1
                    logger.debug(f"[PREPROCESS] Removed macOS metadata file: {path}")
                    pbar.set_postfix({"removed": removed_count})
                except OSError as e:
                    logger.warning(f"[PREPROCESS] Failed to remove {path}: {e}")
            
            # Remove files starting with ._
            elif path.name.startswith("._"):
                try:
                    path.unlink()
                    removed_count += 1
                    logger.debug(f"[PREPROCESS] Removed macOS metadata file: {path}")
                    pbar.set_postfix({"removed": removed_count})
                except OSError as e:
                    logger.warning(f"[PREPROCESS] Failed to remove {path}: {e}")
    
    if removed_count > 0:
        logger.info(f"[PREPROCESS] Removed {removed_count} macOS metadata file(s) from {root}")
    
    return removed_count


def guess_mime_type(path: Path) -> str:
    """
    Guess MIME type for a file path.
    
    Args:
        path: File path to guess MIME type for.
        
    Returns:
        MIME type string, or "application/octet-stream" if unknown.
    """
    mt, _ = mimetypes.guess_type(path.name)
    return mt or "application/octet-stream"


def load_meta_config(meta_json: Path) -> Dict[str, Any]:
    """
    Load metadata configuration from a JSON file.
    
    Args:
        meta_json: Path to metadata JSON file.
        
    Returns:
        Dictionary containing metadata configuration.
    """
    with meta_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def meta_from_config(image_id: str, cfg: Dict[str, Any]) -> Meta:
    """
    Extract metadata for an image from configuration dictionary.
    
    Checks files, prefixes, and defaults in order of specificity.
    
    Args:
        image_id: Image identifier to look up.
        cfg: Metadata configuration dictionary with "default", "files", and "prefixes" keys.
        
    Returns:
        Meta object with license, DOI, and dataset name for the image.
    """
    default = cfg.get("default") or {}
    license_ = default.get("license", "")
    doi = default.get("doi", "")
    dataset_name = default.get("dataset_name", "")

    files = cfg.get("files") or {}
    if image_id in files:
        rec = files[image_id] or {}
        return Meta(
            license=str(rec.get("license", license_)),
            doi=str(rec.get("doi", doi)),
            dataset_name=str(rec.get("dataset_name", dataset_name)),
        )

    prefixes = cfg.get("prefixes") or []
    best: Optional[Tuple[int, Dict[str, Any]]] = None
    for rec in prefixes:
        pref = str(rec.get("prefix", ""))
        if not pref:
            continue
        if image_id.startswith(pref):
            ln = len(pref)
            if best is None or ln > best[0]:
                best = (ln, rec)

    if best is not None:
        rec = best[1]
        return Meta(
            license=str(rec.get("license", license_)),
            doi=str(rec.get("doi", doi)),
            dataset_name=str(rec.get("dataset_name", dataset_name)),
        )

    return Meta(license=str(license_), doi=str(doi), dataset_name=str(dataset_name))


def prompt_default_meta() -> Meta:
    """
    Prompt user for default metadata via command line input.
    
    Returns:
        Meta object with user-provided license, DOI, and dataset name, or "UNKNOWN" if empty.
    """
    license_ = input("License (e.g., CC BY 4.0, custom, UNKNOWN): ").strip()
    doi_val = input("DOI (e.g., HPWREN, Kaggle contributor, UNKNOWN): ").strip()
    dataset_name_val = input("Dataset name (e.g., Sage, Wildfire, UNKNOWN): ").strip()
    return Meta(license=license_ or "UNKNOWN", doi=doi_val or "UNKNOWN", dataset_name=dataset_name_val or "UNKNOWN")


def iter_images(root: Path, follow_symlinks: bool = False) -> Iterable[Path]:
    """
    Recursively iterate over all image files in a directory.
    
    Args:
        root: Root directory to search for images.
        follow_symlinks: If True, follow symbolic links; otherwise skip them.
        
    Yields:
        Path objects for each image file found.
    """
    for path in root.rglob("*"):
        if not follow_symlinks and path.is_symlink():
            continue
        if is_image_file(path):
            yield path


def build_row(
    image_path: Path,
    root: Path,
    meta: Meta,
    image_base_url: str,
    config: BenchmarkConfig,
) -> Dict[str, Any]:
    """
    Build a row dictionary for an image in the images.jsonl format.
    
    Args:
        image_path: Path to the image file.
        root: Root directory for computing relative paths.
        meta: Metadata object containing license, DOI, and dataset name.
        image_base_url: Base URL for constructing image URLs.
        config: BenchmarkConfig instance for column names.
        
    Returns:
        Dictionary with image_id, image_url, mime_type, license, doi, and original_dataset_name columns.
    """
    image_id = posix_relpath(image_path, root)
    base = image_base_url.rstrip("/")
    img_id = image_id.lstrip("/")
    image_url = f"{base}/{img_id}"
    
    # Extract dataset name from meta, or fall back to image_id prefix
    dataset_name = meta.dataset_name
    if not dataset_name or dataset_name == "UNKNOWN":
        # Extract prefix from image_id (everything before first "/")
        if "/" in image_id:
            dataset_name = image_id.split("/")[0]
        else:
            dataset_name = "UNKNOWN"
    
    return {
        config.column_image_id: image_id,
        config.image_url_temp_column: image_url,
        config.column_mime_type: guess_mime_type(image_path),
        config.column_license: meta.license,
        config.column_doi: meta.doi,
        config.column_original_dataset_name: dataset_name,
    }


def write_seeds_jsonl(
    image_ids: List[str],
    out_path: Path,
    num_seeds: int,
    seed_prefix: str,
    config: BenchmarkConfig,
) -> None:
    """
    Write seeds JSONL file with evenly distributed seed images.
    
    Args:
        image_ids: List of all image IDs to select seeds from.
        out_path: Path to write the seeds JSONL file.
        num_seeds: Number of seed queries to create.
        seed_prefix: Prefix for query IDs (e.g., "query_").
        config: BenchmarkConfig instance for column names.
        
    Raises:
        RuntimeError: If there are fewer images than requested seeds.
    """
    if len(image_ids) < num_seeds:
        raise RuntimeError(f"Not enough images ({len(image_ids)}) to create {num_seeds} seeds")

    image_ids_sorted = sorted(image_ids)
    step = len(image_ids_sorted) / num_seeds

    seeds = []
    with tqdm(total=num_seeds, desc="Creating seed images", unit="seed") as pbar:
        for i in range(num_seeds):
            idx = int(i * step)
            seeds.append({
                config.column_query_id: f"{seed_prefix}{i+1:03d}",
                config.query_plan_seed_image_ids_column: [image_ids_sorted[idx]],
            })
            pbar.update(1)

    with out_path.open("w", encoding="utf-8") as f:
        with tqdm(total=len(seeds), desc="Writing seeds.jsonl", unit="seed") as pbar:
            for seed in seeds:
                f.write(json.dumps(seed, ensure_ascii=False) + "\n")
                pbar.update(1)


def build_images_jsonl(
    input_dir: Path,
    out_jsonl: Path,
    image_base_url: Optional[str] = None,
    meta_json: Optional[Path] = None,
    metadata_jsonl: Optional[Path] = None,
    default_license: Optional[str] = None,
    default_doi: Optional[str] = None,
    follow_symlinks: bool = False,
    limit: int = 0,
    skip_percent_paths: bool = True,
    config: Optional[BenchmarkConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Build images.jsonl file from a directory of image files.
    
    Scans the input directory for image files, extracts metadata, constructs image URLs,
    and writes the results to a JSONL file. Supports metadata from JSON config files,
    metadata JSONL files, or interactive prompts.
    
    Args:
        input_dir: Directory containing image files to process.
        out_jsonl: Path to write the output images.jsonl file.
        image_base_url: Base URL for constructing image URLs. If None, uses config.image_base_url.
        meta_json: Optional path to metadata JSON file with license/DOI information.
        metadata_jsonl: Optional path to metadata JSONL file with additional metadata to merge.
            Each row should have image_id plus any additional metadata columns.
        default_license: Default license string if not in metadata.
        default_doi: Default DOI string if not in metadata.
        follow_symlinks: If True, follow symbolic links when scanning directories.
        limit: Maximum number of images to process (0 = no limit).
        skip_percent_paths: If True, skip image paths containing '%' character.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        
    Returns:
        List of row dictionaries written to the JSONL file.
        
    Raises:
        ValueError: If image_base_url is not provided and not in config.
    """
    logger.info(f"[PREPROCESS] Building images.jsonl from {input_dir}...")
    config = config or DEFAULT_BENCHMARK_CONFIG
    image_base_url = image_base_url or config.image_base_url
    if not image_base_url:
        raise ValueError("image_base_url is required to build image URLs.")

    cfg: Optional[Dict[str, Any]] = None
    if meta_json:
        cfg = load_meta_config(meta_json)
    
    # Load metadata JSONL file and create lookup dictionary
    metadata_lookup: Dict[str, Dict[str, Any]] = {}
    if metadata_jsonl and Path(metadata_jsonl).exists():
        logger.info(f"[PREPROCESS] Loading metadata from {metadata_jsonl}...")
        for row in read_jsonl(metadata_jsonl):
            image_id = row.get(config.column_image_id)
            if image_id:
                # Store all columns except image_id as metadata
                metadata_lookup[image_id] = {k: v for k, v in row.items() if k != config.column_image_id}
        logger.info(f"[PREPROCESS] Loaded metadata for {len(metadata_lookup)} image(s)")
    elif metadata_jsonl:
        logger.warning(f"[PREPROCESS] Metadata JSONL file not found: {metadata_jsonl}")

    if default_license is not None or default_doi is not None:
        default_meta = Meta(
            license=(default_license or "UNKNOWN"),
            doi=(default_doi or "UNKNOWN"),
            dataset_name="UNKNOWN",
        )
    elif cfg and cfg.get("default"):
        d = cfg["default"] or {}
        default_meta = Meta(
            license=str(d.get("license", "UNKNOWN")),
            doi=str(d.get("doi", "UNKNOWN")),
            dataset_name=str(d.get("dataset_name", "UNKNOWN")),
        )
    else:
        default_meta = prompt_default_meta()

    rows: List[Dict[str, Any]] = []
    n = 0
    
    # Collect all image paths first for accurate progress tracking
    all_images = list(iter_images(input_dir, follow_symlinks=follow_symlinks))
    total_images = len(all_images) if not limit else min(len(all_images), limit)
    
    with tqdm(total=total_images, desc="Building images.jsonl", unit="image") as pbar:
        for img_path in all_images:
            image_id = posix_relpath(img_path, input_dir)

            if skip_percent_paths and "%" in image_id:
                pbar.update(1)
                continue

            if cfg is not None:
                meta = meta_from_config(image_id, cfg)
                if not meta.license:
                    meta = Meta(license=default_meta.license, doi=meta.doi, dataset_name=meta.dataset_name)
                if not meta.doi:
                    meta = Meta(license=meta.license, doi=default_meta.doi, dataset_name=meta.dataset_name)
            else:
                meta = default_meta

            row = build_row(img_path, input_dir, meta, image_base_url=image_base_url, config=config)
            
            # Merge metadata from metadata_jsonl if available
            if image_id in metadata_lookup:
                row.update(metadata_lookup[image_id])
            
            rows.append(row)
            n += 1
            pbar.update(1)
            pbar.set_postfix({"processed": n, "skipped": pbar.n - n})
            
            if limit and n >= limit:
                break

    write_jsonl(out_jsonl, rows)
    logger.info(f"[PREPROCESS] Wrote {len(rows)} image(s) to {out_jsonl}")
    return rows


def build_seeds_jsonl(
    rows: List[Dict[str, Any]],
    out_seeds_jsonl: Path,
    num_seeds: int,
    seed_prefix: str = "query_",
    config: Optional[BenchmarkConfig] = None,
) -> None:
    """
    Build seeds.jsonl file from image rows.
    
    Creates evenly distributed seed queries from the provided image rows.
    Each seed query contains a single seed image ID.
    
    Args:
        rows: List of image row dictionaries from images.jsonl.
        out_seeds_jsonl: Path to write the output seeds.jsonl file.
        num_seeds: Number of seed queries to create.
        seed_prefix: Prefix for query IDs (default: "query_").
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        
    Raises:
        RuntimeError: If there are fewer images than requested seeds.
    """
    logger.info(f"[PREPROCESS] Building seeds.jsonl from {len(rows)} rows...")
    config = config or DEFAULT_BENCHMARK_CONFIG
    image_ids = [row[config.column_image_id] for row in rows]
    write_seeds_jsonl(
        image_ids=image_ids,
        out_path=out_seeds_jsonl,
        num_seeds=num_seeds,
        seed_prefix=seed_prefix,
        config=config,
    )


def check_image_urls(
    images_jsonl: Optional[Path] = None,
    config: Optional[BenchmarkConfig] = None,
    timeout: int = 10,
    max_workers: int = 10,
) -> Dict[str, Any]:
    """
    Check if all image URLs in images.jsonl are reachable.
    
    Reads images.jsonl and attempts to fetch each image URL to verify accessibility.
    Uses concurrent requests for efficiency.
    
    Args:
        images_jsonl: Path to images.jsonl file. If None, uses config.images_jsonl.
        config: BenchmarkConfig instance. If None, uses DEFAULT_BENCHMARK_CONFIG.
        timeout: Request timeout in seconds (default: 10).
        max_workers: Maximum number of concurrent requests (default: 10).
        
    Returns:
        Dictionary with keys:
            - total_count: Total number of images checked
            - success_count: Number of successfully reachable images
            - failed_count: Number of failed/unreachable images
            - failed_image_ids: List of image IDs that failed
            
    Raises:
        ValueError: If images_jsonl is not provided and not in config.
        FileNotFoundError: If images_jsonl file does not exist.
    """
    import concurrent.futures
    
    config = config or DEFAULT_BENCHMARK_CONFIG
    images_jsonl = Path(images_jsonl) if images_jsonl else (Path(config.images_jsonl) if config.images_jsonl else None)
    
    if images_jsonl is None:
        raise ValueError("images_jsonl must be provided or set in config.images_jsonl")
    if not images_jsonl.exists():
        raise FileNotFoundError(f"Images JSONL file not found: {images_jsonl}")
    
    # Read all image rows
    rows = list(read_jsonl(images_jsonl))
    total_count = len(rows)
    
    if total_count == 0:
        logger.warning("[PREPROCESS] No images found in images.jsonl")
        return {
            "total_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "failed_image_ids": [],
        }
    
    logger.info(f"[PREPROCESS] Checking {total_count} image URLs...")
    
    def check_url(row: Dict[str, Any]) -> Tuple[str, bool]:
        """Check if a single image URL is reachable."""
        image_id = row.get(config.column_image_id, "")
        image_url = row.get(config.image_url_temp_column, "")
        
        if not image_url:
            logger.warning(f"[PREPROCESS] Missing image_url for {image_id}")
            return (image_id, False)
        
        try:
            # Try HEAD first (more efficient), fall back to GET if HEAD is not supported
            try:
                response = requests.head(image_url, timeout=timeout, allow_redirects=True)
            except requests.exceptions.RequestException:
                # If HEAD fails, try GET with stream=True (only download headers)
                response = requests.get(image_url, timeout=timeout, allow_redirects=True, stream=True)
                # Close the connection immediately to avoid downloading the full image
                response.close()
            
            # Accept 2xx and 3xx status codes as success
            is_reachable = 200 <= response.status_code < 400
            if not is_reachable:
                logger.debug(f"[PREPROCESS] Image {image_id} returned status {response.status_code}")
            return (image_id, is_reachable)
        except requests.exceptions.RequestException as e:
            logger.debug(f"[PREPROCESS] Failed to reach {image_id}: {e}")
            return (image_id, False)
    
    # Check URLs concurrently
    success_count = 0
    failed_count = 0
    failed_image_ids = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(check_url, row): row for row in rows}
        
        with tqdm(total=total_count, desc="Checking image URLs", unit="image") as pbar:
            for future in concurrent.futures.as_completed(future_to_row):
                image_id, is_reachable = future.result()
                if is_reachable:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_image_ids.append(image_id)
                pbar.update(1)
                pbar.set_postfix({"success": success_count, "failed": failed_count})
    
    logger.info(f"[PREPROCESS] URL check complete: {success_count}/{total_count} successful, {failed_count} failed")
    
    return {
        "total_count": total_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "failed_image_ids": failed_image_ids,
    }

