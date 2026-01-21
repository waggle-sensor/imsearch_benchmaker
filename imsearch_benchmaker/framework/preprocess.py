"""
preprocess.py

Framework utilities for building images.jsonl and seeds.jsonl.

This module provides functions to preprocess image directories into JSONL format,
extracting image metadata, constructing image URLs, and creating seed image lists
for query planning.
"""

from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io import write_jsonl
from .config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"
}


@dataclass(frozen=True)
class Meta:
    """
    Image metadata container for license and DOI information.
    
    Attributes:
        license: License string for the image (e.g., "CC BY 4.0").
        doi: Digital Object Identifier or source identifier.
    """
    license: str
    doi: str


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
        Meta object with license and DOI for the image.
    """
    default = cfg.get("default") or {}
    license_ = default.get("license", "")
    doi = default.get("doi", "")

    files = cfg.get("files") or {}
    if image_id in files:
        rec = files[image_id] or {}
        return Meta(
            license=str(rec.get("license", license_)),
            doi=str(rec.get("doi", doi)),
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
        )

    return Meta(license=str(license_), doi=str(doi))


def prompt_default_meta() -> Meta:
    """
    Prompt user for default metadata via command line input.
    
    Returns:
        Meta object with user-provided license and DOI, or "UNKNOWN" if empty.
    """
    license_ = input("License (e.g., CC BY 4.0, custom, UNKNOWN): ").strip()
    doi_val = input("DOI (e.g., HPWREN, Kaggle contributor, UNKNOWN): ").strip()
    return Meta(license=license_ or "UNKNOWN", doi=doi_val or "UNKNOWN")


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
        meta: Metadata object containing license and DOI.
        image_base_url: Base URL for constructing image URLs.
        config: BenchmarkConfig instance for column names.
        
    Returns:
        Dictionary with image_id, image_url, mime_type, license, and doi columns.
    """
    image_id = posix_relpath(image_path, root)
    base = image_base_url.rstrip("/")
    img_id = image_id.lstrip("/")
    image_url = f"{base}/{img_id}"
    return {
        config.column_image_id: image_id,
        config.image_url_temp_column: image_url,
        config.column_mime_type: guess_mime_type(image_path),
        config.column_license: meta.license,
        config.column_doi: meta.doi,
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
    for i in range(num_seeds):
        idx = int(i * step)
        seeds.append({
            config.column_query_id: f"{seed_prefix}{i+1:03d}",
            config.query_plan_seed_image_ids_column: [image_ids_sorted[idx]],
        })

    with out_path.open("w", encoding="utf-8") as f:
        for seed in seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")


def build_images_jsonl(
    input_dir: Path,
    out_jsonl: Path,
    image_base_url: Optional[str] = None,
    meta_json: Optional[Path] = None,
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
    and writes the results to a JSONL file. Supports metadata from JSON config files
    or interactive prompts.
    
    Args:
        input_dir: Directory containing image files to process.
        out_jsonl: Path to write the output images.jsonl file.
        image_base_url: Base URL for constructing image URLs. If None, uses config.image_base_url.
        meta_json: Optional path to metadata JSON file with license/DOI information.
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
    config = config or DEFAULT_BENCHMARK_CONFIG
    image_base_url = image_base_url or config.image_base_url
    if not image_base_url:
        raise ValueError("image_base_url is required to build image URLs.")

    cfg: Optional[Dict[str, Any]] = None
    if meta_json:
        cfg = load_meta_config(meta_json)

    if default_license is not None or default_doi is not None:
        default_meta = Meta(
            license=(default_license or "UNKNOWN"),
            doi=(default_doi or "UNKNOWN"),
        )
    elif cfg and cfg.get("default"):
        d = cfg["default"] or {}
        default_meta = Meta(
            license=str(d.get("license", "UNKNOWN")),
            doi=str(d.get("doi", "UNKNOWN")),
        )
    else:
        default_meta = prompt_default_meta()

    rows: List[Dict[str, Any]] = []
    n = 0
    for img_path in iter_images(input_dir, follow_symlinks=follow_symlinks):
        image_id = posix_relpath(img_path, input_dir)

        if skip_percent_paths and "%" in image_id:
            continue

        if cfg is not None:
            meta = meta_from_config(image_id, cfg)
            if not meta.license:
                meta = Meta(license=default_meta.license, doi=meta.doi)
            if not meta.doi:
                meta = Meta(license=meta.license, doi=default_meta.doi)
        else:
            meta = default_meta

        rows.append(build_row(img_path, input_dir, meta, image_base_url=image_base_url, config=config))
        n += 1
        if limit and n >= limit:
            break

    write_jsonl(out_jsonl, rows)
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
    config = config or DEFAULT_BENCHMARK_CONFIG
    image_ids = [row[config.column_image_id] for row in rows]
    write_seeds_jsonl(
        image_ids=image_ids,
        out_path=out_seeds_jsonl,
        num_seeds=num_seeds,
        seed_prefix=seed_prefix,
        config=config,
    )

