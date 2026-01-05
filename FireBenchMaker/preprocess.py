#!/usr/bin/env python3
"""
preprocess.py

Build images.jsonl and seeds.jsonl for FireBench Batch ingestion.

Output JSONL lines:
- images.jsonl:

  {
    "image_id": "<relative/path/to/file.jpg>",
    "image_url": "<public_url_to_image>",
    "mime_type": "image/jpeg",
    "license": "...",
    "doi": "..."
  }

- seeds.jsonl:
  {
    "query_id": "firebench_q001",
    "seed_image_ids": ["<relative/path/to/file.jpg>"]
  }

License + doi sources:
  A) --meta-json <file.json> : resolve per-folder (or per-file) metadata
  B) --license/--rights-holder : set defaults non-interactively
  C) If neither A nor B provided, prompt interactively once for defaults

Folder metadata JSON formats supported:

1) Simple defaults (applies to all):
{
  "default": { "license": "CC BY 4.0", "doi": "HPWREN" }
}

2) Folder prefixes (longest-prefix match):
{
  "default": { "license": "UNKNOWN", "doi": "UNKNOWN" },
  "prefixes": [
    { "prefix": "hpwren/", "license": "custom", "doi": "HPWREN" },
    { "prefix": "kaggle/", "license": "custom", "doi": "Kaggle Uploader" }
  ]
}

3) Explicit per-file overrides (exact match on image_id):
{
  "default": { "license": "UNKNOWN", "doi": "UNKNOWN" },
  "files": {
    "hpwren/fire1/img0001.jpg": { "license": "custom", "doi": "HPWREN" }
  }
}

Notes:
- image_id is always POSIX relative path (forward slashes) from input root
- By default, skips non-image files based on extension
- One seed image per query, chosen by: Enumerate all images (after walking the folder), Sort by image_id (stable, reproducible), Uniformly sample N = 100 images across the list
    - This guarantees: no randomness unless you want it, seeds don’t cluster in one subfolder, same inputs → same seeds
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from config import IMAGE_BASE_URL
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
# Constants
# -----------------------------
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"
}

# -----------------------------
# Helpers
# -----------------------------

@dataclass(frozen=True)
class Meta:
    """
    Metadata for an image.
    Args:
        license: The license of the image.
        doi: The DOI of the image.
    """
    license: str
    doi: str


def posix_relpath(path: Path, root: Path) -> str:
    """
    Get the POSIX relative path of a file from the root directory.
    Args:
        path: The path of the file.
        root: The root directory.
    Returns:
        The POSIX relative path of the file.
    """
    return path.relative_to(root).as_posix()


def is_image_file(p: Path) -> bool:
    """
    Check if a file is an image file.
    Args:
        p: The path of the file.
    Returns:
        True if the file is an image file, False otherwise.
    """
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def guess_mime_type(p: Path) -> str:
    """
    Guess the MIME type of a file.
    Args:
        p: The path of the file.
    Returns:
        The MIME type of the file.
    """
    mt, _ = mimetypes.guess_type(p.name)
    return mt or "application/octet-stream"


def load_meta_config(meta_json: Path) -> Dict[str, Any]:
    """
    Load the metadata configuration from a JSON file.
    Args:
        meta_json: The path of the JSON file.
    Returns:
        The metadata configuration.
    """
    with meta_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def meta_from_config(image_id: str, cfg: Dict[str, Any]) -> Meta:
    """
    Get the metadata for an image from the configuration.
    Args:
        image_id: The ID of the image.
        cfg: The configuration.
    Returns:
        The metadata for the image.
    """
    # Defaults
    default = cfg.get("default") or {}
    license_ = default.get("license", "")
    doi = default.get("doi", "")

    # Per-file exact override
    files = cfg.get("files") or {}
    if image_id in files:
        rec = files[image_id] or {}
        return Meta(
            license=str(rec.get("license", license_)),
            doi=str(rec.get("doi", doi)),
        )

    # Prefix-based override (longest match)
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
    Prompt the user for default metadata to apply to all images.
    Returns:
        The default metadata.
    """
    logger.info("No --meta-json and no --license/--doi provided.")
    logger.info("Enter default metadata to apply to ALL images.")
    license_ = input("License (e.g., CC BY 4.0, custom, UNKNOWN): ").strip()
    doi_val = input("DOI (e.g., HPWREN, Kaggle contributor, UNKNOWN): ").strip()
    return Meta(license=license_ or "UNKNOWN", doi=doi_val or "UNKNOWN")


def iter_images(root: Path, follow_symlinks: bool = False) -> Iterable[Path]:
    """
    Iterate over all images in the root directory.
    Args:
        root: The root directory.
        follow_symlinks: Whether to follow symlinks.
    Returns:
        An iterable of paths to the images.
    """
    # Path.rglob follows symlinks depending on filesystem; keep it simple and allow opt-out.
    for p in root.rglob("*"):
        if not follow_symlinks and p.is_symlink():
            continue
        if is_image_file(p):
            yield p


def build_row(image_path: Path, root: Path, meta: Meta) -> Dict[str, Any]:
    """
    Build a row for the images.jsonl file.
    Args:
        image_path: The path of the image.
        root: The root directory.
        meta: The metadata for the image.
    Returns:
        A row for the images.jsonl file.
    """
    image_id = posix_relpath(image_path, root)
    # Construct image URL from base URL and image_id
    if IMAGE_BASE_URL:
        # Ensure base URL doesn't end with / and image_id doesn't start with /
        base = IMAGE_BASE_URL.rstrip('/')
        img_id = image_id.lstrip('/')
        image_url = f"{base}/{img_id}"
    else:
        raise ValueError("IMAGE_BASE_URL must be set in config.py or environment variable FIREBENCH_IMAGE_BASE_URL")
    
    return {
        "image_id": image_id,
        "image_url": image_url,
        "mime_type": guess_mime_type(image_path),
        "license": meta.license,
        "doi": meta.doi,
    }


def write_jsonl(out_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    Args:
        out_path: The path to the JSONL file.
        rows: An iterable of dictionaries to write to the file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_seeds_jsonl(
    image_ids: List[str],
    out_path: Path,
    num_seeds: int,
) -> None:
    """
    Write a list of seeds to a JSONL file.
    Args:
        image_ids: The IDs of the images.
        out_path: The path to the JSONL file.
        num_seeds: The number of seeds to write.
    """
    if len(image_ids) < num_seeds:
        raise RuntimeError(f"Not enough images ({len(image_ids)}) to create {num_seeds} seeds")

    # Deterministic: sort, then evenly sample
    image_ids_sorted = sorted(image_ids)
    step = len(image_ids_sorted) / num_seeds

    seeds = []
    for i in range(num_seeds):
        idx = int(i * step)
        seeds.append({
            "query_id": f"firebench_q{i+1:03d}",
            "seed_image_ids": [image_ids_sorted[idx]],
        })

    with out_path.open("w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(seeds)} seeds to {out_path}")

def main() -> None:
    """
    Main function.
    """
    ap = argparse.ArgumentParser("preprocess")
    ap.add_argument("--input-dir", required=True, help="Folder containing images (walked recursively)")
    ap.add_argument("--out-jsonl", required=True, help="Output images.jsonl")
    ap.add_argument("--meta-json", default=None, help="Optional metadata JSON file (default/prefixes/files)")
    ap.add_argument("--license", default=None, help="Default license for all images (overridden by meta-json)")
    ap.add_argument("--doi", default=None, help="Default DOI for all images (overridden by meta-json)")
    ap.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks while scanning")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of images (0 = no limit)")
    ap.add_argument("--out-seeds-jsonl", default=None,
                    help="Optional output seeds.jsonl (if set, seeds will be generated)")
    ap.add_argument("--num-seeds", type=int, default=100,
                    help="Number of queries/seeds to generate (default: 100)")
    args = ap.parse_args()

    root = Path(args.input_dir).resolve()
    out = Path(args.out_jsonl).resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--input-dir must be an existing directory: {root}")

    cfg: Optional[Dict[str, Any]] = None
    if args.meta_json:
        cfg = load_meta_config(Path(args.meta_json).resolve())

    # Determine default meta (used only if cfg missing per-image keys)
    if args.license is not None or args.doi is not None:
        default_meta = Meta(
            license=(args.license or "UNKNOWN"),
            doi=(args.doi or "UNKNOWN"),
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
    skipped = 0
    for img_path in iter_images(root, follow_symlinks=args.follow_symlinks):
        image_id = posix_relpath(img_path, root)
        
        # Skip images with % character in image_id (can cause URL encoding issues)
        if "%" in image_id:
            logger.warning(f"Skipping image with '%' can cause URL encoding issues: {image_id}")
            skipped += 1
            continue

        if cfg is not None:
            meta = meta_from_config(image_id, cfg)
            # If cfg omits values, fall back to default_meta
            if not meta.license:
                meta = Meta(license=default_meta.license, doi=meta.doi)
            if not meta.doi:
                meta = Meta(license=meta.license, doi=default_meta.doi)
        else:
            meta = default_meta

        rows.append(build_row(img_path, root, meta))
        n += 1
        if args.limit and n >= args.limit:
            break
        if n % 250 == 0:
            logger.info(f"Processed {n} images...")

    write_jsonl(out, rows)
    logger.info(f"Wrote {len(rows)} lines to {out}")
    if skipped > 0:
        logger.warning(f"Skipped {skipped} images with '%' character in path")

    if args.out_seeds_jsonl:
        image_ids = [r["image_id"] for r in rows]
        write_seeds_jsonl(
            image_ids=image_ids,
            out_path=Path(args.out_seeds_jsonl),
            num_seeds=args.num_seeds,
        )
        logger.info(f"Wrote {args.num_seeds} seeds to {args.out_seeds_jsonl}")


if __name__ == "__main__":
    """
    Main function.
    """
    main()
