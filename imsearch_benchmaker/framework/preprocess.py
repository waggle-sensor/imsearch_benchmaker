"""
preprocess.py

Framework utilities for building images.jsonl and seeds.jsonl.
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
    license: str
    doi: str


def posix_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def guess_mime_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(path.name)
    return mt or "application/octet-stream"


def load_meta_config(meta_json: Path) -> Dict[str, Any]:
    with meta_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def meta_from_config(image_id: str, cfg: Dict[str, Any]) -> Meta:
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
    license_ = input("License (e.g., CC BY 4.0, custom, UNKNOWN): ").strip()
    doi_val = input("DOI (e.g., HPWREN, Kaggle contributor, UNKNOWN): ").strip()
    return Meta(license=license_ or "UNKNOWN", doi=doi_val or "UNKNOWN")


def iter_images(root: Path, follow_symlinks: bool = False) -> Iterable[Path]:
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
    image_id = posix_relpath(image_path, root)
    base = image_base_url.rstrip("/")
    img_id = image_id.lstrip("/")
    image_url = f"{base}/{img_id}"
    return {
        config.image_id_column: image_id,
        config.image_url_column: image_url,
        config.mime_type_column: guess_mime_type(image_path),
        config.license_column: meta.license,
        config.doi_column: meta.doi,
    }


def write_seeds_jsonl(
    image_ids: List[str],
    out_path: Path,
    num_seeds: int,
    seed_prefix: str,
    config: BenchmarkConfig,
) -> None:
    if len(image_ids) < num_seeds:
        raise RuntimeError(f"Not enough images ({len(image_ids)}) to create {num_seeds} seeds")

    image_ids_sorted = sorted(image_ids)
    step = len(image_ids_sorted) / num_seeds

    seeds = []
    for i in range(num_seeds):
        idx = int(i * step)
        seeds.append({
            config.seed_query_id_column: f"{seed_prefix}{i+1:03d}",
            config.seed_image_ids_column: [image_ids_sorted[idx]],
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
    config = config or DEFAULT_BENCHMARK_CONFIG
    image_ids = [row[config.image_id_column] for row in rows]
    write_seeds_jsonl(
        image_ids=image_ids,
        out_path=out_seeds_jsonl,
        num_seeds=num_seeds,
        seed_prefix=seed_prefix,
        config=config,
    )

