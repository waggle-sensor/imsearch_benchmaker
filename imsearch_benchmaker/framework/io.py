"""
io.py

Shared JSONL helpers and batch utilities for benchmark pipelines.

This module provides utilities for reading/writing JSONL files, managing batch references,
and handling batch ID persistence for resumable pipeline execution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class BatchRefs:
    """
    Reference to a batch job, containing input file ID and batch ID.
    
    Used for tracking batch submissions across different adapters. This allows
    the framework to work with adapter-specific batch reference formats while
    maintaining a common interface.
    
    Attributes:
        input_file_id: Identifier for the input file uploaded to the batch service.
        batch_id: Identifier for the batch job submitted to the service.
        input_path: Optional path to the local input/shard file (e.g. vision_shard_0000.jsonl).
    """
    input_file_id: str
    batch_id: str
    input_path: Optional[Path] = None


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Read a JSONL file and yield each line as a dictionary.
    
    Args:
        path: Path to the JSONL file to read.
        
    Yields:
        Dictionary parsed from each non-empty line in the JSONL file.
        
    Raises:
        RuntimeError: If a line contains invalid JSON.
        FileNotFoundError: If the file does not exist.
    """
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {i} of {path}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write dictionaries to a JSONL file.
    
    Creates the parent directory if it doesn't exist. Each dictionary is written
    as a single JSON-encoded line with UTF-8 encoding.
    
    Args:
        path: Path to the JSONL file to write.
        rows: Iterable of dictionaries to write, one per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_failed_ids(error_jsonl: Path, stage: str) -> Set[str]:
    """
    Extract IDs from failed requests in the error JSONL file.
    
    Args:
        error_jsonl: Path to the error JSONL file from batch output.
        stage: Stage identifier (e.g., "vision", "judge") to filter custom_ids.
    
    Returns:
        Set of IDs that failed.
    """
    failed_ids = set()
    for row in read_jsonl(error_jsonl):
        custom_id = row.get("custom_id", "")
        if custom_id.startswith(f"{stage}::"):
            custom_id_id = custom_id.split(f"{stage}::", 1)[1]
            failed_ids.add(custom_id_id)
    return failed_ids


def save_batch_id(batch_id: str, batch_id_file: Path) -> None:
    """
    Save batch ID(s) to a file. Supports single batch ID or comma-separated list.
    
    Args:
        batch_id: Single batch ID or comma-separated list of batch IDs.
        batch_id_file: Path to save the batch ID(s).
    """
    batch_id_file.parent.mkdir(parents=True, exist_ok=True)
    batch_id_file.write_text(batch_id)
    logger.info(f"[IO] Saved batch ID(s) to {batch_id_file}: {batch_id}")


def load_batch_id(batch_id_file: Path) -> str:
    """
    Load batch ID(s) from a file.
    
    Args:
        batch_id_file: Path to the batch ID file.
    
    Returns:
        Batch ID string (may be comma-separated for multiple batches).
    """
    if not batch_id_file.exists():
        raise FileNotFoundError(f"Batch ID file not found: {batch_id_file}")
    return batch_id_file.read_text().strip()


def save_batch_map(entries: List[Dict[str, str]], map_path: Path, merge: bool = True) -> None:
    """
    Save or append input-file -> batch_id mapping for debugging and retries.
    
    Each entry should have "input_file" (e.g. vision_shard_0000.jsonl) and "batch_id".
    If merge is True, appends to existing mapping at map_path; otherwise overwrites.
    
    Args:
        entries: List of {"input_file": str, "batch_id": str}.
        map_path: Path to JSON file (e.g. .vision_batch_map.json).
        merge: If True, load existing entries and extend; otherwise overwrite.
    """
    map_path.parent.mkdir(parents=True, exist_ok=True)
    if merge and map_path.exists():
        try:
            existing = json.loads(map_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                entries = existing + entries
        except (json.JSONDecodeError, OSError):
            pass
    map_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    logger.info(f"[IO] Saved batch map ({len(entries)} entries) to {map_path}")


def load_batch_map(map_path: Path) -> List[Dict[str, str]]:
    """
    Load input-file -> batch_id mapping.
    
    Args:
        map_path: Path to JSON file (e.g. .vision_batch_map.json).
    
    Returns:
        List of {"input_file": str, "batch_id": str}. Empty list if file missing or invalid.
    """
    if not map_path.exists():
        return []
    try:
        data = json.loads(map_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def format_batch_id(batch_ref: Any) -> str:
    """
    Format batch reference(s) into a comma-separated string for saving.
    
    Args:
        batch_ref: Single BatchRefs or list of BatchRefs, or any other object.
    
    Returns:
        Comma-separated batch ID string.
    """
    if isinstance(batch_ref, BatchRefs):
        return batch_ref.batch_id
    elif isinstance(batch_ref, list):
        return ",".join([ref.batch_id for ref in batch_ref if isinstance(ref, BatchRefs)])
    else:
        return str(batch_ref)

