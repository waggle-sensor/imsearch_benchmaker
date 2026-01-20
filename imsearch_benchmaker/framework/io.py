"""
io.py

Shared JSONL helpers and batch utilities for benchmark pipelines.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Set

logger = logging.getLogger(__name__)


@dataclass
class BatchRefs:
    """
    Reference to a batch job, containing input file ID and batch ID.
    Used for tracking batch submissions across different adapters.
    """
    input_file_id: str
    batch_id: str


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Read a JSONL file and yield each line as a dictionary.
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
    logger.info(f"Saved batch ID(s) to {batch_id_file}: {batch_id}")


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

