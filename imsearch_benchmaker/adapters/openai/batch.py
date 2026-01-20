"""
batch.py

OpenAI Batch API helpers (submit, poll).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ...framework.io import read_jsonl, write_jsonl, BatchRefs


def shard_batch_jsonl(
    input_jsonl: Path,
    output_dir: Path,
    max_items_per_shard: int,
    shard_prefix: str,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    current_shard = []
    shard_num = 0

    for row in read_jsonl(input_jsonl):
        current_shard.append(row)
        if len(current_shard) >= max_items_per_shard:
            shard_path = output_dir / f"{shard_prefix}_{shard_num:04d}.jsonl"
            write_jsonl(shard_path, current_shard)
            shard_paths.append(shard_path)
            current_shard = []
            shard_num += 1

    if current_shard:
        shard_path = output_dir / f"{shard_prefix}_{shard_num:04d}.jsonl"
        write_jsonl(shard_path, current_shard)
        shard_paths.append(shard_path)

    return shard_paths


def submit_batch(
    client: OpenAI,
    input_jsonl: Path,
    completion_window: Optional[str],
    metadata: Optional[Dict[str, str]] = None,
) -> BatchRefs:
    up = client.files.create(file=input_jsonl.open("rb"), purpose="batch")
    input_file_id = up.id

    b = client.batches.create(
        input_file_id=input_file_id,
        completion_window=completion_window,
        metadata=metadata or {},
        endpoint="/v1/responses",
    )
    return BatchRefs(input_file_id=input_file_id, batch_id=b.id)


def submit_batch_shards(
    client: OpenAI,
    shard_paths: List[Path],
    completion_window: Optional[str],
    metadata: Optional[Dict[str, str]] = None,
    max_concurrent: int = 1,
) -> List[BatchRefs]:
    all_refs: List[BatchRefs] = []
    in_flight: List[BatchRefs] = []

    for shard_path in shard_paths:
        while len(in_flight) >= max_concurrent:
            completed = []
            for ref in in_flight:
                try:
                    b = client.batches.retrieve(ref.batch_id)
                    if b.status in ("completed", "failed", "expired", "canceled"):
                        completed.append(ref)
                except Exception:
                    pass
            in_flight = [ref for ref in in_flight if ref not in completed]
            if len(in_flight) >= max_concurrent:
                time.sleep(60)

        refs = submit_batch(client, shard_path, completion_window, metadata)
        all_refs.append(refs)
        in_flight.append(refs)

    return all_refs


def wait_for_batch(client: OpenAI, batch_id: str, poll_s: int = 60) -> Dict[str, object]:
    while True:
        b = client.batches.retrieve(batch_id)
        status = b.status
        if status in ("completed", "failed", "expired", "canceled"):
            return b.model_dump() if hasattr(b, "model_dump") else dict(b)
        time.sleep(poll_s)


def list_batches(client: OpenAI, active_only: bool = False, limit: int = 50, stage: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List OpenAI batches.
    
    Args:
        client: OpenAI client instance
        active_only: If True, only return active batches
        limit: Maximum number of batches to return
        stage: If provided, filter batches by stage ("vision" or "judge")
        
    Returns:
        List of batch dictionaries with id, status, endpoint, created_at, metadata, and request_counts
    """
    batches = client.batches.list(limit=limit)
    
    ACTIVE = {"validating", "in_progress", "finalizing"}
    
    result = []
    for b in batches.data:
        if active_only and b.status not in ACTIVE:
            continue
        
        # Convert SDK/Pydantic objects to plain dicts
        if hasattr(b, "model_dump"):
            request_counts = b.request_counts.model_dump() if b.request_counts else None
            metadata = b.metadata.model_dump() if hasattr(b.metadata, "model_dump") else b.metadata
        else:
            # Fallback (older SDK)
            request_counts = dict(b.request_counts) if b.request_counts else None
            metadata = dict(b.metadata) if b.metadata else None
        
        # Filter by stage if specified
        if stage is not None:
            batch_stage = metadata.get("stage") if isinstance(metadata, dict) else None
            if batch_stage != stage:
                continue
        
        result.append({
            "id": b.id,
            "status": b.status,
            "endpoint": b.endpoint,
            "created_at": b.created_at,
            "metadata": metadata,
            "request_counts": request_counts,
        })
    
    return result

