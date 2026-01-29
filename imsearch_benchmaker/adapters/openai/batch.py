"""
batch.py

OpenAI Batch API helpers for submitting, polling, and managing batch jobs.

This module provides utilities for working with OpenAI's Batch API, including
batch submission, sharding large batches, waiting for completion with progress
indicators, downloading results, and listing batches.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from ...framework.io import read_jsonl, write_jsonl, BatchRefs


def shard_batch_jsonl(
    input_jsonl: Path,
    output_dir: Path,
    max_items_per_shard: int,
    shard_prefix: str,
) -> List[Path]:
    """
    Split a large JSONL file into smaller shards.
    
    Reads the input JSONL file and splits it into multiple shard files,
    each containing at most max_items_per_shard items. Shard files are
    named with the pattern {shard_prefix}_{shard_num:04d}.jsonl.
    
    Args:
        input_jsonl: Path to the input JSONL file to shard.
        output_dir: Directory to write shard files to.
        max_items_per_shard: Maximum number of items per shard.
        shard_prefix: Prefix for shard filenames.
        
    Returns:
        List of paths to created shard files.
    """
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
    """
    Submit a single batch job to OpenAI Batch API.
    
    Uploads the input JSONL file, creates a batch job, and returns
    references to the uploaded file and batch job.
    
    Args:
        client: OpenAI client instance.
        input_jsonl: Path to the batch input JSONL file.
        completion_window: Completion window for the batch (e.g., "24h").
        metadata: Optional metadata dictionary to attach to the batch.
        
    Returns:
        BatchRefs object containing input_file_id and batch_id.
    """
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
    """
    Submit multiple batch shards with concurrency control.
    
    Submits multiple batch shards while respecting the max_concurrent limit.
    Monitors in-flight batches and waits for completion before submitting new ones.
    
    Args:
        client: OpenAI client instance.
        shard_paths: List of paths to shard JSONL files to submit.
        completion_window: Completion window for batches (e.g., "24h").
        metadata: Optional metadata dictionary to attach to batches.
        max_concurrent: Maximum number of batches to submit concurrently.
        
    Returns:
        List of BatchRefs objects for all submitted batches.
    """
    all_refs: List[BatchRefs] = []
    in_flight: List[BatchRefs] = []

    with tqdm(total=len(shard_paths), desc="Submitting batch shards", unit="shard") as pbar:
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
                    pbar.set_postfix({"in_flight": len(in_flight), "waiting": "..."})
                    time.sleep(60)

            refs = submit_batch(client, shard_path, completion_window, metadata)
            all_refs.append(refs)
            in_flight.append(refs)
            pbar.update(1)
            pbar.set_postfix({"submitted": len(all_refs), "in_flight": len(in_flight)})

    return all_refs


def wait_for_batch(client: OpenAI, batch_id: str, poll_s: int = 60) -> Dict[str, object]:
    """
    Wait for a batch to complete, with progress indicators using tqdm.
    
    Args:
        client: OpenAI client instance
        batch_id: Batch ID to wait for
        poll_s: Polling interval in seconds
        
    Returns:
        Batch status dictionary
    """
    start_time = time.time()
    poll_count = 0
    
    # Initialize progress bar - start with unknown total, will update when we get request counts
    pbar = tqdm(
        desc=f"Batch {batch_id[:12]}...",
        unit="requests",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
        dynamic_ncols=True,
    )
    
    try:
        while True:
            b = client.batches.retrieve(batch_id)
            status = b.status
            poll_count += 1
            elapsed = int(time.time() - start_time)
            
            # Update progress bar with status information
            postfix_dict = {"status": status}
            
            if hasattr(b, "request_counts") and b.request_counts:
                request_counts = b.request_counts.model_dump() if hasattr(b.request_counts, "model_dump") else dict(b.request_counts)
                total = request_counts.get("total", 0)
                completed = request_counts.get("completed", 0)
                failed = request_counts.get("failed", 0)
                
                if total > 0:
                    # Update progress bar with actual request counts
                    pbar.total = total
                    pbar.n = completed
                    pbar.refresh()
                    
                    progress_pct = int((completed / total) * 100)
                    postfix_dict["progress"] = f"{progress_pct}%"
                    if failed > 0:
                        postfix_dict["failed"] = failed
                else:
                    # No request counts yet, just show polling
                    pbar.total = None
                    pbar.n = poll_count
                    pbar.refresh()
            else:
                # No request counts available
                pbar.total = None
                pbar.n = poll_count
                pbar.refresh()
            
            # Format postfix string
            postfix_str = " | ".join([f"{k}: {v}" for k, v in postfix_dict.items()])
            pbar.set_postfix_str(postfix_str)
            
            if status in ("completed", "failed", "expired", "canceled"):
                elapsed_total = int(time.time() - start_time)
                # Final update
                if hasattr(b, "request_counts") and b.request_counts:
                    request_counts = b.request_counts.model_dump() if hasattr(b.request_counts, "model_dump") else dict(b.request_counts)
                    total = request_counts.get("total", 0)
                    completed = request_counts.get("completed", 0)
                    if total > 0:
                        pbar.n = completed
                        pbar.total = total
                pbar.set_postfix_str(f"status: {status} | {elapsed_total}s")
                pbar.close()
                return b.model_dump() if hasattr(b, "model_dump") else dict(b)
            
            time.sleep(poll_s)
    except KeyboardInterrupt:
        pbar.close()
        raise
    except Exception as e:
        pbar.close()
        raise

def wait_for_batches(client: OpenAI, batch_refs: List[BatchRefs], poll_s: int = 60) -> List[Dict[str, object]]:
    """
    Wait for a list of batches to complete in parallel, showing progress indicators for each.
    
    Args:
        client: OpenAI client instance
        batch_refs: List of BatchRefs to wait for
        poll_s: Polling interval in seconds
        
    Returns:
        List of batch status dictionaries, in the same order as batch_refs
    """
    if not batch_refs:
        return []
    
    if len(batch_refs) == 1:
        # Single batch, just use the regular function
        return [wait_for_batch(client, batch_refs[0].batch_id, poll_s)]
    
    # Multiple batches - wait in parallel using threads
    results: List[Optional[Dict[str, object]]] = [None] * len(batch_refs)
    threads: List[threading.Thread] = []
    exceptions: List[Exception] = []
    
    def wait_single_batch(index: int, batch_ref: BatchRefs) -> None:
        """Wait for a single batch and store the result."""
        try:
            result = wait_for_batch(client, batch_ref.batch_id, poll_s)
            results[index] = result
        except Exception as e:
            exceptions.append(e)
            results[index] = None
    
    # Start a thread for each batch
    for i, batch_ref in enumerate(batch_refs):
        thread = threading.Thread(target=wait_single_batch, args=(i, batch_ref))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check for exceptions
    if exceptions:
        raise RuntimeError(f"Errors occurred while waiting for batches: {exceptions}")
    
    # Verify all results are present
    if any(r is None for r in results):
        raise RuntimeError("Some batches did not complete successfully")
    
    return results  # type: ignore

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

