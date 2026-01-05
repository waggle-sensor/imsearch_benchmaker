#!/usr/bin/env python3
"""
FireBench Batch Tool (OpenAI Batch API)

Two-phase pipeline:
  A) Vision annotation (image -> summary + facet labels) [expensive, do once]
  B) Query + relevance labeling (text-only)              [cheap, repeatable]

Input formats:
  - Images input JSONL: each line is:
      {"image_id": "...", "image_url": "...", "mime_type": "image/jpeg",
       "license": "...", "doi": "..."}
    *image_url must be a publicly accessible URL to the image

  - Query-plan JSONL (for phase B): each line is:
      {"query_id":"firebench_q001",
       "seed_image_ids":["img_...","img_..."],
       "candidate_image_ids":["img_...","img_..."]}

  - Annotations JSONL (output of phase A parse): each line is:
      {"image_id":"...", "summary":"...", "viewpoint":"...", ... "license":"...", "doi":"..."}

Notes:
  - Batch input files must be uploaded with purpose="batch" and are .jsonl.
  - Images can be provided as public URLs in Responses API.
  - Batch input files must be ≤ 200MB.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from openai import OpenAI
from config import (
    VISION_MODEL, 
    TEXT_MODEL, 
    COMPLETION_WINDOW, 
    IMAGE_DETAIL, 
    MAX_CANDIDATES, 
    VIEWPOINT, 
    PLUME_STAGE, 
    LIGHTING, 
    CONFOUNDER, 
    ENVIRONMENT, 
    CONTROLLED_TAG_VOCAB, 
    VISON_ANNOTATION_SYSTEM_PROMPT, 
    VISION_ANNOTATION_USER_PROMPT, 
    JUDGE_SYSTEM_PROMPT, 
    JUDGE_USER_PROMPT,
    VISION_ANNOTATION_MAX_OUTPUT_TOKENS,
    JUDGE_MAX_OUTPUT_TOKENS,
    VISION_ANNOTATION_REASONING_EFFORT,
    JUDGE_REASONING_EFFORT,
    MAX_IMAGES_PER_BATCH,
    MAX_QUERIES_PER_BATCH,
    MAX_CONCURRENT_BATCHES,
)

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)
# Suppress verbose OpenAI library logging (retries, etc.)
logging.getLogger("openai").setLevel(logging.WARNING)

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
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {i} of {path}: {e}") from e


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




def must_get(d: Dict[str, Any], k: str) -> Any:
    """
    Get a value from a dictionary, raising an error if the key is not present.
    Args:
        d: The dictionary to get the value from.
        k: The key to get the value from.
    Returns:
        The value from the dictionary.
    """
    if k not in d:
        raise KeyError(f"Missing required key '{k}' in: {list(d.keys())}")
    return d[k]


def clamp_candidates(ids: List[str], max_n: int) -> List[str]:
    """
    Clamp the number of candidates to a maximum.
    Args:
        ids: A list of image IDs.
        max_n: The maximum number of image IDs to return.
    Returns:
        A list of image IDs, clamped to the maximum number of image IDs.
    """
    if len(ids) <= max_n:
        return ids
    return ids[:max_n]


# -----------------------------
# Prompts + Schemas
# -----------------------------

def vision_system_prompt() -> str:
    """
    Get the system prompt for the vision model.
    Returns:
        A string of the system prompt.
    """
    return VISON_ANNOTATION_SYSTEM_PROMPT


def vision_user_prompt() -> str:
    """
    Get the user prompt for the vision model.
    Returns:
        A string of the user prompt.
    """
    return VISION_ANNOTATION_USER_PROMPT


def vision_json_schema() -> Dict[str, Any]:
    """
    Get the JSON schema object for the vision model.
    Returns:
        A dictionary containing only the JSON Schema object (no wrapper).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "viewpoint": {"type": "string", "enum": VIEWPOINT},
            "plume_stage": {"type": "string", "enum": PLUME_STAGE},
            "flame_visible": {"type": "boolean"},
            "lighting": {"type": "string", "enum": LIGHTING},
            "confounder_type": {"type": "string", "enum": CONFOUNDER},
            "environment_type": {"type": "string", "enum": ENVIRONMENT},

            "tags": {
                "type": "array",
                "minItems": 14,
                "maxItems": 25,
                "items": {"type": "string", "enum": CONTROLLED_TAG_VOCAB},
            },

            "confidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "viewpoint": {"type": "number", "minimum": 0, "maximum": 1},
                    "plume_stage": {"type": "number", "minimum": 0, "maximum": 1},
                    "confounder_type": {"type": "number", "minimum": 0, "maximum": 1},
                    "environment_type": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["viewpoint", "plume_stage", "confounder_type", "environment_type"],
            },
        },
        "required": [
            "summary", "viewpoint", "plume_stage", "flame_visible",
            "lighting", "confounder_type", "environment_type",
            "tags",
            "confidence"
        ],
    }


def judge_system_prompt() -> str:
    """
    Get the system prompt for the judge model.
    Returns:
        A string of the system prompt.
    """
    return JUDGE_SYSTEM_PROMPT


def judge_user_prompt() -> str:
    """
    Get the user prompt for the judge model.
    Returns:
        A string of the user prompt.
    """
    return JUDGE_USER_PROMPT


def judge_json_schema() -> Dict[str, Any]:
    """
    Get the JSON schema object for the judge model.
    Returns:
        A dictionary containing only the JSON Schema object (no wrapper).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query_text": {"type": "string"},
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "image_id": {"type": "string"},
                        "relevance_label": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["image_id", "relevance_label"],
                },
            },
        },
        "required": ["query_text", "judgments"],
    }


# -----------------------------
# Batch JSONL Builders
# -----------------------------

def extract_failed_image_ids(error_jsonl: Path) -> Set[str]:
    """
    Extract image IDs from failed vision requests in the error JSONL file.
    Args:
        error_jsonl: Path to the error JSONL file from batch output.
    Returns:
        Set of image IDs that failed.
    """
    failed_ids = set()
    for row in read_jsonl(error_jsonl):
        custom_id = row.get("custom_id", "")
        if custom_id.startswith("vision::"):
            image_id = custom_id.split("vision::", 1)[1]
            failed_ids.add(image_id)
    return failed_ids


def extract_failed_query_ids(error_jsonl: Path) -> Set[str]:
    """
    Extract query IDs from failed judge requests in the error JSONL file.
    Args:
        error_jsonl: Path to the error JSONL file from batch output.
    Returns:
        Set of query IDs that failed.
    """
    failed_ids = set()
    for row in read_jsonl(error_jsonl):
        custom_id = row.get("custom_id", "")
        if custom_id.startswith("judge::"):
            query_id = custom_id.split("judge::", 1)[1]
            failed_ids.add(query_id)
    return failed_ids


def build_vision_batch_lines(images_jsonl: Path, image_ids_filter: Optional[set[str]] = None) -> Iterable[Dict[str, Any]]:
    """
    Each line becomes a Responses API request with one image.
    Args:
        images_jsonl: The path to the images JSONL file.
        image_ids_filter: Optional set of image IDs to include (if None, includes all).
    Returns:
        An iterable of dictionaries, each containing a Responses API request.
    """
    for row in read_jsonl(images_jsonl):
        image_id = must_get(row, "image_id")
        
        # Filter by image_ids_filter if provided
        if image_ids_filter is not None and image_id not in image_ids_filter:
            continue
        image_id = must_get(row, "image_id")
        image_url = must_get(row, "image_url")

        license_ = row.get("license")
        doi = row.get("doi")

        body = {
            "model": VISION_MODEL,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": vision_system_prompt()}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": vision_user_prompt()},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": IMAGE_DETAIL,
                        },
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "firebench_vision_annotation",
                    "strict": True,
                    "schema": vision_json_schema(),
                }
            },
            "reasoning": {"effort": VISION_ANNOTATION_REASONING_EFFORT},
            "max_output_tokens": VISION_ANNOTATION_MAX_OUTPUT_TOKENS,
            # Carry-through fields for easier parsing/debugging
            "metadata": {
                "image_id": str(image_id),
                "license": "" if license_ is None else str(license_),
                "doi": "" if doi is None else str(doi),
            },
        }

        yield {
            "custom_id": f"vision::{image_id}",
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }


def build_judge_batch_lines(
    query_plan_jsonl: Path,
    annotations_jsonl: Path,
    query_ids_filter: Optional[Set[str]] = None,
) -> Iterable[Dict[str, Any]]:
    """
    For each query, create one text-only request that returns:
      - query_text
      - judgments[]: {image_id, relevance_label}
    Args:
        query_plan_jsonl: The path to the query plan JSONL file.
        annotations_jsonl: The path to the annotations JSONL file.
        query_ids_filter: Optional set of query IDs to include (if None, includes all).
    Returns:
        An iterable of dictionaries, each containing a Responses API request.
    """
    # Load annotations into memory (10k rows is fine)
    ann: Dict[str, Dict[str, Any]] = {}
    for a in read_jsonl(annotations_jsonl):
        ann[must_get(a, "image_id")] = a

    for q in read_jsonl(query_plan_jsonl):
        query_id = must_get(q, "query_id")
        
        # Filter by query_ids_filter if provided
        if query_ids_filter is not None and query_id not in query_ids_filter:
            continue
        query_id = must_get(q, "query_id")
        seed_ids = list(must_get(q, "seed_image_ids"))
        cand_ids = clamp_candidates(list(must_get(q, "candidate_image_ids")), MAX_CANDIDATES)

        # Build compact seed + candidate payloads (text-only)
        seeds = []
        for sid in seed_ids:
            if sid not in ann:
                raise KeyError(f"seed_image_id {sid} not found in annotations")
            a = ann[sid]
            seeds.append({
                "image_id": sid,
                "summary": a["summary"],
                "viewpoint": a["viewpoint"],
                "plume_stage": a["plume_stage"],
                "flame_visible": a["flame_visible"],
                "lighting": a["lighting"],
                "confounder_type": a["confounder_type"],
                "environment_type": a["environment_type"],
                "tags": sorted(list(a.get("tags") or [])),
            })

        candidates = []
        for cid in cand_ids:
            if cid not in ann:
                raise KeyError(f"candidate_image_id {cid} not found in annotations")
            a = ann[cid]
            candidates.append({
                "image_id": cid,
                "summary": a["summary"],
                "viewpoint": a["viewpoint"],
                "plume_stage": a["plume_stage"],
                "flame_visible": a["flame_visible"],
                "lighting": a["lighting"],
                "confounder_type": a["confounder_type"],
                "environment_type": a["environment_type"],
                "tags": sorted(list(a.get("tags") or [])),
            })

        payload = {
            "query_id": query_id,
            "seed_images": seeds,
            "candidates": candidates,
        }

        body = {
            "model": TEXT_MODEL,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": judge_system_prompt()}]},
                {"role": "user", "content": [{"type": "input_text", "text": judge_user_prompt()}]},
                {"role": "user", "content": [{"type": "input_text", "text": f"DATA (JSON):\n{json.dumps(payload, ensure_ascii=False)}"}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "firebench_query_and_judgments",
                    "strict": True,
                    "schema": judge_json_schema(),
                }
            },
            "reasoning": {"effort": JUDGE_REASONING_EFFORT},
            "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
            "metadata": {"query_id": str(query_id)},
        }

        yield {
            "custom_id": f"judge::{query_id}",
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }


# -----------------------------
# Batch submit / poll / download
# -----------------------------

@dataclass
class BatchRefs:
    """
    A class to store the input file ID and batch ID.
    Args:
        input_file_id: The ID of the input file.
        batch_id: The ID of the batch.
    """
    input_file_id: str
    batch_id: str


def shard_batch_jsonl(input_jsonl: Path, output_dir: Path, max_items_per_shard: int, shard_prefix: str) -> List[Path]:
    """
    Split a large batch JSONL file into smaller shards.
    Args:
        input_jsonl: The input batch JSONL file to shard.
        output_dir: Directory to write shard files.
        max_items_per_shard: Maximum number of items per shard.
        shard_prefix: Prefix for shard filenames (e.g., "vision_shard").
    Returns:
        List of paths to shard files.
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
            logger.info(f"Created shard {shard_num}: {shard_path} ({len(current_shard)} items)")
            current_shard = []
            shard_num += 1
    
    # Write final shard if there are remaining items
    if current_shard:
        shard_path = output_dir / f"{shard_prefix}_{shard_num:04d}.jsonl"
        write_jsonl(shard_path, current_shard)
        shard_paths.append(shard_path)
        logger.info(f"Created shard {shard_num}: {shard_path} ({len(current_shard)} items)")
    
    logger.info(f"Split {input_jsonl} into {len(shard_paths)} shards")
    return shard_paths


def submit_batch_shards(
    client: OpenAI,
    shard_paths: List[Path],
    metadata: Optional[Dict[str, str]] = None,
    max_concurrent: int = MAX_CONCURRENT_BATCHES,
) -> List[BatchRefs]:
    """
    Submit multiple batch shards with concurrency control.
    Submits batches up to max_concurrent, then waits for some to complete before submitting more.
    Returns immediately after all batches are submitted (does not wait for completion).
    
    Args:
        client: The OpenAI client.
        shard_paths: List of shard file paths to submit.
        metadata: The metadata to add to each batch.
        max_concurrent: Maximum number of batches to keep in flight.
    Returns:
        List of BatchRefs for all submitted batches.
    """
    all_refs = []
    in_flight = []
    
    for i, shard_path in enumerate(shard_paths):
        # Wait if we've hit the concurrency limit
        while len(in_flight) >= max_concurrent:
            # Check status of in-flight batches
            completed = []
            for ref in in_flight:
                try:
                    b = client.batches.retrieve(ref.batch_id)
                    if b.status in ("completed", "failed", "expired", "canceled"):
                        completed.append(ref)
                        logger.info(f"Batch {ref.batch_id} finished with status: {b.status}")
                except Exception as e:
                    logger.warning(f"Error checking batch {ref.batch_id}: {e}")
            
            # Remove completed batches from in-flight
            in_flight = [ref for ref in in_flight if ref not in completed]
            
            if len(in_flight) >= max_concurrent:
                time.sleep(60)  # Wait before checking again
        
        # Submit this shard
        logger.info(f"Submitting shard {i+1}/{len(shard_paths)}: {shard_path.name}")
        refs = submit_batch(client, shard_path, metadata)
        all_refs.append(refs)
        in_flight.append(refs)
        logger.info(f"Submitted shard {i+1}/{len(shard_paths)}: batch_id={refs.batch_id}")
    
    logger.info(f"All {len(shard_paths)} shards submitted. {len(in_flight)} batches currently in flight.")
    logger.info("Use 'wait' command to monitor batch completion.")
    return all_refs


def submit_batch(client: OpenAI, input_jsonl: Path, metadata: Optional[Dict[str, str]] = None) -> BatchRefs:
    """
    Submit a batch to the OpenAI API.
    Args:
        client: The OpenAI client.
        input_jsonl: The path to the input JSONL file.
        metadata: The metadata to add to the batch.
    Returns:
        A BatchRefs object containing the input file ID and batch ID.
    """
    # Upload JSONL with purpose=batch
    up = client.files.create(file=input_jsonl.open("rb"), purpose="batch")
    input_file_id = up.id

    # Create batch - endpoint specifies which API the batch calls
    # We use /v1/responses since our batch requests target /v1/responses
    b = client.batches.create(
        input_file_id=input_file_id,
        completion_window=COMPLETION_WINDOW,
        metadata=metadata or {},
        endpoint="/v1/responses",
    )
    return BatchRefs(input_file_id=input_file_id, batch_id=b.id)


def wait_for_batch(client: OpenAI, batch_id: str, poll_s: int = 60) -> Dict[str, Any]:
    """
    Wait for a batch to complete.
    Args:
        client: The OpenAI client.
        batch_id: The ID of the batch.
        poll_s: The number of seconds to poll.
    Returns:
        A dictionary containing the batch status.
    """
    while True:
        b = client.batches.retrieve(batch_id)
        status = b.status
        if status in ("completed", "failed", "expired", "canceled"):
            return b.model_dump() if hasattr(b, "model_dump") else dict(b)
        time.sleep(poll_s)


def download_file(client: OpenAI, file_id: str, out_path: Path) -> None:
    """
    Download a file from the OpenAI API.
    Args:
        client: The OpenAI client.
        file_id: The ID of the file.
        out_path: The path to the output file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    # SDK returns a response-like object; handle bytes
    data = content.read() if hasattr(content, "read") else content
    out_path.write_bytes(data)


# -----------------------------
# Output parsing
# -----------------------------

def parse_batch_output_responses(output_jsonl: Path) -> Iterable[Dict[str, Any]]:
    """
    Each line in output is a batch result. We pull:
      - custom_id
      - response.body (Responses API payload) or error
    Args:
        output_jsonl: The path to the output JSONL file.
    Returns:
        An iterable of dictionaries, each containing the custom ID, response body, error (if any), and raw row.
    """
    for row in read_jsonl(output_jsonl):
        custom_id = row.get("custom_id")
        error = row.get("error")
        resp = row.get("response", {})
        body = resp.get("body", {})
        yield {"custom_id": custom_id, "body": body, "error": error, "raw": row}


def extract_parsed_json_from_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tries to get structured output text as JSON.
    Args:
        body: The body of the response.
    Returns:
        A dictionary containing the parsed JSON.
    """
    # Primary: output_text (SDKs often expose body["output_text"] or similar)
    output_text = body.get("output_text")
    if output_text:
        return json.loads(output_text.strip())

    # Fallback: walk nested structure to find output_text blocks
    # Sometimes: body["output"] -> message -> content -> {type:"output_text", text:"..."}
    # Sometimes: body["output"] -> content -> {type:"output_text", output_text:"..."}
    for item in body.get("output", []) or []:
        # Check if item has content directly
        content_list = item.get("content", [])
        if not content_list and isinstance(item, dict):
            # Maybe item itself is a content block
            if item.get("type") == "output_text":
                text = item.get("text") or item.get("output_text")
                if text:
                    return json.loads(text.strip())
        
        # Walk content array
        for c in content_list:
            if c.get("type") == "output_text":
                # Accept both "text" and "output_text" keys
                text = c.get("text") or c.get("output_text")
                if text:
                    return json.loads(text.strip())

    raise RuntimeError(f"Could not locate JSON output in response body. Keys: {list(body.keys())}")


def parse_vision_results(batch_output_jsonl: Path, out_annotations_jsonl: Path, images_jsonl: Path) -> None:
    """
    Parse the vision results from a batch output JSONL file.
    Args:
        batch_output_jsonl: The path to the batch output JSONL file.
        out_annotations_jsonl: The path to the output annotations JSONL file.
        images_jsonl: The path to the original images.jsonl file for metadata join.
    """
    # Load images.jsonl into a dictionary keyed by image_id for metadata join
    images_meta: Dict[str, Dict[str, Any]] = {}
    for img_row in read_jsonl(images_jsonl):
        image_id = img_row.get("image_id")
        if image_id:
            images_meta[image_id] = {
                "license": img_row.get("license"),
                "doi": img_row.get("doi"),
            }
    
    logger.info(f"Loaded metadata for {len(images_meta)} images from {images_jsonl}")
    
    rows_out = []
    failed_rows = []
    
    for rec in parse_batch_output_responses(batch_output_jsonl):
        custom_id = rec["custom_id"] or ""
        if not custom_id.startswith("vision::"):
            continue
        
        image_id = custom_id.split("vision::", 1)[1]
        error = rec.get("error")
        
        # Handle failed requests
        if error:
            failed_rows.append({
                "image_id": image_id,
                "custom_id": custom_id,
                "error": error,
            })
            logger.warning(f"Vision request failed for {image_id}: {error}")
            continue
        
        body = rec["body"]

        try:
            parsed = extract_parsed_json_from_response(body)
            parsed["image_id"] = image_id
            
            # Join with images.jsonl to get license/doi (reliable source)
            img_meta = images_meta.get(image_id, {})
            license_ = img_meta.get("license")
            doi = img_meta.get("doi")
            
            # Add license/doi if available
            if license_:
                parsed["license"] = license_
            if doi:
                parsed["doi"] = doi
            
            rows_out.append(parsed)
        except Exception as e:
            failed_rows.append({
                "image_id": image_id,
                "custom_id": custom_id,
                "error": {"message": str(e), "type": "parse_error"},
            })
            logger.error(f"Failed to parse vision result for {image_id}: {e}")

    write_jsonl(out_annotations_jsonl, rows_out)
    
    # Write failed rows to a separate file for debugging
    if failed_rows:
        failed_path = out_annotations_jsonl.parent / f"{out_annotations_jsonl.stem}_failed.jsonl"
        write_jsonl(failed_path, failed_rows)
        logger.warning(f"Wrote {len(failed_rows)} failed vision requests to {failed_path}")
    
    logger.info(f"Parsed {len(rows_out)} successful vision annotations, {len(failed_rows)} failed")


def parse_judge_results(batch_output_jsonl: Path, out_qrels_jsonl: Path, annotations_jsonl: Path) -> None:
    """
    Parse the judge results from a batch output JSONL file.
    Args:
        batch_output_jsonl: The path to the batch output JSONL file.
        out_qrels_jsonl: The path to the output qrels JSONL file.
        annotations_jsonl: The path to the annotations JSONL file for metadata join.
    """
    # Load annotations into a dictionary keyed by image_id for metadata join
    annotations_meta: Dict[str, Dict[str, Any]] = {}
    for ann_row in read_jsonl(annotations_jsonl):
        image_id = ann_row.get("image_id")
        if image_id:
            annotations_meta[image_id] = {
                "license": ann_row.get("license"),
                "doi": ann_row.get("doi"),
                "tags": ann_row.get("tags"),
                "confidence": ann_row.get("confidence"),
                "environment_type": ann_row.get("environment_type"),
                "confounder_type": ann_row.get("confounder_type"),
                "lighting": ann_row.get("lighting"),
                "flame_visible": ann_row.get("flame_visible"),
                "plume_stage": ann_row.get("plume_stage"),
                "viewpoint": ann_row.get("viewpoint"),
                "summary": ann_row.get("summary"),
            }
    
    logger.info(f"Loaded metadata for {len(annotations_meta)} images from {annotations_jsonl}")
    
    rows_out = []
    failed_rows = []
    
    for rec in parse_batch_output_responses(batch_output_jsonl):
        custom_id = rec["custom_id"] or ""
        if not custom_id.startswith("judge::"):
            continue
        
        query_id = custom_id.split("judge::", 1)[1]
        error = rec.get("error")
        
        # Handle failed requests
        if error:
            failed_rows.append({
                "query_id": query_id,
                "custom_id": custom_id,
                "error": error,
            })
            logger.warning(f"Judge request failed for {query_id}: {error}")
            continue
        
        body = rec["body"]

        try:
            parsed = extract_parsed_json_from_response(body)
            query_text = parsed["query_text"]
            for j in parsed["judgments"]:
                image_id = j["image_id"]
                
                # Start with base fields
                row = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "image_id": image_id,
                    "relevance_label": int(j["relevance_label"]),
                }
                
                # Join with annotations.jsonl to get metadata
                img_meta = annotations_meta.get(image_id, {})
                if img_meta:
                    # Add all available metadata fields
                    if img_meta.get("license") is not None:
                        row["license"] = img_meta["license"]
                    if img_meta.get("doi") is not None:
                        row["doi"] = img_meta["doi"]
                    if img_meta.get("tags") is not None:
                        row["tags"] = img_meta["tags"]
                    if img_meta.get("confidence") is not None:
                        row["confidence"] = img_meta["confidence"]
                    if img_meta.get("environment_type") is not None:
                        row["environment_type"] = img_meta["environment_type"]
                    if img_meta.get("confounder_type") is not None:
                        row["confounder_type"] = img_meta["confounder_type"]
                    if img_meta.get("lighting") is not None:
                        row["lighting"] = img_meta["lighting"]
                    if img_meta.get("flame_visible") is not None:
                        row["flame_visible"] = img_meta["flame_visible"]
                    if img_meta.get("plume_stage") is not None:
                        row["plume_stage"] = img_meta["plume_stage"]
                    if img_meta.get("viewpoint") is not None:
                        row["viewpoint"] = img_meta["viewpoint"]
                    if img_meta.get("summary") is not None:
                        row["summary"] = img_meta["summary"]
                
                rows_out.append(row)
        except Exception as e:
            failed_rows.append({
                "query_id": query_id,
                "custom_id": custom_id,
                "error": {"message": str(e), "type": "parse_error"},
            })
            logger.error(f"Failed to parse judge result for {query_id}: {e}")

    write_jsonl(out_qrels_jsonl, rows_out)
    
    # Write failed rows to a separate file for debugging
    if failed_rows:
        failed_path = out_qrels_jsonl.parent / f"{out_qrels_jsonl.stem}_failed.jsonl"
        write_jsonl(failed_path, failed_rows)
        logger.warning(f"Wrote {len(failed_rows)} failed judge requests to {failed_path}")
    
    logger.info(f"Parsed {len(rows_out)} successful judge qrels, {len(failed_rows)} failed")


# -----------------------------
# CLI
# -----------------------------

def cmd_make_vision_batch(args: argparse.Namespace) -> None:
    """
    Make a vision batch.
    Args:
        args: The arguments.
    """
    out = Path(args.out_jsonl)
    rows = build_vision_batch_lines(Path(args.images_jsonl))
    write_jsonl(out, rows)
    
    # Check file size (Batch API limit is 200MB)
    file_size = out.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > 200:
        raise RuntimeError(
            f"Batch file size ({file_size_mb:.2f} MB) exceeds 200MB limit. "
            f"Reduce number of images or split into multiple batches."
        )
    elif file_size_mb > 150:
        logger.warning(f"Batch file size ({file_size_mb:.2f} MB) is approaching 200MB limit")
    
    logger.info(f"Wrote vision batch input: {out} ({file_size_mb:.2f} MB)")


def cmd_make_judge_batch(args: argparse.Namespace) -> None:
    """
    Make a judge batch.
    Args:
        args: The arguments.
    """
    out = Path(args.out_jsonl)
    rows = build_judge_batch_lines(Path(args.query_plan_jsonl), Path(args.annotations_jsonl))
    write_jsonl(out, rows)
    
    # Check file size (Batch API limit is 200MB)
    file_size = out.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > 200:
        raise RuntimeError(
            f"Batch file size ({file_size_mb:.2f} MB) exceeds 200MB limit. "
            f"Reduce number of queries or split into multiple batches."
        )
    elif file_size_mb > 150:
        logger.warning(f"Batch file size ({file_size_mb:.2f} MB) is approaching 200MB limit")
    
    logger.info(f"Wrote judge batch input: {out} ({file_size_mb:.2f} MB)")


def cmd_submit(args: argparse.Namespace) -> None:
    """
    Submit a batch (with automatic sharding if needed to avoid 5M token limit).
    
    If the batch exceeds the per-shard limit (800 images for vision, 100 queries for judge),
    it will automatically split into multiple shards and submit them with concurrency control.
    
    Args:
        args: The arguments.
    """
    try:
        client = OpenAI()
        input_jsonl = Path(args.input_jsonl)
        
        # Count lines to determine if sharding is needed
        line_count = sum(1 for _ in read_jsonl(input_jsonl))
        logger.info(f"Input batch has {line_count} requests")
        
        # Determine max items per shard based on stage
        if args.stage == "vision":
            max_per_shard = MAX_IMAGES_PER_BATCH
            shard_prefix = "vision_shard"
        elif args.stage == "judge":
            max_per_shard = MAX_QUERIES_PER_BATCH
            shard_prefix = "judge_shard"
        else:
            max_per_shard = 1000  # default
            shard_prefix = f"{args.stage}_shard"
        
        # Check if sharding is needed
        if line_count <= max_per_shard:
            # Single batch - no sharding needed
            refs = submit_batch(
                client,
                input_jsonl,
                metadata={"purpose": args.purpose, "stage": args.stage},
            )
            result = {"input_file_id": refs.input_file_id, "batch_id": refs.batch_id, "sharded": False}
            logger.info(f"Submitted single batch: {json.dumps(result, indent=2)}")
            print(json.dumps(result, indent=2))
        else:
            # Sharding needed
            logger.info(f"Sharding required: {line_count} requests > {max_per_shard} per shard")
            shard_dir = input_jsonl.parent / f"{input_jsonl.stem}_shards"
            shard_paths = shard_batch_jsonl(input_jsonl, shard_dir, max_per_shard, shard_prefix)
            
            # Submit shards with concurrency control
            all_refs = submit_batch_shards(
                client,
                shard_paths,
                metadata={"purpose": args.purpose, "stage": args.stage},
                max_concurrent=MAX_CONCURRENT_BATCHES,
            )
            
            # Output all batch IDs
            batch_ids = [ref.batch_id for ref in all_refs]
            result = {
                "sharded": True,
                "num_shards": len(shard_paths),
                "batch_ids": batch_ids,
                "input_file_ids": [ref.input_file_id for ref in all_refs],
            }
            logger.info(f"Submitted {len(shard_paths)} shards: {json.dumps(result, indent=2)}")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Failed to submit batch: {e}", exc_info=True)
        # Don't output JSON on error - let the error propagate so Makefile can handle it
        raise


def cmd_wait(args: argparse.Namespace) -> None:
    """
    Wait for batch(es) to complete.
    Args:
        args: The arguments. batch_id can be a single ID or comma-separated list.
    """
    client = OpenAI()
    
    # Parse batch_id - can be single ID or comma-separated list
    batch_ids = [bid.strip() for bid in args.batch_id.split(",") if bid.strip()]
    
    if len(batch_ids) == 1:
        # Single batch
        final = wait_for_batch(client, batch_ids[0], poll_s=args.poll_s)
        logger.info(f"Batch status: {json.dumps(final, indent=2)}")
        print(json.dumps(final, indent=2))
    else:
        # Multiple batches
        logger.info(f"Waiting for {len(batch_ids)} batches...")
        all_statuses = []
        for batch_id in batch_ids:
            logger.info(f"Waiting for batch {batch_id}...")
            final = wait_for_batch(client, batch_id, poll_s=args.poll_s)
            all_statuses.append(final)
        
        result = {
            "num_batches": len(batch_ids),
            "batches": all_statuses,
        }
        logger.info(f"All batches completed: {json.dumps(result, indent=2)}")
        print(json.dumps(result, indent=2))


def cmd_download_output(args: argparse.Namespace) -> None:
    """
    Download the output of batch(es) and merge if multiple.
    Args:
        args: The arguments. batch_id can be a single ID or comma-separated list.
    """
    client = OpenAI()
    
    # Parse batch_id - can be single ID or comma-separated list
    batch_ids = [bid.strip() for bid in args.batch_id.split(",") if bid.strip()]
    
    if len(batch_ids) == 1:
        # Single batch - simple download
        b = client.batches.retrieve(batch_ids[0])
        out_file_id = getattr(b, "output_file_id", None) or (b.get("output_file_id") if isinstance(b, dict) else None)
        err_file_id = getattr(b, "error_file_id", None) or (b.get("error_file_id") if isinstance(b, dict) else None)

        if not out_file_id and not err_file_id:
            raise RuntimeError("Batch has no output_file_id or error_file_id yet (not completed?)")

        if out_file_id:
            download_file(client, out_file_id, Path(args.out_output_jsonl))
            logger.info(f"Downloaded output file: {args.out_output_jsonl}")
        if err_file_id:
            download_file(client, err_file_id, Path(args.out_error_jsonl))
            logger.info(f"Downloaded error file: {args.out_error_jsonl}")
    else:
        # Multiple batches - download and merge
        logger.info(f"Downloading and merging {len(batch_ids)} batch outputs...")
        all_output_rows = []
        all_error_rows = []
        
        for i, batch_id in enumerate(batch_ids):
            b = client.batches.retrieve(batch_id)
            out_file_id = getattr(b, "output_file_id", None) or (b.get("output_file_id") if isinstance(b, dict) else None)
            err_file_id = getattr(b, "error_file_id", None) or (b.get("error_file_id") if isinstance(b, dict) else None)
            
            if out_file_id:
                temp_output = Path(args.out_output_jsonl).parent / f"temp_output_{i}.jsonl"
                download_file(client, out_file_id, temp_output)
                # Read and collect rows
                for row in read_jsonl(temp_output):
                    all_output_rows.append(row)
                temp_output.unlink()  # Clean up temp file
                logger.info(f"Downloaded output from batch {batch_id} ({i+1}/{len(batch_ids)})")
            
            if err_file_id:
                temp_error = Path(args.out_error_jsonl).parent / f"temp_error_{i}.jsonl"
                download_file(client, err_file_id, temp_error)
                # Read and collect rows
                for row in read_jsonl(temp_error):
                    all_error_rows.append(row)
                temp_error.unlink()  # Clean up temp file
                logger.info(f"Downloaded errors from batch {batch_id} ({i+1}/{len(batch_ids)})")
        
        # Write merged outputs
        if all_output_rows:
            write_jsonl(Path(args.out_output_jsonl), all_output_rows)
            logger.info(f"Merged {len(all_output_rows)} output rows to {args.out_output_jsonl}")
        
        if all_error_rows:
            write_jsonl(Path(args.out_error_jsonl), all_error_rows)
            logger.info(f"Merged {len(all_error_rows)} error rows to {args.out_error_jsonl}")


def cmd_parse_vision(args: argparse.Namespace) -> None:
    """
    Parse the vision results.
    Args:
        args: The arguments.
    """
    parse_vision_results(
        Path(args.batch_output_jsonl),
        Path(args.out_annotations_jsonl),
        Path(args.images_jsonl),
    )
    logger.info(f"Wrote annotations: {args.out_annotations_jsonl}")


def cmd_parse_judge(args: argparse.Namespace) -> None:
    """
    Parse the judge results.
    Args:
        args: The arguments.
    """
    parse_judge_results(
        Path(args.batch_output_jsonl),
        Path(args.out_qrels_jsonl),
        Path(args.annotations_jsonl),
    )
    logger.info(f"Wrote qrels/judgments: {args.out_qrels_jsonl}")


def cmd_resubmit_vision_errors(args: argparse.Namespace) -> None:
    """
    Re-submit failed vision requests from an error JSONL file.
    Args:
        args: The arguments.
    """
    error_jsonl = Path(args.error_jsonl)
    images_jsonl = Path(args.images_jsonl)
    out_jsonl = Path(args.out_jsonl)
    
    if not error_jsonl.exists():
        raise FileNotFoundError(f"Error file not found: {error_jsonl}")
    if not images_jsonl.exists():
        raise FileNotFoundError(f"Images file not found: {images_jsonl}")
    
    # Extract failed image IDs
    failed_ids = extract_failed_image_ids(error_jsonl)
    if not failed_ids:
        logger.warning(f"No failed vision requests found in {error_jsonl}")
        # Create empty output file
        write_jsonl(out_jsonl, [])
        logger.info(f"Created empty batch file: {out_jsonl}")
        return
    
    logger.info(f"Found {len(failed_ids)} failed image IDs to re-submit")
    
    # Build batch lines for only the failed images
    rows = list(build_vision_batch_lines(images_jsonl, image_ids_filter=failed_ids))
    
    if not rows:
        logger.warning(f"None of the failed image IDs were found in {images_jsonl}")
        write_jsonl(out_jsonl, [])
        return
    
    write_jsonl(out_jsonl, rows)
    
    # Check file size
    file_size = out_jsonl.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > 200:
        raise RuntimeError(
            f"Batch file size ({file_size_mb:.2f} MB) exceeds 200MB limit. "
            f"Consider splitting into multiple batches."
        )
    elif file_size_mb > 150:
        logger.warning(f"Batch file size ({file_size_mb:.2f} MB) is approaching 200MB limit")
    
    logger.info(f"Wrote vision retry batch with {len(rows)} requests: {out_jsonl} ({file_size_mb:.2f} MB)")
    
    # Optionally submit if requested
    if args.submit:
        try:
            client = OpenAI()
            refs = submit_batch(
                client,
                out_jsonl,
                metadata={"purpose": args.purpose, "stage": "vision", "retry": "true"},
            )
            result = {"input_file_id": refs.input_file_id, "batch_id": refs.batch_id, "retry": True}
            logger.info(f"Submitted retry batch: {json.dumps(result, indent=2)}")
            print(json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"Failed to submit retry batch: {e}", exc_info=True)
            raise


def cmd_resubmit_judge_errors(args: argparse.Namespace) -> None:
    """
    Re-submit failed judge requests from an error JSONL file.
    Args:
        args: The arguments.
    """
    error_jsonl = Path(args.error_jsonl)
    query_plan_jsonl = Path(args.query_plan_jsonl)
    annotations_jsonl = Path(args.annotations_jsonl)
    out_jsonl = Path(args.out_jsonl)
    
    if not error_jsonl.exists():
        raise FileNotFoundError(f"Error file not found: {error_jsonl}")
    if not query_plan_jsonl.exists():
        raise FileNotFoundError(f"Query plan file not found: {query_plan_jsonl}")
    if not annotations_jsonl.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_jsonl}")
    
    # Extract failed query IDs
    failed_ids = extract_failed_query_ids(error_jsonl)
    if not failed_ids:
        logger.warning(f"No failed judge requests found in {error_jsonl}")
        # Create empty output file
        write_jsonl(out_jsonl, [])
        logger.info(f"Created empty batch file: {out_jsonl}")
        return
    
    logger.info(f"Found {len(failed_ids)} failed query IDs to re-submit")
    
    # Build batch lines for only the failed queries
    rows = list(build_judge_batch_lines(query_plan_jsonl, annotations_jsonl, query_ids_filter=failed_ids))
    
    if not rows:
        logger.warning(f"None of the failed query IDs were found in {query_plan_jsonl}")
        write_jsonl(out_jsonl, [])
        return
    
    write_jsonl(out_jsonl, rows)
    
    # Check file size
    file_size = out_jsonl.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > 200:
        raise RuntimeError(
            f"Batch file size ({file_size_mb:.2f} MB) exceeds 200MB limit. "
            f"Consider splitting into multiple batches."
        )
    elif file_size_mb > 150:
        logger.warning(f"Batch file size ({file_size_mb:.2f} MB) is approaching 200MB limit")
    
    logger.info(f"Wrote judge retry batch with {len(rows)} requests: {out_jsonl} ({file_size_mb:.2f} MB)")
    
    # Optionally submit if requested
    if args.submit:
        try:
            client = OpenAI()
            refs = submit_batch(
                client,
                out_jsonl,
                metadata={"purpose": args.purpose, "stage": "judge", "retry": "true"},
            )
            result = {"input_file_id": refs.input_file_id, "batch_id": refs.batch_id, "retry": True}
            logger.info(f"Submitted retry batch: {json.dumps(result, indent=2)}")
            print(json.dumps(result, indent=2))
        except Exception as e:
            logger.error(f"Failed to submit retry batch: {e}", exc_info=True)
            raise


def cmd_list_batches(args):
    """
    List batches.
    Args:
        args: The arguments.
    """
    client = OpenAI()
    batches = client.batches.list(limit=args.limit)

    ACTIVE = {"validating", "in_progress", "finalizing"}

    for b in batches.data:
        if args.active_only and b.status not in ACTIVE:
            continue

        # Convert SDK/Pydantic objects to plain dicts
        if hasattr(b, "model_dump"):
            request_counts = b.request_counts.model_dump() if b.request_counts else None
            metadata = b.metadata.model_dump() if hasattr(b.metadata, "model_dump") else b.metadata
        else:
            # Fallback (older SDK)
            request_counts = dict(b.request_counts) if b.request_counts else None
            metadata = dict(b.metadata) if b.metadata else None

        print(json.dumps({
            "id": b.id,
            "status": b.status,
            "endpoint": b.endpoint,
            "created_at": b.created_at,
            "metadata": metadata,
            "request_counts": request_counts,
        }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """
    Build a parser for the command line arguments.
    Returns:
        A parser for the command line arguments.
    """
    p = argparse.ArgumentParser("firebench_openai_batch")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-vision-batch", help="Build Batch JSONL for image annotation (vision)")
    s.add_argument("--images-jsonl", required=True)
    s.add_argument("--out-jsonl", required=True)
    s.set_defaults(func=cmd_make_vision_batch)

    s = sub.add_parser("make-judge-batch", help="Build Batch JSONL for query+relevance labeling (text-only)")
    s.add_argument("--query-plan-jsonl", required=True)
    s.add_argument("--annotations-jsonl", required=True)
    s.add_argument("--out-jsonl", required=True)
    s.set_defaults(func=cmd_make_judge_batch)

    s = sub.add_parser("submit", help="Upload input JSONL + create a batch")
    s.add_argument("--input-jsonl", required=True)
    s.add_argument("--stage", required=True, choices=["vision", "judge"])
    s.add_argument("--purpose", default="firebench")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("wait", help="Poll until batch(es) are terminal (supports comma-separated batch IDs)")
    s.add_argument("--batch-id", required=True, help="Single batch ID or comma-separated list of batch IDs")
    s.add_argument("--poll-s", type=int, default=60)
    s.set_defaults(func=cmd_wait)

    s = sub.add_parser("download-output", help="Download batch output_file and error_file (supports comma-separated batch IDs, merges outputs)")
    s.add_argument("--batch-id", required=True, help="Single batch ID or comma-separated list of batch IDs")
    s.add_argument("--out-output-jsonl", required=True)
    s.add_argument("--out-error-jsonl", required=True)
    s.set_defaults(func=cmd_download_output)

    s = sub.add_parser("parse-vision", help="Parse batch output -> annotations JSONL")
    s.add_argument("--batch-output-jsonl", required=True)
    s.add_argument("--out-annotations-jsonl", required=True)
    s.add_argument("--images-jsonl", required=True, help="Original images.jsonl file for metadata join (license/doi)")
    s.set_defaults(func=cmd_parse_vision)

    s = sub.add_parser("parse-judge", help="Parse batch output -> qrels/judgments LONG JSONL")
    s.add_argument("--batch-output-jsonl", required=True)
    s.add_argument("--out-qrels-jsonl", required=True)
    s.add_argument("--annotations-jsonl", required=True, help="Original annotations.jsonl file for metadata join")
    s.set_defaults(func=cmd_parse_judge)

    s = sub.add_parser("list-batches", help="List OpenAI batches")
    s.add_argument("--active-only", action="store_true")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_list_batches)

    s = sub.add_parser("resubmit-vision-errors", help="Re-submit failed vision requests from error JSONL")
    s.add_argument("--error-jsonl", required=True, help="Error JSONL file from batch output")
    s.add_argument("--images-jsonl", required=True, help="Original images.jsonl file")
    s.add_argument("--out-jsonl", required=True, help="Output batch JSONL file for retry")
    s.add_argument("--submit", action="store_true", help="Automatically submit the retry batch after creating it")
    s.add_argument("--purpose", default="firebench")
    s.set_defaults(func=cmd_resubmit_vision_errors)

    s = sub.add_parser("resubmit-judge-errors", help="Re-submit failed judge requests from error JSONL")
    s.add_argument("--error-jsonl", required=True, help="Error JSONL file from batch output")
    s.add_argument("--query-plan-jsonl", required=True, help="Original query_plan.jsonl file")
    s.add_argument("--annotations-jsonl", required=True, help="Original annotations.jsonl file")
    s.add_argument("--out-jsonl", required=True, help="Output batch JSONL file for retry")
    s.add_argument("--submit", action="store_true", help="Automatically submit the retry batch after creating it")
    s.add_argument("--purpose", default="firebench")
    s.set_defaults(func=cmd_resubmit_judge_errors)

    return p


def main() -> None:
    """
    Main function.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    """
    Main function.
    """
    main()
