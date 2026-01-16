"""
_responses.py

Helpers for parsing OpenAI Responses API outputs.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def extract_json_from_response_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract JSON output from a Responses API response body.
    """
    output_text = body.get("output_text")
    if output_text:
        return json.loads(output_text.strip())

    for item in body.get("output", []) or []:
        content_list = item.get("content", [])
        if not content_list and isinstance(item, dict):
            if item.get("type") == "output_text":
                text = item.get("text") or item.get("output_text")
                if text:
                    return json.loads(text.strip())

        for content in content_list:
            if content.get("type") == "output_text":
                text = content.get("text") or content.get("output_text")
                if text:
                    return json.loads(text.strip())

    raise RuntimeError(f"Could not locate JSON output in response body. Keys: {list(body.keys())}")

