"""
io.py

Shared JSONL helpers for benchmark pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


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

