"""
client.py

OpenAI client creation and file download utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from openai import OpenAI
from .config import OpenAIConfig

def get_openai_client(
    api_key: Optional[str] = None,
    openai_config: Optional[OpenAIConfig] = None,
) -> OpenAI:
    """
    Get OpenAI client.
    
    Args:
        api_key: Optional API key (takes precedence)
        openai_config: Optional OpenAIConfig
    """
    if api_key:
        return OpenAI(api_key=api_key)
    
    if openai_config:
        if openai_config._openai_api_key:
            return OpenAI(api_key=openai_config._openai_api_key)
    
    return OpenAI()


def download_file(client: OpenAI, file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    data = content.read() if hasattr(content, "read") else content
    out_path.write_bytes(data)

