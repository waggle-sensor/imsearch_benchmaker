#!/usr/bin/env python3
"""
Extract version from setup.py file.
"""

import re
import sys
from pathlib import Path


def extract_version(file_path: str = "setup.py") -> str:
    """
    Extract VERSION from setup.py file.
    
    Args:
        file_path: Path to setup.py file
        
    Returns:
        Version string
        
    Raises:
        SystemExit: If version cannot be extracted
    """
    setup_file = Path(file_path)
    if not setup_file.exists():
        print(f"❌ File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    content = setup_file.read_text(encoding="utf-8")
    
    # Match VERSION = "x.y.z" or VERSION = 'x.y.z'
    match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
    
    if not match:
        print(f"❌ Could not extract version from {file_path}", file=sys.stderr)
        sys.exit(1)
    
    version = match.group(1)
    print(version)
    return version


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "setup.py"
    extract_version(file_path)

