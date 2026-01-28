#!/usr/bin/env python3
"""
Create a GitHub release from version information.

This script creates a GitHub release with appropriate release notes based on
version changes in setup.py.
"""

import json
import os
import sys
from pathlib import Path


def main():
    """Create GitHub release payload and make API call."""
    version = os.environ.get("VERSION")
    prev_version = os.environ.get("PREV_VERSION", "")
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    
    if not version:
        print("❌ VERSION environment variable is required")
        sys.exit(1)
    
    if not repo:
        print("❌ GITHUB_REPOSITORY environment variable is required")
        sys.exit(1)
    
    if not token:
        print("❌ GITHUB_TOKEN environment variable is required")
        sys.exit(1)
    
    # Create release body
    if prev_version and prev_version != version:
        body = f"""## Release {version}

Automated release created from version change in setup.py.

### Changes
- Version updated from {prev_version} to {version}

See the [full changelog](https://github.com/{repo}/compare/{prev_version}...{version}) for details."""
    elif prev_version and prev_version == version:
        # Same version but setup.py was modified (e.g., dependency updates)
        body = f"""## Release {version}

Automated release created from version change in setup.py.

### Changes
- Setup.py updated (version remains {version})

See the [full changelog](https://github.com/{repo}/compare/{prev_version}...{version}) for details."""
    else:
        body = f"""## Release {version}

Automated release created from version change in setup.py.

### Changes
- Initial release version {version}"""
    
    # Create JSON payload
    payload = {
        "tag_name": version,
        "name": f"Release {version}",
        "body": body,
        "draft": False,
        "prerelease": False
    }
    
    # Write JSON to file for curl (or we could use requests library)
    output_file = Path("release_payload.json")
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"✅ Created release payload for version {version}")
    print(f"   Tag: {version}")
    print(f"   Name: Release {version}")
    if prev_version:
        print(f"   Previous version: {prev_version}")
    
    # Create release using GitHub REST API
    import subprocess
    
    curl_cmd = [
        "curl", "-s", "-w", "\n%{http_code}",
        "-X", "POST",
        "-H", "Accept: application/vnd.github.v3+json",
        "-H", f"Authorization: token {token}",
        "-H", "Content-Type: application/json",
        "--data-binary", f"@{output_file}",
        f"https://api.github.com/repos/{repo}/releases"
    ]
    
    result = subprocess.run(curl_cmd, capture_output=True, text=True)
    response = result.stdout
    
    # Extract HTTP code (last line)
    lines = response.strip().split("\n")
    http_code = lines[-1] if lines else "000"
    response_body = "\n".join(lines[:-1]) if len(lines) > 1 else response
    
    if http_code == "201":
        print(f"✅ Created GitHub release: Release {version}")
        print(f"   Repository: https://github.com/{repo}")
        sys.exit(0)
    else:
        print(f"❌ Failed to create release. HTTP code: {http_code}")
        print(f"Response: {response_body}")
        sys.exit(1)


if __name__ == "__main__":
    main()

