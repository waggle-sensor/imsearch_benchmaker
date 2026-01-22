"""
Setup configuration for imsearch_benchmaker.

This package provides a framework for creating image search benchmarks with
optional adapters for different services (OpenAI, local CLIP, etc.).
"""

from pathlib import Path
from setuptools import setup, find_packages

VERSION = "0.0.8"

def parse_requirements(requirements_path: Path) -> list[str]:
    """Helper function to parse requirements file, filtering out empty lines and comments."""
    if not requirements_path.exists():
        return []
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Get requirements from files
core_requirements_path = Path(__file__).parent / "imsearch_benchmaker/requirements.txt"
openai_requirements_path = Path(__file__).parent / "imsearch_benchmaker/adapters/openai/requirements.txt"
local_requirements_path = Path(__file__).parent / "imsearch_benchmaker/adapters/local/requirements.txt"

# Core dependencies
core_dependencies = parse_requirements(core_requirements_path)

# Optional dependencies for different adapters
openai_deps = parse_requirements(openai_requirements_path)
local_deps = parse_requirements(local_requirements_path)

extras_require = {
    "openai": openai_deps,
    "local": local_deps,
    "all": openai_deps + local_deps,
}

setup(
    name="imsearch_benchmaker",
    version=VERSION,
    description="Framework for creating image search benchmarks with optional adapters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Francisco Lozano",
    author_email="francisco.lozano@northwestern.edu",
    url="https://github.com/waggle-sensor/imsearch_benchmaker",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.11",
    install_requires=core_dependencies,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "benchmaker=imsearch_benchmaker.framework.cli:main",
        ],
    }
)

