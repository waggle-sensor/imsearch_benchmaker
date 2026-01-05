"""
clipscore package

Service-agnostic CLIPScore calculation framework.

This package provides:
- Base classes for creating CLIP API adapters
- Built-in adapters for common services
- Registry system for managing adapters
"""

from .framework import CLIPAdapter, CLIPAdapterRegistry
from .adapters import HTTPCLIPAdapter, LocalCLIPAdapter

# Register built-in adapters
CLIPAdapterRegistry.register("local", LocalCLIPAdapter)
CLIPAdapterRegistry.register("http", HTTPCLIPAdapter)

__all__ = [
    'CLIPAdapter',
    'CLIPAdapterRegistry',
    'HTTPCLIPAdapter',
    'LocalCLIPAdapter',
]
