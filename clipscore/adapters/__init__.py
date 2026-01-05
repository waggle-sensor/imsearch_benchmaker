"""
adapters package

CLIP API adapters for different services.
"""

from .clip import HTTPCLIPAdapter, LocalCLIPAdapter

__all__ = ['HTTPCLIPAdapter', 'LocalCLIPAdapter']

