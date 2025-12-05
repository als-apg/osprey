"""
Database implementations for Channel Finder.

Provides various database backend implementations:
- legacy: Original flat format
- template: Compact template-based format with expansion
- hierarchical: Hierarchical tree structure for large databases
"""

from .hierarchical import HierarchicalChannelDatabase
from .legacy import ChannelDatabase as LegacyChannelDatabase
from .template import ChannelDatabase as TemplateChannelDatabase

__all__ = [
    "LegacyChannelDatabase",
    "TemplateChannelDatabase",
    "HierarchicalChannelDatabase",
]
