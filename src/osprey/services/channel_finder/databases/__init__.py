"""
Database implementations for Channel Finder.

Provides various database backend implementations:
- flat: Simple flat list format (base implementation for in-context databases)
- hierarchical: Hierarchical tree structure for large databases
- middle_layer: MML functional hierarchy (System→Family→Field)
"""

from .flat import ChannelDatabase as FlatChannelDatabase
from .hierarchical import HierarchicalChannelDatabase
from .middle_layer import MiddleLayerDatabase

__all__ = [
    "FlatChannelDatabase",
    "HierarchicalChannelDatabase",
    "MiddleLayerDatabase",
]
