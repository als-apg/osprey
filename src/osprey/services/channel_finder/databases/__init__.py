"""
Database implementations for Channel Finder.

Provides various database backend implementations:
- flat: Simple flat list format (base implementation for in-context databases)
- template: Compact template-based format with expansion (extends flat)
- hierarchical: Hierarchical tree structure for large databases
- middle_layer: MML functional hierarchy (System→Family→Field)
"""

from .flat import ChannelDatabase as FlatChannelDatabase
from .hierarchical import HierarchicalChannelDatabase
from .middle_layer import MiddleLayerDatabase
from .template import ChannelDatabase as TemplateChannelDatabase

_has_google_sheets = False
try:
    from .google_sheets import GoogleSheetsChannelDatabase

    _has_google_sheets = True
except ImportError:
    pass

# Backward compatibility alias
LegacyChannelDatabase = FlatChannelDatabase

__all__ = [
    "FlatChannelDatabase",
    "LegacyChannelDatabase",  # Backward compatibility
    "TemplateChannelDatabase",
    "HierarchicalChannelDatabase",
    "MiddleLayerDatabase",
]

if _has_google_sheets:
    __all__.append("GoogleSheetsChannelDatabase")
