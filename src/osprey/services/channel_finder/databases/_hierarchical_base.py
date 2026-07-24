"""Shared state and low-level helpers for the hierarchical database.

This module holds :class:`_HierarchicalBase`, the common ancestor of the
hierarchical database mixins. It declares the instance attributes that all
mixins share (populated by ``load_database``) and provides the small,
cross-cutting helpers used by more than one concern (instance-name expansion,
value coercion, and node/expansion counting).

The mixins in the sibling ``_hierarchical_*`` modules build on this base;
:class:`~osprey.services.channel_finder.databases.hierarchical.HierarchicalChannelDatabase`
combines them into the public database class.
"""

from typing import Any

from ..core.base_database import BaseDatabase

_HIER_META_KEYS = frozenset(
    {
        "_description",
        "_expansion",
        "_channel_part",
        "_is_leaf",
        "_separator",
    }
)


class _HierarchicalBase(BaseDatabase):
    """Shared state and utility helpers for the hierarchical database mixins.

    Not instantiated on its own: the concrete
    :class:`~osprey.services.channel_finder.databases.hierarchical.HierarchicalChannelDatabase`
    provides the abstract-method implementations via the functional mixins.
    """

    # Populated by load_database() (see _HierarchicalLoadingMixin).
    tree: dict[str, Any]
    _raw_data: dict[str, Any]
    hierarchy_levels: list[str]
    naming_pattern: str
    hierarchy_config: dict[str, Any]
    default_separators: dict[tuple[str, str], str]
    channel_map: dict[str, dict]
    # Lazily created by generate_tree_preview() (see _HierarchicalPreviewMixin).
    _tree_preview_cache: dict[tuple[int, int], str]

    def _get_instance_names(self, expansion_def: dict) -> list[str]:
        """Get list of instance names from expansion definition."""
        expansion_type = expansion_def.get("_type")

        if expansion_type == "range":
            pattern = expansion_def.get("_pattern", "{}")
            start, end = expansion_def.get("_range", [1, 1])
            return [pattern.format(i) for i in range(start, end + 1)]

        elif expansion_type == "list":
            return expansion_def.get("_instances", [])

        return []

    def _ensure_list(self, value: Any) -> list:
        """Convert value to list if it isn't already."""
        if isinstance(value, list):
            return value
        elif value is None:
            return []
        else:
            return [value]

    def _get_single_value(self, value):
        """Get single value from potentially list value."""
        if isinstance(value, list):
            return value[0] if value else None
        return value

    @staticmethod
    def _child_keys(node: dict) -> list[str]:
        """Return non-metadata child keys of a hierarchical node."""
        return [k for k in node if k not in _HIER_META_KEYS and not k.startswith("_")]

    @staticmethod
    def _expansion_instance_count(expansion: dict) -> int:
        """Return the number of instances defined by an ``_expansion`` descriptor."""
        r = expansion.get("_range") or expansion.get("range", [1, 1])
        if isinstance(r, list) and len(r) >= 2:
            return max(0, r[1] - r[0] + 1)
        return 1

    @classmethod
    def _count_channels(cls, node: dict) -> int:
        """Recursively count leaf channels under a node, accounting for expansion."""
        if not isinstance(node, dict):
            return 0
        count = 0
        for k, v in node.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                child_count = cls._count_channels(v)
                if "_expansion" in v:
                    child_count *= cls._expansion_instance_count(v["_expansion"])
                count += child_count
            else:
                count += 1
        children = cls._child_keys(node)
        if not children:
            count += 1
        return count
