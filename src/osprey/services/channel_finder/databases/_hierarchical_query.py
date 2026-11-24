"""Navigation and read access over the hierarchical database.

Provides the query side of the public API: enumerating the options available at
a hierarchy level (expanding instance ranges/lists as needed), navigating the
tree by prior selections, channel lookup/validation over the flat map, and
database statistics.
"""

from typing import Any

from ._hierarchical_base import _HierarchicalBase


class UnselectedLevelError(ValueError):
    """An options request whose selections leave out a required level above it.

    Attributes:
        level: The required level the selections name no value for.
        requested: The level whose options were asked for.
    """

    def __init__(self, level: str, requested: str) -> None:
        self.level = level
        self.requested = requested
        super().__init__(
            f"selections name no {level!r}; "
            f"get the {level!r} options before the {requested!r} options"
        )


class _HierarchicalQueryMixin(_HierarchicalBase):
    """Level-option enumeration, tree navigation, and channel lookup."""

    def get_hierarchy_definition(self) -> list[str]:
        """Get the hierarchy level names."""
        return self.hierarchy_levels.copy()

    def get_options_at_level(
        self, level: str, previous_selections: dict[str, Any]
    ) -> list[dict[str, str]]:
        """
        Get available options at a specific hierarchy level.

        Args:
            level: Current level name
            previous_selections: Dict mapping previous level names to selected values

        Returns:
            List of options with name and description

        Raises:
            UnselectedLevelError: ``previous_selections`` omit a required tree
                level above ``level``.
            ValueError: ``level`` is no hierarchy level; a key of
                ``previous_selections`` is no hierarchy level; or a value selected at
                a tree level above ``level`` names no node under the selections
                above it. Options are offered only below a selection of every
                required tree level above ``level``; instance levels and tree
                levels declared ``optional`` may be omitted, and an empty
                selection is no selection.
        """
        self._check_selections(previous_selections, level)

        level_type = self.hierarchy_config["levels"][level]["type"]
        if level_type not in ("tree", "instances"):
            return []

        # Every node the selections reach: a level selected with several values
        # offers the options under each of them, each name once, in first-seen order.
        options: dict[str, dict[str, str]] = {}
        for node in self._navigate_to_nodes(level, previous_selections):
            if not node:
                continue
            found = (
                self._extract_tree_options(node)
                if level_type == "tree"
                else self._get_expansion_options(node, level)
            )
            for option in found:
                options.setdefault(option["name"], option)
        return list(options.values())

    def required_tree_levels(self) -> list[str]:
        """The levels an options request must select above the level it asks about.

        Every tree level not declared ``optional``, top to bottom; instance
        levels and optional tree levels may be left out.
        """
        levels = self.hierarchy_config["levels"]
        return [
            level
            for level in self.hierarchy_levels
            if levels[level]["type"] == "tree" and not levels[level].get("optional", False)
        ]

    def _check_selections(
        self, selections: dict[str, Any], target_level: str | None = None
    ) -> None:
        """Raise ``ValueError`` naming the first mistake in a request's selections.

        Every request is refused when ``target_level`` or a key of ``selections``
        is no hierarchy level. An options request (``target_level`` given) is
        also refused when a required tree level above ``target_level`` has no
        selection (an :class:`UnselectedLevelError`), or when a value selected
        at a tree level above it reaches no node from the nodes the levels above
        reach. Instance levels are walked the way navigation walks them; their
        selections are not checked.
        """
        levels = self.hierarchy_levels
        defined = ", ".join(levels)
        if target_level is not None and target_level not in levels:
            raise ValueError(f"no level {target_level!r}; levels: {defined}")
        for key in selections:
            if key not in levels:
                raise ValueError(f"selections name no level {key!r}; levels: {defined}")
        if target_level is None:
            return

        required = set(self.required_tree_levels())
        frontier = [self.tree]
        for prev_level in levels:
            if prev_level == target_level:
                return
            if self.hierarchy_config["levels"][prev_level]["type"] == "instances":
                frontier = [self._enter_instance_container(node, prev_level) for node in frontier]
                continue
            values = [value for value in self._ensure_list(selections.get(prev_level)) if value]
            if not values:
                if prev_level in required:
                    raise UnselectedLevelError(level=prev_level, requested=target_level)
                continue
            reached: dict[int, dict] = {}
            for value in values:
                found = False
                for node in frontier:
                    child = self._follow_tree_value(node, value)
                    if child is not None:
                        reached.setdefault(id(child), child)
                        found = True
                if not found:
                    raise ValueError(
                        f"no {prev_level!r} named {value!r} under the selections above it"
                    )
            frontier = list(reached.values())

    def _enter_instance_container(self, node: dict, level: str) -> dict:
        """The container named after an instance ``level`` (case-folded), else ``node``."""
        for key, value in node.items():
            if key.upper() == level.upper() and isinstance(value, dict):
                return value
        return node

    def _follow_tree_value(self, node: dict, value: Any) -> dict | None:
        """The node a tree-level ``value`` selects under ``node``, or ``None``.

        A value selects a direct child that is a node (a dict whose key does not
        start with ``_``), or else the child container whose ``_expansion``
        generates it (``CH-1`` selects ``CH`` holding a ``CH-{}`` range).
        """
        if not isinstance(value, str):
            return None
        child = node.get(value)
        if isinstance(child, dict) and not value.startswith("_"):
            return child
        for key, child in node.items():
            if (
                not key.startswith("_")
                and isinstance(child, dict)
                and "_expansion" in child
                and value in self._get_instance_names(child["_expansion"])
            ):
                return child
        return None

    def _navigate_to_nodes(
        self, target_level: str, previous_selections: dict[str, Any]
    ) -> list[dict]:
        """
        The nodes the selections above ``target_level`` reach, each once, in order.

        A tree level selected with several values follows each of them; an empty
        selection is no selection. Instance levels do NOT change tree position
        during selection, but we DO navigate INTO their containers to find children.

        Args:
            target_level: Level we're getting options for
            previous_selections: Selections made at previous levels

        Returns:
            The reached nodes; empty when a selected value reaches no node
        """
        frontier = [self.tree]

        for prev_level in self.hierarchy_levels:
            if prev_level == target_level:
                break

            level_type = self.hierarchy_config["levels"][prev_level]["type"]

            # Instance levels: navigate INTO container but don't use selection
            if level_type == "instances":
                frontier = [self._enter_instance_container(node, prev_level) for node in frontier]
                continue

            values = [v for v in self._ensure_list(previous_selections.get(prev_level)) if v]
            if level_type != "tree" or not values:
                continue
            reached: dict[int, dict] = {}
            for value in values:
                for node in frontier:
                    child = self._follow_tree_value(node, value)
                    if child is not None:
                        reached.setdefault(id(child), child)
            frontier = list(reached.values())

        return frontier

    def _extract_tree_options(self, node: dict) -> list[dict[str, str]]:
        """
        Extract options from tree-structured node.

        For nodes with _expansion definitions, expands them inline and returns
        the expanded instances rather than the base container name. This ensures
        that at optional tree levels, only valid navigable options are presented.

        Example:
            If node contains:
            - "PSU": {...} (regular container) → returns "PSU"
            - "CH": {"_expansion": {...}} → returns "CH-1", "CH-2", etc.

        Returns:
            List of options with name and description
        """
        options = []
        for key, value in node.items():
            if not key.startswith("_") and isinstance(value, dict):
                # Check if this node has an expansion definition
                if "_expansion" in value:
                    # Expand inline and add expanded instances
                    expanded = self._expand_instances(value["_expansion"])
                    options.extend(expanded)
                else:
                    # Regular node - add as-is
                    options.append({"name": key, "description": value.get("_description", "")})
        return options

    def _get_expansion_options(self, node: dict, level: str) -> list[dict[str, str]]:
        """
        Get options from expansion definition at current level.

        Looks for a key matching the level name (case-insensitive) with _expansion definition.

        Args:
            node: Current node in tree
            level: Level name to find expansion for

        Returns:
            List of expanded instance options
        """
        # Look for level name key with expansion
        for key, value in node.items():
            if key.upper() == level.upper() and isinstance(value, dict) and "_expansion" in value:
                return self._expand_instances(value["_expansion"])

        # If not found, return empty (will cause navigation to fail)
        return []

    def _expand_instances(self, expansion_def: dict) -> list[dict[str, str]]:
        """
        Expand instance definition into list of options.

        Args:
            expansion_def: Dictionary with _type, _pattern/_instances, _range

        Returns:
            List of instance options
        """
        expansion_type = expansion_def.get("_type")
        options = []

        if expansion_type == "range":
            pattern = expansion_def.get("_pattern", "{}")
            start, end = expansion_def.get("_range", [1, 1])

            for i in range(start, end + 1):
                instance_name = pattern.format(i)
                options.append({"name": instance_name, "description": f"Instance {i}"})

        elif expansion_type == "list":
            instances = expansion_def.get("_instances", [])
            options.extend({"name": instance, "description": ""} for instance in instances)

        return options

    def validate_channel(self, channel: str) -> bool:
        """Check if a channel exists in the database."""
        return channel in self.channel_map

    def get_channel(self, channel_name: str) -> dict | None:
        """Get channel information."""
        channel_data = self.channel_map.get(channel_name)
        # Add address field if not present (use channel name as address)
        if channel_data and "address" not in channel_data:
            channel_data["address"] = channel_name
        return channel_data

    def get_all_channels(self) -> list[dict]:
        """Get all channels in the database."""
        return [
            {"channel": ch_name, "address": ch_data.get("address", ch_name), **ch_data}
            for ch_name, ch_data in self.channel_map.items()
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        stats: dict[str, Any] = {
            "total_channels": len(self.channel_map),
            "hierarchy_levels": self.hierarchy_levels,
        }

        # Count by first level (if it's a tree level)
        first_level = self.hierarchy_levels[0] if self.hierarchy_levels else None
        if first_level:
            first_config = self.hierarchy_config["levels"].get(first_level, {})
            if first_config.get("type") == "tree":
                stats["systems"] = []
                for system_name in self.tree:
                    if not system_name.startswith("_"):
                        system_channels = [
                            ch
                            for ch in self.channel_map.values()
                            if ch["path"].get(first_level) == system_name
                        ]
                        stats["systems"].append(
                            {"name": system_name, "channel_count": len(system_channels)}
                        )

        return stats
