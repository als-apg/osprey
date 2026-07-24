"""Navigation and read access over the hierarchical database.

Provides the query side of the public API: enumerating the options available at
a hierarchy level (expanding instance ranges/lists as needed), navigating the
tree by prior selections, channel lookup/validation over the flat map, and
database statistics.
"""

from typing import Any

from ._hierarchical_base import _HierarchicalBase


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
        """
        # Navigate to current position in tree (skipping instance levels)
        current_node = self._navigate_to_node(level, previous_selections)

        if not current_node:
            return []

        # Get level configuration
        level_config = self.hierarchy_config["levels"][level]
        level_type = level_config["type"]

        # Extract options based on level type
        if level_type == "tree":
            # Direct children of current node
            options = self._extract_tree_options(current_node)

            # OPTIONAL LEVEL BEHAVIOR: For optional tree levels, INCLUDE both:
            # 1. Containers (subdevices, subsystems, etc.) - these lead to deeper navigation
            # 2. Leaf nodes (direct signals) - these skip this optional level
            # This allows the LLM to naturally select either a subdevice (PSU, ADC, etc.)
            # OR a direct signal (Heartbeat, Status, etc.) without needing to reason about
            # "NOTHING_FOUND". The navigation logic will handle both cases appropriately.

            return options

        elif level_type == "instances":
            # Find expansion definition for this level
            return self._get_expansion_options(current_node, level)

        return []

    def _navigate_to_node(
        self, target_level: str, previous_selections: dict[str, Any]
    ) -> dict | None:
        """
        Navigate to current node in tree based on previous selections.

        Key behavior: Instance levels do NOT change tree position during selection,
        but we DO navigate INTO their containers to find children.

        Args:
            target_level: Level we're getting options for
            previous_selections: Selections made at previous levels

        Returns:
            Current node in tree, or None if path invalid
        """
        current_node = self.tree

        # Navigate through previous levels
        for prev_level in self.hierarchy_levels:
            if prev_level == target_level:
                break

            level_config = self.hierarchy_config["levels"][prev_level]

            # Instance levels: navigate INTO container but don't use selection
            if level_config["type"] == "instances":
                # Find and enter the container for this instance level
                for key, value in current_node.items():
                    if key.upper() == prev_level.upper() and isinstance(value, dict):
                        current_node = value
                        break
                continue

            # Tree levels - navigate down using selection
            if level_config["type"] == "tree":
                if prev_level in previous_selections:
                    selection = self._get_single_value(previous_selections[prev_level])

                    if selection and selection in current_node:
                        current_node = current_node[selection]
                    else:
                        # No direct match - check if selection is an expanded instance
                        # (e.g., selection="CH-1" but tree has "CH" with _expansion)
                        found_via_expansion = False
                        for key, value in current_node.items():
                            if (
                                not key.startswith("_")
                                and isinstance(value, dict)
                                and "_expansion" in value
                            ):
                                # This node has an expansion - check if it generates our selection
                                instance_names = self._get_instance_names(value["_expansion"])
                                if selection in instance_names:
                                    # Found the container that expands to our selected instance
                                    current_node = value
                                    found_via_expansion = True
                                    break

                        if not found_via_expansion:
                            return None  # Invalid path

        return current_node

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
            if key.upper() == level.upper() and isinstance(value, dict):
                if "_expansion" in value:
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
            for instance in instances:
                options.append({"name": instance, "description": ""})

        return options

    def validate_channel(self, channel: str) -> bool:
        """Check if a channel exists in the database."""
        return channel in self.channel_map

    def get_channel(self, channel_name: str) -> dict | None:
        """Get channel information."""
        channel_data = self.channel_map.get(channel_name)
        if channel_data:
            # Add address field if not present (use channel name as address)
            if "address" not in channel_data:
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
                for system_name in self.tree.keys():
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
