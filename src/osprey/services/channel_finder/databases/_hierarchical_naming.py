"""Channel-name construction from the hierarchical tree.

Turns tree positions and level selections into fully-qualified channel names:
resolving naming-pattern placeholders and separators (including per-node
``_separator`` overrides and optional-level cleanup), expanding the whole tree
into the flat channel map, and building channels from explicit selections.
"""

import itertools
from typing import Any

from ._hierarchical_base import _HierarchicalBase


class _HierarchicalNamingMixin(_HierarchicalBase):
    """Channel-name construction and flat channel-map expansion."""

    def _get_pattern_levels(self) -> list[str]:
        """
        Extract level names referenced in naming pattern, in order of appearance.

        Returns:
            List of level names that appear as placeholders in naming_pattern,
            in the order they appear in hierarchy_levels (not pattern order).
        """
        import re

        # Extract all placeholders from pattern
        pattern_placeholders = set(re.findall(r"\{(\w+)\}", self.naming_pattern))

        # Return in hierarchy order (not pattern order) for consistent Cartesian product
        return [level for level in self.hierarchy_levels if level in pattern_placeholders]

    def _get_channel_part(self, node: dict, tree_key: str) -> str:
        """
        Get the channel name component for a tree node.

        Supports decoupling tree keys (for navigation) from naming components.
        This enables human-readable tree keys while maintaining technical naming conventions.

        Args:
            node: Tree node dictionary
            tree_key: The key used in the tree structure

        Returns:
            Channel name component:
            - node['_channel_part'] if specified (explicit override)
            - Empty string if _channel_part is explicitly ""  (skip in naming)
            - tree_key if _channel_part not specified (backward compatible default)

        Examples:
            # Backward compatible - uses tree key
            {"MAG": {}} → "MAG"

            # Friendly tree key, technical naming
            {"Magnets": {"_channel_part": "MAG"}} → "MAG"

            # Skip in naming (navigation only)
            {"Building-1": {"_channel_part": ""}} → ""
        """
        if "_channel_part" in node:
            return node["_channel_part"]
        return tree_key  # Default: backward compatible

    def _is_leaf_node(self, node: dict, current_level_idx: int) -> bool:
        """
        Detect if a node represents a complete channel (leaf node).

        Supports optional hierarchy levels by allowing channels to terminate
        before all hierarchy levels are traversed.

        Criteria for leaf detection:
        1. Explicit _is_leaf marker set to True (for nodes with children that are also leaves)
        2. Node has no children (automatic leaf detection - no _is_leaf needed)
        3. All hierarchy levels processed (reached end)
        4. All remaining levels are optional AND node has no children for those levels

        Args:
            node: Current tree node
            current_level_idx: Current position in hierarchy_levels

        Returns:
            True if node is a complete channel, False otherwise

        Examples:
            # Explicit marker (node with children that is also a leaf)
            {"SIGNAL-Y": {"_is_leaf": True, "RB": {...}}} → True

            # Automatic detection (no children, no _is_leaf needed)
            {"RB": {"_description": "Readback"}} → True

            # All levels processed
            (current_level_idx >= len(hierarchy_levels)) → True

            # All remaining optional AND no children
            (subdevice optional, no SUBDEVICE container) → True
        """
        # Explicit _is_leaf marker takes precedence
        if node.get("_is_leaf", False):
            return True

        # AUTOMATIC LEAF DETECTION: Node has no children (only metadata)
        # This eliminates the need for explicit _is_leaf on obvious leaf nodes
        has_children = any(
            not key.startswith("_") and isinstance(node[key], dict) for key in node.keys()
        )
        if not has_children:
            return True

        # All hierarchy levels processed - definitely a leaf
        if current_level_idx >= len(self.hierarchy_levels):
            return True

        # Check if all remaining levels are optional AND no children for next level
        remaining_levels = self.hierarchy_levels[current_level_idx:]
        if remaining_levels:
            all_remaining_optional = all(
                self.hierarchy_config["levels"][level].get("optional", False)
                for level in remaining_levels
            )
            if all_remaining_optional:
                # Check if node has children for the next level
                next_level = remaining_levels[0]
                next_config = self.hierarchy_config["levels"][next_level]

                if next_config["type"] == "instances":
                    # Look for instance container matching next level name
                    has_next_level_container = any(
                        key.upper() == next_level.upper()
                        for key in node.keys()
                        if isinstance(node[key], dict) and not key.startswith("_")
                    )
                    # Only a leaf if there's NO container for the next instance level
                    return not has_next_level_container
                elif next_config["type"] == "tree":
                    # Look for any tree children (non-meta keys)
                    has_tree_children = any(
                        not key.startswith("_") and isinstance(node[key], dict)
                        for key in node.keys()
                    )
                    # Only a leaf if there are NO tree children
                    return not has_tree_children

        return False

    def _clean_optional_separators(self, channel: str) -> str:
        """
        Remove artifacts from skipped optional levels in channel names.

        When optional levels are skipped, their separators may leave artifacts
        like consecutive delimiters (::, --) or trailing separators.
        This method cleans up these artifacts to produce valid channel names.

        Args:
            channel: Raw channel name with potential separator artifacts

        Returns:
            Cleaned channel name with artifacts removed

        Examples:
            # Double colons from skipped subdevice
            "SYSTEM:DEV01::SIGNAL" → "SYSTEM:DEV01:SIGNAL"

            # Trailing underscore from missing suffix
            "SYSTEM:SIGNAL_" → "SYSTEM:SIGNAL"

            # Multiple consecutive dashes
            "SYSTEM--SUBSYS-DEV" → "SYSTEM-SUBSYS-DEV"

            # Leading separator from skipped first optional
            ":SYSTEM:SIGNAL" → "SYSTEM:SIGNAL"
        """
        import re

        # Multiple consecutive separators of same type
        channel = re.sub(r":{2,}", ":", channel)  # :: → :
        channel = re.sub(r"-{2,}", "-", channel)  # -- → -
        channel = re.sub(r"_{2,}", "_", channel)  # __ → _

        # Trailing separators
        channel = re.sub(r"[:_-]+$", "", channel)  # Remove trailing : _ -

        # Leading separators (except intentional ones)
        channel = re.sub(r"^[_:]", "", channel)  # Remove leading _ :

        # Mixed consecutive separators (e.g., ":_" when both parts empty)
        # Keep the first separator type
        channel = re.sub(r"[:_-]([:_-])+", lambda m: m.group(0)[0], channel)

        return channel

    def _find_separator_between_levels(
        self, start_idx: int, end_idx: int, pattern_levels: list[str]
    ) -> str:
        """
        Find the appropriate separator between two non-consecutive pattern levels.

        When optional levels are skipped, we need to find which separator to use.
        We use the first separator encountered when walking from start to end.

        Args:
            start_idx: Index of the starting level (last non-empty)
            end_idx: Index of the ending level (current non-empty)
            pattern_levels: List of all pattern levels in order

        Returns:
            Separator string to use

        Example:
            Levels: [system, device, subdevice, signal]
            Pattern: {system}-{device}:{subdevice}:{signal}
            If subdevice is empty and we're connecting device (idx=1) to signal (idx=3):
            - Check device→subdevice separator: ":"
            - Return ":"
        """
        # Walk through levels from start to end and find first separator
        for i in range(start_idx, end_idx):
            current_level = pattern_levels[i]
            next_level = pattern_levels[i + 1]
            sep_key = (current_level, next_level)

            if sep_key in self.default_separators:
                return self.default_separators[sep_key]

        # Fallback to colon if no separator found
        return ":"

    def _build_channel_with_separators(
        self, path: dict[str, str], separator_overrides: dict[tuple[str, str], str]
    ) -> str:
        """
        Build channel name using path components and custom separators.

        This method constructs channel names by joining path components with
        separators, respecting both:
        - Literal prefixes/suffixes from naming pattern (e.g., S{sector} → S01)
        - Custom separator overrides from _separator metadata in tree nodes

        Args:
            path: Dict mapping level names to values
            separator_overrides: Dict mapping (level, next_level) tuples to custom separators

        Returns:
            Complete channel name with correct separators and literals

        Raises:
            KeyError: If a placeholder in the naming pattern doesn't have a value in path
                     (except for optional levels with empty string values)

        Examples:
            # Pattern with literal prefixes
            path = {"sector": "01", "building": "MAIN", "floor": "1"}
            pattern = "S{sector}:{building}:F{floor}"
            → "S01:MAIN:F1"

            # With custom separator override
            path = {"device": "DEV-01", "signal": "Mode", "suffix": "RB"}
            pattern = "{device}:{signal}:{suffix}"
            overrides = {("signal", "suffix"): "_"}
            → "DEV-01:Mode_RB"  (uses custom _ instead of default :)
        """
        import re

        pattern = self.naming_pattern
        self._get_pattern_levels()

        # Parse the naming pattern to extract literal text and placeholders
        # Pattern: "S{sector}:{building}:F{floor}"
        # → [('S', 'sector'), (':', 'building'), (':F', 'floor')]
        # where each tuple is (prefix_text, level_name)

        # Find all placeholders and text between them
        placeholder_pattern = r"\{(\w+)\}"
        matches = list(re.finditer(placeholder_pattern, pattern))

        # Build a list of (prefix, level_name) pairs
        pattern_parts = []
        for i, match in enumerate(matches):
            level_name = match.group(1)

            # Get the text before this placeholder
            if i == 0:
                # First placeholder - get everything before it
                prefix = pattern[: match.start()]
            else:
                # Get text between previous placeholder end and this one's start
                prefix = pattern[matches[i - 1].end() : match.start()]

            pattern_parts.append((prefix, level_name))

        # Build the channel name
        result_parts = []
        last_level_with_value = None

        for _, (prefix, level_name) in enumerate(pattern_parts):
            # Check if level exists in path (KeyError if not)
            if level_name not in path:
                raise KeyError(level_name)

            value = path.get(level_name, "")

            if value:  # Only include if value is non-empty
                # Check if we need to apply a separator override
                if last_level_with_value is not None:
                    sep_key = (last_level_with_value, level_name)
                    if sep_key in separator_overrides:
                        # Replace the separator in the prefix with the override
                        # The prefix contains the separator from the pattern
                        result_parts.append(separator_overrides[sep_key])
                        result_parts.append(value)
                    else:
                        # Use the prefix as-is from the pattern
                        result_parts.append(prefix)
                        result_parts.append(value)
                else:
                    # First value - include prefix and value
                    result_parts.append(prefix)
                    result_parts.append(value)

                last_level_with_value = level_name

        channel = "".join(result_parts)
        return self._clean_optional_separators(channel)

    def build_channels_from_selections(self, selections: dict[str, Any]) -> list[str]:
        """
        Build fully-qualified channel names from hierarchical selections.

        Works with any number of levels - uses Cartesian product.
        Only includes levels that are referenced in the naming pattern.
        Handles optional levels by treating missing levels as empty strings.
        Respects _separator overrides from tree nodes.

        Args:
            selections: Dict mapping level names to selected values (strings or lists)

        Returns:
            List of complete channel names
        """
        # Get levels that appear in naming pattern (may be subset of all hierarchy levels)
        pattern_levels = self._get_pattern_levels()

        # Convert selections for pattern levels to lists for uniform handling
        selection_lists = []
        for level in pattern_levels:
            if level in selections:
                # Level provided in selections
                values = self._ensure_list(selections.get(level, []))
            else:
                # Level not in selections - check if it's optional
                level_config = self.hierarchy_config["levels"][level]
                is_optional = level_config.get("optional", False)

                if is_optional:
                    # Optional level not provided - use empty string
                    values = [""]
                else:
                    # Required level missing - raise so the MCP tool can surface
                    # which level is absent (silent [] confuses the agent)
                    raise ValueError(
                        f"Required level '{level}' is missing from selections. "
                        f"All required levels must be provided: {pattern_levels}. "
                        f"Received: {list(selections.keys())}"
                    )

            selection_lists.append(values)

        # Collect separator overrides by navigating the tree with selections
        separator_overrides = self._collect_separator_overrides(selections)

        # Generate Cartesian product of all selections
        channels = []
        for combination in itertools.product(*selection_lists):
            # Build channel name with separator overrides
            params = dict(zip(pattern_levels, combination, strict=False))
            channel = self._build_channel_with_separators(params, separator_overrides)

            channels.append(channel)

        return channels

    def _build_channel_map(self) -> dict[str, dict]:
        """
        Expand hierarchical tree into flat channel map.

        Works with flexible hierarchy configuration.
        Only includes pattern-referenced levels in channel names.

        Returns:
            Dict mapping channel names to channel info
        """
        channels = {}
        pattern_levels = self._get_pattern_levels()

        def expand_tree(
            path: dict[str, str],
            node: dict,
            level_idx: int,
            separator_overrides: dict[tuple[str, str], str] | None = None,
        ):
            """Recursively expand tree with flexible level handling and custom separators."""
            if separator_overrides is None:
                separator_overrides = {}

            # Check if this node specifies a custom separator for its children
            # Create a NEW dict for this subtree to avoid polluting siblings
            local_overrides = separator_overrides.copy()

            if "_separator" in node and level_idx > 0:
                # Get current level (where this node is assigned)
                current_level_idx = level_idx - 1
                if current_level_idx < len(self.hierarchy_levels):
                    current_level = self.hierarchy_levels[current_level_idx]

                    # Find next tree level for children
                    for next_idx in range(level_idx, len(self.hierarchy_levels)):
                        next_config = self.hierarchy_config["levels"][
                            self.hierarchy_levels[next_idx]
                        ]
                        if next_config["type"] == "tree":
                            next_level = self.hierarchy_levels[next_idx]
                            # Store the custom separator in LOCAL copy
                            local_overrides[(current_level, next_level)] = node["_separator"]
                            break

            # Check for leaf nodes (supports optional hierarchy levels)
            if self._is_leaf_node(node, level_idx):
                # Build channel from path with custom separators
                # First ensure all pattern levels have values (empty string for skipped optional)
                complete_path = path.copy()
                for level in pattern_levels:
                    if level not in complete_path:
                        complete_path[level] = ""

                # Build channel name using custom separators
                channel_name = self._build_channel_with_separators(complete_path, local_overrides)

                # Store channel
                channels[channel_name] = {"channel": channel_name, "path": path.copy()}

                # Process children of leaf nodes to handle optional levels
                # A node can be both a complete channel AND have children for optional suffixes
                # Example: SIGNAL-Y is a valid channel, but may also have RB/SP suffix variants
                if level_idx < len(self.hierarchy_levels):
                    # Check if node has children (non-meta keys)
                    has_children = any(
                        not k.startswith("_") and isinstance(v, dict) for k, v in node.items()
                    )

                    if has_children:
                        # Find next tree level after current position
                        next_tree_level_idx = None
                        for idx in range(level_idx, len(self.hierarchy_levels)):
                            next_config = self.hierarchy_config["levels"][
                                self.hierarchy_levels[idx]
                            ]
                            if next_config["type"] == "tree":
                                next_tree_level_idx = idx
                                break

                        if next_tree_level_idx is not None:
                            # Process children as nodes at the next tree level
                            next_level = self.hierarchy_levels[next_tree_level_idx]
                            children = {
                                k: v
                                for k, v in node.items()
                                if not k.startswith("_") and isinstance(v, dict)
                            }

                            for child_key, child_node in children.items():
                                channel_part = self._get_channel_part(child_node, child_key)
                                # Assign child to next tree level
                                expand_tree(
                                    {**path, next_level: channel_part},
                                    child_node,
                                    next_tree_level_idx + 1,
                                    local_overrides,
                                )

                            # Children processed, return
                            return

                # No children or all levels processed
                if level_idx >= len(self.hierarchy_levels):
                    return

            # Base case: processed all levels (already handled above if leaf)
            if level_idx >= len(self.hierarchy_levels):
                return

            current_level = self.hierarchy_levels[level_idx]
            level_config = self.hierarchy_config["levels"][current_level]

            # Handle based on level type
            if level_config["type"] == "tree":
                # Tree navigation: iterate direct children
                children = {
                    k: v for k, v in node.items() if not k.startswith("_") and isinstance(v, dict)
                }

                is_optional_level = level_config.get("optional", False)

                for child_key, child_node in children.items():
                    # Get channel part (supports _channel_part override)
                    channel_part = self._get_channel_part(child_node, child_key)

                    # Handle hybrid pattern: tree node with instance expansion
                    # Allows tree categories that expand into multiple numbered instances
                    # Example: A tree node "SUBDEV" that expands to SUBDEV-01, SUBDEV-02, etc.
                    if "_expansion" in child_node:
                        # Tree node with instance expansion - expand instances
                        instances = self._get_instance_names(child_node["_expansion"])
                        for instance_name in instances:
                            # Assign instance to current level and recurse
                            expand_tree(
                                {**path, current_level: instance_name},
                                child_node,
                                level_idx + 1,
                                local_overrides,
                            )
                        # Don't process this node as regular tree node
                        continue

                    # OPTIONAL LEVEL SKIP: If this is an optional level and the child is a leaf,
                    # the child should skip this level and be assigned to the NEXT tree level instead
                    if is_optional_level and child_node.get("_is_leaf", False):
                        # Find next tree level to assign this child to
                        next_level_idx = level_idx + 1
                        if next_level_idx < len(self.hierarchy_levels):
                            next_level = self.hierarchy_levels[next_level_idx]
                            # Skip current optional level, assign child to next level
                            expand_tree(
                                {**path, next_level: channel_part},
                                child_node,
                                next_level_idx + 1,
                                local_overrides,
                            )
                        else:
                            # No next level exists, expand normally
                            expand_tree(path, child_node, level_idx + 1, local_overrides)
                    else:
                        # Normal tree expansion: assign child to current level
                        expand_tree(
                            {**path, current_level: channel_part},
                            child_node,
                            level_idx + 1,
                            local_overrides,
                        )

            elif level_config["type"] == "instances":
                # Expansion: find expansion definition and generate instances
                expansion_def = None
                child_node = node  # Stay at same node

                for key, value in node.items():
                    if key.upper() == current_level.upper() and isinstance(value, dict):
                        if "_expansion" in value:
                            expansion_def = value["_expansion"]
                            # Navigate past the expansion container
                            child_node = value
                            break

                if expansion_def:
                    instances = self._get_instance_names(expansion_def)
                    for instance_name in instances:
                        expand_tree(
                            {**path, current_level: instance_name},
                            child_node,
                            level_idx + 1,
                            local_overrides,
                        )

        expand_tree({}, self.tree, 0, {})
        return channels

    def _collect_separator_overrides(
        self, selections: dict[str, Any]
    ) -> dict[tuple[str, str], str]:
        """
        Collect separator overrides from tree nodes based on selections.

        Navigates through the tree following the provided selections and collects
        all _separator overrides encountered along the path.

        Args:
            selections: Dict mapping level names to selected values

        Returns:
            Dict mapping (current_level, next_level) tuples to separator strings
        """
        separator_overrides = {}
        current_node = self.tree

        # Navigate through each hierarchy level
        for level_idx, level in enumerate(self.hierarchy_levels):
            level_config = self.hierarchy_config["levels"][level]
            is_optional = level_config.get("optional", False)

            # Check if we have a selection at this level
            if level not in selections:
                # No selection at this level
                if is_optional:
                    # Optional level - skip it and continue
                    continue
                else:
                    # Required level missing - can't navigate further
                    break

            selection = self._get_single_value(selections[level])
            if not selection:
                # Empty selection
                if is_optional:
                    # Optional level with empty selection - skip it
                    continue
                else:
                    # Required level with empty selection - can't navigate further
                    break

            # Handle different level types
            if level_config["type"] == "tree":
                # Navigate using the tree key
                if selection in current_node:
                    # Direct match - navigate to this node
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
                        # Selection not found in tree - stop navigation
                        break

            elif level_config["type"] == "instances":
                # Instance levels don't change tree position, but we navigate INTO the container
                #
                # Handle two cases:
                # 1. Selection is an expanded instance (e.g., "CH-1") - find container with matching expansion
                # 2. Selection is a container name (e.g., "DEVICE") - find matching container key

                found_container = False

                # First, try to find a container with an expansion that generates this selection
                for key, value in current_node.items():
                    if (
                        not key.startswith("_")
                        and isinstance(value, dict)
                        and "_expansion" in value
                    ):
                        # This container has an expansion - check if it generates our selection
                        instance_names = self._get_instance_names(value["_expansion"])
                        if selection in instance_names:
                            # Found the container that expands to our selected instance
                            current_node = value
                            found_container = True
                            break

                # If not found via expansion, try matching container key (old logic for compatibility)
                if not found_container:
                    for key, value in current_node.items():
                        if key.upper() == level.upper() and isinstance(value, dict):
                            current_node = value
                            found_container = True
                            break

                if not found_container:
                    # No container found - stop navigation
                    break

            # After navigating to the node, check if it has a separator override for its children
            if "_separator" in current_node:
                # This node's separator applies to the connection between current level and next tree level
                # Find the next tree level
                for next_idx in range(level_idx + 1, len(self.hierarchy_levels)):
                    next_config = self.hierarchy_config["levels"][self.hierarchy_levels[next_idx]]
                    if next_config["type"] == "tree":
                        next_level = self.hierarchy_levels[next_idx]
                        separator_overrides[(level, next_level)] = current_node["_separator"]
                        break

        return separator_overrides
