"""Channel-name construction from the hierarchical tree.

Turns tree positions and level selections into fully-qualified channel names:
resolving naming-pattern placeholders and separators (including per-node
``_separator`` overrides and optional-level cleanup), expanding the whole tree
into the flat channel map, and building channels from explicit selections.
"""

import itertools
from collections.abc import Iterator
from typing import Any

from ._hierarchical_base import _HierarchicalBase

#: Most selection combinations ``build_existing_channels`` lists as holding no channel.
INVALID_EXAMPLE_LIMIT = 10


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
            part: str = node["_channel_part"]
            return part
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
        has_children = any(not key.startswith("_") and isinstance(node[key], dict) for key in node)
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
                        for key in node
                        if isinstance(node[key], dict) and not key.startswith("_")
                    )
                    # Only a leaf if there's NO container for the next instance level
                    return not has_next_level_container
                if next_config["type"] == "tree":
                    # Look for any tree children (non-meta keys)
                    has_tree_children = any(
                        not key.startswith("_") and isinstance(node[key], dict) for key in node
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
        return re.sub(r"[:_-]([:_-])+", lambda m: m.group(0)[0], channel)

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

        # Parse the naming pattern to extract literal text and placeholders
        # Pattern: "S{sector}:{building}:F{floor}"  # noqa: ERA001 -- a naming pattern
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
        # A pattern of bare placeholders puts no separator of its own between
        # the parts, so whatever separators the channel holds are the parts'
        # own characters and stay as written.
        if re.fullmatch(r"(?:\{\w+\})+", pattern):
            return channel
        return self._clean_optional_separators(channel)

    def build_channels_from_selections(self, selections: dict[str, Any]) -> list[str]:
        """
        Build fully-qualified channel names from hierarchical selections.

        Works with any number of levels - uses Cartesian product.
        Only includes levels that are referenced in the naming pattern.
        Handles optional levels by treating missing levels as empty strings.
        Each combination is walked through the tree on its own, so it takes the
        ``_channel_part`` and ``_separator`` values of its own branch. A name is
        built for every combination, whether or not it is a channel; see
        :meth:`build_existing_channels` for the ones that are.

        Args:
            selections: Dict mapping level names to selected values (strings or lists)

        Returns:
            List of complete channel names, one per combination
        """
        return [channel for _, channel in self._build_selection_combinations(selections)]

    def build_existing_channels(self, selections: dict[str, Any]) -> dict[str, Any]:
        """Build the channels a selection holds and count the combinations that hold none.

        Args:
            selections: Dict mapping level names to selected values (strings or lists)

        Returns:
            ``channels``: the built names that are channels of this database, each
            once, in first-seen order. ``total``: their count. ``invalid_count``:
            the number of distinct selection combinations whose name is not a
            channel. ``invalid_examples``: at most
            :data:`INVALID_EXAMPLE_LIMIT` of those combinations, each a dict of
            level name to selected value.

        Raises:
            ValueError: If a required level of the naming pattern is missing.
        """
        channels: dict[str, None] = {}
        invalid: dict[tuple[Any, ...], dict[str, Any]] = {}
        for combination, channel in self._build_selection_combinations(selections):
            if channel in self.channel_map:
                channels[channel] = None
            else:
                invalid.setdefault(tuple(combination.values()), combination)
        return {
            "channels": list(channels),
            "total": len(channels),
            "invalid_count": len(invalid),
            "invalid_examples": list(itertools.islice(invalid.values(), INVALID_EXAMPLE_LIMIT)),
        }

    def _build_selection_combinations(
        self, selections: dict[str, Any]
    ) -> Iterator[tuple[dict[str, Any], str]]:
        """Yield each selection combination with the channel name it builds.

        The product runs over every level that is in the naming pattern or has a
        selection, in hierarchy order, so a multi-value selection at a level the
        pattern leaves out is still walked value by value.

        Args:
            selections: Dict mapping level names to selected values (strings or lists)

        Yields:
            ``(combination, channel)``: the selected value of each level the
            caller selected, and the name built from them.

        Raises:
            ValueError: If a required level of the naming pattern is missing.
        """
        # Get levels that appear in naming pattern (may be subset of all hierarchy levels)
        pattern_levels = self._get_pattern_levels()

        levels: list[str] = []
        selection_lists: list[list[Any]] = []
        for level in self.hierarchy_levels:
            if level in pattern_levels:
                if level in selections:
                    values = self._ensure_list(selections[level])
                elif self.hierarchy_config["levels"][level].get("optional", False):
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
            elif level in selections:
                # Navigation only: an empty selection is no selection, not an empty product
                values = self._ensure_list(selections[level]) or [""]
            else:
                continue
            levels.append(level)
            selection_lists.append(values)

        for values_of_levels in itertools.product(*selection_lists):
            combination = dict(zip(levels, values_of_levels, strict=True))
            parts, separator_overrides = self._walk_selections(combination)
            params = {level: parts.get(level, combination[level]) for level in pattern_levels}
            channel = self._build_channel_with_separators(params, separator_overrides)
            yield {level: combination[level] for level in levels if level in selections}, channel

    def _separator_binding(self, node: dict, level_idx: int) -> tuple[tuple[str, str], str] | None:
        """Return the level pair a node's ``_separator`` binds to, with the separator.

        Args:
            node: Tree node, possibly carrying ``_separator``.
            level_idx: Index in ``hierarchy_levels`` of the level the node sits at.

        Returns:
            ``((level, next_tree_level), separator)``, or None when the node sets
            no separator or no tree level follows it.
        """
        if "_separator" not in node or level_idx >= len(self.hierarchy_levels):
            return None
        for next_level in self.hierarchy_levels[level_idx + 1 :]:
            if self.hierarchy_config["levels"][next_level]["type"] == "tree":
                return (self.hierarchy_levels[level_idx], next_level), node["_separator"]
        return None

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
            separator_overrides: dict[tuple[str, str], str],
        ):
            """Recursively expand tree with flexible level handling and custom separators."""
            # Check if this node specifies a custom separator for its children
            # Create a NEW dict for this subtree to avoid polluting siblings
            local_overrides = separator_overrides.copy()

            # The node sits at the level before ``level_idx``; the root sits at none
            binding = self._separator_binding(node, level_idx - 1) if level_idx > 0 else None
            if binding is not None:
                # Store the custom separator in LOCAL copy
                local_overrides[binding[0]] = binding[1]

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
                    if (
                        key.upper() == current_level.upper()
                        and isinstance(value, dict)
                        and "_expansion" in value
                    ):
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
        all _separator overrides encountered along the path. A multi-value
        selection is followed by its first value.

        Args:
            selections: Dict mapping level names to selected values

        Returns:
            Dict mapping (current_level, next_level) tuples to separator strings
        """
        return self._walk_selections(selections)[1]

    def _walk_selections(
        self, selections: dict[str, Any]
    ) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
        """Follow one selection per level down the tree, as the expansion descends it.

        A selection names tree keys and instance names, the names
        ``get_options_at_level`` offers; ``_build_channel_map`` builds the
        channel from each node's ``_channel_part`` and the ``_separator``
        overrides on the way. The walk collects both for the levels it can
        follow. It skips an optional level with no selection and stops at the
        first level it cannot follow; from there on, values stay as given.

        Args:
            selections: Dict mapping level names to selected values; a list is
                followed by its first value.

        Returns:
            The channel part of each followed tree key, by level, and the dict
            mapping (current_level, next_level) tuples to separator strings.
        """
        channel_parts: dict[str, str] = {}
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
                # Required level missing - can't navigate further
                break

            selection = self._get_single_value(selections[level])
            if not selection:
                # Empty selection
                if is_optional:
                    # Optional level with empty selection - skip it
                    continue
                # Required level with empty selection - can't navigate further
                break

            # Handle different level types
            if level_config["type"] == "tree":
                # Navigate using the tree key
                child = (
                    current_node.get(selection)
                    if isinstance(selection, str) and not selection.startswith("_")
                    else None
                )
                if isinstance(child, dict):
                    # Direct match - navigate to this node
                    channel_parts[level] = self._get_channel_part(child, selection)
                    current_node = child
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
            binding = self._separator_binding(current_node, level_idx)
            if binding is not None:
                separator_overrides[binding[0]] = binding[1]

        return channel_parts, separator_overrides
