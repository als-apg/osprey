"""
Hierarchical Channel Database

Loads and navigates hierarchical channel structures for iterative LLM-based refinement.
Supports flexible hierarchy with arbitrary mixing of:
- Tree levels (semantic categories)
- Instance levels (numbered/patterned expansions)
"""

import itertools
import json
import logging
from typing import Any

from ..core.base_database import BaseDatabase

logger = logging.getLogger(__name__)


class HierarchicalChannelDatabase(BaseDatabase):
    """
    Database for hierarchical channel naming schemes.

    Supports flexible hierarchy with arbitrary mixing of:
    - Tree levels (semantic categories)
    - Instance levels (numbered/patterned expansions)
    """

    def __init__(self, db_path: str):
        """
        Initialize hierarchical database.

        Args:
            db_path: Path to hierarchical database JSON file
        """
        super().__init__(db_path)

    def load_database(self):
        """Load hierarchical database from JSON with flexible configuration."""
        import warnings

        with open(self.db_path) as f:
            data = json.load(f)

        self.tree = data["tree"]

        # Support new unified schema (preferred) or legacy three-field format (deprecated)
        if "hierarchy" in data:
            # NEW UNIFIED SCHEMA: Single "hierarchy" section
            hierarchy_def = data["hierarchy"]

            # Extract levels list and build derived structures
            levels_list = hierarchy_def["levels"]
            self.hierarchy_levels = [level["name"] for level in levels_list]
            self.naming_pattern = hierarchy_def["naming_pattern"]

            # Build hierarchy_config from levels list
            self.hierarchy_config = {"levels": {}}
            for level_def in levels_list:
                self.hierarchy_config["levels"][level_def["name"]] = {
                    "type": level_def["type"],
                    "optional": level_def.get("optional", False),
                }

            # Validate naming_pattern references correct level names
            self._validate_naming_pattern()

        elif "hierarchy_definition" in data:
            # LEGACY/INTERMEDIATE FORMAT: Backward compatibility for old schemas
            self.hierarchy_levels = data["hierarchy_definition"]
            self.naming_pattern = data["naming_pattern"]

            # Load or infer hierarchy configuration
            if "hierarchy_config" in data:
                # Intermediate format (unpublished) - silent backward compatibility
                self.hierarchy_config = data["hierarchy_config"]
            else:
                # TRULY LEGACY FORMAT: Missing hierarchy_config entirely
                # This is the old container-based format with devices/fields/subfields
                warnings.warn(
                    "Legacy hierarchical database format detected (missing 'hierarchy_config' section). "
                    "Automatic conversion applied, but explicit configuration is recommended. "
                    "The new flexible hierarchy format was introduced in Osprey 0.9.4. "
                    "Please update your database by:\n"
                    "  1. Adding 'hierarchy_config' section with explicit level types\n"
                    "  2. Migrating from nested containers (devices/fields/subfields) to DEVICE/FIELD/SUBFIELD with _expansion\n"
                    "See data/channel_databases/hierarchical.json for the new format, "
                    "or data/channel_databases/examples/hierarchical_legacy.json for the legacy format reference.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.hierarchy_config = self._infer_legacy_config()
        else:
            raise ValueError(
                "Invalid database format: must contain either 'hierarchy' (new unified schema) "
                "or 'hierarchy_definition' (legacy format)"
            )

        # Validate configuration
        self._validate_hierarchy_config()

        # Parse default separators from naming pattern
        self.default_separators = self._parse_naming_pattern_separators()

        # Build flat channel map for validation and lookup
        self.channel_map = self._build_channel_map()

    def _infer_legacy_config(self) -> dict:
        """
        Infer hierarchy configuration for legacy databases.

        Assumes traditional accelerator pattern:
        - First 2 levels: tree-based categories (system, family)
        - Remaining levels: container-based instances (device, field, subfield)
        """
        config = {"levels": {}}

        # Map legacy container keys
        legacy_container_keys = {"device": "devices", "field": "fields", "subfield": "subfields"}

        for i, level in enumerate(self.hierarchy_levels):
            if i < 2:
                # First two levels: tree-based categories
                config["levels"][level] = {"type": "tree"}
            else:
                # Later levels: container-based (legacy style)
                config["levels"][level] = {
                    "type": "container",  # Legacy mode
                    "container_key": legacy_container_keys.get(level, f"{level}s"),
                }

        return config

    def _validate_naming_pattern(self):
        """
        Validate that naming_pattern references valid level names.

        Pattern placeholders must be a subset of hierarchy levels.
        Not all hierarchy levels need to appear in the pattern - some levels
        may be used only for navigation/organization.

        Prevents out-of-sync errors between level names and naming pattern.
        """
        import re

        # Extract placeholder names from naming pattern (e.g., {system}, {family}, etc.)
        pattern_placeholders = set(re.findall(r"\{(\w+)\}", self.naming_pattern))
        expected_placeholders = set(self.hierarchy_levels)

        # Pattern placeholders must be subset of hierarchy levels
        # (but hierarchy can have extra levels not used in pattern)
        if not pattern_placeholders.issubset(expected_placeholders):
            extra = pattern_placeholders - expected_placeholders

            error_msg = "naming_pattern references undefined hierarchy levels:\n"
            error_msg += f"  Undefined levels in pattern: {sorted(extra)}\n"
            error_msg += f"  Defined hierarchy levels: {self.hierarchy_levels}\n"
            error_msg += f"  Pattern: {self.naming_pattern}\n\n"
            error_msg += (
                "All placeholders in naming_pattern must correspond to defined hierarchy levels."
            )

            raise ValueError(error_msg)

        # Info message if some levels are not used in pattern (navigation-only levels)
        unused_levels = expected_placeholders - pattern_placeholders
        if unused_levels:
            logger.info(
                f"Note: {len(unused_levels)} hierarchy level(s) not used in naming pattern: {sorted(unused_levels)}. "
                "These levels will be used for navigation only."
            )

        # Validate optional levels are in pattern
        # (Optional levels must be in pattern since they're conditionally included)
        for level in self.hierarchy_levels:
            level_config = self.hierarchy_config["levels"][level]
            is_optional = level_config.get("optional", False)

            if is_optional and level not in pattern_placeholders:
                raise ValueError(
                    f"Optional level '{level}' must appear in naming_pattern.\n"
                    f"Optional levels are conditionally included in channel names, "
                    f"so they need a placeholder in the pattern.\n"
                    f"Current pattern: {self.naming_pattern}\n"
                    f"Missing placeholder: {{{level}}}\n\n"
                    f"Note: If you want a navigation-only level (not in channel names), "
                    f"don't mark it as optional - just omit it from the pattern."
                )

    def _validate_hierarchy_config(self):
        """
        Validate hierarchy configuration structure with helpful error messages.

        Checks:
        1. Configuration structure is valid
        2. All levels are configured
        3. Each level has required fields
        4. Field values are valid
        5. Tree structure matches configuration
        """
        if "levels" not in self.hierarchy_config:
            raise ValueError("hierarchy_config must contain 'levels' key")

        # Check all levels are configured
        for level in self.hierarchy_levels:
            if level not in self.hierarchy_config["levels"]:
                raise ValueError(
                    f"Level '{level}' not found in hierarchy_config.\n"
                    f"All levels from hierarchy_definition must be configured.\n"
                    f"Expected levels: {self.hierarchy_levels}\n"
                    f"Configured levels: {list(self.hierarchy_config['levels'].keys())}"
                )

        # Validate each level config
        for level, config in self.hierarchy_config["levels"].items():
            if "type" not in config:
                raise ValueError(
                    f"Level '{level}' missing required 'type' property.\n"
                    f'Add: "type": "tree" or "instances"\n'
                    f"  - tree: Semantic categories with direct children\n"
                    f"  - instances: Numbered/patterned instances that share structure"
                )

            if config["type"] not in ["tree", "instances", "container"]:
                raise ValueError(
                    f"Level '{level}' has invalid type: '{config['type']}'.\n"
                    f"Must be 'tree', 'instances', or 'container' (legacy).\n"
                    f"Did you mean 'tree' or 'instances'?"
                )

        # Validate tree structure matches configuration
        self._validate_tree_structure()

    def _validate_tree_structure(self):
        """
        Validate tree structure matches hierarchy configuration.

        Checks:
        1. Instance levels have matching containers
        2. Instance containers have _expansion definitions
        3. Expansion definitions are valid
        4. Consecutive instances are properly nested
        """
        for level_idx, level in enumerate(self.hierarchy_levels):
            level_config = self.hierarchy_config["levels"][level]

            # Validate instance levels
            if level_config["type"] == "instances":
                self._validate_instance_level(level, level_idx)

    def _validate_instance_level(self, level_name: str, level_idx: int):
        """
        Validate instance level has proper container and expansion.

        Args:
            level_name: Name of the level to validate
            level_idx: Index in hierarchy_levels
        """
        level_config = self.hierarchy_config["levels"][level_name]
        is_optional = level_config.get("optional", False)

        # Find the container for this level
        container = self._find_level_container(self.tree, level_name, level_idx)

        if not container:
            # Optional levels don't need to have containers in all branches
            if is_optional:
                logger.info(
                    f"Optional instance level '{level_name}' has no container in some branches. "
                    f"This is acceptable for optional levels."
                )
                return

            # Required levels must have containers
            # Helpful error message with suggestions
            raise ValueError(
                f"Instance level '{level_name}' requires container named '{level_name.upper()}' in tree.\n\n"
                f"Expected structure:\n"
                f'  "tree": {{\n'
                f'    "{level_name.upper()}": {{\n'
                f'      "_expansion": {{...}},\n'
                f"      ...(children for next level)\n"
                f"    }}\n"
                f"  }}\n\n"
                f"Troubleshooting:\n"
                f"  1. Check that container name matches level name (case-insensitive)\n"
                f"  2. Verify container is at correct nesting depth\n"
                f"  3. Ensure previous levels are properly configured\n"
                f"  4. If this level is optional, add '\"optional\": true' to its configuration"
            )

        # Validate expansion definition exists
        if "_expansion" not in container:
            # Get the path to this container for error message
            path = self._get_container_path(level_name, level_idx)

            raise ValueError(
                f"Instance level '{level_name}' container missing '_expansion' definition.\n\n"
                f"Found container at: {path}\n"
                f"Missing: {path}['_expansion']\n\n"
                f"Add expansion definition:\n"
                f'  "_expansion": {{\n'
                f'    "_type": "range",\n'
                f'    "_pattern": "{{:02d}}",  // or "{{}}"\n'
                f'    "_range": [1, 10]  // [start, end] inclusive\n'
                f"  }}\n\n"
                f"Or for list-based:\n"
                f'  "_expansion": {{\n'
                f'    "_type": "list",\n'
                f'    "_instances": ["A", "B", "C"]\n'
                f"  }}"
            )

        # Validate expansion definition format
        expansion = container["_expansion"]
        self._validate_expansion_definition(expansion, level_name)

        # Check for consecutive instances that should be nested
        if level_idx < len(self.hierarchy_levels) - 1:
            next_level = self.hierarchy_levels[level_idx + 1]
            next_config = self.hierarchy_config["levels"][next_level]

            if next_config["type"] == "instances":
                # Next level is also instance - verify it's nested
                if next_level.upper() not in container:
                    raise ValueError(
                        f"Consecutive instance levels '{level_name}' and '{next_level}' detected.\n\n"
                        f"'{next_level.upper()}' container must be nested inside '{level_name.upper()}' container.\n\n"
                        f"Current structure (incorrect):\n"
                        f"  tree['{level_name.upper()}'] = {{...}}\n"
                        f"  tree['{next_level.upper()}'] = {{...}}  ← siblings (wrong)\n\n"
                        f"Expected structure (correct):\n"
                        f"  tree['{level_name.upper()}'] = {{\n"
                        f'    "_expansion": {{...}},\n'
                        f'    "{next_level.upper()}": {{  ← nested inside {level_name.upper()}\n'
                        f'      "_expansion": {{...}},\n'
                        f"      ...\n"
                        f"    }}\n"
                        f"  }}\n\n"
                        f"Why: Consecutive instance levels stay at the same tree position,\n"
                        f"so they must be nested to maintain proper navigation."
                    )

    def _validate_expansion_definition(self, expansion: dict, level_name: str):
        """Validate expansion definition has required fields and valid values."""
        if "_type" not in expansion:
            raise ValueError(
                f"Expansion for '{level_name}' missing '_type' field.\nMust be 'range' or 'list'."
            )

        exp_type = expansion["_type"]

        if exp_type == "range":
            if "_pattern" not in expansion:
                raise ValueError(
                    f"Range expansion for '{level_name}' requires '_pattern' field.\n"
                    f'Example: "_pattern": "{{:02d}}" for zero-padded numbers (01, 02, ...)\n'
                    f'Example: "_pattern": "{{}}" for plain numbers (1, 2, ...)\n'
                    f'Example: "_pattern": "B{{:02d}}" for prefixed (B01, B02, ...)'
                )

            if "_range" not in expansion:
                raise ValueError(
                    f"Range expansion for '{level_name}' requires '_range' field.\n"
                    f"Must be [start, end] list (inclusive).\n"
                    f'Example: "_range": [1, 24] generates 1, 2, ..., 24'
                )

            # Validate range format
            if not isinstance(expansion["_range"], list) or len(expansion["_range"]) != 2:
                raise ValueError(
                    f"Range expansion for '{level_name}' '_range' must be [start, end] list.\n"
                    f"Got: {expansion['_range']}"
                )

            start, end = expansion["_range"]
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(
                    f"Range expansion for '{level_name}' start and end must be integers.\n"
                    f"Got: start={start} ({type(start).__name__}), end={end} ({type(end).__name__})"
                )

            if start > end:
                raise ValueError(
                    f"Range expansion for '{level_name}' start must be <= end.\n"
                    f"Got: start={start}, end={end}"
                )

        elif exp_type == "list":
            if "_instances" not in expansion:
                raise ValueError(
                    f"List expansion for '{level_name}' requires '_instances' field.\n"
                    f"Must be a list of strings.\n"
                    f'Example: "_instances": ["MAIN", "BACKUP", "TEST"]'
                )

            if not isinstance(expansion["_instances"], list):
                raise ValueError(
                    f"List expansion for '{level_name}' '_instances' must be a list.\n"
                    f"Got: {type(expansion['_instances']).__name__}"
                )

            if len(expansion["_instances"]) == 0:
                raise ValueError(
                    f"List expansion for '{level_name}' '_instances' cannot be empty.\n"
                    f"Provide at least one instance name."
                )

        else:
            raise ValueError(
                f"Expansion for '{level_name}' has invalid '_type': '{exp_type}'.\n"
                f"Must be 'range' or 'list'."
            )

    def _find_level_container(self, tree: dict, level_name: str, level_idx: int) -> dict | None:
        """
        Find the container for an instance level in the tree.

        Args:
            tree: Tree structure to search
            level_name: Name of level to find
            level_idx: Index in hierarchy

        Returns:
            Container dict if found, None otherwise
        """
        current_node = tree

        # Navigate to the correct position based on previous tree levels
        for prev_idx in range(level_idx):
            prev_level = self.hierarchy_levels[prev_idx]
            prev_config = self.hierarchy_config["levels"][prev_level]

            # Only navigate for tree levels
            if prev_config["type"] == "tree":
                # For validation, we can't navigate without selections
                # Just find the first valid child
                for key, value in current_node.items():
                    if not key.startswith("_") and isinstance(value, dict):
                        current_node = value
                        break

            elif prev_config["type"] == "instances":
                # Find the container and move into it
                for key, value in current_node.items():
                    if key.upper() == prev_level.upper() and isinstance(value, dict):
                        current_node = value
                        break

        # Now look for the current level's container
        for key, value in current_node.items():
            if key.upper() == level_name.upper() and isinstance(value, dict):
                return value

        return None

    def _get_container_path(self, level_name: str, level_idx: int) -> str:
        """Get the path to a container for error messages."""
        path_parts = ["tree"]

        for prev_idx in range(level_idx):
            prev_level = self.hierarchy_levels[prev_idx]
            prev_config = self.hierarchy_config["levels"][prev_level]

            if prev_config["type"] == "tree":
                path_parts.append("[CATEGORY]")
            elif prev_config["type"] == "instances":
                path_parts.append(f"['{prev_level.upper()}']")

        path_parts.append(f"['{level_name.upper()}']")
        return "".join(path_parts)

    def get_hierarchy_definition(self) -> list[str]:
        """Get the hierarchy level names."""
        return self.hierarchy_levels.copy()

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

    def _parse_naming_pattern_separators(self) -> dict[tuple[str, str], str]:
        """
        Extract default separators between levels from naming pattern.

        The naming pattern defines how levels are joined with separators.
        This method extracts those separators for use as defaults.

        Returns:
            Dict mapping (level, next_level) tuples to separator strings

        Example:
            Pattern: "{system}-{subsystem}:{device}_{signal}"
            Returns: {
                ('system', 'subsystem'): '-',
                ('subsystem', 'device'): ':',
                ('device', 'signal'): '_'
            }
        """
        import re

        separators = {}
        pattern = self.naming_pattern

        # Find all {level} placeholders and text between them
        matches = list(re.finditer(r"\{(\w+)\}", pattern))

        for i in range(len(matches) - 1):
            current_level = matches[i].group(1)
            next_level = matches[i + 1].group(1)

            # Extract text between placeholders (the separator)
            start = matches[i].end()
            end = matches[i + 1].start()
            separator = pattern[start:end]

            separators[(current_level, next_level)] = separator

        return separators

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

        elif level_type == "container":
            # Legacy container mode (backward compatibility)
            return self._get_container_options(
                current_node, level, level_config, previous_selections
            )

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

            # Container levels (legacy) also don't change position
            if level_config["type"] == "container":
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

    def _get_container_options(
        self, node: dict, level: str, level_config: dict, previous_selections: dict
    ) -> list[dict[str, str]]:
        """
        Get options from legacy container structure.

        For backward compatibility with existing databases.
        """
        container_key = level_config.get("container_key")

        if not container_key or container_key not in node:
            return []

        container = node[container_key]

        # Check if it's an instance definition or direct dictionary
        if "_type" in container:
            # Instance expansion (range or list)
            return self._expand_instances(container)
        else:
            # Direct dictionary - need special handling for subfield
            if level == "subfield" and "field" in previous_selections:
                # Navigate to specific field first
                field_selection = self._get_single_value(previous_selections["field"])
                if field_selection and field_selection in container:
                    field_node = container[field_selection]
                    if "subfields" in field_node:
                        return self._extract_tree_options(field_node["subfields"])
            else:
                # Regular container
                return self._extract_tree_options(container)

        return []

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
                    # Required level missing - return empty list (invalid selections)
                    return []

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

            elif level_config["type"] == "container":
                # Legacy container mode
                container_key = level_config.get("container_key")

                if container_key and container_key in node:
                    container = node[container_key]

                    if "_type" in container:
                        # Instance expansion
                        instances = self._get_instance_names(container)
                        for instance_name in instances:
                            expand_tree(
                                {**path, current_level: instance_name},
                                node,
                                level_idx + 1,
                                local_overrides,
                            )
                    else:
                        # Container with direct children
                        children = {
                            k: v
                            for k, v in container.items()
                            if not k.startswith("_") and isinstance(v, dict)
                        }

                        for child_name, child_node in children.items():
                            expand_tree(
                                {**path, current_level: child_name},
                                child_node,
                                level_idx + 1,
                                local_overrides,
                            )

        expand_tree({}, self.tree, 0, {})
        return channels

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

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        stats = {
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

    def _get_single_value(self, value):
        """Get single value from potentially list value."""
        if isinstance(value, list):
            return value[0] if value else None
        return value
