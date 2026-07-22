"""Database loading, schema validation, and serialization.

Parses the hierarchical database JSON into the in-memory structures the rest of
the class navigates, validates that the naming pattern and tree agree with the
declared hierarchy, and serializes state back to disk. Builds on the channel-map
construction provided by :class:`_HierarchicalNamingMixin`.
"""

import json
import logging

from ._hierarchical_naming import _HierarchicalNamingMixin

logger = logging.getLogger(__name__)


class _HierarchicalLoadingMixin(_HierarchicalNamingMixin):
    """Loading, schema validation, and serialization for the hierarchical database."""

    def load_database(self):
        """Load and parse the hierarchical database JSON."""
        with open(self.db_path) as f:
            data = json.load(f)

        self._raw_data = data
        self.tree = data["tree"]

        if "hierarchy" not in data:
            raise ValueError(
                "Invalid database format: must contain 'hierarchy' section with "
                "'levels' list and 'naming_pattern'. See data/channel_databases/"
                "hierarchical.json for the expected format."
            )

        # Unified schema: single "hierarchy" section
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

        # Validate configuration
        self._validate_hierarchy_config()

        # Parse default separators from naming pattern
        self.default_separators = self._parse_naming_pattern_separators()

        # Build flat channel map for validation and lookup
        self.channel_map = self._build_channel_map()

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

            if config["type"] not in ["tree", "instances"]:
                raise ValueError(
                    f"Level '{level}' has invalid type: '{config['type']}'.\n"
                    f"Must be 'tree' or 'instances'.\n"
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

    def _serialize(self) -> dict:
        """Serialize in-memory state back to JSON-compatible structure."""
        return self._raw_data
