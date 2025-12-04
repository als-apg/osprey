"""
Hierarchical Channel Database

Loads and navigates hierarchical channel structures for iterative LLM-based refinement.
Supports flexible hierarchy with arbitrary mixing of:
- Tree levels (semantic categories)
- Instance levels (numbered/patterned expansions)
"""

import itertools
import json
from typing import Any, Optional

from ..core.base_database import BaseDatabase


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

        with open(self.db_path, 'r') as f:
            data = json.load(f)

        self.tree = data['tree']

        # Support new unified schema (preferred) or legacy three-field format (deprecated)
        if 'hierarchy' in data:
            # NEW UNIFIED SCHEMA: Single "hierarchy" section
            hierarchy_def = data['hierarchy']

            # Extract levels list and build derived structures
            levels_list = hierarchy_def['levels']
            self.hierarchy_levels = [level['name'] for level in levels_list]
            self.naming_pattern = hierarchy_def['naming_pattern']

            # Build hierarchy_config from levels list
            self.hierarchy_config = {"levels": {}}
            for level_def in levels_list:
                self.hierarchy_config["levels"][level_def['name']] = {
                    "type": level_def['type']
                }

            # Validate naming_pattern references correct level names
            self._validate_naming_pattern()

        elif 'hierarchy_definition' in data:
            # LEGACY/INTERMEDIATE FORMAT: Backward compatibility for old schemas
            self.hierarchy_levels = data['hierarchy_definition']
            self.naming_pattern = data['naming_pattern']

            # Load or infer hierarchy configuration
            if 'hierarchy_config' in data:
                # Intermediate format (unpublished) - silent backward compatibility
                self.hierarchy_config = data['hierarchy_config']
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
                    stacklevel=2
                )
                self.hierarchy_config = self._infer_legacy_config()
        else:
            raise ValueError(
                "Invalid database format: must contain either 'hierarchy' (new unified schema) "
                "or 'hierarchy_definition' (legacy format)"
            )

        # Validate configuration
        self._validate_hierarchy_config()

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
        legacy_container_keys = {
            'device': 'devices',
            'field': 'fields',
            'subfield': 'subfields'
        }

        for i, level in enumerate(self.hierarchy_levels):
            if i < 2:
                # First two levels: tree-based categories
                config["levels"][level] = {
                    "type": "tree"
                }
            else:
                # Later levels: container-based (legacy style)
                config["levels"][level] = {
                    "type": "container",  # Legacy mode
                    "container_key": legacy_container_keys.get(level, f"{level}s")
                }

        return config

    def _validate_naming_pattern(self):
        """
        Validate that naming_pattern references exactly the level names defined in hierarchy.

        Prevents out-of-sync errors between level names and naming pattern.
        """
        import re

        # Extract placeholder names from naming pattern (e.g., {system}, {family}, etc.)
        pattern_placeholders = set(re.findall(r'\{(\w+)\}', self.naming_pattern))
        expected_placeholders = set(self.hierarchy_levels)

        if pattern_placeholders != expected_placeholders:
            missing = expected_placeholders - pattern_placeholders
            extra = pattern_placeholders - expected_placeholders

            error_msg = "naming_pattern does not match hierarchy level names:\n"
            if missing:
                error_msg += f"  Missing from pattern: {sorted(missing)}\n"
            if extra:
                error_msg += f"  Extra in pattern: {sorted(extra)}\n"
            error_msg += f"  Expected levels: {self.hierarchy_levels}\n"
            error_msg += f"  Pattern: {self.naming_pattern}"

            raise ValueError(error_msg)

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
                    f"Add: \"type\": \"tree\" or \"instances\"\n"
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
        # Find the container for this level
        container = self._find_level_container(self.tree, level_name, level_idx)

        if not container:
            # Helpful error message with suggestions
            raise ValueError(
                f"Instance level '{level_name}' requires container named '{level_name.upper()}' in tree.\n\n"
                f"Expected structure:\n"
                f"  \"tree\": {{\n"
                f"    \"{level_name.upper()}\": {{\n"
                f"      \"_expansion\": {{...}},\n"
                f"      ...(children for next level)\n"
                f"    }}\n"
                f"  }}\n\n"
                f"Troubleshooting:\n"
                f"  1. Check that container name matches level name (case-insensitive)\n"
                f"  2. Verify container is at correct nesting depth\n"
                f"  3. Ensure previous levels are properly configured"
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
                f"  \"_expansion\": {{\n"
                f"    \"_type\": \"range\",\n"
                f"    \"_pattern\": \"{{:02d}}\",  // or \"{{}}\"\n"
                f"    \"_range\": [1, 10]  // [start, end] inclusive\n"
                f"  }}\n\n"
                f"Or for list-based:\n"
                f"  \"_expansion\": {{\n"
                f"    \"_type\": \"list\",\n"
                f"    \"_instances\": [\"A\", \"B\", \"C\"]\n"
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
                        f"    \"_expansion\": {{...}},\n"
                        f"    \"{next_level.upper()}\": {{  ← nested inside {level_name.upper()}\n"
                        f"      \"_expansion\": {{...}},\n"
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
                f"Expansion for '{level_name}' missing '_type' field.\n"
                f"Must be 'range' or 'list'."
            )

        exp_type = expansion["_type"]

        if exp_type == "range":
            if "_pattern" not in expansion:
                raise ValueError(
                    f"Range expansion for '{level_name}' requires '_pattern' field.\n"
                    f"Example: \"_pattern\": \"{{:02d}}\" for zero-padded numbers (01, 02, ...)\n"
                    f"Example: \"_pattern\": \"{{}}\" for plain numbers (1, 2, ...)\n"
                    f"Example: \"_pattern\": \"B{{:02d}}\" for prefixed (B01, B02, ...)"
                )

            if "_range" not in expansion:
                raise ValueError(
                    f"Range expansion for '{level_name}' requires '_range' field.\n"
                    f"Must be [start, end] list (inclusive).\n"
                    f"Example: \"_range\": [1, 24] generates 1, 2, ..., 24"
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
                    f"Example: \"_instances\": [\"MAIN\", \"BACKUP\", \"TEST\"]"
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

    def _find_level_container(self, tree: dict, level_name: str, level_idx: int) -> Optional[dict]:
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

    def get_options_at_level(
        self,
        level: str,
        previous_selections: dict[str, Any]
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
            return self._extract_tree_options(current_node)

        elif level_type == "instances":
            # Find expansion definition for this level
            return self._get_expansion_options(current_node, level)

        elif level_type == "container":
            # Legacy container mode (backward compatibility)
            return self._get_container_options(current_node, level, level_config, previous_selections)

        return []

    def _navigate_to_node(self, target_level: str, previous_selections: dict[str, Any]) -> Optional[dict]:
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
                        return None  # Invalid path

        return current_node

    def _extract_tree_options(self, node: dict) -> list[dict[str, str]]:
        """Extract options from tree-structured node."""
        options = []
        for key, value in node.items():
                if not key.startswith('_') and isinstance(value, dict):
                    options.append({
                        'name': key,
                        'description': value.get('_description', '')
                    })
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
        expansion_type = expansion_def.get('_type')
        options = []

        if expansion_type == 'range':
            pattern = expansion_def.get('_pattern', '{}')
            start, end = expansion_def.get('_range', [1, 1])

            for i in range(start, end + 1):
                instance_name = pattern.format(i)
                options.append({
                    'name': instance_name,
                    'description': f"Instance {i}"
                })

        elif expansion_type == 'list':
            instances = expansion_def.get('_instances', [])
            for instance in instances:
                options.append({
                    'name': instance,
                    'description': ''
                })

        return options

    def _get_container_options(
        self,
        node: dict,
        level: str,
        level_config: dict,
        previous_selections: dict
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

        Args:
            selections: Dict mapping level names to selected values (strings or lists)

        Returns:
            List of complete channel names
        """
        # Convert all selections to lists for uniform handling
        selection_lists = []
        for level in self.hierarchy_levels:
            values = self._ensure_list(selections.get(level, []))
            selection_lists.append(values)

        # Generate Cartesian product of all selections
        channels = []
        for combination in itertools.product(*selection_lists):
            # Build channel name using naming pattern
            params = dict(zip(self.hierarchy_levels, combination))
            channel = self.naming_pattern.format(**params)
            channels.append(channel)

        return channels

    def validate_channel(self, channel: str) -> bool:
        """Check if a channel exists in the database."""
        return channel in self.channel_map

    def get_channel(self, channel_name: str) -> Optional[dict]:
        """Get channel information."""
        channel_data = self.channel_map.get(channel_name)
        if channel_data:
            # Add address field if not present (use channel name as address)
            if 'address' not in channel_data:
                channel_data['address'] = channel_name
        return channel_data

    def get_all_channels(self) -> list[dict]:
        """Get all channels in the database."""
        return [
            {
                'channel': ch_name,
                'address': ch_data.get('address', ch_name),
                **ch_data
            }
            for ch_name, ch_data in self.channel_map.items()
        ]

    def _build_channel_map(self) -> dict[str, dict]:
        """
        Expand hierarchical tree into flat channel map.

        Works with flexible hierarchy configuration.

        Returns:
            Dict mapping channel names to channel info
        """
        channels = {}

        def expand_tree(path: dict[str, str], node: dict, level_idx: int):
            """Recursively expand tree with flexible level handling."""
            # Base case: processed all levels
            if level_idx >= len(self.hierarchy_levels):
                # Build channel from complete path
                channel_name = self.naming_pattern.format(**path)
                channels[channel_name] = {
                    'channel': channel_name,
                    'path': path.copy()
                }
                return

            current_level = self.hierarchy_levels[level_idx]
            level_config = self.hierarchy_config["levels"][current_level]

            # Handle based on level type
            if level_config["type"] == "tree":
                # Tree navigation: iterate direct children
                children = {
                    k: v for k, v in node.items()
                    if not k.startswith('_') and isinstance(v, dict)
                }

                for child_name, child_node in children.items():
                    expand_tree({**path, current_level: child_name}, child_node, level_idx + 1)

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
                        expand_tree({**path, current_level: instance_name}, child_node, level_idx + 1)

            elif level_config["type"] == "container":
                # Legacy container mode
                container_key = level_config.get("container_key")

                if container_key and container_key in node:
                    container = node[container_key]

                    if "_type" in container:
                        # Instance expansion
                        instances = self._get_instance_names(container)
                        for instance_name in instances:
                            expand_tree({**path, current_level: instance_name}, node, level_idx + 1)
                    else:
                        # Container with direct children
                        children = {
                            k: v for k, v in container.items()
                            if not k.startswith('_') and isinstance(v, dict)
                        }

                        for child_name, child_node in children.items():
                            expand_tree({**path, current_level: child_name}, child_node, level_idx + 1)

        expand_tree({}, self.tree, 0)
        return channels

    def _get_instance_names(self, expansion_def: dict) -> list[str]:
        """Get list of instance names from expansion definition."""
        expansion_type = expansion_def.get('_type')

        if expansion_type == 'range':
            pattern = expansion_def.get('_pattern', '{}')
            start, end = expansion_def.get('_range', [1, 1])
            return [pattern.format(i) for i in range(start, end + 1)]

        elif expansion_type == 'list':
            return expansion_def.get('_instances', [])

        return []

    def _ensure_list(self, value: Any) -> list:
        """Convert value to list if it isn't already."""
        if isinstance(value, list):
            return value
        elif value is None:
            return []
        else:
            return [value]

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_channels': len(self.channel_map),
            'hierarchy_levels': self.hierarchy_levels,
        }

        # Count by first level (if it's a tree level)
        first_level = self.hierarchy_levels[0] if self.hierarchy_levels else None
        if first_level:
            first_config = self.hierarchy_config["levels"].get(first_level, {})
            if first_config.get("type") == "tree":
                stats['systems'] = []
                for system_name in self.tree.keys():
                    if not system_name.startswith('_'):
                        system_channels = [
                            ch for ch in self.channel_map.values()
                            if ch['path'].get(first_level) == system_name
                        ]
                        stats['systems'].append({
                            'name': system_name,
                            'channel_count': len(system_channels)
                        })

        return stats

    def _get_single_value(self, value):
        """Get single value from potentially list value."""
        if isinstance(value, list):
            return value[0] if value else None
        return value

