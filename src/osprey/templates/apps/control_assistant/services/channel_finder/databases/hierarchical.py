"""
Hierarchical Channel Database

Loads and navigates hierarchical channel structures for iterative LLM-based refinement.
"""

import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from ..core.base_database import BaseDatabase


class HierarchicalChannelDatabase(BaseDatabase):
    """
    Database for hierarchical channel naming schemes.
    
    Supports iterative navigation through levels: SYSTEM → FAMILY → DEVICE → FIELD → SUBFIELD
    """
    
    def __init__(self, db_path: str):
        """
        Initialize hierarchical database.
        
        Args:
            db_path: Path to hierarchical database JSON file
        """
        super().__init__(db_path)
    
    def load_database(self):
        """Load hierarchical database from JSON."""
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        
        self.hierarchy_levels = data['hierarchy_definition']
        self.naming_pattern = data['naming_pattern']
        self.tree = data['tree']
        
        # Build flat channel map for validation and lookup
        self.channel_map = self._build_channel_map()
    
    def get_hierarchy_definition(self) -> List[str]:
        """Get the hierarchy level names."""
        return self.hierarchy_levels.copy()
    
    def get_options_at_level(
        self, 
        level: str, 
        previous_selections: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Get available options at a specific hierarchy level.
        
        Args:
            level: Current level name (e.g., "system", "family", "device")
            previous_selections: Dict mapping previous level names to selected values
            
        Returns:
            List of options with name, description, and metadata
        """
        # Navigate to current position in tree
        current_node = self.tree
        
        # For each previous level, navigate down the tree
        # NOTE: Tree structure is system -> family only
        #       device, field, subfield are accessed via special keys, not direct children
        for prev_level in self.hierarchy_levels:
            if prev_level == level:
                break
            
            # Only navigate for system and family levels
            # (device, field, subfield are accessed differently)
            if prev_level not in ['system', 'family']:
                continue
            
            if prev_level in previous_selections:
                selection = previous_selections[prev_level]
                
                # Handle list selections (take first for navigation)
                if isinstance(selection, list):
                    selection = selection[0] if selection else None
                
                if selection and selection in current_node:
                    current_node = current_node[selection]
                else:
                    # Invalid path
                    return []
        
        # Extract options at current level
        options = []
        
        if level == "system":
            # Level 1: System level - list all top-level systems
            for key, value in current_node.items():
                if not key.startswith('_'):
                    options.append({
                        'name': key,
                        'description': value.get('_description', '')
                    })
        
        elif level == "family":
            # Level 2: Family level - list device families in selected system
            for key, value in current_node.items():
                if not key.startswith('_') and isinstance(value, dict):
                    options.append({
                        'name': key,
                        'description': value.get('_description', '')
                    })
        
        elif level == "device":
            # Level 3: Device level - expand device instances
            if 'devices' in current_node:
                device_config = current_node['devices']
                device_type = device_config.get('_type')
                
                if device_type == 'range':
                    # Generate device list from range
                    pattern = device_config.get('_pattern', '{}')
                    start, end = device_config.get('_range', [1, 1])
                    
                    for i in range(start, end + 1):
                        device_name = pattern.format(i)
                        options.append({
                            'name': device_name,
                            'description': f"Instance {i}"
                        })
                
                
                elif device_type == 'list':
                    # Use explicit list
                    instances = device_config.get('_instances', [])
                    for instance in instances:
                        options.append({
                            'name': instance,
                            'description': ''
                        })
        
        elif level == "field":
            # Level 4: Field level - list physical quantities
            if 'fields' in current_node:
                for key, value in current_node['fields'].items():
                    if not key.startswith('_'):
                        options.append({
                            'name': key,
                            'description': value.get('_description', '')
                        })
        
        elif level == "subfield":
            # Level 5: Subfield level - list measurement types
            # For single field, show its subfields directly
            # (Multi-field handling is done at pipeline level with separate calls)
            if 'fields' in current_node and 'field' in previous_selections:
                field_selection = previous_selections['field']
                
                # Handle both single string and list (take first if list)
                if isinstance(field_selection, list):
                    field_selection = field_selection[0] if field_selection else None
                
                if field_selection and field_selection in current_node['fields']:
                    field_node = current_node['fields'][field_selection]
                    if 'subfields' in field_node:
                        for key, value in field_node['subfields'].items():
                            if not key.startswith('_'):
                                options.append({
                                    'name': key,
                                    'description': value.get('_description', '')
                                })
        
        return options
    
    def build_channels_from_selections(
        self, 
        selections: Dict[str, Any]
    ) -> List[str]:
        """
        Build fully-qualified channel names from hierarchical selections.
        
        With recursive branching, selections will contain single values at branch levels
        (system/family/field) and lists only at device/subfield levels.
        
        Args:
            selections: Dict mapping level names to selected values
                       - Single strings for branch levels (system, family, field)
                       - Lists for device and subfield levels
            
        Returns:
            List of complete channel names
        """
        channels = []
        
        # Get selections at each level (convert to lists for uniform handling)
        systems = self._ensure_list(selections.get('system', []))
        families = self._ensure_list(selections.get('family', []))
        devices = self._ensure_list(selections.get('device', []))
        fields = self._ensure_list(selections.get('field', []))
        subfields = self._ensure_list(selections.get('subfield', []))
        
        # Build Cartesian product of all selections
        for system in systems:
            for family in families:
                for device in devices:
                    for field in fields:
                        for subfield in subfields:
                            # Build channel name using pattern
                            channel = self.naming_pattern.format(
                                system=system,
                                family=family,
                                device=device,
                                field=field,
                                subfield=subfield
                            )
                            channels.append(channel)
        
        return channels
    
    def validate_channel(self, channel: str) -> bool:
        """Check if a channel exists in the database."""
        return channel in self.channel_map
    
    def get_channel(self, channel_name: str) -> Optional[Dict]:
        """Get channel information."""
        channel_data = self.channel_map.get(channel_name)
        if channel_data:
            # Add address field if not present (use channel name as address)
            if 'address' not in channel_data:
                channel_data['address'] = channel_name
        return channel_data
    
    def get_all_channels(self) -> List[Dict]:
        """Get all channels in the database."""
        return [
            {
                'channel': ch_name,
                'address': ch_data.get('address', ch_name),
                **ch_data
            }
            for ch_name, ch_data in self.channel_map.items()
        ]
    
    def _build_channel_map(self) -> Dict[str, Dict]:
        """
        Expand hierarchical tree into flat channel map.
        
        Returns:
            Dict mapping channel names to channel info
        """
        channels = {}
        
        def expand_tree(path: Dict[str, str], node: Dict):
            """Recursively expand tree."""
            current_level_idx = len(path)
            
            if current_level_idx >= len(self.hierarchy_levels):
                # Reached leaf - build channel
                channel_name = self.naming_pattern.format(**path)
                channels[channel_name] = {
                    'channel': channel_name,
                    'path': path.copy()
                }
                return
            
            current_level = self.hierarchy_levels[current_level_idx]
            
            if current_level == "system":
                for key, value in node.items():
                    if not key.startswith('_'):
                        expand_tree({**path, 'system': key}, value)
            
            elif current_level == "family":
                for key, value in node.items():
                    if not key.startswith('_') and isinstance(value, dict):
                        expand_tree({**path, 'family': key}, value)
            
            elif current_level == "device":
                if 'devices' in node:
                    device_config = node['devices']
                    device_type = device_config.get('_type')
                    
                    if device_type == 'range':
                        pattern = device_config.get('_pattern', '{}')
                        start, end = device_config.get('_range', [1, 1])
                        for i in range(start, end + 1):
                            device_name = pattern.format(i)
                            expand_tree({**path, 'device': device_name}, node)
                    
                    
                    elif device_type == 'list':
                        for instance in device_config.get('_instances', []):
                            expand_tree({**path, 'device': instance}, node)
            
            elif current_level == "field":
                if 'fields' in node:
                    for key in node['fields'].keys():
                        if not key.startswith('_'):
                            expand_tree({**path, 'field': key}, node['fields'][key])
            
            elif current_level == "subfield":
                if 'subfields' in node:
                    for key in node['subfields'].keys():
                        if not key.startswith('_'):
                            expand_tree({**path, 'subfield': key}, node['subfields'][key])
        
        expand_tree({}, self.tree)
        return channels
    
    def _ensure_list(self, value: Any) -> List:
        """Convert value to list if it isn't already."""
        if isinstance(value, list):
            return value
        elif value is None:
            return []
        else:
            return [value]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_channels': len(self.channel_map),
            'hierarchy_levels': self.hierarchy_levels,
            'systems': []
        }
        
        # Count by system
        for system_name in self.tree.keys():
            if not system_name.startswith('_'):
                system_channels = [
                    ch for ch in self.channel_map.values()
                    if ch['path'].get('system') == system_name
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

