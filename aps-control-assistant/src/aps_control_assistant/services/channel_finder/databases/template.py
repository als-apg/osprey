"""
Enhanced Database with Template Support and Dual Presentation Modes

Extends the original ChannelDatabase to support:
1. Template-based channel definitions (template storage)
2. Automatic expansion of templates to explicit channels
3. Dual presentation modes: explicit (all names listed) vs template (pattern notation)
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from .legacy import ChannelDatabase as ChannelDatabaseLegacy


class ChannelDatabase(ChannelDatabaseLegacy):
    """Database with template support and dual presentation modes."""

    def __init__(self, db_path: str, presentation_mode: str = "explicit"):
        """
        Initialize database with template support.

        Args:
            db_path: Path to database JSON file (can be template or legacy format)
            presentation_mode: "explicit" (list all names) or "template" (use patterns)
        """
        self.presentation_mode = presentation_mode
        super().__init__(db_path)

    def load_database(self):
        """Load database and expand any template entries into explicit channels."""
        with open(self.db_path, 'r') as f:
            raw_data = json.load(f)

        # Handle both list format and dict format with metadata
        if isinstance(raw_data, dict):
            entries = raw_data.get('channels', [])
            self.metadata = {k: v for k, v in raw_data.items() if k != 'channels'}
        else:
            entries = raw_data
            self.metadata = {}

        # Expand templates into explicit channels
        self.channels = []
        self.template_map = {}  # Track which channels came from which template

        # Track statistics
        self.template_entry_count = 0
        self.standalone_entry_count = 0

        for entry in entries:
            if entry.get('template'):
                self.template_entry_count += 1
                expanded = self._expand_template(entry)
                self.channels.extend(expanded)
                # Track template membership for grouping
                template_key = entry['base_name']
                for ch in expanded:
                    self.template_map[ch['channel']] = template_key
            else:
                self.standalone_entry_count += 1
                self.channels.append(entry)

        # Create lookup map for O(1) access
        self.channel_map = {ch['channel']: ch for ch in self.channels}

    def _expand_template(self, tmpl: dict) -> List[dict]:
        """
        Expand a template entry into multiple explicit channel dictionaries.

        Template format:
        {
            "template": true,
            "base_name": "DipoleMagnet",
            "instances": [2, 10],  # start, end (inclusive)
            "sub_channels": ["SetPoint", "ReadBack"],  # or with axes
            "axes": ["X", "Y"],  # optional, for multi-axis devices
            "address_pattern": "{base}{instance:02d}{suffix}",  # optional
            "description": "Description with {instance} placeholder",  # generic fallback
            "channel_descriptions": {  # optional, per-sub-channel descriptions
                "SetPoint": "Specific description for {instance:02d} SetPoint",
                "ReadBack": "Specific description for {instance:02d} ReadBack"
            }
        }
        """
        base_name = tmpl['base_name']
        start, end = tmpl['instances']
        sub_channels = tmpl.get('sub_channels', [])
        axes = tmpl.get('axes', [None])  # None means no axis
        generic_description = tmpl.get('description', '')
        address_pattern = tmpl.get('address_pattern', '{base}{instance:02d}{suffix}')
        channel_descriptions = tmpl.get('channel_descriptions', {})

        # If no sub_channels, use empty string so loop executes once
        if not sub_channels:
            sub_channels = ['']

        expanded = []

        for instance_num in range(start, end + 1):
            for axis in axes:
                for suffix in sub_channels:
                    # Build channel name
                    if axis:
                        # Multi-axis: SteeringCoil01XSetPoint
                        channel_name = f"{base_name}{instance_num:02d}{axis}{suffix}"
                        actual_suffix = f"{axis}{suffix}"
                    else:
                        # Single: DipoleMagnet02SetPoint
                        channel_name = f"{base_name}{instance_num:02d}{suffix}"
                        actual_suffix = suffix

                    # Build address (use pattern if provided, else same as channel name)
                    # When address_pattern has {axis}, use plain suffix; otherwise use actual_suffix
                    pattern_suffix = suffix if (axis and '{axis}' in address_pattern) else actual_suffix
                    address = address_pattern.format(
                        base=base_name,
                        instance=instance_num,
                        suffix=pattern_suffix,
                        axis=axis if axis else ""
                    )

                    # Choose description: specific sub-channel description or generic fallback
                    if actual_suffix in channel_descriptions:
                        # Use specific sub-channel description
                        desc_template = channel_descriptions[actual_suffix]
                    else:
                        # Fall back to generic description
                        desc_template = generic_description

                    # Build description with instance and axis substitution
                    desc = desc_template.format(
                        instance=instance_num,
                        axis=axis if axis else ""
                    )

                    expanded.append({
                        'channel': channel_name,
                        'address': address,
                        'description': desc,
                        'template_source': base_name
                    })

        return expanded

    def format_chunk_for_prompt(
        self,
        chunk: List[Dict],
        include_addresses: bool = False
    ) -> str:
        """
        Format chunk using selected presentation mode.

        Args:
            chunk: List of channel dictionaries
            include_addresses: Whether to include addresses

        Returns:
            Formatted string for LLM prompt
        """
        if self.presentation_mode == "template":
            return self._format_template(chunk, include_addresses)
        else:
            return self._format_explicit(chunk, include_addresses)

    def _format_explicit(self, chunk: List[Dict], include_addresses: bool = False) -> str:
        """
        Format with explicit names, grouped by device family.

        Lists every channel name explicitly but groups by template family
        to avoid repeating descriptions.
        """
        # Group channels by template source
        grouped = {}
        standalone = []

        for ch in chunk:
            template_source = ch.get('template_source')
            if template_source:
                if template_source not in grouped:
                    grouped[template_source] = []
                grouped[template_source].append(ch)
            else:
                standalone.append(ch)

        formatted = []

        # Format standalone channels first
        for ch in standalone:
            if include_addresses:
                entry = f"- {ch['channel']} (Address: {ch['address']})"
            else:
                entry = f"- {ch['channel']}"

            if ch.get('description'):
                entry += f": {ch['description']}"

            formatted.append(entry)

        # Format grouped channels
        for template_name, channels in grouped.items():
            # Check if all channels have the same description pattern
            # (normalize for instance numbers to detect if descriptions vary by sub-channel)
            normalized_descs = set()
            for ch in channels:
                if ch.get('description'):
                    # Normalize by removing instance numbers
                    normalized = ch['description']
                    for i in range(1, 100):
                        normalized = normalized.replace(f'{i:02d}', '{N}')
                    normalized_descs.add(normalized)

            # If all descriptions are the same, show compact format with header
            # If they differ, show individual descriptions for each channel
            show_individual_descriptions = len(normalized_descs) > 1

            if not show_individual_descriptions and channels and channels[0].get('description'):
                # Compact format: header with generic description, list channel names only
                desc = channels[0]['description']
                # Remove instance-specific parts for header
                desc = desc.replace('01', '{N}').replace('02', '{N}')
                formatted.append(f"\n{template_name} devices: {desc}")

                # List all channel names
                for ch in channels:
                    if include_addresses:
                        formatted.append(f"- {ch['channel']} (Address: {ch['address']})")
                    else:
                        formatted.append(f"- {ch['channel']}")
            else:
                # Explicit format: show description for each channel
                formatted.append(f"\n{template_name} devices:")

                for ch in channels:
                    if include_addresses:
                        entry = f"- {ch['channel']} (Address: {ch['address']})"
                    else:
                        entry = f"- {ch['channel']}"

                    if ch.get('description'):
                        entry += f": {ch['description']}"

                    formatted.append(entry)

        return "\n".join(formatted)

    def _format_template(self, chunk: List[Dict], include_addresses: bool = False) -> str:
        """
        Format using pattern notation with examples.

        Uses range syntax and patterns to minimize tokens.
        """
        # Group channels by template source
        grouped = {}
        standalone = []

        for ch in chunk:
            template_source = ch.get('template_source')
            if template_source:
                if template_source not in grouped:
                    grouped[template_source] = []
                grouped[template_source].append(ch)
            else:
                standalone.append(ch)

        formatted = []

        # Format standalone channels normally
        for ch in standalone:
            if include_addresses:
                entry = f"- {ch['channel']} (Address: {ch['address']})"
            else:
                entry = f"- {ch['channel']}"

            if ch.get('description'):
                entry += f": {ch['description']}"

            formatted.append(entry)

        # Format grouped channels with patterns
        for template_name, channels in grouped.items():
            # Detect pattern structure
            pattern_info = self._detect_pattern(channels)

            # Add group header
            if channels and channels[0].get('description'):
                desc = channels[0]['description']
                desc = desc.replace('01', '{N}').replace('02', '{N}')
                formatted.append(f"\n{template_name} devices: {desc}")
            else:
                formatted.append(f"\n{template_name} devices:")

            # Show pattern
            formatted.append(f"  Pattern: {pattern_info['pattern']}")

            # Show 2 examples
            examples = channels[:2]
            example_str = ", ".join(ch['channel'] for ch in examples)
            if len(channels) > 2:
                example_str += f", ... ({len(channels)} total)"
            formatted.append(f"  Examples: {example_str}")

        return "\n".join(formatted)

    def _detect_pattern(self, channels: List[Dict]) -> Dict:
        """
        Detect pattern structure from a group of channels.

        Returns dict with 'pattern' string representation.
        """
        if not channels:
            return {'pattern': ''}

        # Analyze first and last to detect ranges
        first = channels[0]['channel']
        last = channels[-1]['channel']

        # Simple pattern detection - find numbers and suffixes
        import re

        # Try pattern with optional suffix: Base + Number + OptionalSuffix
        matches = [re.match(r'([A-Za-z]+)(\d+)([A-Za-z]*)', ch['channel']) for ch in channels]

        if all(matches):
            base = matches[0].group(1)
            numbers = [int(m.group(2)) for m in matches]
            suffixes = [m.group(3) for m in matches if m.group(3)]  # Only non-empty suffixes

            min_num = min(numbers)
            max_num = max(numbers)

            # Build pattern
            if suffixes and len(set(suffixes)) > 1:
                # Multiple different suffixes
                suffix_pattern = "{" + "|".join(sorted(set(suffixes))) + "}"
            elif suffixes:
                # Single suffix type
                suffix_pattern = suffixes[0]
            else:
                # No suffix
                suffix_pattern = ""

            pattern = f"{base}{{{min_num:02d}-{max_num:02d}}}{suffix_pattern}"
        else:
            # Fallback
            pattern = f"{first} ... {last}"

        return {'pattern': pattern}

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        base_stats = super().get_statistics()
        base_stats['format'] = 'template'
        base_stats['presentation_mode'] = self.presentation_mode
        base_stats['template_entries'] = self.template_entry_count
        base_stats['standalone_entries'] = self.standalone_entry_count
        return base_stats

