"""
Flat Channel Database - Simple List Format

Manages a flat list of channels with efficient lookup and formatting capabilities.
This is the base implementation for in-context channel databases.
"""

import json

from ..core.base_database import BaseDatabase


class ChannelDatabase(BaseDatabase):
    """Manages the channel database with efficient lookup."""

    def __init__(self, db_path: str):
        """
        Initialize the channel database.

        Args:
            db_path: Path to the channel database JSON file

        Raises:
            FileNotFoundError: If database file doesn't exist
            json.JSONDecodeError: If database file is malformed
        """
        # Note: GoogleSheetsChannelDatabase mirrors these attributes manually
        # in its __init__ (it cannot call super().__init__). Keep in sync.
        self.channels: list[dict] = []
        self.channel_map: dict[str, dict] = {}
        super().__init__(db_path)

    def load_database(self):
        """Load channel database from JSON file."""
        with open(self.db_path) as f:
            self.channels = json.load(f)

        # Create lookup map for O(1) access
        self.channel_map = {ch["channel"]: ch for ch in self.channels}

    def get_statistics(self) -> dict:
        """Get database statistics."""
        return {"total_channels": len(self.channels), "format": "flat"}

    def get_all_channels(self) -> list[dict]:
        """Return all channels."""
        return self.channels

    def get_channel(self, channel_name: str) -> dict | None:
        """Get channel by exact name match."""
        return self.channel_map.get(channel_name)

    def validate_channel(self, channel_name: str) -> bool:
        """Check if channel exists in database."""
        return channel_name in self.channel_map

    def validate_channels(self, channel_names: list[str]) -> list[dict[str, any]]:
        """Validate list of channels, return list of {channel, valid} dicts.

        Args:
            channel_names: List of channel names to validate

        Returns:
            List of validation entries with 'channel' (str) and 'valid' (bool) keys
        """
        validation_results = []
        for name in channel_names:
            validation_results.append({"channel": name, "valid": self.validate_channel(name)})
        return validation_results

    def get_valid_channels(self, validation_results: list[dict]) -> list[str]:
        """Extract only valid channel names from validation results."""
        return [entry["channel"] for entry in validation_results if entry["valid"]]

    def get_invalid_channels(self, validation_results: list[dict]) -> list[str]:
        """Extract only invalid channel names from validation results."""
        return [entry["channel"] for entry in validation_results if not entry["valid"]]

    def format_for_prompt(self, include_addresses: bool = False) -> str:
        """Format channel database for inclusion in LLM prompts.

        Args:
            include_addresses: Whether to include channel addresses in output

        Returns:
            Formatted string of all channels
        """
        formatted = []
        for ch in self.channels:
            if include_addresses:
                entry = f"- {ch['channel']} (Address: {ch['address']})"
            else:
                entry = f"- {ch['channel']}"

            if ch.get("description"):
                entry += f": {ch['description']}"

            formatted.append(entry)

        return "\n".join(formatted)

    def chunk_database(self, chunk_size: int = 50) -> list[list[dict]]:
        """Split database into chunks for processing.

        Args:
            chunk_size: Number of channels per chunk

        Returns:
            List of channel chunks (each chunk is a list of channel dicts)
        """
        chunks = []
        for i in range(0, len(self.channels), chunk_size):
            chunk = self.channels[i : i + chunk_size]
            chunks.append(chunk)
        return chunks

    def format_chunk_for_prompt(self, chunk: list[dict], include_addresses: bool = False) -> str:
        """Format a specific chunk for LLM prompts.

        Args:
            chunk: List of channel dictionaries
            include_addresses: Whether to include addresses in output

        Returns:
            Formatted string of channels in the chunk
        """
        formatted = []
        for ch in chunk:
            if include_addresses:
                entry = f"- {ch['channel']} (Address: {ch['address']})"
            else:
                entry = f"- {ch['channel']}"

            if ch.get("description"):
                entry += f": {ch['description']}"

            formatted.append(entry)

        return "\n".join(formatted)
