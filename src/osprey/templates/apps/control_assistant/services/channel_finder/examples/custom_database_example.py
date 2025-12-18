"""
Example: Custom REST API Database

This example shows how to create a custom database that fetches channels from a REST API.
"""

import logging

import requests

from ..core.base_database import BaseDatabase

logger = logging.getLogger(__name__)


class RESTAPIChannelDatabase(BaseDatabase):
    """
    Channel database backed by REST API.

    Compatible with InContextPipeline.
    """

    def __init__(
        self,
        db_path: str,
        api_url: str = None,
        api_key: str = None,
        timeout: int = 30,
        cache_ttl: int = 300,
        **kwargs,
    ):
        """
        Initialize REST API database.

        Args:
            db_path: Config file path (contains API URL and credentials)
            api_url: API base URL (optional, from config if not provided)
            api_key: API authentication key (optional)
            timeout: Request timeout in seconds
            cache_ttl: Cache time-to-live in seconds
            **kwargs: Additional parameters from config
        """
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.cache_ttl = cache_ttl

        self.channels_cache = None
        self.channel_map_cache = None

        super().__init__(db_path)  # Calls load_database()

    def load_database(self):
        """Load configuration from file and validate API connection."""
        import json

        # Load config file
        with open(self.db_path) as f:
            config = json.load(f)

        # Get API configuration
        if not self.api_url:
            self.api_url = config.get("api_url")

        if not self.api_key:
            self.api_key = config.get("api_key")

        if not self.api_url:
            raise ValueError("api_url must be provided in config file or constructor")

        logger.info(f"Connected to REST API: {self.api_url}")

        # Test connection
        try:
            self._make_request("GET", "/health")
            logger.info("✓ API health check passed")
        except Exception as e:
            logger.warning(f"API health check failed: {e}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make HTTP request to API."""
        url = f"{self.api_url}{endpoint}"

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.request(
            method=method, url=url, headers=headers, timeout=self.timeout, **kwargs
        )

        response.raise_for_status()
        return response.json()

    def get_all_channels(self) -> list[dict]:
        """Fetch all channels from API (with caching)."""
        # Return cached if available
        if self.channels_cache is not None:
            return self.channels_cache

        logger.info("Fetching channels from API...")

        # Fetch from API
        response = self._make_request("GET", "/channels")

        # Parse response
        channels = []
        for item in response.get("channels", []):
            channels.append(
                {
                    "channel": item["name"],
                    "address": item["pv_address"],
                    "description": item.get("description", ""),
                    "unit": item.get("unit"),
                    "type": item.get("type"),
                }
            )

        # Cache results
        self.channels_cache = channels
        self.channel_map_cache = {ch["channel"]: ch for ch in channels}

        logger.info(f"Loaded {len(channels)} channels from API")

        return channels

    def get_channel(self, channel_name: str) -> dict | None:
        """Get single channel (uses cache or API)."""
        # Try cache first
        if self.channel_map_cache:
            return self.channel_map_cache.get(channel_name)

        # Fetch from API
        try:
            response = self._make_request("GET", f"/channels/{channel_name}")
            item = response.get("channel", {})

            return {
                "channel": item["name"],
                "address": item["pv_address"],
                "description": item.get("description", ""),
                "unit": item.get("unit"),
                "type": item.get("type"),
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def validate_channel(self, channel_name: str) -> bool:
        """Check if channel exists."""
        try:
            channel = self.get_channel(channel_name)
            return channel is not None
        except Exception:
            return False

    def get_statistics(self) -> dict:
        """Return database statistics."""
        all_channels = self.get_all_channels()

        return {
            "total_channels": len(all_channels),
            "format": "rest_api",
            "api_url": self.api_url,
            "cache_ttl": self.cache_ttl,
            "cached": self.channels_cache is not None,
        }

    # ===== InContextPipeline-specific methods =====

    def chunk_database(self, chunk_size: int = 50) -> list[list[dict]]:
        """Split into chunks (required by InContextPipeline if chunking enabled)."""
        all_channels = self.get_all_channels()

        chunks = []
        for i in range(0, len(all_channels), chunk_size):
            chunks.append(all_channels[i : i + chunk_size])

        return chunks

    def format_chunk_for_prompt(self, chunk: list[dict], include_addresses: bool = False) -> str:
        """Format chunk for LLM prompt (required by InContextPipeline)."""
        formatted = []

        for ch in chunk:
            if include_addresses:
                entry = f"- {ch['channel']} (Address: {ch['address']})"
            else:
                entry = f"- {ch['channel']}"

            if ch.get("description"):
                entry += f": {ch['description']}"

            # Add API-specific metadata
            if ch.get("unit"):
                entry += f" [Unit: {ch['unit']}]"

            if ch.get("type"):
                entry += f" [Type: {ch['type']}]"

            formatted.append(entry)

        return "\n".join(formatted)


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of custom database registration.
    """

    from services.channel_finder import ChannelFinderService

    # Register the custom database
    ChannelFinderService.register_database("rest_api", RESTAPIChannelDatabase)

    print("✓ Registered 'rest_api' database")
    print("\nAvailable databases:")
    for name, desc in ChannelFinderService.list_available_databases().items():
        print(f"  - {name}: {desc}")

    print("\nConfig file example (rest_api_config.json):")
    print(
        """{
    "api_url": "https://api.example.com/v1",
    "api_key": "your-api-key-here",
    "timeout": 30,
    "cache_ttl": 300
}"""
    )

    print("\nconfig.yml example:")
    print(
        """
    channel_finder:
      pipeline_mode: "in_context"
      pipelines:
        in_context:
          database:
            type: "rest_api"
            path: "config/rest_api_config.json"
            api_url: "https://api.example.com/v1"
            timeout: 30
          processing:
            chunk_dictionary: false
    """
    )
