"""Typesense client factory and collection name resolution for AccelPapers."""

import os

import typesense


def get_client() -> typesense.Client:
    """Create a Typesense client from environment config.

    Environment variables:
        ACCELPAPERS_TYPESENSE_HOST: Server host (default: localhost).
        ACCELPAPERS_TYPESENSE_PORT: Server port (default: 8108).
        ACCELPAPERS_TYPESENSE_API_KEY: API key (default: accelpapers-dev).
    """
    return typesense.Client(
        {
            "api_key": os.environ.get("ACCELPAPERS_TYPESENSE_API_KEY", "accelpapers-dev"),
            "nodes": [
                {
                    "host": os.environ.get("ACCELPAPERS_TYPESENSE_HOST", "localhost"),
                    "port": os.environ.get("ACCELPAPERS_TYPESENSE_PORT", "8108"),
                    "protocol": "http",
                }
            ],
            "connection_timeout_seconds": 5,
        }
    )


def get_collection_name() -> str:
    """Return the Typesense collection name.

    Environment variables:
        ACCELPAPERS_COLLECTION: Collection name (default: papers).
    """
    return os.environ.get("ACCELPAPERS_COLLECTION", "papers")
