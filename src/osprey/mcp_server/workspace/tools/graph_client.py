"""HTTP client for the DePlot graph extraction service.

Thin async httpx wrapper consumed by the graph MCP tools. Reads
connection details from config.yml under ``deplot.host`` / ``deplot.port``.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from osprey.mcp_server.common import load_osprey_config

logger = logging.getLogger("osprey.mcp_server.tools.graph_client")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8095
_TIMEOUT = 120.0  # Model inference can be slow on first call


def get_deplot_url() -> str:
    """Resolve the DePlot service base URL from config.

    Reads ``deplot.host`` and ``deplot.port`` from config.yml.
    Defaults to ``http://127.0.0.1:8095``.
    """
    config = load_osprey_config()
    deplot = config.get("deplot", {})
    host = deplot.get("host", _DEFAULT_HOST)
    port = deplot.get("port", _DEFAULT_PORT)
    return f"http://{host}:{port}"


class DePlotClient:
    """Async HTTP client for the DePlot service."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or get_deplot_url()

    async def health(self) -> dict[str, Any]:
        """Check service health.

        Returns:
            Health status dict from the service.

        Raises:
            httpx.ConnectError: If the service is not reachable.
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def is_available(self) -> bool:
        """Check if the DePlot service is running and healthy."""
        try:
            status = await self.health()
            return status.get("status") == "ok"
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            return False

    async def extract(
        self, image_path: str, preprocess: bool = True
    ) -> dict[str, Any]:
        """Send an image to the DePlot service for data extraction.

        Args:
            image_path: Path to the chart image file.
            preprocess: Whether to apply OpenCV preprocessing.

        Returns:
            Dict with ``columns``, ``data``, ``raw_table``, and ``title``.

        Raises:
            httpx.ConnectError: If the service is not reachable.
            httpx.HTTPStatusError: On non-2xx responses.
            FileNotFoundError: If image_path does not exist.
        """
        from pathlib import Path

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            with open(path, "rb") as f:
                files = {"image": (path.name, f, "image/png")}
                params = {"preprocess": str(preprocess).lower()}
                resp = await client.post(
                    f"{self.base_url}/extract",
                    files=files,
                    params=params,
                )
            resp.raise_for_status()
            return resp.json()
