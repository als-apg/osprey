"""Entry point for ``python -m osprey.mcp_server.dispatch_worker``."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the dispatch worker FastAPI app via uvicorn."""
    uvicorn.run(
        "osprey.mcp_server.dispatch_worker.dispatch_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("DISPATCH_WORKER_PORT", "9190")),
    )


if __name__ == "__main__":
    main()
