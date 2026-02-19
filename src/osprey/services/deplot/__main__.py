"""Entry point for the DePlot service.

Usage:
    uv run python -m osprey.services.deplot
    uv run python -m osprey.services.deplot --port 8095 --host 0.0.0.0
"""

import argparse
import logging

import uvicorn

from osprey.services.deplot.server import create_app

logger = logging.getLogger("osprey.services.deplot")


def main() -> None:
    """Parse arguments and start the DePlot service."""
    parser = argparse.ArgumentParser(description="OSPREY DePlot graph extraction service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8095, help="Bind port (default: 8095)")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger.info("Starting DePlot service on %s:%d", args.host, args.port)

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
