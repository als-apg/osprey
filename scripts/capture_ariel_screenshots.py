#!/usr/bin/env python3
"""Capture ARIEL web interface screenshots for documentation.

Launches the ARIEL web app on a temporary port, captures screenshots
of each view, and saves them to the docs static directory.

Screenshots are referenced by the developer guide at:
  docs/source/developer-guides/05_production-systems/07_logbook-search-service/web-interface.rst

The mapping between views and output files:
  #search  → ariel_search.png   (Search tab-item)
  #browse  → ariel_browse.png   (Browse tab-item)
  #create  → ariel_create.png   (New Entry tab-item)
  #status  → ariel_status.png   (Status tab-item)

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    python scripts/capture_ariel_screenshots.py
    python scripts/capture_ariel_screenshots.py --config /path/to/config.yml

The script uses a built-in minimal configuration by default, pointing to
the standard Osprey PostgreSQL database at localhost:5432. Start it with
``osprey deploy up`` before running this script.  Pass ``--config`` to
override with a custom config file.
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

# Output directory for screenshots
SCREENSHOTS_DIR = Path("docs/source/_static/screenshots")

# Views to capture: (hash, filename, wait_selector)
VIEWS = [
    ("#search", "ariel_search.png", "#view-search"),
    ("#browse", "ariel_browse.png", "#view-browse"),
    ("#create", "ariel_create.png", "#view-create"),
    ("#status", "ariel_status.png", "#view-status"),
]

# Minimal self-contained config for screenshot capture.
# Only needs the database URI and keyword search enabled so the
# web interface can start and render all views.
MINIMAL_CONFIG = """\
ariel:
  database:
    uri: postgresql://ariel:ariel@localhost:5432/ariel
  default_max_results: 10
  search_modules:
    keyword:
      enabled: true
    semantic:
      enabled: false
  pipelines:
    rag:
      enabled: true
      retrieval_modules: [keyword]
    agent:
      enabled: false
  enhancement_modules:
    semantic_processor:
      enabled: false
    text_embedding:
      enabled: false
"""


def find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_server(port: int, config_path: str) -> None:
    """Start the ARIEL web server in a background thread."""
    import uvicorn

    from osprey.interfaces.ariel.app import create_app

    app = create_app(config_path=config_path)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


async def capture_screenshots(port: int) -> None:
    """Capture screenshots of each view using Playwright."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Error: playwright is not installed.")
        print("Install with: pip install playwright && playwright install chromium")
        sys.exit(1)

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"

    # Wait for server to be ready
    print(f"Waiting for server on port {port}...")
    for _ in range(30):
        try:
            import urllib.request

            urllib.request.urlopen(f"{base_url}/health", timeout=2)
            break
        except Exception:
            await asyncio.sleep(1)
    else:
        print("Error: Server did not start within 30 seconds.")
        print("Hint: Is PostgreSQL running? Start it with: osprey deploy up")
        sys.exit(1)

    print("Server ready. Capturing screenshots...")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})

        for view_hash, filename, wait_selector in VIEWS:
            url = f"{base_url}/{view_hash}"
            print(f"  Capturing {view_hash} -> {filename}")

            await page.goto(url)
            await page.wait_for_selector(wait_selector, state="visible", timeout=10000)
            # Allow rendering to settle
            await asyncio.sleep(1)

            output_path = SCREENSHOTS_DIR / filename
            await page.screenshot(path=str(output_path), full_page=False)
            print(f"    Saved: {output_path}")

        await browser.close()

    print(f"\nDone. Screenshots saved to {SCREENSHOTS_DIR}/")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Capture ARIEL web interface screenshots")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yml (default: use built-in minimal config)",
    )
    args = parser.parse_args()

    # Determine config path
    if args.config:
        config_path = args.config
        print(f"Using config: {config_path}")
    else:
        # Write minimal config to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", prefix="ariel_screenshots_", delete=False
        )
        tmp.write(MINIMAL_CONFIG)
        tmp.flush()
        config_path = tmp.name
        print(f"Using built-in minimal config (written to {config_path})")

    port = find_free_port()
    print(f"Starting ARIEL web server on port {port}...")

    # Start server in background thread
    server_thread = threading.Thread(target=start_server, args=(port, config_path), daemon=True)
    server_thread.start()

    # Give the server a moment to start binding
    time.sleep(2)

    # Capture screenshots
    asyncio.run(capture_screenshots(port))


if __name__ == "__main__":
    main()
