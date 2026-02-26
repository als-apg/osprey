#!/usr/bin/env python3
"""Capture ARIEL web interface screenshots for documentation.

Self-contained script that creates a temporary database, ingests demo data
with embeddings, starts the web server with full control_assistant config,
captures screenshots, and tears everything down.

Prerequisites:
    - PostgreSQL reachable at localhost:5432 (via `osprey deploy up`)
    - Ollama reachable at localhost:11434 (via `osprey deploy up`)
    - pip install playwright && playwright install chromium

Usage:
    # Full self-contained run (default):
    python scripts/capture_ariel_screenshots.py

    # Escape hatch: capture from a running instance (skip setup/teardown):
    python scripts/capture_ariel_screenshots.py --url http://127.0.0.1:8085

Screenshots are saved to docs/source/_static/screenshots/ and referenced by:
  docs/source/developer-guides/05_production-systems/07_logbook-search-service/web-interface.rst
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

# Repo root (scripts/ is one level below)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Output directory for screenshots
SCREENSHOTS_DIR = REPO_ROOT / "docs" / "source" / "_static" / "screenshots"

# Demo logbook data bundled with the control_assistant template
DEMO_LOGBOOK = (
    REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "logbook_seed"
    / "demo_logbook.json"
)

# Views to capture: (hash, filename, wait_selector)
VIEWS = [
    ("#search", "ariel_search.png", "#view-search"),
    ("#browse", "ariel_browse.png", "#view-browse"),
    ("#create", "ariel_create.png", "#view-create"),
    ("#status", "ariel_status.png", "#view-status"),
]

# PostgreSQL connection parameters for admin operations (CREATE/DROP DATABASE)
PG_ADMIN_DSN = "postgresql://ariel:ariel@localhost:5432/ariel"


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def check_postgres() -> None:
    """Verify PostgreSQL is reachable at localhost:5432."""
    import psycopg

    try:
        with psycopg.connect(PG_ADMIN_DSN, autocommit=True) as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        print(f"Error: Cannot connect to PostgreSQL at localhost:5432: {e}")
        print("Start it with: osprey deploy up")
        sys.exit(1)


def check_ollama() -> None:
    """Verify Ollama is reachable at localhost:11434."""
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except Exception as e:
        print(f"Error: Cannot reach Ollama at localhost:11434: {e}")
        print("Start it with: osprey deploy up")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Temporary database management
# ---------------------------------------------------------------------------


def create_temp_database(db_name: str) -> None:
    """Create a temporary PostgreSQL database."""
    import psycopg

    with psycopg.connect(PG_ADMIN_DSN, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
        conn.execute(f"CREATE DATABASE {db_name}")


def drop_temp_database(db_name: str) -> None:
    """Drop a temporary PostgreSQL database."""
    import psycopg

    try:
        with psycopg.connect(PG_ADMIN_DSN, autocommit=True) as conn:
            # Terminate existing connections first
            conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname = '{db_name}' AND pid <> pg_backend_pid()"
            )
            conn.execute(f"DROP DATABASE IF EXISTS {db_name}")
    except Exception as e:
        print(f"Warning: Failed to drop temp database {db_name}: {e}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def build_config_dict(db_name: str) -> dict:
    """Build an ARIEL config dict matching the control_assistant template."""
    return {
        "database": {
            "uri": f"postgresql://ariel:ariel@localhost:5432/{db_name}",
        },
        "default_max_results": 10,
        "ingestion": {
            "adapter": "generic_json",
            "source_url": str(DEMO_LOGBOOK),
        },
        "search_modules": {
            "keyword": {"enabled": True},
            "semantic": {
                "enabled": True,
                "provider": "ollama",
                "model": "nomic-embed-text",
            },
        },
        "pipelines": {
            "rag": {"enabled": True, "retrieval_modules": ["keyword", "semantic"]},
            "agent": {"enabled": True, "retrieval_modules": ["keyword", "semantic"]},
        },
        "enhancement_modules": {
            "semantic_processor": {"enabled": False},
            "text_embedding": {
                "enabled": True,
                "provider": "ollama",
                "models": [{"name": "nomic-embed-text", "dimension": 768}],
            },
        },
        "embedding": {"provider": "ollama"},
        "reasoning": {
            "provider": "ollama",
            "model_id": "nomic-embed-text",
            "max_iterations": 5,
            "temperature": 0.1,
            "total_timeout_seconds": 120,
        },
    }


# ---------------------------------------------------------------------------
# Quickstart (migrate, ingest, embed)
# ---------------------------------------------------------------------------


def _init_framework_registry() -> None:
    """Initialize a framework-only registry (no config.yml needed).

    Same approach as osprey.interfaces.ariel.app._create_lifespan — prevents
    get_registry() from failing when ingestion/enhancement code tries to load it.
    """
    import osprey.registry.manager as _reg_mod

    if _reg_mod._registry is None:
        _reg_mod._registry = _reg_mod.RegistryManager(registry_path=None)
        _reg_mod._registry.initialize(silent=True)


async def run_quickstart(config_dict: dict) -> None:
    """Run the ARIEL quickstart workflow: migrate, ingest, embed."""
    from osprey.services.ariel_search import ARIELConfig, create_ariel_service
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrate import run_migrations
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config
    from osprey.services.ariel_search.ingestion import get_adapter

    _init_framework_registry()

    config = ARIELConfig.from_dict(config_dict)

    # Connect and migrate
    print("  Connecting to database...")
    pool = await create_connection_pool(config.database)
    try:
        print("  Running migrations...")
        applied = await run_migrations(pool, config)
        if applied:
            print(f"    {len(applied)} migrations applied")
        else:
            print("    Tables already up to date")

        # Ingest and enhance
        if config.ingestion and config.ingestion.source_url:
            print(f"  Ingesting data from: {config.ingestion.source_url}")
            adapter_instance = get_adapter(config)
            enhancers = create_enhancers_from_config(config)
            if enhancers:
                print(f"    Enhancement modules: {[e.name for e in enhancers]}")

            service = await create_ariel_service(config)
            async with service:
                count = 0
                enhanced_count = 0
                failed_count = 0

                async with service.pool.connection() as conn:
                    async for entry in adapter_instance.fetch_entries():
                        await service.repository.upsert_entry(entry)
                        count += 1

                        for enhancer in enhancers:
                            try:
                                await enhancer.enhance(entry, conn)
                                await service.repository.mark_enhancement_complete(
                                    entry["entry_id"], enhancer.name
                                )
                                enhanced_count += 1
                            except Exception as e:
                                await service.repository.mark_enhancement_failed(
                                    entry["entry_id"], enhancer.name, str(e)
                                )
                                failed_count += 1

                print(f"    {count} entries ingested")
                if enhancers:
                    msg = f"    {enhanced_count} enhancements applied"
                    if failed_count:
                        msg += f", {failed_count} failed"
                    print(msg)
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------


def find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_server(port: int, config_path: str) -> None:
    """Start the ARIEL web server (blocking, intended for daemon thread)."""
    import uvicorn

    from osprey.interfaces.ariel.app import create_app

    app = create_app(config_path=config_path)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def check_server(base_url: str, timeout: int = 5) -> bool:
    """Check if a server is reachable at the given URL."""
    import urllib.request

    try:
        urllib.request.urlopen(f"{base_url}/health", timeout=timeout)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------


async def capture_screenshots(base_url: str, wait_for_startup: bool = False) -> None:
    """Capture screenshots of each view using Playwright."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Error: playwright is not installed.")
        print("Install with: pip install playwright && playwright install chromium")
        sys.exit(1)

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    if wait_for_startup:
        print(f"Waiting for server at {base_url}...")
        for _ in range(60):
            if check_server(base_url):
                break
            await asyncio.sleep(1)
        else:
            print("Error: Server did not become ready within 60 seconds.")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Capture ARIEL web interface screenshots")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL of a running ARIEL instance (skip setup/teardown, just capture)",
    )
    args = parser.parse_args()

    # Escape hatch: capture from a running instance
    if args.url:
        print(f"Connecting to ARIEL instance at {args.url}...")
        if not check_server(args.url):
            print(f"Error: No ARIEL instance found at {args.url}")
            sys.exit(1)
        asyncio.run(capture_screenshots(args.url))
        return

    # Self-contained mode: full setup → capture → teardown
    print("ARIEL Screenshot Capture (self-contained mode)")
    print("=" * 50)

    # Pre-flight checks
    print("\nPre-flight checks...")
    check_postgres()
    print("  PostgreSQL: OK")
    check_ollama()
    print("  Ollama: OK")

    db_name = f"ariel_screenshots_{int(time.time())}"
    tmp_config_path = None

    try:
        # Create temp database
        print(f"\nCreating temp database: {db_name}")
        create_temp_database(db_name)

        # Build config and run quickstart
        config_dict = build_config_dict(db_name)
        print("\nRunning quickstart (migrate, ingest, embed)...")
        asyncio.run(run_quickstart(config_dict))

        # Write temp config file for the web server
        import yaml

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", prefix="ariel_screenshots_", delete=False
        )
        yaml.safe_dump({"ariel": config_dict}, tmp)
        tmp.flush()
        tmp.close()
        tmp_config_path = tmp.name

        # Start web server
        port = find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        print(f"\nStarting web server on port {port}...")

        server_thread = threading.Thread(
            target=start_server, args=(port, tmp_config_path), daemon=True
        )
        server_thread.start()

        # Capture screenshots
        asyncio.run(capture_screenshots(base_url, wait_for_startup=True))

    finally:
        # Teardown
        print("\nCleaning up...")
        drop_temp_database(db_name)
        print(f"  Dropped database: {db_name}")

        if tmp_config_path:
            import os

            try:
                os.unlink(tmp_config_path)
                print(f"  Deleted temp config: {tmp_config_path}")
            except OSError:
                pass


if __name__ == "__main__":
    main()
