"""MCP server lifecycle: startup timing, logging setup, and entry point."""

import logging
import sys
import time
from contextlib import contextmanager

logger = logging.getLogger("osprey.mcp_server.startup")

# ---------------------------------------------------------------------------
# Startup timing instrumentation
# ---------------------------------------------------------------------------
_server_label: str = "unknown"


@contextmanager
def startup_timer(label: str):
    """Context manager that logs ``[STARTUP-TIMING] <server> | <label>: <ms>ms`` to stderr.

    Uses ``time.perf_counter()`` for sub-millisecond precision.
    Output goes directly to stderr so it is visible even before the logging
    subsystem is fully configured.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[STARTUP-TIMING] {_server_label} | {label}: {elapsed_ms:.0f}ms",
            file=sys.stderr,
            flush=True,
        )


def prime_config_builder() -> None:
    """Prime the main ConfigBuilder with config.yml from OSPREY_CONFIG.

    Sets the global ConfigBuilder singleton so that ``get_config_value()``
    works throughout MCP tool code. Does NOT initialize the full framework
    registry — MCP servers don't need it (they use their own lightweight
    registries for connectors, ARIEL, channel-finder, etc.).
    """
    import os

    osprey_config = os.environ.get("OSPREY_CONFIG")
    if osprey_config:
        osprey_config = os.path.expandvars(osprey_config)
        try:
            from osprey.utils.config import get_config_builder

            with startup_timer("config_builder"):
                get_config_builder(config_path=osprey_config, set_as_default=True)
            logger.info("Main ConfigBuilder primed from OSPREY_CONFIG: %s", osprey_config)
            try:
                from osprey.stores.type_registry import load_categories_from_config

                n = load_categories_from_config()
                if n:
                    logger.info("Loaded %d custom category/ies from config", n)
            except Exception as exc:
                logger.warning("Custom category loading failed (non-fatal): %s", exc)
        except Exception as exc:
            logger.warning("ConfigBuilder priming failed (non-fatal): %s", exc)


def initialize_workspace_singletons() -> None:
    """Initialize the ArtifactStore singleton on the SHARED data root.

    The artifact store is served by long-lived daemons (the artifact gallery)
    that read the shared ``_agent_data/`` root. Session isolation is handled
    at the index level via ``ArtifactEntry.session_id`` — never in the store
    path. Rooting the store at the session-relocated path
    (``resolve_agent_data_root`` appends ``sessions/<id>/`` when
    ``OSPREY_SESSION_ID`` is set) would make a session's artifacts invisible
    to the gallery.
    """
    from osprey.stores.artifact_store import initialize_artifact_store
    from osprey.utils.workspace import resolve_shared_data_root

    with startup_timer("workspace_singletons"):
        initialize_artifact_store(workspace_root=resolve_shared_data_root())


def run_mcp_server(server_module: str) -> None:
    """Shared entry point for all MCP servers.

    Handles dotenv loading, logging setup, and server startup. MCP servers speak
    JSON-RPC over stdio, so stdout must carry nothing but protocol frames;
    ``configure_logging()`` routes every record to stderr.

    Args:
        server_module: Dotted path to the module containing ``create_server()``.
    """
    global _server_label

    from importlib import import_module

    # Derive a human-readable label from the module path
    # e.g. "osprey.mcp_server.workspace.server" -> "workspace"
    parts = server_module.split(".")
    _server_label = parts[-2] if len(parts) >= 2 else server_module

    t_total = time.perf_counter()

    from osprey.mcp_env import load_dotenv_from_project
    from osprey.utils.logger import configure_logging

    with startup_timer("dotenv_load"):
        load_dotenv_from_project()

    configure_logging()

    with startup_timer("import_server_module"):
        mod = import_module(server_module)

    with startup_timer("create_server"):
        server = mod.create_server()

    elapsed_total_ms = (time.perf_counter() - t_total) * 1000
    print(
        f"[STARTUP-TIMING] {_server_label} | total_startup: {elapsed_total_ms:.0f}ms",
        file=sys.stderr,
        flush=True,
    )

    server.run()
