"""MCP server lifecycle: startup timing, logging redirect, and entry point."""

import logging
import sys
import time
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler

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


def redirect_logging_to_stderr() -> None:
    """Pre-install a RichHandler that writes to *stderr* on the root logger.

    MCP servers communicate over stdio — stdout is reserved for JSON-RPC
    messages.  The framework's ``_setup_rich_logging()`` (in
    ``osprey.utils.logger``) creates a ``RichHandler`` that writes to
    *stdout* by default, which corrupts the MCP transport.

    By installing a stderr-based ``RichHandler`` first, the framework's
    ``_setup_rich_logging()`` sees an existing ``RichHandler`` and skips
    its own registration, keeping stdout clean for MCP.

    Call this **before** ``create_server()`` in every MCP ``__main__.py``.
    """
    root = logging.getLogger()

    # Guard: only install once
    for handler in root.handlers:
        if isinstance(handler, RichHandler):
            return

    root.setLevel(logging.INFO)

    console = Console(
        stderr=True,
        force_terminal=True,
        width=120,
        color_system="truecolor",
    )

    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_time=True,
        show_level=True,
        tracebacks_show_locals=False,
    )

    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for lib in ["httpx", "httpcore", "requests", "urllib3", "LiteLLM"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


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
        except Exception as exc:
            logger.warning("ConfigBuilder priming failed (non-fatal): %s", exc)


def initialize_workspace_singletons() -> None:
    """Initialize ArtifactStore singleton with the shared (session-independent) root.

    Artifacts are shared across sessions — the gallery daemon needs a single
    stable directory.  Session isolation is handled by the ``session_id`` field
    on each :class:`~osprey.stores.artifact_store.ArtifactEntry`.
    """
    from osprey.stores.artifact_store import initialize_artifact_store
    from osprey.utils.workspace import resolve_shared_data_root

    with startup_timer("workspace_singletons"):
        initialize_artifact_store(workspace_root=resolve_shared_data_root())


def run_mcp_server(server_module: str) -> None:
    """Shared entry point for all MCP servers.

    Handles dotenv loading, stderr logging redirect, and server startup.

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

    with startup_timer("dotenv_load"):
        load_dotenv_from_project()

    redirect_logging_to_stderr()

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
