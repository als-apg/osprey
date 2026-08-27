"""MCP server lifecycle: startup timing, logging setup, and entry point."""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("osprey.mcp_server.startup")

#: The one transport the audit middleware is installed for. Everything else --
#: today only the event dispatcher, which is multi-tenant and carries its own
#: token auth -- is skipped, loudly.
STDIO_TRANSPORT = "stdio"

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
    that read the shared ``var/agent_data/`` root. Session isolation is handled
    at the index level via ``ArtifactEntry.session_id`` — never in the store
    path. Rooting the store at the session-relocated path
    (``resolve_agent_data_root`` appends ``sessions/<id>/`` when
    ``OSPREY_SESSION_ID`` is set) would make a session's artifacts invisible
    to the gallery.

    Also subscribes the artifact-activity listeners, so every save and delete
    an MCP server performs shows up in the Web Terminal. Registration is
    idempotent — this function runs more than once in a process under test.

    The listeners are armed per PROCESS, not per store instance: they hang off
    the ArtifactStore class, so every store built in a process that called this
    emits. Code paths that run in their own process (dispatch ingest, retention
    sweeps, a separately launched gallery) never call this and stay silent.

    .. warning::
        That process boundary is not guaranteed. ``ServerLauncher`` can start
        the artifact gallery IN-THREAD inside this very process — the store
        auto-launches it on first save when no other process owns the port
        (``artifact_store.py`` save paths → ``ensure_artifact_server``). In
        that topology a HUMAN deleting an artifact in the gallery UI fires the
        same delete listener, and the activity frame is attributed to the
        agent. Telling the two apart needs origin plumbing through the store or
        the gallery route; until then this is a known limitation, recorded here
        rather than papered over with a thread-name guess.
    """
    from osprey.mcp_server.artifact_activity import register_artifact_activity_listeners
    from osprey.stores.artifact_store import initialize_artifact_store
    from osprey.utils.workspace import resolve_shared_data_root

    with startup_timer("workspace_singletons"):
        initialize_artifact_store(workspace_root=resolve_shared_data_root())
        register_artifact_activity_listeners()


def fastmcp_transport() -> str:
    """The transport fastmcp itself will use, read from fastmcp's own settings.

    The single seam the audit-middleware skip predicate reads. It is
    deliberately NOT ``os.environ["FASTMCP_TRANSPORT"]``: ``fastmcp.settings``
    snapshots the environment once, when fastmcp is first imported, and it is
    that snapshot -- not the current environment -- that decides which transport
    ``server.run()`` actually speaks. Reading the environment instead would let
    a late write skip the audit layer on a server that is still talking stdio.

    Callers must only reach this AFTER ``load_dotenv_from_project()``, so a
    project ``.env`` has already landed in the environment fastmcp reads.
    """
    import fastmcp

    return fastmcp.settings.transport


def install_audit_middleware(server: Any, label: str) -> None:
    """Install :class:`~osprey.mcp_server.audit_middleware.AuditMiddleware` on ``server``.

    Skipped for any server fastmcp will not serve over stdio -- the event
    dispatcher is the only one today -- and the skip is never silent: one
    WARNING names the server that went without an audit layer.

    The middleware import lives HERE, inside a function, not at module scope:
    the middleware module imports fastmcp, and a module-scope import in
    ``startup.py`` would run before ``run_mcp_server`` has loaded the project
    ``.env``, freezing ``fastmcp.settings`` against the wrong environment. See
    :func:`fastmcp_transport`.

    Args:
        server: The FastMCP instance a server module's ``create_server()`` built.
        label: Human-readable server name, used in the skip warning.
    """
    transport = fastmcp_transport()
    if transport != STDIO_TRANSPORT:
        logger.warning(
            "MCP audit middleware NOT installed on the %s server: fastmcp transport "
            "is %r, not %r. Tool calls on this server are not written to the audit "
            "ledger and the readonly clamp does not apply to them.",
            label,
            transport,
            STDIO_TRANSPORT,
        )
        return

    from osprey.mcp_server.audit_middleware import AuditMiddleware

    server.add_middleware(AuditMiddleware())


def run_mcp_server(server_module: str) -> None:
    """Shared entry point for all MCP servers.

    Handles dotenv loading, logging setup, audit-middleware install, and server
    startup. MCP servers speak JSON-RPC over stdio, so stdout must carry nothing
    but protocol frames; ``configure_logging()`` routes every record to stderr.

    Every framework MCP server reaches its ``create_server()`` through here --
    the channel-finder variants included, via
    :func:`~osprey.mcp_server.channel_finder_common.run_cf_main` -- so the audit
    middleware has exactly one install site.

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

    with startup_timer("audit_middleware"):
        install_audit_middleware(server, _server_label)

    elapsed_total_ms = (time.perf_counter() - t_total) * 1000
    print(
        f"[STARTUP-TIMING] {_server_label} | total_startup: {elapsed_total_ms:.0f}ms",
        file=sys.stderr,
        flush=True,
    )

    server.run()
