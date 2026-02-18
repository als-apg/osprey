"""Shared utilities for OSPREY MCP servers.

Extracted from the original monolithic server.py so that the control-system,
python-executor, and workspace servers can each import a common set of helpers
without depending on each other.
"""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger("osprey.mcp_server.common")


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

    # Reduce third-party noise (mirrors _setup_rich_logging)
    for lib in ["httpx", "httpcore", "requests", "urllib3", "LiteLLM"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Config loader (standalone, no LangGraph dependency)
# ---------------------------------------------------------------------------

_config_cache: dict | None = None
_config_cache_path: Path | None = None


def resolve_config_path() -> Path:
    """Resolve the path to config.yml.

    Resolution order:
      1. ``OSPREY_CONFIG`` environment variable (with shell variable expansion)
      2. ``./config.yml`` relative to the current working directory
    """
    import os

    return Path(os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml"))))


def load_osprey_config() -> dict:
    """Load OSPREY configuration from config.yml (cached after first call).

    Delegates to the framework's ``ConfigBuilder`` so that ``${VAR:-default}``
    environment-variable placeholders are resolved consistently.

    Resolution order:
      1. ``OSPREY_CONFIG`` environment variable
      2. ``./config.yml`` relative to the current working directory

    Returns:
        Parsed YAML dict (with env vars resolved), or empty dict if the file is missing.
    """
    global _config_cache, _config_cache_path

    if _config_cache is not None:
        return _config_cache

    config_path = resolve_config_path()
    _config_cache_path = config_path
    try:
        from osprey.utils.config import get_config_builder

        builder = get_config_builder(config_path=str(config_path), set_as_default=True)
        _config_cache = builder.raw_config  # env vars resolved
    except (FileNotFoundError, Exception):
        _config_cache = {}
    return _config_cache


def reset_config_cache() -> None:
    """Clear the cached config — used between tests."""
    global _config_cache, _config_cache_path
    _config_cache = None
    _config_cache_path = None


def resolve_workspace_root() -> Path:
    """Resolve the workspace root directory from config.

    Uses ``workspace.base_dir`` from config.yml, resolved relative to the
    config file's parent directory (the project root).  Falls back to
    ``./osprey-workspace`` relative to cwd if no config is found.
    """
    config = load_osprey_config()
    base_dir = config.get("workspace", {}).get("base_dir", "./osprey-workspace")

    config_path = resolve_config_path()
    if config_path.exists():
        project_root = config_path.parent
    else:
        project_root = Path.cwd()

    resolved = (project_root / base_dir).resolve()

    import os

    session_id = os.environ.get("OSPREY_SESSION_ID")
    if session_id:
        resolved = resolved / "sessions" / session_id

    logger.debug("Workspace root resolved to %s", resolved)
    return resolved


# ---------------------------------------------------------------------------
# Structured error helper
# ---------------------------------------------------------------------------
def make_error(
    error_type: str,
    error_message: str,
    suggestions: list[str] | None = None,
) -> dict:
    """Build the cross-team standard error envelope.

    All MCP tools must return this shape on failure so that Claude Code
    can reliably detect and display errors.
    """
    return {
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "suggestions": suggestions or [],
    }


# ---------------------------------------------------------------------------
# Config bridge: prime the main ConfigBuilder
# ---------------------------------------------------------------------------
def prime_config_builder() -> None:
    """Prime the main ConfigBuilder with config.yml from OSPREY_CONFIG.

    Also initializes the main OSPREY registry (providers, connectors, etc.)
    so downstream services (ChannelFinderService, LimitsValidator,
    pattern_detection, ARIEL) find their config.
    """
    import os

    osprey_config = os.environ.get("OSPREY_CONFIG")
    if osprey_config:
        osprey_config = os.path.expandvars(osprey_config)
        try:
            from osprey.utils.config import get_config_builder

            get_config_builder(config_path=osprey_config, set_as_default=True)
            logger.info("Main ConfigBuilder primed from OSPREY_CONFIG: %s", osprey_config)
        except Exception as exc:
            logger.warning("ConfigBuilder priming failed (non-fatal): %s", exc)

    # Initialize the main OSPREY registry (providers, connectors, etc.)
    try:
        from osprey.registry import initialize_registry

        initialize_registry(auto_export=False, silent=True, config_path=osprey_config)
        logger.info("Main OSPREY registry initialized")
    except Exception as exc:
        logger.warning("OSPREY registry initialization failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Workspace singleton initialization
# ---------------------------------------------------------------------------
def initialize_workspace_singletons(workspace_root: Path) -> None:
    """Initialize ArtifactStore, DataContext, and MemoryStore singletons for a workspace."""
    from osprey.mcp_server.artifact_store import initialize_artifact_store
    from osprey.mcp_server.data_context import initialize_data_context
    from osprey.mcp_server.memory_store import initialize_memory_store

    initialize_artifact_store(workspace_root=workspace_root)
    initialize_data_context(workspace_root=workspace_root)
    initialize_memory_store(workspace_root=workspace_root)


# ---------------------------------------------------------------------------
# Gallery URL helper (moved from tools/artifact_save.py)
# ---------------------------------------------------------------------------
def gallery_url() -> str:
    """Build the gallery base URL from config."""
    config = load_osprey_config()
    art_config = config.get("artifact_server", {})
    host = art_config.get("host", "127.0.0.1")
    port = art_config.get("port", 8086)
    return f"http://{host}:{port}"


def web_terminal_url() -> str:
    """Build the web terminal base URL from config."""
    config = load_osprey_config()
    wt = config.get("web_terminal", {})
    return f"http://{wt.get('host', '127.0.0.1')}:{wt.get('port', 8087)}"


def post_json(url: str, payload: dict, *, timeout: int = 3) -> None:
    """Fire-and-forget JSON POST to a local HTTP endpoint.

    Non-fatal: logs a warning if the target is unreachable.
    Used by focus tools and panel-focus notifications.
    """
    import json as _json
    import urllib.request

    try:
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout)
    except Exception as exc:
        logger.warning("POST %s failed (non-fatal): %s", url, exc)


def notify_panel_focus(panel_id: str, url: str | None = None) -> None:
    """Fire-and-forget POST to switch the Web Terminal's active panel.

    Non-fatal if the web terminal is not running (CLI-only mode).
    """
    base = web_terminal_url()
    payload: dict = {"panel": panel_id}
    if url is not None:
        payload["url"] = url
    post_json(f"{base}/api/panel-focus", payload, timeout=2)


# ---------------------------------------------------------------------------
# MCP server entry point helper
# ---------------------------------------------------------------------------
def run_mcp_server(server_module: str) -> None:
    """Shared entry point for all MCP servers.

    Handles dotenv loading, stderr logging redirect, and server startup.

    Args:
        server_module: Dotted path to the module containing ``create_server()``.
    """
    from importlib import import_module

    from osprey.mcp_env import load_dotenv_from_project

    load_dotenv_from_project()
    redirect_logging_to_stderr()

    mod = import_module(server_module)
    server = mod.create_server()
    server.run()
