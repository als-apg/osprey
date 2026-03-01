"""Cross-layer workspace and config resolution utilities.

These functions were originally in ``mcp_server.common`` but are used across
``interfaces/``, ``cli/``, ``services/``, and ``mcp_server/`` layers.
Living in ``utils/`` eliminates layering violations.
"""

import logging
from pathlib import Path

logger = logging.getLogger("osprey.utils.workspace")

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


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root from config.

    Absolute paths are returned as-is. Relative paths are resolved
    against ``project_root`` from the active configuration.

    Args:
        path_str: Path string (absolute or relative to project root)

    Returns:
        Resolved absolute Path object
    """
    from osprey.utils.config import get_config_builder

    config_builder = get_config_builder()
    project_root = Path(config_builder.get("project_root"))
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path
