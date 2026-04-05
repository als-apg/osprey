"""Cross-layer workspace and config resolution utilities.

These functions were originally in ``mcp_server.common`` but are used across
``interfaces/``, ``cli/``, ``services/``, and ``mcp_server/`` layers.
Living in ``utils/`` eliminates layering violations.
"""

import logging
from pathlib import Path

logger = logging.getLogger("osprey.utils.workspace")


def resolve_config_path() -> Path:
    """Resolve the path to config.yml.

    Resolution order:
      1. ``OSPREY_CONFIG`` environment variable (with shell variable expansion)
      2. ``./config.yml`` relative to the current working directory
    """
    import os

    return Path(os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml"))))


def load_osprey_config() -> dict:
    """Load OSPREY configuration (delegates to ConfigBuilder singleton).

    Delegates to the framework's ``ConfigBuilder`` so that ``${VAR:-default}``
    environment-variable placeholders are resolved consistently.

    Resolution order:
      1. ``OSPREY_CONFIG`` environment variable
      2. ``./config.yml`` relative to the current working directory

    Returns:
        Parsed YAML dict (with env vars resolved), or empty dict if the file is missing.
    """
    config_path = resolve_config_path()
    try:
        from osprey.utils.config import get_config_builder

        builder = get_config_builder(config_path=str(config_path), set_as_default=True)
        return builder.raw_config
    except (FileNotFoundError, Exception):
        return {}


def reset_config_cache() -> None:
    """Clear all config caches — used between tests."""
    from osprey.utils import config as config_module

    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()


def resolve_agent_data_root() -> Path:
    """Resolve the agent data root directory from config.

    Uses ``agent_data.base_dir`` from config.yml, resolved relative to the
    config file's parent directory (the project root).  Falls back to
    ``./_agent_data`` relative to cwd if no config is found.
    """
    config = load_osprey_config()
    base_dir = config.get("agent_data", {}).get("base_dir", "./_agent_data")

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

    logger.debug("Agent data root resolved to %s", resolved)
    return resolved


def resolve_shared_data_root() -> Path:
    """Resolve the agent data root WITHOUT session-path isolation.

    Use for stores whose data must be visible to long-lived daemons
    (gallery, ARIEL) that run outside any specific session.  Logical
    session isolation is handled at the index level via entry metadata
    (e.g. ``ArtifactEntry.session_id``).
    """
    config = load_osprey_config()
    base_dir = config.get("agent_data", {}).get("base_dir", "./_agent_data")
    config_path = resolve_config_path()
    project_root = config_path.parent if config_path.exists() else Path.cwd()
    resolved = (project_root / base_dir).resolve()
    logger.debug("Shared data root resolved to %s", resolved)
    return resolved


# Backward-compatible alias
resolve_workspace_root = resolve_agent_data_root


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
