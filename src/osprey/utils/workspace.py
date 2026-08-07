"""Cross-layer workspace and config resolution utilities.

These functions were originally in ``mcp_server.common`` but are used across
``interfaces/``, ``cli/``, ``services/``, and ``mcp_server/`` layers.
Living in ``utils/`` eliminates layering violations.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("osprey.utils.workspace")

#: Agent-data root used when a config declares no ``agent_data.base_dir``.
DEFAULT_AGENT_DATA_BASE_DIR = "./_agent_data"

#: Subdirectory of the agent-data root holding the simulation's mutable
#: ``active_scenarios`` state file.
SIMULATION_STATE_DIR_NAME = "simulation"

#: Config key naming that directory explicitly (relative paths resolve against
#: the project root). Unset — the normal case — puts it under the agent-data
#: root. It lives here, rather than in the simulation package, so
#: :data:`osprey.utils.config.RUNTIME_WRITE_PATH_KEYS` and
#: :func:`osprey.simulation.engine.resolve_state_dir` share one spelling without
#: ``config`` having to import the engine (which would pull numpy into every
#: config load).
SIMULATION_STATE_DIR_CONFIG_KEY = "simulation.state_dir"


def dotted_config_str(config: Mapping[str, Any] | None, key: str) -> str | None:
    """Read a dotted config key, or ``None`` if it does not name a real string.

    ``None`` is returned for every way the key can fail to answer — a missing
    segment, a segment that is not a mapping, a non-string value, or an empty
    one — because each of those means the same thing to every caller: the key
    is unset, use the default. A mistyped key therefore falls through to the
    default rather than raising.

    The single reader for the path-shaped config keys, so the checks that
    *reject* a value and the resolvers that *use* it cannot disagree about what
    the config says. They previously differed on ``dict`` versus ``Mapping``,
    which is only invisible while every config arrives straight from the YAML
    loader.

    Args:
        config: Loaded config mapping, or ``None``.
        key: Dotted path, e.g. ``simulation.state_dir``.
    """
    value: Any = config or {}
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, str) and value else None


def anchored_path(value: str, project_root: Path) -> Path:
    """Expand ``~`` in a configured path and anchor a relative one at the project.

    The other half of :func:`dotted_config_str`: a configured path is absolute,
    home-relative, or project-relative, and every consumer has to make the same
    three-way choice before comparing paths or creating directories.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def agent_data_base_dir(config: Mapping[str, Any] | None) -> str:
    """Read the agent-data root out of an already-loaded config mapping.

    ``agent_data.base_dir`` is the only key that names this directory. Callers
    holding a config dict (and their own anchor for relative paths — the health
    checks and the compose generator anchor on ``project_root``) read it through
    here; callers with no config in hand use :func:`resolve_agent_data_root`,
    which anchors on the config file's own directory.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The configured base directory, possibly relative to the caller's anchor.
    """
    section = (config or {}).get("agent_data") or {}
    if not isinstance(section, Mapping):
        return DEFAULT_AGENT_DATA_BASE_DIR
    return str(section.get("base_dir") or DEFAULT_AGENT_DATA_BASE_DIR)


def resolve_simulation_state_dir(config: Mapping[str, Any] | None, project_root: Path) -> Path:
    """Resolve the directory holding the mutable ``active_scenarios`` state file.

    The state file is the one piece of simulation state that changes after a
    build, so it lives under the agent-data root rather than next to
    ``machine.json``: everything in a project's ``data/`` tree is build-owned
    and checksummed (see
    :func:`osprey.cli.templates.manifest.calculate_file_checksums`), and a
    scenario switch is not project drift.

    Lives here rather than in the simulation package so the three sides that
    must agree — the engine that writes the file, the compose generator that
    renders the container's bind-mount source, and the build injector that
    pre-creates it — resolve it identically without ``deployment`` importing
    numpy through :mod:`osprey.simulation.engine`. Re-exported there as
    ``resolve_state_dir``.

    A mistyped :data:`SIMULATION_STATE_DIR_CONFIG_KEY` (anything but a non-empty
    string) falls through to the default rather than raising, matching how
    :func:`osprey.utils.config.find_runtime_write_paths_under_data` reads the
    same key.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.
        project_root: Root the relative paths above resolve against.

    Returns:
        Absolute path to the state directory (not created here).
    """
    config = config or {}
    configured = dotted_config_str(config, SIMULATION_STATE_DIR_CONFIG_KEY)
    relative = configured or f"{agent_data_base_dir(config)}/{SIMULATION_STATE_DIR_NAME}"
    return anchored_path(relative, project_root)


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
    base_dir = agent_data_base_dir(load_osprey_config())

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
    base_dir = agent_data_base_dir(load_osprey_config())
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
