"""Where a process finds its config and its agent data.

The rendered ``config.yml``, the repo root it belongs to, and the agent-data
root under it — resolved once here, for the connectors and for every framework
layer that loads the same config. The rest of the deployment repo's layout
(layer directories, the image staging area, the audit and archive zones) is the
framework's, in ``osprey.utils.workspace``.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "BUILD_DIR_NAME",
    "DEFAULT_AGENT_DATA_BASE_DIR",
    "RENDERED_CONFIG_RELPATH",
    "SIMULATION_STATE_DIR_CONFIG_KEY",
    "SIMULATOR_LOG_DIR_RELPATH",
    "agent_data_base_dir",
    "anchored_path",
    "dotted_config_str",
    "load_osprey_config",
    "rendered_config_path",
    "repo_root_for_config",
    "reset_config_cache",
    "resolve_agent_data_root",
    "resolve_config_path",
    "resolve_path",
    "resolve_project_root",
    "resolve_shared_data_root",
    "resolve_simulation_state_dir",
]

logger = logging.getLogger("osprey_connectors.workspace")

#: The render zone inside a deployment repo, re-created wholesale by every
#: ``osprey build``. The rendered config lives in it
#: (:data:`RENDERED_CONFIG_RELPATH`), which is how :func:`repo_root_for_config`
#: tells a host's render from a container's project directory.
BUILD_DIR_NAME = "build"

#: The durable state zone, which survives ``build/`` being re-created. Private:
#: the framework owns the zone layout (``osprey.utils.workspace.STATE_DIR_NAME``,
#: held equal to this by test); it is spelled here only to place
#: :data:`DEFAULT_AGENT_DATA_BASE_DIR`.
_STATE_DIR_NAME = "var"

#: The file that makes a directory a deployment repo root, the marker
#: :func:`resolve_project_root` walks up to. Private: the framework owns it
#: (``osprey.utils.workspace.PROFILE_FILENAME``, held equal to this by test).
_PROFILE_FILENAME = "profile.yml"

#: The rendered config, relative to the repo root. It is an *output*: the source
#: of truth is ``profile.yml`` at the repo root, and this file is what a build
#: derives from it. Every producer and consumer of that path spells it through
#: this constant so the two cannot drift.
RENDERED_CONFIG_RELPATH = f"{BUILD_DIR_NAME}/config.yml"

#: Agent-data root used when a config declares no ``agent_data.base_dir``.
#: Under ``var/`` rather than beside the render: memory, sessions, and artifacts
#: have to survive ``rm -rf build/``, which is the one guarantee the build zone
#: does not offer.
DEFAULT_AGENT_DATA_BASE_DIR = f"{_STATE_DIR_NAME}/agent_data"

#: The simulator's model-log directory, relative to the repo root: the composite
#: appends one ``<model>.log`` per served physics model here, from whichever
#: process serves it. Durable, like the agent-data root beside it.
SIMULATOR_LOG_DIR_RELPATH = f"{_STATE_DIR_NAME}/simulator"

#: Subdirectory of the agent-data root holding the simulation's mutable
#: ``active_scenarios`` state file.
_SIMULATION_STATE_DIR_NAME = "simulation"

#: Config key naming that directory explicitly (relative paths resolve against
#: the project root). Unset — the normal case — puts it under the agent-data
#: root. :class:`osprey_connectors.config.ConfigBuilder`'s runtime-write
#: advisory and :func:`resolve_simulation_state_dir` read this one spelling.
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
    the config says — including on ``dict`` versus ``Mapping``, a difference
    that stays invisible only while every config arrives straight from the YAML
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


def rendered_config_path(repo_root: Path | str) -> Path:
    """Path to the rendered ``config.yml`` of the repo rooted at *repo_root*.

    The one place the ``build/config.yml`` spelling is assembled, so that a
    caller holding a repo root never has to know which zone the render lands in.
    """
    return Path(repo_root) / RENDERED_CONFIG_RELPATH


def agent_data_base_dir(config: Mapping[str, Any] | None) -> str:
    """Read the agent-data root out of an already-loaded config mapping.

    ``agent_data.base_dir`` is the only key that names this directory. Callers
    holding a config dict read it through here and anchor the result themselves
    (the health checks and the compose generator anchor on ``project_root``);
    callers with no config in hand use :func:`resolve_agent_data_root`, which
    loads the config and anchors on the same ``project_root``.

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
    build, so it lives under the agent-data root rather than in ``data/``:
    everything in a project's ``data/`` tree is build-owned and checksummed
    (see :func:`osprey.cli.templates.manifest.calculate_file_checksums`), and
    a scenario switch is not project drift.

    The one spelling serves config, the compose generator that renders the
    container's bind-mount source, and the build injector that pre-creates it,
    so all three resolve the same directory.

    A mistyped :data:`SIMULATION_STATE_DIR_CONFIG_KEY` (anything but a non-empty
    string) falls through to the default rather than raising, matching how
    :class:`osprey_connectors.config.ConfigBuilder`'s runtime-write advisory
    reads the same key.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.
        project_root: Root the relative paths above resolve against.

    Returns:
        Absolute path to the state directory (not created here).
    """
    config = config or {}
    configured = dotted_config_str(config, SIMULATION_STATE_DIR_CONFIG_KEY)
    relative = configured or f"{agent_data_base_dir(config)}/{_SIMULATION_STATE_DIR_NAME}"
    return anchored_path(relative, project_root)


def resolve_config_path() -> Path:
    """Resolve the path to the rendered ``config.yml``.

    Resolution order:
      1. ``OSPREY_CONFIG`` environment variable (with shell variable expansion).
         The CLI sets it on the host, every framework MCP server definition
         sets it, and so does the compose service of every container whose
         working directory does not hold the render: the Bluesky bridge, its
         queueserver and the Bluesky sidecar run from ``/app`` with the config
         mounted at ``/app/project/config.yml``.
      2. ``build/config.yml`` under the current working directory — standing in
         a repo root with a render in it.
      3. ``config.yml`` under the current working directory — a container
         project directory, whose root *is* the render, and the path reported
         when nothing exists at all.

    ``CONFIG_FILE`` is deliberately NOT read here, even though
    :func:`osprey.agent_runner.artifact_resolve.deployed_config_path` reads it
    ahead of ``OSPREY_CONFIG``. That helper answers for one deployed process
    whose compose service sets the variable; this one answers for the framework
    everywhere, and ``CONFIG_FILE`` is a name generic enough that honoring it
    globally would let an unrelated export redirect every OSPREY process on the
    host. Where a container is handed both, they agree by construction rather
    than by precedence. The Bluesky compose services set both to the one
    mounted path. The dispatch worker's sets ``CONFIG_FILE`` to
    ``/app/<project>/build/config.yml``, and the image's ``WORKDIR`` is
    ``/app/<project>``, so step 2 above finds that same file.
    """
    import os

    configured = os.environ.get("OSPREY_CONFIG")
    if configured:
        return Path(os.path.expandvars(configured))

    cwd = Path.cwd()
    rendered = rendered_config_path(cwd)
    return rendered if rendered.is_file() else cwd / "config.yml"


def load_osprey_config() -> dict:
    """Load OSPREY configuration (delegates to ConfigBuilder singleton).

    Delegates to the framework's ``ConfigBuilder`` so that ``${VAR:-default}``
    environment-variable placeholders are resolved consistently.

    The file is the one :func:`resolve_config_path` names.

    Returns:
        Parsed YAML dict (with env vars resolved), or empty dict if the file is missing.
    """
    config_path = resolve_config_path()
    try:
        from osprey_connectors.config import get_config_builder

        builder = get_config_builder(config_path=str(config_path), set_as_default=True)
        return builder.raw_config
    except (FileNotFoundError, Exception):
        return {}


def reset_config_cache() -> None:
    """Clear all config caches — used between tests.

    The warn-once latch on the facility-timezone fallback is cleared with them.
    It is process state in exactly the way a cache is: left set, whichever test
    happened to run first would consume the single warning and every later test
    asserting on it would see none — an order-dependent pass that says nothing
    about the code.
    """
    from osprey_connectors import config as config_module

    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()
    config_module._tz_fallback_warned = False


def repo_root_for_config(config_path: Path | str) -> Path:
    """The repo root that a config file at *config_path* belongs to.

    The rendered config lives at ``<repo>/build/config.yml``, so the repo is
    two levels up — except in a container, whose project directory *is* the
    render and holds ``config.yml`` at its root. One helper for both, so the
    "is this the build zone?" question is asked in exactly one place.

    Known non-goal: a FLAT project whose own directory is literally named
    ``build`` is read as a render, and its parent is returned. Accepted
    — every runtime path derives the root through this helper, so a regen or a
    status check agreeing with the runtime beats a special case that would make
    the two disagree.
    """
    parent = Path(config_path).parent
    return parent.parent if parent.name == BUILD_DIR_NAME else parent


def resolve_project_root(config: Mapping[str, Any] | None = None) -> Path:
    """Resolve the repo root that relative configured paths anchor against.

    ``project_root`` names the deployment repo — the directory holding
    ``profile.yml`` — and not the directory holding ``config.yml``: the render
    lives one level down, in ``build/``, so anchoring on the config file's own
    parent would put every relative path inside the disposable zone.

    Resolution order:
      1. The config's own ``project_root`` key, when it names a directory that
         exists here. That qualifier is what makes it right inside a container,
         where the repo was built at one path and runs at another: a deployed
         service reads a config STAGED on the host — the compose generator
         flattens the host render into ``build/services/<svc>/config.yml``, and
         the service's compose file bind-mounts that over the one in the image —
         so its ``project_root`` is a host path that names nothing in here.
         Anchoring on it would put the
         agent-data root, and every other relative path, outside the mounted
         volume. Same rule and same reason as
         :func:`osprey.deployment.compose_generator.resolve_repo_root`.
      2. The repo the resolved config path sits in — its parent, or that
         parent's parent when the config is in the ``build/`` zone. Filesystem
         truth, and correct in every container layout: the mount target names
         the render, so unwrapping it names the repo.
      3. The nearest ancestor of the working directory holding ``profile.yml``,
         the marker every OSPREY verb walks up to. Reached only when no config
         file exists, so it can never out-vote one; without it a process started
         in an unbuilt render would anchor every relative path inside
         ``build/``.
      4. The configured value after all, when there is no config file and no
         marker to derive from — a build rendered with ``--runtime-root``
         records a path that exists only on the machine it will run on, and has
         nothing better.
      5. The current working directory, when there is neither.
         ``CLAUDE_PROJECT_DIR`` is not an input: this resolver serves processes
         outside any agent session as well, and a harness variable must not
         redirect them.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None`` to consult only the
            config *path* (callers that have not loaded one).
    """
    configured = dotted_config_str(config, "project_root")
    candidate = Path(configured).expanduser() if configured else None
    if candidate is not None and candidate.is_dir():
        return candidate

    config_path = resolve_config_path()
    if config_path.exists():
        return repo_root_for_config(config_path)
    cwd = Path.cwd()
    for ancestor in (cwd, *cwd.parents):
        if (ancestor / _PROFILE_FILENAME).is_file():
            return ancestor
    return candidate if candidate is not None else cwd


def resolve_agent_data_root() -> Path:
    """Resolve the agent data root directory from config.

    Uses ``agent_data.base_dir`` from config.yml, anchored on the repo root that
    :func:`resolve_project_root` resolves. Falls back to
    :data:`DEFAULT_AGENT_DATA_BASE_DIR` relative to that root when the config
    declares no base directory.
    """
    config = load_osprey_config()
    resolved = anchored_path(agent_data_base_dir(config), resolve_project_root(config)).resolve()

    import os

    session_id = os.environ.get("OSPREY_SESSION_ID")
    if session_id:
        resolved = resolved / "sessions" / session_id

    logger.debug("Agent data root resolved to %s", resolved)
    return resolved


def resolve_shared_data_root(config: Mapping[str, Any] | None = None) -> Path:
    """Resolve the agent data root WITHOUT session-path isolation.

    Use for stores whose data must be visible to long-lived daemons
    (gallery, ARIEL) that run outside any specific session.  Logical
    session isolation is handled at the index level via entry metadata
    (e.g. ``ArtifactEntry.session_id``).

    Args:
        config: An already-loaded config mapping; ``None`` loads the config.
            One derivation serves both a store's writer and a reader that
            already holds a config.
    """
    if config is None:
        config = load_osprey_config()
    resolved = anchored_path(agent_data_base_dir(config), resolve_project_root(config)).resolve()
    logger.debug("Shared data root resolved to %s", resolved)
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
    from osprey_connectors.config import get_config_builder

    config_builder = get_config_builder()
    project_root = Path(config_builder.get("project_root"))
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path
