"""The path rules of a deployment repo: where the render, the repo root and the agent data are.

The rendered ``config.yml``, the repo root it belongs to, and the agent-data
root under it, spelled once here for the connectors and for every framework
layer that reads the same config. A loaded config answers them for itself
(:attr:`osprey_connectors.config.LoadedConfig.repo_root` and the paths beside
it); :func:`locate_repo_root` answers for a process with no config. The rest of
the deployment repo's layout (layer directories, the image staging area, the
audit and archive zones) is the framework's, in ``osprey.utils.workspace``.
"""

from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "BUILD_DIR_NAME",
    "DEFAULT_AGENT_DATA_BASE_DIR",
    "RENDERED_CONFIG_RELPATH",
    "SIMULATION_STATE_DIR_CONFIG_KEY",
    "SIMULATOR_LOG_DIR_RELPATH",
    "agent_data_base_dir",
    "anchored_path",
    "locate_repo_root",
    "rendered_config_path",
    "repo_root_for_config",
    "resolve_simulation_state_dir",
]

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
#: :func:`locate_repo_root` walks up to. Private: the framework owns it
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
#: root. :func:`osprey_connectors.config.load_config`'s runtime-write
#: advisory and :func:`resolve_simulation_state_dir` read this one spelling.
SIMULATION_STATE_DIR_CONFIG_KEY = "simulation.state_dir"


def _configured_str(config: Mapping[str, object] | None, key: str) -> str | None:
    """The dotted *key* of *config* when it names a non-empty string, else ``None``.

    Every way the key can fail to answer (a missing segment, a segment that is
    not a mapping, a non-string or empty value) means the same to the path
    readers: the key is unset, use the default.
    """
    value: object = config or {}
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, str) and value else None


def anchored_path(value: str, project_root: Path) -> Path:
    """Expand ``~`` in a configured path and anchor a relative one at the project.

    A configured path is absolute, home-relative, or project-relative, and every
    consumer has to make the same three-way choice before comparing paths or
    creating directories (:meth:`osprey_connectors.config.LoadedConfig.repo_path`
    makes it for the loaded config).
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def rendered_config_path(repo_root: Path | str) -> Path:
    """Path to the rendered ``config.yml`` of the repo rooted at *repo_root*.

    The one place the ``build/config.yml`` spelling is assembled, so that a
    caller holding a repo root never has to know which zone the render lands in.
    """
    return Path(repo_root) / RENDERED_CONFIG_RELPATH


def agent_data_base_dir(config: Mapping[str, object] | None) -> str:
    """Read the agent-data root out of an already-loaded config mapping.

    ``agent_data.base_dir`` is the only key that names this directory. Callers
    holding a config dict read it through here and anchor the result themselves
    (the health checks and the compose generator anchor on ``project_root``);
    a loaded config answers it anchored
    (:attr:`osprey_connectors.config.LoadedConfig.agent_data_root`).

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The configured base directory, possibly relative to the caller's anchor.
    """
    section = (config or {}).get("agent_data") or {}
    if not isinstance(section, Mapping):
        return DEFAULT_AGENT_DATA_BASE_DIR
    return str(section.get("base_dir") or DEFAULT_AGENT_DATA_BASE_DIR)


def resolve_simulation_state_dir(config: Mapping[str, object] | None, project_root: Path) -> Path:
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
    :func:`osprey_connectors.config.load_config`'s runtime-write advisory
    reads the same key.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.
        project_root: Root the relative paths above resolve against.

    Returns:
        Absolute path to the state directory (not created here).
    """
    config = config or {}
    configured = _configured_str(config, SIMULATION_STATE_DIR_CONFIG_KEY)
    relative = configured or f"{agent_data_base_dir(config)}/{_SIMULATION_STATE_DIR_NAME}"
    return anchored_path(relative, project_root)


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


def locate_repo_root(cwd: Path | None = None) -> Path:
    """The deployment repo a process stands in when it has no loaded config.

    The ladder :attr:`osprey_connectors.config.LoadedConfig.repo_root` cannot
    climb, because there is no config to read a ``project_root`` from:
      1. The repo of the file :func:`osprey_connectors.config.locate_config`
         names, when one is there (a file that exists but does not load still
         says which repo the process belongs to).
      2. The nearest ancestor of *cwd* holding ``profile.yml``, the marker every
         OSPREY verb walks up to; without it a process started in an unbuilt
         render would anchor every relative path inside ``build/``.
      3. *cwd* itself. ``CLAUDE_PROJECT_DIR`` is not an input: this serves
         processes outside any agent session as well, and a harness variable
         must not redirect them.

    Args:
        cwd: The directory to start from; the working directory by default.
    """
    from osprey_connectors.config import locate_config

    here = Path.cwd() if cwd is None else Path(cwd)
    located = locate_config(cwd=here)
    if located.exists():
        return repo_root_for_config(located)
    for ancestor in (here, *here.parents):
        if (ancestor / _PROFILE_FILENAME).is_file():
            return ancestor
    return here
