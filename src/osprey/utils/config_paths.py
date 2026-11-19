"""One rule for resolving a path that a config file names relatively.

Several config keys point at a directory or file inside the project —
``facility_knowledge.bundle_path``, the qmd export mirror, the ARIEL vocabulary
file. Each is read by more than one process (an MCP server, a CLI command, a
web app, a container), and if any of them resolves the value against its own
working directory instead of the project's, the same configured string
silently means a different place in each process. That divergence is exactly
what this module exists to prevent: :func:`resolve_config_relative_path` is the
single rule, and every reader of such a key delegates to it.

The anchor is the **project root** — the deployment repo, the directory holding
``profile.yml`` — and not the directory holding ``config.yml``. The two differ
on a host, where the render lives one zone down in ``build/`` and is re-created
from scratch by every ``osprey build``; they coincide in a container, whose
project directory *is* its own render, and in a flat legacy project. The repo
root is the anchor for one reason beyond durability: it is what every compose
bind source resolves against (every invocation is pinned with
``--project-directory <repo>``, see
:func:`osprey.deployment.compose_generator.compose_base_cmd`), so it is the
only anchor under which the directory a process WRITES is the directory a
container is handed to read. Anchored on the render, the qmd exporter wrote
``build/var/ariel_mirror`` while the sidecar indexed ``var/ariel_mirror`` — an
empty tree — and a render's readers opened the image's baked bundle at
``build/data/...`` while the deployment's live bundle was mounted at
``data/...`` one directory over. Same rule as
:func:`osprey.utils.config_paths.process_repo_root` and every other
relative path the runtime resolves.

One key is the deliberate exception, through :func:`resolve_render_relative_path`:
``services.graphdb.ttl_path``. The corpus it names is an artifact OF the
render: its documented default, ``./data/graph/facility.ttl``, is the graph
view the build writes into the render from the project's ``data/facility/``
sources, so a facility edit reaches the store on the next build, like every
other rendered artifact. It therefore resolves against the ``config.yml``
directory. The store is seeded from it once, by the deploy; nothing writes it
back and nothing mounts it, which is why the argument above does not apply.

Note that :func:`osprey.cli.project_utils.resolve_config_path` is a *different*
function that happens to share a name with the workspace helper used below; the
one consulted here is :func:`osprey_connectors.config.locate_config`, which
answers "which ``config.yml`` is this process running against".
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey_connectors.config import LoadedConfig

__all__ = [
    "deployment_config",
    "loaded_or_empty",
    "process_agent_data_root",
    "process_repo_root",
    "repo_root_of",
    "session_agent_data_root",
    "resolve_config_dir",
    "resolve_config_relative_path",
    "resolve_render_relative_path",
    "project_root_for_config_dir",
]


def _process_config() -> LoadedConfig | None:
    """The process's config (:func:`osprey_connectors.config.current_config`), or ``None``.

    ``None`` for every way it fails to answer: these readers degrade rather
    than fail, and each says what it falls back to.
    """
    from osprey_connectors.config import current_config

    try:
        return current_config()
    except Exception:
        return None


# The tree is the YAML parse's: every framework reader reads the shape its own
# keys have.
def loaded_or_empty() -> dict[str, Any]:
    """The process's config tree, as a copy; ``{}`` when it does not load, for any reason.

    For the framework readers that degrade to their own defaults rather than
    fail when there is no config, or a broken one. The tree is
    :func:`osprey_connectors.config.current_config`'s: the config a
    ``using_config`` block or an entry point installed, else the located file.
    Whether each reader should keep degrading is the reader's to decide.
    """
    config = _process_config()
    return config.raw_tree() if config is not None else {}


def process_repo_root() -> Path:
    """The process config's repo root; without a loaded config, the repo the process stands in.

    :attr:`osprey_connectors.config.LoadedConfig.repo_root`, else
    :func:`osprey_connectors.workspace.locate_repo_root`. Never raises.
    """
    from osprey_connectors.workspace import locate_repo_root

    config = _process_config()
    return config.repo_root if config is not None else locate_repo_root()


def repo_root_of(config: Mapping[str, Any]) -> Path:
    """The repo root a config mapping belongs to, for a caller that holds the mapping only.

    Its ``project_root`` when that names a directory here (a container's config
    names the root it runs at), else
    :func:`osprey_connectors.workspace.locate_repo_root`. Never raises. A caller
    holding the process config uses :func:`process_repo_root`.
    """
    from osprey_connectors.workspace import locate_repo_root

    configured = config.get("project_root")
    if isinstance(configured, str) and configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            return candidate
    return locate_repo_root()


def process_agent_data_root() -> Path:
    """The shared agent-data root of the process config; without one, the default under the repo.

    :attr:`osprey_connectors.config.LoadedConfig.agent_data_root`, else
    ``var/agent_data`` under :func:`process_repo_root`, resolved. Never raises
    on a config that does not load.
    """
    from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR, locate_repo_root

    config = _process_config()
    if config is not None:
        return config.agent_data_root
    return (locate_repo_root() / DEFAULT_AGENT_DATA_BASE_DIR).resolve()


def session_agent_data_root() -> Path:
    """The agent-data directory of the session this process serves, else the shared root.

    The session is the one ``OSPREY_SESSION_ID`` names, which every session's
    MCP servers are spawned with
    (:meth:`osprey_connectors.config.LoadedConfig.session_data_root`); a
    process outside any session gets :func:`process_agent_data_root`.
    """
    from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR, locate_repo_root

    session_id = os.environ.get("OSPREY_SESSION_ID")
    config = _process_config()
    if config is not None:
        return config.session_data_root(session_id) if session_id else config.agent_data_root
    shared = (locate_repo_root() / DEFAULT_AGENT_DATA_BASE_DIR).resolve()
    return shared / "sessions" / session_id if session_id else shared


def resolve_config_dir() -> Path | None:
    """Return the directory of the ``config.yml`` this process runs against.

    The lookup that :func:`resolve_config_relative_path` falls back on, exposed
    for the callers that need the directory itself — a CLI command handing
    ``config_dir`` down to a service operation, for instance, so that it and
    the web panel resolve the same configured string to the same file.

    Returns:
        The config file's parent directory, or None when the workspace lookup
        failed. The blind except is deliberate: callers include a launcher, a
        CLI command and an ingestion pipeline, and a workspace lookup that
        fails for any reason must degrade to a CWD-relative path rather than
        take the caller down.
    """
    try:
        from osprey_connectors.config import locate_config

        return Path(locate_config()).parent
    except Exception:
        return None


def project_root_for_config_dir(config_dir: Path) -> Path:
    """The project root that a ``config.yml`` in *config_dir* belongs to.

    The one place the "is this the build zone?" question is asked for
    config-relative paths, delegating to the workspace helper that answers it
    for every other runtime path (:func:`osprey_connectors.workspace.repo_root_for_config`)
    so the two cannot disagree: the parent when *config_dir* is the render
    zone, *config_dir* itself otherwise.

    Args:
        config_dir: Directory containing ``config.yml``.

    Returns:
        The project root — ``<repo>`` for a host render at ``<repo>/build``, the
        directory itself for a container project or a flat legacy one.
    """
    from osprey_connectors.workspace import repo_root_for_config

    return repo_root_for_config(Path(config_dir) / "config.yml")


def resolve_config_relative_path(value: str | Path, config_dir: Path | None = None) -> Path:
    """Resolve a configured path value to an absolute path.

    ``~`` is expanded first. An absolute value is then returned unchanged — it
    already names one place, and re-resolving it would silently follow symlinks
    the operator wrote deliberately. A relative value is resolved against the
    project root of the config in *config_dir* (see the module docstring for
    why the root, not the config's own directory).

    Args:
        value: The configured value, as read from the config file.
        config_dir: Directory containing ``config.yml``. Callers that loaded a
            config themselves should pass its parent; the project root is
            derived from it here (:func:`project_root_for_config_dir`). When
            omitted it is derived from
            :func:`osprey_connectors.config.locate_config`, which falls
            back to the process CWD when ``OSPREY_CONFIG`` is unset.

    Returns:
        Absolute :class:`~pathlib.Path`.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    if config_dir is None:
        config_dir = resolve_config_dir()
    if config_dir is None:
        return path.resolve()
    return (project_root_for_config_dir(config_dir) / path).resolve()


def resolve_render_relative_path(value: str | Path, config_dir: Path | None = None) -> Path:
    """Resolve a configured path that names an artifact of the render.

    The exception the module docstring describes, for ``services.graphdb.ttl_path``:
    ``~`` expanded, an absolute value returned unchanged, a relative value
    resolved against the directory holding ``config.yml`` itself — the render
    — rather than against the project root.

    Args:
        value: The configured value, as read from the config file.
        config_dir: Directory containing ``config.yml``. When omitted it is
            derived from :func:`osprey_connectors.config.locate_config`,
            which falls back to the process CWD when ``OSPREY_CONFIG`` is unset.

    Returns:
        Absolute :class:`~pathlib.Path`.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    if config_dir is None:
        config_dir = resolve_config_dir()
    if config_dir is None:
        return path.resolve()
    return (Path(config_dir) / path).resolve()


@contextmanager
def deployment_config(config_path: str | Path) -> Iterator[None]:
    """Make *config_path* this process's config for the block; put the previous one back after.

    For a verb that acts on one deployment named by its repo (``--repo``,
    ``osprey up`` on another checkout): it stands in neither the deployment's
    container nor its repo root, so every lookup that names no config (the
    facility zone, the limits database, a scenario apply) would otherwise
    answer from the working directory, and degrade to a default without
    failing. A whole archive could be synthesized against the wrong clock.

    Process-wide (:func:`osprey_connectors.config.install_config`), not a
    context-scoped :func:`~osprey_connectors.config.using_config`, because the
    work the block does runs in executor and server threads that a context
    variable does not reach. Deliberately in-process rather than an exported
    ``OSPREY_CONFIG``: the attached start path ``os.execvpe``-replaces itself
    with the container runtime, and an exported variable would ride into that
    process's interpolation environment.

    Never raises: a config that will not load leaves the previous default in
    place and warns, because a verb whose anchor aborted it would turn a
    degraded lookup into a failed deploy.

    Args:
        config_path: The rendered config this process is acting on, normally
            ``<repo>/build/config.yml``.
    """
    from osprey_connectors.config import install_config, load_config
    from osprey_connectors.errors import ConfigLoadError, ConfigurationError

    try:
        loaded = load_config(config_path)
    except (ConfigLoadError, ConfigurationError) as exc:
        logging.getLogger("CONFIG").warning(
            f"Could not anchor configuration at {config_path} "
            f"({type(exc).__name__}: {exc}). Values this process reads without an "
            "explicit path will fall back to their defaults."
        )
        yield
        return
    previous = install_config(loaded)
    try:
        yield
    finally:
        install_config(previous)
