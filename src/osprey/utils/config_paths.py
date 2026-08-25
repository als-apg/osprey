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
empty tree — and a persona's readers opened the image's baked bundle at
``build/data/...`` while the deployment's live bundle was mounted at
``data/...`` one directory over. Same rule as
:func:`osprey_connectors.workspace.resolve_project_root` and every other
relative path the runtime resolves.

One key is the deliberate exception, through :func:`resolve_render_relative_path`:
``services.graphdb.ttl_path``. The corpus it names is an artifact OF the
render: its documented default, ``./data/demo_machine.ttl``, is read from the
``data/`` tree the build assembled for exactly this project — the app
template's data, or the profile's own ``data:`` tree in its place — so a corpus regenerated
into the profile's data tree reaches the store on the next build, like every
other rendered artifact. It therefore resolves against the ``config.yml``
directory. The store is seeded from it once, by the deploy; nothing writes it
back and nothing mounts it, which is why the argument above does not apply.

Note that :func:`osprey.cli.project_utils.resolve_config_path` is a *different*
function that happens to share a name with the workspace helper used below; the
one consulted here is :func:`osprey.utils.workspace.resolve_config_path`, which
answers "which ``config.yml`` is this process running against".
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "resolve_config_dir",
    "resolve_config_relative_path",
    "resolve_render_relative_path",
    "project_root_for_config_dir",
]


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
        from osprey.utils.workspace import resolve_config_path

        return Path(resolve_config_path()).parent
    except Exception:  # noqa: BLE001
        return None


def project_root_for_config_dir(config_dir: Path) -> Path:
    """The project root that a ``config.yml`` in *config_dir* belongs to.

    The one place the "is this the build zone?" question is asked for
    config-relative paths, delegating to the workspace helper that answers it
    for every other runtime path (:func:`osprey.utils.workspace.repo_root_for_config`)
    so the two cannot disagree: the parent when *config_dir* is the render
    zone, *config_dir* itself otherwise.

    Args:
        config_dir: Directory containing ``config.yml``.

    Returns:
        The project root — ``<repo>`` for a host render at ``<repo>/build``, the
        directory itself for a container project or a flat legacy one.
    """
    from osprey.utils.workspace import repo_root_for_config

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
            :func:`osprey.utils.workspace.resolve_config_path`, which falls
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
            derived from :func:`osprey.utils.workspace.resolve_config_path`,
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
