"""One rule for resolving a path that a config file names relatively.

Several config keys point at a directory or file next to ``config.yml`` —
``facility_knowledge.bundle_path``, the qmd export mirror, the ARIEL vocabulary
file. Each is read by more than one process (an MCP server, a CLI command, a
web app, a container), and if any of them resolves the value against its own
working directory instead of the config's directory, the same configured string
silently means a different place in each process. That divergence is exactly
what this module exists to prevent: :func:`resolve_config_relative_path` is the
single rule, and every reader of such a key delegates to it.

Note that :func:`osprey.cli.project_utils.resolve_config_path` is a *different*
function that happens to share a name with the workspace helper used below; the
one consulted here is :func:`osprey.utils.workspace.resolve_config_path`, which
answers "which ``config.yml`` is this process running against".
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_config_dir", "resolve_config_relative_path"]


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


def resolve_config_relative_path(value: str | Path, config_dir: Path | None = None) -> Path:
    """Resolve a configured path value to an absolute path.

    ``~`` is expanded first. An absolute value is then returned unchanged — it
    already names one place, and re-resolving it would silently follow symlinks
    the operator wrote deliberately. A relative value is resolved against
    *config_dir*.

    Args:
        value: The configured value, as read from the config file.
        config_dir: Directory containing ``config.yml``. Callers that loaded a
            config themselves should pass its parent. When omitted it is derived
            from :func:`osprey.utils.workspace.resolve_config_path`, which falls
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
    return (config_dir / path).resolve()
