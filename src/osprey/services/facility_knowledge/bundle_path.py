"""Single resolution rule for ``facility_knowledge.bundle_path``.

Three consumers read that one config key — the ``osprey_facility_knowledge``
MCP server, the ``osprey knowledge`` CLI, and the OKF knowledge panel — and
they must land on the same directory or a valid relative value silently means
different bundles in different places. :func:`resolve_bundle_path` is that one
rule for this key: it delegates to
:func:`osprey.utils.config_paths.resolve_config_relative_path`, the framework's
rule for every config-relative path — expand ``~``, then resolve a
still-relative value against the directory containing ``config.yml``, which is
what the config template and ``how-to/okf-bundle.rst`` promise.
"""

from __future__ import annotations

from pathlib import Path

from osprey.utils.config_paths import resolve_config_relative_path

__all__ = ["resolve_bundle_path"]


def resolve_bundle_path(raw: str | Path, config_dir: Path | None = None) -> Path:
    """Resolve a configured ``facility_knowledge.bundle_path`` to an absolute path.

    ``~`` is expanded first. An absolute value is then returned unchanged; a
    relative one is resolved against *config_dir*.

    Args:
        raw: The configured value, as read from ``facility_knowledge.bundle_path``.
        config_dir: Directory containing ``config.yml``. When omitted it is
            derived from :func:`osprey.utils.workspace.resolve_config_path`,
            which falls back to the process CWD when ``OSPREY_CONFIG`` is unset.

    Returns:
        Absolute :class:`~pathlib.Path` to the bundle root.
    """
    return resolve_config_relative_path(raw, config_dir)
