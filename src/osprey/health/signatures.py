"""Disk-change signatures for the health surfaces' config-freshness probes.

A "signature" is a cheap ``(mtime_ns, size)`` fingerprint of a file. Comparing a
file's signature across cycles detects a content edit — or the file's appearance
or disappearance — without reading or parsing it. The long-lived health surfaces
use this to decide, on every poll, whether ``config.yml`` / ``.env`` changed on
disk since the last cycle:

* the loader (:mod:`osprey.health.loader`) gates its per-file reload on it;
* the web engine's liveness breaker and the MCP context's wedge breaker each pair
  the two files' signatures into a single change probe.

Single-sourcing the definition here keeps "config changed on disk" meaning the
same thing across all three, so those checks cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def stat_signature(path: Path) -> tuple[int, int] | None:
    """Return a change-detecting ``(mtime_ns, size)`` signature, or ``None``.

    ``None`` means the file is absent. Comparing signatures across cycles
    detects a content edit (mtime and/or size change) as well as the appearance
    or disappearance of the file; pairing size with mtime catches an edit that a
    coarse filesystem clock would leave at the same timestamp.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def disk_signature(config_path: str | Path | None) -> tuple[Any, ...]:
    """Stat ``config.yml`` and the deployment's env chain as one change probe.

    The chain files are the ones at the REPO ROOT, not siblings of the config:
    the config is a render under ``build/``, which no build writes an ``.env``
    into. Every chain member is statted — ``.env.shared`` included — because an
    edit to the shared defaults changes the environment the checks answer from
    exactly the way an edit to ``.env`` does.

    Resolves *config_path* (or the CLI default via
    :func:`osprey.utils.workspace.resolve_config_path` when ``None``) and
    returns the per-file :func:`stat_signature` values the breaker/validity
    checks compare across cycles as one opaque tuple. A changed tuple forces a
    refresh regardless of age.
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        from osprey.utils.workspace import resolve_config_path

        path = resolve_config_path()
    # Same rule as `loader.HealthConfigLoader.load` — deliberately, and
    # through the same helper. A signature that stats different files
    # than the loader watches is a cache that never invalidates.
    from osprey.utils.workspace import deployment_env_chain

    return (
        stat_signature(path),
        *(stat_signature(env_path) for env_path in deployment_env_chain(path)),
    )
