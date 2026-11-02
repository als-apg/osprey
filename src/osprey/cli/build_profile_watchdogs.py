"""What a build carries out of the ``watchdogs:`` block: its enabled watchdogs only.

The rendered ``config.yml`` holds the same name-keyed map the author wrote, cut to
the enabled watchdogs and each saying ``enabled: true``; with none enabled the key
is absent. Their files follow them to ``<build>/watchdogs/<name>/``, so a disabled
watchdog leaves no trace in a render.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.services.watchdog.config import WATCHDOGS_BLOCK


def enabled_watchdogs(block: Any) -> dict[str, dict[str, Any]]:
    """The enabled entries of a ``watchdogs:`` map, in the author's order."""
    if not isinstance(block, Mapping):
        return {}
    return {
        str(name): dict(entry)
        for name, entry in block.items()
        if isinstance(entry, Mapping) and entry.get("enabled") is True
    }


def watchdog_config_overrides(block: Any) -> dict[str, Any]:
    """The ``watchdogs`` key of the rendered ``config.yml``; nothing when none is enabled."""
    enabled = enabled_watchdogs(block)
    return {WATCHDOGS_BLOCK: enabled} if enabled else {}


def copy_enabled_watchdogs(block: Any, profile_dir: Path, render_dir: Path) -> list[str]:
    """Copy each enabled watchdog's directory from the profile into the render.

    Args:
        block: The profile's ``watchdogs:`` map, already validated, which is what
            holds each enabled watchdog's directory to existing.
        profile_dir: The directory holding ``watchdogs/<name>/``.
        render_dir: The render, which gets ``watchdogs/<name>/``.

    Returns:
        The names copied.
    """
    target = render_dir / WATCHDOGS_BLOCK
    if target.exists():
        shutil.rmtree(target)
    names = list(enabled_watchdogs(block))
    for name in names:
        shutil.copytree(profile_dir / WATCHDOGS_BLOCK / name, target / name)
    return names
