"""A synthetic simulator view's scenario state."""

from __future__ import annotations

import os
from pathlib import Path


def activate(state: Path, *names: str) -> None:
    """Write ``names`` (``nominal`` when none) as the active set under ``state``.

    The file's mtime is moved a full second past its last one, so a reader
    that watches the mtime sees the change however soon it follows the last.
    """
    state.mkdir(parents=True, exist_ok=True)
    target = state / "active_scenarios"
    before = target.stat().st_mtime_ns if target.exists() else 0
    target.write_text("\n".join(names or ("nominal",)) + "\n", encoding="utf-8")
    os.utime(target, ns=(before + 1_000_000_000, before + 1_000_000_000))
