"""Session-tier plan files, written where the bridge resolves its writable plan directory."""

from __future__ import annotations

from pathlib import Path

from osprey.services.bluesky_bridge.session_dir import resolve_session_plan_dir


def session_plan_source(name: str, *, body: str = "") -> str:
    """A minimal, bluesky-free session-tier plan file satisfying the load contract."""
    return (
        "PLAN_METADATA = {\n"
        f'    "name": {name!r},\n'
        '    "description": "A session-tier test plan.",\n'
        '    "writes": False,\n'
        "}\n\n\n"
        "def build_plan(devices, params):\n"
        f'    return {{"plan": {name!r}}}\n'
        f"{body}"
    )


def write_session_plan(name: str, source: str) -> Path:
    """Write ``source`` as ``<name>.py`` in this test's session-plan directory."""
    path = resolve_session_plan_dir() / f"{name}.py"
    path.write_text(source)
    return path
