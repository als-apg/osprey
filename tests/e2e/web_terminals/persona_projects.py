"""Two deployment repos that differ by one permission entry.

Persona A's render denies one tool that persona B's allows by default. The
live and the scripted persona tests both build on this pair.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tests.e2e.sdk_helpers import render_dir

#: The swing tool: allowed by default (the ``osprey_workspace`` server's
#: ``permissions_allow``), explicitly denied in persona A's ``config.yml``.
DENIED_TOOL_ENTRY = "mcp__osprey_workspace__facility_description"
DENIED_TOOL_SHORT_NAME = "facility_description"


def deny_swing_tool(repo: Path) -> None:
    """Add the swing tool to the render's deny list and re-render from it.

    ``claude_code.permissions`` is baked into ``.claude/settings.json`` when the
    Claude Code artifacts are rendered, not read live, so flipping the config
    alone leaves ``settings.json`` stale. ``regen_if_drift`` re-renders them
    from the edited config — the same step ``osprey build`` performs.

    The edit lands in the render (``<repo>/build/config.yml``), which is what
    the agent's ``.claude/`` tree is generated from. Nothing here rebuilds the
    repo from ``profile.yml``: a full rebuild would also re-render the ARIEL
    DSN override ``init_project`` applies to the same file.
    """
    from osprey.cli.templates.manager import TemplateManager

    render = render_dir(repo)
    config_path = render / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    permissions = config.setdefault("claude_code", {}).setdefault("permissions", {})
    permissions.setdefault("deny", []).append(DENIED_TOOL_ENTRY)
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    regenerated = TemplateManager().regen_if_drift(render)
    assert regenerated, (
        f"no Claude Code artifact was re-rendered after adding {DENIED_TOOL_ENTRY} to "
        f"{config_path} — settings.json would still carry the pre-edit permissions, and "
        "the denial under test would never reach the CLI's permission engine"
    )
