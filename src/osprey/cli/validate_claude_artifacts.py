"""Validate rendered Claude Code artifacts after ``osprey build``.

Catches two classes of drift between profile inputs and rendered ``.claude/`` output:

1. **Wildcard tools in agent frontmatter** — every agent must list its MCP tools
   explicitly so the lockdown is auditable. ``mcp__<server>__*`` is rejected.
2. **Unbacked tool declarations** — every ``mcp__`` tool entry in an agent's
   ``tools:`` allowlist must have a matching entry in the project's
   ``.claude/settings.json`` ``permissions.allow``. Otherwise the agent thinks
   it has a tool the MCP gateway will refuse.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _parse_frontmatter(md_text: str) -> dict[str, Any] | None:
    m = _FRONTMATTER_RE.match(md_text)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _split_tools(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    return [t.strip() for t in str(value).split(",") if t.strip()]


def validate_agent_tools_against_permissions(project_dir: Path) -> list[str]:
    """Return a list of error strings; empty list ⇒ artifacts are coherent.

    Rules:
      - Non-``mcp__`` entries (``Read``, ``Bash``, …) are Claude Code built-ins
        and are ignored.
      - Wildcards in agent ``tools:`` (``mcp__<server>__*``) are rejected —
        list MCP tools explicitly so the lockdown is auditable.
      - Each literal ``mcp__<server>__<tool>`` must appear in
        ``.claude/settings.json`` ``permissions.allow``.
    """
    project_dir = Path(project_dir)
    settings_path = project_dir / ".claude" / "settings.json"
    agents_dir = project_dir / ".claude" / "agents"

    if not settings_path.exists() or not agents_dir.is_dir():
        return []

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{settings_path}: invalid JSON ({e})"]

    allow_set: set[str] = {
        str(entry)
        for entry in settings.get("permissions", {}).get("allow", [])
        if isinstance(entry, str)
    }

    errors: list[str] = []
    for md_file in sorted(agents_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
        if not fm:
            continue
        agent_name = str(fm.get("name") or md_file.stem)
        for entry in _split_tools(fm.get("tools")):
            if not entry.startswith("mcp__"):
                continue
            if "*" in entry:
                errors.append(
                    f"agent {agent_name}: tool '{entry}' uses wildcard; "
                    "list MCP tools explicitly so the lockdown is auditable"
                )
                continue
            if entry not in allow_set:
                errors.append(
                    f"agent {agent_name}: tool '{entry}' not present in "
                    ".claude/settings.json permissions.allow"
                )

    return errors
