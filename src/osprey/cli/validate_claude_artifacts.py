"""Validate rendered Claude Code artifacts after ``osprey build``.

Catches two classes of drift between profile inputs and rendered ``.claude/`` output:

1. **Wildcard tools in agent frontmatter** — every agent must list its MCP tools
   explicitly so the lockdown is auditable. ``mcp__<server>__*`` is rejected.
2. **Unbacked tool declarations** — every ``mcp__`` tool entry in an agent's
   ``tools:`` allowlist must be *backed*: present in the project's
   ``.claude/settings.json`` ``permissions.allow`` **or** ``permissions.ask``.
   Approval-gated tools (e.g. ``mcp__python__execute``) render into
   ``permissions.ask`` by design — they are available to the agent, just
   subject to an approval prompt. An exact-literal ``permissions.deny`` entry
   removes a tool from the backed set (deny wins at runtime). A tool in neither
   allow nor ask — or one explicitly denied — is a real drift: the agent thinks
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
        ``.claude/settings.json`` ``permissions.allow`` **or** ``permissions.ask``
        (the latter covers approval-gated tools, which are backed but prompt on
        use), and must not appear as an exact-literal ``permissions.deny`` entry
        (deny wins at runtime). A tool that is unbacked or explicitly denied is
        reported.
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

    permissions = settings.get("permissions", {})
    # A tool is "backed" if it renders into either allow (auto-approved) or ask
    # (approval-gated). Approval-gated tools like mcp__python__execute are still
    # available to the agent — they just prompt on use — so both lists count.
    backed_set: set[str] = {
        str(entry)
        for key in ("allow", "ask")
        for entry in permissions.get(key, [])
        if isinstance(entry, str)
    }
    # deny wins at runtime, so a tool explicitly denied is not actually backed —
    # remove exact-literal deny entries. Matching is string equality only; we do
    # not expand wildcard deny patterns (e.g. ``mcp__plugin_playwright_*``), which
    # never coincide with the explicit literals agents are required to declare.
    deny_set: set[str] = {
        str(entry) for entry in permissions.get("deny", []) if isinstance(entry, str)
    }
    backed_set -= deny_set

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
            if entry not in backed_set:
                errors.append(
                    f"agent {agent_name}: tool '{entry}' not present in "
                    ".claude/settings.json permissions.allow or permissions.ask"
                )

    return errors
