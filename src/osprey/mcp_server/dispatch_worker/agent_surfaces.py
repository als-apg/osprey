"""Declared-subagent tool surfaces for dispatch permission policy.

Parses the provisioned ``<project>/.claude/agents/*.md`` files (rendered from
``src/osprey/templates/claude_code/claude/agents/*.md.j2``) into a mapping of
agent name to declared tool surface. The dispatch worker uses this to grant
each subagent exactly its declared tools — parity with the web terminal —
without the trigger having to enumerate them.

Frontmatter contract (matching what the Claude Code CLI loads):

* Only files directly in ``.claude/agents/`` (non-recursive; the templates
  ship a ``_terminology`` sibling directory that must not be scanned).
* A file must start with ``---``; the block up to the closing ``---`` is
  parsed with ``yaml.safe_load``.
* Agents are keyed by the frontmatter ``name:`` (what the CLI dispatches on),
  not the filename. Duplicate names: last file in sorted order wins, with a
  warning.
* ``tools:`` may be a comma-separated scalar (the template form) or a YAML
  list. An absent or malformed ``tools:`` yields ``None`` — the agent exists
  but has inherits-all semantics, which dispatch treats as non-delegable
  rather than granting an unbounded surface.

Parsing is best-effort and never raises: a malformed or unreadable file is
skipped with a warning so one bad agent file cannot take down dispatch runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("osprey.mcp_server.dispatch_worker.agent_surfaces")


def _parse_tools(value: object) -> frozenset[str] | None:
    """Normalize a frontmatter ``tools`` value to a tool-name set.

    Accepts the template's comma-separated scalar or a YAML list; anything
    else (including absent → ``None``) yields ``None`` (non-delegable).
    """
    if isinstance(value, str):
        return frozenset(t.strip() for t in value.split(",") if t.strip())
    if isinstance(value, list):
        return frozenset(str(t).strip() for t in value if str(t).strip())
    return None


def parse_project_agents(project_dir: str | Path) -> dict[str, frozenset[str] | None]:
    """Parse declared subagents and their tool surfaces from a project.

    Args:
        project_dir: Project root containing ``.claude/agents/``.

    Returns:
        Mapping of frontmatter agent name to its declared tool set, or
        ``None`` for agents without an explicit ``tools:`` list. Missing
        directory or no parseable files ⇒ empty mapping.
    """
    agents_dir = Path(project_dir) / ".claude" / "agents"
    surfaces: dict[str, frozenset[str] | None] = {}
    if not agents_dir.is_dir():
        return surfaces

    for path in sorted(agents_dir.glob("*.md")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Skipping unreadable agent file %s", path, exc_info=True)
            continue

        if not text.startswith("---"):
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue

        try:
            frontmatter = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            logger.warning("Skipping agent file with malformed frontmatter: %s", path.name)
            continue
        if not isinstance(frontmatter, dict):
            continue

        name = frontmatter.get("name")
        if not isinstance(name, str) or not name.strip():
            logger.warning("Skipping agent file without a name: %s", path.name)
            continue
        name = name.strip()

        tools = _parse_tools(frontmatter.get("tools"))
        if tools is None:
            logger.warning(
                "Agent %r (%s) declares no explicit tools list — "
                "it will not be delegable in dispatch runs",
                name,
                path.name,
            )
        if name in surfaces:
            logger.warning("Duplicate agent name %r — %s overrides earlier file", name, path.name)
        surfaces[name] = tools

    return surfaces
