"""What a project's subagents declare about themselves.

The rendered ``<project>/.claude/agents/*.md`` files (from
``src/osprey/templates/claude_code/claude/agents/*.md.j2``) are the one place
an agent says what it is: its ``name``, the ``tools`` it may use, and — for an
agent that computes something — the ``results_category`` it owes a data
artifact to. Two runtime readers consume that frontmatter: the dispatch worker
(tool surfaces, :mod:`osprey.mcp_server.dispatch_worker.agent_surfaces`) and
``submit_response`` (the results contract). They share this parser so they
cannot disagree about which files are agents or what an agent is called.

Frontmatter contract (matching what the Claude Code CLI loads):

* Only files directly in ``.claude/agents/`` (non-recursive; the templates
  ship ``_terminology`` and ``_shared`` partial directories alongside the
  agents that must not be scanned).
* A file must start with ``---``; the block up to the closing ``---`` is
  parsed with ``yaml.safe_load``.
* Agents are keyed by the frontmatter ``name:`` (what the CLI dispatches on),
  not the filename. Duplicate names: last file in sorted order wins, with a
  warning.

Parsing is best-effort and never raises: a malformed or unreadable file is
skipped with a warning so one bad agent file cannot take down a run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("osprey.mcp_server.agent_frontmatter")

#: Frontmatter key naming the artifact category an agent owes its computed
#: results to. ``submit_response`` files the agent's ``data`` there and refuses
#: a hand-in that carries none.
RESULTS_CATEGORY_KEY = "results_category"


def parse_agent_frontmatter(project_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Parse every declared subagent's frontmatter from a project.

    Args:
        project_dir: Project root containing ``.claude/agents/``.

    Returns:
        Mapping of frontmatter agent name to its frontmatter dict. Missing
        directory or no parseable files ⇒ empty mapping.
    """
    agents_dir = Path(project_dir) / ".claude" / "agents"
    declared: dict[str, dict[str, Any]] = {}
    if not agents_dir.is_dir():
        return declared

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

        if name in declared:
            logger.warning("Duplicate agent name %r — %s overrides earlier file", name, path.name)
        declared[name] = frontmatter

    return declared


def results_category_for(agent_name: str, project_dir: str | Path) -> str | None:
    """The artifact category *agent_name* declares it owes its results to.

    ``None`` for an agent that declares nothing, for a blank or non-string
    declaration, and for a name no agent file carries — all of which mean the
    same thing to the caller: this agent hands in prose only.
    """
    frontmatter = parse_agent_frontmatter(project_dir).get(agent_name)
    if not frontmatter:
        return None
    category = frontmatter.get(RESULTS_CATEGORY_KEY)
    if not isinstance(category, str) or not category.strip():
        return None
    return category.strip()
