"""Declared-subagent tool surfaces for dispatch permission policy.

Reads the provisioned ``<project>/.claude/agents/*.md`` files into a mapping
of agent name to declared tool surface. The dispatch worker uses this to grant
each subagent exactly its declared tools — parity with the web terminal —
without the trigger having to enumerate them.

Which files are agents, and what they are called, is decided by
:mod:`osprey.mcp_server.agent_frontmatter` (shared with ``submit_response``);
this module only interprets the ``tools:`` key:

* ``tools:`` may be a comma-separated scalar (the template form) or a YAML
  list. An absent or malformed ``tools:`` yields ``None`` — the agent exists
  but has inherits-all semantics, which dispatch treats as non-delegable
  rather than granting an unbounded surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

from osprey.mcp_server.agent_frontmatter import parse_agent_frontmatter

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
    surfaces: dict[str, frozenset[str] | None] = {}
    for name, frontmatter in parse_agent_frontmatter(project_dir).items():
        tools = _parse_tools(frontmatter.get("tools"))
        if tools is None:
            logger.warning(
                "Agent %r declares no explicit tools list — "
                "it will not be delegable in dispatch runs",
                name,
            )
        surfaces[name] = tools
    return surfaces
