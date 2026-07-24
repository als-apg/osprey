"""Claude Code project-path encoding.

Derives the ``~/.claude/projects/<encoded>/`` directory name Claude Code uses
for a project's transcripts, memory, and session state. Consumed by both the
Web Terminal (``interfaces.web_terminal``) and the workspace MCP server
(``mcp_server.workspace``), so it lives in ``agent_runner`` — below both — to
avoid an interface/CLI import from the low layers.
"""

from __future__ import annotations

import re
from pathlib import Path

# Claude Code derives the per-project transcript/memory directory name by
# replacing every non-alphanumeric character of the absolute cwd with '-'.
# Replacing only '/' (the old rule) is wrong for any path containing '_' or
# other special characters — Claude writes to one dir and we look in another,
# producing silent false-negative reads of transcripts, memory, and sessions.
_CLAUDE_PROJECT_DIR_NORMALIZE = re.compile(r"[^A-Za-z0-9-]")


def encode_claude_project_path(project_dir: Path | str) -> str:
    """Return the Claude Code project-directory name for *project_dir*.

    Claude Code stores per-project state in
    ``~/.claude/projects/<encoded>/`` where ``<encoded>`` is the absolute
    cwd with every non-alphanumeric character normalized to ``-``. This
    helper exists so every callsite computes the same name — keep this in
    sync with the bundled CLI if its encoding ever changes.
    """
    abs_path = str(Path(project_dir).resolve())
    return _CLAUDE_PROJECT_DIR_NORMALIZE.sub("-", abs_path)
