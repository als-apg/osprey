"""Claude Code project-path encoding and state-directory resolution.

Derives the ``<config-dir>/projects/<encoded>/`` directory Claude Code uses for
a project's transcripts, memory, and session state. Consumed by both the Web
Terminal (``interfaces.web_terminal``) and the workspace MCP server
(``mcp_server.workspace``), so it lives in ``agent_runner`` — below both — to
avoid an interface/CLI import from the low layers.

``<config-dir>`` is ``$CLAUDE_CONFIG_DIR`` when set and ``~/.claude`` otherwise,
exactly as the CLI resolves it. The distinction matters in the per-user
web-terminal container, which sets ``CLAUDE_CONFIG_DIR`` *and* ``HOME`` to the
same mounted volume: a reader that spells ``~/.claude/projects`` there looks one
``.claude`` too deep, finds nothing, and reports every session as missing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Claude Code derives the per-project transcript/memory directory name by
# replacing every non-alphanumeric character of the absolute cwd with '-'.
# Every special character counts, not just '/': a path containing '_' that
# normalizes only its slashes names a different directory than Claude writes to,
# producing silent false-negative reads of transcripts, memory, and sessions.
_CLAUDE_PROJECT_DIR_NORMALIZE = re.compile(r"[^A-Za-z0-9-]")

#: The variable Claude Code reads for its state root (settings, projects,
#: skills). The web-terminal compose service sets it to the per-user volume.
CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"


def encode_claude_project_path(project_dir: Path | str) -> str:
    """Return the Claude Code project-directory name for *project_dir*.

    Claude Code stores per-project state in
    ``<config-dir>/projects/<encoded>/`` where ``<encoded>`` is the absolute
    cwd with every non-alphanumeric character normalized to ``-``. This
    helper exists so every callsite computes the same name — keep this in
    sync with the bundled CLI if its encoding ever changes.
    """
    abs_path = str(Path(project_dir).resolve())
    return _CLAUDE_PROJECT_DIR_NORMALIZE.sub("-", abs_path)


def claude_config_dir() -> Path:
    """Return Claude Code's state root: ``$CLAUDE_CONFIG_DIR`` or ``~/.claude``.

    The variable is taken verbatim (after ``~`` expansion) — Claude Code does
    not append ``.claude`` to it — and a blank value counts as unset.
    """
    configured = os.environ.get(CLAUDE_CONFIG_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".claude"


def claude_project_dir(project_dir: Path | str) -> Path:
    """Return the directory Claude Code keeps *project_dir*'s state in.

    This is ``<config-dir>/projects/<encoded>/``: the session transcripts
    (``<uuid>.jsonl``), the ``memory/`` directory and per-session subagent
    files all live here.
    """
    return claude_config_dir() / "projects" / encode_claude_project_path(project_dir)
