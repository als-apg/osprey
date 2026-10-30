"""Session discovery for Claude Code JSONL conversation files.

Scans ``<config-dir>/projects/<encoded-path>/`` for JSONL session files. The
session picker reads the agent record, not these files; what remains here is
the set of transcript ids on disk.
"""

from __future__ import annotations

from pathlib import Path

from osprey.agent_runner.project_paths import claude_project_dir


class SessionDiscovery:
    """List the Claude Code session files on disk for one project."""

    def __init__(self, project_dir: str | Path) -> None:
        self._project_dir = Path(project_dir).resolve()

    def _resolve_sessions_dir(self) -> Path:
        """Return the Claude projects directory for this project.

        Claude Code stores sessions in ``<config-dir>/projects/<encoded>/``;
        see :func:`osprey.agent_runner.project_paths.claude_project_dir` for
        how the root and the encoded name are resolved.
        """
        return claude_project_dir(self._project_dir)

    def snapshot_session_ids(self) -> set[str]:
        """Return the current set of JSONL filenames (stems)."""
        sessions_dir = self._resolve_sessions_dir()
        if not sessions_dir.is_dir():
            return set()
        return {p.stem for p in sessions_dir.glob("*.jsonl")}
