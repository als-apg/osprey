"""Session discovery for Claude Code JSONL conversation files.

Scans ``<config-dir>/projects/<encoded-path>/`` for JSONL session files. The
session picker reads the agent record, not these files; what remains here is
the set of transcript ids on disk and the single-file metadata parse.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from osprey.agent_runner.project_paths import claude_project_dir

logger = logging.getLogger(__name__)


def _user_preview(entry: dict) -> str:
    """Return the first 80 characters of a user record's text, or ``""``."""
    if entry.get("type") != "user":
        return ""
    message = entry.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, list):
        # Multi-part content — extract first text block
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                content = part.get("text", "")
                break
        else:
            content = ""
    if isinstance(content, str) and content:
        return content[:80]
    return ""


@dataclass
class SessionInfo:
    """Metadata for a single Claude Code session.

    ``message_count`` counts the transcript's readable records, the lines that
    parse as a JSON object.
    """

    session_id: str
    first_message: str
    last_modified: datetime
    message_count: int


class SessionDiscovery:
    """Discover and inspect Claude Code session files on disk."""

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_session_file(path: Path) -> SessionInfo | None:
        """Extract metadata from a single JSONL session file."""
        stat = path.stat()
        if stat.st_size == 0:
            return None

        first_message = ""
        message_count = 0

        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:  # json.JSONDecodeError is a ValueError
                    continue
                if not isinstance(entry, dict):
                    continue
                message_count += 1
                if not first_message:
                    first_message = _user_preview(entry)

        if message_count == 0:
            logger.debug("Skipping session file with no readable record: %s", path.name)
            return None

        return SessionInfo(
            session_id=path.stem,
            first_message=first_message or "(no user message)",
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            message_count=message_count,
        )
