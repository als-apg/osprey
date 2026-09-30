"""Session discovery for Claude Code JSONL conversation files.

Scans ``<config-dir>/projects/<encoded-path>/`` for JSONL session files,
extracting metadata (first message, modification time, message count)
for the session picker UI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from osprey.agent_runner.project_paths import claude_project_dir

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Metadata for a single Claude Code session."""

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

    def list_sessions(self) -> list[SessionInfo]:
        """Return sessions sorted newest-first.

        Skips empty JSONL files and any file that cannot be read.
        """
        sessions_dir = self._resolve_sessions_dir()
        if not sessions_dir.is_dir():
            return []

        results: list[SessionInfo] = []
        for path in sessions_dir.glob("*.jsonl"):
            try:
                info = self._parse_session_file(path)
                if info is not None:
                    results.append(info)
            except Exception:
                logger.debug("Skipping corrupt session file: %s", path.name)

        results.sort(key=lambda s: s.last_modified, reverse=True)
        return results

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
                message_count += 1
                if not first_message:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "user":
                            msg = entry.get("message", {})
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                # Multi-part content — extract first text block
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        content = part.get("text", "")
                                        break
                                else:
                                    content = ""
                            if content:
                                first_message = content[:80]
                    except (json.JSONDecodeError, AttributeError):
                        pass

        if message_count == 0:
            return None

        return SessionInfo(
            session_id=path.stem,
            first_message=first_message or "(no user message)",
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            message_count=message_count,
        )
