"""Claude Memory Service — reads/writes Claude Code native memory files.

Provides CRUD operations on ``~/.claude/projects/<encoded>/memory/*.md``
files. Stateless service class instantiated per-request with a project
directory, following the same pattern as :class:`PromptGalleryService`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from osprey.cli.project_utils import encode_claude_project_path

logger = logging.getLogger(__name__)

# Maximum line count at which MEMORY.md gets truncated in Claude's context
MEMORY_TRUNCATION_LIMIT = 200


class MemoryFileNotFoundError(Exception):
    """Raised when the requested memory file does not exist."""


class MemoryFileExistsError(Exception):
    """Raised when creating a file that already exists."""


class MemoryValidationError(Exception):
    """Raised for invalid filenames or content."""


# Filename validation: alphanumeric, hyphens, underscores, dots (must end .md)
_VALID_FILENAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*\.md$")


class ClaudeMemoryService:
    """Service for the Memory Gallery web UI.

    Instantiated per-request with the project directory. All methods are
    synchronous since the web terminal runs them in a thread pool.
    """

    def __init__(self, project_dir: str | Path) -> None:
        self._project_dir = Path(project_dir).resolve()

    def _resolve_memory_dir(self) -> Path:
        """Return the Claude Code memory directory for this project.

        Claude Code stores memory files in
        ``~/.claude/projects/<encoded>/memory/``;
        see :func:`osprey.cli.project_utils.encode_claude_project_path`
        for the encoding rule.
        """
        encoded = encode_claude_project_path(self._project_dir)
        return Path.home() / ".claude" / "projects" / encoded / "memory"

    # ── List ──────────────────────────────────────────────────────────

    def list_files(self) -> list[dict]:
        """Return metadata for all *.md files in the memory directory."""
        memory_dir = self._resolve_memory_dir()
        if not memory_dir.is_dir():
            return []

        files = []
        for path in sorted(memory_dir.glob("*.md")):
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            files.append(
                {
                    "filename": path.name,
                    "line_count": line_count,
                    "size": path.stat().st_size,
                    "is_primary": path.name == "MEMORY.md",
                }
            )
        return files

    # ── Read ──────────────────────────────────────────────────────────

    def read_file(self, filename: str) -> dict:
        """Read a single memory file by filename."""
        self._validate_filename(filename)
        path = self._resolve_memory_dir() / filename
        if not path.is_file():
            raise MemoryFileNotFoundError(f"Memory file not found: {filename}")
        content = path.read_text(encoding="utf-8")
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return {
            "filename": filename,
            "content": content,
            "line_count": line_count,
            "size": path.stat().st_size,
            "is_primary": filename == "MEMORY.md",
        }

    # ── Create ────────────────────────────────────────────────────────

    def create_file(self, filename: str, content: str = "") -> dict:
        """Create a new memory file."""
        self._validate_filename(filename)
        memory_dir = self._resolve_memory_dir()
        path = memory_dir / filename
        if path.exists():
            raise MemoryFileExistsError(f"File already exists: {filename}")
        memory_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return {
            "filename": filename,
            "line_count": line_count,
            "size": path.stat().st_size,
            "is_primary": filename == "MEMORY.md",
        }

    # ── Update ────────────────────────────────────────────────────────

    def update_file(self, filename: str, content: str) -> dict:
        """Update an existing memory file."""
        self._validate_filename(filename)
        path = self._resolve_memory_dir() / filename
        if not path.is_file():
            raise MemoryFileNotFoundError(f"Memory file not found: {filename}")
        path.write_text(content, encoding="utf-8")
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return {
            "filename": filename,
            "line_count": line_count,
            "size": path.stat().st_size,
            "is_primary": filename == "MEMORY.md",
        }

    # ── Delete ────────────────────────────────────────────────────────

    def delete_file(self, filename: str) -> dict:
        """Delete a memory file."""
        self._validate_filename(filename)
        path = self._resolve_memory_dir() / filename
        if not path.is_file():
            raise MemoryFileNotFoundError(f"Memory file not found: {filename}")
        path.unlink()
        return {"filename": filename, "deleted": True}

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate_filename(filename: str) -> None:
        """Validate that a filename is safe and ends with .md."""
        if not filename or not _VALID_FILENAME_RE.match(filename):
            raise MemoryValidationError(
                f"Invalid filename: '{filename}'. "
                "Must be alphanumeric with hyphens/underscores, ending in .md"
            )
        # Reject path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise MemoryValidationError(f"Path traversal not allowed: '{filename}'")
