"""Service for managing Claude Code integration files."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from osprey.utils.logger import get_logger

logger = get_logger("claude_code_files")


class ClaudeCodeFileService:
    """Service for discovering, reading, writing, and creating Claude Code files.

    Centralises file system logic and path security checks so that API route
    handlers stay thin.
    """

    ALLOWED_DIRS = {"rules", "agents", "commands", "hooks", "skills"}
    ROOT_FILES = {"CLAUDE.md", ".mcp.json"}

    # Category assignments for well-known files
    _KNOWN_CATEGORIES: dict[str, str] = {
        "CLAUDE.md": "System Prompt",
        "settings.json": "Permissions",
        ".mcp.json": "MCP Servers",
        "safety.md": "Safety",
    }

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_files(self) -> list[dict]:
        """Discover all Claude Code integration files in the project."""
        targets = self._collect_targets()
        files: list[dict] = []

        for fpath, rel_path in targets:
            if not fpath.exists() or not fpath.is_file():
                continue
            try:
                content = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            files.append(
                {
                    "name": fpath.name,
                    "path": rel_path,
                    "category": self.categorize(fpath.name, rel_path),
                    "content": content,
                    "language": self.detect_language(fpath.name),
                }
            )

        return files

    def read_file(self, rel_path: str) -> dict:
        """Read a single file by relative path."""
        resolved = self._validate_path(rel_path)

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")
        if not resolved.is_file():
            raise ValueError(f"Not a file: {rel_path}")

        content = resolved.read_text(encoding="utf-8")
        return {
            "name": resolved.name,
            "path": rel_path,
            "category": self.categorize(resolved.name, rel_path),
            "content": content,
            "language": self.detect_language(resolved.name),
        }

    def write_file(self, rel_path: str, content: str) -> dict:
        """Write content to an existing file with path security + syntax validation."""
        resolved = self._validate_path(rel_path)

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")

        self._validate_content(resolved, content)

        resolved.write_text(content, encoding="utf-8")
        logger.info("Claude Code file updated: %s", rel_path)

        return {
            "status": "saved",
            "name": resolved.name,
            "path": rel_path,
            "category": self.categorize(resolved.name, rel_path),
            "language": self.detect_language(resolved.name),
        }

    def create_file(self, rel_path: str, content: str) -> dict:
        """Create a new file in an allowed .claude/ subdirectory."""
        resolved = self._validate_path(rel_path)

        # Must be inside .claude/<allowed_dir>/
        self._validate_allowed_dir(rel_path)

        if resolved.exists():
            raise FileExistsError(f"File already exists: {rel_path}")

        self._validate_content(resolved, content)

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        logger.info("Claude Code file created: %s", rel_path)

        return {
            "status": "created",
            "name": resolved.name,
            "path": rel_path,
            "category": self.categorize(resolved.name, rel_path),
            "language": self.detect_language(resolved.name),
        }

    # ------------------------------------------------------------------
    # Path security
    # ------------------------------------------------------------------

    def _validate_path(self, rel_path: str) -> Path:
        """Resolve path and check for traversal attacks."""
        resolved = (self.project_dir / rel_path).resolve()

        if not resolved.is_relative_to(self.project_dir):
            raise PermissionError(f"Path traversal blocked: {rel_path}")

        return resolved

    def _validate_allowed_dir(self, rel_path: str) -> None:
        """Ensure the path is inside .claude/<allowed_dir>/."""
        parts = Path(rel_path).parts

        if len(parts) < 3 or parts[0] != ".claude" or parts[1] not in self.ALLOWED_DIRS:
            allowed = ", ".join(sorted(self.ALLOWED_DIRS))
            raise PermissionError(
                f"New files must be in .claude/<dir>/ where <dir> is one of: {allowed}"
            )

    # ------------------------------------------------------------------
    # Content validation
    # ------------------------------------------------------------------

    def _validate_content(self, path: Path, content: str) -> None:
        """Syntax-check JSON and YAML files before writing."""
        suffix = path.suffix.lower()

        if suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}") from e

        elif suffix in (".yml", ".yaml"):
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML: {e}") from e

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def categorize(name: str, rel_path: str) -> str:
        """Determine the category for a Claude Code integration file."""
        if name in ClaudeCodeFileService._KNOWN_CATEGORIES:
            return ClaudeCodeFileService._KNOWN_CATEGORIES[name]
        if "agents/" in rel_path or name.endswith("-agent.md"):
            return "Agents"
        if "skills/" in rel_path:
            return "Skills"
        if "commands/" in rel_path:
            return "Commands"
        if "hooks/" in rel_path:
            return "Hooks"
        if "rules/" in rel_path:
            return "Safety"
        return "Other"

    @staticmethod
    def detect_language(name: str) -> str:
        """Infer language/format from filename."""
        if name.endswith(".md"):
            return "markdown"
        if name.endswith(".json"):
            return "json"
        if name.endswith((".yml", ".yaml")):
            return "yaml"
        if name.endswith(".sh"):
            return "shell"
        if name.endswith(".py"):
            return "python"
        return "text"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_targets(self) -> list[tuple[Path, str]]:
        """Build the ordered list of (absolute_path, relative_path) targets."""
        targets: list[tuple[Path, str]] = [
            (self.project_dir / "CLAUDE.md", "CLAUDE.md"),
            (self.project_dir / ".mcp.json", ".mcp.json"),
            (self.project_dir / ".claude" / "settings.json", ".claude/settings.json"),
        ]

        claude_dir = self.project_dir / ".claude"
        for subdir in sorted(self.ALLOWED_DIRS):
            sub = claude_dir / subdir
            if sub.is_dir():
                for f in sorted(sub.rglob("*")):
                    if f.is_file() and not f.name.startswith("."):
                        rel = str(f.relative_to(self.project_dir))
                        targets.append((f, rel))

        return targets
