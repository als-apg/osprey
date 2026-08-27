"""Service for managing Claude Code integration files."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from osprey.audit.envelope import POSTURE_SOURCE_APP
from osprey.audit.protected import SURFACE_CLAUDE_SETUP, record_protected_refusal
from osprey.interfaces.web_terminal.ownership import reserved_write_channel
from osprey.utils.logger import get_logger

logger = get_logger("claude_code_files")

#: What the panel tells an operator about the files it will not save. Lives
#: here, beside the check that produces the refusals, so the copy and the gate
#: cannot drift apart; the ``GET /api/claude-setup`` handler serves it.
PROFILE_EDIT_NOTICE = (
    "CLAUDE.md, .mcp.json, .claude/settings.json and the rules and skills "
    "below .claude/ are rendered by the build profile. Edit them in the "
    "profile and rebuild the project — saves aimed at them here are refused, "
    "because a profile that no longer describes the project it built is worse "
    "than an edit that did not happen."
)


class ProtectedWriteError(PermissionError):
    """A write aimed at the protected set — the framework that constrains the agent.

    A :class:`PermissionError` so that the existing route mapping turns it into
    a 403 whether or not a handler knows this subclass exists; the subclass
    exists so that a handler that *does* know can tell "you may not rewrite the
    framework" apart from "that path escapes the project" and surface the two
    differently.

    Attributes:
        rel_path: The project-relative path the write was aimed at.
        channel: The channel that owns that path, phrased as
            :func:`~osprey.interfaces.web_terminal.ownership.reserved_write_channel`
            phrases it, so the message, the audit record and the log all name
            the same way in.
    """

    def __init__(self, rel_path: str, channel: str):
        self.rel_path = rel_path
        self.channel = channel
        super().__init__(
            f"Refused: nothing was written to {rel_path}. The change belongs in {channel}."
        )


def _refuse_if_reserved(project_dir: Path, rel_path: str) -> None:
    """Refuse *rel_path* if the protected set owns it, recording the refusal.

    Both write routes ask this first, so the two cannot drift on either the
    question or the audit record it leaves.

    The question goes to :func:`~osprey.interfaces.web_terminal.ownership.reserved_write_channel`
    rather than to the lexical ``is_reserved_write``, because a save reaches
    disk through the filesystem and the filesystem follows links: a link at an
    unprotected name (``.claude/agents/x.md`` -> ``../rules/safety.md``) is
    lexically an agent and physically a rule. The gallery already asks the
    resolving question; the panel is the same policy through a different door,
    and a second answer here would be a second policy.

    Args:
        project_dir: Resolved project root ``rel_path`` is relative to.
        rel_path: Project-relative path the panel would write.

    Raises:
        ProtectedWriteError: ``rel_path`` names — or lands on — a file in the
            protected set, or is not project-relative at all.
    """
    channel = reserved_write_channel(project_dir, rel_path)
    if channel is not None:
        record_protected_refusal(
            surface=SURFACE_CLAUDE_SETUP,
            target_file=rel_path,
            key_or_path=rel_path,
            channel=channel,
            reason="reserved path",
            # The panel is only ever reached over HTTP, and a web request
            # belongs to no session: the server process carries no posture
            # stamp of its own, so the env ladder would file this as a bare
            # ``process`` next to the ``app`` ``HttpAuditMiddleware`` stamps
            # for the same request.
            posture_source=POSTURE_SOURCE_APP,
        )
        raise ProtectedWriteError(rel_path, channel)


class ClaudeCodeFileService:
    """Service for discovering, reading, writing, and creating Claude Code files.

    Centralises file system logic and path security checks so that API route
    handlers stay thin.
    """

    ALLOWED_DIRS = {"rules", "agents", "commands", "hooks", "skills", "output-styles"}
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
                    # Computed from the same call ``write_file`` makes -- link
                    # resolution included -- so the editor greys out exactly the
                    # files a save would refuse.
                    "read_only": reserved_write_channel(self.project_dir, rel_path) is not None,
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
        """Write content to an existing file with path security + syntax validation.

        The protected set is consulted *first*: the question "may a running
        agent rewrite the file this write would land on at all" is answered
        before anything else, so a reserved path gets the refusal that names
        its channel rather than whichever generic error the later checks would
        have produced.

        Raises:
            ProtectedWriteError: ``rel_path`` is in the protected set, or is
                not project-relative at all.
            PermissionError: the resolved path escapes the project root.
            FileNotFoundError: the file does not exist (this route edits, it
                does not create).
            ValueError: the content is not valid JSON/YAML for the suffix.
        """
        _refuse_if_reserved(self.project_dir, rel_path)

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
        """Create a new file in an allowed .claude/ subdirectory.

        The protected set is consulted *first*, on the same terms as
        :meth:`write_file` and for the same reason: a subtree closed to
        rewrites is not closed at all while a new file may still be dropped
        into it. Asking first also keeps the refusal specific — a reserved
        path is told which channel owns it, rather than being handed the
        allowlist message, which would send an operator off to pick a
        different directory instead of to the profile.

        Raises:
            ProtectedWriteError: ``rel_path`` is in the protected set, or is
                not project-relative at all.
            PermissionError: the resolved path escapes the project root, or
                the path is outside the allowed ``.claude/`` subdirectories.
            FileExistsError: the file already exists (this route creates, it
                does not overwrite).
            ValueError: the content is not valid JSON/YAML for the suffix.
        """
        _refuse_if_reserved(self.project_dir, rel_path)

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
        if "output-styles/" in rel_path:
            return "Output Styles"
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
