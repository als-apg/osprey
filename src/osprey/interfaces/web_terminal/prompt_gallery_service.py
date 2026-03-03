"""Prompt Gallery Service — bridges PromptRegistry + TemplateManager for the web UI.

Stateless service class instantiated per-request with a project directory.
Provides list/get/diff/claim/save/unclaim operations that the frontend
gallery consumes via the ``/api/prompts`` route family.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

import yaml

from osprey.cli.prompt_registry import PromptArtifact, PromptRegistry
from osprey.cli.prompts_cmd import (
    _get_user_owned,
    _update_config_add_user_owned,
    _update_config_remove_user_owned,
    _update_manifest_add_user_owned,
    _update_manifest_remove_user_owned,
)
from osprey.cli.templates.manager import TemplateManager
from osprey.utils.config import resolve_env_vars

# Language inference from file extension
_EXT_LANG = {
    ".md": "markdown",
    ".py": "python",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
}

# Front-matter extraction patterns
_FM_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_PY_FM_RE = re.compile(r'^(?:#!.*\n)?"""\n---\n(.*?)\n---', re.DOTALL)

# Subdirectories under .claude/ to scan for artifact files
_CLAUDE_SUBDIRS = ("agents", "commands", "hooks", "output-styles", "rules", "skills")

# Config-tier files: (relative_path, relative_path) — checked first in scan order
_CONFIG_FILES = (
    ("CLAUDE.md", "CLAUDE.md"),
    (".mcp.json", ".mcp.json"),
    (".claude/settings.json", ".claude/settings.json"),
)


class PromptGalleryService:
    """Service for the Prompt Gallery web UI.

    Instantiated per-request with the project directory. All methods are
    synchronous since the web terminal runs them in a thread pool.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self._registry = PromptRegistry.default()
        self._config = self._load_config()
        self._user_owned = _get_user_owned(self._config)
        self._manager: TemplateManager | None = None
        self._ctx: dict[str, Any] | None = None

    def _load_config(self) -> dict[str, Any]:
        config_file = self.project_dir / "config.yml"
        if not config_file.exists():
            return {}
        with open(config_file, encoding="utf-8") as f:
            return resolve_env_vars(yaml.safe_load(f) or {})

    # ── List ──────────────────────────────────────────────────────────

    def list_artifacts(self) -> list[dict[str, Any]]:
        """Return all artifacts with status and metadata.

        Filesystem-first: only artifacts whose files exist on disk are
        returned, enriched with registry metadata where available.
        Custom user-owned files (registered via :meth:`register_untracked`)
        are also included.
        """
        result: list[dict[str, Any]] = []
        registry_names: set[str] = set()

        # Phase 1: filesystem-driven discovery
        for _abs_path, rel_path in self._scan_all_files():
            art = self._registry.get_by_output(rel_path)

            if art is not None:
                # Registry-backed artifact that exists on disk
                registry_names.add(art.canonical_name)
                is_owned = art.canonical_name in self._user_owned
                category = (
                    art.canonical_name.split("/")[0] if "/" in art.canonical_name else "config"
                )
                fm = self._read_front_matter_from_disk(rel_path, art)
                summary = fm.get("summary") or art.description
                description = fm.get("description") or art.description

                result.append(
                    {
                        "name": art.canonical_name,
                        "category": category,
                        "summary": summary,
                        "description": description,
                        "output_path": art.output_path,
                        "status": "user-owned" if is_owned else "framework",
                        "custom": False,
                        "language": self._infer_language(art.output_path),
                    }
                )
            else:
                # Not in registry — check if it's a registered custom artifact
                canonical = self._path_to_canonical(rel_path)
                if canonical in self._user_owned and canonical not in registry_names:
                    registry_names.add(canonical)
                    fm = self._read_front_matter_from_disk(rel_path)
                    summary = fm.get("summary", "(custom user file)")
                    description = fm.get(
                        "description", "(custom user file — no framework template)"
                    )
                    category = canonical.split("/")[0] if "/" in canonical else "other"
                    result.append(
                        {
                            "name": canonical,
                            "category": category,
                            "summary": summary,
                            "description": description,
                            "output_path": rel_path,
                            "status": "user-owned",
                            "custom": True,
                            "language": self._infer_language(rel_path),
                        }
                    )

        return result

    # ── Content retrieval ─────────────────────────────────────────────

    def get_content(self, name: str) -> dict[str, Any]:
        """Get the active content for an artifact (always reads from disk)."""
        art = self._registry.get(name)
        if art is None:
            return self._get_custom_content(name)

        disk_content = self._read_user_file(art)
        if disk_content is not None:
            source = "user-owned" if art.canonical_name in self._user_owned else "framework"
            return {
                "content": disk_content,
                "source": source,
                "language": self._infer_language(art.output_path),
            }
        # File not on disk — render from template as fallback
        return {
            "content": self._render_framework(art),
            "source": "framework",
            "language": self._infer_language(art.output_path),
        }

    def get_framework_content(self, name: str) -> str:
        """Render and return the framework template content."""
        art = self._get_artifact(name)
        return self._render_framework(art)

    def get_override_content(self, name: str) -> str | None:
        """Read the user-owned file content, or None if not user-owned."""
        art = self._get_artifact(name)
        if art.canonical_name not in self._user_owned:
            return None
        return self._read_user_file(art)

    # ── Diff ──────────────────────────────────────────────────────────

    def compute_diff(self, name: str) -> dict[str, Any]:
        """Compute unified diff between framework and user-owned file."""
        art = self._get_artifact(name)
        if art.canonical_name not in self._user_owned:
            raise FileNotFoundError(f"'{name}' is not user-owned — no diff available")

        user_content = self._read_user_file(art)
        if user_content is None:
            raise FileNotFoundError(f"User-owned file not found: {art.output_path}")

        framework_content = self._render_framework(art)
        framework_lines = framework_content.splitlines(keepends=True)
        user_lines = user_content.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                framework_lines,
                user_lines,
                fromfile=f"framework:{art.template_path}",
                tofile=f"yours:{art.output_path}",
            )
        )

        additions = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
        deletions = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))

        return {
            "unified_diff": "".join(diff_lines),
            "has_diff": len(diff_lines) > 0,
            "additions": additions,
            "deletions": deletions,
        }

    # ── Claim ─────────────────────────────────────────────────────────

    def scaffold_override(self, name: str) -> dict[str, Any]:
        """Claim an artifact for in-place editing."""
        art = self._get_artifact(name)

        if name in self._user_owned:
            raise FileExistsError(f"'{name}' is already user-owned")

        # Render framework content if file doesn't exist
        output_file = self.project_dir / art.output_path
        if not output_file.exists():
            content = self._render_framework(art)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            if output_file.suffix == ".py":
                output_file.chmod(output_file.stat().st_mode | 0o755)
        else:
            content = output_file.read_text(encoding="utf-8")

        # Update config.yml
        _update_config_add_user_owned(self.project_dir, name)

        # Update manifest
        manager, ctx = self._ensure_template_context()
        _update_manifest_add_user_owned(self.project_dir, manager, ctx, name)

        # Refresh internal state
        self._config = self._load_config()
        self._user_owned = _get_user_owned(self._config)

        return {
            "status": "claimed",
            "output_path": art.output_path,
            "content": content,
        }

    # ── Save ──────────────────────────────────────────────────────────

    def save_override(self, name: str, content: str) -> dict[str, Any]:
        """Write content to a user-owned file."""
        if name not in self._user_owned:
            raise FileNotFoundError(f"'{name}' is not user-owned — claim first")

        art = self._registry.get(name)
        out = art.output_path if art else self._canonical_to_path(name)

        output_path = self.project_dir / out
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        return {"status": "saved", "path": out}

    # ── Unclaim ──────────────────────────────────────────────────────

    def unoverride(self, name: str, delete_file: bool = False) -> dict[str, Any]:
        """Release ownership, restoring framework management."""
        if name not in self._user_owned:
            raise FileNotFoundError(f"'{name}' is not user-owned")

        is_custom = self._registry.get(name) is None

        # Remove from config.yml
        _update_config_remove_user_owned(self.project_dir, name)

        # Remove from manifest (only for registry artifacts)
        if not is_custom:
            _update_manifest_remove_user_owned(self.project_dir, name)

        deleted = False
        restored = False
        if delete_file and is_custom:
            # Custom artifact (no framework template) — delete the file
            out = self.project_dir / self._canonical_to_path(name)
            if out.exists():
                out.unlink()
                deleted = True
        elif delete_file and not is_custom:
            # Framework artifact — restore file to rendered template
            art = self._registry.get(name)
            try:
                content = self._render_framework(art)
                out = self.project_dir / art.output_path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content, encoding="utf-8")
                restored = True
            except Exception:
                pass  # ownership already removed; stale file stays

        # Refresh internal state
        self._config = self._load_config()
        self._user_owned = _get_user_owned(self._config)

        return {"status": "removed", "deleted_file": deleted, "restored_file": restored}

    # ── Untracked file detection ─────────────────────────────────────

    @property
    def _scan_dirs(self) -> tuple[str, ...]:
        """Categories where users may create new .md files (excludes hooks and config)."""
        return tuple(sorted(self._registry.categories - {"config", "hooks"}))

    def scan_untracked(self) -> list[dict[str, Any]]:
        """Find files in .claude/ that are active in Claude Code but not managed.

        Scans .claude/{rules,agents,commands,skills}/ for .md files whose
        output path doesn't match any registered artifact and whose derived
        canonical name isn't already in ``prompts.user_owned``.
        """
        claude_dir = self.project_dir / ".claude"
        if not claude_dir.is_dir():
            return []

        registered_outputs: set[str] = {a.output_path for a in self._registry.all_artifacts()}

        untracked: list[dict[str, Any]] = []
        for subdir_name in self._scan_dirs:
            subdir = claude_dir / subdir_name
            if not subdir.is_dir():
                continue
            for fpath in sorted(subdir.rglob("*.md")):
                if not fpath.is_file() or fpath.name.startswith("."):
                    continue
                rel_path = str(fpath.relative_to(self.project_dir))
                if rel_path in registered_outputs:
                    continue
                canonical = self._path_to_canonical(rel_path)
                if canonical in self._user_owned:
                    continue
                preview = self._safe_preview(fpath)
                untracked.append(
                    {
                        "canonical_name": canonical,
                        "output_path": rel_path,
                        "category": canonical.split("/")[0] if "/" in canonical else "other",
                        "preview": preview,
                    }
                )
        return untracked

    def register_untracked(self, canonical_name: str) -> dict[str, Any]:
        """Register an untracked file by adding it to ``prompts.user_owned``."""
        output_path = self._canonical_to_path(canonical_name)
        full_path = self.project_dir / output_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found on disk: {output_path}")
        if canonical_name in self._user_owned:
            raise FileExistsError(f"'{canonical_name}' is already registered")

        _update_config_add_user_owned(self.project_dir, canonical_name)

        self._config = self._load_config()
        self._user_owned = _get_user_owned(self._config)

        return {"status": "registered", "canonical_name": canonical_name}

    def delete_untracked(self, canonical_name: str) -> dict[str, Any]:
        """Delete an untracked file from disk."""
        if self._registry.get(canonical_name) is not None:
            raise ValueError(f"'{canonical_name}' is a framework artifact — use unoverride instead")
        output_path = self._canonical_to_path(canonical_name)
        full_path = self.project_dir / output_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found on disk: {output_path}")

        full_path.unlink()
        return {"status": "deleted", "output_path": output_path}

    @staticmethod
    def _path_to_canonical(rel_path: str) -> str:
        """Derive a canonical name from a relative output path.

        ``.claude/rules/test.md`` -> ``rules/test``
        """
        # Strip leading .claude/
        name = rel_path
        if name.startswith(".claude/"):
            name = name[len(".claude/") :]
        # Strip .md extension
        if name.endswith(".md"):
            name = name[: -len(".md")]
        return name

    @staticmethod
    def _canonical_to_path(canonical_name: str) -> str:
        """Convert a canonical name back to a relative output path."""
        return f".claude/{canonical_name}.md"

    @staticmethod
    def _safe_preview(fpath: Path, max_chars: int = 200) -> str:
        """Read the first ``max_chars`` characters of a file for preview."""
        try:
            text = fpath.read_text(encoding="utf-8")
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."
        except (UnicodeDecodeError, PermissionError, OSError):
            return "(unable to read)"

    # ── Private helpers ───────────────────────────────────────────────

    def _get_custom_content(self, name: str) -> dict[str, Any]:
        """Read content for a custom user-owned artifact (no framework template)."""
        if name not in self._user_owned:
            raise KeyError(f"Unknown artifact: '{name}'")
        output_path = self._canonical_to_path(name)
        fpath = self.project_dir / output_path
        if not fpath.exists():
            raise FileNotFoundError(f"Custom file not found: {output_path}")
        content = fpath.read_text(encoding="utf-8")
        return {
            "content": content,
            "source": "user-owned",
            "language": self._infer_language(output_path),
        }

    def _scan_all_files(self) -> list[tuple[Path, str]]:
        """Yield all existing Claude Code files as (absolute_path, rel_path) pairs.

        Mirrors ClaudeCodeFileService._collect_targets() but used for prompt
        gallery discovery.  Config-tier files come first, then rglob on each
        subdirectory.
        """
        targets: list[tuple[Path, str]] = []

        # Config-tier files
        for rel_path, _ in _CONFIG_FILES:
            fpath = self.project_dir / rel_path
            if fpath.is_file():
                targets.append((fpath, rel_path))

        # Subdirectory files
        claude_dir = self.project_dir / ".claude"
        for subdir_name in _CLAUDE_SUBDIRS:
            subdir = claude_dir / subdir_name
            if not subdir.is_dir():
                continue
            for fpath in sorted(subdir.rglob("*")):
                if fpath.is_file() and not fpath.name.startswith("."):
                    rel = str(fpath.relative_to(self.project_dir))
                    targets.append((fpath, rel))

        return targets

    def _read_front_matter_from_disk(
        self,
        rel_path: str,
        art: PromptArtifact | None = None,
    ) -> dict[str, str]:
        """Extract front-matter from on-disk content, falling back to rendered template.

        Reads the file at *rel_path* first (reflects user modifications).
        Falls back to ``_render_framework(art)`` only if the file is absent
        AND a ``PromptArtifact`` is provided.  Uses both ``_FM_RE`` and
        ``_PY_FM_RE`` patterns so ``.py`` hook files are handled correctly.
        """
        content: str | None = None
        fpath = self.project_dir / rel_path
        if fpath.is_file():
            try:
                content = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError, OSError):
                pass

        if content is None and art is not None:
            try:
                content = self._render_framework(art)
            except Exception:
                return {}

        if content is None:
            return {}

        match = _FM_RE.match(content) or _PY_FM_RE.match(content)
        if not match:
            return {}

        fields: dict[str, str] = {}
        for line in match.group(1).split("\n"):
            kv = re.match(r'^(\w[\w-]*):\s*"?(.*?)"?\s*$', line)
            if kv:
                fields[kv.group(1)] = kv.group(2)
        return fields

    def _get_artifact(self, name: str) -> PromptArtifact:
        """Look up an artifact by name, raising if unknown."""
        art = self._registry.get(name)
        if art is None:
            raise KeyError(f"Unknown artifact: '{name}'")
        return art

    def _ensure_template_context(self) -> tuple[TemplateManager, dict[str, Any]]:
        """Return cached (manager, context) pair, creating on first call."""
        if self._manager is None:
            self._manager = TemplateManager()
            from osprey.cli.templates.claude_code import build_claude_code_context

            self._ctx = build_claude_code_context(
                self._manager.template_root, self._manager.jinja_env, self.project_dir, self._config
            )
        return self._manager, self._ctx  # type: ignore[return-value]

    def _render_framework(self, art: PromptArtifact) -> str:
        """Render the framework template with the current config context."""
        manager, ctx = self._ensure_template_context()
        claude_code_dir = manager.template_root / "claude_code"
        template_file = claude_code_dir / art.template_path

        if not template_file.exists():
            return f"# Template not found: {art.template_path}\n"

        if template_file.suffix == ".j2":
            template_rel = f"claude_code/{art.template_path}"
            template = manager.jinja_env.get_template(template_rel)
            return template.render(**ctx)
        else:
            return template_file.read_text(encoding="utf-8")

    def _read_user_file(self, art: PromptArtifact) -> str | None:
        """Read the user's file at the canonical output path."""
        output_path = self.project_dir / art.output_path
        if not output_path.exists():
            return None
        return output_path.read_text(encoding="utf-8")

    @staticmethod
    def _infer_language(output_path: str) -> str:
        """Infer language from the output file extension."""
        from pathlib import PurePosixPath

        ext = PurePosixPath(output_path).suffix
        return _EXT_LANG.get(ext, "text")
