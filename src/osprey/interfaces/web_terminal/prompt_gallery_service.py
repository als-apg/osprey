"""Prompt Gallery Service — bridges PromptRegistry + TemplateManager for the web UI.

Stateless service class instantiated per-request with a project directory.
Provides list/get/diff/scaffold/save/unoverride operations that the frontend
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
    _cleanup_empty_dirs,
    _get_overrides,
    _update_config_add_override,
    _update_config_remove_override,
    _update_manifest_add_override,
    _update_manifest_remove_override,
)
from osprey.cli.templates import TemplateManager

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


class PromptGalleryService:
    """Service for the Prompt Gallery web UI.

    Instantiated per-request with the project directory. All methods are
    synchronous since the web terminal runs them in a thread pool.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self._registry = PromptRegistry.default()
        self._config = self._load_config()
        self._overrides = _get_overrides(self._config)
        self._manager: TemplateManager | None = None
        self._ctx: dict[str, Any] | None = None

    def _load_config(self) -> dict[str, Any]:
        config_file = self.project_dir / "config.yml"
        if not config_file.exists():
            return {}
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ── List ──────────────────────────────────────────────────────────

    def list_artifacts(self) -> list[dict[str, Any]]:
        """Return all artifacts with status and metadata."""
        result = []
        for art in self._registry.all_artifacts():
            override_path = self._overrides.get(art.canonical_name)
            status = "overridden" if override_path else "framework"
            category = (
                art.canonical_name.split("/")[0]
                if "/" in art.canonical_name
                else "config"
            )

            fm = self._extract_front_matter(art)
            summary = fm.get("summary") or art.description
            description = fm.get("description") or art.description

            result.append({
                "name": art.canonical_name,
                "category": category,
                "summary": summary,
                "description": description,
                "output_path": art.output_path,
                "status": status,
                "override_path": override_path,
                "language": self._infer_language(art.output_path),
            })
        return result

    # ── Content retrieval ─────────────────────────────────────────────

    def get_content(self, name: str) -> dict[str, Any]:
        """Get the active content for an artifact (override takes priority)."""
        art = self._get_artifact(name)
        override_content = self._read_override(art)
        if override_content is not None:
            return {
                "content": override_content,
                "source": "override",
                "language": self._infer_language(art.output_path),
            }
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
        """Read the override file content, or None if not overridden."""
        art = self._get_artifact(name)
        return self._read_override(art)

    # ── Diff ──────────────────────────────────────────────────────────

    def compute_diff(self, name: str) -> dict[str, Any]:
        """Compute unified diff between framework and override."""
        art = self._get_artifact(name)
        override_content = self._read_override(art)
        if override_content is None:
            raise FileNotFoundError(
                f"'{name}' is not overridden — no diff available"
            )

        framework_content = self._render_framework(art)
        framework_lines = framework_content.splitlines(keepends=True)
        override_lines = override_content.splitlines(keepends=True)

        diff_lines = list(difflib.unified_diff(
            framework_lines,
            override_lines,
            fromfile=f"framework:{art.template_path}",
            tofile=f"override:{self._overrides[name]}",
        ))

        additions = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
        deletions = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))

        return {
            "unified_diff": "".join(diff_lines),
            "has_diff": len(diff_lines) > 0,
            "additions": additions,
            "deletions": deletions,
        }

    # ── Scaffold ──────────────────────────────────────────────────────

    def scaffold_override(self, name: str) -> dict[str, Any]:
        """Create an override file from the framework template."""
        art = self._get_artifact(name)

        if name in self._overrides:
            raise FileExistsError(
                f"Override for '{name}' already exists at {self._overrides[name]}"
            )

        # Render framework content
        content = self._render_framework(art)

        # Write override file
        override_rel = f"overrides/{art.output_path}"
        override_path = self.project_dir / override_rel
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(content, encoding="utf-8")
        if override_path.suffix == ".py":
            override_path.chmod(override_path.stat().st_mode | 0o755)

        # Update config.yml
        _update_config_add_override(self.project_dir, name, override_rel)

        # Update manifest
        manager = TemplateManager()
        ctx = manager._build_claude_code_context(self.project_dir, self._config)
        _update_manifest_add_override(
            self.project_dir, manager, ctx, name, override_rel
        )

        # Refresh internal state
        self._config = self._load_config()
        self._overrides = _get_overrides(self._config)

        return {
            "status": "created",
            "override_path": override_rel,
            "content": content,
        }

    # ── Save ──────────────────────────────────────────────────────────

    def save_override(self, name: str, content: str) -> dict[str, Any]:
        """Write content to an existing override file."""
        self._get_artifact(name)  # validate name

        if name not in self._overrides:
            raise FileNotFoundError(
                f"'{name}' is not overridden — scaffold first"
            )

        override_path = self.project_dir / self._overrides[name]
        override_path.parent.mkdir(parents=True, exist_ok=True)
        override_path.write_text(content, encoding="utf-8")

        return {"status": "saved", "path": self._overrides[name]}

    # ── Unoverride ────────────────────────────────────────────────────

    def unoverride(self, name: str, delete_file: bool = False) -> dict[str, Any]:
        """Remove an override, restoring framework management."""
        self._get_artifact(name)  # validate name

        if name not in self._overrides:
            raise FileNotFoundError(
                f"'{name}' is not overridden"
            )

        override_rel = self._overrides[name]

        # Remove from config.yml
        _update_config_remove_override(self.project_dir, name)

        # Remove from manifest
        _update_manifest_remove_override(self.project_dir, name)

        # Optionally delete the file
        if delete_file:
            override_path = self.project_dir / override_rel
            if override_path.exists():
                override_path.unlink()
            _cleanup_empty_dirs(
                override_path.parent, self.project_dir / "overrides"
            )

        # Refresh internal state
        self._config = self._load_config()
        self._overrides = _get_overrides(self._config)

        return {"status": "removed", "deleted_file": delete_file}

    # ── Private helpers ───────────────────────────────────────────────

    def _get_artifact(self, name: str) -> PromptArtifact:
        """Look up an artifact by name, raising if unknown."""
        art = self._registry.get(name)
        if art is None:
            raise KeyError(f"Unknown artifact: '{name}'")
        return art

    def _ensure_template_context(self) -> tuple["TemplateManager", dict[str, Any]]:
        """Return cached (manager, context) pair, creating on first call."""
        if self._manager is None:
            self._manager = TemplateManager()
            self._ctx = self._manager._build_claude_code_context(
                self.project_dir, self._config
            )
        return self._manager, self._ctx  # type: ignore[return-value]

    def _extract_front_matter(self, art: PromptArtifact) -> dict[str, str]:
        """Extract front matter fields from the artifact's rendered content."""
        try:
            content = self._render_framework(art)
        except Exception:
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

    def _read_override(self, art: PromptArtifact) -> str | None:
        """Read the override file if the artifact is overridden."""
        override_rel = self._overrides.get(art.canonical_name)
        if not override_rel:
            return None
        override_path = self.project_dir / override_rel
        if not override_path.exists():
            return None
        return override_path.read_text(encoding="utf-8")

    @staticmethod
    def _infer_language(output_path: str) -> str:
        """Infer language from the output file extension."""
        from pathlib import PurePosixPath

        ext = PurePosixPath(output_path).suffix
        return _EXT_LANG.get(ext, "text")
