"""Prompt ownership helpers — config.yml and manifest updates.

These functions manage the ``prompts.user_owned`` list in ``config.yml``
and the corresponding ``user_owned`` section of ``.osprey-manifest.json``.
Extracted from ``cli.prompts_cmd`` so that both the CLI and web UI can
share ownership logic without a layering violation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.services.prompts.catalog import PromptCatalog


def get_user_owned(config: dict) -> list[str]:
    """Extract prompts.user_owned list from config."""
    return config.get("prompts", {}).get("user_owned", [])


# ── Config.yml helpers ───────────────────────────────────────────────


def update_config_add_user_owned(project_dir: Path, name: str) -> bool:
    """Add a name to prompts.user_owned list in config.yml, preserving comments.

    Returns True if the name was added, False if it was already present.
    """
    from osprey.utils.config_writer import config_add_to_list

    config_path = project_dir / "config.yml"
    return config_add_to_list(config_path, ["prompts", "user_owned"], name)


def update_config_remove_user_owned(project_dir: Path, name: str) -> None:
    """Remove a name from prompts.user_owned list in config.yml."""
    from osprey.utils.config_writer import config_remove_from_list

    config_path = project_dir / "config.yml"
    config_remove_from_list(config_path, ["prompts", "user_owned"], name)


# ── Manifest helpers ─────────────────────────────────────────────────


def update_manifest_add_user_owned(
    project_dir: Path,
    manager,
    ctx: dict,
    name: str,
) -> None:
    """Add a user_owned entry to .osprey-manifest.json."""
    from osprey.cli.templates import manifest as manifest_mod
    from osprey.cli.templates.manifest import MANIFEST_FILENAME

    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    if "user_owned" not in manifest:
        manifest["user_owned"] = {}

    # Compute framework hash
    registry = PromptCatalog.default()
    artifact = registry.get(name)
    framework_hash = None
    if artifact:
        claude_code_dir = manager.template_root / "claude_code"
        template_file = claude_code_dir / artifact.template_path
        if template_file.exists():
            try:
                if template_file.suffix == ".j2":
                    import tempfile

                    template_rel = f"claude_code/{artifact.template_path}"
                    template = manager.jinja_env.get_template(template_rel)
                    rendered = template.render(**ctx)
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".tmp", delete=False, encoding="utf-8"
                    ) as tmp:
                        tmp.write(rendered)
                        tmp_path = Path(tmp.name)
                    framework_hash = f"sha256:{manifest_mod.sha256_file(tmp_path)}"
                    tmp_path.unlink(missing_ok=True)
                else:
                    framework_hash = f"sha256:{manifest_mod.sha256_file(template_file)}"
            except Exception:
                pass

    entry: dict[str, Any] = {
        "claimed_at": datetime.now(UTC).isoformat(),
    }
    if framework_hash:
        entry["framework_hash"] = framework_hash

    manifest["user_owned"][name] = entry

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)


def update_manifest_remove_user_owned(project_dir: Path, name: str) -> None:
    """Remove a user_owned entry from .osprey-manifest.json."""
    from osprey.cli.templates.manifest import MANIFEST_FILENAME

    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    user_owned = manifest.get("user_owned", {})
    if name in user_owned:
        del user_owned[name]
        if not user_owned:
            manifest.pop("user_owned", None)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
