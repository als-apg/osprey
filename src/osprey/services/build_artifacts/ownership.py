"""Build artifact ownership helpers — config.yml and manifest updates.

These functions manage the ``scaffold.user_owned`` list in ``config.yml``
and the corresponding ``user_owned`` section of ``.osprey-manifest.json``.
Extracted from ``cli.scaffold_cmd`` so that both the CLI and web UI can
share ownership logic without a layering violation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.services.build_artifacts.catalog import BuildArtifactCatalog


def get_user_owned(config: dict) -> list[str]:
    """Extract scaffold.user_owned list from config."""
    return config.get("scaffold", {}).get("user_owned", [])


# ── Config.yml helpers ───────────────────────────────────────────────


def update_config_add_user_owned(project_dir: Path, name: str) -> bool:
    """Add a name to scaffold.user_owned list in config.yml, preserving comments.

    Returns True if the name was added, False if it was already present.
    """
    from osprey.utils.config_writer import config_add_to_list

    config_path = project_dir / "config.yml"
    return config_add_to_list(config_path, ["scaffold", "user_owned"], name)


def update_config_remove_user_owned(project_dir: Path, name: str) -> None:
    """Remove a name from scaffold.user_owned list in config.yml."""
    from osprey.utils.config_writer import config_remove_from_list

    config_path = project_dir / "config.yml"
    config_remove_from_list(config_path, ["scaffold", "user_owned"], name)


# ── Manifest helpers ─────────────────────────────────────────────────


def sha256_directory(directory: Path) -> str:
    """Compute a stable content hash of a directory tree.

    Hashes the sorted sequence of (relative path, file sha256) pairs so the
    result is deterministic across filesystems and independent of mtimes.
    Used for directory artifacts (service compose templates), which are
    copied verbatim — never rendered — so no template context is needed.
    """
    import hashlib

    from osprey.build.manifest import sha256_file

    digest = hashlib.sha256()
    for file in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = file.relative_to(directory).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(sha256_file(file).encode("ascii"))
    return digest.hexdigest()


def update_manifest_add_user_owned(
    project_dir: Path,
    manager,
    ctx: dict,
    name: str,
) -> None:
    """Add a user_owned entry to .osprey-manifest.json."""
    from osprey.build.manifest import MANIFEST_FILENAME, sha256_file

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
    registry = BuildArtifactCatalog.default()
    artifact = registry.get(name)
    framework_hash = None
    if artifact and artifact.is_directory:
        # Directory artifacts are copied verbatim (no render pass), so the
        # framework hash is a content hash of the packaged template tree.
        template_dir = manager.template_root / artifact.template_root / artifact.template_path
        if template_dir.is_dir():
            try:
                framework_hash = f"sha256:{sha256_directory(template_dir)}"
            except Exception:
                pass
    elif artifact:
        template_root_dir = manager.template_root / artifact.template_root
        template_file = template_root_dir / artifact.template_path
        if template_file.exists():
            try:
                if template_file.suffix == ".j2":
                    import tempfile

                    template_rel = f"{artifact.template_root}/{artifact.template_path}"
                    template = manager.jinja_env.get_template(template_rel)
                    rendered = template.render(**ctx)
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".tmp", delete=False, encoding="utf-8"
                    ) as tmp:
                        tmp.write(rendered)
                        tmp_path = Path(tmp.name)
                    framework_hash = f"sha256:{sha256_file(tmp_path)}"
                    tmp_path.unlink(missing_ok=True)
                else:
                    framework_hash = f"sha256:{sha256_file(template_file)}"
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
    from osprey.build.manifest import MANIFEST_FILENAME

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
