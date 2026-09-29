"""Build artifact ownership helpers — config.yml and manifest updates.

These functions manage the ``scaffold.user_owned`` list in ``config.yml``
and the corresponding ``user_owned`` section of ``.osprey-manifest.json``.
Extracted from ``cli.scaffold_cmd`` so that both the CLI and web UI can
share ownership logic without a layering violation.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.agent_runner.build_artifacts.catalog import BuildArtifact, BuildArtifactCatalog

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jinja2 import Environment


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


def framework_template_hash(
    templates_dir: Path,
    artifact: BuildArtifact,
    jinja_env: Environment,
    context: Mapping[str, Any],
) -> str | None:
    """``sha256:`` digest of the framework's own version of one artifact.

    Recorded when an artifact is claimed and recomputed on every regen, so the
    two must be computed identically or every regen would report drift that is
    not there. That is the whole reason this lives in one function: the
    callers are in different modules and would otherwise be free to differ on
    the template location, the render context, the encoding, or the
    ``sha256:`` prefix.

    A ``.j2`` template is rendered first — the digest is of what the framework
    would *write*, not of the template that writes it, so a context change is
    drift and a comment change in the template is not. A directory artifact is
    copied verbatim and rendered later by ``osprey up``, so its hash is the
    tree digest of the packaged directory, with no render and no context.

    Args:
        templates_dir: The bundled templates directory; the artifact's
            ``template_root`` and ``template_path`` are resolved below it.
        artifact: The catalog artifact to hash.
        jinja_env: Jinja environment the render goes through.
        context: Template context for the render.

    Returns:
        ``sha256:<hex>``, or ``None`` when the template is missing or will not
        render. Callers treat ``None`` as "no comparison possible" rather than
        as drift: a template that cannot render is a framework problem, and
        reporting it as the operator's artifact having drifted would misdirect.
    """
    from osprey.build.manifest import sha256_file

    source = templates_dir / artifact.template_root / artifact.template_path
    try:
        if artifact.is_directory:
            return f"sha256:{sha256_directory(source)}" if source.is_dir() else None
        if not source.is_file():
            return None
        if source.suffix != ".j2":
            return f"sha256:{sha256_file(source)}"
        template = jinja_env.get_template(f"{artifact.template_root}/{artifact.template_path}")
        rendered = template.render(**context)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=source.stem, delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(rendered)
            return f"sha256:{sha256_file(tmp_path)}"
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
    except Exception:
        return None


def update_manifest_add_user_owned(
    project_dir: Path,
    manager,
    ctx: dict,
    name: str,
) -> None:
    """Add a user_owned entry to .osprey-manifest.json."""
    from osprey.build.manifest import MANIFEST_FILENAME

    manifest_path = project_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    if "user_owned" not in manifest:
        manifest["user_owned"] = {}

    artifact = BuildArtifactCatalog.default().get(name)
    framework_hash = (
        framework_template_hash(manager.template_root, artifact, manager.jinja_env, ctx)
        if artifact
        else None
    )

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
