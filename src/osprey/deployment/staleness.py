"""Deploy-time project staleness advisory.

``osprey build`` stamps provenance into ``.osprey-manifest.json`` — the
osprey version that rendered the project and a content hash of the resolved
preset/profile (``creation.preset_hash``). This module reads that provenance
back when the project is *deployed* and warns when the render predates the
installed framework or preset, with the exact rebuild command as the remedy.

Rationale: a rendered project is a generated artifact. Its ``config.yml``
self-describes what to deploy, so a stale render deploys "successfully" while
silently missing every service the preset gained since — with no error
anywhere. Every other drift check in the codebase renders *from*
``config.yml`` and therefore cannot see this; the comparison here is against
the installed preset instead.

Advisory like
:func:`osprey.deployment.web_terminals.postup_hooks.warn_if_web_stack_unreachable`:
it warns loudly but never fails a deploy, and it fails open — a legacy
project without a manifest (or with a pre-hash manifest) deploys silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("deployment.staleness")

_MANIFEST_FILENAME = ".osprey-manifest.json"


def _installed_version() -> str:
    # The release lineage, matching what the manifest stores. Comparing running
    # versions would flag every commit in a development checkout as drift.
    from osprey.cli.templates.manifest import get_framework_release_version

    return get_framework_release_version()


def _load_manifest(project_dir: Path) -> dict[str, Any] | None:
    """Read the project manifest; ``None`` on missing/corrupt (fail open)."""
    try:
        data = json.loads((project_dir / _MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def staleness_reasons(project_dir: Path) -> list[str]:
    """Human-readable reasons this project's render trails the installed code.

    Empty list when the project is current — or when staleness cannot be
    determined (no manifest, unknown versions, preset no longer shipped):
    "can't compare" must never read as drift.

    :param project_dir: Project root (the directory holding config.yml and
        the manifest)
    """
    project_dir = Path(project_dir)
    manifest = _load_manifest(project_dir)
    if not manifest:
        return []

    reasons: list[str] = []
    creation = manifest.get("creation") or {}

    stored_version = creation.get("osprey_version")
    installed_version = _installed_version()
    if (
        stored_version
        and "unknown" not in (stored_version, installed_version)
        and stored_version != installed_version
    ):
        reasons.append(
            f"rendered by osprey {stored_version}; installed osprey is {installed_version}"
        )

    stored_hash = creation.get("preset_hash")
    if stored_hash:
        build_args = manifest.get("build_args") or {}
        # `profile_path_abs` is preferred over `profile_path`: the latter is the
        # string the user typed, and a relative one re-resolves against the
        # deploy's working directory, where it names nothing. Manifests written
        # before `profile_path_abs` existed still fall back to it. Resolved once
        # — the comparison below and the message further down must name the same
        # file, or the advisory would report a hash it did not compute.
        hashed_profile = build_args.get("profile_path_abs") or build_args.get("profile_path")
        current_hash = None
        try:
            from osprey.cli import build_profile

            if hashed_profile:
                # The profile comes first, including for a `--preset` build:
                # that build rendered from the profile it materialized, so the
                # profile is what the project must be compared against. Judging
                # it by the bundled preset instead would stay silent on a hand
                # edit to that profile — the very change the advisory exists to
                # catch, since editing the profile is how a facility changes
                # its project.
                current_hash = build_profile.compute_profile_hash(Path(hashed_profile))
            elif build_args.get("preset"):
                # A preset-built project from before profile-always builds, or
                # one whose profile path could not be recorded.
                current_hash = build_profile.compute_preset_hash(build_args["preset"])
        except Exception:
            current_hash = None
        if current_hash is not None and current_hash != stored_hash:
            # Name what was actually compared. Saying "preset X has changed" for
            # a project whose *profile* was edited would send the reader to the
            # wrong file — and under profile-always builds that is the common
            # case, since a preset build renders from a materialized profile.
            source = hashed_profile or build_args.get("preset") or "profile"
            kind = "profile" if hashed_profile else "preset"
            reasons.append(f"{kind} '{source}' has changed since this project was rendered")

    return reasons


def warn_if_project_stale(project_dir: Path) -> None:
    """Advisory: warn (never fail) when the project's render looks stale.

    Called by ``osprey deploy up`` before any container is touched, so the
    warning lands even when the deploy then "succeeds" at deploying an
    out-of-date service set.
    """
    try:
        project_dir = Path(project_dir)
        reasons = staleness_reasons(project_dir)
        if not reasons:
            return
        manifest = _load_manifest(project_dir) or {}
        message = (
            f"This project looks stale ({'; '.join(reasons)}). "
            "The deployed stack may be missing services or configuration the "
            "current framework provides."
        )
        remedy = manifest.get("reproducible_command")
        if remedy:
            message += f" Re-render it with:\n    {remedy} --force\nthen re-run osprey deploy up."
        logger.warning(message)
    except Exception as exc:  # advisory must never break a deploy
        logger.debug(f"Project staleness advisory skipped: {exc}")
