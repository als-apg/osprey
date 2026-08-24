"""Deploy-time project staleness advisory and the build-drift verdict.

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

Two readings of the same provenance live here, and they differ in severity:

- :func:`staleness_reasons` / :func:`warn_if_project_stale` — the legacy
  advisory over a rendered project directory. Warns loudly, never fails, and
  fails open: a project without a manifest deploys silently.
- :func:`check_drift` — the deployment-repo verdict the lifecycle verbs act
  on. ``up``/``restart`` refuse on it, ``status`` prints it, ``chat``/``query``
  warn on it. Here "cannot compare" is *not* silence: a build whose profile no
  longer parses, or which carries no fingerprint at all, is reported as
  ``UNRESOLVABLE`` so a start verb refuses rather than guessing. ``--as-built``
  is the documented escape from both refusals — ``build/`` is self-contained,
  so starting it as rendered stays a valid, knowing choice.

Both read the *same* fingerprint: ``compute_profile_hash`` over the resolved
profile document, the file material it names, and the host-variant overlay
``.env.variant`` selects, plus the ``creation.osprey_version`` stamp. The
variant is part of it because the overlay is part of what was rendered: without
it, editing the selected ``profiles/<host>.yml`` — or pointing the setting at a
different one — would leave ``build/`` reading CLEAN while it holds the render
of an overlay nobody is using any more. There is deliberately no second hash
mechanism — the per-key digests :func:`profile_fingerprint` derives are
diagnostic only, used to *name* what changed, never to decide whether
anything did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger
from osprey.utils.workspace import BUILD_DIR_NAME

logger = get_logger("deployment.staleness")

_MANIFEST_FILENAME = ".osprey-manifest.json"

#: Where ``osprey build`` renders the deployment repo's output zone. The
#: manifest sits at its root: ``build/`` *is* the rendered project, so there is
#: exactly one place a repo's build provenance can be. Aliased to
#: :data:`osprey.utils.workspace.BUILD_DIR_NAME` rather than spelled again — a
#: second literal for the same directory is a second thing to get wrong.
BUILD_DIRNAME = BUILD_DIR_NAME

#: Manifest key (under ``creation``) holding the per-top-level-key digests
#: :func:`profile_fingerprint` derives. Optional: the drift *verdict* never
#: depends on it, only the "what changed" detail in the refusal message.
KEY_DIGESTS_MANIFEST_KEY = "profile_key_digests"

#: Prefix marking a digest key as *file material* — something the profile YAML
#: points at rather than says. Prefixed so the two kinds of key stay tellable
#: apart when they are read back out of a manifest: :func:`_changed_keys` has to
#: know whether a named input already accounts for the material having moved,
#: and a profile key called ``data`` must never be confused for the ``data/``
#: tree it names.
MATERIAL_PREFIX = "material:"

#: Digest over *all* the material at once — the ``data:`` tree, the convention
#: directories, ``triggers.yml`` and the ``personas/`` deltas — folded by the
#: build's own :func:`~osprey.cli.build_profile_merge._fold_profile_material`.
#: Stamped alongside the per-input digests as the backstop that keeps material
#: the per-input split does not enumerate from going unreported (see
#: :func:`_material_input_digests`).
MATERIAL_KEY = f"{MATERIAL_PREFIX}all"

#: How :data:`MATERIAL_KEY` is printed. Reached only when the material moved and
#: no single input can be named for it — material the split does not enumerate.
MATERIAL_DISPLAY = "files the profile points at"

#: The refusal headline for "the fingerprint could not be computed at all".
#: Pinned as a constant because ``up``, ``restart`` and ``status`` must all say
#: the same thing, and the spec names this exact phrase.
UNVERIFIABLE_MESSAGE = "cannot verify profile ↔ build consistency"

_AS_BUILT_REMEDY = (
    "Run `osprey build` to re-render build/ from the profile, "
    "or `osprey up --as-built` to start the existing build anyway."
)


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

    Called by ``osprey up`` before any container is touched, so the
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
            # Printed verbatim. It is the manifest's own account of how this
            # render is reproduced, and no suffix is appended to it: a repo
            # render is unconditional, so there is no force flag to add.
            message += f" Re-render it with:\n    {remedy}\nthen re-run osprey up."
        logger.warning(message)
    except Exception as exc:  # advisory must never break a deploy
        logger.debug(f"Project staleness advisory skipped: {exc}")


# ---------------------------------------------------------------------------
# Deployment-repo drift verdict
# ---------------------------------------------------------------------------


class DriftState(StrEnum):
    """What a deployment repo's ``build/`` can be said to be, versus its source.

    ``CLEAN`` and ``DRIFT`` are verdicts; ``UNRESOLVABLE`` and ``NO_BUILD`` are
    the two honest ways of having no verdict, kept apart because their remedies
    are opposites: ``--as-built`` is the escape from an unverifiable profile,
    and is no escape at all when there is nothing built to start.
    """

    #: Recomputed fingerprint equals the stamp — ``build/`` is what the profile
    #: currently says.
    CLEAN = "clean"
    #: Fingerprints differ: the working tree has moved on since the render.
    DRIFT = "drift"
    #: The comparison could not be made — no profile, an unparseable one, or a
    #: build carrying no fingerprint.
    UNRESOLVABLE = "unresolvable"
    #: No ``build/`` (or no manifest inside it) to compare against.
    NO_BUILD = "no_build"


@dataclass(frozen=True)
class VersionSkew:
    """The installed framework differs from the one that rendered ``build/``.

    Always a warning, never a refusal: a patch bump must not stop an operator
    from starting the stack they already have.
    """

    stamped: str
    installed: str

    @property
    def message(self) -> str:
        return (
            f"build/ was rendered by osprey {self.stamped}; installed osprey is "
            f"{self.installed}. Re-run `osprey build` to pick up framework changes."
        )


@dataclass(frozen=True)
class ProfileFingerprint:
    """A profile's content hash plus the per-key digests that explain a diff.

    ``profile_hash`` is the verdict — the existing
    :func:`~osprey.cli.build_profile_merge.compute_profile_hash` value, byte for
    byte the same thing the build stamps. ``key_digests`` is commentary: it
    exists so a refusal can say *which* part moved, and nothing decides drift by
    reading it.
    """

    profile_hash: str
    key_digests: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftReport:
    """One drift verdict, shaped for every consumer of it.

    ``up``/``restart`` refuse when :attr:`refuses`; ``status`` prints
    :attr:`status_line` plus the version line; ``chat``/``query`` log
    :attr:`message` as a warning and continue.

    Attributes:
        state: The verdict.
        message: What was found, as a sentence — no remedy attached, because
            the same finding is a refusal in one verb and a warning in another.
        remedy: The suggested next step, or ``""`` when there is nothing to
            suggest.
        changed_keys: What moved, named as the operator would spell it: the
            top-level profile keys, and the paths of the file inputs the profile
            points at (``data/``, a convention directory, ``triggers.yml``, a
            ``personas/`` delta). Empty on a ``DRIFT`` whose build stamped no
            per-key digests — the verdict still stands, only the detail is
            missing.
        version_skew: Present whenever installed and stamped versions differ,
            independent of :attr:`state`.
        profile_path: The profile the comparison used, when there was one.
        build_dir: The output zone the comparison read.
        stamped_hash: Fingerprint recorded at build time, when present.
        current_hash: Fingerprint recomputed now, when it could be computed.
    """

    state: DriftState
    message: str
    remedy: str = ""
    changed_keys: tuple[str, ...] = ()
    version_skew: VersionSkew | None = None
    profile_path: Path | None = None
    build_dir: Path | None = None
    stamped_hash: str | None = None
    current_hash: str | None = None

    @property
    def refuses(self) -> bool:
        """Whether a start verb must stop unless the operator passed ``--as-built``."""
        return self.state in (DriftState.DRIFT, DriftState.UNRESOLVABLE)

    @property
    def status_line(self) -> str:
        """One line for ``osprey status``."""
        if self.state is DriftState.CLEAN:
            return "build: in sync with profile.yml"
        if self.state is DriftState.DRIFT:
            detail = f" ({', '.join(self.changed_keys)})" if self.changed_keys else ""
            return (
                "build: OUT OF DATE — profile.yml or a file it points at has changed "
                f"since the last build{detail}"
            )
        if self.state is DriftState.NO_BUILD:
            return "build: none — run `osprey build`"
        return f"build: unverifiable — {UNVERIFIABLE_MESSAGE}"


def build_manifest_path(repo_root: Path) -> Path:
    """Where a deployment repo's build provenance lives.

    ``build/`` is the rendered project itself, not a directory of them, so the
    manifest is at its root. One rule, deliberately: a search would let a stray
    nested render answer for the repo.
    """
    return Path(repo_root) / BUILD_DIRNAME / _MANIFEST_FILENAME


def _material_input_digests(resolved: dict[str, Any], profile_dir: Path) -> dict[str, str]:
    """One digest per *named* file input, so a refusal can point at a path.

    :data:`MATERIAL_KEY` alone can only say "something the profile points at
    moved". That is what made the refusal misleading: a persona delta edit was
    reported under the label ``data/``, which the operator had not touched.
    Splitting the fold per input lets the message name ``personas/readonly.yml``
    — the file that actually changed.

    Each input is folded with the same label and the same
    :func:`~osprey.cli.build_profile_merge._fold_source_tree` call the build's
    material fold uses, into its own hash rather than a shared one, so the value
    here answers "did *this* input move" independently.

    The enumeration mirrors
    :func:`~osprey.cli.build_profile_merge._fold_profile_material` — plus the
    selected host overlay, which the hash folds one level up in
    :func:`~osprey.cli.build_profile_merge._hash_resolved_profile` — and is
    therefore a second copy of it. That copy is deliberately diagnostic-only:
    nothing decides drift by reading these keys, and :data:`MATERIAL_KEY` —
    computed by the fold itself, not by this function — still covers whatever
    this enumeration misses. An input the build learns to fold and this does not
    is reported as :data:`MATERIAL_DISPLAY` rather than going unmentioned.

    Args:
        resolved: The profile after ``extends`` (and persona-delta) resolution.
        profile_dir: The profile root every relative path anchors at.

    Returns:
        Digests keyed by :data:`MATERIAL_PREFIX` plus the input's path as the
        operator would spell it. Inputs the profile does not have are absent,
        so creating or deleting one reads as a change to that key.
    """
    import hashlib

    from osprey.cli.build_profile_merge import _fold_source_tree
    from osprey.cli.build_profile_model import BuildProfile
    from osprey.cli.profile_conventions import CONVENTION_SOURCES
    from osprey.cli.profile_root import PERSONA_DIRNAME
    from osprey.cli.variant_selection import VARIANT_DIRNAME, active_variant_overlay

    digests: dict[str, str] = {}

    def fold(key: str, label: str, source: Path) -> None:
        digest = hashlib.sha256()
        _fold_source_tree(digest, label, source)
        digests[f"{MATERIAL_PREFIX}{key}"] = digest.hexdigest()

    data_root = BuildProfile(name="", data=resolved.get("data")).resolved_data_root(profile_dir)
    if data_root is not None:
        fold("data/", "data", data_root)

    for source in sorted(CONVENTION_SOURCES):
        candidate = profile_dir / source
        if candidate.exists():
            fold(f"{source}/", f"convention:{source}", candidate)

    triggers = profile_dir / "triggers.yml"
    if triggers.is_file():
        fold("triggers.yml", "triggers", triggers)

    # The host variant this repo builds, keyed by the overlay's own path so a
    # switch between two variants reads as one key gone and another arrived —
    # which is what happened. Only the SELECTED overlay is digested, matching
    # what `compute_profile_hash` folds: an overlay for another host is not an
    # input to this build and must not be reported as one.
    overlay = active_variant_overlay(profile_dir)
    if overlay is not None:
        fold(f"{VARIANT_DIRNAME}/{overlay.name}", "variant", overlay)

    # Per delta rather than per directory: ``personas/`` is flat and each child
    # IS a build input of its own, so naming the one file that moved costs
    # nothing and is what the operator edited.
    persona_dir = profile_dir / PERSONA_DIRNAME
    if persona_dir.is_dir():
        for delta in sorted(
            (
                entry
                for entry in persona_dir.iterdir()
                if entry.is_file() and not entry.name.startswith(".")
            ),
            key=lambda entry: entry.name,
        ):
            fold(f"{PERSONA_DIRNAME}/{delta.name}", "persona", delta)

    return digests


def profile_fingerprint(profile_path: Path) -> ProfileFingerprint | None:
    """Fingerprint a profile, or ``None`` when it cannot be resolved.

    ``None`` means exactly one thing: the profile does not parse or does not
    resolve (a broken ``extends``, a non-mapping document, an unreadable file).
    Callers turn that into :attr:`DriftState.UNRESOLVABLE` rather than into a
    clean bill of health.

    ``osprey build`` persists :attr:`ProfileFingerprint.key_digests` under
    ``creation`` → :data:`KEY_DIGESTS_MANIFEST_KEY`; :func:`check_drift` reads
    it back to name what changed.

    Args:
        profile_path: The repo's root ``profile.yml``.
    """
    import hashlib

    from osprey.cli.build_profile import compute_profile_hash

    profile_path = Path(profile_path)
    profile_hash = compute_profile_hash(profile_path)
    if profile_hash is None:
        return None

    digests: dict[str, str] = {}
    try:
        from osprey.cli.build_profile_document import _read_profile_document
        from osprey.cli.build_profile_merge import (
            _fold_profile_material,
            resolve_profile_document,
        )

        raw = _read_profile_document(profile_path)
        document = resolve_profile_document(raw, profile_path, warn=False)
        for key, value in document.raw.items():
            canonical = json.dumps(value, sort_keys=True, default=str)
            digests[str(key)] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        material = hashlib.sha256()
        _fold_profile_material(material, document.raw, document.root_dir)
        digests[MATERIAL_KEY] = material.hexdigest()
        digests.update(_material_input_digests(document.raw, document.root_dir))
    except Exception as exc:
        # The hash resolved, so the profile is fine; only the commentary failed.
        # Losing the "what changed" detail must not turn a computable verdict
        # into an unverifiable one.
        logger.debug(f"Profile key digests unavailable: {exc}")
        digests = {}

    return ProfileFingerprint(profile_hash=profile_hash, key_digests=digests)


def _is_material(name: str) -> bool:
    """Whether a digest key names file material rather than a profile key."""
    return name.startswith(MATERIAL_PREFIX)


def _display_key(name: str) -> str:
    """The key as the operator would recognize it, not as the digest stores it.

    Profile keys are digested by canonical field name, which is what keeps a
    pure change of YAML-surface spelling from reading as drift. That canonical
    name is the wrong thing to *print*: an operator who edited ``app_template:``
    would be sent looking for a ``data_bundle:`` key their file does not
    contain. Material keys carry :data:`MATERIAL_PREFIX`, an internal marker
    that has no business in a message; underneath it they are already the paths
    the operator edited.
    """
    from osprey.cli.build_profile_document import _YAML_TO_FIELD

    if name == MATERIAL_KEY:
        return MATERIAL_DISPLAY
    if _is_material(name):
        return name[len(MATERIAL_PREFIX) :]

    for yaml_key, field_name in _YAML_TO_FIELD.items():
        if name == field_name:
            return yaml_key
    return name


def _changed_keys(current: dict[str, str], stamped: Any) -> tuple[str, ...]:
    """Names of the keys whose digests disagree, sorted; empty when unknowable.

    An empty stamp is unknowable, not "everything changed". A build that
    recorded no digests — or recorded something that is not a mapping — supports
    no comparison at all, and diffing against it would name every key the
    profile has, which reads as a total rewrite of a profile nobody touched.
    The caller turns the empty answer into the honest "this build recorded no
    detail about which".
    """
    if not isinstance(stamped, dict) or not stamped or not current:
        return ()

    # Over the union, so a key the operator added or deleted is named too — the
    # absence of a key is as much a change as a different value under it.
    names = {key for key in set(current) | set(stamped) if current.get(key) != stamped.get(key)}
    if MATERIAL_KEY in names and any(_is_material(name) for name in names - {MATERIAL_KEY}):
        # A named input already accounts for the material having moved; saying
        # "files the profile points at, personas/readonly.yml" would report the
        # same edit twice, once vaguely. The combined key survives only when it
        # moved and no input explains it — material this module does not
        # enumerate, which is the one case worth admitting to.
        names.discard(MATERIAL_KEY)
    return tuple(sorted(_display_key(name) for name in names))


def _version_skew(stamped: Any) -> VersionSkew | None:
    installed = _installed_version()
    if not isinstance(stamped, str) or not stamped:
        return None
    if "unknown" in (stamped, installed) or stamped == installed:
        return None
    return VersionSkew(stamped=stamped, installed=installed)


def check_drift(repo_root: Path) -> DriftReport:
    """Decide whether a deployment repo's ``build/`` still matches its source.

    The comparison recomputes the profile fingerprint now and holds it against
    the one ``osprey build`` stamped into the build manifest. Both sides come
    from the same machinery, so the answer is about content, not about
    timestamps, file order, or comments.

    Unlike the legacy advisory in this module, this does **not** fail open. A
    profile that no longer resolves, and a build that carries no fingerprint,
    are both ``UNRESOLVABLE``: silently starting a stack nobody can vouch for
    is the failure mode the refusal exists to prevent. ``--as-built`` remains
    the operator's way through, since ``build/`` is self-contained.

    Args:
        repo_root: The deployment repo root — the directory holding
            ``profile.yml``, ``build/`` and ``var/``.

    Returns:
        A report every lifecycle verb can act on; never raises.
    """
    from osprey.cli.repo_resolver import PROFILE_FILENAME

    repo_root = Path(repo_root)
    profile_path = repo_root / PROFILE_FILENAME
    build_dir = repo_root / BUILD_DIRNAME

    manifest = _load_manifest(build_dir)
    if manifest is None:
        return DriftReport(
            state=DriftState.NO_BUILD,
            message=f"No build found in {build_dir}.",
            remedy="Run `osprey build`.",
            build_dir=build_dir,
        )

    creation = manifest.get("creation")
    if not isinstance(creation, dict):
        creation = {}
    skew = _version_skew(creation.get("osprey_version"))
    stamped_hash = creation.get("preset_hash")

    def report(
        state: DriftState,
        message: str,
        *,
        remedy: str = "",
        changed_keys: tuple[str, ...] = (),
        current_hash: str | None = None,
    ) -> DriftReport:
        """Every exit below carries the same provenance; only the verdict differs."""
        return DriftReport(
            state=state,
            message=message,
            remedy=remedy,
            changed_keys=changed_keys,
            version_skew=skew,
            profile_path=profile_path,
            build_dir=build_dir,
            stamped_hash=stamped_hash if isinstance(stamped_hash, str) else None,
            current_hash=current_hash,
        )

    if not isinstance(stamped_hash, str) or not stamped_hash:
        return report(
            DriftState.UNRESOLVABLE,
            f"{UNVERIFIABLE_MESSAGE}: this build carries no profile fingerprint, "
            "so there is nothing to compare the profile against.",
            remedy=_AS_BUILT_REMEDY,
        )

    if not profile_path.is_file():
        return report(
            DriftState.UNRESOLVABLE,
            f"{UNVERIFIABLE_MESSAGE}: no {PROFILE_FILENAME} at {repo_root}.",
            remedy=_AS_BUILT_REMEDY,
        )

    fingerprint = profile_fingerprint(profile_path)
    if fingerprint is None:
        return report(
            DriftState.UNRESOLVABLE,
            f"{UNVERIFIABLE_MESSAGE}: {profile_path} could not be read or resolved.",
            remedy=(
                f"Fix {PROFILE_FILENAME} (`osprey validate` reports what is wrong), "
                "or run `osprey up --as-built` to start the existing build anyway."
            ),
        )

    if fingerprint.profile_hash == stamped_hash:
        return report(
            DriftState.CLEAN,
            f"build/ matches {PROFILE_FILENAME}.",
            current_hash=fingerprint.profile_hash,
        )

    changed = _changed_keys(fingerprint.key_digests, creation.get(KEY_DIGESTS_MANIFEST_KEY))
    if changed:
        detail = f" Changed: {', '.join(changed)}."
    else:
        # No stamped digests to diff against. Say so rather than inventing a key
        # list — the headline already names both places the change can be.
        detail = " This build recorded no detail about which."
    return report(
        DriftState.DRIFT,
        f"{PROFILE_FILENAME} or a file it points at has changed since build/ was "
        f"rendered, so starting it would deploy something other than what the "
        f"profile now describes.{detail}",
        remedy=_AS_BUILT_REMEDY,
        changed_keys=changed,
        current_hash=fingerprint.profile_hash,
    )
