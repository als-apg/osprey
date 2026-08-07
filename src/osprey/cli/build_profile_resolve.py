"""Multi-source profile resolution: preset / file + overlays + ``--set``.

The entry point the ``build`` command actually calls. Picks the base layer
(bundled preset or on-disk file), deep-merges override files and ``--set``
values over it, then hands the assembled raw dict to ``extends`` resolution and
:func:`osprey.cli.build_profile_load._parse_profile`. Also owns the ``--set``
mini-parser.

Every build reads a profile directory, so this module also owns the two ways a
CLI invocation gets one: :func:`materialize_or_reuse_profile`, which gives a
``--preset`` build the profile it builds from, and
:func:`write_back_cli_overrides`, which turns ``--set`` / ``-O`` / ``--tier`` on
a build from an existing profile into an edit of that profile. The profile is
the source of truth, so an explicit override *is* a profile edit.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import click
import yaml

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger

from .build_profile_document import _normalize_profile_aliases, _read_profile_document
from .build_profile_load import LoadedProfile, _parse_profile
from .build_profile_merge import _deep_merge, _resolve_extends, resolve_profile_document
from .build_profile_model import BuildProfile
from .build_profile_presets import _load_preset_raw, _normalize_preset_name

logger = get_logger("build")

#: Refusal shared by the two places CLI layers reach a materialized profile —
#: `osprey profile new` baking them in, and a build writing them back. One
#: constant because the rule is one rule: a materialized profile is standalone,
#: so nothing may give it an `extends:` parent. Two spellings of it would mean
#: the same override file is refused on the build that materializes and accepted
#: on the next one.
EXTENDS_OVERRIDE_REFUSAL = (
    "Cannot override 'extends' — a materialized profile is standalone and "
    "inherits nothing at build time."
)


def _parse_set_pairs(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``--set KEY.PATH=VALUE`` pairs into a nested dict.

    The right-hand side is parsed with ``yaml.safe_load`` so callers get
    type coercion for free: ``true``/``false`` -> bool, ``[a,b]`` -> list,
    bare ints/floats -> numeric, anything else -> string.
    """
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise BuildProfileError(f"--set expects KEY=VALUE (with '='), got: {pair!r}")
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        if not key:
            raise BuildProfileError(f"--set key must be non-empty: {pair!r}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as e:
            raise BuildProfileError(f"--set value for {key!r} is not valid YAML: {e}") from e
        target: dict[str, Any] = result
        parts = key.split(".")
        for part in parts[:-1]:
            existing = target.get(part)
            if existing is None:
                existing = {}
                target[part] = existing
            elif not isinstance(existing, dict):
                raise BuildProfileError(
                    f"--set key {key!r} conflicts with earlier scalar at {part!r}"
                )
            target = existing
        target[parts[-1]] = value
    # ``--set`` is its own authored layer, so it normalizes like a document —
    # but the value loads above are scalar, not document, reads.
    return _normalize_profile_aliases(result, "--set")


def merge_cli_overrides(
    base: dict[str, Any],
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
) -> dict[str, Any]:
    """Layer ``-O`` override files and ``--set`` pairs over ``base``.

    The shared CLI layering step: override files deep-merge in declaration
    order, then ``--set`` values merge on top. Used by the project-render path
    (:func:`resolve_build_profile`) and by ``osprey profile new``, which bakes
    the merged result into the materialized ``profile.yml``.
    """
    raw = base
    for override_path in overrides:
        if not override_path.exists():
            raise BuildProfileError(f"Override not found: {override_path}")
        override_raw = _read_profile_document(override_path)
        if override_raw is None:
            continue
        if not isinstance(override_raw, dict):
            raise BuildProfileError(f"Override must be a YAML mapping: {override_path}")
        raw = _deep_merge(raw, override_raw)

    if set_pairs:
        raw = _deep_merge(raw, _parse_set_pairs(set_pairs))
    return raw


def resolve_build_profile(
    profile_path: Path | None,
    preset: str | None,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
) -> tuple[BuildProfile, Path]:
    """The two fields most callers need from :func:`resolve_build_document`.

    Answers "what does this profile say, and where does it anchor". A caller
    that must also honor what resolution *derived* — which convention artifacts
    the profile excludes — wants :func:`resolve_build_document` instead; the
    build does, because an excluded artifact it copies anyway would shadow the
    framework's own version of that file.

    Returns:
        ``(profile, profile_dir)``. ``profile_dir`` is the profile ROOT — where
        every profile-relative path anchors, and what
        :meth:`BuildProfile.validate` resolves overlay/services lookups
        against. For a persona delta that is the directory above ``personas/``,
        never the delta's own parent; for preset mode it is the bundled
        ``profiles/presets/`` package directory.

    Raises:
        BuildProfileError: Whatever :func:`resolve_build_document` raises.
    """
    document = resolve_build_document(profile_path, preset, overrides, set_pairs)
    return document.profile, document.profile_dir


def resolve_build_document(
    profile_path: Path | None,
    preset: str | None,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
) -> LoadedProfile:
    """Resolve a build profile from any combination of preset / file / overlays.

    Mode is determined by which of ``profile_path`` and ``preset`` is given;
    they are mutually exclusive and exactly one is required.

    Layers are applied in order: base -> override file(s) -> --set values.
    All layers are merged via :func:`_deep_merge` (string lists union-dedup,
    other lists concatenate) before ``extends:`` is resolved.

    The multi-source counterpart of
    :func:`~osprey.cli.build_profile_load.load_profile_document`, and it returns
    the same record for the same reason: resolution knows two things the parsed
    :class:`BuildProfile` does not — the profile ROOT, and the convention
    artifacts the profile excludes — and both are things the build must act on.
    Returning the record rather than a widening tuple is what keeps the next
    thing resolution learns from breaking every callsite.

    Returns:
        The parsed, validated profile with its root and exclusion record.

    Raises:
        BuildProfileError: For mutual-exclusion violations, missing files,
        invalid YAML, a ``data:`` tree in preset mode, or validation failures.
    """
    if profile_path is not None and preset is not None:
        raise BuildProfileError("Pass either a profile path or --preset, not both.")
    if profile_path is None and preset is None:
        raise BuildProfileError("Either a profile path or --preset is required.")

    # A preset is one file in a shared package directory: it is never a persona
    # delta and carries no convention material to exclude (compute_preset_hash
    # folds none either, for the same reason).
    is_persona_delta = False
    excluded_artifacts: frozenset[str] = frozenset()

    if preset is not None:
        raw, base_anchor = _load_preset_raw(preset)
        profile_dir = base_anchor.parent
        raw = merge_cli_overrides(raw, overrides, set_pairs)
        raw = _resolve_extends(raw, base_anchor)

        # Checked after extends resolution so no injection path escapes: the
        # preset itself, a -O file, a --set pair, or an extends parent. A preset
        # has no profile directory to anchor a data tree against (profile_dir is
        # the bundled package dir), so carrying one is always a mistake.
        if raw.get("data") is not None:
            raise BuildProfileError(
                f"Profile key 'data' is not supported with --preset (got {raw['data']!r}). "
                f"A preset carries no profile directory to resolve the data tree against. "
                f"Materialize the preset first — 'osprey profile new DIR --preset {preset}' — "
                f"then build from that directory."
            )
    else:
        assert profile_path is not None  # narrows for type-checkers
        if not profile_path.exists():
            raise BuildProfileError(f"Profile not found: {profile_path}")
        raw = _read_profile_document(profile_path)
        if not isinstance(raw, dict):
            raise BuildProfileError(f"Profile must be a YAML mapping, got {type(raw).__name__}")
        raw = merge_cli_overrides(raw, overrides, set_pairs)
        # Resolution goes through the one call that decides what a profile file
        # *means* — the same one the loader and the content hash make. A file
        # under `personas/` is a delta merged over the `profile.yml` beside it
        # and anchors at that root; resolving `extends` here instead, against
        # the file's own parent, would build a hollow project from the delta
        # alone and read its data tree from `personas/`.
        document = resolve_profile_document(raw, profile_path.resolve())
        raw, profile_dir = document.raw, document.root_dir
        is_persona_delta = document.is_persona_delta
        excluded_artifacts = document.excluded_artifacts

    profile = _parse_profile(raw)
    profile.validate(profile_dir)
    return LoadedProfile(
        profile=profile,
        profile_dir=profile_dir,
        is_persona_delta=is_persona_delta,
        excluded_artifacts=excluded_artifacts,
    )


# ---------------------------------------------------------------------------
# The profile a build reads
# ---------------------------------------------------------------------------

#: Directory-name suffix a ``--preset`` build materializes its profile under,
#: beside the project it builds. One constant so the create branch and the
#: reuse branch cannot name different directories.
PROFILE_DIR_SUFFIX = "-profile"

#: The profile file inside a materialized profile directory.
PROFILE_FILENAME = "profile.yml"


def preset_profile_dir(output_dir: Path, project_name: str) -> Path:
    """Where a ``--preset`` build keeps the profile it builds from."""
    return output_dir / f"{project_name}{PROFILE_DIR_SUFFIX}"


def materialize_or_reuse_profile(
    preset: str,
    output_dir: Path,
    project_name: str,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
    tier: int | None = None,
) -> Path:
    """Give a ``--preset`` build the profile directory it builds from.

    Materialized on first use through the one materialization path ``osprey
    profile new`` uses; reused verbatim afterwards, because from then on the
    profile — not the preset — is what the project is built from. A preset that
    has moved on since is *reported*, never re-applied, and a build naming a
    *different* preset is refused outright (see
    :func:`_check_preset_provenance`). ``--force`` on the build wipes the
    project, never this directory: a profile is replaced only by ``osprey
    profile new --force``.

    The CLI overrides reach the profile either way, which is what makes the
    profile a complete description of the project: baked in as layers at
    materialization, written into the file on a reuse.

    Args:
        preset: Bundled preset name, in either spelling.
        output_dir: Where the project is being built; the profile is its
            sibling.
        project_name: Names the profile directory (``<name>-profile``).
        overrides: ``-O`` files.
        set_pairs: ``--set`` pairs.
        tier: ``--tier``, the profile's ``tier:`` key.

    Returns:
        Path to the profile's ``profile.yml``.

    Raises:
        click.UsageError: For anything the caller could have got wrong — an
            unknown preset, or layers that produce an invalid profile.
    """
    # Imported here: the materialization verb lives with the `profile` command
    # group, which imports this module.
    from .profile_cmd import _materialize_profile_directory

    target = preset_profile_dir(output_dir, project_name)
    profile_path = target / PROFILE_FILENAME

    if profile_path.is_file():
        logger.info("  Profile: %s (reused)", target)
        _check_preset_provenance(profile_path, preset)
        write_back_cli_overrides(profile_path, overrides, set_pairs, tier)
        return profile_path

    logger.info("  Profile: %s (materializing from preset '%s')", target, preset)
    if tier is not None:
        set_pairs = (*set_pairs, f"tier={tier}")
    try:
        _materialize_profile_directory(target, preset, overrides, set_pairs)
    except click.UsageError as e:
        raise _rewrite_already_exists(e, target, preset) from e
    return profile_path


def _rewrite_already_exists(error: click.UsageError, target: Path, preset: str) -> click.UsageError:
    """Give the "directory is in the way" refusal build-appropriate remediation.

    ``osprey profile new`` tells the user to re-run *itself* with ``--force``,
    which is right there and wrong here: a build reaches this only when a
    ``<name>-profile/`` directory exists without a ``profile.yml`` in it (an
    interrupted materialization), and ``osprey build --force`` replaces the
    *project*, never the profile. Repeating that advice sends the user round a
    loop that cannot terminate, so the message names the two things that do
    clear it.

    Anything else is that command's own refusal and is passed through unchanged.
    """
    from .profile_cmd import _ALREADY_EXISTS

    if str(error) != _ALREADY_EXISTS.format(target=target.resolve()):
        return error
    return click.UsageError(
        f"{target} already exists but holds no {PROFILE_FILENAME} — an interrupted "
        f"materialization leaves that behind. `osprey build --force` replaces the "
        f"project, never the profile, so it cannot clear this. Delete {target}, or "
        f"materialize over it: osprey profile new {target} --preset {preset} --force"
    )


def _check_preset_provenance(profile_path: Path, preset: str) -> None:
    """Vet a reused profile against the ``--preset`` the build named.

    Two questions of the profile's ``provenance`` block, in the order that
    matters. *Which preset is this?* — a different one than the build asked for
    is a hard error (:func:`_require_matching_preset`), because the profile is
    built verbatim and would quietly produce the other project. *Has that preset
    moved on?* — advisory only (:func:`_report_preset_drift`).

    Silent when the profile carries no provenance at all: hand-written and
    not-preset-derived profiles answer neither question, and "cannot tell" must
    never read as an answer.
    """
    try:
        raw = _read_profile_document(profile_path)
    except BuildProfileError:
        # An unreadable profile is about to fail the build with a much better
        # message than either check could give.
        return
    provenance = raw.get("provenance") if isinstance(raw, dict) else None
    if not isinstance(provenance, dict):
        return
    _require_matching_preset(profile_path, preset, provenance)
    _report_preset_drift(profile_path, preset, provenance)


def _require_matching_preset(profile_path: Path, preset: str, provenance: dict[str, Any]) -> None:
    """Refuse a ``--preset`` that is not the one this profile was made from.

    A reused profile is built verbatim, so the preset named on the command line
    selects nothing — accepting a different one would build the *stored*
    preset's project while every artifact that records the build (the manifest's
    ``preset``, its reproducible command) named the requested one. The two
    presets are named in the message because which one the user meant is
    exactly what the CLI cannot know.
    """
    stored = provenance.get("preset")
    if not isinstance(stored, str) or not stored:
        return
    requested = _normalize_preset_name(preset)
    if _normalize_preset_name(stored) == requested:
        return
    raise click.UsageError(
        f"{profile_path} was materialized from preset {_normalize_preset_name(stored)!r}, "
        f"but this build asks for {requested!r}. The profile is the source of truth and "
        f"is never re-materialized, so building it now would build the "
        f"{_normalize_preset_name(stored)!r} project under a {requested!r} label. "
        f"Build {requested!r} under a different project name so it gets a profile of "
        f"its own, or remove {profile_path.parent} and rebuild."
    )


def profile_provenance_preset(profile_path: Path) -> str | None:
    """The normalized preset a profile records having been materialized from.

    ``None`` when the profile is unreadable or carries no such record. What the
    build stamps into the manifest: the profile is what was built, so the preset
    it came from is the honest answer even where the CLI named one too.
    """
    try:
        raw = _read_profile_document(profile_path)
    except BuildProfileError:
        return None
    provenance = raw.get("provenance") if isinstance(raw, dict) else None
    stored = provenance.get("preset") if isinstance(provenance, dict) else None
    if not isinstance(stored, str) or not stored:
        return None
    return _normalize_preset_name(stored)


def _report_preset_drift(profile_path: Path, preset: str, provenance: dict[str, Any]) -> None:
    """Advise when the bundled preset moved on since the profile was written.

    Advisory only, and deliberately narrow: the comparison is between the
    ``provenance.preset_hash`` the materialization stamped and the installed
    preset's hash today, both of which cover the preset's resolved YAML and
    nothing else — not the packaged data tree the profile copied, and not the
    framework artifacts it selects. So this can say the preset's *settings*
    changed; it cannot say the profile is otherwise current.

    Silent when either hash is unavailable: "cannot compare" must never read as
    drift. Only reached once :func:`_require_matching_preset` has established
    that the two hashes are of the same preset.
    """
    from .build_profile_merge import compute_preset_hash

    stored = provenance.get("preset_hash")
    current = compute_preset_hash(preset)
    if not stored or not current or stored == current:
        return

    logger.warning(
        "  ! Preset '%s' has changed since %s was materialized from it "
        "(YAML settings only — this cannot see the data tree or framework artifacts).",
        preset,
        profile_path,
    )
    logger.warning(
        "      The profile is the source of truth and is being built verbatim; "
        "nothing from the preset is re-applied."
    )
    logger.warning(
        "      To see what moved, materialize the current preset elsewhere: "
        "osprey profile new <DIR> --preset %s",
        preset,
    )


def write_back_cli_overrides(
    profile_path: Path,
    overrides: tuple[Path, ...] = (),
    set_pairs: tuple[str, ...] = (),
    tier: int | None = None,
) -> list[str]:
    """Write ``-O`` / ``--set`` / ``--tier`` into the profile, before building.

    The profile is the source of truth, so an explicit override on a build from
    an existing profile is an edit *of that profile* — made here, announced, and
    then read back by the ordinary resolution path like any other profile
    content. Nothing is layered at build time and thrown away afterwards.

    The edit **replaces** the value at each dotted key path. That is a
    deliberate change from the old build-time layering, where a list-valued
    ``--set`` union-deduped with what the profile already had: a value written
    into a file has to be the value the file then holds, or the profile stops
    describing the project. (A first materialization still bakes its layers in
    through :func:`merge_cli_overrides`, so ``--set`` on a *fresh* preset build
    keeps layering semantics.)

    ``config:`` is written the way a profile spells it — one mapping key holding
    the whole dotted path (``control_system.type``) rather than a nested map, so
    a write-back addresses the same rendered-config leaf the profile's own
    entries do instead of wholesale-replacing a config subtree.

    Args:
        profile_path: The ``profile.yml`` (or persona delta) being built from.
        overrides: ``-O`` files, deep-merged in declaration order.
        set_pairs: ``--set`` pairs, merged on top of the ``-O`` layers.
        tier: ``--tier``, written as the profile's ``tier:`` key.

    Returns:
        The dotted key paths written, in write order; empty when there was
        nothing to write.

    Raises:
        BuildProfileError: If an override file is missing, unreadable, or not a
            YAML mapping, or a ``--set`` pair is malformed.
        click.UsageError: If a layer sets ``extends``, which a materialized
            profile cannot have — the same refusal materialization makes, so the
            same override file is answered the same way on every build.
    """
    # Layered against an empty base so only what the CLI supplied is written —
    # the profile's own content is never rewritten as a side effect.
    layer = merge_cli_overrides({}, overrides, set_pairs)
    if tier is not None:
        layer["tier"] = tier
    if "extends" in layer:
        raise click.UsageError(EXTENDS_OVERRIDE_REFUSAL)
    if not layer:
        return []

    updates = _flatten_override_layer(layer)
    _write_profile_values(profile_path, updates)
    written = [".".join(key_path) for key_path, _ in updates]
    logger.info(
        "  ✓ Wrote %d override(s) into %s: %s",
        len(written),
        profile_path,
        ", ".join(written),
    )
    return written


def _flatten_override_layer(
    layer: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[list[str], Any]]:
    """Flatten a CLI override layer into ``(key_path, value)`` leaf writes.

    Descends mappings so an override touches only the leaf it names; scalars,
    lists and empty mappings are leaves. ``config:`` is the one block that does
    not nest — its keys are dotted paths into the *rendered* config, held as
    single mapping keys — so its interior is flattened into one such key rather
    than into further levels.
    """
    flat: list[tuple[list[str], Any]] = []
    for key, value in layer.items():
        key_path = (*prefix, str(key))
        if not isinstance(value, dict) or not value:
            flat.append((list(key_path), value))
        elif key_path == ("config",):
            flat.extend(
                (["config", ".".join(sub_path)], leaf) for sub_path, leaf in _dotted_leaves(value)
            )
        else:
            flat.extend(_flatten_override_layer(value, key_path))
    return flat


def _dotted_leaves(
    mapping: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    """Every leaf of ``mapping`` as a ``(path_segments, value)`` pair."""
    leaves: list[tuple[tuple[str, ...], Any]] = []
    for key, value in mapping.items():
        path = (*prefix, str(key))
        if isinstance(value, dict) and value:
            leaves.extend(_dotted_leaves(value, path))
        else:
            leaves.append((path, value))
    return leaves


def _write_profile_values(profile_path: Path, updates: list[tuple[list[str], Any]]) -> None:
    """Set each ``key_path`` in ``profile_path`` to its value, keeping comments.

    Uses the shared round-trip YAML handle rather than a private one: the
    profile is a hand-edited, heavily commented document, and a second handle
    with a different line width would silently re-wrap the whole file on the
    first write-back.

    Rendered to text first and then written by :func:`_atomic_write_bytes`, so
    the document either lands whole or not at all: this file is the facility's
    source of truth, and a truncate-in-place write interrupted halfway would
    leave it neither the profile it was nor the one it was becoming.
    """
    from ruamel.yaml import CommentedMap

    from osprey.utils.config_writer import _load, _yaml

    data = _load(profile_path)
    for key_path, value in updates:
        node = data
        for segment in key_path[:-1]:
            if not isinstance(node.get(segment), dict):
                node[segment] = CommentedMap()
            node = node[segment]
        node[key_path[-1]] = value
    rendered = io.StringIO()
    _yaml.dump(data, rendered)
    _atomic_write_bytes(profile_path, rendered.getvalue().encode("utf-8"))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload`` via a same-directory temp file.

    The same shape as the manifest and ``.env`` writers this feature added, for
    the same reason: a reader (or a crash) never sees a half-written file, and
    the previous contents survive any failure before the ``os.replace``. The
    existing file's mode is carried over so an atomic rewrite does not quietly
    re-permission the profile.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        if path.exists():
            os.chmod(tmp_name, path.stat().st_mode & 0o7777)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class ProfileWriteBackGuard:
    """Undo a build's write-back into a profile when that build then fails.

    ``--set`` / ``-O`` / ``--tier`` are written into the profile *before* the
    build resolves and validates it, because the profile is what the build then
    reads. Without this, a value the profile cannot hold — ``--set
    provider=nonexistent``, or a mistyped model — would stay in the file after
    the build that introduced it failed, and so fail every later flag-free build
    too, with a message that never mentions the edit: a typo turns into a stuck
    project, recoverable only by hand-editing the profile. The edit is therefore
    part of the build: it survives a build that completes and is undone by one
    that does not.

    Restores the exact bytes, so a profile that a build did not change is left
    untouched (down to formatting) and one it did change is put back verbatim.
    A restore that itself fails is reported rather than raised: the build's own
    failure is the news, and burying it under an I/O error would cost the
    operator the reason they are looking for. The report names the file and the
    keys, which is what they need to repair it by hand.

    Constructed *before* the write-back, whichever branch makes it, and the two
    branches are symmetric because the principle is one principle — a failed
    build leaves the facility as it found it. Where the profile already existed
    the edit is rolled back; where *this invocation materialized it*
    (``materializes_into``, a directory that did not exist at entry) there is no
    earlier state to return to, so the directory goes: its ``profile.yml``
    carries the same bad override baked in as a layer, and leaving it behind
    turns the next flag-free build into the same stuck project by a different
    route. Only ever the directory this invocation created — one that existed in
    any form at entry is never touched, so an interrupted materialization still
    meets the refusal that names it (see :func:`_rewrite_already_exists`).
    """

    def __init__(
        self,
        profile_path: Path | None,
        overrides: tuple[Path, ...] = (),
        set_pairs: tuple[str, ...] = (),
        tier: int | None = None,
        *,
        materializes_into: Path | None = None,
    ) -> None:
        self._path = Path(profile_path) if profile_path is not None else None
        self._overrides = overrides
        self._set_pairs = set_pairs
        self._tier = tier
        self._snapshot: bytes | None = None
        if self._path is not None and self._path.is_file():
            try:
                self._snapshot = self._path.read_bytes()
            except OSError:
                # Unreadable now means the build is about to fail on it anyway.
                self._snapshot = None
        # Recorded at entry, which is the only moment "this invocation created
        # it" can be established: after materialization the two cases are
        # indistinguishable from the directory alone.
        self._created_dir: Path | None = None
        if materializes_into is not None and not materializes_into.exists():
            self._created_dir = materializes_into

    def rollback(self) -> None:
        """Undo what this build did to the profile: restore it, or remove it."""
        if self._created_dir is not None:
            self._remove_created_dir()
            return
        if self._path is None or self._snapshot is None:
            return
        try:
            current = self._path.read_bytes() if self._path.is_file() else None
            if current == self._snapshot:
                return
            _atomic_write_bytes(self._path, self._snapshot)
        except OSError as e:
            logger.error(
                "✗ Could not restore %s after the failed build (%s). It still holds "
                "this build's override(s): %s — remove or correct them before rebuilding.",
                self._path,
                e,
                self._describe_edit(),
            )
            return
        logger.warning(
            "  ! Build failed — restored %s to its pre-build contents; "
            "the override(s) it wrote (%s) were not kept.",
            self._path,
            self._describe_edit(),
        )

    def _remove_created_dir(self) -> None:
        """Drop the profile directory this failed build materialized.

        Mirrors :func:`_materialize_profile_directory`'s own cleanup of a
        materialization that fails partway: the difference is only *when* the
        build gave up, and an operator has no more use for a profile the build
        that made it could not use either.
        """
        import shutil

        assert self._created_dir is not None  # narrows for type-checkers
        if not self._created_dir.is_dir():
            return
        carried = self._describe_edit()
        try:
            shutil.rmtree(self._created_dir)
        except OSError as e:
            logger.error(
                "✗ Could not remove %s after the failed build (%s). It was materialized by "
                "this build and bakes in the override(s) it died on: %s — remove the "
                "directory, or correct them, before rebuilding.",
                self._created_dir,
                e,
                carried,
            )
            return
        logger.warning(
            "  ! Build failed — removing %s: it was materialized by this build and "
            "carries the override(s) the build died on (%s). Re-run without the bad "
            "flag to materialize cleanly.",
            self._created_dir,
            carried,
        )

    def _describe_edit(self) -> str:
        """The dotted key paths this invocation asked to write, for a message.

        Re-derived from the CLI arguments rather than reported by the write-back
        so the guard can also name them when the write-back is what failed.
        """
        try:
            layer = merge_cli_overrides({}, self._overrides, self._set_pairs)
        except BuildProfileError:
            layer = {}
        if self._tier is not None:
            layer["tier"] = self._tier
        keys = [".".join(key_path) for key_path, _ in _flatten_override_layer(layer)]
        return ", ".join(keys) if keys else "(none)"
