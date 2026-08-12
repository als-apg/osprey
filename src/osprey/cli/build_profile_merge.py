"""Profile inheritance merging and content hashing.

Resolves the ``extends`` chain into a single raw YAML dict — deep-merging each
layer, applying ``exclude:`` list subtraction, and detecting circular
references — and derives the canonical content hashes stamped into build
manifests. Hashing lives here rather than beside preset discovery because it
hashes the *resolved* profile, which requires ``extends`` resolution; keeping it
here leaves :mod:`osprey.cli.build_profile_presets` a leaf.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.errors import BuildProfileError

from .build_profile_document import _normalize_profile_aliases, _read_profile_document
from .build_profile_presets import _load_preset_raw, _preset_exists, list_presets
from .profile_root import PERSONA_DIRNAME, ROOT_PROFILE_FILENAME, resolve_profile_root

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile inheritance helpers
# ---------------------------------------------------------------------------


def _merge_lists(base: list, child: list) -> list:
    """Merge two YAML lists.

    String lists: union with dedup, base order preserved.
    Other lists (e.g. lifecycle step dicts): concatenate.
    """
    if not base and not child:
        return []
    all_items = base + child
    if all(isinstance(x, str) for x in all_items):
        seen: set[str] = set()
        merged: list[str] = []
        for item in all_items:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged
    return list(base) + list(child)


def _deep_merge(base: dict, child: dict) -> dict:
    """Deep-merge two raw YAML profile dicts (child wins on conflict)."""
    merged = dict(base)
    for key, child_val in child.items():
        if key not in base:
            merged[key] = child_val
        else:
            base_val = base[key]
            if isinstance(base_val, dict) and isinstance(child_val, dict):
                merged[key] = _deep_merge(base_val, child_val)
            elif isinstance(base_val, list) and isinstance(child_val, list):
                merged[key] = _merge_lists(base_val, child_val)
            else:
                merged[key] = child_val
    return merged


# String-list profile fields that ``exclude:`` may subtract from. Deliberately
# excludes ``config`` — list subtraction only makes sense for the plain string
# collections a child inherits via ``extends``, not for a mapping whose entries
# are key/value settings rather than names. The dict-shaped convention fields
# (``mcp_servers``, ``services``) are reachable through ``exclude:`` by a
# different route: _apply_exclude drops the named declaration alongside the
# files, which is why they are absent here rather than unreachable.
_EXCLUDABLE_FIELDS: frozenset[str] = frozenset(
    {
        "skills",
        "rules",
        "hooks",
        "agents",
        "output_styles",
        "web_panels",
        "dependencies",
    }
)


def _convention_source_for_field(field_name: str) -> str | None:
    """The convention directory, if any, an ``exclude:`` field names.

    Args:
        field_name: One key of an ``exclude:`` mapping.

    Returns:
        The matching entry of
        :data:`~osprey.cli.profile_conventions.CONVENTION_SOURCES`, or ``None``
        when the field names no convention directory. A profile spells its list
        fields with underscores (``output_styles``) while the directory is
        hyphenated (``output-styles/``), so an underscore fallback bridges the
        two; the identity match runs first, which keeps ``mcp_servers`` — whose
        directory really does carry an underscore — from being rewritten.
    """
    from .profile_conventions import CONVENTION_SOURCES

    if field_name in CONVENTION_SOURCES:
        return field_name
    hyphenated = field_name.replace("_", "-")
    return hyphenated if hyphenated in CONVENTION_SOURCES else None


# ``exclude:`` fields whose entries name built-in library artifacts rather than
# profile-authored files. ``validate_artifacts`` hard-rejects any name in these
# profile lists that the built-in library does not carry, so a profile's own
# ``agents/orbit-writer.md`` can never legally appear in ``agents:`` — the two
# namespaces are disjoint, which is why the two ``exclude:`` spellings below
# have to mean different things.
#
# ``hooks`` is the one field whose two namespaces cannot collide by accident:
# the library selects a hook by a readable short name (``writes-check``) while
# a profile-shipped one is named by the filename the render wires into
# ``settings.json`` (``hooks/osprey_writes_check.py``). The library probe below
# still decides, as it does for every other field — the two spellings simply
# cannot be confused for one another the way ``rules/safety`` and ``safety``
# can.
_ARTIFACT_LIBRARY_FIELDS: frozenset[str] = frozenset(
    {"hooks", "rules", "skills", "agents", "output_styles"}
)


def _is_library_selection(field_name: str, name: str) -> bool:
    """Whether ``name`` under ``field_name`` names a built-in library artifact.

    Args:
        field_name: The ``exclude:`` key the entry was listed under.
        name: The bare entry.

    Returns:
        ``True`` only for a field the built-in library governs whose entry that
        library actually carries. An unreadable library answers ``False``: the
        exclusion still applies, it is merely also recorded as a file omission,
        which is the harmless direction to be wrong in.
    """
    if field_name not in _ARTIFACT_LIBRARY_FIELDS:
        return False
    from .templates.artifact_library import list_artifacts

    try:
        return name in set(list_artifacts(field_name))
    except Exception:
        return False


def _split_exclude_entry(field_name: str, source: str | None, entry: Any) -> tuple[str, bool]:
    """Split one ``exclude:`` entry into its bare name and its spelling.

    The spelling is the contract, because the two things a profile can select
    by name live in disjoint namespaces. ``channel-finder`` is a built-in
    library artifact the profile *selects*; ``agents/channel-finder`` is a file
    the profile *ships*, which shadows the built-in one. Excluding the shadow
    must leave the selection standing — that is what makes the built-in version
    render again — so the two spellings cannot be treated as equivalent.

    Args:
        field_name: The ``exclude:`` key the entry was listed under.
        source: That key's convention directory, or ``None``.
        entry: The raw list entry.

    Returns:
        ``(name, qualified)`` — the bare name, and whether it was written with
        its convention directory as a prefix.

    Raises:
        BuildProfileError: If the entry is not a string, is qualified with a
            *different* convention directory than the key it sits under, or is
            a prefix naming nothing. Each of those would otherwise exclude the
            wrong artifact, or nothing at all, while looking like it worked.
    """
    if not isinstance(entry, str):
        raise BuildProfileError(
            f"exclude.{field_name}: entries must be strings naming one artifact "
            f"(got {type(entry).__name__}: {entry!r})"
        )
    head, sep, tail = entry.partition("/")
    if not sep:
        return entry, False
    if source is not None and head in (source, field_name):
        if not tail:
            raise BuildProfileError(
                f"exclude.{field_name}: entry {entry!r} names no artifact after the "
                f"{head + '/'!r} prefix."
            )
        return tail, True
    # Checked for every field, not only convention-backed ones: a qualified
    # entry under an unrelated key (``hooks: [agents/x]``) is the same mistake
    # and must not pass as a literal hook name that matches nothing.
    if _convention_source_for_field(head) is not None:
        raise BuildProfileError(
            f"exclude.{field_name}: entry {entry!r} names a {head!r} artifact under "
            f"the {field_name!r} key. List it under {head!r} instead, or drop the "
            f"{head + '/'!r} prefix if you did mean a {field_name!r} entry."
        )
    return entry, False


def _apply_exclude(
    merged: dict[str, Any],
    exclude: Any,
    *,
    artifacts: set[str] | None = None,
    shadow_candidates: set[str] | None = None,
) -> None:
    """Subtract ``exclude`` entries from ``merged`` in place, recording artifacts.

    ``exclude`` is a mapping of field name to a list of entries to remove, and
    the SPELLING of each entry decides which of the two things it removes:

    * bare (``channel-finder``) subtracts from the string list
      (:data:`_EXCLUDABLE_FIELDS`), so a built-in artifact an ``extends`` base
      selected is no longer selected;
    * qualified (``agents/channel-finder``) records the entry in ``artifacts``
      as ``"<source>/<name>"`` so the build omits the profile's own file for
      it, leaving the list alone.

    Keeping them apart is what makes FR-10's promise true: a profile that ships
    ``agents/channel-finder.md`` shadows the built-in agent of that name, and
    excluding the shadow by its qualified spelling drops only the file, so the
    still-selected built-in version renders again. Conflating the two would
    delete both and the artifact would vanish entirely.

    A bare entry that the built-in library does *not* carry is recorded as a
    file omission too — it cannot be a library selection (``validate_artifacts``
    would reject it), so the only thing it can mean is the profile's own file.
    Dict-shaped fields (``mcp_servers``, ``services``) drop the declaration on
    either spelling: the declaration names the profile's own directory, so
    omitting the files while leaving it standing would deploy a server whose
    source is missing, and there is no built-in version to fall back to.

    Excluding an entry that is not present is a silent no-op. Because this runs
    after each ``_deep_merge`` in :func:`_resolve_extends`, a deeper ``extends``
    layer that re-adds an entry merges in afterwards and wins; an entry re-added
    by an override file or ``--set`` merges *before* extends resolution and is
    stripped again here, so it cannot win.

    Args:
        merged: The merged raw profile dict (mutated in place).
        exclude: The raw ``exclude`` value from a profile layer.
        artifacts: Set collecting convention-artifact exclusions, mutated in
            place. ``None`` skips recording, for callers that only need list
            subtraction.
        shadow_candidates: Set collecting the *other* half — bare entries that
            named a built-in library artifact, as ``"<source>/<name>"`` — so a
            caller that knows the profile root can tell whether the profile also
            shadows that name (see :func:`_warn_shadowed_bare_exclusions`).
            Deliberately disjoint from ``artifacts``: these entries record no
            file omission, which is exactly why they need a separate diagnostic.

    Raises:
        BuildProfileError: If ``exclude`` is not a mapping, names a field that
            is neither an excludable list nor a convention directory, or maps a
            field to a non-list value.
    """
    from .profile_conventions import CONVENTION_SOURCES

    if not isinstance(exclude, dict):
        raise BuildProfileError(
            f"Profile 'exclude' must be a mapping of field name to list "
            f"(got {type(exclude).__name__})"
        )
    for field_name, entries in exclude.items():
        source = _convention_source_for_field(field_name)
        if field_name not in _EXCLUDABLE_FIELDS and source is None:
            raise BuildProfileError(
                f"exclude: unknown or non-list field {field_name!r} (must be one of "
                f"{sorted(_EXCLUDABLE_FIELDS)}, or a convention directory: "
                f"{sorted(CONVENTION_SOURCES)})"
            )
        if not isinstance(entries, list):
            raise BuildProfileError(
                f"exclude.{field_name} must be a list of entries to remove "
                f"(got {type(entries).__name__})"
            )
        split = [_split_exclude_entry(field_name, source, entry) for entry in entries]
        if source is not None:
            for name, qualified in split:
                # One library probe per entry: list_artifacts() walks a
                # directory, and both records below key off the same answer.
                library = not qualified and _is_library_selection(field_name, name)
                if artifacts is not None and not library:
                    artifacts.add(f"{source}/{name}")
                if shadow_candidates is not None and library:
                    shadow_candidates.add(f"{source}/{name}")

        current = merged.get(field_name)
        if isinstance(current, dict):
            for name, _qualified in split:
                current.pop(name, None)
        elif isinstance(current, list):
            removal = {name for name, qualified in split if not qualified}
            merged[field_name] = [item for item in current if item not in removal]


def _resolve_extends(
    raw: dict[str, Any],
    profile_path: Path,
    chain: list[Path] | None = None,
    *,
    artifacts: set[str] | None = None,
    shadow_candidates: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve ``extends`` chain, returning a fully merged raw YAML dict.

    Args:
        raw: The raw YAML dict from the current file.
        profile_path: Resolved path to the current YAML file.
        chain: Paths already visited (for circular-reference detection).
        artifacts: Set collecting convention-artifact exclusions from every
            layer in the chain, mutated in place. ``None`` skips recording.
        shadow_candidates: Set collecting bare library-selection exclusions from
            every layer, mutated in place. ``None`` skips recording.

    Returns:
        Merged raw dict with ``extends`` consumed.

    Raises:
        BuildProfileError: On missing base, circular reference, or bad YAML.
    """
    if chain is None:
        chain = []

    resolved = profile_path.resolve()
    if resolved in chain:
        cycle = " -> ".join(str(p) for p in chain) + f" -> {resolved}"
        raise BuildProfileError(f"Circular extends detected: {cycle}")
    chain.append(resolved)

    extends_value = raw.pop("extends", None)
    if extends_value is None:
        # No base to subtract from — ``exclude`` here can only touch this file's
        # own declarations, which is an author mistake. Apply-to-self (a no-op in
        # practice) and log so it's discoverable, matching the recursive path's
        # "pop exclude before returning" contract.
        exclude_value = raw.pop("exclude", None)
        if exclude_value is not None:
            _LOGGER.debug(
                "Profile %s declares 'exclude' without 'extends'; it can only "
                "affect its own declarations (no inherited entries to remove).",
                resolved,
            )
            _apply_exclude(
                raw, exclude_value, artifacts=artifacts, shadow_candidates=shadow_candidates
            )
        return raw

    # Try a bundled preset by name first; fall through to filesystem-path
    # resolution. Path-shaped values like ``als-base.yml`` correctly miss the
    # preset probe (it looks up ``als-base.yml.yml``) and resolve as paths,
    # preserving the sibling-file semantics ALS-style profiles depend on.
    preset_path = _preset_exists(extends_value)
    if preset_path is not None:
        base_path = preset_path
    else:
        base_path = (profile_path.parent / extends_value).resolve()
        if not base_path.exists():
            available = ", ".join(list_presets()) or "(none)"
            raise BuildProfileError(
                f"Cannot resolve extends: {extends_value!r}. "
                f"No bundled preset by that name (available: {available}), "
                f"and no file at {base_path}."
            )

    base_raw = _read_profile_document(base_path)

    if not isinstance(base_raw, dict):
        raise BuildProfileError(f"Extended profile must be a YAML mapping: {base_path}")

    # Recurse: the base may itself extend another profile
    base_raw = _resolve_extends(
        base_raw, base_path, chain, artifacts=artifacts, shadow_candidates=shadow_candidates
    )

    merged = _deep_merge(base_raw, raw)
    # Apply this layer's ``exclude`` to the merged result and consume it. The
    # recursively-resolved ``base_raw`` has already had its own ``exclude``
    # popped, so the only ``exclude`` present here is this layer's own.
    exclude_value = merged.pop("exclude", None)
    if exclude_value is not None:
        _apply_exclude(
            merged, exclude_value, artifacts=artifacts, shadow_candidates=shadow_candidates
        )
    return merged


def merge_persona_delta(
    root_resolved: dict[str, Any],
    delta: dict[str, Any],
    *,
    artifacts: set[str] | None = None,
    shadow_candidates: set[str] | None = None,
) -> dict[str, Any]:
    """Merge a persona delta over its resolved root profile.

    The implicit merge a ``personas/<name>.yml`` file stands for: it carries
    only what differs from the profile it lives under, so it is meaningless on
    its own. The sequence mirrors one ``extends`` layer exactly — the delta's
    ``exclude:`` is consumed first, then the delta wins key-by-key over the
    root, then ``exclude:`` subtracts from the *merged* result — so a persona
    resolves identically whether it is spelled as a delta or as an ``extends``
    child.

    Args:
        root_resolved: The root profile, already ``extends``-resolved.
        delta: The persona's own layer.
        artifacts: Set collecting the delta's convention-artifact exclusions,
            mutated in place. ``None`` skips recording.
        shadow_candidates: Set collecting the delta's bare library-selection
            exclusions, mutated in place. ``None`` skips recording.

    Returns:
        A new merged dict; neither argument is mutated.
    """
    delta = dict(delta)
    exclude_value = delta.pop("exclude", None)
    merged = _deep_merge(root_resolved, delta)
    if exclude_value is not None:
        _apply_exclude(
            merged, exclude_value, artifacts=artifacts, shadow_candidates=shadow_candidates
        )
    return merged


def _profile_ships(root_dir: Path, artifact: str) -> bool:
    """Whether the profile carries its own file for a ``"<source>/<name>"`` entry.

    Args:
        root_dir: The profile root the exclusions anchor at.
        artifact: A recorded exclusion, ``"<source>/<name>"``.

    Returns:
        ``True`` when the profile ships that artifact. Markdown categories name
        the artifact without its extension; the directory-shaped ones name the
        directory itself, so both spellings are probed.

        File-shaped categories (``hooks/``) need a third probe. A profile ships
        a hook under the filename the render wires into ``settings.json``
        (``hooks/osprey_writes_check.py``), while the built-in library selects
        the same hook by a readable short name (``writes-check``) — so a bare
        library exclusion arrives here spelled neither way, and only the
        library's own resolution maps one to the other. A short name that
        resolves to nothing is not shipped, which is the answer that keeps the
        unmatched-exclusion warning loud about a misspelling.
    """
    from .profile_conventions import EntryShape, convention_for

    source, _, name = artifact.partition("/")
    base = root_dir / source / name
    if base.exists() or (base.parent / f"{base.name}.md").exists():
        return True

    convention = convention_for(source)
    if convention is None or convention.shape is not EntryShape.FILE:
        return False
    from .templates.artifact_library import resolve_artifact

    try:
        library_file = resolve_artifact(source, name)
    except Exception:
        return False
    return (base.parent / library_file.name).is_file()


def _warn_shadowed_bare_exclusions(root_dir: Path, shadow_candidates: set[str]) -> None:
    """Warn when a bare exclusion names a library artifact the profile shadows.

    The one exclusion mistake neither :func:`_warn_unmatched_exclusions` nor the
    build itself can see. A bare entry unselects the built-in artifact, and that
    is all it does — so when the profile *also* ships a file of that name, the
    file keeps rendering and the project comes out byte-identical. Nothing is
    recorded in the exclusion record (by design: no file is omitted), so the
    unmatched-exclusion warning has nothing to look at, and the author reads a
    successful build as a successful exclusion.

    The remedy is the other spelling, so the warning names it: the qualified
    ``<source>/<name>`` form drops the profile's own file, which is what someone
    who shadows an artifact and then excludes it almost always meant.

    Args:
        root_dir: The profile root the exclusions anchor at.
        shadow_candidates: Bare library-selection exclusions recorded by
            :func:`_apply_exclude`, as ``"<source>/<name>"``.
    """
    for artifact in sorted(shadow_candidates):
        if not _profile_ships(root_dir, artifact):
            continue
        source, _, name = artifact.partition("/")
        _LOGGER.warning(
            "Profile %s excludes the bare name %r under %r, but it also ships its own "
            "%r in %s — that file still renders, so the exclusion has no effect. "
            "Did you mean %r? A bare name only unselects the BUILT-IN artifact, which "
            "this profile already shadows; the qualified spelling omits the profile's "
            "own file, which is what makes the built-in version render again.",
            root_dir,
            name,
            source,
            name,
            root_dir / source,
            artifact,
        )


def _warn_unmatched_exclusions(root_dir: Path, artifacts: set[str]) -> None:
    """Warn about recorded exclusions that match no file under the profile root.

    A misspelled shadow exclusion is otherwise a silent no-op: the artifact it
    meant to drop keeps rendering, and nothing says why. This warns rather than
    raises because a root profile may legitimately exclude something only some
    of its personas carry.

    Args:
        root_dir: The profile root the exclusions anchor at.
        artifacts: Recorded ``"<source>/<name>"`` exclusions.
    """
    for artifact in sorted(artifacts):
        if _profile_ships(root_dir, artifact):
            continue
        source, _, _name = artifact.partition("/")
        _LOGGER.warning(
            "Profile %s excludes %r, but no such artifact exists under %s — the "
            "exclusion has no effect. Check the spelling.",
            root_dir,
            artifact,
            root_dir / source,
        )


@dataclass(frozen=True)
class ResolvedProfileDocument:
    """One profile document resolved into what the build actually reads.

    Attributes:
        raw: The fully resolved profile mapping — ``extends`` consumed and, for
            a persona delta, merged over its root.
        root_dir: The profile root every profile-relative path anchors at
            (``data:``, convention directories, ``.env``, hash material). For a
            persona delta this is the directory *above* ``personas/``, never the
            delta's own parent.
        is_persona_delta: Whether the document was resolved as a persona delta.
        excluded_artifacts: Convention-dir artifacts the resolved profile
            excludes, as ``"<source>/<name>"`` (e.g. ``"agents/orbit-writer"``).
            The build omits these files, and ownership derivation scans the
            post-exclude set — so excluding a shadow restores the framework's
            own version of that artifact.
    """

    raw: dict[str, Any]
    root_dir: Path
    is_persona_delta: bool
    excluded_artifacts: frozenset[str]


def resolve_profile_document(
    raw: dict[str, Any], profile_path: Path, *, warn: bool = True
) -> ResolvedProfileDocument:
    """Resolve a profile document, following the implicit persona merge.

    The one place that decides what a profile file *means*. A file under
    ``personas/`` beside a ``profile.yml`` carries only a delta: living there
    IS its inheritance, so it is resolved against that root rather than on its
    own, and everything it names anchors at the root. Every other file resolves
    standalone through its ``extends`` chain, as before.

    Both the loader and the content hash go through here, which is what keeps a
    persona's built project and its staleness hash describing the same merge.

    Args:
        raw: The profile document as read.
        profile_path: Where that document lives — the classification is made
            from its location, not its contents.
        warn: Whether to log the diagnostic for exclusions that match no file.
            ``False`` for the hash path, which resolves the same document a
            build already loaded — without it every build reports each stale
            exclusion twice, and a warning that repeats is one people learn to
            skip. The loader is the user-facing surface, so it keeps the
            warning.

    Returns:
        The resolved document and its anchor.

    Raises:
        BuildProfileError: If a persona delta declares ``extends``, if the root
            of a persona delta is missing or is not a YAML mapping, or if
            ``extends`` resolution fails.
    """
    root_dir, is_persona_delta = resolve_profile_root(profile_path)
    normalized = _normalize_profile_aliases(dict(raw), str(profile_path))
    artifacts: set[str] = set()
    shadowed: set[str] = set()

    def finish(resolved: dict[str, Any], as_delta: bool) -> ResolvedProfileDocument:
        """Report the exclusion diagnostics, then package the result.

        Both branches end the same way, and they must: the two diagnostics read
        the sets every layer of either branch fed, so a branch that skipped one
        would leave that whole shape of profile without the warning.
        """
        if warn:
            _warn_unmatched_exclusions(root_dir, artifacts)
            _warn_shadowed_bare_exclusions(root_dir, shadowed)
        return ResolvedProfileDocument(resolved, root_dir, as_delta, frozenset(artifacts))

    if not is_persona_delta:
        return finish(
            _resolve_extends(
                normalized, profile_path, artifacts=artifacts, shadow_candidates=shadowed
            ),
            False,
        )

    if "extends" in normalized:
        raise BuildProfileError(
            f"Persona delta {profile_path} declares 'extends'. A file in "
            f"{PERSONA_DIRNAME}/ already inherits from the {ROOT_PROFILE_FILENAME} beside "
            f"it — that is what makes it a delta — so a second base would be ambiguous. "
            f"Remove the 'extends' key; to change what the whole stack inherits, edit "
            f"{root_dir / ROOT_PROFILE_FILENAME} instead."
        )

    root_path = root_dir / ROOT_PROFILE_FILENAME
    root_raw = _read_profile_document(root_path)
    if not isinstance(root_raw, dict):
        raise BuildProfileError(f"Profile root must be a YAML mapping: {root_path}")
    root_resolved = _resolve_extends(
        _normalize_profile_aliases(dict(root_raw), str(root_path)),
        root_path,
        artifacts=artifacts,
        shadow_candidates=shadowed,
    )
    # The delta goes into the merge unresolved on purpose: it carries no
    # ``extends:`` of its own (rejected above), and _resolve_extends would
    # consume its ``exclude:`` against its own layer instead of against the
    # root, silently dropping the exclusion.
    return finish(
        merge_persona_delta(
            root_resolved, normalized, artifacts=artifacts, shadow_candidates=shadowed
        ),
        True,
    )


# ---------------------------------------------------------------------------
# Profile content hashing
# ---------------------------------------------------------------------------


def _fold_source_tree(digest: Any, label: str, source: Path) -> None:
    """Fold one file-or-directory build input into ``digest``, in place.

    Every regular file under ``source`` contributes its path relative to
    ``source`` (posix separators, so the digest matches across platforms) NUL-
    joined with the SHA-256 of its bytes. Entries are sorted by that relative
    path, and directories themselves contribute nothing — only their files —
    so the digest depends on content alone and not on filesystem walk order.

    A ``source`` that does not exist folds only its ``label``: the profile keys
    that name it are already in the canonical JSON, so a vanished tree still
    reads differently from a populated one without this function having to
    invent a marker for it.

    Two limits of the walk, both deliberate:

    * Symlinks. A symlinked *file* is folded — ``is_file`` and ``read_bytes``
      both follow it — and so is a ``source`` that is itself a symlink to a
      directory, which arrives here already resolved. A symlinked *directory
      nested inside* the tree is not: ``rglob`` does not recurse into one, so
      its contents are invisible to the digest while ``shutil.copytree``
      (``symlinks=False``) does follow it and copy them into the project. A
      data tree that reaches its content through a nested directory symlink
      therefore has a known blind spot in the staleness advisory: edits behind
      that link do not move the hash. Recursing resolved targets would change
      the pinned algorithm and is left to the maintainer.
    * Unicode. Relative paths are folded as their UTF-8 bytes with no
      normalization, so cross-platform determinism is guaranteed for ASCII
      names; a non-ASCII name stored NFD on one filesystem and NFC on another
      hashes differently.
    """
    import hashlib

    if source.is_dir():
        entries = sorted(
            (
                (path.relative_to(source).as_posix(), path)
                for path in source.rglob("*")
                if path.is_file()
            ),
            key=lambda entry: entry[0],
        )
    elif source.is_file():
        entries = [(source.name, source)]
    else:
        entries = []

    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    for rel_posix, path in entries:
        digest.update(rel_posix.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")


def _fold_profile_material(
    digest: Any, resolved: dict[str, Any], profile_dir: Path, *, conventions: bool = True
) -> None:
    """Fold the *file* inputs a resolved profile names into ``digest``.

    The canonical JSON alone captures what a profile says; it cannot see what
    the trees it points at contain. Without this, editing a facility's channel
    database, a rule the agent reads, or the trigger table a dispatcher fires
    on leaves the profile hash untouched and the deploy-side staleness advisory
    stays silent about a project that no longer matches its source.

    Four kinds of input are covered:

    * the ``data:`` tree, anchored via
      :meth:`~osprey.cli.build_profile_model.BuildProfile.resolved_data_root`
      so it resolves exactly where the build copies it from;
    * every convention directory the profile carries
      (:data:`~osprey.cli.profile_conventions.CONVENTION_SOURCES`, which
      includes the ``project/`` verbatim mirror), folded in sorted name order
      so reordering the mapping table cannot move a hash;
    * ``triggers.yml``, the dispatch trigger table the build materializes;
    * every persona delta in ``personas/`` — the files a deployment's own
      per-persona projects are rendered from. Without them, editing a delta
      moves no hash at all: the deploy-side check would call the build clean,
      no rebuild would follow, and the persona render already on disk (which
      ``up`` re-renders only when it is *absent*) would keep shipping the
      superseded delta into that persona's image.

    Personas are folded as the *direct children* of ``personas/`` and nothing
    deeper, because that is exactly the set that can be built: a file is read
    as a delta only when its parent directory IS ``personas/``
    (:func:`~osprey.cli.profile_root.resolve_profile_root`), so a tree nested
    below it is no more build input than a ``README`` beside ``profile.yml``.
    Suffixes are not filtered — a delta is named by the persona catalog, which
    accepts any direct child — while dot-prefixed entries are skipped, matching
    the walk every other convention read uses: a ``.DS_Store`` must not report a
    deployment as stale.

    One consequence, accepted rather than worked around: this is folded for a
    persona delta too (its root_dir is the profile root), so editing ONE delta
    moves every sibling persona's hash as well. That is the same over-report as
    a root edit reaching every persona — it costs an idempotent rebuild, where
    subtracting siblings would cost a persona that silently no longer matches
    its source.

    A convention directory, ``triggers.yml`` or ``personas/`` that is absent
    folds nothing at all, so a profile carrying none of them hashes exactly as
    it did before they existed. Creating an empty convention directory does
    move the hash — it folds its label — which is honest: the profile tree
    changed.

    Args:
        digest: The hash object to update in place.
        resolved: The profile after ``extends`` (and persona-delta) resolution.
        profile_dir: The profile root every relative path anchors at.
        conventions: Whether to fold convention material. ``False`` for a
            bundled preset, which is one file among many in a shared package
            directory rather than a profile root of its own — folding there
            would make every preset's hash depend on every other preset's
            neighbouring directories.
    """
    from .build_profile_model import BuildProfile
    from .profile_conventions import CONVENTION_SOURCES
    from .profile_root import PERSONA_DIRNAME

    data_root = BuildProfile(name="", data=resolved.get("data")).resolved_data_root(profile_dir)
    if data_root is not None:
        _fold_source_tree(digest, "data", data_root)

    if not conventions:
        return

    for source in sorted(CONVENTION_SOURCES):
        candidate = profile_dir / source
        if candidate.exists():
            _fold_source_tree(digest, f"convention:{source}", candidate)

    triggers = profile_dir / "triggers.yml"
    if triggers.is_file():
        _fold_source_tree(digest, "triggers", triggers)

    # Sorted by name, which is the whole ordering rule a flat directory needs;
    # `_fold_source_tree` contributes each delta's own name beside its content
    # digest, so a rename moves the hash exactly as an edit does.
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
            _fold_source_tree(digest, "persona", delta)


def _hash_resolved_profile(
    raw: dict[str, Any], profile_path: Path, *, conventions: bool = True
) -> str:
    """Canonical content hash of a profile dict after ``extends`` resolution.

    Hashes the *resolved* content (canonical JSON, sorted keys) rather than
    file bytes, so comment/ordering churn is invisible while a change in any
    ``extends`` parent is not. The file inputs the resolved profile names — its
    ``data:`` tree, convention directories, ``triggers.yml`` and the persona
    deltas in ``personas/`` — are folded in on top by
    :func:`_fold_profile_material`, so a project whose facility data changed
    under an unchanged profile still reads as stale.

    Resolution goes through :func:`resolve_profile_document`, the same call the
    loader makes: the hash can only say honestly whether a built project still
    matches its source if it resolves the *same* document the build resolved.
    That is also what makes a root edit move every persona project's hash.

    The excluded-artifact record is folded in as well. It is derived state, not
    profile content, so the canonical JSON cannot see it — yet two personas
    differing only by a ``commands:`` exclusion build different projects, and
    without this they would hash identically and neither would ever read stale.

    What goes in is the profile's declared selection lists, not the *effective*
    post-shadow ones: a bare exclusion of a name the profile also shadows moves
    the hash while rendering byte-identical output, and that over-report is
    accepted on purpose. Subtracting the effective selection would mean
    re-deriving the build's shadow rule inside the hash, and any drift between
    the two would make the hash miss a real divergence — the one failure this
    advisory exists to prevent. Over-reporting costs an idempotent rebuild;
    under-reporting costs a project that silently no longer matches its source,
    so the conservative input is the right one.
    :func:`_warn_shadowed_bare_exclusions` keeps that case loud rather than
    letting an author sit on a no-op exclusion forever.

    ``raw`` is normalized to the canonical field spellings first, so a caller
    that hands over a dict which did not come from
    :func:`~osprey.cli.build_profile_document._read_profile_document` hashes
    the same as one that did — the YAML-surface rename must not move any
    profile's hash. Normalizing before ``extends`` resolution (rather than
    after) keeps the deep merge child-wins across a mixed-spelling chain.
    """
    import hashlib
    import json

    document = resolve_profile_document(raw, profile_path, warn=False)
    canonical = json.dumps(document.raw, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8"))
    _fold_profile_material(digest, document.raw, document.root_dir, conventions=conventions)
    # Sorted so the record's iteration order cannot move a hash; a profile with
    # no exclusions folds nothing and keeps the digest it had before.
    for artifact in sorted(document.excluded_artifacts):
        digest.update(b"exclude:")
        digest.update(artifact.encode("utf-8"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def compute_preset_hash(preset_name: str) -> str | None:
    """Content hash of a bundled preset as resolved (post-``extends``).

    Stamped into ``.osprey-manifest.json`` at build time and compared by the
    deploy-side staleness advisory
    (:mod:`osprey.deployment.staleness`). Returns ``None`` when the preset is
    unknown or unreadable — callers treat that as "cannot compare", never as
    drift.

    Convention material is deliberately not folded: bundled presets share one
    package directory, so folding it would make every preset's hash move when
    any other preset's neighbouring directories changed.
    """
    try:
        raw, path = _load_preset_raw(preset_name)
        return _hash_resolved_profile(raw, path, conventions=False)
    except Exception:
        return None


def compute_profile_hash(profile_path: Path) -> str | None:
    """Content hash of a repo's own ``profile.yml`` as resolved (post-``extends``).

    Counterpart of :func:`compute_preset_hash` for profile-built repos — the
    drift-fingerprint input, hashed against the profile path the manifest
    recorded. Returns ``None`` when the file is missing or unreadable.
    """
    try:
        path = Path(profile_path)
        raw = _read_profile_document(path)
        if not isinstance(raw, dict):
            return None
        return _hash_resolved_profile(raw, path)
    except Exception:
        return None
