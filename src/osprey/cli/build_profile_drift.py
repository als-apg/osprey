"""The preset-drift lint — what a materialized profile no longer shares with its preset.

``osprey init`` writes a preset out as ``profile.yml`` and stamps
``provenance:`` with the preset's name and content hash. From then on the
profile is the source of truth, and the emitted header invites editing it. The
cost of that copy is silence: when the preset later gains a line — a panel, a
hook, a permission — no build says so, and a profile maintained as a delta over
the preset drifts without anyone choosing to. This module is the comparison the
stamp was written for.

The reference is not the preset's raw layer, which no repo ever holds, but
:func:`~osprey.cli.build_profile_emit.materialized_profile` — the document
``osprey init`` would write from the installed preset for this repo today. The
profile is compared with it key by key. Each tier and app file is compared the
same way with :func:`~osprey.cli.build_profile_emit.emit_layer_delta_yaml` —
the file ``osprey init`` would write from the bundled tier or app its own
``provenance:`` block names. What a repo writes for itself rather than copies
from the preset is not compared: the display name, the materialized data tree,
the schema floor and the provenance stamp.

A difference is deliberate when a ``# <TAG>: <why>`` comment says so — within
three lines above the profile line it describes, or, for a line the profile
does not have, anywhere in the file naming what is absent. The tag is
``provenance.deviation_marker`` (default ``DEVIATION``), so a facility's
existing marker convention is the first consumer rather than a second one.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML, CommentedMap, CommentedSeq

from osprey import __version__
from osprey.profiles.providers import (
    PROVIDERS_FILENAME,
    load_provider_catalog,
    packaged_catalog_path,
)
from osprey.utils.workspace import APPS_DIRNAME, LAYER_DIR_KINDS, PROFILE_FILENAME, TIERS_DIRNAME
from osprey_connectors.errors import BuildProfileError

from .build_profile_document import _parse_profile_document, _read_profile_document
from .build_profile_emit import emit_layer_delta_yaml, materialized_profile
from .build_profile_merge import (
    _resolve_extends,
    compute_preset_hash,
    resolve_profile_document,
)
from .build_profile_presets import _load_preset_raw, _normalize_preset_name
from .build_profile_resolve import _dotted_leaves
from .build_profile_schema import DEFAULT_DEVIATION_MARKER, ProfileProvenance
from .profile_root import layer_files, resolve_profile_root

#: How far above a profile line a marker comment reaches.
MARKER_REACH = 3

#: The finding kinds ``osprey validate --drift=error`` refuses: one document
#: has a key, a block or a list member the other has not. These are the
#: silent failure the lint was written for — a profile that never picked up a
#: line its preset gained builds green forever, and nothing says so.
REFUSAL_KINDS: frozenset[str] = frozenset(
    {"missing", "missing_member", "extra", "extra_member", "exclude"}
)

#: The finding kind that is reported and never refused: both documents carry
#: the key and disagree on its value. That is a knob a facility turned — the
#: thing the emitted profile invites — so a preset changing a default in a
#: newer OSPREY surfaces as a note rather than as a refusal to build.
NOTE_KINDS: frozenset[str] = frozenset({"value"})

#: What joins two advisories into :attr:`DriftReport.note`. The printer indents
#: the note it is handed by one step and no more, so a second line carries that
#: indent itself rather than starting at the margin under the first.
_NOTE_SEPARATOR = "\n  "


def _compare_catalog(root_dir: Path, tag: str) -> tuple[list[DriftFinding], list[str]]:
    """``providers.yml`` against the catalog OSPREY ships, entry by entry.

    The repo's catalog is the operator's the way ``profile.yml`` is, and ``osprey
    init --revalidate`` never changes a value it states; it only adds what the
    packaged catalog has and the file lacks. So the entries this OSPREY ships
    are compared key by key, with the profile's own rules: a value the
    facility turned is a note, a key or a list member one side has and the
    other has not is a refusal, and a ``# <TAG>:`` comment claims either. An
    entry the packaged catalog does not declare is the operator's own gateway
    and is not compared.

    Silent for a repo with no catalog: the packaged copy IS what it resolves,
    so nothing has drifted from anything.

    Args:
        root_dir: The deployment repo root, where a ``providers.yml`` sits
            beside ``profile.yml``.
        tag: The deviation marker that claims a finding.

    Returns:
        ``(findings, stale markers)``, as for any other compared document.

    Raises:
        BuildProfileError: The repo's catalog does not parse. Left to surface
            rather than swallowed — every other reader of that file refuses
            too, and a lint that quietly said nothing would be the one place
            the broken file looked fine.
    """
    catalog = load_provider_catalog(root_dir)
    if catalog.source != "repo" or catalog.path is None:
        return [], []
    shipped_path = packaged_catalog_path()
    actual = _parse_profile_document(catalog.path.read_text(encoding="utf-8"), PROVIDERS_FILENAME)
    shipped = _parse_profile_document(shipped_path.read_text(encoding="utf-8"), "packaged")
    actual_entries = _mapping(_mapping(actual).get("providers"))
    shipped_entries = _mapping(_mapping(shipped).get("providers"))
    comparison = _Comparison(
        PROVIDERS_FILENAME,
        _Lines(catalog.path),
        f"the catalog OSPREY {__version__} ships",
        [_Lines(shipped_path)],
    )
    comparison.compare(
        {
            "providers": {
                name: actual_entries[name] for name in shipped_entries if name in actual_entries
            }
        },
        {"providers": shipped_entries},
    )
    return _apply_markers(comparison.findings, catalog.path, root_dir, tag)


# Top-level keys `osprey init` writes for the repo rather than copies from the
# preset: the display name, the deployment's own name, the materialized data
# tree, the schema floor, and the provenance stamp itself — whose hash
# difference is the note, not a finding.
_MATERIALIZATION_KEYS: frozenset[str] = frozenset(
    {"name", "project_name", "data", "provenance", "requires_osprey_version"}
)


@dataclass(frozen=True)
class DriftFinding:
    """One place the profile and its preset disagree.

    Attributes:
        subject: What differs, as an operator would name it — a dotted key
            (``config.web.theme``, ``bluesky``) or a list member
            (``web_panels: lattice``).
        detail: Which way it differs.
        profile_ref: ``file:line`` in the profile, or the file alone when the
            profile has no line for it.
        preset_ref: ``file:line`` in the preset, or the preset's name when the
            value is one the emitter synthesizes.
        token: What a marker comment must name to silence this finding.
        line: The profile line the finding sits on, or ``None`` for an absence.
        kind: Which sort of difference this is, and so what it costs:
            ``value`` when both documents carry the key and give it different
            values, or one of the structural kinds — ``missing``,
            ``missing_member``, ``extra``, ``extra_member``, ``exclude`` — when
            one document has something the other has not. See
            :data:`REFUSAL_KINDS` and :data:`NOTE_KINDS`.
        marked: Whether a marker comment claims it.
    """

    subject: str
    detail: str
    profile_ref: str
    preset_ref: str
    token: str
    line: int | None
    kind: str
    marked: bool = False

    def render(self) -> str:
        """The one-line form the verbs print."""
        return f"{self.subject} — {self.detail} ({self.profile_ref}; {self.preset_ref})"


@dataclass(frozen=True)
class DriftReport:
    """Everything the lint has to say about one profile.

    Attributes:
        preset: The bundled preset the profile's provenance names.
        note: The advisories about the comparison itself rather than about any
            one key — distinct from :attr:`notes`, which are findings. The preset has
            moved on since materialization, or is not bundled here. One line
            each, joined with the printer's own indent when both apply.
        findings: Every difference found, marked or not.
        stale_markers: Marker comments that silence nothing.
    """

    preset: str
    note: str | None
    findings: list[DriftFinding]
    stale_markers: list[str]

    @property
    def unmarked(self) -> list[DriftFinding]:
        """The differences nobody has claimed as deliberate.

        The union of :attr:`refusals` and :attr:`notes`, in the order the
        comparison found them — what a surface prints when it prints
        everything.
        """
        return [finding for finding in self.findings if not finding.marked]

    @property
    def refusals(self) -> list[DriftFinding]:
        """The unclaimed structural differences — what ``osprey validate`` refuses."""
        return [finding for finding in self.unmarked if finding.kind in REFUSAL_KINDS]

    @property
    def notes(self) -> list[DriftFinding]:
        """The unclaimed value differences — reported by every surface, refused by none."""
        return [finding for finding in self.unmarked if finding.kind in NOTE_KINDS]


def preset_drift_report(profile_file: Path, provenance: ProfileProvenance | None) -> DriftReport:
    """Compare the profile at *profile_file* with what it was materialized from.

    Reads the tracked documents — ``profile.yml`` and every tier and app file
    beside it — never a host-variant overlay: drift is a property of what the
    repo tracks, and an overlay is one host's layer over it. ``profile.yml``
    pairs with the preset its provenance names; each ``tiers/<n>.yml`` and
    ``apps/<n>.yml`` pairs with the bundled tier or app ITS provenance names
    (see :func:`_layer_pairs`).

    Args:
        profile_file: A repo's ``profile.yml``, or one tier or app file under
            its ``tiers/`` or ``apps/``. A layer file is compared alone; the
            root compares itself and every layer it carries.
        provenance: The parsed ``provenance:`` block the loader resolved for
            *profile_file*. For a layer file it may be the layer's own block
            (a tier is merged over the root), so the root's stamp is re-read
            from ``profile.yml`` itself. ``None`` when the root has none.

    Returns:
        The report. A root whose preset is not bundled is noted and not
        compared; its layers still are.

    Raises:
        BuildProfileError: When a document the lint has to read is not a YAML
            mapping, the preset will not emit, two layer files share a stem,
            or the repo's ``providers.yml`` does not parse.
    """
    root_dir, is_delta = resolve_profile_root(profile_file)
    root_path = root_dir / PROFILE_FILENAME

    root_raw = _read_profile_document(root_path)
    if not isinstance(root_raw, dict):
        raise BuildProfileError(f"Profile must be a YAML mapping: {root_path}")
    if is_delta:
        provenance = _root_provenance(root_raw)
    profile = resolve_profile_document(copy.deepcopy(root_raw), root_path, warn=False).raw
    tag = provenance.deviation_marker if provenance is not None else DEFAULT_DEVIATION_MARKER

    notes: list[str] = []
    findings: list[DriftFinding] = []
    stale: list[str] = []
    preset = _normalize_preset_name(provenance.preset) if provenance is not None else ""
    if provenance is not None:
        installed_hash = compute_preset_hash(preset)
        if installed_hash is None:
            notes.append(
                f"provenance names preset {preset!r}, which OSPREY {__version__} does not "
                f"bundle — the profile cannot be compared with it"
            )
        else:
            if installed_hash != provenance.preset_hash:
                emitted = _emitted_version(root_path)
                by = f" by OSPREY {emitted}" if emitted else ""
                notes.append(
                    f"{PROFILE_FILENAME} was materialized from preset {preset} at "
                    f"{_short_hash(provenance.preset_hash)}{by}; the preset bundled with "
                    f"OSPREY {__version__} is {_short_hash(installed_hash)} — it has moved on "
                    f"since"
                )
            if not is_delta:
                findings_catalog, stale_catalog = _compare_catalog(root_dir, tag)
                findings.extend(findings_catalog)
                stale.extend(stale_catalog)
                findings_root, stale_root = _compare_root(profile, preset, root_path, root_dir, tag)
                findings.extend(findings_root)
                stale.extend(stale_root)

    profile_name = str(profile.get("name", ""))
    for pair in _layer_pairs(
        root_dir, str(profile.get("project_name") or ""), profile_file if is_delta else None
    ):
        relative = pair.path.relative_to(root_dir).as_posix()
        word = LAYER_DIR_KINDS[pair.dirname]
        installed = compute_preset_hash(pair.source)
        if installed is None:
            notes.append(
                f"{relative}: provenance names {word} {pair.stem!r}, which OSPREY "
                f"{__version__} does not bundle — the file cannot be compared with it"
            )
            continue
        if installed != pair.preset_hash:
            notes.append(
                f"{relative} was materialized from {word} {pair.stem} at "
                f"{_short_hash(pair.preset_hash)}; the {word} bundled with OSPREY "
                f"{__version__} is {_short_hash(installed)} — it has moved on since"
            )
        expected = _parse_profile_document(
            emit_layer_delta_yaml(pair.source, profile_name, relative), relative
        )
        if not isinstance(expected, dict):  # pragma: no cover - the emitter writes a mapping
            continue
        comparison = _Comparison(
            relative,
            _Lines(pair.path),
            f"{word} {pair.stem}",
            [_Lines(_load_preset_raw(pair.source)[1])],
        )
        comparison.compare(pair.document, expected)
        claimed, unclaimed = _apply_markers(
            comparison.findings, pair.path, root_dir, pair.marker or tag
        )
        findings.extend(claimed)
        stale.extend(unclaimed)

    return DriftReport(preset, _NOTE_SEPARATOR.join(notes) or None, findings, stale)


def _root_provenance(root_raw: Mapping[str, Any]) -> ProfileProvenance | None:
    """The root profile's own ``provenance:`` stamp, or ``None`` without a usable one."""
    block = root_raw.get("provenance")
    if not isinstance(block, Mapping) or not block.get("preset") or not block.get("preset_hash"):
        return None
    marker = block.get("deviation_marker")
    return ProfileProvenance(
        preset=str(block["preset"]),
        preset_hash=str(block["preset_hash"]),
        providers_hash=str(block.get("providers_hash") or ""),
        deviation_marker=(
            marker.strip()
            if isinstance(marker, str) and marker.strip()
            else DEFAULT_DEVIATION_MARKER
        ),
    )


def _compare_root(
    profile: Mapping[str, Any], preset: str, root_path: Path, root_dir: Path, tag: str
) -> tuple[list[DriftFinding], list[str]]:
    """``profile.yml`` against the document ``osprey init`` would write from *preset*."""
    preset_raw, preset_path = _load_preset_raw(preset)
    chain: list[Path] = []
    _resolve_extends(preset_raw, preset_path, chain)
    # The two names are copied from the profile itself, so neither ever reads as
    # a difference and the findings are a function of the tracked documents alone.
    expected = materialized_profile(
        preset,
        project_name=str(profile.get("project_name") or ""),
        profile_name=str(profile.get("name", "")),
    )
    comparison = _Comparison(
        PROFILE_FILENAME, _Lines(root_path), preset, [_Lines(path) for path in chain]
    )
    comparison.compare(profile, expected)
    return _apply_markers(comparison.findings, root_path, root_dir, tag)


@dataclass(frozen=True)
class _LayerPair:
    """One materialized tier or app file and the bundled source it names."""

    path: Path
    dirname: str
    stem: str
    preset_hash: str
    marker: str | None
    document: dict[str, Any]

    @property
    def source(self) -> str:
        """The kind-qualified bundled name: ``tiers/<stem>`` or ``apps/<stem>``."""
        return f"{self.dirname}/{self.stem}"


def _layer_pairs(root_dir: Path, project_name: str, only: Path | None) -> list[_LayerPair]:
    """Each tier and app file with the bundled source its provenance block names.

    Enumerated over the ``renders`` projection
    (:func:`~osprey.deployment.web_terminals.render_naming.render_projection`)
    — the same set of files the build renders — so a layer is compared exactly
    when it is built. The pairing is by provenance, not by file name, the way
    ``profile.yml`` pairs with its preset: a copy of ``apps/logbook.yml`` saved
    as ``apps/night-shift.yml`` still pairs with the bundled ``logbook``. A file
    with no provenance block was not materialized by ``osprey init`` and has
    nothing to drift from; a file that is not a mapping is the build's to name.

    Args:
        root_dir: The profile root holding ``tiers/`` and ``apps/``.
        project_name: The deployment's ``project_name``, which names renders.
        only: Restrict to this one layer file, if given.
    """
    from osprey.deployment.web_terminals.render_naming import RENDER_KIND_APP, render_projection

    paths = {path.stem: path for _kind, path in layer_files(root_dir)}
    target = only.resolve() if only is not None else None
    pairs: list[_LayerPair] = []
    for name, entry in render_projection(root_dir, project_name or "project").items():
        path = paths.get(name)
        if path is None or (target is not None and path.resolve() != target):
            continue
        document = _read_profile_document(path)
        if not isinstance(document, dict):
            continue
        block = document.get("provenance")
        if not isinstance(block, Mapping) or not block.get("preset"):
            continue
        marker = block.get("deviation_marker")
        pairs.append(
            _LayerPair(
                path=path,
                dirname=APPS_DIRNAME if entry.get("kind") == RENDER_KIND_APP else TIERS_DIRNAME,
                stem=_normalize_preset_name(str(block["preset"])),
                preset_hash=str(block.get("preset_hash") or ""),
                marker=marker.strip() if isinstance(marker, str) and marker.strip() else None,
                document=document,
            )
        )
    return pairs


class _Comparison:
    """Key-level diff of one document against its reference, collecting findings."""

    def __init__(self, label: str, lines: _Lines, preset: str, preset_lines: list[_Lines]) -> None:
        self.label = label
        self.lines = lines
        self.preset = preset
        self.preset_lines = preset_lines
        self.findings: list[DriftFinding] = []

    def compare(self, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
        """Diff two resolved profile documents from the top."""
        from osprey.deployment.web_terminals.lint import _nest_dotted

        for key in sorted(set(actual) | set(expected), key=str):
            if key in _MATERIALIZATION_KEYS:
                continue
            path = (str(key),)
            if key not in expected:
                self._extra(path)
            elif key not in actual:
                self._missing(path)
            elif key == "config":
                # Dotted keys, nested mappings and any mix of the two address
                # the same rendered leaves; both sides are read the way the
                # renderer reads them before a single leaf is compared.
                self._compare(
                    _nest_dotted(_mapping(actual[key])), _nest_dotted(_mapping(expected[key])), path
                )
            else:
                self._compare(actual[key], expected[key], path)

    def _compare(self, actual: Any, expected: Any, path: tuple[str, ...]) -> None:
        if isinstance(actual, Mapping) and isinstance(expected, Mapping):
            for key in sorted(set(actual) | set(expected), key=str):
                sub = (*path, str(key))
                if key not in expected:
                    self._extra(sub)
                elif key not in actual:
                    self._missing(sub)
                else:
                    self._compare(actual[key], expected[key], sub)
        elif _is_name_list(actual) and _is_name_list(expected):
            own, replaced = _own_entries(path, actual, expected)
            for member in expected:
                if member not in actual and member not in replaced:
                    self._missing_member(path, member)
            for member in actual:
                if member not in expected and member not in own:
                    self._extra_member(path, member)
        elif actual != expected:
            self.findings.append(
                DriftFinding(
                    _dotted(path),
                    f"{self.label} has {_short_repr(actual)}, the preset has "
                    f"{_short_repr(expected)}",
                    self._profile_ref(path),
                    self._preset_ref(path),
                    _token(path),
                    self.lines.line(path),
                    kind="value",
                )
            )

    def _extra(self, path: tuple[str, ...]) -> None:
        """A key the profile sets and the preset does not.

        Reported under ``config:`` exactly as anywhere else: the preset carries
        the whole configuration a deployment renders, so a key it does not
        carry is one nothing documents — a typo, or a knob that has been
        retired — rather than a layer beneath the preset answering for it.
        """
        self.findings.append(
            DriftFinding(
                _dotted(path),
                f"set by {self.label}, unknown to the preset",
                self._profile_ref(path),
                f"preset {self.preset}",
                _token(path),
                self.lines.line(path),
                kind="extra",
            )
        )

    def _missing(self, path: tuple[str, ...]) -> None:
        self.findings.append(
            DriftFinding(
                _dotted(path),
                f"set by the preset, absent from {self.label}",
                self.label,
                self._preset_ref(path),
                _token(path),
                None,
                kind="missing",
            )
        )

    def _missing_member(self, path: tuple[str, ...], member: str) -> None:
        # A document subtracts a member with `exclude:`; that line, when it
        # exists, is where the operator would write the marker.
        line = self.lines.line(("exclude", *path, member))
        self.findings.append(
            DriftFinding(
                f"{_dotted(path)}: {member}",
                f"selected by the preset, not by {self.label}",
                self._profile_ref(("exclude", *path, member)) if line else self.label,
                self._preset_ref((*path, member)),
                member,
                line,
                # An `exclude:` line is a subtraction the profile spells out;
                # without one the member is simply not there. Both are the
                # preset selecting something this document does not.
                kind="exclude" if line else "missing_member",
            )
        )

    def _extra_member(self, path: tuple[str, ...], member: str) -> None:
        self.findings.append(
            DriftFinding(
                f"{_dotted(path)}: {member}",
                f"selected by {self.label}, not by the preset",
                self._profile_ref((*path, member)),
                f"preset {self.preset}",
                member,
                self.lines.line((*path, member)),
                kind="extra_member",
            )
        )

    def _profile_ref(self, path: tuple[str, ...]) -> str:
        line = self.lines.line(path)
        return f"{self.label}:{line}" if line else self.label

    def _preset_ref(self, path: tuple[str, ...]) -> str:
        for lines in self.preset_lines:
            line = lines.line(path)
            if line:
                return f"{lines.path.name}:{line}"
        return f"preset {self.preset}"


class _Lines:
    """The 1-based line of every key and list member in one YAML file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.root: Any = YAML(typ="rt").load(path.read_text(encoding="utf-8"))
        except Exception:  # an unreadable file simply has no lines to cite
            self.root = None

    def line(self, path: tuple[str, ...]) -> int | None:
        """Where *path* is written, whichever legal spelling it uses.

        A ``config:`` leaf may be one dotted key, a nested mapping, or any
        split between the two, so every cut of the segments is tried in turn.
        """
        return _locate(self.root, list(path))


def _locate(node: Any, segments: list[str]) -> int | None:
    if not segments:
        return None
    if isinstance(node, CommentedMap):
        for cut in range(1, len(segments) + 1):
            head = ".".join(segments[:cut])
            if head not in node:
                continue
            rest = segments[cut:]
            if not rest:
                return int(node.lc.key(head)[0]) + 1
            found = _locate(node[head], rest)
            if found is not None:
                return found
        # A block path the file spells only as dotted leaves (``control_system``
        # under ``control_system.writes: never``) is cited at its first leaf.
        prefix = ".".join(segments) + "."
        for key in node:
            if isinstance(key, str) and key.startswith(prefix):
                return int(node.lc.key(key)[0]) + 1
        return None
    if isinstance(node, CommentedSeq) and len(segments) == 1:
        for index, item in enumerate(node):
            if isinstance(item, str) and item == segments[0]:
                return int(node.lc.item(index)[0]) + 1
    return None


def _apply_markers(
    findings: list[DriftFinding], path: Path, root_dir: Path, tag: str
) -> tuple[list[DriftFinding], list[str]]:
    """Claim findings with the ``# <tag>:`` comments in *path*; name the rest.

    A marker reaches a finding that sits within :data:`MARKER_REACH` lines
    below it, and any finding whose token it names. A marker reaching nothing
    is stale: the line it described has come back into step with the preset,
    or it has drifted too far from that line.
    """
    pattern = re.compile(rf"^\s*#\s*{re.escape(tag)}:\s*(?P<text>.*)$")
    markers: list[tuple[int, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = pattern.match(line)
        if match:
            markers.append((number, match.group("text")))

    used: set[int] = set()
    claimed: list[DriftFinding] = []
    for finding in findings:
        covering = [number for number, text in markers if _covers(number, text, finding)]
        used.update(covering)
        claimed.append(replace(finding, marked=bool(covering)))

    relative = path.relative_to(root_dir).as_posix()
    stale = [
        f"{relative}:{number}: `# {tag}:` marks no difference from the preset — the line it "
        f"describes matches again, or sits more than {MARKER_REACH} lines below it"
        for number, _text in markers
        if number not in used
    ]
    return claimed, stale


def _covers(marker_line: int, text: str, finding: DriftFinding) -> bool:
    if finding.line is not None and 1 <= finding.line - marker_line <= MARKER_REACH:
        return True
    return re.search(rf"(?<![\w.-]){re.escape(finding.token)}(?![\w.-])", text) is not None


def lacking_config_keys(profile_raw: Mapping[str, Any], preset: str) -> list[str]:
    """Every ``config:`` leaf *preset* carries that *profile_raw* does not spell.

    The missing half of the comparison :class:`_Comparison` reports, asked the
    way ``osprey init --revalidate`` fills a profile: not which branches differ but
    which leaves are absent, flattened, because the fill writes one dotted key per leaf
    with the preset's own comment above it. A branch the profile lacks whole —
    an entire ``approval`` section — comes back as its leaves rather than as
    its root, so what lands in the profile is what the preset documents rather
    than one opaque mapping.

    Both sides are flattened to the dotted keys they address, so any split
    between dotted and nested counts as present: a leaf the preset spells
    ``a.b.c``, a profile nesting it whole and a profile spelling it ``{a:
    {"b.c": ...}}`` are one rendered key, and the profile already has it.

    Args:
        profile_raw: The profile document as it sits on disk, unresolved — a
            profile old enough to need expanding is one that may not resolve.
        preset: A bundled preset, in any CLI spelling.

    Returns:
        Dotted keys into the rendered config, in the preset's own order, so a
        caller writing them out follows the document it copies from.

    Raises:
        BuildProfileError: When the preset is unknown or will not emit.
    """
    name = str(profile_raw.get("name", ""))
    # Only which keys the reference carries is read here, never their values.
    expected = materialized_profile(
        _normalize_preset_name(preset),
        project_name=str(profile_raw.get("project_name") or ""),
        profile_name=name,
    )
    # Flattening each side to its dotted leaves normalizes every spelling at
    # once, which nesting the dots does not: a mapping only ever splits the
    # dotted keys it holds directly, so a dot inside a nested key survives it
    # and the same rendered leaf reads as two different ones.
    present = {_dotted(path) for path, _ in _dotted_leaves(_mapping(profile_raw.get("config")))}
    return [
        key
        for key in (_dotted(path) for path, _ in _dotted_leaves(_mapping(expected.get("config"))))
        if key not in present
    ]


def _emitted_version(profile_path: Path) -> str | None:
    """The OSPREY version the emitted header names, if the header is still there."""
    for line in profile_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        match = re.search(r"emitted by OSPREY (\S+)", line)
        if match:
            return match.group(1)
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _own_entries(
    path: tuple[str, ...], actual: list[str], expected: list[str]
) -> tuple[set[str], set[str]]:
    """The deployment's own entries in a top-level list, and the preset entries they replace.

    A path entry (``skills/beam-check``) names a file the repo ships, which no
    preset can carry, so it is the repo writing for itself rather than drift. A
    path that names the deployment's copy of a library artifact the preset
    selects (``rules/facility`` for ``facility``) stands in for that entry.

    Returns:
        ``(own, replaced)``: the path entries of *actual*, and the bare entries
        of *expected* one of them replaces.
    """
    from .profile_conventions import library_entry_shadow, split_convention_selection

    if len(path) != 1:
        return set(), set()
    own = set(split_convention_selection(path[0], actual)[1])
    replaced = {member for member in expected if library_entry_shadow(path[0], member) in own}
    return own, replaced


def _is_name_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _dotted(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _token(path: tuple[str, ...]) -> str:
    """What a marker names for a key finding: the key as the profile spells it."""
    return _dotted(path[1:]) if path[0] == "config" and len(path) > 1 else _dotted(path)


def _short_repr(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _short_hash(digest: str) -> str:
    return digest if len(digest) <= 19 else digest[:19] + "…"
