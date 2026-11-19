"""Writing the screens to disk, and checking a written set against the model.

:func:`write` draws every target of a graph (or the subset ``only`` keeps) as a
Screen-IR file and records the set in ``index.json``; :func:`check` regenerates
each screen and compares it with the file on disk. Neither holds the whole set
in memory: a file is drawn, written and forgotten, and the index keeps one small
entry per screen.

Layout under the output directory ``out``::

    out/index.json
    out/<view>/<screen id>.screen.json

``<view>`` is the target's view kind (``card``, ``place``, ...). The suffix is
the one the canopy library reads, so ``out`` can be copied under a library's
``screens/`` directory (without ``index.json``, which the library refuses).

Screen ids
    :func:`screen_id` gives a lowercase ``[a-z0-9-]`` slug of the thing's id,
    prefixed by the view kind (and the roll-up grouping). Two different
    addresses can have one slug (ids that differ only in case or punctuation,
    a device and a group with one id), so :func:`screen_ids` gives ids to a
    whole set and appends the 8-hex digest of the full address to every member
    of a colliding group. The result does not depend on the order of the
    targets. A slug over :data:`SLUG_MAX` characters is cut and gets the
    digest.

Links
    A link's ``file`` is the target's path relative to the directory of the
    file that holds it, in the form the library's own screens use
    (``../card/xr01-qf1.screen.json``). A link to a target outside the written
    subset is still written.

The stamp
    The screen format has no free metadata field, but ``metadata.description``
    is free text, so every file ends its description with
    ``Facility <sha256>, panels format <N>.``. ``index.json`` records the same
    two values as ``facility_hash`` and ``format``. The hash is
    :attr:`~osprey.facility.panels.model.PanelGraph.content_hash`. No file
    holds a time, an absolute path or a host name, so equal inputs give equal
    bytes on any machine.

Ownership
    ``write`` deletes only files its earlier run listed in ``index.json`` and
    that are no target's file any more. A non-empty directory with no index is
    an error.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import posixpath
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from osprey.facility.panels import layout
from osprey.facility.panels.describe import (
    NO_CLASS,
    NOT_PLACED,
    describe,
    drawn_on,
    light,
    targets,
    unset_groups,
)
from osprey.facility.panels.health import supported
from osprey.facility.panels.model import SIMULATION_PREFIX, PanelGraph
from osprey.facility.panels.symbols import WIDTH_UNITS, near_state
from osprey.facility.panels.view import VIEW_KINDS, Node, Target, View

__all__ = [
    "FORMAT_VERSION",
    "INDEX",
    "SCREEN_SUFFIX",
    "SLUG_MAX",
    "Draw",
    "DRAW_FILE",
    "ID_MAX",
    "EmitError",
    "Problem",
    "check",
    "load_draw",
    "named_classes",
    "picture_classes",
    "unplaced_parts",
    "screen_id",
    "screen_ids",
    "under",
    "write",
]

#: Bumped when a change of the generator changes the files it writes.
FORMAT_VERSION = 1
#: The index file, at the root of the output directory.
INDEX = "index.json"
#: The suffix of a screen file (the canopy library's).
SCREEN_SUFFIX = ".screen.json"
#: The longest slug of a thing's id kept in a screen id; a longer one is cut.
SLUG_MAX = 64
#: Hex digits of the digest appended to an id.
DIGEST_LEN = 8
#: Failing problem kinds are all but these.
NOTE_KINDS = frozenset({"outside_set"})
#: View kinds whose ``over_budget`` is a note: drawn in full, the excess is reported.
NOTED_OVER_BUDGET = frozenset({"card", "channels", "checkout"})

_STAMP = re.compile(r" Facility ([0-9a-f]{64}), panels format (\d+), draw ([0-9a-f]{16})\.$")
_STAMP_IN_TEXT = re.compile(r" Facility [0-9a-f]{64}, panels format \d+, draw [0-9a-f]{16}\.\"")
#: The longest screen id :func:`screen_ids` can make: ``kind-by-slug-digest``,
#: the digest lengthened to its longest (a SHA-256 in hex).
ID_MAX = (
    max(map(len, VIEW_KINDS)) + 1 + max(map(len, ("place", "system"))) + 2 * (1 + SLUG_MAX) + 1 + 64
)
#: The only paths ``write`` creates or deletes below ``out``.
_FILE_SHAPE = re.compile(
    rf"(?:{'|'.join(VIEW_KINDS)})/[a-z0-9-]{{1,{ID_MAX}}}{re.escape(SCREEN_SUFFIX)}"
)
#: A temporary file :func:`_atomic` leaves behind when a run is killed mid-write:
#: ``.<name>.<random>.tmp`` (``atomic_replace``), or ``.<name>.tmp`` (earlier runs).
_TEMPORARY = re.compile(r"\.[a-z0-9-]+" + re.escape(SCREEN_SUFFIX) + r"(?:\.[a-z0-9_]+)?\.tmp")
_INDEX_TEMPORARY = re.compile(r"\.index\.json(?:\.[a-z0-9_]+)?\.tmp")
#: Screens and the index are readable by all, writable by the owner.
_FILE_MODE = 0o644
#: The drawing settings file, beside the rule tables.
DRAW_FILE = "draw.json"
_NON_SLUG = re.compile(r"[^a-z0-9]+")
_FORMULA_PREFIX = "formula://"
_LOCAL_PREFIX = "loc://"


class EmitError(Exception):
    """The output directory cannot be written as asked."""


@dataclass(frozen=True)
class Problem:
    """One thing :func:`check` found.

    Attributes:
        kind: ``stale``, ``missing``, ``orphan``, ``dangling_link``,
            ``unknown_channel``, ``stamp``, ``invalid``, ``oversize``,
            ``incomplete`` (an earlier write did not finish), ``unrolled``
            (a channel with a rule a light can read that no roll-up screen
            reads, nor a screen one of its rows opens),
            ``static_rollup`` (a roll-up that reads a rule a light can read,
            has no live light, and has a child, or a root without children,
            with no screen of the set to open), ``long_formula`` (an expression the formula engine
            would refuse as too long), ``over_budget`` (more rules or bound
            channels than one screen may have), ``unbound_probe`` (a note: a probe
            row of a device card names a signal no channel states), ``unbound_picture`` (a note: a
            pictures row's signal resolves to no channel on a device),
            ``ambiguous_picture`` (a note: it matches channels of several devices),
            ``unset_group`` (a note: a family that is no set only because some of its
            members are no device of a stated class) or ``outside_set``.
        file: The file the problem is in, relative to the output directory.
        detail: What is wrong, in one line.
    """

    kind: str
    file: str
    detail: str
    note: bool = False

    @property
    def failing(self) -> bool:
        """False for a note: a screen a scoped write names without a link (or a link
        to a target outside the written set), or a card or
        checkout over its budget (:data:`NOTED_OVER_BUDGET`)."""
        return not self.note and self.kind not in NOTE_KINDS


@dataclass(frozen=True)
class Draw:
    """The parameters a screen is described with; ``write`` and ``check`` must agree on them."""

    colours: Mapping[str, str] = field(default_factory=dict)
    annotate: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The places a write keeps to (:func:`write_scope`); empty: everything. The
    #: screens do not depend on it, so it is no part of :attr:`digest`.
    write_under: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        """16 hex digits naming these settings, whatever the order they were given in."""
        text = json.dumps(
            {
                "colours": sorted(self.colours.items()),
                "annotate": sorted((k, list(v)) for k, v in self.annotate.items()),
            },
            sort_keys=True,
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _strings(value: Any, name: str, path: Path) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EmitError(f"{path}: {name} must be a list of strings")
    return value


def load_draw(path: Path) -> Draw:
    """Read the drawing settings from a JSON file.

    The keys are ``colours`` (symbol name to
    the ``#rrggbb`` its strip symbols are filled with) and ``annotate`` (symbol
    name to at most :data:`MAX_ANNOTATIONS` stated signals whose values a strip
    shows under its symbols) and ``write_under``
    (place ids a write keeps to, :func:`write_scope`; each must be a place of the
    graph). All are optional.

    Raises:
        EmitError: The file is missing or unreadable, is not JSON, holds a key
            this reader does not know, a value of the wrong type, a symbol
            name the engine does not have, a colour that is no ``#rrggbb`` or
            that could be read as a state colour (:func:`symbols.near_state`).
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EmitError(f"{path} cannot be read as {DRAW_FILE}: {error}") from error
    if not isinstance(document, dict):
        raise EmitError(f"{path}: the file must hold one JSON object")
    keys = {"colours", "annotate", "write_under"}
    unknown = sorted(set(document) - keys)
    if unknown:
        raise EmitError(f"{path}: unknown key {', '.join(unknown)}")
    write_under = _strings(document.get("write_under", []), "write_under", path)
    known = {*WIDTH_UNITS, "generic"}
    colours = document.get("colours", {})
    if not isinstance(colours, dict) or not all(isinstance(v, str) for v in colours.values()):
        raise EmitError(f"{path}: colours must map symbol names to #rrggbb")
    theme = _theme_roles()
    for name, colour in sorted(colours.items()):
        if name not in known:
            raise EmitError(f"{path}: colours names {name}, which is not a symbol of the engine")
        if not _HEX.fullmatch(colour):
            raise EmitError(f"{path}: the colour of {name} is {colour!r}, not #rrggbb")
        state = near_state(colour, theme)
        if state is not None:
            raise EmitError(
                f"{path}: the colour of {name}, {colour}, is too near the state colour {state}"
            )
    annotate = document.get("annotate", {})
    if not isinstance(annotate, dict):
        raise EmitError(f"{path}: annotate must map symbol names to lists of signals")
    for name, signals in sorted(annotate.items()):
        _strings(signals, f"annotate {name}", path)
        if name not in known:
            raise EmitError(f"{path}: annotate names {name}, which is not a symbol of the engine")
        if len(signals) > MAX_ANNOTATIONS:
            raise EmitError(f"{path}: annotate {name} lists more than {MAX_ANNOTATIONS} signals")
    return Draw(
        colours={name: colour.lower() for name, colour in colours.items()},
        annotate={name: tuple(signals) for name, signals in annotate.items()},
        write_under=tuple(dict.fromkeys(write_under)),
    )


def check_draw(graph: PanelGraph, draw: Draw) -> None:
    """Hold the drawing settings to the graph: each place named in ``write_under``
    exists.

    Raises:
        EmitError: A place named in ``write_under`` is no place of the graph.
    """
    for place_id in draw.write_under:
        try:
            graph.thing(place_id, "place")
        except KeyError:
            raise EmitError(
                f"write_under names {place_id}, which is no place of the facility"
            ) from None


_HEX = re.compile(r"#[0-9a-fA-F]{6}")
#: The most signals a symbol shows the values of on a strip.
MAX_ANNOTATIONS = 2
#: The theme the panels' screens are drawn for, shipped with OSPREY.
THEME_FILE = Path(__file__).parents[2] / "templates/apps/demo/data/canopy/theme/osprey.json"


def _theme_roles() -> Mapping[str, Mapping[str, str]]:
    """The shipped theme's roles (name to its light and dark colour); none when absent."""
    try:
        roles: Mapping[str, Mapping[str, str]] = json.loads(
            THEME_FILE.read_text(encoding="utf-8")
        ).get("roles", {})
        return roles
    except (OSError, ValueError):
        return {}


# -- ids ------------------------------------------------------------------------------


def _slug(text: str) -> str:
    return _NON_SLUG.sub("-", text.lower()).strip("-") or "x"


def _digest(*parts: str, length: int = DIGEST_LEN) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:length]


def _stem(kind: str, thing_id: str, by: str | None, system: str | None = None) -> tuple[str, bool]:
    """The readable id, and whether a slug had to be cut."""
    slugs = [_slug(text) for text in (thing_id, system) if text is not None]
    cut = any(len(slug) > SLUG_MAX for slug in slugs)
    slugs = [slug[:SLUG_MAX].rstrip("-") for slug in slugs]
    return "-".join(part for part in (kind, by, *slugs) if part), cut


def screen_id(kind: str, thing_id: str, by: str | None = None, system: str | None = None) -> str:
    """The readable screen id of one address.

    Args:
        kind: The view kind.
        thing_id: The thing's id.
        by: A roll-up's grouping.
        system: A roll-up's one system.

    Returns:
        ``kind[-by]-slug`` in lowercase ``[a-z0-9-]``; a slug over
        :data:`SLUG_MAX` characters is cut and the digest of the address
        follows. Different ids can still give one result; use
        :func:`screen_ids` for a set.
    """
    stem, cut = _stem(kind, thing_id, by, system)
    if not cut:
        return stem
    parts = (kind, thing_id, by or "") if system is None else (kind, thing_id, by or "", system)
    return f"{stem}-{_digest(*parts)}"


def _identity(target: Target) -> tuple[str, ...]:
    identity = (target.view, target.kind, target.id, target.by or "")
    return identity if target.system is None else (*identity, target.system)


def screen_ids(found: list[Target] | tuple[Target, ...]) -> dict[Target, str]:
    """A screen id for every target, no two alike, whatever the order.

    Every member of a group that would collide, and every cut slug, gets the
    8-hex digest of its full address (view, thing kind, id, grouping). Should
    two digests collide as well, the digests of that group are lengthened.
    """
    stems = {}
    for target in found:
        stems[target] = _stem(target.view, target.id, target.by, target.system)
    groups: dict[str, list[Target]] = {}
    for target, (stem, _) in stems.items():
        groups.setdefault(stem, []).append(target)
    ids: dict[Target, str] = {}
    for stem, members in groups.items():
        crowded = len(members) > 1
        for target in members:
            if crowded or stems[target][1]:
                ids[target] = f"{stem}-{_digest(*_identity(target))}"
            else:
                ids[target] = stem
    length = DIGEST_LEN
    while True:
        seen: dict[str, list[Target]] = {}
        for target, id_ in ids.items():
            seen.setdefault(id_, []).append(target)
        clashes = [members for members in seen.values() if len(members) > 1]
        if not clashes:
            return ids
        length *= 2
        for members in clashes:
            for target in members:
                ids[target] = f"{stems[target][0]}-{_digest(*_identity(target), length=length)}"


def _path(target: Target, ids: Mapping[Target, str]) -> str:
    """The file of a target, relative to the output directory, always with ``/``."""
    id_ = ids.get(target)
    if id_ is None:
        id_ = screen_id(target.view, target.id, target.by, target.system)
    return f"{target.view}/{id_}{SCREEN_SUFFIX}"


def under(graph: PanelGraph, place_id: str) -> Callable[[Target], bool]:
    """A test for the targets of a place and everything within it.

    Args:
        graph: The panel graph.
        place_id: A place.

    Returns:
        A function true for a target whose thing is the place or a device or
        place of its contents (:meth:`PanelGraph.contents`), and for the
        calibration views of a set with a member there. Other views of a group
        are never under a place.

    Raises:
        KeyError: No such place.
    """
    within = {place_id, *(thing.id for thing in graph.contents(place_id))}
    sets = {
        thing.id
        for thing in graph.things()
        if thing.kind == "group" and any(m.id in within for m in graph.members(thing.id))
    }

    def test(target: Target) -> bool:
        if target.kind == "group":
            return target.view == "calibration" and target.id in sets
        return target.id in within

    return test


@dataclass(frozen=True)
class Scope:
    """What a write kept to ``write_under`` holds (:func:`write_scope`)."""

    places: frozenset[str]
    devices: frozenset[str]
    groups: frozenset[str]
    #: The classes of the kept devices in no place.
    loose_classes: frozenset[str]
    #: The places the write was asked to keep to.
    under: tuple[str, ...] = ()
    #: Each kept group's members cut to the kept devices, in the group's order.
    members: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The groups cut to the same members as a kept one: written once, as that one.
    dropped: frozenset[str] = frozenset()

    def __call__(self, target: Target) -> bool:
        if target.view == "home":
            return target.by is None or target.system in self.loose_classes
        if target.kind == "device":
            return target.id in self.devices
        if target.kind == "group":
            return target.id in self.groups
        if target.id == NOT_PLACED:
            return bool(self.loose_classes)
        return target.id in self.places

    def holds(self, thing_id: str | None) -> bool:
        """Whether a device or place of this id is kept."""
        return thing_id in self.devices or thing_id in self.places


def write_scope(graph: PanelGraph, places: Iterable[str]) -> Scope:
    """What a write kept to *places* writes: the home screen; each place and the places
    within it; the devices placed there (:meth:`PanelGraph.contents`); each device in no
    place that one relation joins to a kept device, with the not-placed screens of its
    class; and each group with a kept device as a member. The other screens are not
    written; links to them still are, and the index lists them under ``outside``.

    Raises:
        KeyError: A place that is no place of the graph.
    """
    kept_places: set[str] = set()
    placed: set[str] = set()
    for place_id in places:
        graph.thing(place_id, "place")
        kept_places.add(place_id)
        for thing in graph.contents(place_id):
            (kept_places if thing.kind == "place" else placed).add(thing.id)
    loose: set[str] = set()
    classes: set[str] = set()
    for thing in graph.unplaced():
        joined = any(
            edge.other_kind == "device" and edge.other in placed
            for edges in graph.neighbours(thing.id, "device").values()
            for edge in edges
        )
        if joined:
            loose.add(thing.id)
            classes.add(thing.cls or NO_CLASS)
    devices = placed | loose
    cut: dict[str, tuple[str, ...]] = {}
    whole: set[str] = set()  # the groups all of whose members are kept
    for thing in graph.things():
        if thing.kind != "group":
            continue
        members = [m.id for m in graph.members(thing.id)]
        kept = tuple(m for m in members if m in devices or m in kept_places)
        if kept:
            cut[thing.id] = kept
            if len(kept) == len(members):
                whole.add(thing.id)
    # a group cut to the same members as another is written once: the one the
    # facility defined whole within the scope, else the first by id
    by_members: dict[frozenset[str], list[str]] = {}
    for group, kept in cut.items():
        by_members.setdefault(frozenset(kept), []).append(group)
    groups = {
        min(same, key=lambda group: (group not in whole, group)) for same in by_members.values()
    }
    return Scope(
        frozenset(kept_places),
        frozenset(devices),
        frozenset(groups),
        frozenset(classes),
        tuple(places),
        MappingProxyType({group: cut[group] for group in sorted(groups)}),
        frozenset(cut) - frozenset(groups),
    )


def scoped(graph: PanelGraph, scope: Scope) -> PanelGraph:
    """*graph* as a write kept to *scope* describes it: each kept group holds only its
    kept members (:attr:`Scope.members`), a dropped duplicate (:attr:`Scope.dropped`)
    none, so no screen of the write links to it rather than to the kept one, and the
    home and the not-placed indexes list only what the scope keeps (the graph carries
    it as ``write_scope``)."""
    out = copy.copy(graph)
    out._members = {
        **graph._members,
        **scope.members,
        **dict.fromkeys(scope.dropped, ()),
    }
    out.write_scope = scope
    return out


# -- drawing --------------------------------------------------------------------------


def _stamp(graph: PanelGraph, draw: Draw) -> str:
    return f" Facility {graph.content_hash}, panels format {FORMAT_VERSION}, draw {draw.digest}."


def _generate(
    graph: PanelGraph, target: Target, ids: Mapping[Target, str], draw: Draw
) -> tuple[bytes, str, View, tuple[Target, ...]]:
    """The bytes of a target's file, its title, its view, and the screens it names
    without a link because a write kept to ``write_under`` does not produce them."""
    own = _path(target, ids)
    scope = graph.write_scope
    unlinked: list[Target] = []

    def screens(other: Target) -> str | None:
        if scope is not None and not scope(other):
            unlinked.append(other)
            return None  # a written screen links only to a file of the same write
        # Every file sits one directory below ``out``, so this is relative to it.
        return "../" + _path(other, ids)

    view = describe(
        graph,
        target,
        annotate=draw.annotate,
    )
    document = layout.render(
        view, graph=graph, screens=screens, doc_id=own[: -len(SCREEN_SUFFIX)], colours=draw.colours
    )
    document["metadata"]["description"] += _stamp(graph, draw)
    text = json.dumps(
        document, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return (text + "\n").encode("utf-8"), view.title, view, tuple(dict.fromkeys(unlinked))


def _atomic(path: Path, data: bytes) -> None:
    from osprey_connectors.safety_file import atomic_replace

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace(path, data, mode=_FILE_MODE)


def _guard(out: Path, relative: str, seen: set[str]) -> Path:
    """The path of *relative* below *out*, refused when a link lies on the way.

    *out* itself may be a link; nothing below it may be, the file included.

    Raises:
        EmitError: A component below *out* is a symbolic link.
    """
    parts = relative.split("/")
    current = out
    for number, part in enumerate(parts, 1):
        current = current / part
        key = "/".join(parts[:number])
        if number < len(parts) and key in seen:
            continue
        if current.is_symlink():
            raise EmitError(f"{current} is a symbolic link; refusing to go through it")
        if number < len(parts):
            seen.add(key)
    return out.joinpath(*parts)


def _put(out: Path, relative: str, data: bytes, seen: set[str]) -> Path:
    path = _guard(out, relative, seen)
    _atomic(path, data)
    return path


def _remove(out: Path, relative: str, seen: set[str]) -> None:
    """Delete one file this generator owns, then its directory when that is empty."""
    path = _guard(out, relative, seen)
    if path.is_file():
        path.unlink()
        parent = path.parent
        if parent != out and not any(parent.iterdir()):
            parent.rmdir()
            seen.discard(posixpath.dirname(relative))


def _read_index(out: Path) -> dict[str, Any] | None:
    path = out / INDEX
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as error:
        raise EmitError(f"{path} cannot be read: {error}") from error
    try:
        index = json.loads(text)
    except ValueError as error:
        raise EmitError(f"{path} is not JSON: {error}") from error
    if (
        not isinstance(index, dict)
        or not isinstance(index.get("screens"), list)
        or not all(isinstance(entry, dict) for entry in index["screens"])
        or not isinstance(index.get("outside", []), list)
    ):
        raise EmitError(f"{path} is not an index this generator wrote")
    return index


def _clean_temporaries(out: Path) -> None:
    """Remove the half-written files of an interrupted run (ours by name, nothing else)."""
    stale = [out / n for n in os.listdir(out) if _INDEX_TEMPORARY.fullmatch(n)]
    for view in VIEW_KINDS:
        directory = out / view
        if directory.is_symlink():
            raise EmitError(f"{directory} is a symbolic link; refusing to go through it")
        if directory.is_dir():
            stale += [directory / n for n in os.listdir(directory) if _TEMPORARY.fullmatch(n)]
    for path in stale:
        if path.is_file() or path.is_symlink():
            path.unlink()


def _entry(target: Target, ids: Mapping[Target, str], title: str, size: int) -> dict[str, Any]:
    return {
        "id": ids[target],
        "kind": target.view,
        "thing_kind": target.kind,
        "thing": target.id,
        "by": target.by,
        "system": target.system,
        "title": title,
        "file": _path(target, ids),
        "bytes": size,
    }


def write(
    graph: PanelGraph,
    out: Path,
    *,
    only: Callable[[Target], bool] | None = None,
    draw: Draw = Draw(),
) -> list[Path]:
    """Write the screens of the graph under *out*.

    Each file is written to a temporary name and renamed, so a reader never
    sees half a file. Files an earlier run listed in ``index.json`` that are no
    target's file any more are deleted; files that run did not list never are.
    Files of earlier runs that are still targets stay, listed with their old
    size, so subset writes add up.

    Args:
        graph: The panel graph.
        out: The output directory; created when absent.
        only: Keeps the targets to write; ``None`` writes them all. Links to
            the others are still written, and the index lists them under
            ``outside``.
        draw: How a screen is described.

    Returns:
        The screen files written, in target order (not ``index.json``).

    Raises:
        EmitError: *out* holds files and no ``index.json``, the index is not
            one this generator wrote, or *draw* does not hold for the graph
            (:func:`check_draw`).
    """
    check_draw(graph, draw)
    if draw.write_under:
        scope = write_scope(graph, draw.write_under)
        graph = scoped(graph, scope)
        given = only
        only = scope if given is None else (lambda t: scope(t) and given(t))
    out = Path(out)
    previous = _read_index(out)
    if previous is None and out.exists() and any(out.iterdir()):
        raise EmitError(f"{out} is not empty and holds no {INDEX}; refusing to write into it")
    seen: set[str] = set()
    # The index is input from the disk: only paths of our own shape count as ours.
    owned: dict[str, dict[str, Any]] = {}
    ignored: list[str] = []
    for entry in (previous or {}).get("screens", ()):
        name = entry.get("file")
        if isinstance(name, str) and _FILE_SHAPE.fullmatch(name):
            owned[name] = entry
        else:
            ignored.append(str(name))
    found = targets(graph)
    ids = screen_ids(found)
    paths = {_path(t, ids) for t in found}
    # Pre-flight: every path the run will write, delete or prune is looked at
    # before the first byte is written. A link is an error; a listed name the
    # file system cannot inspect is ignored and not carried on.
    _guard(out, INDEX, seen)
    for relative in sorted(paths | owned.keys()):
        try:
            _guard(out, relative, seen)
        except OSError as error:
            if relative in paths:
                raise EmitError(f"{out / relative} cannot be inspected: {error}") from error
            del owned[relative]
            ignored.append(relative)
    if ignored:
        shown = ", ".join(ignored[:5]) + (" ..." if len(ignored) > 5 else "")
        warnings.warn(
            f"{out / INDEX} lists {len(ignored)} files that are no screen files of this "
            f"generator; they were left alone: {shown}",
            UserWarning,
            stacklevel=2,
        )
    if out.is_dir():
        _clean_temporaries(out)
    head = {
        "format": FORMAT_VERSION,
        "facility_hash": graph.content_hash,
        "draw_hash": draw.digest,
    }

    # a device with no card of its own: the screen of the owner it is drawn on
    written_ids = set(found)
    owners = [
        {"thing": thing, "file": _path(target, ids)}
        for thing, target in drawn_on(graph)
        if target in written_ids
    ]

    def put_index(state: str, screens: list[dict[str, Any]], outside: list[dict[str, Any]]) -> None:
        index = {
            **head,
            "state": state,
            "total_bytes": sum(e.get("bytes", 0) for e in screens),
            "screens": sorted(screens, key=lambda e: e["file"]),
            "outside": sorted(outside, key=lambda e: e["file"]),
            "drawn_on": owners,
        }
        text = json.dumps(index, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        _put(out, INDEX, text.encode("utf-8"), seen)

    # First the index in the "writing" state, listing every file this run may
    # leave: an interrupted run is then finished, not leaked, by the next one.
    announced = dict(owned)
    for target in found:
        announced.setdefault(_path(target, ids), _entry(target, ids, "", 0))
    put_index("writing", list(announced.values()), [])

    entries: dict[str, dict[str, Any]] = {}
    written: list[Path] = []
    outside: list[dict[str, Any]] = []
    for target in found:
        relative = _path(target, ids)
        if only is None or only(target):
            data, title, _view, _unlinked = _generate(graph, target, ids, draw)
            written.append(_put(out, relative, data, seen))
            entries[relative] = _entry(target, ids, title, len(data))
        elif relative in owned and _guard(out, relative, seen).is_file():
            old = owned[relative]
            entries[relative] = _entry(
                target, ids, str(old.get("title", "")), (out / relative).stat().st_size
            )
        else:
            outside.append(
                {k: v for k, v in _entry(target, ids, "", 0).items() if k not in {"title", "bytes"}}
            )
    for relative in sorted(owned.keys() - paths):
        _remove(out, relative, seen)
    put_index("complete", list(entries.values()), outside)
    return written


# -- check ----------------------------------------------------------------------------


def _bound_channels(document: Any) -> Iterator[str]:
    """Every channel a screen binds: a widget's ``pv``, a curve's ``y_channel``, each
    name in a rule's ``pvs`` but a screen formula's, and each binding of a screen
    formula; a screen's own local value (``loc://``, a typed value before Apply) is
    no channel."""
    yield from (name for name in _bound_names(document) if not name.startswith(_LOCAL_PREFIX))


def _bound_names(document: Any) -> Iterator[str]:
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key in ("pv", "y_channel"):
                value = node.get(key)
                if isinstance(value, str) and not value.startswith(_FORMULA_PREFIX):
                    yield value  # a screen formula's channels are its bindings
            pvs = node.get("pvs")
            if isinstance(pvs, list):
                for item in pvs:
                    name = item.get("name") if isinstance(item, dict) else None
                    if isinstance(name, str) and not name.startswith(_FORMULA_PREFIX):
                        yield name
            bindings = node.get("bindings")
            if isinstance(bindings, dict):
                yield from (name for name in bindings.values() if isinstance(name, str))
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _carried(graph: PanelGraph, channel: str) -> bool:
    """Whether a light can read the channel: its rule writes an arm, or would but is too long.

    A channel whose rows the engine cannot evaluate, or whose rule can give no
    state but ``unknown`` (no arm), or whose rows no condition can say (a
    label), has nothing a light could read.
    """
    if not any(supported(row.value) for row in graph.rule(channel)):
        return False
    rule = layout.ladder(graph, [light(graph, channel)])
    if isinstance(rule, str):
        return rule != "label condition"
    return bool(rule.arms)


def _reads(graph: PanelGraph, view: View) -> bool:
    """Whether a roll-up view's root reads a channel a light can read (:func:`_carried`);
    a roll-up of none is not static."""
    return any(
        _carried(graph, channel)
        for section in view.sections
        for item in section.items
        if isinstance(item, Node)
        for channel in item.channels
    )


def _unopened(
    graph: PanelGraph, view: View, ids: Mapping[Target, str], valid: set[str]
) -> list[str]:
    """The nodes of a roll-up that read a rule a light can read and have no screen of
    the set to open: each child of the tree, or the root when it has none. A child
    that opens its own screen is a drill-down, not a static roll-up."""
    found: list[str] = []
    for section in view.sections:
        for item in section.items:
            if not isinstance(item, Node):
                continue
            for node in item.children or (item,):
                if not any(_carried(graph, c) for c in node.channels):
                    continue
                if node.link is None or _path(node.link.target, ids) not in valid:
                    found.append(node.label)
    return found


def _live(document: Any) -> bool:
    """Whether any widget of a document carries a rule."""
    stack = [document.get("root", {})] if isinstance(document, dict) else []
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("rules"):
                return True
            stack.extend(node.get("children") or ())
    return False


def _over_budget(document: Any, relative: str) -> list[Problem]:
    """An ``over_budget`` problem when the screen subscribes more rules, or binds more
    channels, than one screen may (:data:`layout.SCREEN_RULE_BUDGET`,
    :data:`layout.SCREEN_CHANNEL_BUDGET`)."""
    rules = 0
    for node in _walk_nodes(document.get("root")):
        rules += sum(bool(rule.get("conditions")) for rule in node.get("rules") or ())
    channels = len(set(_bound_channels(document)))
    note = relative.split("/", 1)[0] in NOTED_OVER_BUDGET
    found = []
    if rules > layout.SCREEN_RULE_BUDGET:
        detail = f"{rules} rules subscribe, over {layout.SCREEN_RULE_BUDGET}"
        found.append(Problem("over_budget", relative, detail, note))
    if channels > layout.SCREEN_CHANNEL_BUDGET:
        detail = f"{channels} channels bound, over {layout.SCREEN_CHANNEL_BUDGET}"
        found.append(Problem("over_budget", relative, detail, note))
    return found


def _links(document: Any) -> Iterator[str]:
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            ref = node.get("file")
            if isinstance(ref, str):
                yield ref
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _long_formulas(document: Any, relative: str) -> list[Problem]:
    """A ``long_formula`` problem per expression the engine would refuse as too long.

    That is a screen formula's own expression, and every rule's expression
    as the viewer compiles it (:func:`layout.engine_expression`).
    """
    limit = layout.RULE_EXPRESSION_LIMIT
    found: list[Problem] = []
    formulas = {
        f["name"]: f
        for f in document.get("formulas") or ()
        if isinstance(f, Mapping) and isinstance(f.get("expression"), str)
    }
    for name, formula in formulas.items():
        if len(formula["expression"]) > limit:
            detail = f"formula {name} is {len(formula['expression'])} characters, over {limit}"
            found.append(Problem("long_formula", relative, detail))
    for node in _walk_nodes(document.get("root")):
        for rule in node.get("rules") or ():
            if not rule.get("conditions"):
                continue
            try:
                length = len(layout.engine_expression(rule, formulas))
            except (KeyError, TypeError):
                continue  # a dangling formula is reported as a dangling link
            if length > limit:
                detail = f"rule {rule.get('id')} compiles to {length} characters, over {limit}"
                found.append(Problem("long_formula", relative, detail))
    return found


def _walk_nodes(node: Any) -> Iterator[Mapping[str, Any]]:
    if not isinstance(node, Mapping):
        return
    yield node
    for child in node.get("children") or ():
        yield from _walk_nodes(child)


def check(
    graph: PanelGraph,
    out: Path,
    *,
    max_bytes: int | None = None,
    draw: Draw = Draw(),
) -> list[Problem]:
    """Compare a written set with what the graph would generate now; writes nothing.

    Args:
        graph: The panel graph.
        out: The output directory.
        max_bytes: A screen over this many bytes is an ``oversize`` problem.
        draw: How a screen is described; the one ``write`` was given. With
            ``write_under``, a screen outside :func:`write_scope` may be absent,
            and only the health rules of kept devices and places must be read.

    Returns:
        The problems, in file order. ``outside_set`` ones are notes
        (:attr:`Problem.failing` is false).
    """
    out = Path(out)
    try:
        check_draw(graph, draw)
    except EmitError as error:
        return [Problem("invalid", DRAW_FILE, str(error))]
    try:
        index = _read_index(out)
    except (EmitError, ValueError) as error:
        return [Problem("invalid", INDEX, str(error))]
    if index is None:
        return [Problem("invalid", INDEX, "no index.json: nothing was written here")]
    problems: list[Problem] = []
    if index.get("state") != "complete":
        problems.append(
            Problem("incomplete", INDEX, "an earlier write did not finish; write again")
        )
    changed = [
        name
        for name, then, now in (
            ("facility", index.get("facility_hash"), graph.content_hash),
            ("format", index.get("format"), FORMAT_VERSION),
            ("draw settings", index.get("draw_hash"), draw.digest),
        )
        if then != now
    ]
    if changed:
        problems.append(
            Problem("stamp", INDEX, f"written from other {' and '.join(changed)} than now")
        )
    found = targets(graph)
    ids = screen_ids(found)
    wanted = {_path(t, ids): t for t in found}
    scope = write_scope(graph, draw.write_under) if draw.write_under else None
    if scope is not None:
        graph = scoped(graph, scope)
    listed = {e["file"]: e for e in index["screens"] if isinstance(e, dict) and "file" in e}
    outside = {e["file"] for e in index.get("outside", ()) if isinstance(e, dict) and "file" in e}
    problems.extend(
        Problem("orphan", relative, "listed in the index, and no screen of the model")
        for relative in sorted(listed.keys() - wanted.keys())
    )
    for view in VIEW_KINDS:
        directory = out / view
        if directory.is_symlink():
            problems.append(Problem("invalid", view, "is a symbolic link; its files are not read"))
        elif directory.is_dir():
            problems.extend(
                Problem("orphan", f"{view}/{name}", "a screen file the index does not list")
                for name in sorted(os.listdir(directory))
                if _FILE_SHAPE.fullmatch(f"{view}/{name}") and f"{view}/{name}" not in listed
            )

    def readable(relative: str) -> bool:
        path = out / relative
        return not path.parent.is_symlink() and not path.is_symlink() and path.is_file()

    valid = {r for r in wanted if r in listed and readable(r)}
    for relative, target in wanted.items():
        if relative not in listed and (
            relative in outside or (scope is not None and not scope(target))
        ):
            continue
        if relative not in listed or not readable(relative):
            problems.append(
                Problem("missing", relative, f"no file for {target.view} of {target.id}")
            )
    rolled: set[str] | None = set()
    opened: set[str] = set()
    for relative in wanted:
        if relative not in valid:
            if wanted[relative].view == "rollup" and (scope is None or scope(wanted[relative])):
                rolled = None  # a roll-up is not on disk: what is read cannot be told
            continue
        problems.extend(
            _check_file(
                graph, out, relative, wanted[relative], ids, valid, outside, max_bytes, draw
            )
        )
        problems.extend(_unbound_probe(graph, relative, wanted[relative]))
        problems.extend(_unbound_picture(graph, relative, wanted[relative]))
        if rolled is not None and wanted[relative].view == "rollup":
            document = json.loads((out / relative).read_bytes())
            rolled.update(_bound_channels(document))
            here = posixpath.dirname(relative)
            opened.update(posixpath.normpath(posixpath.join(here, ref)) for ref in _links(document))
    if rolled is not None:
        # a row with no live light opens the screen that reads its channels (one level)
        for relative in sorted(opened & valid):
            if wanted[relative].view != "rollup":
                try:
                    rolled.update(_bound_channels(json.loads((out / relative).read_bytes())))
                except (OSError, ValueError):
                    continue  # reported as invalid
        # a channel no light can read has no rule to read (see _carried)
        ruled = dict.fromkeys(row.channel for row in graph.rules())
        problems.extend(
            Problem(
                "unrolled",
                INDEX,
                f"{name} has a health rule and no roll-up screen, or screen one opens, reads it",
            )
            for name in ruled
            if name not in rolled
            and (scope is None or scope.holds(graph.channel(name).thing))
            and _carried(graph, name)
        )
    problems.extend(
        Problem(
            "unset_group",
            INDEX,
            f"{group} is no set: {_named(odd)} {'is' if len(odd) == 1 else 'are'} no device "
            "of a stated class, while its other members share one",
            note=True,
        )
        for group, odd in unset_groups(graph)
    )
    return problems


#: How many members an ``unset_group`` note names; the rest are counted.
NAMED_MEMBERS = 8


def _named(ids: tuple[str, ...]) -> str:
    shown = ", ".join(ids[:NAMED_MEMBERS])
    return shown + (f" and {len(ids) - NAMED_MEMBERS} more" if len(ids) > NAMED_MEMBERS else "")


#: How many class names a ``pictures`` note names; the rest are counted.
NAMED_CLASSES = 12


def picture_classes(graph: PanelGraph) -> dict[str, list[str]]:
    """The device classes with devices, by how ``pictures.csv`` states their drawing:
    ``drawing`` (a body: a shape row or an outline), ``parts only`` (parts at
    their positions, no body) and ``none`` (no row); each list sorted."""
    classes = sorted({t.cls for t in graph.things() if t.kind == "device" and t.cls})
    out: dict[str, list[str]] = {"drawing": [], "parts only": [], "none": []}
    for cls in classes:
        rows = graph.pictures(cls)
        if any(row.shape or row.kind == layout.OUTLINE_KIND for row in rows):
            out["drawing"].append(cls)
        elif rows:
            out["parts only"].append(cls)
        else:
            out["none"].append(cls)
    return out


def unplaced_parts(graph: PanelGraph) -> dict[str, int]:
    """Per device class with devices, how many of its ``pictures.csv`` parts are not
    placed (no position and no angle): drawn hollow beside the drawing. Information,
    not a problem: a facility states no number it has no source for."""
    classes = sorted({t.cls for t in graph.things() if t.kind == "device" and t.cls})
    out: dict[str, int] = {}
    for cls in classes:
        rows = graph.pictures(cls)
        count = sum(
            1 for r in rows if not r.shape and r.kind != layout.OUTLINE_KIND and not r.placed
        )
        if count:
            out[cls] = count
    return out


def named_classes(classes: list[str]) -> str:
    """A count of *classes* and the first :data:`NAMED_CLASSES` of them."""
    if not classes:
        return "0 classes"
    shown = ", ".join(classes[:NAMED_CLASSES])
    more = f" and {len(classes) - NAMED_CLASSES} more" if len(classes) > NAMED_CLASSES else ""
    return f"{len(classes)} {'class' if len(classes) == 1 else 'classes'} ({shown}{more})"


def _unbound_picture(graph: PanelGraph, relative: str, target: Target) -> list[Problem]:
    """A note per pictures row of a device card's class whose ``signal`` resolves to no
    channel on the device or its parts (:meth:`PanelGraph.picture_matches`): drawn
    plain; and one per row whose signal matches channels of more than one device of
    one level (two parts, say): the first is shown, which is no choice of the data."""
    if (target.view, target.kind) != ("card", "device"):
        return []
    cls = graph.thing(target.id, "device").cls
    if cls is None:
        return []
    found = []
    for row in graph.pictures(cls, target.id):
        if row.signal is None:
            continue
        matches = graph.picture_matches(target.id, row.signal)
        what = f"{target.id}: {row.view} view {row.kind} {row.part!r} names {row.signal}"
        owners = list(dict.fromkeys(device for device, _channel in matches))
        if not matches:
            detail = f"{what}, which resolves to no channel; it is drawn plain"
            found.append(Problem("unbound_picture", relative, detail, note=True))
        elif len(owners) > 1:
            detail = (
                f"{what}, which matches channels of {len(owners)} devices "
                f"({', '.join(owners)}); the first, {matches[0][1]}, is shown"
            )
            found.append(Problem("ambiguous_picture", relative, detail, note=True))
    return found


def _unbound_probe(graph: PanelGraph, relative: str, target: Target) -> list[Problem]:
    """A note per probe row of a device card whose signal no channel states for the
    device (:meth:`PanelGraph.signal_channels`): the element is drawn without its value."""
    if (target.view, target.kind) != ("card", "device"):
        return []
    found = []
    for row in graph.probe(target.id):
        for signal in (row.signal, row.signal_y):
            if signal is not None and not graph.signal_channels(target.id, signal):
                detail = (
                    f"{target.id}: probe element {row.element!r} names the signal "
                    f"{signal}, which no channel states; it is drawn without a value"
                )
                found.append(Problem("unbound_probe", relative, detail, note=True))
    return found


def _check_file(
    graph: PanelGraph,
    out: Path,
    relative: str,
    target: Target,
    ids: Mapping[Target, str],
    valid: set[str],
    outside: set[str],
    max_bytes: int | None,
    draw: Draw,
) -> list[Problem]:
    path = out / relative
    data = path.read_bytes()
    problems: list[Problem] = []
    if max_bytes is not None and len(data) > max_bytes:
        problems.append(Problem("oversize", relative, f"{len(data)} bytes, over {max_bytes}"))
    try:
        document = json.loads(data)
        description = document["metadata"]["description"]
    except (ValueError, KeyError, TypeError):
        return [*problems, Problem("invalid", relative, "not a screen document")]
    stamped = _STAMP.search(description) if isinstance(description, str) else None
    now = (graph.content_hash, str(FORMAT_VERSION), draw.digest)
    if stamped is None or stamped.groups() != now:
        changed = (
            [
                n
                for n, a, b in zip(
                    ("facility", "format", "draw settings"), stamped.groups(), now, strict=True
                )
                if a != b
            ]
            if stamped
            else ["stamp"]
        )
        problems.append(
            Problem("stamp", relative, f"written from other {' and '.join(changed)} than now")
        )
    problems += _long_formulas(document, relative)
    fresh, _title, view, unlinked = _generate(graph, target, ids, draw)
    problems.extend(
        Problem(
            "outside_set",
            relative,
            f"names {other.kind} {other.id} without a link: its {other.view} screen is "
            "outside the written set",
        )
        for other in unlinked
    )
    if target.view == "rollup" and _reads(graph, view) and not _live(document):
        unopened = _unopened(graph, view, ids, valid)
        if unopened:
            problems.append(
                Problem(
                    "static_rollup",
                    relative,
                    "reads rule channels with no live light; no screen to open for "
                    + ", ".join(unopened),
                )
            )
    problems += _over_budget(document, relative)
    if fresh != data:
        # The stamp is reported on its own; stale is what differs besides it.
        bare = _STAMP_IN_TEXT.sub('"', data.decode("utf-8", "replace"))
        if bare != _STAMP_IN_TEXT.sub('"', fresh.decode("utf-8")):
            problems.append(Problem("stale", relative, "differs from what the model generates now"))
    names = set()
    for name in _bound_channels(document):
        if name in names:
            continue
        names.add(name)
        try:
            graph.channel(name)
        except KeyError:
            why = (
                "a simulation-only channel no screen may bind"
                if name.startswith(SIMULATION_PREFIX)
                else "no channel of the facility"
            )
            problems.append(Problem("unknown_channel", relative, f"binds {name}, which is {why}"))
    here = posixpath.dirname(relative)
    for ref in dict.fromkeys(_links(document)):
        resolved = posixpath.normpath(posixpath.join(here, ref))
        if resolved in valid:
            continue
        kind = "outside_set" if resolved in outside else "dangling_link"
        why = "is outside the written set" if kind == "outside_set" else "is no file of the set"
        problems.append(Problem(kind, relative, f"links to {ref}, which {why}"))
    return problems
