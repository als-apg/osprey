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

import hashlib
import json
import os
import posixpath
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.facility.panels import layout
from osprey.facility.panels.describe import describe, targets
from osprey.facility.panels.model import SIMULATION_PREFIX, PanelGraph
from osprey.facility.panels.symbols import WIDTH_UNITS
from osprey.facility.panels.view import VIEW_KINDS, Target

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

_STAMP = re.compile(r" Facility ([0-9a-f]{64}), panels format (\d+), draw ([0-9a-f]{16})\.$")
_STAMP_IN_TEXT = re.compile(r" Facility [0-9a-f]{64}, panels format \d+, draw [0-9a-f]{16}\.\"")
#: The longest screen id :func:`screen_ids` can make: ``kind-by-slug-digest``,
#: the digest lengthened to its longest (a SHA-256 in hex).
ID_MAX = max(map(len, VIEW_KINDS)) + 1 + max(map(len, ("place", "system"))) + 1 + SLUG_MAX + 1 + 64
#: The only paths ``write`` creates or deletes below ``out``.
_FILE_SHAPE = re.compile(
    rf"(?:{'|'.join(VIEW_KINDS)})/[a-z0-9-]{{1,{ID_MAX}}}{re.escape(SCREEN_SUFFIX)}"
)
_TEMPORARY = re.compile(r"\.[a-z0-9-]+" + re.escape(SCREEN_SUFFIX) + r"\.tmp")
#: The drawing settings file, beside the rule tables.
DRAW_FILE = "draw.json"
_NON_SLUG = re.compile(r"[^a-z0-9]+")


class EmitError(Exception):
    """The output directory cannot be written as asked."""


@dataclass(frozen=True)
class Problem:
    """One thing :func:`check` found.

    Attributes:
        kind: ``stale``, ``missing``, ``orphan``, ``dangling_link``,
            ``unknown_channel``, ``stamp``, ``invalid``, ``oversize``,
            ``incomplete`` (an earlier write did not finish) or
            ``outside_set``.
        file: The file the problem is in, relative to the output directory.
        detail: What is wrong, in one line.
    """

    kind: str
    file: str
    detail: str

    @property
    def failing(self) -> bool:
        """False for a note (a link to a target outside the written set)."""
        return self.kind not in NOTE_KINDS


@dataclass(frozen=True)
class Draw:
    """The parameters a screen is described with; ``write`` and ``check`` must agree on them."""

    rack_classes: frozenset[str] = frozenset()
    symbols: Mapping[str, str] = field(default_factory=dict)
    above: frozenset[str] = frozenset()

    @property
    def digest(self) -> str:
        """16 hex digits naming these settings, whatever the order they were given in."""
        text = json.dumps(
            {
                "rack_classes": sorted(self.rack_classes),
                "symbols": sorted(self.symbols.items()),
                "above": sorted(self.above),
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

    The keys are ``rack_classes`` (place levels that mark a rack), ``symbols``
    (device class to symbol name) and ``above`` (device classes drawn on the
    upper lane of a strip; each needs a symbol). All are optional.

    Raises:
        EmitError: The file is missing or unreadable, is not JSON, holds a key
            this reader does not know, a value of the wrong type, or a symbol
            name the engine does not have.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EmitError(f"{path} cannot be read as {DRAW_FILE}: {error}") from error
    if not isinstance(document, dict):
        raise EmitError(f"{path}: the file must hold one JSON object")
    unknown = sorted(set(document) - {"rack_classes", "symbols", "above"})
    if unknown:
        raise EmitError(f"{path}: unknown key {', '.join(unknown)}")
    racks = _strings(document.get("rack_classes", []), "rack_classes", path)
    above = _strings(document.get("above", []), "above", path)
    symbols = document.get("symbols", {})
    if not isinstance(symbols, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in symbols.items()
    ):
        raise EmitError(f"{path}: symbols must map class names to symbol names")
    for name in sorted(set(symbols.values()) - set(WIDTH_UNITS) - {"generic"}):
        raise EmitError(f"{path}: {name} is not a symbol of the engine")
    for cls in above:
        if cls not in symbols:
            raise EmitError(f"{path}: above names {cls}, which has no entry in symbols")
    return Draw(
        rack_classes=frozenset(racks),
        symbols=dict(symbols),
        above=frozenset(symbols[cls] for cls in above),
    )


# -- ids ------------------------------------------------------------------------------


def _slug(text: str) -> str:
    return _NON_SLUG.sub("-", text.lower()).strip("-") or "x"


def _digest(*parts: str, length: int = DIGEST_LEN) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:length]


def _stem(kind: str, thing_id: str, by: str | None) -> tuple[str, bool]:
    """The readable id, and whether the slug had to be cut."""
    slug = _slug(thing_id)
    cut = len(slug) > SLUG_MAX
    if cut:
        slug = slug[:SLUG_MAX].rstrip("-")
    return "-".join(part for part in (kind, by, slug) if part), cut


def screen_id(kind: str, thing_id: str, by: str | None = None) -> str:
    """The readable screen id of one address.

    Args:
        kind: The view kind.
        thing_id: The thing's id.
        by: A roll-up's grouping.

    Returns:
        ``kind[-by]-slug`` in lowercase ``[a-z0-9-]``; a slug over
        :data:`SLUG_MAX` characters is cut and the digest of the address
        follows. Different ids can still give one result; use
        :func:`screen_ids` for a set.
    """
    stem, cut = _stem(kind, thing_id, by)
    return f"{stem}-{_digest(kind, thing_id, by or '')}" if cut else stem


def _identity(target: Target) -> tuple[str, ...]:
    return (target.view, target.kind, target.id, target.by or "")


def screen_ids(found: list[Target] | tuple[Target, ...]) -> dict[Target, str]:
    """A screen id for every target, no two alike, whatever the order.

    Every member of a group that would collide, and every cut slug, gets the
    8-hex digest of its full address (view, thing kind, id, grouping). Should
    two digests collide as well, the digests of that group are lengthened.
    """
    stems = {}
    for target in found:
        stem, cut = _stem(target.view, target.id, target.by)
        stems[target] = (stem, cut)
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
        id_ = screen_id(target.view, target.id, target.by)
    return f"{target.view}/{id_}{SCREEN_SUFFIX}"


def under(graph: PanelGraph, place_id: str) -> Callable[[Target], bool]:
    """A test for the targets of a place and everything within it.

    Args:
        graph: The panel graph.
        place_id: A place.

    Returns:
        A function true for a target whose thing is the place or a device or
        place within it, at any depth. Groups are never under a place.

    Raises:
        KeyError: No such place.
    """
    within = {place_id, *(thing.id for thing in graph.within(place_id, deep=True))}
    return lambda target: target.kind != "group" and target.id in within


# -- drawing --------------------------------------------------------------------------


def _stamp(graph: PanelGraph, draw: Draw) -> str:
    return f" Facility {graph.content_hash}, panels format {FORMAT_VERSION}, draw {draw.digest}."


def _generate(
    graph: PanelGraph, target: Target, ids: Mapping[Target, str], draw: Draw
) -> tuple[bytes, str]:
    """The bytes of a target's file and its title."""
    own = _path(target, ids)

    def screens(other: Target) -> str:
        # Every file sits one directory below ``out``, so this is relative to it.
        return "../" + _path(other, ids)

    view = describe(
        graph,
        target,
        rack_classes=draw.rack_classes,
        symbols=draw.symbols,
        above=draw.above,
    )
    document = layout.render(view, graph=graph, screens=screens, doc_id=own[: -len(SCREEN_SUFFIX)])
    document["metadata"]["description"] += _stamp(graph, draw)
    text = json.dumps(
        document, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return (text + "\n").encode("utf-8"), view.title


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.unlink(missing_ok=True)  # never write through a link left at this name
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    stale = [out / f".{INDEX}.tmp"]
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
        EmitError: *out* holds files and no ``index.json``, or the index is
            not one this generator wrote.
    """
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
    found = targets(graph, rack_classes=draw.rack_classes)
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

    def put_index(state: str, screens: list[dict[str, Any]], outside: list[dict[str, Any]]) -> None:
        index = {
            **head,
            "state": state,
            "total_bytes": sum(e.get("bytes", 0) for e in screens),
            "screens": sorted(screens, key=lambda e: e["file"]),
            "outside": sorted(outside, key=lambda e: e["file"]),
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
            data, title = _generate(graph, target, ids, draw)
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
    """Every channel a screen binds: a widget's ``pv`` and each name in a rule's ``pvs``."""
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            pv = node.get("pv")
            if isinstance(pv, str):
                yield pv
            pvs = node.get("pvs")
            if isinstance(pvs, list):
                for item in pvs:
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        yield item["name"]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


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
        draw: How a screen is described; the one ``write`` was given.

    Returns:
        The problems, in file order. ``outside_set`` ones are notes
        (:attr:`Problem.failing` is false).
    """
    out = Path(out)
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
    found = targets(graph, rack_classes=draw.rack_classes)
    ids = screen_ids(found)
    wanted = {_path(t, ids): t for t in found}
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
        if relative in outside and relative not in listed:
            continue
        if relative not in listed or not readable(relative):
            problems.append(
                Problem("missing", relative, f"no file for {target.view} of {target.id}")
            )
    for relative in wanted:
        if relative not in valid:
            continue
        problems.extend(
            _check_file(
                graph, out, relative, wanted[relative], ids, valid, outside, max_bytes, draw
            )
        )
    return problems


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
    fresh = _generate(graph, target, ids, draw)[0]
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
