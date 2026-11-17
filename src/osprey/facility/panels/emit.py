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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.facility.panels import layout
from osprey.facility.panels.describe import describe, targets
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.view import Target

__all__ = [
    "FORMAT_VERSION",
    "INDEX",
    "SCREEN_SUFFIX",
    "SLUG_MAX",
    "Draw",
    "EmitError",
    "Problem",
    "check",
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

_STAMP = re.compile(r" Facility ([0-9a-f]{64}), panels format (\d+)\.$")
_STAMP_IN_TEXT = re.compile(r" Facility [0-9a-f]{64}, panels format \d+\.\"")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


class EmitError(Exception):
    """The output directory cannot be written as asked."""


@dataclass(frozen=True)
class Problem:
    """One thing :func:`check` found.

    Attributes:
        kind: ``stale``, ``missing``, ``orphan``, ``dangling_link``,
            ``unknown_channel``, ``stamp``, ``invalid``, ``oversize`` or
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


def _stamp(graph: PanelGraph) -> str:
    return f" Facility {graph.content_hash}, panels format {FORMAT_VERSION}."


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
    document["metadata"]["description"] += _stamp(graph)
    text = json.dumps(
        document, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return (text + "\n").encode("utf-8"), view.title


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_index(out: Path) -> dict[str, Any] | None:
    try:
        index = json.loads((out / INDEX).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(index, dict) or not isinstance(index.get("screens"), list):
        raise EmitError(f"{out / INDEX} is not an index this generator wrote")
    return index


def _owned(out: Path, relative: Any) -> Path | None:
    """The path of a file the index lists, when it lies strictly inside *out*."""
    if not isinstance(relative, str) or not relative or relative == INDEX:
        return None
    clean = posixpath.normpath(relative)
    if clean.startswith(("/", "..")) or clean != relative or "\\" in relative:
        return None
    return out.joinpath(*clean.split("/"))


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
    found = targets(graph, rack_classes=draw.rack_classes)
    ids = screen_ids(found)
    carried = {
        entry.get("file"): entry
        for entry in (previous or {}).get("screens", ())
        if isinstance(entry, dict)
    }
    paths = {_path(t, ids) for t in found}
    entries: dict[str, dict[str, Any]] = {}
    written: list[Path] = []
    outside: list[dict[str, Any]] = []
    for target in found:
        relative = _path(target, ids)
        if only is None or only(target):
            data, title = _generate(graph, target, ids, draw)
            destination = out / relative
            _atomic(destination, data)
            written.append(destination)
            entries[relative] = _entry(target, ids, title, len(data))
        elif relative in carried and (out / relative).is_file():
            old = carried[relative]
            entries[relative] = _entry(
                target, ids, str(old.get("title", "")), (out / relative).stat().st_size
            )
        else:
            outside.append(
                {k: v for k, v in _entry(target, ids, "", 0).items() if k not in {"title", "bytes"}}
            )
    index = {
        "format": FORMAT_VERSION,
        "facility_hash": graph.content_hash,
        "total_bytes": sum(e["bytes"] for e in entries.values()),
        "screens": [entries[k] for k in sorted(entries)],
        "outside": sorted(outside, key=lambda e: e["file"]),
    }
    _atomic(
        out / INDEX,
        (json.dumps(index, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    for relative in carried:
        owned = _owned(out, relative)
        if owned is not None and relative not in paths and owned.is_file():
            owned.unlink()
            _prune(owned.parent, out)
    return written


def _prune(directory: Path, out: Path) -> None:
    """Remove *directory* when it is empty, up to but not including *out*."""
    while directory != out and directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
        directory = directory.parent


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
    if index.get("facility_hash") != graph.content_hash or index.get("format") != FORMAT_VERSION:
        problems.append(
            Problem(
                "stamp",
                INDEX,
                f"written from facility {index.get('facility_hash')} format {index.get('format')}; "
                f"now facility {graph.content_hash} format {FORMAT_VERSION}",
            )
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
    valid = {r for r in wanted if r in listed and (out / r).is_file()}
    for relative, target in wanted.items():
        if relative in outside and relative not in listed:
            continue
        if relative not in listed or not (out / relative).is_file():
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
    if (
        stamped is None
        or stamped.group(1) != graph.content_hash
        or int(stamped.group(2)) != FORMAT_VERSION
    ):
        got = f"facility {stamped.group(1)} format {stamped.group(2)}" if stamped else "no stamp"
        problems.append(Problem("stamp", relative, f"{got}; now facility {graph.content_hash}"))
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
            problems.append(
                Problem(
                    "unknown_channel",
                    relative,
                    f"binds {name}, which is no channel of the facility",
                )
            )
    here = posixpath.dirname(relative)
    for ref in dict.fromkeys(_links(document)):
        resolved = posixpath.normpath(posixpath.join(here, ref))
        if resolved in valid:
            continue
        kind = "outside_set" if resolved in outside else "dangling_link"
        why = "is outside the written set" if kind == "outside_set" else "is no file of the set"
        problems.append(Problem(kind, relative, f"links to {ref}, which {why}"))
    return problems
