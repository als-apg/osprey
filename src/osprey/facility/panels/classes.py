"""Screen classes: which screens are drawn from one standard, and how a member differs.

A reviewer judges one screen per class, its *standard*, and reads, for every
other member, how it differs from that standard. Two screens are of one class
when they have the same key:

``card`` of a device, and its ``channels`` list
    The device type, its class and the card recipe it is drawn from, and the
    set of role words its card shows. A repeated role counts once in the key;
    how many channels play it is a difference. The driver-internal role is
    never shown and channels with no role are listed only in the grid, so
    neither is in the key.
``set``
    The device types of its members.
``calibration``
    The calibration kind and the device types of the set's members.
Every other screen
    Its own class: a place, a rack, a roll-up, a checkout, the home screen, and
    the card of a place or a group draw the facility's own structure.

Sameness is measured on the screen's view (:mod:`~osprey.facility.panels.view`),
not on the drawn file: :func:`structure` keeps every section, item, row and
block of a view and drops its text and numbers, so a longer name or a wider
number does not make two screens differ. The standard is the member whose
structure most members share (a tie goes to the smaller structure hash), so it
does not depend on the order of the files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from osprey.facility.panels.describe import DEFAULT_RECIPE, INTERNAL_ROLE
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.view import Fact, Target, View

__all__ = [
    "KEYED_VIEWS",
    "MAX_WORDS",
    "Sample",
    "class_key",
    "classify",
    "differences",
    "structure",
    "summary",
]

#: The view kinds whose screens are classed by a key; every other screen is its own class.
KEYED_VIEWS = frozenset({"card", "channels", "set", "calibration"})
#: The most differences stated for one member; the rest are counted.
MAX_WORDS = 8
_DIGEST = 8
_NON_SLUG = re.compile(r"[^a-z0-9]+")
_TEXT = "~"
#: What a reviewer calls the parts of a view, by the field that holds them: one, many.
_NOUNS: Mapping[str, tuple[str, str]] = {
    "basis": ("basis mark", "basis marks"),
    "blocks": ("relation block", "relation blocks"),
    "cells": ("cell", "cells"),
    "children": ("entry below", "entries below"),
    "citations": ("citation", "citations"),
    "claims": ("place claim", "place claims"),
    "controls": ("press", "presses"),
    "dashed": ("drawn dashed, as not stated", "drawn dashed, as not stated"),
    "groups": ("group", "groups"),
    "items": ("entry", "entries"),
    "link": ("link to a screen", "links to screens"),
    "members": ("member", "members"),
    "nodes": ("entry", "entries"),
    "parts": ("drawn part", "drawn parts"),
    "provenance": ("source", "sources"),
    "rows": ("row", "rows"),
    "sets": ("set it is in", "sets it is in"),
    "standing": ("standing mark", "standing marks"),
    "tiles": ("key tile", "key tiles"),
    "where": ("line under where", "lines under where"),
}


@dataclass(frozen=True)
class Sample:
    """What one written screen tells about its class.

    Attributes:
        file: The screen's file, relative to the output directory.
        target: Its address.
        cls: Its class id (:func:`class_key`).
        key: The class key in parts, as written to the index.
        structure: The hash of its view's structure (:func:`structure`).
        height: The drawn screen's height in pixels.
        cuts: The labels the layout cut to fit, before the cut.
    """

    file: str
    target: Target
    cls: str
    key: Mapping[str, Any]
    structure: str
    height: int
    cuts: tuple[str, ...] = ()


def _slug(text: str) -> str:
    return _NON_SLUG.sub("-", text.lower()).strip("-")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _device_type(graph: PanelGraph, device_id: str) -> tuple[str, str]:
    """A device's type: its class (``""`` with none) and the card recipe it is drawn from."""
    thing = graph.thing(device_id, "device")
    recipe = graph.recipe(device_id)
    return thing.cls or "", recipe.recipe if recipe is not None else DEFAULT_RECIPE


def _member_types(graph: PanelGraph, group_id: str) -> list[list[str]]:
    """The sorted device types of a group's members; a member that is no device by its kind."""
    found = {
        _device_type(graph, m.id) if m.kind == "device" else (m.kind, "")
        for m in graph.members(group_id)
    }
    return [list(pair) for pair in sorted(found)]


def class_key(graph: PanelGraph, target: Target, screen: str) -> tuple[str, dict[str, Any]]:
    """The class of a screen: its id and its key in parts.

    Args:
        graph: The panel graph.
        target: The screen's address.
        screen: The screen's id, which is the class id of a screen of its own class.

    Returns:
        ``(class id, key)``. The id is a slug of the key's readable part and the
        first 8 hex digits of the key's digest; it depends on the key alone.
    """
    if target.view in ("card", "channels") and target.kind == "device":
        cls, recipe = _device_type(graph, target.id)
        roles = sorted({hit.role.role for hit in graph.roles(target.id)} - {INTERNAL_ROLE})
        key: dict[str, Any] = {
            "view": target.view,
            "device_class": cls,
            "recipe": recipe,
            "roles": roles,
        }
        readable = f"{cls}-{recipe}"
    elif target.view in ("set", "calibration"):
        key = {"view": target.view, "members": _member_types(graph, target.id)}
        readable = "-".join(sorted({cls for cls, _recipe in key["members"]}))
        if target.view == "calibration":
            key["calibration"] = target.system or ""
            readable = f"{target.system or ''}-{readable}"
    else:
        return screen, {"view": target.view, "screen": screen}
    digest = _digest(repr(sorted(key.items())))[:_DIGEST]
    slug = _slug(readable)[:48].strip("-")
    return f"{target.view}-class-{slug}-{digest}" if slug else f"{target.view}-class-{digest}", key


def _shape(value: Any) -> Any:
    """A view part with its text and numbers replaced by one mark, all else kept."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if isinstance(value, Target):
            return _TEXT
        return (
            type(value).__name__,
            tuple((f.name, _shape(getattr(value, f.name))) for f in dataclasses.fields(value)),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_shape(v) for v in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _shape(v)) for k, v in value.items()))
    if isinstance(value, (frozenset, set)):
        return tuple(sorted(repr(_shape(v)) for v in value))
    if value is None or isinstance(value, bool):
        return value
    return _TEXT


def structure(view: View) -> str:
    """16 hex digits naming a view's structure: its sections, items, rows and blocks,
    without the text and numbers they show."""
    return _digest(repr(_shape(view.sections)))[:16]


def standard(samples: Iterable[Sample]) -> Sample:
    """The member whose structure most members share; a tie goes to the smaller
    structure hash, then to the smaller file name."""
    members = list(samples)
    counts = Counter(s.structure for s in members)
    best = min(counts, key=lambda h: (-counts[h], h))
    return min((s for s in members if s.structure == best), key=lambda s: s.file)


def _noun(field_name: str, count: int = 1) -> str:
    """What a reviewer calls *count* of what a field holds."""
    plain = field_name.replace("_", " ")
    one, many = _NOUNS.get(field_name, (plain, plain))
    return one if count == 1 else many


def _facts(items: Iterable[Any]) -> list[str]:
    """The labels of the facts among *items*, and of facts in their pairs."""
    return [
        part.label
        for item in items
        for part in (item if isinstance(item, tuple) else (item,))
        if isinstance(part, Fact)
    ]


def _walk(ours: Any, theirs: Any, where: str, field_name: str, out: list[str]) -> None:
    """Differences of a member's view part *ours* from the standard's *theirs*."""
    if _shape(ours) == _shape(theirs):
        return
    at = f"{where}: " if where else ""
    if isinstance(ours, tuple) and isinstance(theirs, tuple):
        if len(ours) != len(theirs):
            missing = Counter(_facts(theirs)) - Counter(_facts(ours))
            extra = Counter(_facts(ours)) - Counter(_facts(theirs))
            if missing or extra:
                out.extend(f"{at}no {label}" for label in sorted(missing))
                out.extend(f"{at}{label}, which the standard has not" for label in sorted(extra))
            else:
                out.append(
                    f"{at}{len(ours)} {_noun(field_name, len(ours))}, "
                    f"the standard has {len(theirs)}"
                )
            return
        for a, b in zip(ours, theirs, strict=True):
            _walk(a, b, where, field_name, out)
        return
    if ours is None or theirs is None:
        noun = _noun(field_name)
        out.append(f"{at}no {noun}" if ours is None else f"{at}{noun}, which the standard has not")
        return
    if isinstance(ours, bool) and isinstance(theirs, bool):
        out.append(
            f"{at}{_noun(field_name)}: {'yes' if ours else 'no'}, "
            f"the standard: {'yes' if theirs else 'no'}"
        )
        return
    if dataclasses.is_dataclass(ours) and type(ours) is type(theirs) and not isinstance(ours, type):
        for f in dataclasses.fields(ours):
            _walk(getattr(ours, f.name), getattr(theirs, f.name), where, f.name, out)
        return
    out.append(f"{at}{_noun(field_name)} drawn otherwise than on the standard")


def differences(
    graph: PanelGraph, member: Sample, ours: View, std: Sample, theirs: View
) -> list[str]:
    """How a member differs from its class's standard, in words a reviewer reads
    without the code: a section or a fact missing or extra, another number of
    channels of a role, of rows, tiles or blocks, a label cut to fit, another
    height. At most :data:`MAX_WORDS` of each, structure and cut labels, the rest
    counted."""
    found: list[str] = []
    if member.target.view in ("card", "channels") and member.target.kind == "device":
        ours_roles = Counter(h.role.role for h in graph.roles(member.target.id))
        their_roles = Counter(h.role.role for h in graph.roles(std.target.id))
        for role in sorted(set(ours_roles) - {INTERNAL_ROLE}):
            if ours_roles[role] != their_roles[role]:
                rows = "row" if ours_roles[role] == 1 else "rows"
                found.append(
                    f"{ours_roles[role]} {rows} of {role}, the standard has {their_roles[role]}"
                )
    if member.structure != std.structure:
        sections = {s.title: s for s in ours.sections}
        theirs_by = {s.title: s for s in theirs.sections}
        found.extend(f"no {title}" for title in theirs_by if title not in sections)
        for title, section in sections.items():
            if title not in theirs_by:
                found.append(f"a {title} section, which the standard has not")
            else:
                _walk(section.items, theirs_by[title].items, title, "items", found)
    elif member.height != std.height:
        found.append(f"{member.height} px high, the standard {std.height}")
    return [*_capped(list(dict.fromkeys(found))), *_cut_words(member.cuts)]


def _capped(words: list[str]) -> list[str]:
    """At most :data:`MAX_WORDS` of *words*, the rest counted."""
    if len(words) <= MAX_WORDS:
        return words
    return [*words[: MAX_WORDS - 1], f"and {len(words) - MAX_WORDS + 1} more"]


def _cut_words(cuts: Iterable[str]) -> list[str]:
    """The labels cut to fit, at most :data:`MAX_WORDS`, the rest counted."""
    return _capped([f"label cut: {text}" for text in cuts])


def classify(
    graph: PanelGraph,
    samples: Iterable[Sample],
    view_of: Callable[[Target], View],
    *,
    stamp: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """The class table and each member's differences.

    Args:
        graph: The panel graph.
        samples: One per written screen.
        view_of: The view of a target (``describe``), called only for the
            standards and members that differ.
        stamp: What the layout hash adds to the standard's structure: the
            format and the drawing settings the screens are drawn with.

    Returns:
        ``(classes, differs)``: per class id its key, standard file, member
        count, how many members are off the standard's structure, how many
        have a label cut, and its layout hash; per file the differences of a
        member that has any.
    """
    by_class: dict[str, list[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample.cls, []).append(sample)
    classes: dict[str, dict[str, Any]] = {}
    differs: dict[str, list[str]] = {}
    for cls, members in sorted(by_class.items()):
        std = standard(members)
        theirs: View | None = None
        off = 0
        for member in members:
            if member is std or (
                member.structure == std.structure
                and member.height == std.height
                and not member.cuts
                and not _role_counts_differ(graph, member, std)
            ):
                continue
            if theirs is None:
                theirs = view_of(std.target)
            words = differences(graph, member, view_of(member.target), std, theirs)
            off += member.structure != std.structure
            if words:
                differs[member.file] = words
        if std.cuts:
            differs[std.file] = _cut_words(std.cuts)
        classes[cls] = {
            "key": dict(std.key),
            "standard": std.file,
            "members": len(members),
            "off_standard": off,
            "cut": sum(1 for m in members if m.cuts),
            "layout": _digest(f"{std.structure} {stamp}")[:16],
        }
    return classes, differs


def _role_counts_differ(graph: PanelGraph, member: Sample, std: Sample) -> bool:
    if member.target.view not in ("card", "channels") or member.target.kind != "device":
        return False
    return Counter(h.role.role for h in graph.roles(member.target.id)) != Counter(
        h.role.role for h in graph.roles(std.target.id)
    )


def summary(classes: Mapping[str, Mapping[str, Any]]) -> str:
    """One line a gate reads: classes, those of more than one member, members off
    their standard, and members with a label cut."""
    shared = sum(1 for c in classes.values() if c.get("members", 0) > 1)
    off = sum(c.get("off_standard", 0) for c in classes.values())
    cut = sum(c.get("cut", 0) for c in classes.values())
    return (
        f"{len(classes)} classes ({shared} of more than one screen), "
        f"{off} screens off their standard, {cut} with a label cut"
    )
