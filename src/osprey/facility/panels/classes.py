"""Screen classes: which screens are drawn from one standard, and how a member differs.

A reviewer judges one screen per class, its *standard*, and reads, for every
other member, how it differs from that standard. Two screens are of one class
when they have the same key:

``card`` of a device, and its ``channels`` list
    The device type, its class and the card recipe it is drawn from, and the
    set of role words its card shows. A repeated role counts once in the key.
    The driver-internal role is never shown and channels with no role are
    listed only in the grid, so neither is in the key. Where the default
    recipe's cards of one such key show different blocks, the key also holds
    ``blocks`` (:func:`blocks`, :func:`split_by_blocks`); a key whose cards all
    show the same blocks stays as it is, and so does its id.
``calibration``
    The calibration kind and its structure: gains and offsets are different
    things to judge even where the layout agrees.
Every other screen, a set among them
    Its view kind, the kind of thing it shows, and its structure: places,
    racks, roll-ups, sets and the cards of places and groups that are built
    alike share a class, and only a screen of a truly single structure stays
    alone.

Sameness is measured on the screen's view (:mod:`~osprey.facility.panels.view`),
not on the drawn file. :func:`measure` reduces a view to its *structure*: the
sections, the kinds of item, block and relation it shows, and nothing else. Four
things are left out, because a reviewer would not call them another card:

* text and numbers;
* how many items a repeated block holds, and their order: a block of twelve tiles
  has the structure of a block of nine, and the count is a *size*;
* the parts that state what is known rather than what is drawn, named in
  :data:`GAP_PARTS` and :data:`GAP_FIELDS` (a source, a unit, a link to a screen
  of the thing): where one is missing the screen has a *gap*;
* the data-state flags named in :data:`FLAGS` and :data:`VALUE_FLAGS` (inferred,
  contested, disagreeing, writable).

A member whose structure is not its standard's is *off standard*: a section, a
kind of item, a block or a relation is present or missing. Its other differences,
in sizes, gaps, flags and the size-driven drawing variants of :data:`VARIANTS`,
are stated apart as what *varies*. The standard is the member whose structure
most members share (a tie goes to the smaller structure hash, then the smaller
file name), so it does not depend on the order of the files.

Each class names a *sample* of up to five members that span its variation (the
standard, the smallest, the largest, the one of the longest labels, the first off
standard) and, for every variation that changes what is drawn, one member on each
side of it, so a reviewer who opens the sample has seen every drawing the class
makes. Each member of the sample is named with what it stands for (``as``): one
or more of :data:`SAMPLE_STOPS`, and the drawing sides it is the first to show.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from osprey.facility.panels.describe import DEFAULT_RECIPE, INTERNAL_ROLE
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.view import Fact, Operate, Target, View

__all__ = [
    "DRAWING_RULE",
    "DRAWN_GAPS",
    "FLAGS",
    "GAP_FIELDS",
    "GAP_PARTS",
    "KEYED_VIEWS",
    "MAX_WORDS",
    "SIZE_PARTS",
    "STRUCTURE_FLAGS",
    "STRUCTURE_PARTS",
    "VALUE_FLAGS",
    "VARIANTS",
    "Measure",
    "Sample",
    "class_key",
    "classify",
    "measure",
    "standard",
    "summary",
    "summary_line",
]

#: The rule a ``drawing`` digest is taken by, stated in the index beside the digests:
#: 1, the written file without the model stamp; 2, also without any widget's
#: ``meta``. Two sets of one rule differ in a class's ``drawing`` only where the
#: drawing differs; across rules, every digest may differ once.
DRAWING_RULE = 2

#: The view kinds whose screens are classed by what they show rather than by structure.
KEYED_VIEWS = frozenset({"card", "channels", "set", "calibration"})
#: The most differences stated for one member in one list; the rest are counted.
MAX_WORDS = 8
#: The most members a class's sample names before the drawing sides are added.
SAMPLE_BASE = 5
#: What the members of a class's sample stand for, before the drawing sides.
SAMPLE_STOPS = ("standard", "smallest", "largest", "longest labels", "off standard")

#: Typed parts that state what is known about a thing, not what its screen draws.
#: Each is left out of the structure; where it is missing, or a plain fact stands
#: where a link could, the screen has a gap named by the field.
GAP_PARTS = frozenset(
    {
        "CalibrationRow.link",
        "CalibrationTally.link",
        "CardFoot.citations",
        "CardFoot.claims",
        "CardFoot.sets",
        "CardFoot.where",
        "ChainBlock.link",
        "Fact.citation",
        "FarEnd.link",
        "GridCell.member",
        "GridRow.label",
        "Node.link",
        "PartFacts.stated",
        "Placement.citation",
        "RackUnit.citations",
        "RackUnit.link",
        "RingMember.link",
        "StripElement.link",
        "StripMember.link",
        "TrayEntry.link",
    }
)
#: Links that are drawn only when a block is too long to show whole ("more", the
#: full list): a size, neither structure nor a gap.
SIZE_PARTS = frozenset(
    {"CalibrationTallies.more", "ChannelGrid.full", "Group.more", "Strip.others"}
)
#: Typed parts that may be absent, or be one of several kinds, and are structure:
#: present, missing or of another kind, they make another card.
STRUCTURE_PARTS = frozenset(
    {"GridCell.calibration", "GridCell.pair", "GridCell.tile", "Section.items"}
)
#: Yes/no fields that state the data's state, not the card's structure.
FLAGS = frozenset(
    {
        "CalibrationRow.below_wanted",
        "CardHead.nowhere",
        "ChainBlock.dashed",
        "ChainBlock.focus",
        "CheckRow.has_rule",
        "CheckRow.has_source",
        "CheckRow.on_device",
        "CheckRow.typed",
        "Control.inferred",
        "Light.disagree",
        "Light.inferred",
        "Light.unsupported",
        "Node.no_rule",
        "PicturePart.as_point",
        "Probe.inferred",
        "RackUnit.conflicting",
        "RingMember.focus",
        "RingMember.stated",
        "StripMember.focus",
        "ValueRow.inferred",
        "ValueRow.writable",
    }
)
#: Text fields that state the data's state: field to the flag it counts, and the
#: value that sets it (``None``: any value).
VALUE_FLAGS: Mapping[str, tuple[str, str | None]] = {
    "basis": ("inferred", "inferred"),
    "standing": ("contested", "contested"),
    "ChainBlock.inferred": ("inferred", None),
    "FarEnd.inferred": ("inferred", None),
    "RingMember.inferred": ("inferred", None),
}
#: Yes/no fields that are structure.
STRUCTURE_FLAGS = frozenset({"CardHead.default"})
#: Text and number fields counted as a gap where they are empty.
GAP_FIELDS = frozenset(
    {
        "angle_deg",
        "caveat",
        "channel",
        "channel_y",
        "h_mm",
        "position",
        "readback",
        "readback_unit",
        "signal",
        "size_u",
        "tolerance",
        "unit",
        "w_mm",
        "x_mm",
        "y_mm",
    }
)
#: Gaps and flags that change what is drawn: a sample shows a member on each side.
DRAWN_GAPS = frozenset({"x_mm", "y_mm", "w_mm", "h_mm", "angle_deg", "as_point"})
#: Item kinds that are one item drawn another way by its size: kind to the item's name.
VARIANTS: Mapping[str, str] = {"ChannelGrid": "channel list", "ChannelCount": "channel list"}
#: Fields whose text is a label a reader reads, measured for the longest label.
_LABEL_FIELDS = frozenset({"label", "title", "short", "short_name"})

_DIGEST = 8
_NON_SLUG = re.compile(r"[^a-z0-9]+")
_TEXT = "~"
_MANY = "*"
_EITHER = "|"
#: What a reviewer calls the parts of a view, by the kind or field that holds them.
_NOUNS: Mapping[str, str] = {
    "*": "list of entries",
    "~": "entry of text",
    "Band": "band of key tiles",
    "BlockRow": "block row",
    "CalibrationCell": "calibration cell",
    "ChainBlock": "relation block",
    "GridCell": "cell",
    "GridRow": "grid row",
    "KeyTile": "key tile",
    "ValueRow": "value row",
    "calibration": "calibration cell",
    "pair": "paired readback",
    "tile": "key tile",
}


@dataclass(frozen=True)
class Measure:
    """What a screen's view is made of, as :func:`measure` reads it.

    Attributes:
        sections: Each section's title and the structure of its items.
        structure: 16 hex digits naming the structure (the sections' structures).
        sizes: Per repeated block, ``"<section>: <field>"``, how many items it holds.
        gaps: Per gap (:data:`GAP_PARTS`, :data:`GAP_FIELDS`), how many times it occurs.
        flags: Per flag (:data:`FLAGS`), how many items carry it.
        variants: Per size-driven item (:data:`VARIANTS`), the kind it is drawn as.
        label: The longest label the view shows, in characters.
    """

    sections: tuple[tuple[str, Any], ...]
    structure: str
    sizes: Mapping[str, int] = field(default_factory=dict)
    gaps: Mapping[str, int] = field(default_factory=dict)
    flags: Mapping[str, int] = field(default_factory=dict)
    variants: Mapping[str, str] = field(default_factory=dict)
    label: int = 0


@dataclass(frozen=True)
class Sample:
    """What one written screen tells about its class.

    Attributes:
        file: The screen's file, relative to the output directory.
        target: Its address.
        cls: Its class id (:func:`class_key`).
        key: The class key in parts, as written to the index.
        measure: Its view's structure and sizes (:func:`measure`).
        cuts: The labels the layout cut to fit, before the cut.
        others: What the screen shows of other things on purpose
            (:func:`~osprey.facility.panels.describe.others`).
        blocks: For a device card drawn by the default recipe, the kinds of
            block it shows (:func:`blocks`); empty otherwise.
        drawing: The digest of its written file without the model stamp
            (:func:`drawing_digest`): it moves when the screen is drawn
            otherwise, not when only the facility file's content hash does.
    """

    file: str
    target: Target
    cls: str
    key: Mapping[str, Any]
    measure: Measure
    cuts: tuple[str, ...] = ()
    others: tuple[Mapping[str, Any], ...] = ()
    blocks: tuple[str, ...] = ()
    drawing: str = ""

    @property
    def structure(self) -> str:
        """The hash of its view's structure."""
        return self.measure.structure


class _Reader:
    """One walk over a view: its structure, sizes, gaps, flags and longest label."""

    def __init__(self) -> None:
        self.sizes: Counter[str] = Counter()
        self.gaps: Counter[str] = Counter()
        self.flags: Counter[str] = Counter()
        self.variants: dict[str, str] = {}
        self.label = 0
        self.section = ""

    def _size(self, name: str, count: int) -> None:
        self.sizes[f"{self.section}: {name}" if self.section else name] += count

    def shape(self, value: Any, name: str = "") -> Any:
        """*value* with text, numbers, counts, order, gaps and flags left out."""
        if isinstance(value, Target):
            return _TEXT
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            kind = type(value).__name__
            fields = tuple(
                (f.name, self._field(kind, f.name, getattr(value, f.name)))
                for f in dataclasses.fields(value)
            )
            if kind in VARIANTS:
                self.variants[VARIANTS[kind]] = kind
                return ("variant", VARIANTS[kind])
            return (kind, fields)
        if isinstance(value, (tuple, list)):
            if not _holds_parts(value):
                if value and all(isinstance(item, tuple) for item in value):
                    self._size(name, len(value))  # a table of text rows: its length is a size
                for item in value:  # the labels in pairs of plain text still count
                    self.shape(item, name)
                return _TEXT
            self._size(name, len(value))
            return _many(self.shape(item, name) for item in value)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str) and name in _LABEL_FIELDS:
            self.label = max(self.label, len(value))
        return _TEXT

    def _field(self, kind: str, name: str, value: Any) -> Any:
        where = f"{kind}.{name}"
        if where in GAP_PARTS:
            self._gap(name, value)
            return _TEXT
        if where in SIZE_PARTS:
            return _TEXT
        if where in FLAGS:
            self.flags[name] += bool(value)
            return _TEXT
        flag = VALUE_FLAGS.get(where) or VALUE_FLAGS.get(name)
        if flag is not None and not isinstance(value, bool):
            word, wanted = flag
            self.flags[word] += value is not None and (wanted is None or value == wanted)
            return _TEXT
        if isinstance(value, bool) or where in STRUCTURE_PARTS or where in STRUCTURE_FLAGS:
            return self.shape(value, name)
        if value is None:
            if name in GAP_FIELDS:
                self.gaps[name] += 1
            return _TEXT
        if name == "not_stated" and isinstance(value, tuple):
            self.gaps["not stated"] += len(value)
        return self.shape(value, name)

    def _gap(self, name: str, value: Any) -> None:
        """Count what a gap part says is missing: the part, or the link a fact stands for."""
        if value is None:
            self.gaps[f"no {name}"] += 1
            return
        if isinstance(value, Fact):
            self.gaps[f"{name} without a link"] += 1
            self.shape(value, name)
            return
        if isinstance(value, tuple):
            self._size(name, len(value))
            unlinked = sum(
                1
                for item in value
                if isinstance(item, Fact)
                or (isinstance(item, tuple) and len(item) > 1 and item[-1] is None)
            )
            if unlinked:
                self.gaps[f"{name} without a link"] += unlinked
            for item in value:
                self.shape(item, name)  # labels and inner gaps still count
            return
        self.shape(value, name)


def _many(shapes: Iterable[Any]) -> tuple[str, tuple[Any, ...]]:
    """The structure of a repeated block: per kind of item, the parts its items show.

    Items of one kind are merged (:func:`_merge`), so the block's structure is
    what kinds of item it holds and what parts they show between them, not how
    many there are, in what order, or which item shows which part.
    """
    by_kind: dict[str, Any] = {}
    for shape in shapes:
        kind = _kind_of(shape) if _is_part(shape) else repr(shape)
        by_kind[kind] = _merge(by_kind[kind], shape) if kind in by_kind else shape
    return (_MANY, tuple(by_kind[k] for k in sorted(by_kind)))


def _merge(a: Any, b: Any) -> Any:
    """Two structures of one kind of item as one: what either shows, the other may not."""
    if a == b:
        return a
    if a in (_TEXT, None) and _is_part(b):
        return b  # an empty block or an absent part beside a present one
    if b in (_TEXT, None) and _is_part(a):
        return a
    if _is_part(a) and _is_part(b) and a[0] == b[0] == _MANY:
        return _many((*a[1], *b[1]))
    if _is_part(a) and _is_part(b) and a[0] == b[0] and isinstance(a[1], tuple):
        return (a[0], tuple((n, _merge(x, y)) for (n, x), (_n, y) in zip(a[1], b[1], strict=True)))
    alternatives = {repr(x): x for x in (*_alternatives(a), *_alternatives(b))}
    return (_EITHER, tuple(alternatives[k] for k in sorted(alternatives)))


def _alternatives(shape: Any) -> tuple[Any, ...]:
    if _is_part(shape) and shape[0] == _EITHER:
        return tuple(shape[1])
    return (shape,)


def _holds_parts(value: Iterable[Any]) -> bool:
    """Whether a sequence holds view parts, not only text and numbers."""
    return any(
        (dataclasses.is_dataclass(item) and not isinstance(item, (type, Target)))
        or (isinstance(item, (tuple, list)) and _holds_parts(item))
        for item in value
    )


def measure(view: View) -> Measure:
    """A view's structure, sizes, gaps, flags and longest label (see the module text)."""
    reader = _Reader()
    sections: list[tuple[str, Any]] = []
    for section in view.sections:
        reader.section = section.title
        sections.append((section.title, reader.shape(section.items, "items")))
        reader.label = max(reader.label, len(section.title))
    reader.label = max(reader.label, len(view.title))
    shapes = sorted({repr(shape) for _title, shape in sections})
    return Measure(
        tuple(sections),
        _digest(repr(shapes))[:16],
        dict(sorted(reader.sizes.items())),
        dict(sorted((k, v) for k, v in reader.gaps.items() if v)),
        dict(sorted((k, v) for k, v in reader.flags.items() if v)),
        dict(sorted(reader.variants.items())),
        reader.label,
    )


def _slug(text: str) -> str:
    return _NON_SLUG.sub("-", text.lower()).strip("-")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _device_type(graph: PanelGraph, device_id: str) -> tuple[str, str]:
    """A device's type: its class (``""`` with none) and the card recipe it is drawn from."""
    thing = graph.thing(device_id, "device")
    recipe = graph.recipe(device_id)
    return thing.cls or "", recipe.recipe if recipe is not None else DEFAULT_RECIPE


def class_key(graph: PanelGraph, target: Target, structure: str) -> tuple[str, dict[str, Any]]:
    """The class of a screen: its id and its key in parts.

    Args:
        graph: The panel graph.
        target: The screen's address.
        structure: The structure of its view (:func:`measure`), the key of a
            screen that is not classed by what it shows.

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
    elif target.view == "calibration":
        # Gains and offsets are different things to judge even where the layout
        # agrees: the calibration kind is the kind of signal it states.
        key = {"view": target.view, "calibration": target.system or "", "structure": structure}
        readable = target.system or ""
    else:
        key = {"view": target.view, "kind": target.kind, "structure": structure}
        readable = target.kind
    return _class_id(target.view, readable, key), key


def _class_id(view: str, readable: str, key: Mapping[str, Any]) -> str:
    """A slug of the key's readable part and the first 8 hex digits of its digest."""
    digest = _digest(repr(sorted(key.items())))[:_DIGEST]
    slug = _slug(readable)[:48].strip("-")
    return f"{view}-class-{slug}-{digest}" if slug else f"{view}-class-{digest}"


def blocks(view: View, key: Mapping[str, Any]) -> tuple[str, ...]:
    """The kinds of block a device card drawn by the default recipe shows, by
    presence: the item kinds of its sections, the size-driven drawings of one item
    (:data:`VARIANTS`) as that item, and the operate block named by whether it
    holds controls. Empty for every other screen."""
    if key.get("view") != "card" or key.get("recipe") != DEFAULT_RECIPE:
        return ()
    found: set[str] = set()
    for section in view.sections:
        for item in section.items:
            kind = type(item).__name__
            if isinstance(item, Operate):
                kind += " with controls" if item.groups else " without controls"
            found.add(VARIANTS.get(kind, kind))
    return tuple(sorted(found))


def drawing_digest(text: str) -> str:
    """16 hex digits naming a written screen's drawing, by :data:`DRAWING_RULE`:
    its file without the model stamp (the stamp in its description and the one
    its header shows) and without any widget's ``meta``."""
    return _digest(text)[:16]


def split_by_blocks(samples: list[Sample]) -> list[Sample]:
    """Default-recipe device cards of one class whose blocks differ go to one class
    per kind of card: their key gains ``blocks``. A class whose cards all show the
    same blocks keeps its key and its id."""
    by_class: dict[str, list[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample.cls, []).append(sample)
    out: list[Sample] = []
    for members in by_class.values():
        if len({m.blocks for m in members}) < 2:
            out.extend(members)
            continue
        for m in members:
            key = {**m.key, "blocks": list(m.blocks)}
            readable = f"{key['device_class']}-{key['recipe']}"
            out.append(replace(m, cls=_class_id(m.target.view, readable, key), key=key))
    return out


def standard(samples: Iterable[Sample]) -> Sample:
    """The member whose structure most members share; a tie goes to the smaller
    structure hash, then to the smaller file name."""
    members = list(samples)
    counts = Counter(s.structure for s in members)
    best = min(counts, key=lambda h: (-counts[h], h))
    return min((s for s in members if s.structure == best), key=lambda s: s.file)


def _noun(name: str) -> str:
    return _NOUNS.get(name, name.replace("_", " "))


def _is_part(shape: Any) -> bool:
    return (
        isinstance(shape, tuple)
        and len(shape) == 2
        and isinstance(shape[0], str)
        and isinstance(shape[1], (tuple, str))
    )


def _kind_of(shape: Any) -> str:
    if _is_part(shape):
        return str(shape[1] if shape[0] == "variant" else shape[0])
    return "entry"


def _off(ours: Any, theirs: Any, at: str, name: str, out: list[str]) -> None:
    """How a member's structure *ours* departs from the standard's *theirs*, in words."""
    if ours == theirs:
        return
    empty = (None, _TEXT)
    if ours in empty or theirs in empty:  # a part absent, or a block without items
        noun = _noun(name)
        out.append(f"{at}no {noun}" if ours in empty else f"{at}{noun}, which the standard has not")
        return
    if _is_part(ours) and _is_part(theirs) and ours[0] == theirs[0] == _MANY:
        mine = {_kind_of(s) for s in ours[1]}
        std = {_kind_of(s) for s in theirs[1]}
        out.extend(f"{at}no {_noun(k)}" for k in sorted(std - mine))
        out.extend(f"{at}a {_noun(k)}, which the standard has not" for k in sorted(mine - std))
        for kind in sorted(mine & std):
            a = [s for s in ours[1] if _kind_of(s) == kind]
            b = [s for s in theirs[1] if _kind_of(s) == kind]
            if a == b:
                continue
            if len(a) == len(b) == 1:
                _off(a[0], b[0], at, kind, out)
            else:
                out.append(f"{at}a {_noun(kind)} drawn otherwise than on the standard")
        return
    if _is_part(ours) and _is_part(theirs) and ours[0] == theirs[0] != "variant":
        for (field_name, a), (_same, b) in zip(ours[1], theirs[1], strict=True):
            _off(a, b, at, field_name, out)
        return
    out.append(f"{at}{_noun(name)} drawn otherwise than on the standard")


def _differs(member: Measure, std: Measure) -> list[str]:
    """The words of a member off its standard: sections, kinds, blocks or relations."""
    found: list[str] = []
    mine: dict[str, list[Any]] = {}
    theirs: dict[str, list[Any]] = {}
    for title, shape in member.sections:
        mine.setdefault(title, []).append(shape)
    for title, shape in std.sections:
        theirs.setdefault(title, []).append(shape)
    found.extend(f"no {title} section" for title in theirs if title not in mine)
    found.extend(
        f"a {title} section, which the standard has not" for title in mine if title not in theirs
    )
    for title, shapes in mine.items():
        if title in theirs and shapes != theirs[title]:
            for a, b in zip(shapes, theirs[title], strict=False):
                _off(a, b, f"{title}: ", "items", found)
    if not found:  # the same parts under other section titles
        found.append("its sections are titled otherwise than on the standard")
    return _capped(list(dict.fromkeys(found)))


def _varies(member: Measure, std: Measure) -> list[str]:
    """The words of what varies from the standard: sizes, gaps, flags, drawing variants."""
    found: list[str] = []
    for name in sorted(member.sizes.keys() | std.sizes.keys()):
        a, b = member.sizes.get(name, 0), std.sizes.get(name, 0)
        if a != b:
            found.append(f"{name}: {a}, the standard {b}")
    for title, mine, theirs in (("gap", member.gaps, std.gaps), ("flag", member.flags, std.flags)):
        for name in sorted(mine.keys() | theirs.keys()):
            a, b = mine.get(name, 0), theirs.get(name, 0)
            if a != b:
                found.append(f"{title} {name}: {a}, the standard {b}")
    for name in sorted(member.variants.keys() | std.variants.keys()):
        mine_as, std_as = member.variants.get(name), std.variants.get(name)
        if mine_as != std_as:
            found.append(
                f"{name} drawn as {mine_as or 'nothing'}, the standard as {std_as or 'nothing'}"
            )
    return _capped(found)


def _capped(words: list[str]) -> list[str]:
    """At most :data:`MAX_WORDS` of *words*, the rest counted."""
    if len(words) <= MAX_WORDS:
        return words
    return [*words[: MAX_WORDS - 1], f"and {len(words) - MAX_WORDS + 1} more"]


def _drawn_sides(sample: Sample) -> set[tuple[str, str]]:
    """Per variation that changes the drawing, the side of it a member is on."""
    m = sample.measure
    sides = {(f"variant {k}", v) for k, v in m.variants.items()}
    for name in sorted(DRAWN_GAPS):
        sides.add((f"gap {name}", str(m.gaps.get(name, 0) > 0)))
        sides.add((f"flag {name}", str(m.flags.get(name, 0) > 0)))
    return sides


def _side_name(side: tuple[str, str]) -> str:
    """What a reviewer calls one side of a drawing variation (:func:`_drawn_sides`)."""
    variation, value = side
    kind, name = variation.split(" ", 1)
    if kind == "variant":
        return f"{name} drawn as {value}"
    present = value == "True"
    if kind == "gap":
        return f"{name} missing" if present else f"{name} stated"
    return name if present else f"not {name}"


def _sample(members: list[Sample], std: Sample, off: list[Sample]) -> list[dict[str, Any]]:
    """Up to :data:`SAMPLE_BASE` members that span the class, then one per drawing
    side; each with the stops it stands for (:data:`SAMPLE_STOPS`, then the drawing
    sides it is the first to show)."""
    ordered = sorted(members, key=lambda s: s.file)

    def total(s: Sample) -> int:
        return sum(s.measure.sizes.values())

    picks = [
        ("standard", std),
        ("smallest", min(ordered, key=total)),
        ("largest", max(ordered, key=total)),
        ("longest labels", max(ordered, key=lambda s: s.measure.label)),
        *(("off standard", s) for s in off[:1]),
    ]
    stops: dict[str, list[str]] = {}
    for stop, member in picks:
        stops.setdefault(member.file, []).append(stop)
    by_file = {s.file: s for s in ordered}
    seen = set().union(*(_drawn_sides(by_file[f]) for f in stops))
    for member in ordered:
        sides = _drawn_sides(member) - seen
        if sides:
            stops.setdefault(member.file, []).extend(_side_name(s) for s in sorted(sides))
            seen |= sides
    return [{"file": file, "as": named} for file, named in stops.items()]


def classify(
    samples: Iterable[Sample], *, stamp: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """The class table and what each screen's index entry states about its class.

    Args:
        samples: One per written screen.
        stamp: What the layout hash adds to the standard's structure: the
            format and the drawing settings the screens are drawn with.

    Returns:
        ``(classes, entries)``. Per class id: its key, standard file, member
        count, how many members are off the standard, how many have a label
        cut, its layout hash and its sample. Per file, the fields its index
        entry carries, each only when it has something to say: ``differs``
        (off standard, in words), ``varies`` (sizes, gaps, flags and drawing
        variants other than the standard's, in words), ``cuts`` (labels cut to
        fit), ``others`` (what it shows of other things on purpose), ``sizes``,
        ``gaps`` and ``flags``; and always ``label``, the longest label in
        characters.
    """
    by_class: dict[str, list[Sample]] = {}
    for sample in split_by_blocks(list(samples)):
        by_class.setdefault(sample.cls, []).append(sample)
    classes: dict[str, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    for cls, members in sorted(by_class.items()):
        std = standard(members)
        off: list[Sample] = []
        for member in sorted(members, key=lambda s: s.file):
            m = member.measure
            entry: dict[str, Any] = {
                "sizes": dict(m.sizes),
                "gaps": dict(m.gaps),
                "flags": dict(m.flags),
            }
            if member is not std:
                if member.structure != std.structure:
                    off.append(member)
                    entry["differs"] = _differs(m, std.measure)
                entry["varies"] = _varies(m, std.measure)
            entry["cuts"] = _capped(list(member.cuts))
            entry["others"] = [dict(o) for o in member.others]
            entries[member.file] = {"label": m.label, **{k: v for k, v in entry.items() if v}}
        classes[cls] = {
            "key": dict(std.key),
            "standard": std.file,
            "members": len(members),
            "off_standard": len(off),
            "cut": sum(1 for m in members if m.cuts),
            "layout": _digest(f"{std.structure} {stamp}")[:16],
            "drawing": std.drawing,
            "sample": _sample(members, std, off),
        }
    return classes, entries


def summary(classes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The counts a gate and a dashboard read: classes, those of more than one
    member, members off their standard, members with a label cut, and the
    classes of one member per view kind."""
    single = Counter(
        str(c.get("key", {}).get("view", "")) for c in classes.values() if c.get("members") == 1
    )
    return {
        "classes": len(classes),
        "shared": sum(1 for c in classes.values() if c.get("members", 0) > 1),
        "off_standard": sum(c.get("off_standard", 0) for c in classes.values()),
        "cut": sum(c.get("cut", 0) for c in classes.values()),
        "single": dict(sorted(single.items())),
    }


def summary_line(counts: Mapping[str, Any]) -> str:
    """:func:`summary` as one line."""
    single = ", ".join(f"{view} {n}" for view, n in (counts.get("single") or {}).items())
    return (
        f"{counts['classes']} classes ({counts['shared']} of more than one screen), "
        f"{counts['off_standard']} screens off their standard, {counts['cut']} with a label cut"
        + (f"; single: {single}" if single else "")
    )
