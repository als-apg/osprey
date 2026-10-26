"""C-LAYOUT: the dock layout tree, validated and normalised.

A layout is a tile ``{tile, tabs, showing, size?, from?}`` or a split
``{split: "row"|"column", items, size?}``. The JavaScript engine
(``ts/layout.ts``: ``validateLayout`` / ``normalize``) holds the same rules; both
replay the ``validate`` and ``normalize`` cases under ``cases/``, and where a
refusal is in doubt the case's ``(reason, where)`` decides. This is the one Python
implementation OSPREY and CANOPY share, so both refuse a layout for the same
named reason at the same path.

Two forms exist. ``dock`` is the live dock: every tile has an id, keys are panel
ids (``"terminal"`` included), ``showing`` is required and sizes are fractions that
sum to 1 among siblings. ``collection`` is a saved arrangement: no tile ids, keys are
member indexes (or panel ids), ``showing`` optional and sizes are weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

__all__ = [
    "LAYOUT_REASONS",
    "REASON_TEXT",
    "SIZE_TOLERANCE",
    "SPLITS",
    "LayoutForm",
    "LayoutRefused",
    "layout_keys",
    "layout_refusal",
    "normalize_layout",
    "old_shape_refusal",
    "validate_layout",
]

SPLITS = ("row", "column")

# The named reasons a layout is refused: the C-CASES ``validate`` vocabulary.
LAYOUT_REASONS: tuple[str, ...] = (
    "not_layout",
    "unknown_field",
    "bad_split",
    "bad_tile_id",
    "duplicate_tile",
    "bad_key",
    "empty_tabs",
    "duplicate_key",
    "showing_not_in_tabs",
    "split_too_few_items",
    "bad_size",
    "unknown_from",
    "old_slot_shape",
    "old_key_shape",
    "member_missing",
)
REASON_TEXT: dict[str, str] = {
    "not_layout": "a layout node is a tile (tabs) or a split (split, items)",
    "unknown_field": "unknown field",
    "bad_split": "split must be 'row' or 'column'",
    "bad_tile_id": "tile must be a non-empty string",
    "duplicate_tile": "a tile id is used twice",
    "bad_key": "a tab must be a non-empty panel id (or a member's 0-based index)",
    "empty_tabs": "tabs must list at least one tab",
    "duplicate_key": "a tab is placed twice",
    "showing_not_in_tabs": "showing must be one of the tile's tabs",
    "split_too_few_items": "a split holds at least two items",
    "bad_size": "size must be a positive number",
    "unknown_from": "from must name another tile of the layout",
    "old_slot_shape": "the slot shape is retired; a tile lists its members in tabs",
    "old_key_shape": "the key-leaf shape is retired; a tile lists its keys in tabs",
    "member_missing": "every member is placed in a tile",
}
LayoutForm = Literal["dock", "collection"]
# Dock form: sibling fractions sum to 1 within this.
SIZE_TOLERANCE = 0.001
_SPLIT_FIELDS = frozenset({"split", "items", "size"})
_DOCK_TILE_FIELDS = frozenset({"tile", "tabs", "showing", "size", "from"})
_COLLECTION_TILE_FIELDS = frozenset({"tabs", "showing", "size"})


class LayoutRefused(ValueError):
    """The layout breaks a C-LAYOUT rule: ``reason`` (one of :data:`LAYOUT_REASONS`) at ``where``."""

    def __init__(self, reason: str, where: str) -> None:
        super().__init__(f"{where}: {reason}: {REASON_TEXT[reason]}")
        self.reason = reason
        self.where = where


class _Refused(Exception):
    """The walk's own control flow: the first broken rule, before it is reported."""

    def __init__(self, reason: str, where: str) -> None:
        super().__init__(f"{where}: {reason}")
        self.reason = reason
        self.where = where


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    """A JSON number that is finite (an int of any length is)."""
    return _is_number(value) and (isinstance(value, int) or math.isfinite(value))


def _is_index(value: Any) -> bool:
    """A JSON number that is a non-negative integer (``2.0`` counts, as in JavaScript)."""
    if not _is_number(value) or value < 0:
        return False
    return isinstance(value, int) or (math.isfinite(value) and value.is_integer())


def _same(a: Any, b: Any) -> bool:
    """JavaScript's ``SameValueZero`` for JSON keys: numbers by value, strings by text, never across."""
    if _is_number(a) and _is_number(b):
        return bool(a == b)
    return isinstance(a, str) and isinstance(b, str) and a == b


def _old_shape(value: Any, where: str) -> None:
    """The old leaf shapes are named before anything else, wherever they sit."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            _old_shape(item, f"{where}[{index}]")
        return
    if not isinstance(value, Mapping):
        return
    if "slot" in value:
        raise _Refused("old_slot_shape", where)
    if "key" in value and "tabs" not in value:
        raise _Refused("old_key_shape", where)
    if isinstance(value.get("items"), list):
        _old_shape(value["items"], f"{where}.items")


def old_shape_refusal(value: Any, where: str = "layout") -> tuple[str, str] | None:
    """``(reason, where)`` of the first old leaf shape in ``value``; ``None`` when there is none.

    The reason is ``old_slot_shape`` or ``old_key_shape``; a tree too deep to walk is
    ``("not_layout", "layout")``, as :func:`layout_refusal` reports it.
    """
    try:
        _old_shape(value, where)
    except _Refused as refused:
        return refused.reason, refused.where
    except RecursionError:
        return "not_layout", "layout"
    return None


class _Walk:
    def __init__(self, form: str, members: int | None) -> None:
        self.form = form
        self.members = members
        self.keys: list[Any] = []
        self.tiles: set[str] = set()
        self.froms: list[tuple[str, str]] = []

    def placed(self, key: Any) -> bool:
        return any(_same(key, seen) for seen in self.keys)


def _check_key(key: Any, walk: _Walk, where: str) -> None:
    if walk.form == "dock":
        if not isinstance(key, str) or key == "":
            raise _Refused("bad_key", where)
        return
    if walk.members is not None:
        if not _is_index(key) or key >= walk.members:
            raise _Refused("bad_key", where)
        return
    if not (_is_index(key) or (isinstance(key, str) and key != "")):
        raise _Refused("bad_key", where)


def _check_size(node: Mapping[str, Any], form: str, where: str) -> None:
    if "size" not in node:
        return
    size = node["size"]
    ok = _finite(size) and size > 0 and (form == "collection" or size <= 1 + SIZE_TOLERANCE)
    if not ok:
        raise _Refused("bad_size", where)


def _check_fractions(items: list[Mapping[str, Any]], where: str) -> None:
    """Dock form: given fractions sum to 1, or leave a remainder for the missing ones."""
    given = 0.0
    missing = 0
    for item in items:
        if _is_number(item.get("size")):
            given += item["size"]
        else:
            missing += 1
    ok = abs(given - 1) <= SIZE_TOLERANCE if missing == 0 else given < 1 - SIZE_TOLERANCE
    if not ok:
        raise _Refused("bad_size", where)


def _check_fields(node: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(str(field) for field in node if field not in allowed)
    if unknown:
        raise _Refused("unknown_field", f"{where}.{unknown[0]}")


def _check_tile(node: Mapping[str, Any], walk: _Walk, where: str) -> None:
    _check_fields(
        node, _DOCK_TILE_FIELDS if walk.form == "dock" else _COLLECTION_TILE_FIELDS, where
    )
    tile_id = ""
    if walk.form == "dock":
        raw_id = node.get("tile")
        if not isinstance(raw_id, str) or raw_id == "":
            raise _Refused("bad_tile_id", f"{where}.tile")
        tile_id = raw_id
        if tile_id in walk.tiles:
            raise _Refused("duplicate_tile", f"{where}.tile")
        walk.tiles.add(tile_id)
    tabs = node.get("tabs")
    if not isinstance(tabs, list) or not tabs:
        raise _Refused("empty_tabs", f"{where}.tabs")
    for index, key in enumerate(tabs):
        at = f"{where}.tabs[{index}]"
        _check_key(key, walk, at)
        if walk.placed(key):
            raise _Refused("duplicate_key", at)
        walk.keys.append(key)
    if ("showing" in node or walk.form == "dock") and not any(
        _same(node.get("showing"), key) for key in tabs
    ):
        raise _Refused("showing_not_in_tabs", f"{where}.showing")
    _check_size(node, walk.form, f"{where}.size")
    if "from" in node:
        origin = node["from"]
        if not isinstance(origin, str) or origin == tile_id:
            raise _Refused("unknown_from", f"{where}.from")
        walk.froms.append((origin, f"{where}.from"))


def _check_node(value: Any, walk: _Walk, where: str) -> None:
    if not isinstance(value, Mapping):
        raise _Refused("not_layout", where)
    if "split" not in value and "items" not in value:
        if "tabs" not in value and "tile" not in value:
            raise _Refused("not_layout", where)
        _check_tile(value, walk, where)
        return
    _check_fields(value, _SPLIT_FIELDS, where)
    if value.get("split") not in SPLITS:
        raise _Refused("bad_split", f"{where}.split")
    items = value.get("items")
    if not isinstance(items, list) or len(items) < 2:
        raise _Refused("split_too_few_items", f"{where}.items")
    _check_size(value, walk.form, f"{where}.size")
    for index, item in enumerate(items):
        _check_node(item, walk, f"{where}.items[{index}]")
    if walk.form == "dock":
        _check_fractions(items, f"{where}.items")


def layout_refusal(
    layout: Any, form: LayoutForm = "dock", members: int | None = None
) -> tuple[str, str] | None:
    """``(reason, where)`` of the first C-LAYOUT rule ``layout`` breaks, in tree order; ``None`` when valid.

    ``form`` is ``dock`` (tile ids, string keys, fractions) or ``collection`` (no ids,
    weights, ``showing`` optional). With ``members`` (collection form), every key must be a
    member index below it and every member placed once. A tree too deep to walk is
    ``("not_layout", "layout")``.
    """
    walk = _Walk(form, members if form == "collection" else None)
    try:
        _old_shape(layout, "layout")
        _check_node(layout, walk, "layout")
        for origin, where in walk.froms:
            if origin not in walk.tiles:
                raise _Refused("unknown_from", where)
        if walk.members is not None:
            for index in range(walk.members):
                if not walk.placed(index):
                    raise _Refused("member_missing", "layout")
    except _Refused as refused:
        return refused.reason, refused.where
    except RecursionError:
        return "not_layout", "layout"
    return None


def validate_layout(layout: Any, form: LayoutForm = "dock", members: int | None = None) -> None:
    """Raise :class:`LayoutRefused` when ``layout`` breaks a C-LAYOUT rule."""
    refusal = layout_refusal(layout, form, members)
    if refusal is not None:
        raise LayoutRefused(*refusal)


def _fractions(items: list[Mapping[str, Any]], form: str) -> list[float]:
    """Dock form: missing sizes share the remainder; collection form: a missing weight is 1. Then sum to 1."""
    given = [
        item["size"] if _finite(item.get("size")) and item["size"] > 0 else None for item in items
    ]
    missing = sum(1 for size in given if size is None)
    fill = 1.0
    if form == "dock" and missing:
        remainder = 1 - sum(size for size in given if size is not None)
        fill = remainder / missing if remainder > 1e-9 else 1 / len(items)
    raw = [fill if size is None else size for size in given]
    total = sum(raw)
    return [size / total for size in raw]


def _with_size(node: Mapping[str, Any], size: Any) -> dict[str, Any]:
    out = dict(node)
    if size is None:
        out.pop("size", None)
    else:
        out["size"] = size
    return out


def _normalize_node(node: Mapping[str, Any], form: str, tiles: set[str]) -> dict[str, Any]:
    if "split" not in node:
        out = {**node, "tabs": list(node["tabs"])}
        if "showing" not in out:
            out["showing"] = out["tabs"][0]
        if "from" in out and (
            not isinstance(out["from"], str)
            or out["from"] not in tiles
            or out["from"] == out.get("tile")
        ):
            del out["from"]
        return out
    fractions = _fractions(node["items"], form)
    items: list[dict[str, Any]] = []
    for share, item in zip(fractions, node["items"], strict=True):
        child = _normalize_node(item, form, tiles)
        if "split" in child and child["split"] == node["split"]:
            items.extend(
                _with_size(grandchild, share * (grandchild.get("size") or 0))
                for grandchild in child["items"]
            )
        else:
            items.append(_with_size(child, share))
    if len(items) == 1:
        return _with_size(items[0], node.get("size"))
    return _with_size({"split": node["split"], "items": items}, node.get("size"))


def _tile_ids(node: Mapping[str, Any], out: set[str]) -> None:
    if "split" in node:
        for item in node["items"]:
            _tile_ids(item, out)
    elif isinstance(node.get("tile"), str):
        out.add(node["tile"])


def normalize_layout(layout: Mapping[str, Any], form: LayoutForm = "dock") -> dict[str, Any]:
    """A valid ``layout`` in normalised form, as a fresh tree (the engine's ``normalize``).

    Same-direction splits are flattened, one-item splits collapsed, every size filled in
    as a fraction with siblings summing to 1 (collection weights normalised, a missing
    weight counting 1), every ``showing`` filled in (default ``tabs[0]``) and a ``from``
    whose tile is gone cleared. The root carries no size.
    """
    tiles: set[str] = set()
    _tile_ids(layout, tiles)
    return _with_size(_normalize_node(layout, form, tiles), None)


def layout_keys(layout: Mapping[str, Any] | None) -> list[str]:
    """Every tab key of a valid dock ``layout`` in reading order (tiles depth-first, tabs in order)."""
    out: list[str] = []

    def walk(node: Mapping[str, Any]) -> None:
        if "split" in node:
            for item in node["items"]:
                walk(item)
        else:
            out.extend(str(key) for key in node["tabs"])

    if layout is not None:
        walk(layout)
    return out
