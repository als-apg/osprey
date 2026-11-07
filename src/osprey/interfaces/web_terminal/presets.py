"""Dock-layout presets: the rules the profile build, the config loader and the op routes share.

A preset (``web.presets``, the profile's ``panel_presets``) is the collection
form of the dock layout (:mod:`osprey.interfaces.web_terminal.dock_layout`)
whose tiles each name one panel id in their ``tabs``. The terminal is never one
of them: a preset arranges service panels only. The ``label: [ids]`` list form
of earlier releases is not read anywhere; :func:`preset_refusal` names it so
each reader can say what to write instead.

One panel per tile is OSPREY's policy, not the layout grammar's. The dock
engine, and :mod:`~osprey.interfaces.web_terminal.dock_layout` which replays its
pinned cases, lets a tile hold several panels; :func:`crowded_tile` finds such a
tile, and every server path that takes a layout (a preset, an agent's
arrangement, a dock's report or op result) refuses it by the one name
:data:`ONE_PANEL_PER_TILE`, the name the browser refuses it by too.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from osprey.interfaces.web_terminal.dock_layout import REASON_TEXT, layout_refusal, normalize_layout

__all__ = [
    "ONE_PANEL_PER_TILE",
    "ONE_PANEL_PER_TILE_TEXT",
    "TERMINAL_KEY",
    "crowded_tile",
    "names_terminal",
    "preset_refusal",
    "prune_preset_layout",
]

#: The dock key of the operator's Agent pane.
TERMINAL_KEY = "terminal"

#: The refusal of a layout that puts several panels in one tile.
ONE_PANEL_PER_TILE = "one_panel_per_tile"

#: What that refusal says to the operator or the agent.
ONE_PANEL_PER_TILE_TEXT = (
    "a tile holds one panel; to show several, put them side by side in tiles of their own"
)

#: How a preset is spelled, for the refusals that say what to write.
LAYOUT_SHAPE = "a tile {tabs: [<panel id>]} or a split {split: row|column, items: [...]}"


def names_terminal(node: object) -> bool:
    """Whether any ``tabs`` list in a (possibly malformed) layout names the terminal."""
    if isinstance(node, dict):
        tabs = node.get("tabs")
        if isinstance(tabs, list) and TERMINAL_KEY in tabs:
            return True
        items = node.get("items")
        return isinstance(items, list) and any(names_terminal(item) for item in items)
    return False


def crowded_tile(layout: Mapping) -> str | None:
    """Where the first tile holding more than one panel is, in tree order; ``None`` when none does.

    ``layout`` is structurally valid, in either form: it passed
    :func:`~osprey.interfaces.web_terminal.dock_layout.layout_refusal`. The
    answer is spelled like that function's paths and ends at the tile's
    ``tabs``, e.g. ``layout.items[1].tabs``.
    """
    pending: list[tuple[Mapping, str]] = [(layout, "layout")]
    while pending:
        node, where = pending.pop()
        if "split" in node:
            items = [(item, f"{where}.items[{n}]") for n, item in enumerate(node["items"])]
            pending.extend(reversed(items))
        elif len(node["tabs"]) > 1:
            return f"{where}.tabs"
    return None


def _raw_keys(node: Mapping) -> list[object]:
    """Every key of a structurally valid layout, as written (indexes stay ints)."""
    if "split" in node:
        return [key for item in node["items"] for key in _raw_keys(item)]
    return list(node["tabs"])


def preset_refusal(layout: object) -> str | None:
    """Why a raw preset value is not a dock-layout preset; ``None`` when it is.

    A preset is the collection form with panel-id keys (no member indexes) that
    leaves the terminal alone and puts one panel in each tile. The phrase
    completes "``web.presets['X']`` …".
    """
    if isinstance(layout, list):
        return "is the retired list-of-panel-ids form, not a layout"
    if not isinstance(layout, dict):
        return f"must be a layout: {LAYOUT_SHAPE} (got {type(layout).__name__})"
    refusal = layout_refusal(layout, form="collection")
    if refusal is not None:
        reason, where = refusal
        return f"is not a valid layout: {where}: {reason}: {REASON_TEXT[reason]}"
    if not all(isinstance(key, str) for key in _raw_keys(layout)):
        return "tabs name panel ids, not member indexes"
    if names_terminal(layout):
        return "names the terminal: a preset arranges service panels and leaves the terminal where it is"
    crowded = crowded_tile(layout)
    if crowded is not None:
        return f"puts several panels in one tile ({crowded}): {ONE_PANEL_PER_TILE_TEXT}"
    return None


def prune_preset_layout(layout: Mapping, keep: Callable[[str], bool]) -> dict | None:
    """A valid preset with every panel ``keep`` rejects removed, normalised.

    A panel that is not kept leaves the tree with its tile, and a split left
    with one item collapses. The survivors keep their weights, so normalising
    shares the space among what is left. ``None`` when no panel survives.
    """

    def prune(node: Mapping) -> dict | None:
        if "split" in node:
            items = [kept for kept in (prune(item) for item in node["items"]) if kept]
            if not items:
                return None
            out: dict = {"split": node["split"], "items": items}
        else:
            tabs = [key for key in node["tabs"] if keep(key)]
            if not tabs:
                return None
            out = {"tabs": tabs}
            if node.get("showing") in tabs:
                out["showing"] = node["showing"]
        if "size" in node:
            out["size"] = node["size"]
        return out

    pruned = prune(layout)
    return None if pruned is None else normalize_layout(pruned, form="collection")
