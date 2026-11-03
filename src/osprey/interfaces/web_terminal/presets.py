"""Dock-layout presets: the rules the profile build, the config loader and the op routes share.

A preset (``web.presets``, the profile's ``panel_presets``) is the collection
form of the dock layout (:mod:`osprey.interfaces.web_terminal.dock_layout`) with
panel ids in its ``tabs``. The terminal is never one of them: a preset arranges
service panels only. The ``label: [ids]`` list form of earlier releases is not
read anywhere; :func:`preset_refusal` names it so each reader can say what to
write instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from osprey.interfaces.web_terminal.dock_layout import REASON_TEXT, layout_refusal, normalize_layout

__all__ = ["TERMINAL_KEY", "names_terminal", "preset_refusal", "prune_preset_layout"]

#: The dock key of the operator's Agent pane.
TERMINAL_KEY = "terminal"

#: How a preset is spelled, for the refusals that say what to write.
LAYOUT_SHAPE = "a tile {tabs: [<panel id>, ...]} or a split {split: row|column, items: [...]}"


def names_terminal(node: object) -> bool:
    """Whether any ``tabs`` list in a (possibly malformed) layout names the terminal."""
    if isinstance(node, dict):
        tabs = node.get("tabs")
        if isinstance(tabs, list) and TERMINAL_KEY in tabs:
            return True
        items = node.get("items")
        return isinstance(items, list) and any(names_terminal(item) for item in items)
    return False


def _raw_keys(node: Mapping) -> list[object]:
    """Every tab key of a structurally valid layout, as written (indexes stay ints)."""
    if "split" in node:
        return [key for item in node["items"] for key in _raw_keys(item)]
    return list(node["tabs"])


def preset_refusal(layout: object) -> str | None:
    """Why a raw preset value is not a dock-layout preset; ``None`` when it is.

    A preset is the collection form with panel-id tabs (no member indexes) that
    leaves the terminal alone. The phrase completes "``web.presets['X']`` …".
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
    return None


def prune_preset_layout(layout: Mapping, keep: Callable[[str], bool]) -> dict | None:
    """A valid preset with every tab ``keep`` rejects removed, normalised.

    A tab that is not kept leaves its tile; a tile left with no tab leaves the
    tree, and a split left with one item collapses; a ``showing`` that left
    defaults to the tile's first tab. The survivors keep their weights, so
    normalising shares the space among what is left. ``None`` when no tab
    survives.
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
