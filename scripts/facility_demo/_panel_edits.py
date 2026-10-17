"""The structural edits ``panel_rename.py apply`` makes to GTLView/BTSView.

The rename map rewrites binding values; these edits change the panels' shape.
Each one is keyed by the panel's path under ``screens/`` and, where the source
panel offers no better handle, by widget id. Every edit is idempotent: it
checks for its own result first, so a second ``apply`` finds nothing to do.

* **Paddle split.** A TV paddle toggle writes the paddle setpoint and shows
  the paddle readback. The ``_tv`` sub-screen declares a ``paddle_rb`` macro
  beside ``paddle``; its paddle toggle reads ``${paddle_rb}``; and every
  ``_tv`` instance passes ``paddle_rb``, which the rename map resolves to the
  paddle's readback. A toggle bound directly to a paddle needs no edit: the
  rename map already resolves its ``pv`` to the setpoint and its
  ``readbackPV`` to the readback.
* **Lamp.** The ``_tv`` lamp is a readback, so its toggle becomes a ``pv-led``.
* **Dead macros.** ``light_on``/``paddle_in`` (named only in a script comment)
  leave every ``_tv`` instance, and the knob-assignment macros ``P``, ``R``,
  ``GAIN``, ``MIN``, ``MAX`` leave every declaration and instance.
* **Cuts.** The two hidden knob-assignment buttons of ``_KnobDevice``, both
  "Open KnobPanel" buttons, both Matlab ``execute`` buttons and an empty 1x1
  button of GTLView. ``common/KnobPanel.screen.json`` is deleted.
* **Cross-links.** GTLView opens BTSView and BTSView opens GTLView, each by a
  path relative to the panel's own file. BTSView's link takes the slot of its
  cut "Open KnobPanel" button. GTLView's KnobPanel button sat far from the
  booster/BTS TV bank, so its link takes the free slot under the BTS TV bank.
* **Source bugs.** The LTB Q5 glyph read Q4.2's supply status; it now reads the
  status its own knob shows. The BTS BPM5 plot was titled "BPM1". Two more
  source bugs, the LTB Q4.1 knob reading its setpoint and the BTS Q6.1 knob
  writing its readback, need no edit here: the element table gives each of
  those legacy names a setpoint and a readback row, and the rename map
  resolves every site to the row of its kind.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

#: The panel files, relative to ``screens/``.
GTL_VIEW = "gtb/GTLView.screen.json"
BTS_VIEW = "bts/BTSView.screen.json"

#: The file the cut "Open KnobPanel" buttons opened, relative to the library.
KNOB_PANEL = "screens/common/KnobPanel.screen.json"

#: Widgets removed, by file under ``screens/``.
CUT_NODES: dict[str, frozenset[str]] = {
    GTL_VIEW: frozenset({"w-322", "w-323", "w-415"}),
    BTS_VIEW: frozenset({"w-194", "w-195"}),
    "gtb/embedded/_KnobDevice.screen.json": frozenset({"w-006", "w-007"}),
    "bts/embedded/_KnobDevice.screen.json": frozenset({"w-006", "w-007"}),
}

#: Macros removed from every screen declaration and every embed instance.
CUT_MACROS = ("P", "R", "GAIN", "MIN", "MAX")

#: Macros removed from every ``_tv`` instance and declaration.
CUT_TV_MACROS = ("light_on", "paddle_in")

TV_FILE = "_tv.screen.json"
PADDLE = "paddle"
PADDLE_RB = "paddle_rb"

#: The props of the ``_tv`` lamp once it is a ``pv-led``.
LAMP_PROPS: dict[str, Any] = {
    "pv": "${light}",
    "onColor": "#ffff00",
    "offColor": {"$role": "Background", "fallback": "#808080"},
}


@dataclass(frozen=True)
class Link:
    """A cross-link button one panel gets.

    ``slot`` names the cut widget whose geometry the link takes; ``geometry``
    is used instead when that widget does not sit where the link belongs.
    """

    panel: str
    label: str
    target: str
    slot: str
    geometry: dict[str, int] | None = None


#: GTLView's link: the free slot under the BTS TV PADDLES bank.
GTL_LINK_GEOMETRY = {"x": 1960, "y": 490, "width": 130, "height": 24}

LINKS = (
    Link(GTL_VIEW, "Open BTSView", "../bts/BTSView.screen.json", "w-415", GTL_LINK_GEOMETRY),
    Link(BTS_VIEW, "Open GTLView", "../gtb/GTLView.screen.json", "w-195"),
)


@dataclass(frozen=True)
class CopyRulePv:
    """Point a widget's first rule PV at an embed macro of another widget."""

    panel: str
    node: str
    source: str
    macro: str


@dataclass(frozen=True)
class FixMacro:
    """Replace one free-text macro value of a widget, ``was`` -> ``now``."""

    panel: str
    node: str
    macro: str
    was: str
    now: str


#: The LTB Q5 glyph shows the supply status of the LTB Q5 knob.
Q5_GLYPH = CopyRulePv(GTL_VIEW, "w-300", "w-410", "COLOR_PV")
#: The BTS BPM5 plot carries its own title.
BPM5_TITLE = FixMacro(BTS_VIEW, "w-157", "BPM", "BTS BPM1", "BTS BPM5")


@dataclass
class Edits:
    """The structural edits, bound to how a paddle's readback is named.

    ``paddle_readback`` maps the value of a ``_tv`` instance's ``paddle`` macro
    to its ``paddle_rb`` value. A legacy name maps to itself, so the rename
    map resolves it to the readback row; a demo name maps to its readback.
    """

    paddle_readback: Callable[[str], str]
    errors: list[str] = field(default_factory=list)

    def edit(self, relative: str, document: dict[str, Any]) -> None:
        """Apply every structural edit to one screen, in place.

        ``relative`` is the screen's path under ``screens/`` (POSIX form).
        Problems are appended to ``errors``; the document may be partly edited.
        """
        _cut_declared(document, CUT_MACROS)
        if relative.rsplit("/", 1)[-1] == TV_FILE:
            _cut_declared(document, CUT_TV_MACROS)
            self._tv_screen(document)
        for node in _nodes(document.get("root")):
            if node.get("type") == "embedded-display":
                self._embed(node)
        for link in LINKS:
            if link.panel == relative:
                self._link(document, link)
        if Q5_GLYPH.panel == relative:
            self._copy_rule_pv(document, Q5_GLYPH)
        if BPM5_TITLE.panel == relative:
            self._fix_macro(document, BPM5_TITLE)
        cuts = CUT_NODES.get(relative)
        if cuts:
            _cut_nodes(document.get("root"), cuts)

    # --- the _tv sub-screen and its instances ------------------------------------

    def _tv_screen(self, document: dict[str, Any]) -> None:
        macros = document.setdefault("macros", [])
        names = [m.get("name") for m in macros if isinstance(m, dict)]
        if PADDLE_RB not in names:
            at = names.index(PADDLE) + 1 if PADDLE in names else len(macros)
            macros.insert(at, {"name": PADDLE_RB, "default": ""})
        for node in _nodes(document.get("root")):
            props = node.get("props")
            if node.get("type") != "pv-button" or not isinstance(props, dict):
                continue
            if props.get("pv") == "${paddle}" and props.get("readbackPV") == "${paddle}":
                props["readbackPV"] = "${paddle_rb}"
            elif props.get("pv") == "${light}":
                node["type"] = "pv-led"
                node["props"] = dict(LAMP_PROPS)

    def _embed(self, node: dict[str, Any]) -> None:
        props = node.get("props")
        macros = props.get("macros") if isinstance(props, dict) else None
        if not isinstance(macros, dict):
            return
        for name in CUT_MACROS:
            macros.pop(name, None)
        if not str(props.get("file", "")).endswith(TV_FILE):
            return
        for name in CUT_TV_MACROS:
            macros.pop(name, None)
        paddle = macros.get(PADDLE)
        if isinstance(paddle, str) and PADDLE_RB not in macros:
            try:
                macros[PADDLE_RB] = self.paddle_readback(paddle)
            except Exception as error:  # the rename map's error, reported with the rest
                self.errors.append(f"{node.get('id')}: macros.{PADDLE_RB}: {error}")

    # --- links and fixes ---------------------------------------------------------

    def _link(self, document: dict[str, Any], link: Link) -> None:
        root = document.get("root")
        if any(_opens(node, link.target) for node in _nodes(root)):
            return
        found = _find(root, link.slot)
        if link.geometry is not None:
            geometry = dict(link.geometry)
        elif found is not None:
            geometry = dict(found[0]["geometry"])
        else:
            self.errors.append(f"no {link.slot} to take the slot of '{link.label}'")
            return
        button = {
            "id": _next_id(document),
            "type": "regular-button",
            "props": {
                "label": link.label,
                "actions": [{"type": "open_display", "file": link.target, "target": "window"}],
            },
            "geometry": geometry,
        }
        if found is not None and link.geometry is None:
            _, siblings, index = found
            siblings.insert(index, button)
        else:
            root.setdefault("children", []).append(button)

    def _copy_rule_pv(self, document: dict[str, Any], fix: CopyRulePv) -> None:
        root = document.get("root")
        target, source = _find(root, fix.node), _find(root, fix.source)
        if target is None or source is None:
            missing = fix.node if target is None else fix.source
            self.errors.append(f"no {missing} for the {fix.node} rule fix")
            return
        value = ((source[0].get("props") or {}).get("macros") or {}).get(fix.macro)
        if not isinstance(value, str):
            self.errors.append(f"{fix.source} has no macro {fix.macro}")
            return
        try:
            target[0]["rules"][0]["pvs"][0]["name"] = value
        except (KeyError, IndexError, TypeError):
            self.errors.append(f"{fix.node} has no rule PV")

    def _fix_macro(self, document: dict[str, Any], fix: FixMacro) -> None:
        found = _find(document.get("root"), fix.node)
        macros = (found[0].get("props") or {}).get("macros") if found else None
        if not isinstance(macros, dict):
            self.errors.append(f"no {fix.node} with macros")
            return
        value = macros.get(fix.macro)
        if value == fix.was:
            macros[fix.macro] = fix.now
        elif value != fix.now:
            self.errors.append(
                f"{fix.node} macros.{fix.macro} is {value!r}, expected {fix.was!r} or {fix.now!r}"
            )


# --- tree helpers --------------------------------------------------------------------


def _nodes(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for child in node.get("children") or []:
            yield from _nodes(child)


def _find(node: Any, node_id: str) -> tuple[dict[str, Any], list[Any], int] | None:
    """The widget ``node_id``, its parent's child list and its index there."""
    if not isinstance(node, dict):
        return None
    children = node.get("children") or []
    for index, child in enumerate(children):
        if isinstance(child, dict) and child.get("id") == node_id:
            return child, children, index
        found = _find(child, node_id)
        if found is not None:
            return found
    return None


def _cut_nodes(node: Any, ids: frozenset[str]) -> None:
    if not isinstance(node, dict) or not isinstance(node.get("children"), list):
        return
    node["children"] = [
        child
        for child in node["children"]
        if not (isinstance(child, dict) and child.get("id") in ids)
    ]
    for child in node["children"]:
        _cut_nodes(child, ids)


def _cut_declared(document: dict[str, Any], names: tuple[str, ...]) -> None:
    macros = document.get("macros")
    if isinstance(macros, list):
        document["macros"] = [
            m for m in macros if not (isinstance(m, dict) and m.get("name") in names)
        ]


def _opens(node: dict[str, Any], target: str) -> bool:
    actions = (node.get("props") or {}).get("actions")
    return any(
        isinstance(a, dict) and a.get("type") == "open_display" and a.get("file") == target
        for a in (actions if isinstance(actions, list) else [])
    )


_ID = re.compile(r"^w-(\d+)$")


def _next_id(document: dict[str, Any]) -> str:
    numbers = [
        int(match.group(1))
        for node in _nodes(document.get("root"))
        if (match := _ID.match(str(node.get("id", ""))))
    ]
    return f"w-{max(numbers, default=0) + 1:03d}"
