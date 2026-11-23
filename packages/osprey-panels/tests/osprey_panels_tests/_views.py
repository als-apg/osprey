"""Views for the layout tests: every fixture screen, and invented views for the cases it lacks."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from typing import Any

from osprey_panels.describe import describe, targets
from osprey_panels.model import PanelGraph
from osprey_panels.view import (
    Fact,
    Light,
    Link,
    Node,
    Section,
    Strip,
    StripElement,
    Target,
    TrayEntry,
    ValueRow,
    View,
)

#: Class to symbol for the fixture's place views; a quadrupole class names no polarity.
SYMBOLS = {"Quadrupole": "quadrupole", "Dipole": "bend", "Pump": "ion_pump"}
ABOVE = frozenset({"ion_pump"})


class Screens:
    """A stub ``screens``: a file name per target, and every target it was asked for."""

    def __init__(self) -> None:
        self.asked: list[Target] = []

    def __call__(self, target: Target) -> str:
        self.asked.append(target)
        return f"{self.doc_id(target)}.screen.json"

    @staticmethod
    def doc_id(target: Target) -> str:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", target.id)
        by = f"-{target.by}" if target.by else ""
        return f"stub/{target.view}{by}-{target.kind}-{name}"


def stub_render(
    view: View, graph: PanelGraph, screens: Screens | None = None, colours: dict | None = None
) -> dict:
    """A view drawn with the stub ``screens``, its document named the stub's way."""
    from osprey_panels.layout import render

    screens = Screens() if screens is None else screens
    return render(
        view,
        graph=graph,
        screens=screens,
        doc_id=Screens.doc_id(view.id),
        colours=colours or {},
    )


def with_symbols(
    graph: PanelGraph, symbols: dict[str, str], above: frozenset[str] = frozenset()
) -> PanelGraph:
    """*graph* with ``symbols.csv`` replaced: each class drawn as its symbol, in the
    lane off the beam when the symbol is in *above*, else on the beam."""
    import copy

    from osprey_panels.tables import StripSymbol

    out = copy.copy(graph)
    out._symbols = {  # type: ignore[attr-defined]
        cls: [StripSymbol(cls, None, kind, "above" if kind in above else "beam", None)]
        for cls, kind in symbols.items()
    }
    return out


def fixture_views(graph: PanelGraph) -> list[View]:
    """One view per target of the fixture graph."""
    return [describe(graph, t) for t in targets(graph)]  # the fixture's symbols.csv


def card(title: str, *items: Any) -> View:
    return View(Target("card", "device", title), title, (), (Section("items", tuple(items)),))


def strip_view(elements: list[StripElement], tray: tuple[TrayEntry, ...] = ()) -> View:
    extent = (min(e.s_m for e in elements), max(e.s_m for e in elements)) if elements else None
    return View(
        Target("place", "place", "XR/TEST"),
        "test strip",
        (),
        (Section("strip", (Strip(tuple(elements), tray, extent),)),),
    )


def element(thing: str, s: float, symbol: str, lane: str = "beam", link: bool = False):
    target = Link(thing, Target("card", "device", thing)) if link else None
    return StripElement(thing, None, s, symbol, lane, target)


def invented_views() -> list[View]:
    """Views with the cases the fixture lacks."""
    dense = [
        element(f"D{i:03d}", i * 0.1, ("quadrupole_f", "quadrupole_d", "bend")[i % 3])
        for i in range(300)
    ]
    ruler = [element(f"R{i}", float(s), "bpm", link=True) for i, s in enumerate((0, 7, 12.5, 30))]
    polarity = [
        element("QF", 1.0, "quadrupole_f"),
        element("QD", 2.0, "quadrupole_d"),
        element("QX", 3.0, "quadrupole"),
        element("IP", 3.5, "ion_pump", lane="above"),
        element("ODD", 4.0, "no-such-symbol"),
    ]
    tray = (TrayEntry("T1", None, "no position", Link("T1", Target("card", "device", "T1"))),)
    many = tuple(f"XR99:X{i}:State-Sts" for i in range(40))
    return [
        strip_view(dense),
        strip_view(ruler),
        strip_view(polarity, tray),
        card(
            "values",
            ValueRow("sp", "A:SP", "A:RB", ("absolute", 0.05), True),
            ValueRow("rel", "B:SP", "B:RB", ("relative", 0.01), True, inferred=True),
            ValueRow("ro", "C:RB", None, None, False),
            Light("XR01-PS1:On-Sts", inferred=True),
            Fact("note", "text", basis="inferred", standing="contested", caveat="assumed"),
            Link("other", Target("card", "device", "XR01:PS1"), basis="inferred"),
        ),
        card("many", Node("big", None, many, False)),
    ]


#: Rule-bearing fixture channels, for an invented roll-up.
RULED = (
    "XR01-IP1:State-Sts",
    "XR01-PS1:On-Sts",
    "XR02-IP1:Pressure-RB",
    "XR02-QD1:Current-RB",
    "XR02-QF1:Current-RB",
    "XR02-BEND1:Current-RB",
)

#: Three of them whose joint rule is longer than any one channel's.
THREE = (RULED[0], RULED[1], RULED[3])


def rollup_view() -> View:
    """A roll-up: a root over every ruled channel; one child per channel, a three-channel
    child (with a grandchild the screen must not draw), and a child with no rule."""
    grandchild = Node("deep", None, RULED[:1], False)
    children = (
        *(Node(c, Link(c, Target("card", "device", c)), (c,), False) for c in RULED),
        Node("three", None, THREE, False, (grandchild,)),
        Node("none", None, (), True),
    )
    root = Node("ROOT", None, RULED, False, children)
    target = Target("rollup", "place", "ROOT", by="place")
    return View(target, "ROOT health", (), (Section("roll-up", (root,)),))


def links(value: Any) -> Iterator[Link]:
    """Every link inside a view, at any depth."""
    if isinstance(value, Link):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from links(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            if (type(value).__name__, f.name) == ("CalibrationTallies", "more"):
                continue  # drawn only when the chips overflow their line
            yield from links(getattr(value, f.name))


def walk(node: dict) -> Iterator[dict]:
    yield node
    for child in node.get("children", ()):
        yield from walk(child)


def nodes(doc: dict) -> list[dict]:
    return list(walk(doc["root"]))[1:]


FORMULA = "formula://"


def formulas(doc: dict) -> dict[str, dict]:
    """The document's screen formulas by the name a rule's pv reads them under."""
    return {FORMULA + f["name"]: f for f in doc.get("formulas", ())}


def printed(doc: dict, node: dict) -> str | None:
    """The channel whose value a formula text label prints, or ``None``."""
    by_name = {f["name"]: f for f in doc.get("formulas", ())}
    for rule in node.get("rules", ()):
        if rule.get("targetProperty") == "text" and "formula" in rule:
            return by_name[rule["formula"].removeprefix(FORMULA)]["bindings"]["v"]
    return None


def bound(doc: dict, node: dict) -> tuple[str, ...]:
    """The channels a node's rules read, each once, through the screen formulas they
    name: in the light's order (a formula's ``v<i>`` is its channel ``i``)."""
    found: dict[str, tuple[int, int]] = {}
    for rule in node["rules"]:
        for pv in rule["pvs"]:
            formula = formulas(doc).get(pv["name"])
            if formula is None:
                found.setdefault(pv["name"], (len(found), 0))
                continue
            for key, channel in formula["bindings"].items():
                found.setdefault(channel, (int(key[1:]), 0))
    return tuple(sorted(found, key=lambda channel: found[channel]))


def resolved(doc: dict, values: dict) -> dict:
    """*values* by channel, with each screen formula's value under its pv name, as the
    formula engine gives it: no value while any of its channels has none."""
    out = dict(values)
    for name, found in formulas(doc).items():
        inputs = {var: values.get(channel) for var, channel in found["bindings"].items()}
        if any(value is None for value in inputs.values()):
            out[name] = None
            continue
        try:
            out[name] = eval(found["expression"], {"__builtins__": {}}, inputs)
        except TypeError:
            out[name] = None
    return out
