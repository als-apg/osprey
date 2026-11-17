"""Views for the layout tests: every fixture screen, and invented views for the cases it lacks."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from typing import Any

from osprey.facility.panels.describe import describe, targets
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.view import (
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
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", target.id)
        by = f"-{target.by}" if target.by else ""
        return f"stub/{target.view}{by}-{target.kind}-{name}.screen.json"


def fixture_views(graph: PanelGraph) -> list[View]:
    """One view per target of the fixture graph."""
    return [describe(graph, t, symbols=SYMBOLS, above=ABOVE) for t in targets(graph)]


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


def rollup_view() -> View:
    """A roll-up: a root over every ruled channel; one child per channel, a three-channel
    child (with a grandchild the screen must not draw), and a child with no rule."""
    grandchild = Node("deep", None, RULED[:1], False)
    children = (
        *(Node(c, Link(c, Target("card", "device", c)), (c,), False) for c in RULED),
        Node("three", None, RULED[1:4], False, (grandchild,)),
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
            yield from links(getattr(value, f.name))


def walk(node: dict) -> Iterator[dict]:
    yield node
    for child in node.get("children", ()):
        yield from walk(child)


def nodes(doc: dict) -> list[dict]:
    return list(walk(doc["root"]))[1:]
