"""The schematic arrangement of a place: devices in true order, drawn compressed.

A sector is metres long and centimetres wide, so a place view cannot be drawn
to scale. :func:`arrange` keeps the true order of the devices and gives each a
symbol of fixed width; the space between neighbours grows with their distance
along the beam up to a cap. :func:`locator` is the to-scale companion: where
each device sits on a ruler of the place's whole extent.

Both are pure: no screen format, no colour.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from osprey_panels.view import Link, StripElement

#: Width, in units, of a symbol with no entry in the widths mapping.
DEFAULT_WIDTH = 24.0
#: Fraction of its symbol's width an above-lane element takes along the strip.
ABOVE_FRACTION = 0.55
#: Space between neighbours: ``min(GAP_MAX, GAP_MIN + GAP_PER_M * metres)``.
GAP_MIN = 4.0
GAP_PER_M = 14.0
GAP_MAX = 26.0
#: The narrowest a gap gets, in parts of its natural width, before symbols shrink.
GAP_FLOOR = 0.25


@dataclass(frozen=True)
class Placed:
    """One element placed on the strip.

    Attributes:
        thing: The element's id.
        symbol: The symbol it is drawn as.
        lane: ``above`` or ``beam``.
        s_m: Its true ring position.
        x: The centre of its symbol, in strip units.
        width: Its symbol's width after scaling.
        level: Height of an ``above`` element, 0 or 1, alternating in order of
            appearance; 0 for a beam element.
        link: Its card, or ``None``.
        channels: Its rule-bearing channels, as the element states them.
        short: Its stated short name, or ``None``.
        annotations: The channels whose values it shows, each with its unit.
    """

    thing: str
    symbol: str
    lane: str
    s_m: float
    x: float
    width: float
    level: int
    link: Link | None
    channels: tuple[str, ...] = ()
    short: str | None = None
    annotations: tuple[tuple[str, str | None], ...] = ()


def arrange(
    elements: Sequence[StripElement],
    widths: Mapping[str, float] | None = None,
    strip_width: float = 1000.0,
) -> list[Placed]:
    """Place the elements along a strip of ``strip_width`` units.

    See :func:`arrange_scaled`; this drops the scale.
    """
    return arrange_scaled(elements, widths, strip_width)[0]


def arrange_scaled(
    elements: Sequence[StripElement],
    widths: Mapping[str, float] | None = None,
    strip_width: float = 1000.0,
    *,
    fill: bool = False,
) -> tuple[list[Placed], float]:
    """Place the elements along a strip of ``strip_width`` units.

    Sorts by ``s_m`` (equal positions keep their input order) and walks in
    order: a gap, then the symbol. Symbols are never enlarged. A strip too
    long for ``strip_width`` first narrows its gaps, down to
    :data:`GAP_FLOOR` of their natural width; only then are the symbols
    compressed too. A shorter strip is centred as a block; with *fill* its
    gaps widen, each in proportion to its natural width, so that it spans
    the strip.

    Args:
        elements: The devices; any order.
        widths: Symbol name to width in units; others get :data:`DEFAULT_WIDTH`.
        strip_width: The width of the strip.
        fill: Widen the gaps of a short strip to span the width.

    Returns:
        One :class:`Placed` per element, in beam order, and the scale applied
        to the symbols (at most 1; below 1 the strip was compressed, so a
        renderer may choose to break it into rows).
    """
    widths = widths or {}
    ordered = sorted(elements, key=lambda e: e.s_m)
    sizes: list[float] = []
    gaps: list[float] = []
    previous: float | None = None
    for e in ordered:
        if previous is not None:
            gaps.append(min(GAP_MAX, GAP_MIN + GAP_PER_M * (e.s_m - previous)))
        w = float(widths.get(e.symbol, DEFAULT_WIDTH))
        sizes.append(w * ABOVE_FRACTION if e.lane == "above" else w)
        previous = e.s_m
    drawn, spaces = sum(sizes), sum(gaps)
    scale, stretch = 1.0, 1.0
    if drawn + spaces > strip_width:
        if spaces and drawn + spaces * GAP_FLOOR <= strip_width:
            stretch = (strip_width - drawn) / spaces
        else:
            stretch = GAP_FLOOR
            scale = strip_width / (drawn + spaces * GAP_FLOOR)
    elif fill and spaces:
        stretch = (strip_width - drawn) / spaces
    total = (drawn + spaces * stretch) * scale
    cursor = (strip_width - total) / 2
    out: list[Placed] = []
    above = 0
    for i, (e, w) in enumerate(zip(ordered, sizes, strict=True)):
        if i:
            cursor += gaps[i - 1] * stretch * scale
        level = 0
        if e.lane == "above":
            level = above % 2
            above += 1
        width = w * scale
        out.append(
            Placed(
                e.thing,
                e.symbol,
                e.lane,
                e.s_m,
                cursor + width / 2,
                width,
                level,
                e.link,
                e.channels,
                e.short,
                e.annotations,
            )
        )
        cursor += width
    return out, scale


def locator(
    elements: Iterable[StripElement],
    extent_m: float,
    strip_width: float = 1000.0,
) -> list[tuple[str, float]]:
    """Each element's to-scale position on a ruler of ``extent_m`` metres.

    ``s_m`` is taken from zero; an element outside the extent is clamped to its
    end, and a zero extent puts everything at the centre.
    """
    out: list[tuple[str, float]] = []
    for e in elements:
        fraction = 0.5 if extent_m <= 0 else min(max(e.s_m / extent_m, 0.0), 1.0)
        x = fraction * strip_width
        out.append((e.thing, x))
    return out
