"""The geometry of a device class's drawing: its body's shapes in mm, and the tags of its parts.

Pure functions of the ``pictures.csv`` rows of one view
(:class:`~osprey.facility.panels.tables.PicturePart`); the layout scales and draws
what they return. A shape row that states no size is drawn at a default
proportion of the bore (:func:`body`), and the body says so (:attr:`Body.sized`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from osprey.facility.panels.tables import PicturePart

__all__ = [
    "DEFAULT_BORE_MM",
    "Body",
    "CoilMark",
    "Shape",
    "body",
    "member_number",
    "tag_prefixes",
]

Point = tuple[float, float]

#: The bore radius a body is drawn about when no row sizes a bore, a gap or a yoke.
DEFAULT_BORE_MM = 15.0
#: Of a yoke's radius, the bore radius a body with only its yoke sized is drawn about.
BORE_OF_YOKE = 0.2
#: Of a yoke's radius (or of the nearest point of its outline), where its inner edge is.
YOKE_INNER = 0.8
#: An unsized pole's length when no yoke states where it ends, in bore radii.
POLE_LENGTH = 2.5
#: An unsized pole's width: this many times the chord its share of the bore subtends.
POLE_WIDTH = 1.4
#: A pole's tip (of more than two poles): this share of the circle's chord its pole has there.
POLE_TIP = 0.8
#: An unsized coil side: of its pole's width and of its pole's length.
COIL_WIDTH = 0.3
COIL_LENGTH = 0.55
#: An unsized corrector winding: of its coil side's width.
WINDING_WIDTH = 0.3
#: Between a coil side and a corrector winding, of the pole's width.
WINDING_GAP = 0.05
#: An unsized yoke's outer radius: of the pole's root radius.
YOKE_OUTER = 1.25
#: A coil and a pole this close in angle, in degrees, are one pole's.
SAME_ANGLE = 0.5
TURN = 360.0


@dataclass(frozen=True)
class Shape:
    """One shape of a body, in mm in the view's plane (y up).

    Attributes:
        row: The row it draws.
        form: ``ellipse`` (``points`` are two opposite corners of its box) or
            ``polygon`` (``points`` are its corners).
        points: As ``form`` says.
    """

    row: PicturePart
    form: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class CoilMark:
    """Where a coil's number goes: at the coil's centre, on its pole between its sides."""

    row: PicturePart
    number: str
    at: Point  # mm


@dataclass(frozen=True)
class Body:
    """A class's body in one view.

    Attributes:
        shapes: In drawing order: yokes and the outline, poles, coils, corrector
            windings, bores.
        coils: Where each coil's number goes.
        beam: Where the beam is (a bore's or a gap's centre), or ``None``.
        sized: Every shape row states its size, and the bore radius is stated.
    """

    shapes: tuple[Shape, ...]
    coils: tuple[CoilMark, ...]
    beam: Point | None
    sized: bool


def _rectangle(a: float, b: float, w: float, h: float, turn: float) -> tuple[Point, ...]:
    """The corners of a rectangle *w* wide and *h* long centred at (*a*, *b*), its length
    along the direction *turn* (radians, counter-clockwise from +x)."""
    ux, uy = math.cos(turn), math.sin(turn)
    vx, vy = -uy, ux
    return tuple(
        (a + s * ux * h / 2 + t * vx * w / 2, b + s * uy * h / 2 + t * vy * w / 2)
        for s, t in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    )


def _pole(a: float, b: float, w: float, h: float, turn: float, count: int) -> tuple[Point, ...]:
    """A pole centred at (*a*, *b*), *w* wide at its root and *h* long along *turn*: a
    rectangle for two poles (a bend's flat faces); for more, its tip toward the
    beam narrowed to :data:`POLE_TIP` of its share of the circle there, so poles
    do not meet at the bore."""
    ux, uy = math.cos(turn), math.sin(turn)
    vx, vy = -uy, ux
    reach = a * ux + b * uy  # the centre's distance out along the pole
    inward = -1 if reach >= 0 else 1  # the tip is the end nearer the beam
    tip_r = abs(reach) - h / 2
    tip = w / 2
    if count > 2 and tip_r > 0:
        tip = min(w / 2, POLE_TIP * tip_r * math.sin(math.pi / count))
    root = (a - inward * ux * h / 2, b - inward * uy * h / 2)
    front = (a + inward * ux * h / 2, b + inward * uy * h / 2)
    return (
        (root[0] + vx * w / 2, root[1] + vy * w / 2),
        (front[0] + vx * tip, front[1] + vy * tip),
        (front[0] - vx * tip, front[1] - vy * tip),
        (root[0] - vx * w / 2, root[1] - vy * w / 2),
    )


def _xy(row: PicturePart) -> Point:
    """A row's coordinates in its view's plane: x, y on an end view; z, y on a side view."""
    return (row.z_mm, row.y_mm) if row.view == "side" else (row.x_mm, row.y_mm)


def _same_angle(a: PicturePart, b: PicturePart) -> bool:
    apart = ((a.angle_deg or 0.0) - (b.angle_deg or 0.0)) % TURN
    return min(apart, TURN - apart) <= SAME_ANGLE


def _half(row: PicturePart) -> float:
    """Half the smaller extent of a sized ellipse row."""
    w = row.w_mm or 0.0
    h = row.h_mm if row.h_mm is not None else w
    return min(w, h) / 2


def _bore_radius(rows: Sequence[PicturePart], outline: Sequence[Point]) -> tuple[float, bool]:
    """The radius an unsized body is drawn about, and whether a row states it."""
    for row in rows:
        if row.kind == "bore" and row.sized:
            return _half(row), True
        if row.kind == "gap" and row.sized:
            return (row.w_mm or 0.0) / 2, True
    for row in rows:
        if row.kind == "yoke" and row.sized:
            return BORE_OF_YOKE * _half(row), False
    if outline:
        return BORE_OF_YOKE * _inner_edge(outline), False
    return DEFAULT_BORE_MM, False


def _pole_root(rows: Sequence[PicturePart], outline: Sequence[Point], bore: float) -> float:
    """The radius an unsized pole ends at: the yoke's inner edge, as far as it is stated."""
    for row in rows:
        if row.kind == "yoke" and row.sized:
            return YOKE_INNER * _half(row)
    if outline:
        return _inner_edge(outline)
    return bore * (1 + POLE_LENGTH)


def _inner_edge(outline: Sequence[Point]) -> float:
    """Where a yoke stated as a polygon has its inner edge, from the beam: a convex
    outline (a hexagon, a rectangle) is the yoke's outside, its inner edge
    :data:`YOKE_INNER` of the nearest side; a concave one (a C, an H) draws its
    window, whose nearest side is the inner edge."""
    near = min(_to_segment(outline[i - 1], outline[i]) for i in range(len(outline)))
    return YOKE_INNER * near if _convex(outline) else near


def _to_segment(p: Point, q: Point) -> float:
    """The distance from the origin to the segment *p*-*q*."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, -(p[0] * dx + p[1] * dy) / length))
    return math.hypot(p[0] + t * dx, p[1] + t * dy)


def _convex(points: Sequence[Point]) -> bool:
    """Whether a polygon turns one way only at every corner."""
    turns = set()
    for i in range(len(points)):
        (ax, ay), (bx, by), (cx, cy) = points[i - 2], points[i - 1], points[i]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross:
            turns.add(cross > 0)
    return len(turns) <= 1


def body(rows: Sequence[PicturePart], outline: Sequence[PicturePart] = ()) -> Body:
    """The body that *rows* (the shape rows of one view) and *outline* (its outline
    rows, in file order) state, unsized rows at their defaults."""
    ring = [_xy(row) for row in outline]
    bore, bore_stated = _bore_radius(rows, ring)
    root = max(_pole_root(rows, ring, bore), bore * 1.5)
    poles = [r for r in rows if r.kind == "pole"]
    count = max(len(poles), 2)
    pole_w = POLE_WIDTH * 2 * bore * math.tan(math.pi / (2 * count))
    pole_h = root - bore
    shapes: list[Shape] = []
    coils: list[CoilMark] = []
    beam: Point | None = None
    sized = bore_stated
    pole_size: dict[int, tuple[float, float]] = {}  # by row index: width, length
    coil_size: dict[int, float] = {}  # by pole index: the coil side's width

    def pole_of(row: PicturePart) -> int | None:
        return next((i for i, p in enumerate(poles) if _same_angle(p, row)), None)

    for row in (r for r in rows if r.kind == "yoke"):
        sized = sized and row.sized
        a, b = _xy(row)
        if row.sized:
            w = row.w_mm or 0.0
            h = row.h_mm if row.h_mm is not None else w
        else:
            w = h = 2 * YOKE_OUTER * root
        shapes.append(Shape(row, "ellipse", ((a - w / 2, b - h / 2), (a + w / 2, b + h / 2))))
    if ring and outline:
        shapes.append(Shape(outline[0], "polygon", tuple(ring)))
    for i, row in enumerate(poles):
        sized = sized and row.sized
        turn = math.radians(row.angle_deg or 0.0)
        if row.sized:
            w = row.w_mm or 0.0
            h = row.h_mm if row.h_mm is not None else w
            a, b = _xy(row)
        else:
            w, h = pole_w, pole_h
            r = bore + h / 2
            a, b = r * math.cos(turn), r * math.sin(turn)
        pole_size[i] = (w, h)
        shapes.append(Shape(row, "polygon", _pole(a, b, w, h, turn, count)))
    for kind in ("coil", "corrector_winding"):
        for row in (r for r in rows if r.kind == kind):
            sized = sized and row.sized
            turn = math.radians(row.angle_deg or 0.0)
            along = (math.cos(turn), math.sin(turn))
            across = (-along[1], along[0])
            i = pole_of(row)
            pw, ph = pole_size.get(i, (pole_w, pole_h)) if i is not None else (pole_w, pole_h)
            side = coil_size.get(i, COIL_WIDTH * pw) if i is not None else COIL_WIDTH * pw
            if row.sized:
                w = row.w_mm or 0.0
                h = row.h_mm if row.h_mm is not None else w
                a, b = _xy(row)
            else:
                w = side if kind == "coil" else WINDING_WIDTH * side
                h = COIL_LENGTH * ph
                r = bore + ph - h / 2  # at the pole's root
                if i is not None and poles[i].sized:  # along the stated pole
                    pa, pb = _xy(poles[i])
                    r = pa * along[0] + pb * along[1] + ph / 2 - h / 2
                a, b = r * along[0], r * along[1]
            if i is None and row.sized:  # no pole to flank: where the row puts it
                shapes.append(Shape(row, "polygon", _rectangle(a, b, w, h, turn)))
            else:
                offset = pw / 2 + w / 2
                if kind == "coil":
                    coil_size[i if i is not None else -1] = w
                else:
                    offset += side + WINDING_GAP * pw
                for sign in (1, -1):
                    ca, cb = a + sign * offset * across[0], b + sign * offset * across[1]
                    shapes.append(Shape(row, "polygon", _rectangle(ca, cb, w, h, turn)))
            if kind == "coil":
                number = member_number("coil", row.part, len(coils) + 1)
                coils.append(CoilMark(row, number, (a, b)))
    for row in (r for r in rows if r.kind in ("bore", "gap")):
        a, b = _xy(row)
        beam = beam or (a, b)
        if row.kind == "gap":
            continue  # the pole faces show it
        if row.sized:
            w = row.w_mm or 0.0
            h = row.h_mm if row.h_mm is not None else w
        else:
            w = h = 2 * bore
        shapes.append(Shape(row, "ellipse", ((a - w / 2, b - h / 2), (a + w / 2, b + h / 2))))
    return Body(tuple(shapes), tuple(coils), beam, sized)


def member_number(word: str, name: str, ordinal: int) -> str:
    """A member's number within its kind: what follows the kind's *word* in *name*
    (``thermal switch 3`` under thermal switch: ``3``), else its *ordinal*."""
    rest = name.removeprefix(word).strip()
    return rest if name.startswith(word + " ") and rest else str(ordinal)


def tag_prefixes(kinds: Sequence[str]) -> dict[str, str]:
    """A short tag per kind of part, its words' initials in capitals (``thermal_switch``:
    ``TS``); kinds that would share one take more letters of their first word until
    none do (``terminal``, ``thermocouple``: ``TE``, ``TH``)."""
    kinds = list(dict.fromkeys(kinds))
    more = dict.fromkeys(kinds, 0)

    def tag(kind: str) -> str:
        words = [w for w in kind.replace("_", " ").split() if w] or [kind]
        first = words[0][: 1 + more[kind]]
        return (first + "".join(w[0] for w in words[1:])).upper()

    for _round in range(max((len(k) for k in kinds), default=0) + 1):
        tags = {kind: tag(kind) for kind in kinds}
        clashing = [k for k in kinds if list(tags.values()).count(tags[k]) > 1]
        if not clashing:
            return tags
        for kind in clashing:
            more[kind] += 1
    return {kind: tag(kind) for kind in kinds}
