"""The geometry of a device class's drawing: its body's shapes in mm, by material.

Pure functions of the ``pictures.csv`` rows of one view
(:class:`~osprey.facility.panels.tables.PicturePart`); the layout scales and paints
what they return. The body is built as a magnet is seen from its end: a ring
yoke (an outer and an inner shape), poles from the yoke's inner edge to just
short of the bore, narrowing to a rounded nose; each coil as two rounded
blocks lying beside its pole; a corrector winding as a thin band outside the
coil; the bore. A shape row that states no size is drawn at a default
proportion of the bore (:func:`body`), and the body says so (:attr:`Body.sized`).

The format draws polygons, axis-aligned ellipses and axis-aligned rectangles
only: a rounded nose and a rounded block at an angle are polygons whose curves
are sampled (:data:`ARC_STEPS` points per quarter).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from osprey.facility.panels.tables import PicturePart

__all__ = [
    "DEFAULT_BORE_MM",
    "Body",
    "CoilMark",
    "Shape",
    "body",
    "member_number",
]

Point = tuple[float, float]

#: The bore radius a body is drawn about when no row sizes a bore, a gap or a yoke.
DEFAULT_BORE_MM = 15.0
#: Of a yoke's outer radius, the bore radius a body with only its yoke sized is drawn about.
BORE_OF_YOKE = 0.19
#: Of a yoke's outer radius (or of the nearest side of a convex outline): its inner edge.
YOKE_INNER = 0.82
#: An unsized yoke's inner edge, in bore radii (its outer edge follows by :data:`YOKE_INNER`).
YOKE_INNER_BORES = 4.3
#: An unsized pole's nose stops this many bore radii from the beam: short of the bore.
POLE_NOSE = 1.15
#: An unsized pole's width at its root: of its length for a tapered pole (more than
#: two); for two, this many bore radii (a bend's face is wide).
POLE_WIDTH = 0.4
FACE_WIDTH = 4.8
#: A tapered pole (of more than two): where its flanks start to narrow, as a share of
#: its length from the nose; how narrow its nose is, of its root width; the nose's
#: length, of the pole's length.
POLE_SHOULDER = 0.25
POLE_NOSE_WIDTH = 0.58
POLE_NOSE_LENGTH = 0.08
#: Of the room a pole's share of the circle has at a radius, the most it takes there.
POLE_ROOM = 0.75
#: An unsized coil block: of its pole's root width across, of its pole's length along;
#: how far short of the pole's root it ends, of the pole's length.
COIL_WIDTH = 0.44
COIL_LENGTH = 0.46
COIL_ROOT_GAP = 0.08
#: Between a pole and its coil, and between a coil and its winding: of the pole's width.
CLEARANCE = 0.02
#: An unsized corrector winding: of its pole's width across, of its coil's length along.
WINDING_WIDTH = 0.12
WINDING_LENGTH = 0.8
#: A rounded block's corner radius, of its smaller side.
ROUNDING = 0.15
#: Points per quarter circle of a sampled curve.
ARC_STEPS = 3
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
        role: What it is painted as: ``yoke``, ``window`` (the yoke's inside,
            the panel's ground), ``pole``, ``coil``, ``winding`` or ``bore``.
    """

    row: PicturePart
    form: str
    points: tuple[Point, ...]
    role: str


@dataclass(frozen=True)
class CoilMark:
    """Where a coil's number goes: on its pole's axis at *at* (mm), on out along
    *along* (a unit vector) by the layout's offset."""

    row: PicturePart
    number: str
    at: Point
    along: Point


@dataclass(frozen=True)
class Body:
    """A class's body in one view.

    Attributes:
        shapes: In painting order: yoke and its window, poles, coils, windings, bores.
        coils: Where each coil's number goes.
        beam: Where the beam is (a bore's or a gap's centre), or ``None``.
        reach: How far from the beam the yoke reaches, in mm, or ``None`` without
            a yoke: a part within it sits on the body.
        sized: Every shape row states its size, and the bore radius is stated.
    """

    shapes: tuple[Shape, ...]
    coils: tuple[CoilMark, ...]
    beam: Point | None
    reach: float | None
    sized: bool


# -- shapes ------------------------------------------------------------------------


def _local(a: float, b: float, turn: float, points: Sequence[Point]) -> tuple[Point, ...]:
    """*points* given as (along, across) in a frame turned by *turn* about (*a*, *b*)."""
    ux, uy = math.cos(turn), math.sin(turn)
    return tuple((a + u * ux - v * uy, b + u * uy + v * ux) for u, v in points)


def _arc(cu: float, cv: float, ru: float, rv: float, start: float, end: float) -> list[Point]:
    """Points of an elliptic arc about (*cu*, *cv*), from *start* to *end* degrees."""
    steps = max(1, round(abs(end - start) / 90 * ARC_STEPS))
    out = []
    for k in range(steps + 1):
        t = math.radians(start + (end - start) * k / steps)
        out.append((cu + ru * math.cos(t), cv + rv * math.sin(t)))
    return out


def _block(a: float, b: float, w: float, h: float, turn: float) -> tuple[Point, ...]:
    """A rounded block *w* across and *h* along *turn*, centred at (*a*, *b*)."""
    r = ROUNDING * min(w, h)
    hu, hv = h / 2 - r, w / 2 - r
    points: list[Point] = []
    for cu, cv, start in ((hu, hv, 0), (-hu, hv, 90), (-hu, -hv, 180), (hu, -hv, 270)):
        points += _arc(cu, cv, r, r, start, start + 90)
    return _local(a, b, turn, points)


def _pole(nose: float, root: float, w: float, turn: float, count: int) -> tuple[Point, ...]:
    """A pole along *turn* from *nose* to *root* (mm from the beam), *w* wide at its root:
    of more than two poles, narrowing to a rounded nose, never wider at a radius than
    :data:`POLE_ROOM` of the room its share of the circle has there; of two, a flat
    face with rounded corners."""
    length = root - nose
    half = w / 2
    if count <= 2:
        r = min(ROUNDING * w, length / 2)
        points = [(root, half)]
        points += _arc(nose + r, half - r, r, r, 90, 180)
        points += _arc(nose + r, -half + r, r, r, 180, 270)
        points.append((root, -half))
        return _local(0.0, 0.0, turn, points)

    def room(radius: float, wanted: float) -> float:
        return min(wanted, POLE_ROOM * max(radius, 0.0) * math.sin(math.pi / count))

    shoulder = nose + POLE_SHOULDER * length
    tip = POLE_NOSE_LENGTH * length
    narrow = room(nose + tip, POLE_NOSE_WIDTH * half)
    upper = [(root, room(root, half)), (shoulder, room(shoulder, half))]
    points = upper + _arc(nose + tip, 0.0, tip, narrow, 90, 270)
    points += [(u, -v) for u, v in reversed(upper)]
    return _local(0.0, 0.0, turn, points)


def _ellipse(a: float, b: float, w: float, h: float) -> tuple[Point, ...]:
    return ((a - w / 2, b - h / 2), (a + w / 2, b + h / 2))


# -- what the rows state -----------------------------------------------------------


def _xy(row: PicturePart) -> Point:
    """A row's coordinates in its view's plane: x, y on an end view; z, y on a side view."""
    return (row.z_mm, row.y_mm) if row.view == "side" else (row.x_mm, row.y_mm)


def _wh(row: PicturePart) -> tuple[float, float]:
    w = row.w_mm or 0.0
    return w, row.h_mm if row.h_mm is not None else w


def _same_angle(a: PicturePart, b: PicturePart) -> bool:
    apart = ((a.angle_deg or 0.0) - (b.angle_deg or 0.0)) % TURN
    return min(apart, TURN - apart) <= SAME_ANGLE


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


def _ellipse_reach(a: float, b: float, along: Point) -> float:
    """How far from its centre an ellipse of semi-axes *a*, *b* reaches along *along*."""
    if a <= 0 or b <= 0:
        return 0.0
    return 1 / math.sqrt((along[0] / a) ** 2 + (along[1] / b) ** 2)


def _ray_exit(ring: Sequence[Point], along: Point) -> float:
    """How far from the beam a ray along *along* last crosses the polygon *ring*."""
    far = 0.0
    dx, dy = along
    for i in range(len(ring)):
        (px, py), (qx, qy) = ring[i - 1], ring[i]
        ex, ey = qx - px, qy - py
        det = dx * (-ey) + ex * dy
        if abs(det) < 1e-12:
            continue
        t = (px * (-ey) + ex * py) / det  # along the ray
        u = (dx * py - dy * px) / det  # along the side
        if t > 0 and 0 <= u <= 1:
            far = max(far, t)
    return far


def _nearest_side(outline: Sequence[Point]) -> float:
    return min(_to_segment(outline[i - 1], outline[i]) for i in range(len(outline)))


def _yoke_edges(
    yoke: PicturePart | None, ring: Sequence[Point], bore: float
) -> tuple[float, float]:
    """The yoke's inner and outer reach from the beam: a sized yoke's; a convex
    outline's (its nearest side the outside, the inside :data:`YOKE_INNER` of it);
    a concave outline draws its window, whose nearest side is the inside; else
    from the bore."""
    if yoke is not None and yoke.sized:
        outer = min(_wh(yoke)) / 2
        return YOKE_INNER * outer, outer
    if ring:
        near = _nearest_side(ring)
        far = max(math.hypot(a, b) for a, b in ring)
        return (YOKE_INNER * near, far) if _convex(ring) else (near, far)
    inner = YOKE_INNER_BORES * bore
    return inner, inner / YOKE_INNER


def _bore_radius(
    rows: Sequence[PicturePart], yoke: PicturePart | None, ring: Sequence[Point]
) -> tuple[float, bool]:
    """The radius an unsized body is drawn about, and whether a row states it."""
    for row in rows:
        if row.kind == "bore" and row.sized:
            return min(_wh(row)) / 2, True
        if row.kind == "gap" and row.sized:
            return (row.w_mm or 0.0) / 2, True
    if yoke is not None and yoke.sized:
        return BORE_OF_YOKE * min(_wh(yoke)) / 2, False
    if ring:
        return BORE_OF_YOKE * _nearest_side(ring), False
    return DEFAULT_BORE_MM, False


def body(rows: Sequence[PicturePart], outline: Sequence[PicturePart] = ()) -> Body:
    """The body that *rows* (the shape rows of one view) and *outline* (its outline
    rows, in file order) state, unsized rows at their defaults."""
    ring = [_xy(row) for row in outline]
    yoke = next((r for r in rows if r.kind == "yoke"), None)
    bore, sized = _bore_radius(rows, yoke, ring)
    inner, outer = _yoke_edges(yoke, ring, bore)
    poles = [r for r in rows if r.kind == "pole"]
    count = max(len(poles), 2)
    tapered = count > 2
    default_length = inner * (1 + CLEARANCE) - POLE_NOSE * bore
    pole_w = POLE_WIDTH * default_length if tapered else FACE_WIDTH * bore
    shapes: list[Shape] = []
    reach: float | None = None
    edge: Callable[[Point], float] | None = None  # the yoke's outer edge along a direction
    if yoke is not None:
        sized = sized and yoke.sized
        a, b = _xy(yoke)
        w, h = _wh(yoke) if yoke.sized else (2 * outer, 2 * outer)
        shapes.append(Shape(yoke, "ellipse", _ellipse(a, b, w, h), "yoke"))
        window = _ellipse(a, b, YOKE_INNER * w, YOKE_INNER * h)
        shapes.append(Shape(yoke, "ellipse", window, "window"))
        reach = max(w, h) / 2
        edge = lambda along, a=w / 2, b=h / 2: _ellipse_reach(a, b, along)  # noqa: E731
    elif ring:
        shapes.append(Shape(outline[0], "polygon", tuple(ring), "yoke"))
        if _convex(ring):  # the outside only: its window at the inner edge
            window = tuple((YOKE_INNER * a, YOKE_INNER * b) for a, b in ring)
            shapes.append(Shape(outline[0], "polygon", window, "window"))
        reach = outer
        edge = lambda along, ring=tuple(ring): _ray_exit(ring, along)  # noqa: E731
    # each pole: (nose, root, width) along its angle, mm from the beam
    spans: list[tuple[float, float, float]] = []
    for row in poles:
        sized = sized and row.sized
        turn = math.radians(row.angle_deg or 0.0)
        if row.sized:
            w, h = _wh(row)
            a, b = _xy(row)
            centre = a * math.cos(turn) + b * math.sin(turn)
            nose, root = centre - h / 2, centre + h / 2
        else:
            w, nose, root = pole_w, POLE_NOSE * bore, inner * (1 + CLEARANCE)
        spans.append((nose, root, w))
        shapes.append(Shape(row, "polygon", _pole(nose, root, w, turn, count), "pole"))
    coils: list[CoilMark] = []
    coil_w: dict[int, float] = {}
    for kind in ("coil", "corrector_winding"):
        for row in (r for r in rows if r.kind == kind):
            sized = sized and row.sized
            turn = math.radians(row.angle_deg or 0.0)
            along = (math.cos(turn), math.sin(turn))
            i = next((n for n, p in enumerate(poles) if _same_angle(p, row)), None)
            nose, root, pw = spans[i] if i is not None else (POLE_NOSE * bore, inner, pole_w)
            length = root - nose
            key = i if i is not None else -1
            middle = root - COIL_ROOT_GAP * length - COIL_LENGTH * length / 2
            flank = pw / 2  # the pole's half width where the coil lies
            if count > 2:  # and the room between two poles there: two coils share it
                flank = min(flank, POLE_ROOM * middle * math.sin(math.pi / count))
                between = 2 * middle * math.sin(math.pi / count) - 2 * flank
                fit = (between - 4 * CLEARANCE * pw) / (2 * (1 + WINDING_WIDTH / COIL_WIDTH))
            else:
                fit = COIL_WIDTH * pw
            side = coil_w.get(key, max(0.0, min(COIL_WIDTH * pw, fit)))
            a = b = 0.0
            if row.sized:
                w, h = _wh(row)
                a, b = _xy(row)
                middle = a * along[0] + b * along[1]
            else:
                h = COIL_LENGTH * length
                w = side
                if kind == "corrector_winding":
                    w, h = side * WINDING_WIDTH / COIL_WIDTH, WINDING_LENGTH * h
            if kind == "coil":
                coil_w[key] = w
            role = "coil" if kind == "coil" else "winding"
            if i is None and row.sized:  # no pole to lie beside: where the row puts it
                shapes.append(Shape(row, "polygon", _block(a, b, w, h, turn), role))
            else:
                offset = flank + CLEARANCE * pw + w / 2
                if kind == "corrector_winding":
                    offset += side + CLEARANCE * pw
                for sign in (1, -1):
                    ((ca, cb),) = _local(0.0, 0.0, turn, [(middle, sign * offset)])
                    shapes.append(Shape(row, "polygon", _block(ca, cb, w, h, turn), role))
            if kind == "coil":
                number = member_number("coil", row.part, len(coils) + 1)
                out = edge(along) if edge is not None else middle + h / 2
                coils.append(CoilMark(row, number, (out * along[0], out * along[1]), along))
    beam: Point | None = None
    for row in (r for r in rows if r.kind in ("bore", "gap")):
        a, b = _xy(row)
        beam = beam or (a, b)
        if row.kind == "bore":
            w, h = _wh(row) if row.sized else (2 * bore, 2 * bore)
            shapes.append(Shape(row, "ellipse", _ellipse(a, b, w, h), "bore"))
    return Body(tuple(shapes), tuple(coils), beam, reach, sized)


def member_number(word: str, name: str, ordinal: int) -> str:
    """A member's number within its kind: what follows the kind's *word* in *name*
    (``thermal switch 3`` under thermal switch: ``3``), else its *ordinal*."""
    rest = name.removeprefix(word).strip()
    return rest if name.startswith(word + " ") and rest else str(ordinal)
