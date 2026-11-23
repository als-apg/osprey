"""The geometry of a device class's drawing: a magnet seen from its end, in mm.

Pure functions of the ``pictures.csv`` rows of one view
(:class:`~osprey_panels.tables.PicturePart`); the layout scales and paints
what they return. Every proportion is a drawing unit (a *unit*, :data:`BORE` of
them the bore's radius); a stated size sets the unit, and a shape row that states
no size is drawn at the default proportion (:attr:`Body.sized` says so).

A magnet of more than two poles (:func:`body`): a yoke ring (or the polygon the
outline rows state) with its cavity cut out; each pole a broad body from the
cavity inwards, shoulders at :data:`POLE_SHOULDER`, a rounded tip that stops
short of the bore; each coil two rounded blocks along the pole's flanks; a
corrector winding a thin band outside each block (a second winding at the
same angle stacked outside the first); the bore a disc with the beam point;
each coil's number on the yoke ring at its angle. A magnet of two poles: a
rectangular yoke, a broad pole above and below the gap (its face tilted by the
gap row's angle), a coil block each side of each pole, its winding on the gap
side, the gap a flat ellipse.

Parts that are not the body: a sensor whose row states a coil (an angle, or a
position) is a dot on that coil's block, two on one coil side by side; one with
neither is not placed, a hollow ring in a row of its kind beside the drawing. A water part (a kind
:data:`WATER_WORDS` names) or a terminal block is a small block in a column
beside the yoke (right of it, or left when its row places it left), a word
above; with no position, a dashed block in a row of its kind with the other parts
not placed, on no side.

The format draws polygons, axis-aligned ellipses and axis-aligned rectangles
only: a rounded tip and a rounded block at an angle are polygons whose curves
are sampled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from osprey_panels.tables import PicturePart

__all__ = [
    "BORE",
    "Block",
    "Body",
    "CoilMark",
    "Dot",
    "Shape",
    "Word",
    "body",
    "member_number",
]

Point = tuple[float, float]

# -- proportions, in drawing units ------------------------------------------------
#: The bore's radius. A stated bore sets the unit; else a stated yoke or outline.
BORE = 40.0
#: A yoke ring's cavity (its inner edge) and its outer edge; a polygon outline's
#: cavity, of its nearest side, is ``CAVITY / YOKE``.
CAVITY = 172.0
YOKE = 212.0
#: The least bore radius a pole tip reaches for, as a share of :data:`BORE`: a bore
#: stated smaller still is drawn as stated, the tips stop here.
BORE_LEAST = 0.3
#: A pole: where its root ends (inside the yoke), where its flanks start to narrow,
#: where its rounded tip starts, the tip's control point (the tip stops half-way
#: between, short of the bore), the tip's half width of the pole's.
POLE_ROOT = 176.0
POLE_SHOULDER = 80.0
POLE_TIP_START = 58.0
POLE_TIP_CONTROL = 47.0
POLE_TIP_WIDTH = 0.58
#: A pole's half width, and its coil block's thickness, by pole count; another count
#: by :data:`HALF_WIDTH_SPAN` and :data:`COIL_OF_POLE`.
HALF_WIDTH = {4: 38.0, 6: 26.0}
HALF_WIDTH_SPAN = 52.0  # times the sine of half a pole's share of the circle
COIL_THICKNESS = {4: 30.0, 6: 23.0}
COIL_OF_POLE = 0.88
COIL_MOST = 30.0
#: A coil block: from and to (from the beam), its clearance from the pole, its corners.
COIL_FROM, COIL_TO = 104.0, 166.0
COIL_CLEARANCE = 1.0
COIL_CORNER = 3.0
#: A corrector winding band: from and to, thickness, clearance from the coil, the step
#: to a second band, corners.
WINDING_FROM, WINDING_TO = 110.0, 160.0
WINDING = 6.0
WINDING_CLEARANCE = 3.0
WINDING_STEP = 8.0
WINDING_CORNER = 2.0
#: A sensor dot on a coil block: its radius; where it sits along the block (a second
#: and a third on one block step along it).
SENSOR = 7.0
SENSOR_ALONG = (135.0, 155.0, 115.0)
#: A coil's number: on the yoke ring, this far from the beam.
NUMBER_AT = 194.0
#: The side columns: their centre's distance from the beam axis; a water block, its
#: step; a terminal block, its step, the gap above the first; the word above a column.
SIDE_COLUMN = 277.0
#: A water or terminal part placed by its angle alone stands left of the yoke when
#: the angle's cosine is below minus this, else right (straight up or down: right).
SIDE_TOLERANCE = 1e-9
WATER_BLOCK = (30.0, 16.0)
WATER_STEP = 26.0
TERMINAL_BLOCK = (30.0, 28.0)
TERMINAL_STEP = 34.0
TERMINAL_GAP = 12.0
TERMINAL_UNPLACED_STEP = 40.0
WORD_ABOVE = 22.0
BLOCK_CORNER = 3.0
#: The hollow rings of sensors with no place: the first one's centre, the step
#: across, the least step down, the clear space between two rows of unlike height,
#: how far after the last ring its kind's word starts.
UNPLACED = (-284.0, 222.0)
UNPLACED_ACROSS = 20.0
UNPLACED_DOWN = 26.0
UNPLACED_ROW_GAP = 10.0
UNPLACED_WORD = 4.0
UNPLACED_BLOCK_GAP = 6.0

# two poles: a rectangular yoke and its cavity (half sizes), the pole's sides, the
# gap's half height at the middle of the face, the coil blocks' sides and heights,
# the winding band's height on the gap side, the gap's flat ellipse, the coil number
BEND_YOKE = (200.0, 190.0)
BEND_CAVITY = (160.0, 128.0)
BEND_YOKE_CORNER = 6.0
BEND_POLE = (-70.0, 62.0)
BEND_GAP = 16.5
BEND_COILS = ((-136.0, -76.0), (68.0, 128.0))
BEND_COIL_HEIGHT = (44.0, 114.0)
BEND_COIL_CORNER = 4.0
BEND_WINDING_HEIGHT = (35.0, 42.0)
BEND_WINDING_INSET = 6.0
BEND_BORE = (34.0, 9.0)
BEND_NUMBER = (-178.0, 84.0)

#: The bore radius, in mm, a body that states no size is drawn about.
DEFAULT_BORE_MM = 15.0
#: Points per quarter circle of a rounded corner; of a pole's rounded tip.
ARC_STEPS = 3
TIP_STEPS = 8
#: A coil and a pole this close in angle, in degrees, are one pole's.
SAME_ANGLE = 0.5
TURN = 360.0
#: Kinds drawn as water (a manifold, a cooling line) and as a terminal block.
WATER_WORDS = ("manifold", "water", "cooling")
TERMINAL_WORDS = ("terminal",)


@dataclass(frozen=True)
class Shape:
    """One shape of a body, in mm in the view's plane (y up).

    Attributes:
        row: The row it draws, or ``None`` for a yoke no row states.
        form: ``ellipse`` (``points`` are two opposite corners of its box) or
            ``polygon`` (``points`` are its corners).
        points: As ``form`` says.
        role: What it is painted as: ``yoke``, ``window`` (the yoke's cavity, the
            panel's ground), ``pole``, ``coil``, ``winding`` (a corrector winding),
            ``winding2`` (a second winding stacked outside the first) or ``bore``.
        name: Its tooltip.
        dashed: Its outline is not stated: a yoke drawn at its default.
    """

    row: PicturePart | None
    form: str
    points: tuple[Point, ...]
    role: str
    name: str
    dashed: bool = False


@dataclass(frozen=True)
class CoilMark:
    """A coil's number, centred at *at* (mm): on the yoke ring at its angle."""

    row: PicturePart
    number: str
    at: Point


@dataclass(frozen=True)
class Dot:
    """A sensor: on its coil's block at *at* when *placed*, else a hollow ring in its
    kind's row beside the drawing. *radius* in mm."""

    row: PicturePart
    at: Point
    radius: float
    placed: bool


@dataclass(frozen=True)
class Block:
    """A part beside the yoke: a rounded block centred at *at*, *size* (w, h) in mm.

    Attributes:
        role: ``water`` or ``terminal``.
        placed: Its row places it; else it is dashed.
        label: The text inside it (a terminal block's ``TB``, ``TB1``...), or ``None``.
    """

    row: PicturePart
    at: Point
    size: Point
    role: str
    placed: bool
    label: str | None = None


@dataclass(frozen=True)
class Word:
    """A muted word beside the drawing: centred at *at*, or starting there (*left*)."""

    text: str
    at: Point
    left: bool = False


@dataclass(frozen=True)
class Body:
    """A class's body in one view.

    Attributes:
        shapes: In painting order: yoke and its cavity, poles, coils, windings, bore.
        coils: Each coil's number.
        dots: The sensors.
        blocks: The water and terminal blocks.
        words: The words over the columns and after the rows of hollow rings.
        beam: Where the beam is (a bore's or a gap's centre), or ``None``.
        unit: mm per drawing unit.
        sized: The unit is stated (by a bore, a gap, a yoke or an outline) and every
            shape row states its size.
        scaled_from: The kind of the row that states the unit (``bore``, ``gap``,
            ``yoke``, ``outline``), or ``None``: the default proportions.
        schematic: The kinds of shape rows (other than those) that state no size,
            drawn at the default proportion, in file order.
    """

    shapes: tuple[Shape, ...]
    coils: tuple[CoilMark, ...] = ()
    dots: tuple[Dot, ...] = ()
    blocks: tuple[Block, ...] = ()
    words: tuple[Word, ...] = ()
    beam: Point | None = None
    unit: float = 1.0
    sized: bool = True
    scaled_from: str | None = None
    schematic: tuple[str, ...] = ()


# -- primitives --------------------------------------------------------------------


def _turned(turn: float, points: Sequence[Point]) -> tuple[Point, ...]:
    """*points*, given as (along, across) a direction *turn* radians from x."""
    ux, uy = math.cos(turn), math.sin(turn)
    return tuple((u * ux - v * uy, u * uy + v * ux) for u, v in points)


def _arc(cu: float, cv: float, r: float, start: float) -> list[Point]:
    """A quarter circle about (*cu*, *cv*) from *start* degrees."""
    out = []
    for k in range(ARC_STEPS + 1):
        t = math.radians(start + 90 * k / ARC_STEPS)
        out.append((cu + r * math.cos(t), cv + r * math.sin(t)))
    return out


def _rounded(u1: float, u2: float, v1: float, v2: float, corner: float) -> list[Point]:
    """A rectangle from *u1* to *u2* along and *v1* to *v2* across, corners rounded."""
    r = max(0.0, min(corner, (u2 - u1) / 2, (v2 - v1) / 2))
    points: list[Point] = []
    for cu, cv, start in ((u2 - r, v2 - r, 0), (u1 + r, v2 - r, 90), (u1 + r, v1 + r, 180)):
        points += _arc(cu, cv, r, start)
    return points + _arc(u2 - r, v1 + r, r, 270)


def _pole(
    nose: float, root: float, half: float, shoulder: float, start: float, tip: float | None = None
) -> list[Point]:
    """A pole along +u: broad from *root* to *shoulder*, narrowing to its tip at
    *start*, rounded to *nose* (all from the beam), *half* its half width, *tip* the
    tip's (default :data:`POLE_TIP_WIDTH` of *half*)."""
    tip = POLE_TIP_WIDTH * half if tip is None else tip
    control = 2 * nose - start  # a quadratic curve's middle lies half-way to its control
    curve = []
    for k in range(TIP_STEPS + 1):
        t = k / TIP_STEPS
        u = (1 - t) ** 2 * start + 2 * t * (1 - t) * control + t**2 * start
        v = (1 - t) ** 2 * -tip + t**2 * tip
        curve.append((u, v))
    return [(root, -half), (shoulder, -half), *curve, (shoulder, half), (root, half)]


def _box(a: float, b: float, w: float, h: float) -> tuple[Point, Point]:
    return (a - w / 2, b - h / 2), (a + w / 2, b + h / 2)


# -- what the rows state -----------------------------------------------------------


def _xy(row: PicturePart) -> Point:
    """A row's coordinates in its view's plane: x, y on an end view; z, y on a side view;
    the beam axis where the row states none."""
    across = row.z_mm if row.view == "side" else row.x_mm
    return (across or 0.0, row.y_mm or 0.0)


def _wh(row: PicturePart) -> Point:
    w = row.w_mm or 0.0
    return w, row.h_mm if row.h_mm is not None else w


def _angle(row: PicturePart) -> float:
    """A row's angle in degrees: its own, else its position's about the beam."""
    if row.angle_deg is not None:
        return row.angle_deg
    a, b = _xy(row)
    return math.degrees(math.atan2(b, a))


def _apart(a: float, b: float) -> float:
    d = (a - b) % TURN
    return min(d, TURN - d)


def _nearest_side(outline: Sequence[Point]) -> float:
    """How near the beam the polygon *outline* comes: its nearest side."""

    def to_segment(p: Point, q: Point) -> float:
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else max(0.0, min(1.0, -(p[0] * dx + p[1] * dy) / length))
        return math.hypot(p[0] + t * dx, p[1] + t * dy)

    return min(to_segment(outline[i - 1], outline[i]) for i in range(len(outline)))


def _convex(points: Sequence[Point]) -> bool:
    """Whether a polygon turns one way only at every corner."""
    turns = set()
    for i in range(len(points)):
        (ax, ay), (bx, by), (cx, cy) = points[i - 2], points[i - 1], points[i]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross:
            turns.add(cross > 0)
    return len(turns) <= 1


def _kind(row: PicturePart) -> str:
    """A part's place: ``water`` and ``terminal`` beside the yoke, ``sensor`` on it."""
    if any(word in row.kind for word in WATER_WORDS):
        return "water"
    if any(word in row.kind for word in TERMINAL_WORDS):
        return "terminal"
    return "sensor"


#: The kinds of shape row that frame a drawing: one of them states its unit.
_FRAMES = frozenset({"yoke", "bore", "gap"})


def _unit(
    rows: Sequence[PicturePart], ring: Sequence[Point], two: bool
) -> tuple[float, str | None]:
    """mm per drawing unit, and the kind of the row that states it (``None``: none
    does, the default bore's). Two poles: a sized gap's,
    else a sized yoke's, else an outline's. More: a sized yoke's, else an
    outline's, else a sized bore's; the frame sets the unit, so the coils and the
    parts keep their proportion to the yoke however small the bore is, and the pole
    tips reach for the bore as stated. Else the default bore's."""
    if two or not any(r.kind == "yoke" and r.sized for r in rows) and not ring:
        for row in rows:
            if row.kind in ("bore", "gap") and row.sized:
                w, h = _wh(row)
                return (min(w, h) / (2 * BEND_GAP) if two else min(w, h) / 2 / BORE), row.kind
    for row in rows:
        if row.kind == "yoke" and row.sized:
            w, h = _wh(row)
            if two:
                return min(w / 2 / BEND_YOKE[0], h / 2 / BEND_YOKE[1]), "yoke"
            return min(w, h) / 2 / YOKE, "yoke"
    if ring:
        near = _nearest_side(ring)
        if two:
            return near / (BEND_YOKE[1] if _convex(ring) else BEND_CAVITY[1]), "outline"
        return near / (YOKE if _convex(ring) else CAVITY), "outline"
    return DEFAULT_BORE_MM / (BEND_GAP if two else BORE), None


# -- the body ----------------------------------------------------------------------


def body(
    rows: Sequence[PicturePart],
    outline: Sequence[PicturePart] = (),
    parts: Sequence[PicturePart] = (),
) -> Body:
    """The body that *rows* (the shape rows of one view), *outline* (its outline rows,
    in file order) and *parts* (its other rows) state, unsized rows at the default
    proportions."""
    poles = [r for r in rows if r.kind == "pole"]
    coils = [r for r in rows if r.kind == "coil"]
    count = len(poles) or len(coils)
    two = count == 2 or (count < 3 and any(r.kind == "gap" for r in rows))
    ring = [_xy(row) for row in outline]
    unit, scaled_from = _unit(rows, ring, two)
    schematic = tuple(dict.fromkeys(r.kind for r in rows if r.kind not in _FRAMES and not r.sized))
    sized = scaled_from is not None and not schematic
    draw = _two_poles if two else _radial
    shapes, marks, seats = draw(rows, outline, ring, unit, max(count, 3))
    steel = [pt for sh in shapes if sh.role == "yoke" for pt in sh.points]
    reach = (
        max((abs(a) for a, _b in steel), default=0.0),
        max((abs(b) for _a, b in steel), default=0.0),
    )
    dots, blocks, words = _parts(parts, seats, unit, reach)
    beam = next((_xy(r) for r in rows if r.kind in ("bore", "gap")), None)
    if beam is None and (poles or coils):
        beam = (0.0, 0.0)
    return Body(
        tuple(shapes),
        tuple(marks),
        dots,
        blocks,
        words,
        beam,
        unit,
        sized,
        scaled_from,
        schematic,
    )


#: Where a sensor may sit: (angle in degrees, its block's centre line as two points
#: (along the block, mm), the side it is on). A sensor goes to the nearest angle.
Seat = tuple[float, Point, Point]


def _yoke(
    rows: Sequence[PicturePart],
    outline: Sequence[PicturePart],
    ring: Sequence[Point],
    unit: float,
    two: bool,
) -> tuple[list[Shape], float]:
    """The yoke and its cavity, and how far the cavity is from the beam (mm)."""
    yoke = next((r for r in rows if r.kind == "yoke"), None)
    name = "yoke"
    if ring:
        shapes = [Shape(outline[0], "polygon", tuple(ring), "yoke", name)]
        if not _convex(ring):  # the outline is the steel, its cavity included
            return shapes, _nearest_side(ring)
        near = _nearest_side(ring)
        if two:
            ratio = BEND_CAVITY[1] / BEND_YOKE[1]
            cavity = tuple((a * ratio, b * ratio) for a, b in ring)
            shapes.append(Shape(outline[0], "polygon", cavity, "window", name))
            return shapes, near * ratio
        cavity_r = near * CAVITY / YOKE
        shapes.append(Shape(outline[0], "ellipse", _box(0, 0, 2 * cavity_r, 2 * cavity_r), "window", name))  # fmt: skip
        return shapes, cavity_r
    stated = yoke is not None and yoke.sized
    a, b = _xy(yoke) if yoke is not None else (0.0, 0.0)
    if two:
        w, h = _wh(yoke) if yoke is not None and stated else (2 * BEND_YOKE[0] * unit, 2 * BEND_YOKE[1] * unit)  # fmt: skip
        cw, ch = w * BEND_CAVITY[0] / BEND_YOKE[0], h * BEND_CAVITY[1] / BEND_YOKE[1]
        outer = _rounded(a - w / 2, a + w / 2, b - h / 2, b + h / 2, BEND_YOKE_CORNER * unit)
        inner = _rounded(a - cw / 2, a + cw / 2, b - ch / 2, b + ch / 2, BEND_YOKE_CORNER * unit)
        return [
            Shape(yoke, "polygon", tuple(outer), "yoke", name, dashed=not stated),
            Shape(yoke, "polygon", tuple(inner), "window", name),
        ], ch / 2
    w, h = _wh(yoke) if yoke is not None and stated else (2 * YOKE * unit, 2 * YOKE * unit)
    cw, ch = w * CAVITY / YOKE, h * CAVITY / YOKE
    return [
        Shape(yoke, "ellipse", _box(a, b, w, h), "yoke", name, dashed=not stated),
        Shape(yoke, "ellipse", _box(a, b, cw, ch), "window", name),
    ], min(cw, ch) / 2


def _windings(rows: Sequence[PicturePart]) -> dict[int, int]:
    """Each corrector winding row's place in the stack at its angle: 0 the first."""
    stack: dict[int, int] = {}
    for i, row in enumerate(rows):
        if row.kind != "corrector_winding":
            continue
        below = [
            j for j in stack if _apart(rows[j].angle_deg or 0.0, row.angle_deg or 0.0) <= SAME_ANGLE
        ]
        stack[i] = len(below)
    return stack


def _radial(
    rows: Sequence[PicturePart],
    outline: Sequence[PicturePart],
    ring: Sequence[Point],
    unit: float,
    count: int,
) -> tuple[list[Shape], list[CoilMark], list[Seat]]:
    """A magnet of more than two poles."""
    shapes, cavity = _yoke(rows, outline, ring, unit, two=False)
    half = HALF_WIDTH.get(count, HALF_WIDTH_SPAN * math.sin(math.pi / count)) * unit
    thick = COIL_THICKNESS.get(count, min(COIL_MOST, COIL_OF_POLE * half / unit)) * unit
    root = cavity + (POLE_ROOT - CAVITY) * unit
    mid = (POLE_TIP_START + POLE_TIP_CONTROL) / 2
    inward = _bore_ratio(rows, unit)  # a bore stated small for its yoke: tips reach in
    for row in (r for r in rows if r.kind == "pole"):
        turn = math.radians(row.angle_deg or 0.0)
        tip = None
        if row.sized:
            w, h = _wh(row)
            a, b = _xy(row)
            centre = a * math.cos(turn) + b * math.sin(turn)
            k = h / (POLE_ROOT - mid)
            nose, end, side = centre - h / 2, centre + h / 2, w / 2
            shoulder = nose + (POLE_SHOULDER - mid) * k
        else:
            k, nose, end, side = inward * unit, mid * inward * unit, root, half
            shoulder = POLE_SHOULDER * unit  # the flanks as drawn; the tip tapers in
            tip = POLE_TIP_WIDTH * half * inward
        start = nose + (POLE_TIP_START - POLE_TIP_CONTROL) / 2 * k
        points = _turned(turn, _pole(nose, end, side, shoulder, start, tip))
        shapes.append(Shape(row, "polygon", points, "pole", row.part))
    # coils, then the windings outside them: each a block on both flanks
    reach = cavity - CAVITY * unit  # how far a stated yoke moves the blocks out
    marks: list[CoilMark] = []
    seats: list[Seat] = []
    block: dict[float, tuple[float, float, float]] = {}  # angle: from, to, outer edge
    for row in (r for r in rows if r.kind == "coil"):
        angle = row.angle_deg or 0.0
        turn = math.radians(angle)
        u1, u2 = (COIL_FROM * unit + reach, COIL_TO * unit + reach)
        t = thick
        if row.sized:
            t, h = _wh(row)
            a, b = _xy(row)
            mid = a * math.cos(turn) + b * math.sin(turn) or (u1 + u2) / 2
            u1, u2 = mid - h / 2, mid + h / 2
        v1 = half + COIL_CLEARANCE * unit
        for sign in (1, -1):
            box = [(u, sign * v) for u, v in _rounded(u1, u2, v1, v1 + t, COIL_CORNER * unit)]
            shapes.append(Shape(row, "polygon", _turned(turn, box), "coil", row.part))
        block[angle] = (u1, u2, v1 + t)
        across = v1 + t / 2
        near, far = _turned(turn, [(u1, across), (u2, across)])
        seats.append((angle, near, far))
        number = member_number("coil", row.part, len(marks) + 1)
        at = cavity + (NUMBER_AT - CAVITY) * unit
        marks.append(CoilMark(row, number, (at * math.cos(turn), at * math.sin(turn))))
    stack = _windings(rows)
    for i, row in enumerate(rows):
        if i not in stack:
            continue
        angle = row.angle_deg or 0.0
        turn = math.radians(angle)
        mine = next((v for a, v in block.items() if _apart(a, angle) <= SAME_ANGLE), None)
        coil_from, _to, edge = mine or (COIL_FROM * unit, 0.0, half + thick + COIL_CLEARANCE * unit)  # fmt: skip
        u1 = coil_from + (WINDING_FROM - COIL_FROM) * unit
        u2 = u1 + (WINDING_TO - WINDING_FROM) * unit
        t = WINDING * unit
        if row.sized:
            t, h = _wh(row)
            u2 = u1 + h
        v1 = edge + (WINDING_CLEARANCE - COIL_CLEARANCE) * unit + stack[i] * WINDING_STEP * unit
        role = "winding" if stack[i] == 0 else "winding2"
        for sign in (1, -1):
            band = [(u, sign * v) for u, v in _rounded(u1, u2, v1, v1 + t, WINDING_CORNER * unit)]
            shapes.append(Shape(row, "polygon", _turned(turn, band), role, row.part))
    for row in (r for r in rows if r.kind == "bore"):
        a, b = _xy(row)
        w, h = _wh(row) if row.sized else (2 * BORE * unit, 2 * BORE * unit)
        shapes.append(Shape(row, "ellipse", _box(a, b, w, h), "bore", row.part))
    if not any(r.kind == "bore" for r in rows) and any(r.kind in ("pole", "coil") for r in rows):
        shapes.append(Shape(None, "ellipse", _box(0, 0, 2 * BORE * unit, 2 * BORE * unit), "bore", "bore"))  # fmt: skip
    return shapes, marks, seats


def _bore_ratio(rows: Sequence[PicturePart], unit: float) -> float:
    """A stated bore's radius over the default's, in this drawing's units: below 1
    when the frame (a stated yoke) sets the unit and the bore is small for it; 1 when
    no bore is stated or the bore sets the unit. At most 1: a pole is never drawn
    shorter than the default proportion."""
    bore = next((r for r in rows if r.kind == "bore" and r.sized), None)
    if bore is None:
        return 1.0
    w, h = _wh(bore)
    return max(BORE_LEAST, min(1.0, min(w, h) / 2 / unit / BORE))


def _two_poles(
    rows: Sequence[PicturePart],
    outline: Sequence[PicturePart],
    ring: Sequence[Point],
    unit: float,
    _count: int,
) -> tuple[list[Shape], list[CoilMark], list[Seat]]:
    """A magnet of two poles, drawn for the first pole's angle (in quarter turns)."""
    poles = [r for r in rows if r.kind == "pole"]
    first = next(iter(poles or [r for r in rows if r.kind == "coil"]), None)
    angle = first.angle_deg if first is not None and first.angle_deg is not None else 90.0
    quarter = round((angle - 90) / 90)
    turn = math.radians(90 * quarter)
    shapes, cavity = _yoke(rows, outline, ring, unit, two=True)  # a stated outline: as stated
    gap = next((r for r in rows if r.kind in ("gap", "bore")), None)
    tilt = math.tan(math.radians(gap.angle_deg or 0.0)) if gap is not None else 0.0
    half_gap = BEND_GAP * unit
    if gap is not None and gap.sized:
        half_gap = min(_wh(gap)) / 2
    p1, p2 = (x * unit for x in BEND_POLE)
    mid = (p1 + p2) / 2

    def face(x: float) -> float:
        return half_gap - tilt * (x - mid)

    def turned(points: Sequence[Point]) -> tuple[Point, ...]:
        return _turned(turn, points)

    for row in poles[:2]:
        sign = 1 if math.sin(math.radians((row.angle_deg or 90.0) - 90 * quarter)) >= 0 else -1
        points = [(p1, sign * cavity), (p1, sign * face(p1)), (p2, sign * face(p2)), (p2, sign * cavity)]  # fmt: skip
        shapes.append(Shape(row, "polygon", turned(points), "pole", row.part))
    marks: list[CoilMark] = []
    seats: list[Seat] = []
    y1, y2 = (y * unit for y in BEND_COIL_HEIGHT)
    for row in (r for r in rows if r.kind == "coil"):
        local = (row.angle_deg if row.angle_deg is not None else 90.0) - 90 * quarter
        sign = 1 if math.sin(math.radians(local)) >= 0 else -1
        for x1, x2 in BEND_COILS:
            box = _rounded(x1 * unit, x2 * unit, y1, y2, BEND_COIL_CORNER * unit)
            points = [(a, sign * b) for a, b in box]
            shapes.append(Shape(row, "polygon", turned(points), "coil", row.part))
            middle = (y1 + y2) / 2 * sign
            ((a1, b1), (a2, b2)) = turned([(x1 * unit, middle), (x2 * unit, middle)])
            seat_angle = math.degrees(math.atan2(b1 + b2, a1 + a2))
            seats.append((seat_angle, (a1, b1), (a2, b2)))
        number = member_number("coil", row.part, len(marks) + 1)
        nx, ny = BEND_NUMBER
        ((ax, ay),) = turned([(nx * unit, sign * ny * unit)])
        marks.append(CoilMark(row, number, (ax, ay)))
    w1, w2 = (y * unit for y in BEND_WINDING_HEIGHT)
    for row in (r for r in rows if r.kind == "corrector_winding"):
        local = (row.angle_deg if row.angle_deg is not None else 90.0) - 90 * quarter
        sign = 1 if math.sin(math.radians(local)) >= 0 else -1
        for x1, x2 in BEND_COILS:
            inset = BEND_WINDING_INSET * unit
            box = _rounded(x1 * unit + inset, x2 * unit - inset, w1, w2, WINDING_CORNER * unit)
            points = [(a, sign * b) for a, b in box]
            shapes.append(Shape(row, "polygon", turned(points), "winding", row.part))
    rx, ry = (r * unit for r in BEND_BORE)
    if quarter % 2:
        rx, ry = ry, rx
    if gap is not None:
        a, b = _xy(gap)
        shapes.append(Shape(gap, "ellipse", _box(a, b, 2 * rx, 2 * ry), "bore", gap.part))
    else:
        shapes.append(Shape(None, "ellipse", _box(0, 0, 2 * rx, 2 * ry), "bore", "gap"))
    return shapes, marks, seats


# -- the parts ---------------------------------------------------------------------


def _parts(
    parts: Sequence[PicturePart], seats: Sequence[Seat], unit: float, reach: Point
) -> tuple[tuple[Dot, ...], tuple[Block, ...], tuple[Word, ...]]:
    """Where each part goes: a sensor on its coil (or in a row of hollow rings), a
    water or terminal part in a column beside the yoke; *reach* is how far the yoke
    reaches across and up (mm): a stated yoke larger than its default moves them out."""
    dots: list[Dot] = []
    words: list[Word] = []
    taken: dict[int, int] = {}
    unplaced: dict[str, list[PicturePart]] = {}
    sides: dict[float, dict[str, list[tuple[PicturePart, bool]]]] = {}
    for row in parts:
        kind = _kind(row)
        if kind == "sensor":
            if seats and row.placed:
                angle = _angle(row)
                n = min(range(len(seats)), key=lambda i: _apart(seats[i][0], angle))
                k = taken.get(n, 0)
                taken[n] = k + 1
                _a, (a1, b1), (a2, b2) = seats[n]
                share = (SENSOR_ALONG[k % len(SENSOR_ALONG)] - COIL_FROM) / (COIL_TO - COIL_FROM)
                at = (a1 + share * (a2 - a1), b1 + share * (b2 - b1))
                dots.append(Dot(row, at, SENSOR * unit, True))
            elif row.positioned:  # no coil to sit on: where its row puts it
                dots.append(Dot(row, _xy(row), SENSOR * unit, True))
            else:
                unplaced.setdefault(row.kind, []).append(row)
            continue
        if not row.placed:  # no side stated: with the other parts not placed
            unplaced.setdefault(row.kind, []).append(row)
            continue
        across = _xy(row)[0] if row.positioned else math.cos(math.radians(_angle(row)))
        side = -1.0 if across < -SIDE_TOLERANCE else 1.0
        sides.setdefault(side, {}).setdefault(kind, []).append((row, True))
    x0 = min(UNPLACED[0] * unit, -reach[0] + (UNPLACED[0] + YOKE) * unit)
    y0 = max(UNPLACED[1] * unit, reach[1] + (UNPLACED[1] - YOKE) * unit)
    blocks: list[Block] = []
    rows_at = _unplaced_rows([_kind(m[0]) for m in unplaced.values()])
    for r, (kind, members) in enumerate(unplaced.items()):
        y = y0 + rows_at[r] * unit  # the last row lowest
        role = _kind(members[0])
        if role == "sensor":
            for i, row in enumerate(members):
                dots.append(Dot(row, (x0 + i * UNPLACED_ACROSS * unit, y), SENSOR * unit, False))
            x = x0 + ((len(members) - 1) * UNPLACED_ACROSS + SENSOR + UNPLACED_WORD) * unit
        else:  # a block each, dashed, in a row: no column, so no side
            step = WATER_BLOCK[0] + UNPLACED_BLOCK_GAP
            left = x0 - SENSOR * unit + WATER_BLOCK[0] / 2 * unit
            size = TERMINAL_BLOCK if role == "terminal" else WATER_BLOCK
            for i, row in enumerate(members):
                label = (f"TB{i + 1}" if len(members) > 1 else "TB") if role == "terminal" else None
                at = (left + i * step * unit, y)
                blocks.append(Block(row, at, _scaled(size, unit), role, False, label))
            x = left + ((len(members) - 1) * step + WATER_BLOCK[0] / 2 + UNPLACED_WORD) * unit
        words.append(Word(kind.replace("_", " "), (x, y), left=True))
    for side, kinds in sides.items():
        x = side * max(SIDE_COLUMN * unit, reach[0] + (SIDE_COLUMN - YOKE) * unit)
        water = kinds.get("water", [])
        terminals = kinds.get("terminal", [])
        n = len(water)
        top = (n - 1) * WATER_STEP / 2
        for i, (row, placed) in enumerate(water):
            at = (x, (top - i * WATER_STEP) * unit)
            blocks.append(Block(row, at, _scaled(WATER_BLOCK, unit), "water", placed))
        if water:
            words.append(Word("water", (x, (top + WORD_ABOVE) * unit)))
        for i, (row, placed) in enumerate(terminals):
            label = f"TB{i + 1}" if len(terminals) > 1 else "TB"
            if water:  # under the water column
                y = top - n * WATER_STEP - TERMINAL_GAP - i * TERMINAL_STEP - TERMINAL_BLOCK[1] / 2
            else:
                y = -(i - (len(terminals) - 1) / 2) * TERMINAL_UNPLACED_STEP
            at = (x, y * unit)
            blocks.append(Block(row, at, _scaled(TERMINAL_BLOCK, unit), "terminal", placed, label))
    return tuple(dots), tuple(blocks), tuple(words)


def _unplaced_rows(roles: Sequence[str]) -> list[float]:
    """How far above the lowest row each row of parts with no place sits (drawing
    units, in the order given, the last row lowest): rows step by at least
    :data:`UNPLACED_DOWN`, and a taller row (a terminal block) keeps clear of its
    neighbours."""
    heights = [_unplaced_height(role) for role in roles]
    at = [0.0] * len(roles)
    for r in range(len(roles) - 2, -1, -1):
        pitch = (heights[r] + heights[r + 1]) / 2 + UNPLACED_ROW_GAP
        at[r] = at[r + 1] + max(UNPLACED_DOWN, pitch)
    return at


def _unplaced_height(role: str) -> float:
    if role == "sensor":
        return 2 * SENSOR
    return TERMINAL_BLOCK[1] if role == "terminal" else WATER_BLOCK[1]


def _scaled(size: Point, unit: float) -> Point:
    return size[0] * unit, size[1] * unit


def member_number(word: str, name: str, ordinal: int) -> str:
    """A member's number within its kind: what follows the kind's *word* in *name*
    (``thermal switch 3`` under thermal switch: ``3``), else its *ordinal*."""
    rest = name.removeprefix(word).strip()
    return rest if name.startswith(word + " ") and rest else str(ordinal)
