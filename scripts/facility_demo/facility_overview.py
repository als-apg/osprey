#!/usr/bin/env python3
"""Generate the control-assistant facility overview, booster and sector screens from the records.

The ``write`` mode renders ``facility/overview``, ``br/overview``, the storage-ring
sectors ``sr/sector-NN`` and their vacuum and absorber sub-screens from the example
facility's channel records (and the device records beside them, which place each
peripheral in its sector) into a CANOPY screen library; ``check`` renders them in
memory and exits 1 when a committed file differs. Every bound channel comes from the
records, sorted by device number, and the plan draws as many symbols and dots as the
records hold. A missing record, a count the drawing cannot hold, or a section with no
bound status stops the run with a named error and writes nothing.

Usage::

    uv run --no-sync python scripts/facility_demo/facility_overview.py write --library <dir>
    uv run --no-sync python scripts/facility_demo/facility_overview.py check --library <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_RECORDS = (
    REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "facilities"
    / "example"
    / "records"
    / "channels.yaml"
)

# The device records, beside the channel records: each device's place and label.
DEVICES_FILE = "devices.yaml"

# The workspace target of every section area. The pinned CANOPY viewer appends a
# workspace tab for an ``open_display`` action whose target is ``tab`` and keeps the
# declaring screen open in its own tab.
NAVIGATION_TARGET = "tab"

# How a rule condition or formula receives a labelled channel (an enum or a bool flag)
# under the pinned CANOPY: as its option index, never its label. A flag that is on
# reads 1, and an enum reads the position of its option in the record's list.
LABELLED_INPUT_FORM = "index"

OVERVIEW_ID = "facility/overview"
BOOSTER_ID = "br/overview"
GTL_ID = "gtl/GTL"
BTS_ID = "bts/BTS"

# Stored beam above this current (mA) is beam; at or below it the ring has no beam.
STORED_BEAM_MA = 1.0
USER_OPERATION = "User operation"
# The storage ring's beam energy, as its deck sets it.
ENERGY_TEXT = "2.0 GeV"

# The longest formula text the viewer may send for one rule.
FORMULA_LIMIT = 2048

SR_CELLS = 12
# The storage-ring sector screens, one per cell: ``sr/sector-NN``. The records number
# a family's devices ring-wide (QF 01..24, HCM 01..72), so a sector's channel names
# are not its number substituted into one parameterised screen.
SECTOR_PREFIX = "sr/sector-"
# The straights that hold devices of the records, and the part of a sector screen's
# strip that is straight.
INJECTION_STRAIGHT = 1
RF_STRAIGHT = 2
STRAIGHT_FRACTION = 0.22
BR_CELLS = 4
SECTOR_PITCH = 27.0

# Lattice families the plan draws, by symbol kind, in the order they fill a cell.
SYMBOL_KIND = {
    "DIPOLE": "dipole",
    "QF": "quadrupole",
    "QD": "quadrupole",
    "QFA": "quadrupole",
    "SF": "sextupole",
    "SD": "sextupole",
    "SHF": "sextupole",
    "SHD": "sextupole",
}
# The storage-ring supply families of the magnet matrix, top to bottom.
MATRIX_FAMILIES = ("DIPOLE", "QF", "QD", "QFA", "SF", "SD", "SHF", "SHD")
FAMILY_LABEL = {"DIPOLE": "Dipole"}
# The corrector planes counted as on/total.
COUNTED_FAMILIES = ("HCM", "VCM")

# The clickable plan areas above the storage-ring sectors, bottom to top: id, the
# screen name its hover label offers, and the screen it opens. GTL is one line, gun to
# booster, and one area; no two areas share a point.
AREAS = (
    ("BTS", "BTS", BTS_ID),
    ("GTL", "GTL", GTL_ID),
    ("BR", "Booster", BOOSTER_ID),
)
# The prop that carries an area's hover text.
HOVER_LABEL = "hoverLabel"
# The prop that carries a lattice element's device name for a hover tooltip. The
# pinned runtime does not show it yet (it hovers only clickable nodes); the name is
# in the screen as data so a runtime that reads it needs no regeneration.
TOOLTIP = "tooltip"

# Text sizes of the overview, px at authored size.
TEXT_SECTION = 18
TEXT = 16
TEXT_SMALL = 14


def rounded(expression: str, decimals: int) -> str:
    """``expression`` rounded half up to ``decimals`` places, in the formula language.

    The pinned runtime shows a formula's value as it comes, with every digit, and the
    language has ``floor`` but no ``round``. A value n / 10**k reads with at most k
    decimals, and a value that rounds to zero reads ``0``, never ``-0``.
    """
    scale = 10**decimals
    return f"floor(({expression}) * {scale} + 0.5) / {scale}"


def text_width(text: str, size: float) -> float:
    """A box wide enough for ``text`` at ``size`` px in the screen's sans face."""
    return math.ceil(len(text) * size * 0.6) + 6


# A box on the canvas: x, y, width, height.
Box = tuple[float, float, float, float]


def _role(name: str, light: str, dark: str) -> dict[str, str]:
    return {"$role": name, "fallback": light, "fallbackDark": dark}


# OSPREY's own roles: the light and dark tokens of the OSPREY themes.
PAGE_BACKGROUND = _role("Page_Background", "#f4f5f5", "#111217")
PAGE_TEXT = _role("Page_Text", "#24292e", "#ccccdc")
FIELD_BACKGROUND = _role("Field_Background", "#fafbfc", "#181b1f")
BOX_BORDER = _role("Box_Border", "#d8d9da", "#2c3235")
LED_ON = _role("Led_On", "#56a64b", "#73bf69")
LED_OFF = _role("Led_Off", "#ecedef", "#22252b")
BEAM = _role("Beam", "#1f62c4", "#6e9fff")
DIPOLE = _role("Dipole", "#e08e00", "#f5b73d")
QUADRUPOLE = _role("Quadrupole", "#1f62c4", "#6e9fff")
SEXTUPOLE = _role("Sextupole", "#7c4dff", "#b59cff")
KIND_ROLE = {"dipole": DIPOLE, "quadrupole": QUADRUPOLE, "sextupole": SEXTUPOLE}

# Scheme roles, used only where a colour carries alarm meaning.
ALARM_OK = _role("OK", "#56a64b", "#73bf69")
ALARM_MAJOR = _role("MAJOR", "#c4162a", "#e05454")
ALARM_DISCONNECTED = _role("DISCONNECTED", "#86878f", "#6e6e77")


class GeneratorError(Exception):
    """The records cannot be drawn as asked; nothing is written."""


class MissingRecordError(GeneratorError):
    """A channel the screens bind is not in the records."""


class GeometryMismatchError(GeneratorError):
    """A record count does not fit the drawing that shows it."""


class MissingScreenError(GeneratorError):
    """A section area would open a screen the library does not hold."""


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Records:
    """The channel records by name."""

    by_id: dict[str, dict[str, Any]]
    device_records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def place(self, device: str) -> str | None:
        """The ``place`` the records give ``device``; ``None`` when it states none."""
        return self.device_records.get(device, {}).get("place")

    def need(self, name: str) -> str:
        if name not in self.by_id:
            raise MissingRecordError(f"no record {name}")
        return name

    def devices(self, prefix: str, suffix: str) -> list[str]:
        """Every ``<prefix>:<NN>:<suffix>`` record, sorted by device number."""
        pattern = re.compile(rf"^{re.escape(prefix)}:(\d+):{re.escape(suffix)}$")
        found = [(int(m.group(1)), name) for name in self.by_id if (m := pattern.match(name))]
        return [name for _, name in sorted(found)]

    def device_numbers(self, prefix: str) -> list[int]:
        """The distinct device numbers of every ``<prefix>:<NN>:...`` record."""
        pattern = re.compile(rf"^{re.escape(prefix)}:(\d+):")
        return sorted({int(m.group(1)) for name in self.by_id if (m := pattern.match(name))})

    def device_name(self, prefix: str, number: int) -> str:
        """The records' device name (``on.device``) of ``<prefix>:<NN>:...``."""
        head = f"{prefix}:{number:02d}:"
        for name in sorted(self.by_id):
            if name.startswith(head):
                device = (self.by_id[name].get("on") or {}).get("device")
                if device:
                    return device
        raise MissingRecordError(f"no device name for {prefix}:{number:02d}")

    def option_index(self, name: str, option: str) -> int:
        options = self.by_id[self.need(name)].get("options") or []
        if option not in options:
            raise MissingRecordError(f"{name} has no option {option!r}")
        return options.index(option)


def _safe_load(text: str) -> Any:
    """``yaml.safe_load`` of *text*, on libyaml's scanner when it is installed.

    ``yaml.safe_load`` never reaches libyaml; the two loaders share the
    constructor and resolver, so the records parse to the same objects.
    """
    return yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def load_records(path: Path) -> Records:
    """The channel records at ``path`` and the device records beside them."""
    rows = _safe_load(path.read_text(encoding="utf-8"))
    devices_path = path.with_name(DEVICES_FILE)
    devices = _safe_load(devices_path.read_text(encoding="utf-8")) if devices_path.is_file() else []
    return Records({row["id"]: row for row in rows}, {row["id"]: row for row in devices})


def _need_all(records: Records, names: Iterable[str]) -> list[str]:
    return [records.need(name) for name in names]


# --------------------------------------------------------------------------- labelled inputs


def labelled_is(token: str, records: Records, channel: str, option: str) -> str:
    """A condition over ``{token}`` that is true while ``channel`` reads ``option``."""
    if LABELLED_INPUT_FORM == "index":
        return f"{token} == {records.option_index(channel, option)}"
    return f"{token} == {json.dumps(option)}"


def flag_on(token: str) -> str:
    """A condition over ``{token}`` that is true while a flag reads on."""
    if LABELLED_INPUT_FORM == "index":
        return f"{token} == 1"
    raise GeneratorError(f"no flag condition for the labelled-input form {LABELLED_INPUT_FORM!r}")


# --------------------------------------------------------------------------- screen builder


def _n(value: float) -> float | int:
    """A coordinate rounded to a tenth, integral values as int."""
    rounded = round(value, 1)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


@dataclass
class Screen:
    """One Screen-IR document under construction, ids in build order."""

    screen_id: str
    title: str
    description: str
    width: int
    height: int
    nodes: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[dict[str, Any]] = field(default_factory=list)
    _widgets: int = 1
    _rules: int = 0

    def add(
        self,
        kind: str,
        x: float,
        y: float,
        w: float,
        h: float,
        props: dict[str, Any],
        *,
        parent: dict[str, Any] | None = None,
        rules: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._widgets += 1
        node: dict[str, Any] = {
            "id": f"w-{self._widgets:03d}",
            "type": kind,
            "props": props,
            "geometry": {"x": _n(x), "y": _n(y), "width": _n(w), "height": _n(h)},
        }
        if rules:
            node["rules"] = rules
        if parent is None:
            self.nodes.append(node)
        else:
            parent.setdefault("children", []).append(node)
        return node

    def rule(self, name: str, target: str, **arm: Any) -> dict[str, Any]:
        self._rules += 1
        return {
            "id": f"r-{self._rules:03d}",
            "name": name,
            "targetProperty": target,
            **arm,
        }

    def formula(self, name: str, expression: str, bindings: dict[str, str]) -> str:
        if len(expression) > FORMULA_LIMIT:
            raise GeometryMismatchError(
                f"formula {name} is {len(expression)} characters, over {FORMULA_LIMIT}"
            )
        self.formulas.append(
            {
                "name": name,
                "kind": "scalar",
                "expression": expression,
                "bindings": bindings,
            }
        )
        return f"formula://{name}"

    # -- widgets --------------------------------------------------------------

    def label(
        self,
        x,
        y,
        w,
        h,
        text,
        *,
        size=12,
        align="left",
        bg=PAGE_BACKGROUND,
        parent=None,
        rules=None,
        fg=PAGE_TEXT,
    ):
        props = {
            "text": text,
            "fontSize": size,
            "align": align,
            "foregroundColor": fg,
            "backgroundColor": bg,
        }
        return self.add("text-label", x, y, w, h, props, parent=parent, rules=rules)

    def field(
        self, x, y, w, pv, *, units=True, precision=2, parent=None, h=22, size=None, tooltip=None
    ):
        props = {
            "pv": pv,
            "showUnits": units,
            "alarmSensitive": True,
            "alarmBorder": True,
            "precision": precision,
            "align": "right",
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": FIELD_BACKGROUND,
        }
        if size is not None:
            props["fontSize"] = size
        if tooltip:
            props[TOOLTIP] = tooltip
        return self.add("pv-label", x, y, w, h, props, parent=parent)

    def led(self, x, y, size, pv, *, parent=None, tooltip=None):
        props = {"pv": pv, "onColor": LED_ON, "offColor": LED_OFF}
        if tooltip:
            props[TOOLTIP] = tooltip
        return self.add("pv-led", x, y, size, size, props, parent=parent)

    def group(self, x, y, w, h, title):
        return self.add("group-box", x, y, w, h, {"title": title})

    def area(self, area: Area, text: str, target: str):
        """A transparent shape that opens ``target``; hovering it reads ``text``.

        The shape is the area's own: an ``ellipse``, or a ``rectangle`` turned
        ``rotation`` degrees about its centre (the runtime turns the hover layer and the
        hit box with it).
        """
        props: dict[str, Any] = {
            "lineColor": PAGE_BACKGROUND,
            "lineWidth": 0,
            "fill": False,
        }
        if area.rotation:
            props["rotation"] = area.rotation
        props[HOVER_LABEL] = text
        props["actions"] = [
            {
                "type": "open_display",
                "file": relative_file(self.screen_id, target),
                "target": NAVIGATION_TARGET,
            }
        ]
        return self.add(area.kind, *area.box, props)

    def box(self, x, y, w, h, *, fill=FIELD_BACKGROUND, line=BOX_BORDER, tooltip=None, parent=None):
        props = {"lineColor": line, "lineWidth": 1, "fill": True, "fillColor": fill}
        if tooltip:
            props[TOOLTIP] = tooltip
        return self.add("rectangle", x, y, w, h, props, parent=parent)

    def dot(self, x, y, size, fill, *, rules=None):
        props = {"lineColor": fill, "lineWidth": 1, "fill": True, "fillColor": fill}
        return self.add("ellipse", x, y, size, size, props, rules=rules)

    def polyline(self, points: Sequence[tuple[float, float]], colour, width=2):
        pad = width + 1
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = math.floor(min(xs)) - pad, math.floor(min(ys)) - pad
        x1, y1 = math.ceil(max(xs)) + pad, math.ceil(max(ys)) + pad
        props = {
            "lineColor": colour,
            "lineWidth": width,
            "points": [{"x": _n(px - x0), "y": _n(py - y0)} for px, py in points],
            "arrowStart": False,
            "arrowEnd": False,
        }
        return self.add("line", x0, y0, x1 - x0, y1 - y0, props)

    def bar(self, x, y, angle, length, thick, colour, *, tooltip=None):
        """A filled bar centred on ``(x, y)``: ``length`` along ``angle``, ``thick`` across it."""
        props: dict[str, Any] = {"lineColor": colour, "fill": True, "fillColor": colour}
        degrees = _n(math.degrees(angle) % 180)
        if degrees:
            props["rotation"] = degrees
        if tooltip:
            props[TOOLTIP] = tooltip
        return self.add("rectangle", x - length / 2, y - thick / 2, length, thick, props)

    # -- document -------------------------------------------------------------

    def document(self) -> dict[str, Any]:
        root = {
            "id": "w-001",
            "type": "absolute-canvas",
            "props": {
                "width": self.width,
                "height": self.height,
                "backgroundColor": PAGE_BACKGROUND,
            },
            "geometry": {"x": 0, "y": 0, "width": self.width, "height": self.height},
            "children": self.nodes,
        }
        return {
            "schemaVersion": "1.1",
            "kind": "screen",
            "id": self.screen_id,
            "metadata": {
                "title": self.title,
                "source": {"type": "native"},
                "size": {"width": self.width, "height": self.height},
                "description": self.description,
            },
            "macros": [],
            "formulas": self.formulas,
            "root": root,
        }


def relative_file(declaring: str, target: str) -> str:
    """The screen file of ``target`` relative to the directory of ``declaring``."""
    up = declaring.count("/")
    return "../" * up + f"{target}.screen.json"


def serialise(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=True) + "\n"


# --------------------------------------------------------------------------- geometry


def pol(cx: float, cy: float, r: float, a: float) -> tuple[float, float]:
    return cx + r * math.cos(a), cy + r * math.sin(a)


@dataclass(frozen=True)
class Ring:
    """A ring of ``cells`` straights and arcs, drawn clockwise from ``start``."""

    cx: float
    cy: float
    r: float
    cells: int
    straight: float
    start: float

    def cell(self, k: int) -> tuple[tuple[float, float], tuple[float, float]]:
        seg = 2 * math.pi / self.cells
        half = seg * self.straight / 2
        a0 = self.start + k * seg
        return (a0 - half, a0 + half), (a0 + half, a0 + seg - half)

    def points(self) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for k in range(self.cells):
            (s0, _), (a1, a2) = self.cell(k)
            pts.append(pol(self.cx, self.cy, self.r, s0))
            steps = max(2, math.ceil((a2 - a1) / math.radians(5)))
            pts.extend(
                pol(self.cx, self.cy, self.r, a1 + (a2 - a1) * i / steps) for i in range(steps + 1)
            )
        pts.append(pts[0])
        return pts


def per_cell(line: str, what: str, count: int, cells: int) -> int:
    if count % cells:
        raise GeometryMismatchError(f"{count} {line} {what} do not divide into {cells} cells")
    return count // cells


def spread(count: int, lo: float, hi: float) -> list[float]:
    """``count`` fractions spread evenly over ``[lo, hi]``."""
    if count == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def sequence(per_family: dict[str, int]) -> list[str]:
    """One cell's symbol families in beam order: dipoles split the cell, the rest fill it."""
    dipoles = per_family.get("DIPOLE", 0)
    others: list[str] = []
    for repeat in range(max(per_family.values(), default=0)):
        others += [f for f, n in per_family.items() if f != "DIPOLE" and n > repeat]
    segments: list[list[str]] = [[] for _ in range(dipoles + 1)]
    for i, family in enumerate(others):
        segments[i % len(segments)].append(family)
    order: list[str] = []
    for i, segment in enumerate(segments):
        order += segment
        if i < dipoles:
            order.append("DIPOLE")
    return order


def layout(
    families: list[str], sizes: dict[str, tuple[float, float]], span: float, where: str
) -> list[float]:
    """Centres along a span of ``span`` px for ``families``, with equal gaps between them."""
    lengths = [sizes[SYMBOL_KIND[f]][0] for f in families]
    gap = (span - sum(lengths)) / (len(families) + 1)
    if gap < 1.0:
        raise GeometryMismatchError(
            f"{len(families)} symbols do not fit the {where} ({span:.0f} px)"
        )
    centres, at = [], gap
    for length in lengths:
        centres.append(at + length / 2)
        at += length + gap
    return centres


class Curve:
    """A cubic Bezier sampled by arc length."""

    def __init__(self, p0, p1, p2, p3, samples: int = 400):
        pts = []
        for i in range(samples + 1):
            t = i / samples
            u = 1 - t
            pts.append(
                (
                    u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
                )
            )
        self._measure(pts)

    @classmethod
    def through(cls, corners: Sequence[tuple[float, float]], step: float = 2.0) -> Curve:
        """A path of straight segments through ``corners``, sampled every ``step`` px or less."""
        pts = [corners[0]]
        for a, b in zip(corners, corners[1:], strict=False):
            n = max(1, math.ceil(math.dist(a, b) / step))
            pts += [
                (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                for i in range(1, n + 1)
            ]
        path = cls.__new__(cls)
        path._measure(pts)
        path.corners = list(corners)
        return path

    def _measure(self, pts: list[tuple[float, float]]) -> None:
        self.pts = pts
        self.corners = None
        self.lengths = [0.0]
        for a, b in zip(self.pts, self.pts[1:], strict=False):
            self.lengths.append(self.lengths[-1] + math.dist(a, b))
        self.length = self.lengths[-1]

    def at(self, s: float) -> tuple[float, float, float]:
        """Point and tangent angle at arc length ``s``."""
        s = min(max(s, 0.0), self.length)
        i = next(k for k in range(1, len(self.lengths)) if self.lengths[k] >= s)
        a, b = self.pts[i - 1], self.pts[i]
        seg = self.lengths[i] - self.lengths[i - 1] or 1.0
        f = (s - self.lengths[i - 1]) / seg
        return (
            a[0] + (b[0] - a[0]) * f,
            a[1] + (b[1] - a[1]) * f,
            math.atan2(b[1] - a[1], b[0] - a[0]),
        )

    def coarse(self, step: int = 10) -> list[tuple[float, float]]:
        """Points to draw the path through: its corners, or every ``step``-th sample."""
        if self.corners is not None:
            return self.corners
        return self.pts[::step] + ([self.pts[-1]] if len(self.pts) % step != 1 else [])


# --------------------------------------------------------------------------- shared parts


def line_families(records: Records, line: str) -> dict[str, int]:
    """Symbol families of a line and their device counts, in ``SYMBOL_KIND`` order."""
    counts = {}
    for family in SYMBOL_KIND:
        n = len(records.device_numbers(f"{line}:MAG:{family}"))
        if n:
            counts[family] = n
    return counts


def draw_ring_lattice(
    screen: Screen,
    records: Records,
    line: str,
    ring: Ring,
    sizes,
    bpm_size: float,
    bpm_inset: float,
    *,
    tooltips: bool = False,
) -> list[str]:
    """The ring's beam path, lattice symbols and BPM dots; returns the BPM status channels.

    With ``tooltips`` each symbol and dot carries its device's name from the records.
    """
    screen.polyline(ring.points(), BEAM)
    counts = line_families(records, line)
    cell_counts = {f: per_cell(line, f"{f} magnets", n, ring.cells) for f, n in counts.items()}
    order = sequence(cell_counts)
    for k in range(ring.cells):
        _, (a1, a2) = ring.cell(k)
        span = (a2 - a1) * ring.r
        devices = cell_devices(records, line, order, k, cell_counts)
        centres = layout(order, sizes, span, f"{line} cell")
        for (family, number), centre in zip(devices, centres, strict=True):
            a = a1 + centre / ring.r
            x, y = pol(ring.cx, ring.cy, ring.r, a)
            length, thick = sizes[SYMBOL_KIND[family]]
            tip = records.device_name(f"{line}:MAG:{family}", number) if tooltips else None
            screen.bar(
                x, y, a + math.pi / 2, length, thick, KIND_ROLE[SYMBOL_KIND[family]], tooltip=tip
            )
    bpms = records.devices(f"{line}:DIAG:BPM", "STATUS:VALID")
    per = per_cell(line, "BPMs", len(bpms), ring.cells)
    for i, name in enumerate(bpms):
        _, (a1, a2) = ring.cell(i // per)
        a = a1 + (a2 - a1) * spread(per, 0.1, 0.9)[i % per]
        x, y = pol(ring.cx, ring.cy, ring.r - bpm_inset, a)
        tip = device_of(records, name) if tooltips else None
        screen.led(x - bpm_size / 2, y - bpm_size / 2, bpm_size, name, tooltip=tip)
    return bpms


def cell_devices(
    records: Records, line: str, order: list[str], cell: int, per_family: dict[str, int]
) -> list[tuple[str, int]]:
    """The device numbers of ``cell``'s symbols in ``order``: each family's next devices.

    The records number a ring's devices ring-wide, so cell ``k`` holds the family's
    devices ``k * per + 1`` to ``(k + 1) * per`` in device order.
    """
    numbers = {f: records.device_numbers(f"{line}:MAG:{f}") for f in per_family}
    seen: dict[str, int] = {}
    out = []
    for family in order:
        j = seen.get(family, 0)
        seen[family] = j + 1
        out.append((family, numbers[family][cell * per_family[family] + j]))
    return out


def device_of(records: Records, channel: str) -> str:
    """The records' device name of ``channel``."""
    device = (records.by_id[records.need(channel)].get("on") or {}).get("device")
    if not device:
        raise MissingRecordError(f"{channel} names no device")
    return device


def draw_curve_lattice(
    screen: Screen,
    records: Records,
    line: str,
    curve: Curve,
    sizes,
    bpm_size: float = 6,
    bpm_offset: float = 9,
    *,
    upto: float | None = None,
) -> list[str]:
    """A transfer line's symbols and BPM dots along ``curve``; returns the BPM status channels.

    They spread over the curve's first ``upto`` px, or all of it.
    """
    length = curve.length if upto is None else upto
    counts = line_families(records, line)
    order = sequence(counts) if counts else []
    if order:
        lo = length * 0.08
        span = length * 0.84
        for family, centre in zip(order, layout(order, sizes, span, f"{line} line"), strict=True):
            x, y, ang = curve.at(lo + centre)
            length, thick = sizes[SYMBOL_KIND[family]]
            screen.bar(x, y, ang, length, thick, KIND_ROLE[SYMBOL_KIND[family]])
    bpms = records.devices(f"{line}:DIAG:BPM", "STATUS:VALID")
    for f, name in zip(spread(len(bpms), 0.1, 0.9), bpms, strict=True):
        x, y, ang = curve.at(length * f)
        ox, oy = x - math.sin(ang) * bpm_offset, y + math.cos(ang) * bpm_offset
        screen.led(ox - bpm_size / 2, oy - bpm_size / 2, bpm_size, name)
    return bpms


def led_row(
    screen: Screen, group: dict, x: float, y: float, items: Sequence[tuple[str, str]]
) -> None:
    """LEDs with their text, left to right from ``x``, as option 4's rows."""
    for text, pv in items:
        screen.led(x, y + 4, 13, pv, parent=group)
        width = 7 * len(text)
        screen.label(x + 18, y - 1, width, 20, text, size=11.5, parent=group)
        x += 18 + width + 14


def overview_led_row(
    screen: Screen,
    group: dict,
    x: float,
    y: float,
    h: float,
    items: Sequence[tuple[str, str]],
) -> float:
    """LEDs with their text at the overview's text size, left to right; returns the end x."""
    led = 15
    for text, pv in items:
        screen.led(x, y + (h - led) / 2, led, pv, parent=group)
        width = text_width(text, TEXT_SMALL + 1)
        screen.label(x + led + 5, y, width, h, text, size=TEXT_SMALL + 1, parent=group)
        x += led + 5 + width + 8
    return x


def booster_supplies(records: Records) -> tuple[list[str], list[str], str]:
    """The booster's ramp supplies (dipole, QF, QD), its two septa and its kicker."""
    ramps = _need_all(records, (f"BR:MAG:{f}-RAMP:01:STATUS:ON" for f in ("DIPOLE", "QF", "QD")))
    septa = records.devices("BR:MAG:SEPTUM", "STATUS:ON")
    kicker = records.need("BR:MAG:KICKER:01:STATUS:ON")
    if len(septa) != 2:
        raise GeometryMismatchError(f"{len(septa)} booster septa; the rows hold two")
    return ramps, septa, kicker


# --------------------------------------------------------------------------- facility overview


@dataclass(frozen=True)
class Area:
    """The clickable shape of one section of the plan, in the section's own drawn shape.

    ``kind`` is ``ellipse`` or ``rectangle``; a rectangle is turned ``rotation``
    degrees about its centre, kept within (-90, 90] so its hover label never reads
    upside down.
    """

    kind: str
    box: Box
    rotation: float = 0

    def holds(self, x: float, y: float, margin: float = 0) -> bool:
        """Whether ``(x, y)`` lies in the shape grown by ``margin`` px."""
        bx, by, w, h = self.box
        mx, my = bx + w / 2, by + h / 2
        if self.kind == "ellipse":
            return math.hypot(x - mx, y - my) <= w / 2 + margin
        a = math.radians(self.rotation)
        dx, dy = x - mx, y - my
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        return abs(u) <= w / 2 + margin and abs(v) <= h / 2 + margin

    def edge(self, step: float = 2.0) -> list[tuple[float, float]]:
        """Points on the shape's outline, ``step`` px apart or less."""
        bx, by, w, h = self.box
        mx, my = bx + w / 2, by + h / 2
        if self.kind == "ellipse":
            n = max(8, math.ceil(math.pi * w / step))
            return [pol(mx, my, w / 2, 2 * math.pi * i / n) for i in range(n)]
        a = math.radians(self.rotation)
        corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        pts = []
        for (u0, v0), (u1, v1) in zip(corners, corners[1:] + corners[:1], strict=True):
            n = max(1, math.ceil(math.hypot(u1 - u0, v1 - v0) / step))
            for i in range(n):
                u, v = u0 + (u1 - u0) * i / n, v0 + (v1 - v0) * i / n
                pts.append(
                    (mx + u * math.cos(a) - v * math.sin(a), my + u * math.sin(a) + v * math.cos(a))
                )
        return pts

    def meets(self, other: Area) -> bool:
        """Whether the two shapes share a point (both are convex)."""
        return any(other.holds(x, y) for x, y in self.edge()) or any(
            self.holds(x, y) for x, y in other.edge()
        )

    @classmethod
    def round(cls, x: float, y: float, r: float) -> Area:
        """A circle of radius ``r`` about ``(x, y)``."""
        return cls("ellipse", (x - r, y - r, 2 * r, 2 * r))

    @classmethod
    def turned(cls, mx: float, my: float, length: float, thick: float, angle: float) -> Area:
        """A ``length`` x ``thick`` box centred on ``(mx, my)``, its length along ``angle``."""
        degrees = math.degrees(angle) % 180
        if degrees > 90:
            degrees -= 180
        box = (mx - length / 2, my - thick / 2, length, thick)
        return cls("rectangle", box, _n(degrees))

    @classmethod
    def sector(cls, ring: Ring, k: int, inner: float, outer: float) -> Area:
        """Cell ``k`` of ``ring`` (its straight, then its arc): a box along the cell's chord.

        The box runs from ``inner`` (on the chord) to ``outer`` radially and spans the
        cell's angle at its own middle radius, so it holds the arc, its symbols and dots
        and the straight's number, and meets its neighbours only in thin corners.
        """
        (s0, _), (_, a2) = ring.cell(k)
        half = (a2 - s0) / 2
        mid = s0 + half
        r0 = inner * math.cos(half)
        rc = (r0 + outer) / 2
        mx, my = pol(ring.cx, ring.cy, rc, mid)
        return cls.turned(mx, my, 2 * rc * math.sin(half), outer - r0, mid + math.pi / 2)

    @classmethod
    def along(
        cls, curve: Curve, pad: float, trim: tuple[float, float], text: str, side: int
    ) -> tuple[Area, Box]:
        """A box turned along the chord of ``curve``, ``pad`` clear of it, and its name's box.

        ``trim`` shortens the box at the start and end of the curve, where the line meets
        the section it leaves and the one it reaches. The name sits clear of the box on
        ``side`` (+1 to the right of the direction of travel).
        """
        (x0, y0), (x1, y1) = curve.pts[0], curve.pts[-1]
        angle = math.atan2(y1 - y0, x1 - x0)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux
        us = [(px - x0) * ux + (py - y0) * uy for px, py in curve.pts]
        vs = [(px - x0) * nx + (py - y0) * ny for px, py in curve.pts]
        u0, u1 = min(us) + trim[0], max(us) - trim[1]
        v0, v1 = min(vs) - pad, max(vs) + pad
        uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
        mx, my = x0 + uc * ux + vc * nx, y0 + uc * uy + vc * ny
        length, thick = u1 - u0, v1 - v0
        w, h = text_width(text, TEXT_SECTION), 24
        off = thick / 2 + 6 + abs(w / 2 * nx) + abs(h / 2 * ny)
        lx, ly = mx + nx * off * side, my + ny * off * side
        label = (lx - w / 2, ly - h / 2, w, h)
        return cls.turned(mx, my, length, thick, angle), label


def facility_overview(records: Records) -> Screen:
    # The ring fills the height at the left edge; the rail follows it on the right.
    height = 940
    big_r = height * 0.445
    sr_out = big_r + 36
    cx, cy = sr_out + 12, height * 0.5
    rx, rw = cx + sr_out + 24, 500
    width = math.ceil(rx + rw + 12)
    screen = Screen(
        OVERVIEW_ID,
        "Facility overview",
        "The control-assistant facility at a glance: the plan of gun, linac, booster and "
        "storage ring with BPM status, each storage-ring sector and each other section "
        "opening its own screen, and the storage-ring status, RF, magnet supplies, injector "
        "and booster, and charge along the chain.",
        width,
        height,
    )
    areas: dict[str, Area] = {}

    # ---- the plan: the left of the canvas, top to bottom ------------------------
    a0 = -math.pi / 3
    sr = Ring(cx, cy, big_r, SR_CELLS, 0.30, a0)
    draw_ring_lattice(
        screen,
        records,
        "SR",
        sr,
        {"dipole": (14, 10), "quadrupole": (5, 16), "sextupole": (4, 13)},
        8,
        16,
    )
    for k in range(SR_CELLS):
        x, y = pol(cx, cy, big_r + 22, a0 + k * 2 * math.pi / SR_CELLS)
        screen.label(x - 14, y - 11, 28, 22, str(k + 1), size=TEXT_SMALL, align="center")
    sectors = [Area.sector(sr, k, big_r - 24, big_r + 34) for k in range(SR_CELLS)]

    # RF at straight 2: device boxes on the ring.
    rf_a = a0 + 2 * math.pi / SR_CELLS
    cavities = records.device_numbers("SR:RF:CAVITY")
    for d in spread(len(cavities), -0.03, 0.03):
        x, y = pol(cx, cy, big_r, rf_a + d)
        screen.box(x - 8, y - 8, 16, 16)
    x, y = pol(cx, cy, sr_out + 22, rf_a)
    screen.label(x - 20, y - 12, 40, 24, "RF", size=TEXT_SECTION, align="center")
    ix, iy = pol(cx, cy, sr_out, a0)
    injection = "Injection"
    screen.label(
        ix + 12,
        iy - 26,
        text_width(injection, TEXT_SECTION),
        24,
        injection,
        size=TEXT_SECTION,
    )

    # Booster right of the centre, its top straight level with the end of GTL.
    bx, by, br_r = cx + big_r * 0.30, cy + big_r * 0.22, big_r * 0.30
    booster = Ring(bx, by, br_r, BR_CELLS, 0.55, -math.pi / 2)
    draw_ring_lattice(
        screen,
        records,
        "BR",
        booster,
        {"dipole": (16, 10), "quadrupole": (5, 15)},
        8,
        12,
    )
    br_out = br_r + 14
    screen.label(bx - 60, by + br_out + 6, 120, 24, "Booster", size=TEXT_SECTION, align="center")
    br_current = records.need("BR:DIAG:DCCT:01:CURRENT:RB")
    screen.label(bx - 62, by - 38, 124, 22, "Beam current", size=TEXT_SMALL, align="center")
    screen.field(bx - 62, by - 12, 124, br_current, precision=2, h=28, size=TEXT)
    areas["BR"] = Area.round(bx, by, br_out)

    # GTL: one line, as gtl/GTL draws it. Gun and linac run straight; the LTB
    # steps down in a dogleg and runs on into the booster's top straight.
    straight_half = math.pi / BR_CELLS * 0.55 / 2
    y2 = by - br_r * math.cos(straight_half)
    ly = y2 - big_r * 0.13
    lx0, lx1 = cx - big_r * 0.72, cx - big_r * 0.08
    ltb_corners = [(lx1, ly), (cx - big_r * 0.04, ly), (cx + big_r * 0.04, y2), (bx, y2)]
    gtl = Curve.through([(lx0, ly), *ltb_corners])
    screen.polyline(gtl.coarse(), BEAM)
    span = lx1 - lx0
    for _ in records.device_numbers("GTL:GUN:EGUN"):
        screen.box(lx0 - 18, ly - 11, 18, 22)
    for lo, hi, prefix, h in (
        (0.06, 0.26, "GTL:RF:SHB", 13),
        (0.30, 0.80, "LN:RF:ACC", 18),
    ):
        n = len(records.device_numbers(prefix))
        if not n:
            raise MissingRecordError(f"no {prefix} device records")
        w = (span * (hi - lo) - 5 * (n - 1)) / n
        for i in range(n):
            screen.box(lx0 + span * lo + i * (w + 5), ly - h / 2, w, h)
    ln_counts = line_families(records, "LN")
    ln_order = sequence(ln_counts) if ln_counts else []
    for f, family in zip(spread(len(ln_order), 0.85, 0.96), ln_order, strict=True):
        screen.bar(lx0 + span * f, ly, 0.0, 5, 16, KIND_ROLE[SYMBOL_KIND[family]])
    gtl_bpms = records.devices("GTL:DIAG:BPM", "STATUS:VALID")
    ln_bpms = records.devices("LN:DIAG:BPM", "STATUS:VALID")
    for names, lo, hi in ((gtl_bpms, 0.04, 0.26), (ln_bpms, 0.5, 0.92)):
        for f, name in zip(spread(len(names), lo, hi), names, strict=True):
            screen.led(lx0 + span * f - 4, ly + 14 - 4, 8, name)
    ltb = Curve.through(ltb_corners)
    draw_curve_lattice(
        screen,
        records,
        "LTB",
        ltb,
        {"dipole": (16, 10), "quadrupole": (5, 16)},
        8,
        12,
        upto=ltb.length - br_r * math.sin(straight_half),
    )
    for text, x in (("Gun", lx0 - 9), ("Linac", lx0 + span * 0.55), ("LTB", cx + big_r * 0.04)):
        w = text_width(text, TEXT_SECTION) + 8
        screen.label(x - w / 2, ly - 42, w, 24, text, size=TEXT_SECTION, align="center")
    # One upright box over gun, linac and the dogleg, up to the booster's area.
    gtl_top, gtl_bottom = ly - 46, y2 + 20
    gtl_right = bx
    while gtl_right > lx1 and Area(
        "rectangle", (lx0 - 24, gtl_top, gtl_right - lx0 + 24, gtl_bottom - gtl_top)
    ).meets(areas["BR"]):
        gtl_right -= 2
    areas["GTL"] = Area(
        "rectangle", (lx0 - 24, gtl_top, gtl_right - 4 - lx0 + 24, gtl_bottom - gtl_top)
    )

    # BTS: on from the end of the booster's right straight, up into the injection
    # straight.
    sx, sy = bx + br_r * math.cos(straight_half), by - br_r * math.sin(straight_half)
    inj_x, inj_y = pol(cx, cy, big_r, a0 + 0.06)
    bts = Curve(
        (sx, sy),
        (sx, sy - big_r * 0.45),
        (inj_x - big_r * 0.24, inj_y + big_r * 0.18),
        (inj_x, inj_y),
    )
    screen.polyline(bts.coarse(), BEAM)

    # The injection septa, on the BTS just inside the ring and clear of sector 1's
    # area, septum 01 nearest the ring. Septum 01 names its pulse waveform.
    septa = records.device_numbers("SR:MAG:SEPTUM")
    if not septa:
        raise MissingRecordError("no SR:MAG:SEPTUM device records")
    side = 18
    at = bts.length
    while at > 0 and sectors[0].holds(*bts.at(at)[:2], side):
        at -= 2
    first = None
    for number in septa:
        at -= side / 2 + 2
        x, y, _ = bts.at(at)
        first = first or (x, y)
        device = records.device_name("SR:MAG:SEPTUM", number)
        tip = f"{device} injection septum"
        if number == septa[0]:
            tip += f" · pulse {records.need('SR:MAG:SEPTUM:01:CURRENT:WAVEFORM')}"
        screen.box(x - side / 2, y - side / 2, side, side, tooltip=tip)
        at -= side / 2 + 2
    sep_text = "SR septum"
    sep_w = text_width(sep_text, TEXT_SMALL)
    screen.label(
        first[0] - sep_w - 20,
        first[1] - 28,
        sep_w,
        20,
        sep_text,
        size=TEXT_SMALL,
        align="right",
    )

    draw_curve_lattice(
        screen,
        records,
        "BTS",
        bts,
        {"dipole": (16, 10), "quadrupole": (5, 16)},
        8,
        12,
        upto=at,
    )
    trim = 0.0
    while True:
        areas["BTS"], bts_label = Area.along(bts, 18, (trim, bts.length - at + 4), "BTS", 1)
        if not areas["BTS"].meets(areas["BR"]):
            break
        trim += 2
    for aid in ("GTL", "BTS", "BR"):
        for k, sector in enumerate(sectors):
            if areas[aid].meets(sector):
                raise GeometryMismatchError(f"the {aid} area meets SR sector {k + 1:02d}'s")
    screen.label(*bts_label, "BTS", size=TEXT_SECTION)

    legend(screen, 12, height - 8 - 6 * 26, (cx, cy, sr_out))

    # ---- the rail ---------------------------------------------------------------
    pad = 14
    right = rx + rw - pad
    row_h = 28
    current = records.need("SR:DIAG:DCCT:01:CURRENT:RB")
    valid = records.need("SR:DIAG:DCCT:01:STATUS:VALID")
    lifetime = records.need("SR:DIAG:DCCT:01:LIFETIME:RB")
    mode = records.need("SR:OPS:MODE:RB")
    rms = orbit_rms_formulas(records)
    for formula in rms:
        screen.formula(formula["name"], formula["expression"], formula["bindings"])

    top = 12
    rows = (
        "Beam current",
        "Lifetime",
        "Energy",
        "Orbit RMS X / Y",
        "SR status",
        "Operation mode",
    )
    pitch = 34
    status_h = 2 * pad + (len(rows) - 1) * pitch + row_h
    screen.box(rx, top, rw, status_h)
    ys = [top + pad + i * pitch for i in range(len(rows))]
    for text, y in zip(rows, ys, strict=True):
        screen.label(rx + pad, y, 190, row_h, text, size=TEXT, bg=FIELD_BACKGROUND)
    screen.field(right - 160, ys[0], 160, current, precision=1, h=row_h, size=TEXT)
    screen.field(right - 160, ys[1], 160, lifetime, precision=2, h=row_h, size=TEXT)
    screen.label(
        right - 160,
        ys[2],
        160,
        row_h,
        ENERGY_TEXT,
        size=TEXT,
        align="right",
        bg=FIELD_BACKGROUND,
    )
    unit_w, rms_w = 30, 78
    screen.label(right - unit_w, ys[3], unit_w, row_h, "µm", size=TEXT_SMALL, bg=FIELD_BACKGROUND)
    screen.field(
        right - unit_w - 4 - rms_w,
        ys[3],
        rms_w,
        "formula://orbit_rms_y",
        units=False,
        h=row_h,
        size=TEXT,
    )
    screen.label(
        right - unit_w - 4 - rms_w - 20,
        ys[3],
        16,
        row_h,
        "/",
        size=TEXT,
        align="center",
        bg=FIELD_BACKGROUND,
    )
    screen.field(
        right - unit_w - 4 - 2 * rms_w - 24,
        ys[3],
        rms_w,
        "formula://orbit_rms_x",
        units=False,
        h=row_h,
        size=TEXT,
    )
    sr_status(
        screen,
        records,
        right - 200,
        ys[4],
        200,
        valid,
        current,
        mode,
        h=row_h,
        size=TEXT,
    )
    screen.field(right - 200, ys[5], 200, mode, units=False, precision=0, h=row_h, size=TEXT)

    # Group boxes: their title band, then rows.
    title_band = 36
    gap = 10

    # RF: cavities and klystrons.
    rf_y = top + status_h + gap
    rf_h = title_band + 2 * 44 + 6
    rf = rail_group(screen, rx, rf_y, rw, rf_h, "RF")
    cav_ready = records.devices("SR:RF:CAVITY", "STATUS:READY")
    kly_on = records.devices("SR:RF:KLYSTRON", "STATUS:ON")
    for row, (text, names) in enumerate((("Cavities", cav_ready), ("Klystrons", kly_on))):
        if not names or len(names) > 6:
            raise GeometryMismatchError(f"{len(names)} {text.lower()} do not fit the RF grid")
        y = rf_y + title_band + row * 44
        screen.label(rx + pad, y, 96, row_h, text, size=TEXT, parent=rf)
        for i, name in enumerate(names):
            lx = rx + pad + 104 + i * 32
            screen.led(lx, y + 3, 16, name, parent=rf)
            number = name.split(":")[3]
            screen.label(lx - 4, y + 20, 24, 16, number, size=12, align="center", parent=rf)
    fwd = records.need("SR:RF:CAVITY:01:POWER:FWD")
    temp = records.need("SR:RF:CAVITY:01:TEMPERATURE:RB")
    for row, (text, pv, precision) in enumerate((("Fwd power", fwd, 0), ("Cav 1 temp", temp, 1))):
        y = rf_y + title_band + row * 44
        screen.label(rx + 262, y, 110, row_h, text, size=TEXT, parent=rf)
        screen.field(right - 106, y, 106, pv, precision=precision, parent=rf, h=row_h, size=TEXT)

    # Storage-ring magnet supplies: one LED per supply, corrector planes as counts.
    mag_y = rf_y + rf_h + gap
    mag_pitch = 23
    rows_y = [mag_y + title_band + 2 + i * mag_pitch for i in range(len(MATRIX_FAMILIES))]
    counts_y = rows_y[-1] + mag_pitch + 6
    mag_h = counts_y + row_h + 8 - mag_y
    mag = rail_group(screen, rx, mag_y, rw, mag_h, "Storage ring magnets · supplies on")
    found = {
        name.split(":")[2]
        for name in records.by_id
        if re.match(r"^SR:MAG:[^:]+:\d+:STATUS:ON$", name)
    }
    unexpected = sorted(found - set(MATRIX_FAMILIES) - set(COUNTED_FAMILIES) - {"SEPTUM"})
    if unexpected:
        raise GeometryMismatchError(
            f"SR supply families with no matrix row: {', '.join(unexpected)}"
        )
    led_x = rx + pad + 76
    cell_w = (right - led_x) / SR_CELLS
    for family, y in zip(MATRIX_FAMILIES, rows_y, strict=True):
        names = records.devices(f"SR:MAG:{family}", "STATUS:ON")
        if not names:
            raise MissingRecordError(f"no SR:MAG:{family}:NN:STATUS:ON records")
        per = per_cell("SR", f"{family} supplies", len(names), SR_CELLS)
        step = cell_w / per
        size = min(12.0, step - 2)
        if size < 6:
            raise GeometryMismatchError(
                f"{len(names)} SR {family} supplies do not fit the matrix row"
            )
        screen.label(
            rx + pad,
            y,
            72,
            mag_pitch,
            FAMILY_LABEL.get(family, family),
            size=TEXT_SMALL,
            parent=mag,
        )
        for j, name in enumerate(names):
            screen.led(led_x + j * step, y + (mag_pitch - size) / 2, size, name, parent=mag)
    for i, family in enumerate(COUNTED_FAMILIES):
        names = records.devices(f"SR:MAG:{family}", "STATUS:ON")
        if not names:
            raise MissingRecordError(f"no SR:MAG:{family}:NN:STATUS:ON records")
        tokens = {f"{k:04d}": name for k, name in enumerate(names)}
        ref = screen.formula(f"{family.lower()}_on", "+".join(f"{{{t}}}" for t in tokens), tokens)
        x = rx + pad + i * 236
        screen.label(x, counts_y, 48, row_h, family, size=TEXT_SMALL, parent=mag)
        screen.field(
            x + 50,
            counts_y,
            64,
            ref,
            units=False,
            precision=0,
            parent=mag,
            h=row_h,
            size=TEXT,
        )
        screen.label(
            x + 120,
            counts_y,
            100,
            row_h,
            f"/ {len(names)} on",
            size=TEXT_SMALL,
            parent=mag,
        )

    # Injector and booster rows.
    inj_y = mag_y + mag_h + gap
    inj_pitch = 30
    inj_h = title_band + 4 * inj_pitch + 4
    inj = rail_group(screen, rx, inj_y, rw, inj_h, "Injector · booster")
    gun = _need_all(records, (f"GTL:GUN:EGUN:01:{s}:RB" for s in ("HV_ON", "HEATER_ON", "AUX_ON")))
    shb = records.devices("GTL:RF:SHB", "HV_ON:RB")
    linac = _need_all(
        records,
        (
            "LN:RF:ACC:02:STATUS:ON",
            "LN:RF:MASTER:01:STATUS:ON",
            "LN:RF:WATER:01:STATUS:ON",
        ),
    )
    ramps, br_septa, kicker = booster_supplies(records)
    sr_septum = records.need("SR:MAG:SEPTUM:01:STATUS:ON")
    if len(shb) != 2:
        raise GeometryMismatchError(f"{len(shb)} SHB; the linac row holds two")
    rows_inj = (
        ("Gun", list(zip(("HV", "heater", "aux"), gun, strict=True))),
        (
            "Linac",
            list(zip(("SHB 1", "2", "ACC", "master", "water"), shb + linac, strict=True)),
        ),
        (
            "Booster",
            list(
                zip(
                    ("ramp", "QF", "QD", "sept 1", "2", "kick"),
                    ramps + br_septa + [kicker],
                    strict=True,
                )
            ),
        ),
        ("Injection", [("SR septum", sr_septum)]),
    )
    for i, (text, items) in enumerate(rows_inj):
        y = inj_y + title_band + i * inj_pitch
        screen.label(rx + pad, y, 88, inj_pitch - 2, text, size=TEXT, parent=inj)
        end = overview_led_row(screen, inj, rx + pad + 92, y, inj_pitch - 2, items)
        if end > right:
            raise GeometryMismatchError(f"the {text.lower()} row does not fit the rail")

    # Charge along the chain.
    ch_y = inj_y + inj_h + gap
    ch_h = title_band + 22 + row_h + 8
    charge = rail_group(screen, rx, ch_y, rw, ch_h, "Charge along the chain")
    lines = ("GTL", "LN", "LTB", "BTS")
    col = (rw - 2 * pad + 8) / len(lines)
    for i, line in enumerate(lines):
        ict = records.need(f"{line}:DIAG:ICT:01:CHARGE:RB")
        x = rx + pad + i * col
        screen.label(
            x,
            ch_y + title_band,
            col - 8,
            22,
            f"{line} ICT",
            size=TEXT_SMALL,
            parent=charge,
        )
        screen.field(
            x,
            ch_y + title_band + 22,
            col - 8,
            ict,
            precision=2,
            parent=charge,
            h=row_h,
            size=TEXT,
        )
    if ch_y + ch_h > height - 8:
        raise GeometryMismatchError(f"the rail ends at {ch_y + ch_h}, the canvas at {height}")

    # ---- section areas, above everything ----------------------------------------
    for k, area in enumerate(sectors):
        n = k + 1
        screen.area(area, f"Open SR sector {n:02d} →", sector_id(n))
    for aid, text, target in AREAS:
        screen.area(areas[aid], f"Open {text} →", target)
    return screen


def rail_group(screen: Screen, x: float, y: float, w: float, h: float, title: str) -> dict:
    """A rail group headed by its title over a rule; a group box alone draws no title."""
    group = screen.group(x, y, w, h, title)
    screen.label(x, y + 4, w, 24, title, size=TEXT_SMALL + 1, parent=group)
    screen.polyline([(x, y + 31), (x + w, y + 31)], BOX_BORDER, 1)
    return group


def orbit_rms_formulas(records: Records) -> list[dict[str, Any]]:
    """The orbit RMS per plane in µm over every storage-ring BPM, to 0.1 µm."""
    out = []
    for plane in ("x", "y"):
        names = records.devices("SR:DIAG:BPM", f"POSITION:{plane.upper()}")
        bindings = {f"{plane}{name.split(':')[3]}": name for name in names}
        terms = " + ".join(f"{{{key}}}**2" for key in bindings)
        out.append(
            {
                "name": f"orbit_rms_{plane}",
                "kind": "scalar",
                "expression": rounded(f"sqrt(({terms}) / {len(names)}) * 1e6", 1),
                "bindings": bindings,
            }
        )
    return out


def sr_condition(records: Records, mode: str) -> dict[str, str]:
    """The SR-status conditions over ``{0}`` valid, ``{1}`` current, ``{2}`` mode."""
    beam = f"{flag_on('{0}')} and {{1}} > {STORED_BEAM_MA:g}"
    user = labelled_is("{2}", records, mode, USER_OPERATION)
    return {
        "delivered": f"{beam} and {user}",
        "stored": beam,
        "valid": flag_on("{0}"),
        "user_no_beam": f"{flag_on('{0}')} and {user}",
    }


def sr_status(
    screen: Screen, records: Records, x, y, w, valid, current, mode, *, h=22, size=13
) -> None:
    """The SR status: a text ladder over the DCCT and the mode; "—" until it knows."""
    c = sr_condition(records, mode)
    pvs = [{"name": name, "trigger": True} for name in (valid, current, mode)]
    text = screen.rule(
        "SR status text",
        "text",
        pvs=pvs,
        conditions=[
            {"expression": c["delivered"], "value": "Beam delivered"},
            {"expression": c["stored"], "value": "Stored beam"},
            {"expression": c["valid"], "value": "No beam"},
        ],
        default="—",
    )
    colour = screen.rule(
        "SR status colour",
        "foregroundColor",
        pvs=[dict(p) for p in pvs],
        conditions=[
            {"expression": c["delivered"], "value": ALARM_OK},
            {"expression": c["stored"], "value": PAGE_TEXT},
            {"expression": c["user_no_beam"], "value": ALARM_MAJOR},
            {"expression": c["valid"], "value": PAGE_TEXT},
        ],
        default=ALARM_DISCONNECTED,
    )
    screen.label(
        x,
        y,
        w,
        h,
        "—",
        size=size,
        align="right",
        bg=FIELD_BACKGROUND,
        fg=ALARM_DISCONNECTED,
        rules=[text, colour],
    )


def legend(screen: Screen, x: float, y: float, ring: tuple[float, float, float]) -> None:
    """The plan's symbols, one per line from ``(x, y)``, in the corner below-left of the ring.

    The rows run from the shortest text to the longest, so each clears the ring's outline
    (centre and radius in ``ring``) where the corner widens toward the bottom.
    """
    items = (
        ("line", BEAM, 18, "beam"),
        ("bar", DIPOLE, 18, "dipole"),
        ("led", LED_ON, 10, "BPM ok"),
        ("bar", SEXTUPOLE, 5, "sextupole"),
        ("bar", QUADRUPOLE, 6, "quadrupole"),
        ("box", FIELD_BACKGROUND, 18, "RF · septum · kicker"),
    )
    rcx, rcy, r = ring
    for i, (kind, colour, w, text) in enumerate(items):
        row = y + i * 26
        mx = x + (18 - w) / 2
        tw = text_width(text, TEXT_SMALL)
        right = x + 26 + tw
        if math.hypot(max(rcx - right, 0), max(row - rcy, 0)) <= r:
            raise GeometryMismatchError(f"the legend row {text!r} meets the storage ring")
        if kind == "bar":
            screen.bar(mx + w / 2, row + 13, 0.0, w, 10, colour)
        elif kind == "led":
            screen.dot(mx, row + 8, 10, colour)
        elif kind == "box":
            screen.box(mx, row + 7, w, 12)
        else:
            screen.polyline([(mx, row + 13), (mx + w, row + 13)], colour)
        screen.label(x + 26, row + 1, tw, 24, text, size=TEXT_SMALL)


# --------------------------------------------------------------------------- booster


def booster_overview(records: Records) -> Screen:
    width, height = 880, 500
    screen = Screen(
        BOOSTER_ID,
        "Booster",
        "The control-assistant booster: the ring with its BPM status, the dipole and "
        "quadrupole ramps, the injection and extraction septa and kicker, and the booster DCCT.",
        width,
        height,
    )
    screen.label(16, 8, 300, 26, "Booster", size=18)
    screen.label(140, 12, 420, 20, "ramps, injection and extraction, beam current", size=12)
    ring = Ring(250, 280, 170, BR_CELLS, 0.55, -math.pi / 2)
    draw_ring_lattice(
        screen,
        records,
        "BR",
        ring,
        {"dipole": (18, 10), "quadrupole": (5, 15)},
        10,
        16,
        tooltips=True,
    )
    ramps, septa, kicker = booster_supplies(records)
    current = records.need("BR:DIAG:DCCT:01:CURRENT:RB")
    valid = records.need("BR:DIAG:DCCT:01:STATUS:VALID")
    gx, gw = 500, 364
    g = screen.group(gx, 56, gw, 60, "Ramps")
    led_row(screen, g, gx + 12, 86, list(zip(("dipole", "QF", "QD"), ramps, strict=True)))
    g = screen.group(gx, 132, gw, 60, "Injection · extraction")
    led_row(
        screen,
        g,
        gx + 12,
        162,
        list(zip(("septum 1", "septum 2", "kicker"), septa + [kicker], strict=True)),
    )
    g = screen.group(gx, 208, gw, 92, "Beam current (DCCT)")
    screen.label(gx + 12, 238, 110, 22, "Current", size=12, parent=g)
    screen.field(gx + 130, 238, 120, current, precision=2, parent=g)
    screen.label(gx + 12, 268, 110, 22, "Reading valid", size=12, parent=g)
    screen.led(gx + 130, 272, 13, valid, parent=g)
    return screen


# --------------------------------------------------------------------------- storage-ring sector


def sector_id(n: int) -> str:
    """The screen of storage-ring sector ``n`` (1-based): straight ``n`` and its arc."""
    return f"{SECTOR_PREFIX}{n:02d}"


def sector_numbers(records: Records, prefix: str, sector: int, what: str) -> list[int]:
    """The device numbers ``<prefix>:<NN>`` of ``sector``: its share of the ring-wide count."""
    numbers = records.device_numbers(prefix)
    per = per_cell("SR", what, len(numbers), SR_CELLS)
    return numbers[(sector - 1) * per : sector * per]


def straight_devices(records: Records, sector: int) -> tuple[str, list[tuple[str, int, str, str]]]:
    """The straight's own devices: its name and (family prefix, number, status, readback) rows.

    Straight 1 holds the injection septa, straight 2 the RF cavities and klystrons;
    the others hold no device of these families.
    """
    if sector == INJECTION_STRAIGHT:
        rows = [
            ("SR:MAG:SEPTUM", n, "STATUS:ON", "CURRENT:RB")
            for n in records.device_numbers("SR:MAG:SEPTUM")
        ]
        return "Injection", rows
    if sector == RF_STRAIGHT:
        rows = [
            ("SR:RF:CAVITY", n, "STATUS:READY", "VOLTAGE:RB")
            for n in records.device_numbers("SR:RF:CAVITY")
        ]
        rows += [
            ("SR:RF:KLYSTRON", n, "STATUS:ON", "POWER:RB")
            for n in records.device_numbers("SR:RF:KLYSTRON")
        ]
        return "RF", rows
    return "", []


def short(device: str) -> str:
    """A device's short name on the sector screen: ``SR/QF05`` reads ``QF 05``."""
    name = device.split("/", 1)[-1]
    match = re.match(r"^(.*?)(\d+)$", name)
    return f"{match.group(1)} {match.group(2)}" if match else name


# The peripheral families a sector screen draws: the channel prefix each device's
# records share. A device belongs to the sector its record places it in.
ION_PUMP = "SR:VAC:ION-PUMP"
GAUGE = "SR:VAC:GAUGE"
VALVE = "SR:VAC:VALVE"
INSERTION_DEVICE = "SR:ID"
SHUTTER = "SR:FE:SHUTTER"
THERMOCOUPLE = "SR:VAC:TC"
SCRAPER = "SR:DIAG:SCRAPER"
AIR = "SR:ENV:AIR"
AIR_HANDLER = "SR:ENV:AHU"
HARMONIC = "SR:RF:HARMONIC"
GAMMA = "SR:DIAG:GAMMA"
NEUTRON = "SR:DIAG:NEUTRON"
# The screen at the end of the BTS line, inside straight 1: its records belong to the
# line, its paddle to the injection straight.
INJECTION_SCREEN = "BTS:DIAG:TV:07"
TV_EMBED = "common/embedded/_tv"

# Pressure bands in Torr: below the first reads good, below the second high.
PRESSURE_BANDS_TORR = (2e-9, 1e-8)
# Absorber temperature bands in degC, against the absorbers' 80 degC trip limit.
TC_BANDS_C = (50.0, 65.0, 80.0)
# The scrapers' full travel in mm, and a scraper's device-list index that is in the
# straight (1) or at the arc's end (6); any other index is mid-arc.
SCRAPER_TRAVEL_MM = 6.4
SCRAPER_STRAIGHT_INDEX = 1
# The device name that says where a scraper sits, by that index.
SCRAPER_WHERE = {1: "straight scraper", 4: "mid-arc scraper", 6: "arc-end scraper"}
SCRAPER_ARC_END_INDEX = 6
# An insertion device's gap that draws its magnet rows fully apart, in mm, how far
# each row then stands off its closed place, and a magnet block's height, in px.
ID_GAP_DRAWN_MM = 60.0
ID_HALF_OPEN = 18
ID_BLOCK_HEIGHT = 8
# An insertion device's magnet rows: blocks, a block's width and the pitch, in px.
ID_BLOCKS, ID_BLOCK_WIDTH, ID_PITCH = 9, 9, 11
# The drawn width of each kind of element in a straight, and the least gap between two.
STRAIGHT_WIDTHS = {
    "box": 26,
    "harmonic": 20,
    "id": ID_BLOCKS * ID_PITCH - (ID_PITCH - ID_BLOCK_WIDTH),
}
STRAIGHT_MIN_GAP = 6
STRAIGHT_NAME_ROWS = 3
# The orbit drawn at the BPMs: a reading of ORBIT_FULL_SCALE_M draws full size, and a
# reading's magnitude tints it OK below the first edge, MINOR below the second, MAJOR
# beyond.
ORBIT_FULL_SCALE_M = 50e-6
ORBIT_BANDS_M = (50e-6, 100e-6)
ORBIT_SCREEN = "sr/orbit-feedback"
# The orbit band under the strip: its top below the beam, a bar's half height, its height.
ORBIT_BAND_TOP = 172
ORBIT_BAR_HALF = 18
ORBIT_BAND_HEIGHT = 96
# The interlock word's bits, from bit 0: the septum's permits.
INTERLOCK_BIT_TEXT = ("Vac", "Crash", "Cover", "Temp", "Ready", "HV")

SECTOR_WIDTH = 1780
SECTOR_STRAIGHT_FRACTION = 0.30
BEAM_Y = 250.0
MINOR = _role("MINOR", "#e0b400", "#f2cc0c")
SHUTTER_OPEN = BEAM


def _natural(name: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.findall(r"\d+|\D+", name))


def placed(records: Records, prefix: str, sector: int) -> list[str]:
    """The names ``<NAME>`` of every ``<prefix>:<NAME>:...`` device placed in ``sector``.

    A device's sector is its record's ``place``, ``SR/SECT<n>``; the names come back
    in natural order (``ST01A`` before ``ST01B``, ``07`` before ``12``).
    """
    pattern = re.compile(rf"^{re.escape(prefix)}:([^:]+):")
    names: dict[str, str] = {}
    for channel in records.by_id:
        if m := pattern.match(channel):
            names.setdefault(m.group(1), device_of(records, channel))
    missing = sorted({d for d in names.values() if d not in records.device_records})
    if missing:
        raise MissingRecordError(f"no device record for {', '.join(missing[:3])}")
    want = f"SR/SECT{sector}"
    return sorted(
        (name for name, device in names.items() if records.place(device) == want), key=_natural
    )


def device_named(records: Records, prefix: str, name: str) -> str:
    """The device of the ``<prefix>:<name>:...`` records."""
    head = f"{prefix}:{name}:"
    for channel in sorted(records.by_id):
        if channel.startswith(head):
            return device_of(records, channel)
    raise MissingRecordError(f"no records under {head}")


def band(token: str, edges: Sequence[float], colours: Sequence[dict]) -> list[dict[str, Any]]:
    """Rule conditions colouring ``{token}`` by the band it reads in, lowest first."""
    conditions = [
        {"expression": f"{token} < {edge:g}", "value": colour}
        for edge, colour in zip(edges, colours, strict=False)
    ]
    conditions.append({"expression": f"{token} >= {edges[-1]:g}", "value": colours[-1]})
    return conditions


@dataclass
class SectorParts:
    """What one sector holds beyond its lattice, each family's names in beam order."""

    pumps: list[str]
    gauges: list[str]
    valves: list[str]
    ids: list[str]
    shutters: list[str]
    tcs: list[str]
    scrapers: list[str]
    air: list[str]
    ahu: list[str]
    harmonic: list[str]
    gamma: list[str]
    neutron: list[str]

    @classmethod
    def of(cls, records: Records, sector: int) -> SectorParts:
        gauges = placed(records, GAUGE, sector)
        # The straight's gauges (ST...) come before the arc's (SR...).
        gauges.sort(key=lambda name: (not name.startswith("ST"), _natural(name)))
        return cls(
            pumps=placed(records, ION_PUMP, sector),
            gauges=gauges,
            valves=placed(records, VALVE, sector),
            ids=placed(records, INSERTION_DEVICE, sector),
            shutters=placed(records, SHUTTER, sector),
            tcs=placed(records, THERMOCOUPLE, sector),
            scrapers=placed(records, SCRAPER, sector),
            air=placed(records, AIR, sector),
            ahu=placed(records, AIR_HANDLER, sector),
            harmonic=placed(records, HARMONIC, sector),
            gamma=placed(records, GAMMA, sector),
            neutron=placed(records, NEUTRON, sector),
        )


def vacuum_id(n: int) -> str:
    """The vacuum detail screen of sector ``n``: every pump, gauge and valve signal."""
    return f"{SECTOR_PREFIX[: -len('sector-')]}_vacuum-{n:02d}"


def absorbers_id(n: int) -> str:
    """The absorber detail screen of straight ``n``: every thermocouple's signals."""
    return f"{SECTOR_PREFIX[: -len('sector-')]}_absorbers-{n:02d}"


def scraper_index(records: Records, name: str) -> int:
    """Where a scraper sits, read from its device names: 1 the straight's start, 4 the
    middle of the arc, 6 the arc's end."""
    names = records.device_records.get(device_named(records, SCRAPER, name), {}).get("names")
    for index, text in SCRAPER_WHERE.items():
        if text in (names or []):
            return index
    raise MissingRecordError(
        f"scraper {name} names no position ({', '.join(SCRAPER_WHERE.values())})"
    )


def shutter_port(records: Records, name: str) -> str:
    """A shutter's port: ``insertion device`` (the straight's end) or ``bend``."""
    names = records.device_records.get(device_named(records, SHUTTER, name), {}).get("names")
    for port in ("insertion device", "bend"):
        if f"{port} port" in (names or []):
            return port
    raise MissingRecordError(f"shutter {name} names no port")


def scraper_family(records: Records, name: str) -> str:
    """A scraper's blade: ``top``, ``bottom`` or ``inside``, read from its device names."""
    device = records.device_records.get(device_named(records, SCRAPER, name), {})
    for family in ("top", "bottom", "inside"):
        if f"{family} scraper" in (device.get("names") or []):
            return family
    raise MissingRecordError(f"scraper {name} names no blade")


def record_label(records: Records, device: str) -> str:
    return records.device_records.get(device, {}).get("label") or short(device)


class SectorDrawing:
    """One storage-ring sector screen under construction: the strip, then the panels."""

    def __init__(self, records: Records, sector: int) -> None:
        self.records = records
        self.sector = sector
        self.parts = SectorParts.of(records, sector)
        self.straight, self.straight_rows = straight_devices(records, sector)
        self.width = SECTOR_WIDTH
        self.x0, self.x1, self.y = 40.0, self.width - 40.0, BEAM_Y
        self.xs = self.x0 + (self.x1 - self.x0) * SECTOR_STRAIGHT_FRACTION
        self.screen = Screen(
            sector_id(sector),
            f"SR sector {sector:02d}",
            f"Storage-ring sector {sector:02d} of the control-assistant facility: straight "
            f"{sector:02d} and its arc in beam order with every magnet, corrector and BPM of "
            "the records, the straight's own devices, the vacuum (ion pumps, gauges, gate "
            "valves), insertion devices, front-end shutters, absorber temperatures, scrapers "
            "and the tunnel environment, with their status and readbacks, and the orbit at "
            "the arc's BPMs as X and Y bars.",
            self.width,
            0,
        )
        self.names: list[tuple[float, str]] = []

    # -- helpers ----------------------------------------------------------------

    def need(self, name: str) -> str:
        return self.records.need(name)

    def name_below(self, cx: float, text: str) -> None:
        """A peripheral's short name in the name lanes under the vacuum lane."""
        self.names.append((cx, text))

    def link(self, x, y, w, text, target, parent=None) -> None:
        """A text that opens ``target`` in a new tab; hovering it reads the screen's name."""
        s = self.screen
        s.label(x, y, w, 20, text, size=12, align="right", parent=parent, fg=BEAM)
        props: dict[str, Any] = {
            "lineColor": PAGE_BACKGROUND,
            "lineWidth": 0,
            "fill": False,
            HOVER_LABEL: f"Open {text.rstrip(' ›')}",
            "actions": [
                {
                    "type": "open_display",
                    "file": relative_file(s.screen_id, target),
                    "target": NAVIGATION_TARGET,
                }
            ],
        }
        s.add("rectangle", x, y, w, 20, props, parent=parent)

    def pressure_rule(self, channel: str, prop: str) -> dict:
        return self.screen.rule(
            f"{channel} band",
            prop,
            pvs=[{"name": channel, "trigger": True}],
            conditions=band("{0}", PRESSURE_BANDS_TORR, (ALARM_OK, MINOR, ALARM_MAJOR)),
            default=ALARM_DISCONNECTED,
        )

    def ntorr(self, name: str, channel: str) -> str:
        if self.records.by_id[channel].get("unit") != "Torr":
            raise GeometryMismatchError(f"{channel} does not read in Torr")
        return self.screen.formula(name, rounded("{p} * 1e9", 2), {"p": channel})

    # -- the strip ----------------------------------------------------------------

    def strip(self) -> None:
        s, y = self.screen, self.y
        s.polyline([(self.x0, y), (self.x1, y)], BEAM)
        s.polyline([(self.xs, y - 190), (self.xs, y + 140)], BOX_BORDER, 1)
        heading = f"Straight {self.sector:02d}" + (f" · {self.straight}" if self.straight else "")
        s.label(self.x0, y - 196, self.xs - self.x0 - 8, 24, heading, size=TEXT)
        s.label(self.xs + 8, y - 196, 200, 24, f"Arc {self.sector:02d}", size=TEXT)
        self.straight_strip()
        self.arc_strip()
        self.vacuum_strip()
        self.shutter_strip()
        self.name_lanes()
        self.orbit_band()

    def straight_strip(self) -> None:
        s, y, r = self.screen, self.y, self.records
        valve, gauges = self.parts.valves[:1], [g for g in self.parts.gauges if g.startswith("ST")]
        in_straight = [
            n for n in self.parts.scrapers if scraper_index(r, n) == SCRAPER_STRAIGHT_INDEX
        ]
        # Beam order: the upstream valve, the first gauge, the scrapers, the screen, the
        # devices, the last gauge.
        if valve:
            self.valve_icon(valve[0], self.x0 + 14)
        if gauges:
            self.gauge_icon(gauges[0], self.x0 + 52)
        if len(gauges) > 1:
            self.gauge_icon(gauges[-1], self.xs - 22)
        if in_straight:
            self.scraper_icons(in_straight, self.x0 + 96)
        lo = self.x0 + 124
        if self.sector == INJECTION_STRAIGHT and self.has_injection_screen():
            self.tv(lo + 4)
            lo += 44
        boxes: list[tuple[str, str, Any]] = [
            ("box", r.device_name(prefix, number), None)
            for prefix, number, _, _ in self.straight_rows
        ]
        boxes.extend(
            ("harmonic", device_named(r, HARMONIC, name), name) for name in self.parts.harmonic
        )
        boxes.extend(
            ("id", device_named(r, INSERTION_DEVICE, name), name) for name in self.parts.ids
        )
        hi = self.xs - (48 if len(gauges) > 1 else 16)
        # Equal gaps between the elements by their drawn widths, so a wide insertion
        # device never reaches its neighbour.
        widths = [STRAIGHT_WIDTHS[kind] for kind, _, _ in boxes]
        gap = (hi - lo - sum(widths)) / (len(boxes) + 1)
        if boxes and gap < STRAIGHT_MIN_GAP:
            raise GeometryMismatchError(
                f"sector {self.sector:02d}: the straight's elements leave {gap:.0f} px between them"
            )
        self.id_x: dict[str, float] = {}
        # Each name takes the first row under the strip where it clears the name before
        # it; an insertion device's open magnet rows fill the top row over its width,
        # so its name takes the second.
        ends = [-math.inf] * STRAIGHT_NAME_ROWS
        left = lo
        for kind, device, name in boxes:
            width = STRAIGHT_WIDTHS[kind]
            cx = left + gap + width / 2
            left = cx + width / 2
            text = record_label(r, device).split(" (")[0] if kind == "id" else short(device)
            size = 13 if kind == "id" else 11
            w = text_width(text, size)
            if kind == "id":
                ends[0] = max(ends[0], cx + width / 2)
            free = [
                k for k, end in enumerate(ends) if cx - w / 2 >= end + 2 and (k or kind != "id")
            ]
            if not free:
                raise GeometryMismatchError(
                    f"sector {self.sector:02d}: no free name row for {text} at x {cx:.0f}"
                )
            row = free[0]
            ends[row] = cx + w / 2
            ty = y + 20 + row * 16
            if kind == "box":
                s.box(cx - 13, y - 14, 26, 28, tooltip=device)
            elif kind == "harmonic":
                s.add(
                    "ellipse",
                    cx - 10,
                    y - 10,
                    20,
                    20,
                    {
                        "lineColor": PAGE_TEXT,
                        "lineWidth": 1,
                        "fill": True,
                        "fillColor": FIELD_BACKGROUND,
                        TOOLTIP: device,
                    },
                )
            else:
                self.id_icon(name, device, cx)
                self.id_x[name] = cx
            s.label(cx - w / 2, ty, w, 16, text, size=size, align="center")
        if self.parts.tcs:
            self.heat_strip(self.x0 + 124, self.xs - 40)

    def has_injection_screen(self) -> bool:
        return f"{INJECTION_SCREEN}:PADDLE:SP" in self.records.by_id

    def tv(self, x: float) -> None:
        s, r = self.screen, self.records
        macros = {
            "TV": "SR TV",
            "light": r.need(f"{INJECTION_SCREEN}:LAMP:RB"),
            "loctv": "loc://srtv1",
            "paddle": r.need(f"{INJECTION_SCREEN}:PADDLE:SP"),
            "paddle_rb": r.need(f"{INJECTION_SCREEN}:PADDLE:RB"),
        }
        s.add(
            "embedded-display",
            x,
            self.y - 170,
            30,
            110,
            {
                "file": relative_file(s.screen_id, TV_EMBED),
                "macros": macros,
                TOOLTIP: device_of(r, f"{INJECTION_SCREEN}:PADDLE:RB"),
            },
        )

    def arc_strip(self) -> None:
        s, y, r, sector = self.screen, self.y, self.records, self.sector
        led = 14
        counts = line_families(r, "SR")
        per_family = {f: per_cell("SR", f"{f} magnets", n, SR_CELLS) for f, n in counts.items()}
        order = sequence(per_family)
        self.magnets = cell_devices(r, "SR", order, sector - 1, per_family)
        sizes = {"dipole": (34, 20), "quadrupole": (10, 32), "sextupole": (8, 26)}
        self.span = self.x1 - self.xs
        centres = layout(order, sizes, self.span, f"SR sector {sector:02d} arc")
        self.dipole_x: list[float] = []
        for i, ((family, number), centre) in enumerate(zip(self.magnets, centres, strict=True)):
            device = r.device_name(f"SR:MAG:{family}", number)
            length, thick = sizes[SYMBOL_KIND[family]]
            cx = self.xs + centre
            if family == "DIPOLE":
                self.dipole_x.append(cx)
            superbend = family == "DIPOLE" and "superbend" in (
                r.device_records.get(device, {}).get("names") or []
            )
            if superbend:
                self.superbend_icon(cx, device)
            else:
                s.bar(cx, y, 0.0, length, thick, KIND_ROLE[SYMBOL_KIND[family]], tooltip=device)
            ty = y + 20 + (i % 2) * 16
            s.label(cx - 30, ty, 60, 16, short(device), size=11, align="center")

        self.hcm = sector_numbers(r, "SR:MAG:HCM", sector, "HCM correctors")
        self.vcm = sector_numbers(r, "SR:MAG:VCM", sector, "VCM correctors")
        self.bpms = sector_numbers(r, "SR:DIAG:BPM", sector, "BPMs")
        if not len(self.hcm) == len(self.vcm) == len(self.bpms):
            raise GeometryMismatchError(
                f"sector {sector:02d}: {len(self.hcm)} HCM, {len(self.vcm)} VCM and "
                f"{len(self.bpms)} BPMs; the strip pairs one of each"
            )
        s.label(self.xs + 8, y - 72, 20, 18, "H", size=12)
        s.label(self.xs + 8, y - 52, 20, 18, "V", size=12)
        s.label(self.xs + 8, y + 56, 40, 18, "BPM", size=12)
        self.bpm_x: list[float] = []
        for f, h, v, b in zip(
            spread(len(self.bpms), 0.1, 0.94), self.hcm, self.vcm, self.bpms, strict=True
        ):
            cx = self.xs + self.span * f
            for row, (prefix, number) in enumerate((("SR:MAG:HCM", h), ("SR:MAG:VCM", v))):
                s.led(
                    cx - led / 2,
                    y - 70 + row * 20,
                    led,
                    r.need(f"{prefix}:{number:02d}:STATUS:ON"),
                    tooltip=r.device_name(prefix, number),
                )
            self.bpm_x.append(cx)
            s.led(
                cx - led / 2,
                y + 58,
                led,
                r.need(f"SR:DIAG:BPM:{b:02d}:STATUS:VALID"),
                tooltip=r.device_name("SR:DIAG:BPM", b),
            )
            s.label(cx - 30, y + 74, 60, 16, f"{b:02d}", size=11, align="center")

        # The arc's gauge, its scrapers and the downstream valve.
        for name in self.parts.gauges:
            if not name.startswith("ST"):
                self.gauge_icon(name, self.xs + self.span * 0.52)
        middle = [
            n
            for n in self.parts.scrapers
            if scraper_index(r, n) not in (SCRAPER_STRAIGHT_INDEX, SCRAPER_ARC_END_INDEX)
        ]
        end = [n for n in self.parts.scrapers if scraper_index(r, n) == SCRAPER_ARC_END_INDEX]
        if middle:
            self.scraper_icons(middle, self.xs + self.span * 0.36)
        if end:
            self.scraper_icons(end, self.xs + self.span * 0.975)
        if len(self.parts.valves) > 1:
            self.valve_icon(self.parts.valves[-1], self.x1 - 8)

    def orbit_tint(self, name: str, position: str, valid: str) -> dict:
        """A fill rule by the reading's magnitude band; grey while the BPM is invalid."""
        return self.screen.rule(
            name,
            "fillColor",
            pvs=[{"name": valid, "trigger": True}, {"name": position, "trigger": True}],
            conditions=[
                {"expression": f"not ({flag_on('{0}')})", "value": ALARM_DISCONNECTED},
                {"expression": f"abs({{1}}) < {ORBIT_BANDS_M[0]:g}", "value": ALARM_OK},
                {"expression": f"abs({{1}}) < {ORBIT_BANDS_M[1]:g}", "value": MINOR},
            ],
            default=ALARM_MAJOR,
        )

    def orbit_band(self) -> None:
        """The orbit at the arc's BPMs: an X row over a Y row of bars from a zero line.

        A bar stands up for a positive reading and hangs down for a negative one, full
        height at the full scale; its colour is the reading's band. The straight's end of
        the band names the scale and opens the orbit feedback.
        """
        s, r, top = self.screen, self.records, self.y + ORBIT_BAND_TOP
        half, bar = ORBIT_BAR_HALF, 16
        k = half / ORBIT_FULL_SCALE_M
        s.polyline([(self.xs, top - 6), (self.x1, top - 6)], BOX_BORDER, 1)
        s.label(
            self.x0, top, self.xs - self.x0 - 16, 18, "Orbit at the BPMs", size=12, align="right"
        )
        s.label(
            self.x0,
            top + 20,
            self.xs - self.x0 - 16,
            16,
            f"bar full height ±{ORBIT_FULL_SCALE_M * 1e6:g} µm · green < {ORBIT_BANDS_M[0] * 1e6:g}, "
            f"amber < {ORBIT_BANDS_M[1] * 1e6:g}",
            size=10,
            align="right",
        )
        self.link(self.xs - 156, top + 42, 140, "Orbit feedback ›", ORBIT_SCREEN)
        for row, plane in enumerate(("X", "Y")):
            zero = top + half + 2 + row * (2 * half + 10)
            s.label(self.xs + 8, zero - 9, 20, 18, plane, size=12)
            s.add(
                "rectangle",
                self.xs + 30,
                zero - half,
                self.x1 - self.xs - 30,
                2 * half,
                {
                    "lineColor": BOX_BORDER,
                    "lineWidth": 1,
                    "fill": True,
                    "fillColor": FIELD_BACKGROUND,
                },
            )
            s.polyline([(self.xs + 30, zero), (self.x1, zero)], BOX_BORDER, 1)
            for b, cx in zip(self.bpms, self.bpm_x, strict=True):
                head = f"SR:DIAG:BPM:{b:02d}"
                position = r.need(f"{head}:POSITION:{plane}")
                scaled = f"{{p}} * {k:.6g}"
                y = s.formula(
                    f"bpm{b:02d}_{plane.lower()}_bar_y",
                    f"{zero:g} - clip({scaled}, 0, {half:g})",
                    {"p": position},
                )
                h = s.formula(
                    f"bpm{b:02d}_{plane.lower()}_bar_h",
                    f"abs(clip({scaled}, -{half:g}, {half:g})) + 1",
                    {"p": position},
                )
                props = {
                    "lineColor": ALARM_DISCONNECTED,
                    "lineWidth": 0,
                    "fill": True,
                    "fillColor": ALARM_DISCONNECTED,
                    TOOLTIP: r.device_name("SR:DIAG:BPM", b),
                }
                rules = [
                    s.rule(f"bpm {b:02d} {plane} bar y", "y", formula=y, default=zero),
                    s.rule(f"bpm {b:02d} {plane} bar height", "height", formula=h, default=1),
                    self.orbit_tint(
                        f"bpm {b:02d} {plane} bar colour",
                        position,
                        r.need(f"{head}:STATUS:VALID"),
                    ),
                ]
                s.add("rectangle", cx - bar / 2, zero, bar, 1, props, rules=rules)

    def superbend_icon(self, cx: float, device: str) -> None:
        """A superconducting bend: the dipole bar inside a cryostat outline."""
        s, y = self.screen, self.y
        s.add(
            "rectangle",
            cx - 22,
            y - 16,
            44,
            32,
            {"lineColor": DIPOLE, "lineWidth": 2, "fill": False, TOOLTIP: device},
        )
        s.bar(cx, y, 0.0, 34, 14, DIPOLE, tooltip=device)

    def vacuum_strip(self) -> None:
        y = self.y
        # Between the BPMs (spread over 0.1 to 0.94), so a stalk never meets a BPM number.
        xs = spread(len(self.parts.pumps), 0.03, 0.87) if self.parts.pumps else []
        for name, f in zip(self.parts.pumps, xs, strict=True):
            self.pump_icon(name, self.xs + self.span * f, y + 104)

    def shutter_strip(self) -> None:
        r = self.records
        ports = [(n, shutter_port(r, n)) for n in self.parts.shutters]
        id_ports = [n for n, port in ports if port == "insertion device"]
        bend_ports = [n for n, port in ports if port == "bend"]
        id_xs = list(self.id_x.values()) or [self.xs - 30]
        for i, name in enumerate(id_ports):
            self.shutter_icon(name, id_xs[min(i, len(id_xs) - 1)] + 6 * (i // len(id_xs)))
        per_dipole: dict[int, int] = {}
        for i, name in enumerate(bend_ports):
            d = i % len(self.dipole_x)
            k = per_dipole.get(d, 0)
            per_dipole[d] = k + 1
            self.shutter_icon(name, self.dipole_x[d] + 20 + 15 * k)

    def name_lanes(self) -> None:
        """Every peripheral's name under the vacuum lane, each on the first free row.

        A name centres under its element and takes the first of the rows where it
        clears the name before it; a name that clears none is a drawing error.
        """
        ends = [-math.inf] * NAME_ROWS
        for cx, text in sorted(self.names):
            w = text_width(text, 10)
            row = next((k for k, end in enumerate(ends) if cx - w / 2 >= end + 2), None)
            if row is None:
                raise GeometryMismatchError(
                    f"sector {self.sector:02d}: no free name row for {text} at x {cx:.0f}"
                )
            ends[row] = cx + w / 2
            ty = self.y + 120 + row * 13
            self.screen.label(cx - w / 2, ty, w, 13, text, size=10, align="center")

    # -- icons ----------------------------------------------------------------

    def pump_icon(self, name: str, cx: float, cy: float) -> None:
        """An ion pump: a four-petal flower on a stalk under the pipe, coloured by pressure."""
        s, r = self.screen, self.records
        device = device_named(r, ION_PUMP, name)
        pressure = r.need(f"{ION_PUMP}:{name}:PRESSURE:RB")
        s.polyline([(cx, self.y + 84), (cx, cy - 6)], BOX_BORDER, 1)
        for dx, dy in ((-5, 0), (5, 0), (0, -5), (0, 5)):
            props = {
                "lineColor": PAGE_TEXT,
                "lineWidth": 1,
                "fill": True,
                "fillColor": ALARM_DISCONNECTED,
                TOOLTIP: device,
            }
            s.add(
                "ellipse",
                cx + dx - 4,
                cy + dy - 4,
                8,
                8,
                props,
                rules=[self.pressure_rule(pressure, "fillColor")],
            )
        s.dot(cx - 2, cy - 2, 4, PAGE_TEXT)
        self.name_below(cx, short(device).replace("ION-PUMP", "IP"))

    def gauge_icon(self, name: str, cx: float) -> None:
        """An ion gauge: a half-dial on a stalk above the pipe, coloured by pressure."""
        s, r, y = self.screen, self.records, self.y
        device = device_named(r, GAUGE, name)
        pressure = r.need(f"{GAUGE}:{name}:PRESSURE:RB")
        s.polyline([(cx, y - 4), (cx, y - 26)], BOX_BORDER, 1)
        s.add(
            "arc",
            cx - 10,
            y - 38,
            20,
            20,
            {
                "lineColor": ALARM_DISCONNECTED,
                "lineWidth": 3,
                "startAngle": 0,
                "spanAngle": 180,
                TOOLTIP: device,
            },
            rules=[self.pressure_rule(pressure, "lineColor")],
        )
        s.polyline([(cx, y - 28), (cx + 5, y - 34)], PAGE_TEXT, 1)
        self.name_below(cx, f"IG {name}")

    def valve_icon(self, name: str, cx: float) -> None:
        """A gate valve: a bow-tie across the pipe, green open, red closed, amber moving."""
        s, r, y = self.screen, self.records, self.y
        device = device_named(r, VALVE, name)
        head = f"{VALVE}:{name}"
        pvs = [
            {"name": r.need(f"{head}:{signal}"), "trigger": True}
            for signal in ("POSITION:OPEN", "POSITION:CLOSED", "STATUS:MOVING", "STATUS:FAULT")
        ]
        colour = s.rule(
            f"valve {name} state",
            "fillColor",
            pvs=pvs,
            conditions=[
                {"expression": flag_on("{3}"), "value": ALARM_MAJOR},
                {"expression": flag_on("{2}"), "value": MINOR},
                {"expression": flag_on("{1}"), "value": ALARM_MAJOR},
                {"expression": flag_on("{0}"), "value": ALARM_OK},
            ],
            default=ALARM_DISCONNECTED,
        )
        w, h = 16, 16
        props = {
            "lineColor": PAGE_TEXT,
            "lineWidth": 1,
            "fill": True,
            "fillColor": ALARM_DISCONNECTED,
            "closed": True,
            "points": [{"x": 0, "y": 0}, {"x": w, "y": h}, {"x": w, "y": 0}, {"x": 0, "y": h}],
            TOOLTIP: device,
        }
        s.add("polygon", cx - w / 2, y - h / 2, w, h, props, rules=[colour])
        self.name_below(cx, short(device).replace("VALVE", "V"))

    def scraper_icons(self, names: list[str], cx: float) -> None:
        """Scraper blades at one position: top above the pipe, bottom and inside below it.

        A blade's tip follows its position readback, from 26 px off the beam retracted to
        3 px at full travel.
        """
        s, r, y = self.screen, self.records, self.y
        for name in names:
            family = scraper_family(r, name)
            device = device_named(r, SCRAPER, name)
            position = r.need(f"{SCRAPER}:{name}:POSITION:RB")
            depth = f"(4 + {23 / SCRAPER_TRAVEL_MM:.4f} * clip({{p}}, 0, {SCRAPER_TRAVEL_MM:g}))"
            x = cx - 3 + (9 if family == "inside" else 0)
            props = {
                "lineColor": PAGE_TEXT,
                "lineWidth": 1,
                "fill": True,
                "fillColor": PAGE_TEXT if family != "inside" else DIPOLE,
                TOOLTIP: device,
            }
            height = s.formula(f"scraper{name}_h", depth, {"p": position})
            rules = [s.rule(f"scraper {name} depth", "height", formula=height, default=4)]
            if family == "top":
                top = y - 30
            else:
                top = y + 26
                ref = s.formula(f"scraper{name}_y", f"{y + 30:g} - {depth}", {"p": position})
                rules.append(s.rule(f"scraper {name} tip", "y", formula=ref, default=top))
            s.add("rectangle", x, top, 6, 4, props, rules=rules)
        self.name_below(cx + 3, "SCR " + "·".join(names))

    def id_icon(self, name: str, device: str, cx: float) -> None:
        """An insertion device: two rows of magnet blocks whose spacing follows the gap.

        An elliptically polarising one's upper row also slides with its row offset.
        """
        s, r, y = self.screen, self.records, self.y
        gap = r.need(f"{INSERTION_DEVICE}:{name}:GAP:RB")
        offset = f"{INSERTION_DEVICE}:{name}:OFFSET:RB"
        bh = ID_BLOCK_HEIGHT
        half = f"(3 + {ID_HALF_OPEN / ID_GAP_DRAWN_MM:.4f} * clip({{g}}, 0, {ID_GAP_DRAWN_MM:g}))"
        top = s.formula(f"id{name}_top_y", f"{y - bh:g} - {half}", {"g": gap})
        bottom = s.formula(f"id{name}_bottom_y", f"{y:g} + {half}", {"g": gap})
        blocks, bw, pitch = ID_BLOCKS, ID_BLOCK_WIDTH, ID_PITCH
        x0 = cx - STRAIGHT_WIDTHS["id"] / 2
        for k in range(blocks):
            for row, ref, rest in (("top", top, y - bh - 9), ("bottom", bottom, y + 9)):
                fill = DIPOLE if k % 2 else QUADRUPOLE
                props = {"lineColor": fill, "fill": True, "fillColor": fill, TOOLTIP: device}
                rules = [s.rule(f"id {name} {row} gap", "y", formula=ref, default=rest)]
                bx = x0 + k * pitch
                if row == "top" and offset in r.by_id:
                    shift = s.formula(
                        f"id{name}_block{k}_x",
                        f"{bx:g} + 0.5 * clip({{o}}, -25, 25)",
                        {"o": offset},
                    )
                    rules.append(s.rule(f"id {name} offset {k}", "x", formula=shift, default=bx))
                s.add("rectangle", bx, rest, bw, bh, props, rules=rules)
        self.owner_badge(name, cx - 14, y - bh - ID_HALF_OPEN - 26)

    def owner_badge(self, name: str, x: float, y: float, parent=None) -> None:
        """BL while the beamline holds the gap, OPS while operations holds it."""
        s, r = self.screen, self.records
        grant = r.need(f"{INSERTION_DEVICE}:{name}:STATUS:GRANT")
        pvs = [{"name": grant, "trigger": True}]
        text = s.rule(
            f"id {name} owner",
            "text",
            pvs=pvs,
            conditions=[{"expression": flag_on("{0}"), "value": "BL"}],
            default="OPS",
        )
        colour = s.rule(
            f"id {name} owner colour",
            "foregroundColor",
            pvs=[dict(p) for p in pvs],
            conditions=[{"expression": flag_on("{0}"), "value": BEAM}],
            default=DIPOLE,
        )
        s.label(x, y, 28, 16, "—", size=11, align="center", parent=parent, rules=[text, colour])

    def shutter_icon(self, name: str, x: float) -> None:
        """A front-end port: a line leaving outboard to a shutter square, filled while open."""
        s, r, y = self.screen, self.records, self.y
        device = device_named(r, SHUTTER, name)
        is_open = r.need(f"{SHUTTER}:{name}:OPEN:RB")
        tip = (x + 10, y - 136)
        s.polyline([(x, y - 8), tip], BOX_BORDER, 1)
        fill = s.rule(
            f"shutter {name} open",
            "fillColor",
            pvs=[{"name": is_open, "trigger": True}],
            conditions=[{"expression": flag_on("{0}"), "value": SHUTTER_OPEN}],
            default=PAGE_BACKGROUND,
        )
        props = {
            "lineColor": PAGE_TEXT,
            "lineWidth": 1,
            "fill": True,
            "fillColor": PAGE_BACKGROUND,
            TOOLTIP: device,
        }
        s.add("rectangle", tip[0] - 5, tip[1] - 12, 10, 10, props, rules=[fill])
        s.label(tip[0] - 10, tip[1] - 26, 20, 13, name, size=9, align="center")

    def heat_cell(self, name: str, x, y, w, h, parent=None) -> None:
        s, r = self.screen, self.records
        temperature = r.need(f"{THERMOCOUPLE}:{name}:TEMPERATURE:RB")
        colour = s.rule(
            f"tc {name} heat",
            "fillColor",
            pvs=[{"name": temperature, "trigger": True}],
            conditions=band("{0}", TC_BANDS_C, (ALARM_OK, MINOR, DIPOLE, ALARM_MAJOR)),
            default=ALARM_DISCONNECTED,
        )
        props = {
            "lineColor": PAGE_BACKGROUND,
            "lineWidth": 1,
            "fill": True,
            "fillColor": ALARM_DISCONNECTED,
            TOOLTIP: device_named(r, THERMOCOUPLE, name),
        }
        s.add("rectangle", x, y, w, h, props, parent=parent, rules=[colour])

    def heat_strip(self, lo: float, hi: float) -> None:
        """The straight's absorber thermocouples: upper row over lower row, along the beam."""
        s, y = self.screen, self.y
        rows: dict[str, list[str]] = {}
        for name in self.parts.tcs:
            rows.setdefault(name[2], []).append(name)
        per = max(len(v) for v in rows.values())
        w = min(24.0, (hi - lo) / per - 2)
        s.label(lo - 64, y + 92, 60, 14, "Absorber", size=10, align="right")
        for k, (_, names) in enumerate(sorted(rows.items(), key=lambda kv: kv[0] != "U")):
            for i, name in enumerate(names):
                self.heat_cell(name, lo + i * (w + 2), y + 92 + k * 11, w, 9)

    # -- panels ---------------------------------------------------------------

    def header(self, group: dict, cells: Sequence[tuple[float, float, str]], y: float) -> None:
        for i, (hx, hw, text) in enumerate(cells):
            align = "left" if i == 0 else "center"
            self.screen.label(hx, y, hw, ROW_H, text, size=12, parent=group, align=align)

    def table(self, x, y, w, title, head, rows: int) -> tuple[dict, float, float]:
        """A rail group with a header row and ``rows`` rows: (group, first row y, bottom)."""
        h = TITLE_BAND + (PITCH if head else 0) + rows * PITCH + 8
        group = rail_group(self.screen, x, y, w, h, title)
        if head:
            self.header(group, head, y + TITLE_BAND)
        first = y + TITLE_BAND + (PITCH if head else 0)
        return group, first, y + h

    def panels(self) -> float:
        top = self.y + 160 + ORBIT_BAND_HEIGHT
        bottoms = [
            self.magnet_panel(16, top),
            self.corrector_panels(592, top),
            self.vacuum_panels(988, top),
            self.side_panels(1404, top),
        ]
        return max(bottoms)

    def magnet_panel(self, mx: float, top: float) -> float:
        s, r = self.screen, self.records
        mw = 560
        mag, y, bottom = self.table(
            mx,
            top,
            mw,
            "Magnet supplies",
            [
                (mx + 14, 120, "Supply"),
                (mx + 150, 60, "On"),
                (mx + 220, 60, "Ready"),
                (mx + 300, 160, "Current"),
            ],
            len(self.magnets),
        )
        for i, (family, number) in enumerate(self.magnets):
            prefix = f"SR:MAG:{family}:{number:02d}"
            ry = y + PITCH * i
            device = r.device_name(f"SR:MAG:{family}", number)
            s.label(mx + 14, ry, 120, ROW_H, short(device), size=TEXT_SMALL, parent=mag)
            s.led(mx + 173, ry + 4, LED, r.need(f"{prefix}:STATUS:ON"), parent=mag)
            s.led(mx + 243, ry + 4, LED, r.need(f"{prefix}:STATUS:READY"), parent=mag)
            s.field(mx + 300, ry, 160, r.need(f"{prefix}:CURRENT:RB"), parent=mag)
        return bottom

    def corrector_panels(self, cx0: float, top: float) -> float:
        s, r = self.screen, self.records
        cw = 380
        cor, y, bottom = self.table(
            cx0,
            top,
            cw,
            "Correctors",
            [
                (cx0 + 14, 80, "H · V"),
                (cx0 + 96, 40, "H on"),
                (cx0 + 140, 84, "H current A"),
                (cx0 + 236, 40, "V on"),
                (cx0 + 280, 84, "V current A"),
            ],
            len(self.hcm),
        )
        for i, (h, v) in enumerate(zip(self.hcm, self.vcm, strict=True)):
            ry = y + PITCH * i
            s.label(cx0 + 14, ry, 80, ROW_H, f"{h:02d} · {v:02d}", size=TEXT_SMALL, parent=cor)
            for lx, fx, prefix, number in (
                (cx0 + 109, cx0 + 140, "SR:MAG:HCM", h),
                (cx0 + 249, cx0 + 280, "SR:MAG:VCM", v),
            ):
                channel = f"{prefix}:{number:02d}"
                s.led(lx, ry + 4, LED, r.need(f"{channel}:STATUS:ON"), parent=cor)
                ref = s.formula(
                    f"{prefix.split(':')[2].lower()}{number:02d}_a",
                    rounded("{c}", 2),
                    {"c": r.need(f"{channel}:CURRENT:RB")},
                )
                s.field(fx, ry, 84, ref, units=False, parent=cor)

        bpm, y, bottom = self.table(
            cx0,
            bottom + 10,
            cw,
            "BPMs · position µm",
            [
                (cx0 + 14, 80, "BPM"),
                (cx0 + 96, 40, "Valid"),
                (cx0 + 140, 110, "X"),
                (cx0 + 258, 110, "Y"),
            ],
            len(self.bpms),
        )
        for i, b in enumerate(self.bpms):
            ry = y + PITCH * i
            channel = f"SR:DIAG:BPM:{b:02d}"
            name = short(r.device_name("SR:DIAG:BPM", b))
            s.label(cx0 + 14, ry, 80, ROW_H, name, size=TEXT_SMALL, parent=bpm)
            s.led(cx0 + 109, ry + 4, LED, r.need(f"{channel}:STATUS:VALID"), parent=bpm)
            for fx, plane in ((cx0 + 140, "x"), (cx0 + 258, "y")):
                ref = s.formula(
                    f"bpm{b:02d}_{plane}_um",
                    rounded("{p} * 1e6", 1),
                    {"p": r.need(f"{channel}:POSITION:{plane.upper()}")},
                )
                s.field(fx, ry, 110, ref, units=False, precision=1, parent=bpm)
        if self.parts.harmonic:
            bottom = self.harmonic_panel(cx0, bottom + 10, cw)
        return bottom

    def harmonic_panel(self, x: float, top: float, w: float) -> float:
        """The harmonic cavities: tuner, body temperature, water flow, power, flags.

        Each readout reads with its record's precision and unit.
        """
        s, r = self.screen, self.records
        cells = (
            (x + 52, 68, "TUNER:RB"),
            (x + 122, 70, "TEMPERATURE:RB"),
            (x + 194, 80, "FLOW:RB"),
            (x + 276, 60, "POWER:RB"),
        )
        group, y, bottom = self.table(
            x,
            top,
            w,
            "Harmonic cavities",
            [
                (x + 14, 38, "Cavity"),
                (x + 52, 68, "Tuner"),
                (x + 122, 70, "Body T"),
                (x + 194, 80, "Water flow"),
                (x + 276, 60, "Power"),
                (x + 336, 22, "OK"),
                (x + 356, 24, "Lp"),
            ],
            len(self.parts.harmonic),
        )
        for i, name in enumerate(self.parts.harmonic):
            ry = y + PITCH * i
            head = f"{HARMONIC}:{name}"
            device = device_named(r, HARMONIC, name)
            s.label(x + 14, ry, 38, ROW_H, f"HC {name}", size=12, parent=group)
            for fx, fw, signal in cells:
                s.field(fx, ry, fw - 2, r.need(f"{head}:{signal}"), parent=group, size=12)
            for lx, signal in ((x + 340, "STATUS:TEMP_OK"), (x + 361, "STATUS:LOOP")):
                s.led(lx, ry + 4, LED, r.need(f"{head}:{signal}"), parent=group, tooltip=device)
        return bottom

    def vacuum_panels(self, vx: float, top: float) -> float:
        s, r = self.screen, self.records
        vw = 400
        pumps = self.parts.pumps
        group, y, bottom = self.table(
            vx,
            top,
            vw,
            "Ion pumps",
            [
                (vx + 14, 100, "Pump"),
                (vx + 120, 90, "P nTorr"),
                (vx + 216, 90, "I"),
                (vx + 314, 36, "On"),
                (vx + 352, 40, "Fault"),
            ],
            len(pumps),
        )
        self.link(vx + vw - 150, top + 6, 140, "Vacuum detail ›", vacuum_id(self.sector), group)
        for i, name in enumerate(pumps):
            ry = y + PITCH * i
            head = f"{ION_PUMP}:{name}"
            device = device_named(r, ION_PUMP, name)
            s.label(vx + 14, ry, 100, ROW_H, short(device), size=TEXT_SMALL, parent=group)
            ref = self.ntorr(f"ip{name}_ntorr", r.need(f"{head}:PRESSURE:RB"))
            s.field(vx + 120, ry, 90, ref, units=False, parent=group)
            s.field(vx + 216, ry, 90, r.need(f"{head}:CURRENT:RB"), parent=group)
            s.led(vx + 325, ry + 4, LED, r.need(f"{head}:STATUS:ON"), parent=group)
            s.led(vx + 365, ry + 4, LED, r.need(f"{head}:STATUS:FAULT"), parent=group)

        gauges = self.parts.gauges
        group, y, bottom = self.table(
            vx,
            bottom + 10,
            vw,
            "Ion gauges",
            [
                (vx + 14, 100, "Gauge"),
                (vx + 120, 90, "P nTorr"),
                (vx + 214, 44, "Valid"),
                (vx + 262, 40, "Trip"),
                (vx + 312, 74, ""),
            ],
            len(gauges),
        )
        for i, name in enumerate(gauges):
            ry = y + PITCH * i
            head = f"{GAUGE}:{name}"
            s.label(vx + 14, ry, 100, ROW_H, f"IG {name}", size=TEXT_SMALL, parent=group)
            ref = self.ntorr(f"ig{name.lower()}_ntorr", r.need(f"{head}:PRESSURE:RB"))
            s.field(vx + 120, ry, 90, ref, units=False, parent=group)
            s.led(vx + 229, ry + 4, LED, r.need(f"{head}:STATUS:VALID"), parent=group)
            s.led(vx + 275, ry + 4, LED, r.need(f"{head}:STATUS:TRIP"), parent=group)
            self.write_button(vx + 316, ry - 1, 66, "Reset", r.need(f"{head}:RESET:SP"), "1", group)

        valves = self.parts.valves
        group, y, bottom = self.table(
            vx,
            bottom + 10,
            vw,
            "Gate valves",
            [
                (vx + 14, 70, "Valve"),
                (vx + 88, 44, "Open"),
                (vx + 136, 50, "Closed"),
                (vx + 190, 52, "Moving"),
                (vx + 246, 40, "Fault"),
                (vx + 290, 100, ""),
            ],
            len(valves),
        )
        for i, name in enumerate(valves):
            ry = y + PITCH * i
            head = f"{VALVE}:{name}"
            device = device_named(r, VALVE, name)
            s.label(vx + 14, ry, 70, ROW_H, short(device), size=TEXT_SMALL, parent=group)
            for lx, signal in (
                (vx + 103, "POSITION:OPEN"),
                (vx + 154, "POSITION:CLOSED"),
                (vx + 209, "STATUS:MOVING"),
                (vx + 259, "STATUS:FAULT"),
            ):
                s.led(lx, ry + 4, LED, r.need(f"{head}:{signal}"), parent=group)
            setpoint = r.need(f"{head}:OPEN:SP")
            self.write_button(vx + 292, ry - 1, 50, "Open", setpoint, "1", group)
            self.write_button(vx + 346, ry - 1, 50, "Close", setpoint, "0", group)
        return bottom

    def write_button(self, x, y, w, label, pv, value, parent) -> None:
        props = {
            "label": label,
            "fontSize": 12,
            "actions": [{"type": "write_pv", "pvName": pv, "value": value}],
        }
        self.screen.add("regular-button", x, y, w, ROW_H + 2, props, parent=parent)

    def side_panels(self, x: float, top: float) -> float:
        bottom = top - 10
        for panel in (
            self.straight_panel,
            self.id_panel,
            self.shutter_panel,
            self.absorber_panel,
            self.scraper_panel,
            self.environment_panel,
            self.radiation_panel,
        ):
            bottom = panel(x, bottom + 10, SIDE_WIDTH)
        return bottom

    def straight_panel(self, x, top, w) -> float:
        s, r = self.screen, self.records
        rows = self.straight_rows
        if not rows:
            return top - 10
        words = [
            (prefix, number)
            for prefix, number, _, _ in rows
            if f"{prefix}:{number:02d}:INTERLOCK:WORD" in r.by_id
        ]
        extra = 1 + len(words) if words else 0
        group, y, bottom = self.table(x, top, w, self.straight, None, len(rows) + extra)
        for i, (prefix, number, status, readback) in enumerate(rows):
            ry = y + PITCH * i
            channel = f"{prefix}:{number:02d}"
            device = short(r.device_name(prefix, number))
            s.label(x + 14, ry, 100, ROW_H, device, size=TEXT_SMALL, parent=group)
            s.led(x + 118, ry + 4, LED, r.need(f"{channel}:{status}"), parent=group)
            s.field(
                x + 140, ry, w - 154, r.need(f"{channel}:{readback}"), precision=1, parent=group
            )
        if words:
            ry = y + PITCH * len(rows)
            s.label(x + 14, ry, 100, ROW_H, "Interlocks", size=12, parent=group)
            cell = 16
            for k, text in enumerate(INTERLOCK_BIT_TEXT):
                s.label(
                    x + 118 + k * (cell + 18) - 6,
                    ry + 4,
                    cell + 22,
                    14,
                    text,
                    size=9,
                    align="center",
                    parent=group,
                )
            for j, (prefix, number) in enumerate(words):
                wy = ry + PITCH * (j + 1)
                channel = r.need(f"{prefix}:{number:02d}:INTERLOCK:WORD")
                device = r.device_name(prefix, number)
                s.label(x + 14, wy, 100, ROW_H, short(device), size=TEXT_SMALL, parent=group)
                for k in range(len(INTERLOCK_BIT_TEXT)):
                    props = {
                        "pv": channel,
                        "numBits": 1,
                        "shift": k,
                        "onColor": LED_ON,
                        "offColor": ALARM_MAJOR,
                        TOOLTIP: device,
                    }
                    s.add(
                        "pv-byte-led",
                        x + 122 + k * (cell + 18),
                        wy + 3,
                        cell,
                        cell,
                        props,
                        parent=group,
                    )
        return bottom

    def id_panel(self, x, top, w) -> float:
        """The straight's insertion devices: gap, row offset, owner, moving, feed-forward."""
        s, r = self.screen, self.records
        ids = self.parts.ids
        if not ids:
            return top - 10
        group, y, bottom = self.table(
            x,
            top,
            w,
            "Insertion devices",
            [
                (x + 14, 100, "Device"),
                (x + 114, 78, "Gap"),
                (x + 194, 74, "Offset"),
                (x + 270, 30, "Own"),
                (x + 300, 18, "Mv"),
                (x + 320, 18, "FF"),
            ],
            len(ids),
        )
        for i, name in enumerate(ids):
            ry = y + PITCH * i
            head = f"{INSERTION_DEVICE}:{name}"
            device = device_named(r, INSERTION_DEVICE, name)
            s.label(x + 14, ry, 100, ROW_H, record_label(r, device), size=11, parent=group)
            s.field(x + 114, ry, 76, r.need(f"{head}:GAP:RB"), parent=group, size=12)
            if f"{head}:OFFSET:RB" in r.by_id:
                s.field(x + 194, ry, 72, r.need(f"{head}:OFFSET:RB"), parent=group, size=12)
            self.owner_badge(name, x + 271, ry + 3, parent=group)
            s.led(x + 302, ry + 4, LED, r.need(f"{head}:STATUS:MOVING"), parent=group)
            s.led(x + 322, ry + 4, LED, r.need(f"{head}:FF:ENABLE"), parent=group)
        return bottom

    def shutter_panel(self, x, top, w) -> float:
        s, r = self.screen, self.records
        names = self.parts.shutters
        if not names:
            return top - 10
        per_row = 9
        rows = math.ceil(len(names) / per_row)
        h = TITLE_BAND + rows * 34 + 8
        group = rail_group(s, x, top, w, h, "Front-end shutters · filled open")
        for i, name in enumerate(names):
            cx = x + 26 + (i % per_row) * 32
            cy = top + TITLE_BAND + (i // per_row) * 34
            is_open = r.need(f"{SHUTTER}:{name}:OPEN:RB")
            fill = s.rule(
                f"shutter {name} open panel",
                "fillColor",
                pvs=[{"name": is_open, "trigger": True}],
                conditions=[{"expression": flag_on("{0}"), "value": SHUTTER_OPEN}],
                default=PAGE_BACKGROUND,
            )
            props = {
                "lineColor": PAGE_TEXT,
                "lineWidth": 1,
                "fill": True,
                "fillColor": PAGE_BACKGROUND,
                TOOLTIP: device_named(r, SHUTTER, name),
            }
            s.add("rectangle", cx - 6, cy, 12, 12, props, parent=group, rules=[fill])
            s.label(cx - 14, cy + 14, 28, 14, name, size=10, align="center", parent=group)
        return top + h

    def absorber_panel(self, x, top, w) -> float:
        s, r = self.screen, self.records
        names = self.parts.tcs
        if not names:
            return top - 10
        group, y, bottom = self.table(x, top, w, "Absorber temperatures", None, 1)
        self.link(x + w - 130, top + 6, 120, "Detail ›", absorbers_id(self.sector), group)
        tokens = {
            f"t{i:02d}": r.need(f"{THERMOCOUPLE}:{n}:TEMPERATURE:RB") for i, n in enumerate(names)
        }
        terms = ", ".join(f"{{{t}}}" for t in tokens)
        ref = s.formula("tc_hottest_c", rounded(f"max(({terms},))", 1), tokens)
        s.label(x + 14, y, 140, ROW_H, f"Hottest of {len(names)}", size=TEXT_SMALL, parent=group)
        s.field(x + 160, y, 80, ref, units=False, parent=group)
        s.label(x + 246, y, 60, ROW_H, "degC", size=TEXT_SMALL, parent=group)
        return bottom

    def scraper_panel(self, x, top, w) -> float:
        """The sector's scraper blades: insertion depth, moving, at a limit, homed."""
        s, r = self.screen, self.records
        names = self.parts.scrapers
        if not names:
            return top - 10
        group, y, bottom = self.table(
            x,
            top,
            w,
            "Scrapers",
            [
                (x + 14, 110, "Blade"),
                (x + 126, 82, "Depth"),
                (x + 214, 36, "Mv"),
                (x + 250, 36, "Lim"),
                (x + 286, 50, "Homed"),
            ],
            len(names),
        )
        for i, name in enumerate(names):
            ry = y + PITCH * i
            head = f"{SCRAPER}:{name}"
            text = f"SCR {name} {scraper_family(r, name)}"
            s.label(x + 14, ry, 110, ROW_H, text, size=12, parent=group)
            s.field(x + 126, ry, 80, r.need(f"{head}:POSITION:RB"), parent=group, size=12)
            s.led(x + 225, ry + 4, LED, r.need(f"{head}:STATUS:MOVING"), parent=group)
            s.led(x + 261, ry + 4, LED, r.need(f"{head}:STATUS:LIMIT"), parent=group)
            s.led(x + 304, ry + 4, LED, r.need(f"{head}:STATUS:HOMED"), parent=group)
        return bottom

    def environment_panel(self, x, top, w) -> float:
        s, r = self.screen, self.records
        air, ahu = self.parts.air, self.parts.ahu
        if not (air or ahu):
            return top - 10
        h = TITLE_BAND + max(len(air) + len(ahu), 1) * PITCH + (34 if ahu else 0) + 8
        group = rail_group(s, x, top, w, h, "Tunnel environment")
        ry = top + TITLE_BAND
        for name in air:
            s.label(x + 14, ry, 130, ROW_H, "Air temperature", size=TEXT_SMALL, parent=group)
            s.field(
                x + 150,
                ry,
                110,
                r.need(f"{AIR}:{name}:TEMPERATURE:RB"),
                parent=group,
                tooltip=device_named(r, AIR, name),
            )
            ry += PITCH
        for name in ahu:
            channel = r.need(f"{AIR_HANDLER}:{name}:VALVE:RB")
            s.label(x + 14, ry, 130, ROW_H, "AHU water valve", size=TEXT_SMALL, parent=group)
            s.field(x + 150, ry, 70, channel, parent=group)
            props = {
                "pv": channel,
                "min": 0,
                "max": 100,
                "showValue": False,
                TOOLTIP: device_named(r, AIR_HANDLER, name),
            }
            s.add("pv-meter", x + 228, ry - 4, 64, 56, props, parent=group)
            ry += PITCH
        return top + h

    def radiation_panel(self, x, top, w) -> float:
        s, r = self.screen, self.records
        rows = [(GAMMA, n, "Gamma") for n in self.parts.gamma]
        rows += [(NEUTRON, n, "Neutron") for n in self.parts.neutron]
        if not rows:
            return top - 10
        group, y, bottom = self.table(
            x,
            top,
            w,
            "Radiation",
            [(x + 14, 100, "Monitor"), (x + 120, 50, "Alarm"), (x + 176, 130, "Dose rate")],
            len(rows),
        )
        for i, (prefix, name, text) in enumerate(rows):
            ry = y + PITCH * i
            head = f"{prefix}:{name}"
            s.label(x + 14, ry, 100, ROW_H, f"{text} {name}", size=TEXT_SMALL, parent=group)
            s.led(
                x + 138,
                ry + 4,
                LED,
                r.need(f"{head}:STATUS:ALARM"),
                parent=group,
                tooltip=device_named(r, prefix, name),
            )
            s.field(x + 176, ry, 130, r.need(f"{head}:DOSE_RATE:INST"), parent=group)
        return bottom

    # -- assembly -------------------------------------------------------------

    def draw(self) -> Screen:
        s = self.screen
        s.label(16, 10, 160, 30, s.title, size=22)
        s.label(
            180,
            16,
            700,
            22,
            "straight and arc in beam order · device names under the strip",
            size=TEXT_SMALL,
        )
        self.strip()
        bottom = self.panels()
        s.height = int(math.ceil(bottom + 16))
        return s


# The rows of peripheral names under a sector screen's vacuum lane.
NAME_ROWS = 3

# Panel geometry of the sector and detail screens.
TITLE_BAND, PITCH, ROW_H, LED = 36, 26, 22, 14
SIDE_WIDTH = 360


def sector_screen(records: Records, sector: int) -> Screen:
    """One storage-ring sector, unrolled: its lattice and peripherals in beam order, then panels.

    The strip runs left to right through the straight and the arc: the straight's own
    devices (septa, RF cavities and klystrons, harmonic cavities, insertion devices whose
    magnet rows open with their gap), the magnets in the order the overview draws them,
    a corrector pair over each BPM, gate valves across the pipe at both ends, gauges
    above it and ion pumps under it coloured by pressure, scraper blades whose depth
    follows their position, front-end ports leaving outboard to their shutters, and the
    straight's absorber temperatures as a heat strip. Under the arc the orbit band draws
    each BPM's X and Y reading as a bar below it. Each element carries its device
    name as its tooltip. Below the strip one panel per family binds the status and
    readbacks; the vacuum and the absorbers each open a detail screen.
    """
    return SectorDrawing(records, sector).draw()


def detail_screen(records: Records, sector: int, kind: str) -> Screen:
    """A sector's vacuum or absorber detail: every signal of the family's devices."""
    parts = SectorParts.of(records, sector)
    if kind == "vacuum":
        screen_id, title = vacuum_id(sector), f"SR sector {sector:02d} · vacuum"
        what = "every ion pump, ion gauge and gate valve signal"
    else:
        screen_id, title = absorbers_id(sector), f"SR straight {sector:02d} · absorbers"
        what = "every absorber thermocouple's temperature, limit and over-temperature flag"
    screen = Screen(
        screen_id,
        title,
        f"Detail of storage-ring sector {sector:02d} of the control-assistant facility: {what}.",
        1200,
        0,
    )
    s, r = screen, records
    s.label(16, 10, 520, 30, title, size=22)
    back = SectorDrawing.__new__(SectorDrawing)
    back.screen, back.records = s, r
    back.link(900, 16, 280, f"SR sector {sector:02d} ›", sector_id(sector))
    top = 60
    if kind == "vacuum":
        bottom = _vacuum_detail(back, parts, top)
    else:
        bottom = _absorber_detail(back, parts, top)
    s.height = int(math.ceil(bottom + 16))
    return s


def _vacuum_detail(d: SectorDrawing, parts: SectorParts, top: float) -> float:
    s, r, x, w = d.screen, d.records, 16, 1168
    group, y, bottom = d.table(
        x,
        top,
        w,
        "Ion pumps",
        [
            (x + 14, 120, "Pump"),
            (x + 150, 100, "P nTorr"),
            (x + 260, 110, "Current"),
            (x + 380, 110, "Voltage"),
            (x + 500, 110, "Voltage set"),
            (x + 620, 40, "On"),
            (x + 670, 50, "Fault"),
        ],
        len(parts.pumps),
    )
    for i, name in enumerate(parts.pumps):
        ry, head = y + PITCH * i, f"{ION_PUMP}:{name}"
        s.label(
            x + 14,
            ry,
            120,
            ROW_H,
            short(device_named(r, ION_PUMP, name)),
            size=TEXT_SMALL,
            parent=group,
        )
        s.field(
            x + 150,
            ry,
            100,
            d.ntorr(f"ip{name}_ntorr", r.need(f"{head}:PRESSURE:RB")),
            units=False,
            parent=group,
        )
        for fx, signal in (
            (x + 260, "CURRENT:RB"),
            (x + 380, "VOLTAGE:RB"),
            (x + 500, "VOLTAGE:SP"),
        ):
            s.field(fx, ry, 110, r.need(f"{head}:{signal}"), parent=group)
        s.led(x + 633, ry + 4, LED, r.need(f"{head}:STATUS:ON"), parent=group)
        s.led(x + 688, ry + 4, LED, r.need(f"{head}:STATUS:FAULT"), parent=group)

    group, y, bottom = d.table(
        x,
        bottom + 10,
        w,
        "Ion gauges",
        [
            (x + 14, 120, "Gauge"),
            (x + 150, 100, "P nTorr"),
            (x + 260, 40, "On"),
            (x + 310, 50, "Valid"),
            (x + 370, 40, "Trip"),
            (x + 420, 80, ""),
        ],
        len(parts.gauges),
    )
    for i, name in enumerate(parts.gauges):
        ry, head = y + PITCH * i, f"{GAUGE}:{name}"
        s.label(x + 14, ry, 120, ROW_H, f"IG {name}", size=TEXT_SMALL, parent=group)
        s.field(
            x + 150,
            ry,
            100,
            d.ntorr(f"ig{name.lower()}_ntorr", r.need(f"{head}:PRESSURE:RB")),
            units=False,
            parent=group,
        )
        for lx, signal in (
            (x + 273, "STATUS:ON"),
            (x + 328, "STATUS:VALID"),
            (x + 383, "STATUS:TRIP"),
        ):
            s.led(lx, ry + 4, LED, r.need(f"{head}:{signal}"), parent=group)
        d.write_button(x + 424, ry - 1, 70, "Reset", r.need(f"{head}:RESET:SP"), "1", group)

    signals = (
        ("Open", "POSITION:OPEN"),
        ("Closed", "POSITION:CLOSED"),
        ("Moving", "STATUS:MOVING"),
        ("Fault", "STATUS:FAULT"),
        ("Ready", "STATUS:READY"),
        ("Open cmd", "CONTROL:OPEN"),
        ("Close cmd", "CONTROL:CLOSE"),
        ("Open set", "OPEN:SP"),
    )
    head_cells = [(x + 14, 120, "Valve")]
    head_cells += [(x + 150 + 76 * k, 72, text) for k, (text, _) in enumerate(signals)]
    group, y, bottom = d.table(x, bottom + 10, w, "Gate valves", head_cells, len(parts.valves))
    for i, name in enumerate(parts.valves):
        ry, head = y + PITCH * i, f"{VALVE}:{name}"
        s.label(
            x + 14,
            ry,
            120,
            ROW_H,
            short(device_named(r, VALVE, name)),
            size=TEXT_SMALL,
            parent=group,
        )
        for k, (_, signal) in enumerate(signals):
            s.led(x + 179 + 76 * k, ry + 4, LED, r.need(f"{head}:{signal}"), parent=group)
        setpoint = r.need(f"{head}:OPEN:SP")
        d.write_button(x + 770, ry - 1, 60, "Open", setpoint, "1", group)
        d.write_button(x + 836, ry - 1, 60, "Close", setpoint, "0", group)
    return bottom


def _absorber_detail(d: SectorDrawing, parts: SectorParts, top: float) -> float:
    s, r, x, w = d.screen, d.records, 16, 1168
    rows: dict[str, list[str]] = {}
    for name in parts.tcs:
        rows.setdefault(name[2], []).append(name)
    bottom = top - 10
    for row, names in sorted(rows.items(), key=lambda kv: kv[0] != "U"):
        title = "Upper row" if row == "U" else "Lower row"
        group, y, bottom = d.table(x, bottom + 10, w, title, None, 4)
        cw = min(104.0, (w - 140) / len(names))
        for k, text in enumerate(("", "T degC", "Limit degC", "Over")):
            if text:
                s.label(x + 14, y + PITCH * k, 110, ROW_H, text, size=12, parent=group)
        for i, name in enumerate(names):
            cx = x + 130 + i * cw
            head = f"{THERMOCOUPLE}:{name}"
            d.heat_cell(name, cx + 2, y + 4, cw - 4, 14, parent=group)
            s.label(cx, y - 2, cw, 14, name[3:], size=9, align="center", parent=group)
            s.field(
                cx + 2,
                y + PITCH,
                cw - 4,
                r.need(f"{head}:TEMPERATURE:RB"),
                units=False,
                parent=group,
            )
            s.field(
                cx + 2, y + 2 * PITCH, cw - 4, r.need(f"{head}:LIMIT:RB"), units=False, parent=group
            )
            s.led(
                cx + cw / 2 - LED / 2,
                y + 3 * PITCH + 4,
                LED,
                r.need(f"{head}:STATUS:OVERTEMP"),
                parent=group,
            )
    return bottom


# --------------------------------------------------------------------------- CLI


def render(records: Records, library: Path) -> dict[str, str]:
    """The generated screen files by library-relative path."""
    docs = {
        OVERVIEW_ID: facility_overview(records).document(),
        BOOSTER_ID: booster_overview(records).document(),
    }
    for n in range(1, SR_CELLS + 1):
        docs[sector_id(n)] = sector_screen(records, n).document()
        docs[vacuum_id(n)] = detail_screen(records, n, "vacuum").document()
        if SectorParts.of(records, n).tcs:
            docs[absorbers_id(n)] = detail_screen(records, n, "absorbers").document()
    held = (
        {
            p.relative_to(library / "screens").as_posix()[: -len(".screen.json")]
            for p in (library / "screens").rglob("*.screen.json")
        }
        if (library / "screens").is_dir()
        else set()
    )
    held |= set(docs)
    for screen_id, doc in docs.items():
        for target in _targets(doc):
            resolved = _resolve(screen_id, target)
            if resolved not in held:
                raise MissingScreenError(
                    f"{screen_id} opens {target}, which the library does not hold"
                )
    return {f"screens/{screen_id}.screen.json": serialise(doc) for screen_id, doc in docs.items()}


def _targets(doc: dict[str, Any]) -> list[str]:
    out = []

    def walk(node):
        out.extend(
            action["file"]
            for action in node.get("props", {}).get("actions", [])
            if action.get("type") == "open_display"
        )
        for child in node.get("children", []):
            walk(child)

    walk(doc["root"])
    return out


def _resolve(declaring: str, file: str) -> str:
    parts = declaring.split("/")[:-1]
    for piece in file.split("/"):
        if piece == "..":
            parts.pop()
        else:
            parts.append(piece)
    return "/".join(parts)[: -len(".screen.json")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("write", "check"))
    parser.add_argument("--library", type=Path, required=True, help="The CANOPY screen library.")
    parser.add_argument(
        "--records",
        type=Path,
        default=EXAMPLE_RECORDS,
        help="The facility's channels.yaml.",
    )
    args = parser.parse_args(argv)
    try:
        files = render(load_records(args.records), args.library)
    except GeneratorError as error:
        print(f"facility_overview: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    stale = []
    for rel, text in files.items():
        path = args.library / rel
        if args.mode == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            print(f"wrote {rel}")
        elif not path.is_file() or path.read_bytes() != text.encode("utf-8"):
            stale.append(rel)
    for rel in stale:
        print(f"stale: {rel}", file=sys.stderr)
    if args.mode == "check" and not stale:
        print(f"current: {', '.join(files)}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
