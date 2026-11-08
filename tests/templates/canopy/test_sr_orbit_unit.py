"""The storage-ring screens scale the orbit by the unit its BPM channels declare.

The SR BPM positions read in the unit ``records/channels.yaml`` gives them, and
every SR screen that shows the orbit in µm multiplies by the µm in that unit:
the orbit-feedback plots and RMS readouts, the facility overview's RMS, and each
sector's BPM readouts, orbit bars and band rules. A unit moved in the records but
not in the screens (or the reverse) shows the orbit off by a power of ten, so
this module ties each scale factor to the declared unit. It reads only the
screen JSON and the records, and so runs in every lane.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
LIBRARY = (
    ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "demo"
    / "data"
    / "canopy"
    / "library"
    / "screens"
)
CHANNELS = (
    ROOT / "src" / "osprey" / "templates" / "facilities" / "example" / "records" / "channels.yaml"
)
GENERATOR = ROOT / "scripts" / "facility_demo" / "facility_overview.py"
ORBIT_FEEDBACK = LIBRARY / "sr" / "orbit-feedback.screen.json"
OVERVIEW = LIBRARY / "facility" / "overview.screen.json"
SECTORS = [LIBRARY / "sr" / f"sector-{n:02d}.screen.json" for n in range(1, 13)]

#: µm in one of each length unit a BPM channel may declare.
UM_PER = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0}

BPM_POSITION = re.compile(r"SR:DIAG:BPM:\d+:POSITION:[XY]")
# A readout in µm: ``floor((<reading> * <factor>) * 10 + 0.5) / 10``.
READOUT_FACTOR = re.compile(r"\* ([0-9.e+-]+)\) \* 10 \+ 0\.5\) / 10$")
# An orbit bar's height: ``abs(clip({p} * <px per unit>, -<half>, <half>)) + 1``.
BAR_HEIGHT = re.compile(r"^abs\(clip\(\{p\} \* ([0-9.e+-]+), -([0-9.]+), ([0-9.]+)\)\) \+ 1$")
BAND_EDGE = re.compile(r"^abs\(\{1\}\) < ([0-9.e+-]+)$")
SCALE_TEXT = re.compile(r"±([0-9.]+) µm · green < ([0-9.]+), amber < ([0-9.]+)")


@pytest.fixture(scope="module")
def um_per_unit() -> float:
    rows = yaml.safe_load(CHANNELS.read_text(encoding="utf-8"))
    units = {row.get("unit") for row in rows if BPM_POSITION.fullmatch(row["id"])}
    assert len(units) == 1, units
    [unit] = units
    assert unit in UM_PER, unit
    return UM_PER[unit]


def _doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _nodes(node: dict):
    yield node
    for child in node.get("children", []):
        yield from _nodes(child)


def _binds_orbit(formula: dict) -> bool:
    channels = list(formula["bindings"].values())
    return bool(channels) and all(BPM_POSITION.fullmatch(c) for c in channels)


def _readout_factor(expression: str) -> float:
    match = READOUT_FACTOR.search(expression)
    assert match, expression
    return float(match.group(1))


def test_the_generator_draws_in_the_declared_unit(um_per_unit: float) -> None:
    spec = importlib.util.spec_from_file_location("facility_overview_unit", GENERATOR)
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gen
    try:
        spec.loader.exec_module(gen)
    finally:
        sys.modules.pop(spec.name, None)
    assert UM_PER[gen.ORBIT_UNIT] == um_per_unit
    assert gen.UM_PER_ORBIT_UNIT == um_per_unit


def test_orbit_feedback_plots_and_rms_scale_by_the_declared_unit(um_per_unit: float) -> None:
    doc = _doc(ORBIT_FEEDBACK)
    plots = [
        n["props"]
        for n in _nodes(doc["root"])
        if n["type"] == "pv-array-plot"
        and all(BPM_POSITION.fullmatch(c) for c in n["props"]["curves"][0]["y_channels"])
    ]
    assert len(plots) == 2
    for props in plots:
        assert props["yAxes"][0]["label"] == "µm"
        assert props["scale"] == um_per_unit
    rms = [f for f in doc["formulas"] if f["name"].startswith("orbit_rms_")]
    assert len(rms) == 2
    for formula in rms:
        assert _binds_orbit(formula), formula["name"]
        assert _readout_factor(formula["expression"]) == um_per_unit, formula["name"]


def test_the_overview_rms_scales_by_the_declared_unit(um_per_unit: float) -> None:
    rms = [f for f in _doc(OVERVIEW)["formulas"] if f["name"].startswith("orbit_rms_")]
    assert len(rms) == 2
    for formula in rms:
        assert _readout_factor(formula["expression"]) == um_per_unit, formula["name"]


@pytest.mark.parametrize("path", SECTORS, ids=lambda p: p.name.split(".")[0])
def test_sector_readouts_bars_and_bands_follow_the_declared_unit(
    path: Path, um_per_unit: float
) -> None:
    """Readouts multiply into µm; bars and band edges agree with the µm the band label states."""
    doc = _doc(path)
    [text] = [
        n["props"]["text"]
        for n in _nodes(doc["root"])
        if SCALE_TEXT.search(n.get("props", {}).get("text") or "")
    ]
    full_um, green_um, amber_um = (float(v) for v in SCALE_TEXT.search(text).groups())
    readouts = bars = 0
    for formula in doc["formulas"]:
        if not _binds_orbit(formula):
            continue
        name, expression = formula["name"], formula["expression"]
        if name.endswith("_um"):
            assert _readout_factor(expression) == um_per_unit, name
            readouts += 1
        elif match := BAR_HEIGHT.match(expression):
            px_per_unit, low, high = (float(v) for v in match.groups())
            assert low == high
            assert high / px_per_unit * um_per_unit == pytest.approx(full_um), name
            bars += 1
    assert readouts > 0 and bars == readouts
    edges = set()
    for node in _nodes(doc["root"]):
        for rule in node.get("rules", []):
            pvs = [p["name"] for p in rule.get("pvs", [])]
            if rule["targetProperty"] != "fillColor" or not any(
                BPM_POSITION.fullmatch(p) for p in pvs
            ):
                continue
            edges |= {
                float(m.group(1))
                for c in rule["conditions"]
                if (m := BAND_EDGE.match(c["expression"]))
            }
    assert sorted(e * um_per_unit for e in edges) == pytest.approx([green_um, amber_um])


def test_the_library_report_covers_the_rescaled_screens():
    """A rescaled screen ships with a regenerated ``LIBRARY-REPORT.json``.

    The build's ``canopy-payload`` step refuses a library whose report does not
    cover its files, so a unit move that edits the screens without regenerating
    the report breaks every build of the template.
    """
    from osprey.integrations.canopy.library import library_is_fresh

    answer = library_is_fresh(LIBRARY.parent)
    assert answer.fresh, "\n".join(answer.lines)
