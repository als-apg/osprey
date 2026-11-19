"""Every rendered panel view is a valid Screen-IR 1.1 document, by CANOPY's own checks.

``osprey.facility.panels.layout`` draws the panels fixture's views (every view
kind) and the invented views of ``tests/facility/panels``.

The contract is the schema and runtime this repository pins: CANOPY and the PyDM
converter at the commits ``osprey.integrations.canopy`` proves
(``PROVEN_COMMITS``), the same ones the rest of ``tests/templates/canopy``
validates against. It is not a copy in another checkout, such as
``canopy-sdk-js``. The schema is the PyDM converter's copy
(``pydmconverter.ir``), staged the way ``tests/templates/canopy`` stages it;
``check_schema`` and ``check_semantic`` are CANOPY's. Importing CANOPY at
module level carries the ``canopy`` mark.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest
from canopy_mcp.check.schema import check_schema, schema_path
from canopy_mcp.check.semantic import check_semantic
from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility
from tests.facility.panels._views import (
    RULED,
    fixture_views,
    invented_views,
    nodes,
    rollup_view,
    strip_view,
    stub_render,
)

from osprey.facility.panels import layout, symbols
from osprey.facility.panels.describe import light
from osprey.facility.panels.model import PanelGraph, load_graph
from osprey.facility.panels.view import StripElement

WRITE_NOT_RELAYED = "write_not_relayed"


@pytest.fixture(scope="module")
def panel_graph(tmp_path_factory: pytest.TempPathFactory) -> PanelGraph:
    facility = write_fixture_facility(tmp_path_factory.mktemp("panels") / "facility.json")
    return load_graph(facility, PANELS_DIR)


@pytest.fixture(scope="module")
def spec_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("spec")
    source = files("pydmconverter.ir").joinpath("data", "screen-ir.schema.json")
    target = schema_path(root, "1.1")
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    return root


def _check(doc: dict, spec_dir: Path, view) -> None:
    assert check_schema(doc, spec_dir) == [], view.id
    findings = check_semantic(doc)
    assert [f for f in findings if f.severity == "error"] == [], view.id
    # A setpoint is an input; its only warning says this server relays no writes.
    assert {f.code for f in findings} <= {WRITE_NOT_RELAYED}, view.id


def test_an_aggregate_roll_up_validates(panel_graph, spec_dir: Path, monkeypatch) -> None:
    # A limit that every single fixture channel fits and the three-channel child does not.
    singles = [
        layout.engine_length(layout.ladder(panel_graph, [light(panel_graph, c)])) for c in RULED
    ]
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", max(singles))
    doc = stub_render(rollup_view(), panel_graph)
    assert layout.aggregates(doc) == 1  # the three-channel child; the root says it needs one
    assert "server-side aggregate" in layout.needs(doc)
    _check(doc, spec_dir, rollup_view())


def test_every_view_of_every_kind_validates(panel_graph, spec_dir: Path) -> None:
    views = [*fixture_views(panel_graph), *invented_views(), rollup_view()]
    kinds = {"home", "card", "set", "calibration", "place", "rack", "rollup", "checkout"}
    assert {v.id.view for v in views} == kinds
    for view in views:
        _check(stub_render(view, panel_graph), spec_dir, view)


def test_a_strip_of_every_symbol_in_a_live_state_validates(panel_graph, spec_dir: Path) -> None:
    channel = "XR02-IP1:Pressure-RB"  # stated literal rows the format can write
    elements = [
        StripElement(f"E{i}", None, float(i), kind, "beam", None, (channel,))
        for i, kind in enumerate(symbols.KINDS)
    ]
    view = strip_view(elements)
    doc = stub_render(view, panel_graph)
    ruled = {n["type"] for n in nodes(doc) if n.get("rules")}
    assert {"polygon", "text-label"} == ruled  # the ring and its word; a symbol has no rule
    _check(doc, spec_dir, view)
