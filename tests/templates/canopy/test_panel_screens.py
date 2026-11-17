"""Every rendered panel view is a valid Screen-IR 1.1 document, by CANOPY's own checks.

``osprey.facility.panels.layout`` draws the panels fixture's views (every view
kind) and the invented views of ``tests/facility/panels``. The schema is the PyDM converter's copy (``pydmconverter.ir``), staged the way
``tests/templates/canopy`` stages it; ``check_schema`` and ``check_semantic``
are CANOPY's. Importing CANOPY at module level carries the ``canopy`` mark.
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
    Screens,
    fixture_views,
    invented_views,
    rollup_view,
)

from osprey.facility.panels import layout
from osprey.facility.panels.describe import light
from osprey.facility.panels.layout import render
from osprey.facility.panels.model import PanelGraph, load_graph

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
        layout.rule_length(layout.ladder(panel_graph, [light(panel_graph, c)])) for c in RULED
    ]
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", max(singles))
    doc = render(rollup_view(), graph=panel_graph, screens=Screens())
    assert layout.aggregates(doc) == 2
    _check(doc, spec_dir, rollup_view())


def test_every_view_of_every_kind_validates(panel_graph, spec_dir: Path) -> None:
    views = [*fixture_views(panel_graph), *invented_views(), rollup_view()]
    assert {v.id.view for v in views} == {"card", "set", "place", "rack", "rollup", "checkout"}
    for view in views:
        _check(render(view, graph=panel_graph, screens=Screens()), spec_dir, view)
