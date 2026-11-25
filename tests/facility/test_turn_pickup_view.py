"""A pickup's table travels with the deck into the served view.

A pickup statement names its table relative to the directory of the model's
deck. A render's simulator view holds a copy of the deck and nothing else of
that directory, so the view carries each table beside the copy and its
settings name what it carries. The turn-by-turn fixture is built and rendered
with tables stated outside the deck's directory, and the view is then opened
the way a served virtual accelerator opens it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from osprey.facility.views.outputs import render_facility_outputs
from osprey.simulation.engines import pyat as engine
from osprey_connectors.simulation.composite import Composite
from osprey_facility.build import build_facility
from osprey_facility.errors import FacilityBuildError
from tests._builds import FRAMEWORK_CHECKS
from tests.facility._turn_by_turn_fixture import (
    ARMS,
    FIXTURE,
    MONITORS,
    REQUEST,
    RING,
    TURN_SUM,
    TURN_X,
)
from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG

PLANES: dict[str, Any] = {
    "x": {"plus": [1, 4], "minus": [2, 3]},
    "y": {"plus": [1, 2], "minus": [3, 4]},
}
#: Stated from the deck's directory, ``decks/``: a file of another directory.
FLAT = "../tables/flat.json"
OTHER = "../tables/other/flat.json"


def _table(path: Path, slope: float) -> bytes:
    grid = [-0.004, -0.002, 0.0, 0.002, 0.004]
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    values = [[[1.0 + slope * (cx * x + cy * y) for x in grid] for y in grid] for cx, cy in corners]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"x": grid, "y": grid, "buttons": values}))
    return path.read_bytes()


def _statement(table: str, **more: Any) -> dict[str, Any]:
    return {"shape": "table", "table": table, "buttons": 4, **PLANES, **more}


def _tree(root: Path, **block: Any) -> Path:
    facility_dir = root / "data" / "facility"
    shutil.copytree(FIXTURE, facility_dir)
    models = yaml.safe_load((facility_dir / "models.yaml").read_text())
    (ring,) = [model for model in models if model["name"] == RING]
    ring["settings"]["pyat"]["turn_by_turn"].update(block)
    (facility_dir / "models.yaml").write_text(yaml.safe_dump(models, sort_keys=False))
    return facility_dir


def _render(root: Path, facility_dir: Path) -> Path:
    document = build_facility(
        facility_dir, project_name="turn-by-turn", extra_checks=FRAMEWORK_CHECKS
    )
    render_dir = root / "build"
    render_dir.mkdir()
    render_facility_outputs(
        render_dir, document, IN_PROCESS_RENDER_CONFIG, facility_dir, omitted_reported=set()
    )
    return render_dir / "data" / "simulator"


def _turn_block(view: Path) -> dict[str, Any]:
    variables = json.loads((view / "variables.json").read_text())
    (ring,) = [model for model in variables["models"] if model["name"] == RING]
    block: dict[str, Any] = ring["settings"]["pyat"]["turn_by_turn"]
    return block


def test_the_view_carries_the_table_beside_the_deck_and_names_the_copy(tmp_path: Path) -> None:
    facility_dir = _tree(tmp_path, pickup=_statement(FLAT))
    stated = _table(facility_dir / "tables" / "flat.json", 60.0)

    view = _render(tmp_path, facility_dir)

    assert (view / "decks" / f"{RING}.flat.json").read_bytes() == stated
    assert _turn_block(view)["pickup"] == _statement(f"{RING}.flat.json")
    # The facility file itself still names the table where the author put it.
    authored = yaml.safe_load((facility_dir / "models.yaml").read_text())
    (ring,) = [model for model in authored if model["name"] == RING]
    assert ring["settings"]["pyat"]["turn_by_turn"]["pickup"]["table"] == FLAT


def test_the_view_opens_without_the_tree_it_was_rendered_from(tmp_path: Path) -> None:
    """Moved away from its facility tree, the view still serves a shot through the table."""
    facility_dir = _tree(tmp_path, pickup=_statement(FLAT))
    _table(facility_dir / "tables" / "flat.json", 60.0)
    view = tmp_path / "elsewhere" / "simulator"
    shutil.copytree(_render(tmp_path, facility_dir), view)
    shutil.rmtree(facility_dir)

    now = [1000.0]
    composite = Composite(view, clock=lambda: now[0], evolve=True, model_log=False)
    for values in ({ARMS[monitor]: 1.0 for monitor in MONITORS}, {REQUEST: 1.0}):
        now[0] += 1.0
        composite.set(values)
    now[0] += 0.1
    composite.advance()

    assert composite.status(RING) == "ok"
    sums = np.asarray(composite.get(TURN_SUM[MONITORS[0]]), dtype=float)
    positions = np.asarray(composite.get(TURN_X[MONITORS[0]]), dtype=float)
    assert sums.max() > 0.5
    assert np.abs(positions).max() > 1e-4


def test_two_tables_of_one_name_are_carried_apart(tmp_path: Path) -> None:
    pickups = [
        {"elements": [MONITORS[0]], **_statement(FLAT)},
        {"elements": [MONITORS[1]], **_statement(OTHER)},
        {"elements": [MONITORS[2]], **_statement(FLAT, interpolation="bicubic")},
    ]
    facility_dir = _tree(tmp_path, pickups=pickups)
    first = _table(facility_dir / "tables" / "flat.json", 60.0)
    second = _table(facility_dir / "tables" / "other" / "flat.json", 40.0)

    view = _render(tmp_path, facility_dir)

    carried = [item["table"] for item in _turn_block(view)["pickups"]]
    assert carried == [f"{RING}.flat.json", f"{RING}.2.flat.json", f"{RING}.flat.json"]
    assert (view / "decks" / carried[0]).read_bytes() == first
    assert (view / "decks" / carried[1]).read_bytes() == second
    assert sorted(path.name for path in (view / "decks").glob(f"{RING}.*flat.json")) == sorted(
        set(carried)
    )


def test_a_round_pickup_and_a_ring_without_one_carry_nothing(tmp_path: Path) -> None:
    round_pickup = {
        "shape": "round",
        "radius": 0.016,
        "buttons": [45.0, 135.0, 225.0, 315.0],
        "half_width": 14.0,
        **PLANES,
    }
    with_pickup = _render(tmp_path / "a", _tree(tmp_path / "a", pickup=round_pickup))
    without = _render(tmp_path / "b", _tree(tmp_path / "b"))

    assert _turn_block(with_pickup)["pickup"] == round_pickup
    for view in (with_pickup, without):
        assert sorted(path.name for path in (view / "decks").iterdir()) == [f"{RING}.json"]


def test_a_missing_table_stops_the_build_before_a_view_is_written(tmp_path: Path) -> None:
    facility_dir = _tree(tmp_path, pickup=_statement(FLAT))

    with pytest.raises(FacilityBuildError) as caught:
        _render(tmp_path, facility_dir)

    assert "pyat.turn_by_turn.pickup.table ../tables/flat.json" in str(caught.value)
    assert "cannot be read" in str(caught.value)
    assert not (tmp_path / "build").exists()


def test_the_engine_names_each_table_once_and_leaves_the_settings_alone() -> None:
    settings = {
        "pyat": {
            "radiation": True,
            "turn_by_turn": {
                "injection_element": "INJ",
                "pickup": _statement(FLAT),
                "pickups": [{"elements": ["BPM01"], **_statement(OTHER)}, "not a statement"],
            },
        }
    }
    kept = json.loads(json.dumps(settings))
    asked: list[str] = []

    def carry(stated: str) -> str:
        asked.append(stated)
        return f"carried-{len(asked)}.json"

    served = engine.served_settings(settings, carry)

    assert asked == [FLAT, OTHER]
    assert settings == kept
    block = served["pyat"]["turn_by_turn"]
    assert block["pickup"]["table"] == "carried-1.json"
    assert block["pickups"][0]["table"] == "carried-2.json"
    assert block["pickups"][1] == "not a statement"
    assert served["pyat"]["radiation"] is True
    # Nothing to carry: the settings come back as they were given.
    for plain in ({}, {"pyat": {}}, {"pyat": {"turn_by_turn": {"injection_element": "INJ"}}}):
        assert engine.served_settings(plain, carry) is plain
    assert asked == [FLAT, OTHER]
