"""The simulator view reads a model's inputs, and still reads a released ``/3`` view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    ADDRESSES_SCHEMA,
    VARIABLES_FILE,
    VARIABLES_SCHEMA,
    SimulatorView,
    ViewSchemaError,
)

RELEASED_SCHEMA = "osprey.facility.simulator/3"


def _view(tmp_path: Path, schema: str, models: list[dict[str, Any]]) -> SimulatorView:
    (tmp_path / ADDRESSES_FILE).write_text(
        json.dumps({"schema": ADDRESSES_SCHEMA}), encoding="utf-8"
    )
    (tmp_path / VARIABLES_FILE).write_text(
        json.dumps({"schema": schema, "code": "AR", "models": models, "channels": []}),
        encoding="utf-8",
    )
    return SimulatorView.open(tmp_path)


def _model(name: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "engine": "pyat", "served": True, "settings": {}, "wiring": [], **extra}


def test_the_writer_identifier_is_four() -> None:
    assert VARIABLES_SCHEMA == "osprey.facility.simulator/4"


def test_inputs_round_trip(tmp_path: Path) -> None:
    inputs = [
        {"address": "INJ:BEAM:X", "engine": {"take": "x", "unit": "m"}},
        {"address": "INJ:BEAM:Y", "engine": {"take": "y"}},
    ]
    view = _view(tmp_path, VARIABLES_SCHEMA, [_model("AR", inputs=inputs), _model("INJ")])

    model = view.model("AR")
    assert [dict(item) for item in model.inputs] == [
        {"address": "INJ:BEAM:X", "engine": {"take": "x", "unit": "m"}},
        {"address": "INJ:BEAM:Y", "engine": {"take": "y"}},
    ]
    assert model.inputs[0]["address"] == "INJ:BEAM:X"
    assert model.inputs[0]["engine"]["unit"] == "m"
    assert view.model("INJ").inputs == ()


def test_inputs_are_read_only(tmp_path: Path) -> None:
    inputs = [{"address": "INJ:BEAM:X", "engine": {"take": "x"}}]
    view = _view(tmp_path, VARIABLES_SCHEMA, [_model("AR", inputs=inputs)])

    record = view.model("AR").inputs[0]
    with pytest.raises(TypeError):
        record["address"] = "OTHER"  # type: ignore[index]
    with pytest.raises(TypeError):
        record["engine"]["take"] = "y"  # type: ignore[index]


def test_a_model_without_inputs_has_none(tmp_path: Path) -> None:
    view = _view(tmp_path, VARIABLES_SCHEMA, [_model("AR")])

    assert view.model("AR").inputs == ()


def test_a_released_view_loads_with_no_inputs(tmp_path: Path) -> None:
    view = _view(tmp_path, RELEASED_SCHEMA, [_model("AR"), _model("INJ")])

    assert view.code == "AR"
    assert [model.name for model in view.models()] == ["AR", "INJ"]
    assert all(model.inputs == () for model in view.models())


@pytest.mark.parametrize(
    "schema",
    ["osprey.facility.simulator/2", "osprey.facility.simulator/5", None],
    ids=["older", "newer", "absent"],
)
def test_an_unknown_identifier_is_refused(tmp_path: Path, schema: str | None) -> None:
    view = _view(tmp_path, schema, [_model("AR")])  # type: ignore[arg-type]

    with pytest.raises(ViewSchemaError, match="rebuild with osprey build") as raised:
        view.models()
    assert repr(VARIABLES_SCHEMA) in str(raised.value)
