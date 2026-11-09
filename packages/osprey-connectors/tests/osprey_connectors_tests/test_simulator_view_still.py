"""The simulator view reads a scenario's ``still`` as a list of addresses only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors.simulation.view import (
    ADDRESSES_FILE,
    ADDRESSES_SCHEMA,
    SCENARIOS_FILE,
    SCENARIOS_SCHEMA,
    SimulatorView,
    ViewSchemaError,
)


def _view(tmp_path: Path, scenario: dict[str, Any]) -> SimulatorView:
    (tmp_path / ADDRESSES_FILE).write_text(
        json.dumps({"schema": ADDRESSES_SCHEMA}), encoding="utf-8"
    )
    (tmp_path / SCENARIOS_FILE).write_text(
        json.dumps({"schema": SCENARIOS_SCHEMA, "scenarios": [scenario]}), encoding="utf-8"
    )
    return SimulatorView.open(tmp_path)


@pytest.mark.parametrize("still", ["all", [{"model": "SR"}], [1.0]], ids=["word", "map", "number"])
def test_a_still_other_than_a_list_of_addresses_is_refused(tmp_path: Path, still: Any) -> None:
    view = _view(tmp_path, {"name": "quiet", "still": still})

    with pytest.raises(ViewSchemaError, match="not a list of addresses"):
        view.scenarios()


def test_a_still_of_addresses_reads_as_it_is(tmp_path: Path) -> None:
    view = _view(tmp_path, {"name": "quiet", "still": ["SR:RF:TEMP"]})

    assert view.scenarios()[0]["still"] == ("SR:RF:TEMP",)
