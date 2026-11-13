"""The row and column positions carried from the engine to the callers.

The child's result line carries the positions the engine's answer states, as
whole numbers, and ``null`` for an engine that states none. Positions that do
not line up with the answer's rows or columns fail the solve. A line is read
back with ``line_positions``, which fills nothing in. The render is the SDK
tests' two-cell ring.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from osprey.simulation import model_request
from osprey.simulation.response import (
    COLUMN_POSITIONS_KEY,
    RESULT_TAG,
    ROW_POSITIONS_KEY,
)
from tests.apps.test_sdk_model import _ring_render
from tests.simulation.test_response_kind import _payload, _printed

ROWS = ["BPM_1:X", "BPM_2:X"]


@pytest.fixture
def ring(tmp_path: Path) -> Path:
    return _ring_render(tmp_path / "render")


def _engine(monkeypatch: pytest.MonkeyPatch, name: str, **positions: Any) -> None:
    """Install an engine whose answer states exactly the given position attributes."""
    module = types.ModuleType(name)

    def response_matrix(_deck, _wiring, _settings, correctors, _step):
        return types.SimpleNamespace(
            rows=ROWS,
            columns=correctors,
            matrix=[[1.0] * len(correctors) for _ in ROWS],
            unsolved={},
            **positions,
        )

    module.response_matrix = response_matrix  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)


class TestChildLine:
    def test_the_keys_are_the_attribute_names(self):
        assert ROW_POSITIONS_KEY == "row_positions"
        assert COLUMN_POSITIONS_KEY == "column_positions"

    def test_an_engine_without_positions_prints_null_for_both(self, ring, monkeypatch):
        _engine(monkeypatch, "engine_without_positions")

        line = _printed(_payload(ring, "engine_without_positions"))

        assert line["rows"] == ROWS
        assert line[ROW_POSITIONS_KEY] is None
        assert line[COLUMN_POSITIONS_KEY] is None

    def test_an_engine_stating_none_prints_null(self, ring, monkeypatch):
        _engine(monkeypatch, "engine_stating_none", row_positions=None, column_positions=None)

        line = _printed(_payload(ring, "engine_stating_none"))

        assert line[ROW_POSITIONS_KEY] is None
        assert line[COLUMN_POSITIONS_KEY] is None

    def test_the_engines_positions_are_printed_as_whole_numbers(self, ring, monkeypatch):
        _engine(
            monkeypatch,
            "engine_with_positions",
            row_positions=np.array([2, 11]),
            column_positions=[np.int64(5)],
        )

        line = _printed(_payload(ring, "engine_with_positions"))

        assert line[ROW_POSITIONS_KEY] == [2, 11]
        assert line[COLUMN_POSITIONS_KEY] == [5]
        assert all(type(position) is int for position in line[ROW_POSITIONS_KEY])

    def test_one_stated_and_one_not_are_carried_each_as_given(self, ring, monkeypatch):
        _engine(monkeypatch, "engine_with_rows_only", row_positions=[2, 11])

        line = _printed(_payload(ring, "engine_with_rows_only"))

        assert line[ROW_POSITIONS_KEY] == [2, 11]
        assert line[COLUMN_POSITIONS_KEY] is None

    @pytest.mark.parametrize(
        "positions",
        [
            {"row_positions": [2], "column_positions": [5]},
            {"row_positions": [2, 11], "column_positions": [5, 6]},
        ],
        ids=["rows", "columns"],
    )
    def test_positions_that_do_not_line_up_fail_the_solve_naming_the_engine(
        self, ring, monkeypatch, positions
    ):
        _engine(monkeypatch, "engine_out_of_line", **positions)

        with pytest.raises(ValueError, match="engine_out_of_line states"):
            _printed(_payload(ring, "engine_out_of_line"))


class TestLinePositions:
    def _line(self, **extra: Any) -> dict[str, Any]:
        text = json.dumps(
            {"rows": ROWS, "columns": ["COR_1:X:SP"], "matrix": [[1.0], [2.0]], "missing": {}}
            | extra
        )
        line = model_request.result_line(f"{RESULT_TAG} {text}\n")
        assert line is not None
        return line

    def test_a_line_from_before_positions_is_still_a_solved_line_with_none(self):
        assert model_request.line_positions(self._line()) == (None, None)

    def test_stated_positions_are_returned_aligned(self):
        line = self._line(row_positions=[2, 11], column_positions=[5])

        assert model_request.line_positions(line) == ([2, 11], [5])

    def test_null_positions_are_none(self):
        line = self._line(row_positions=None, column_positions=None)

        assert model_request.line_positions(line) == (None, None)

    @pytest.mark.parametrize(
        "rows",
        [[2], [2, 11, 12], [2.0, 11.0], ["2", "11"], [True, False], "2,11", {"0": 2}],
    )
    def test_positions_that_are_not_one_whole_number_per_row_are_none(self, rows):
        line = self._line(row_positions=rows, column_positions=[5])

        assert model_request.line_positions(line) == (None, [5])
