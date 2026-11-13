"""The example ring's full orbit response, solved for real through the SDK.

The orbit-bump application builds its response matrix with
``model_response_matrix`` over every storage-ring corrector of the example
render. The shape test runs in the parallel lane; the wall-clock budget is
marked ``wallclock`` and timed alone in the serial step, where no other worker
distorts the clock it reads.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from osprey.apps import sdk

RING = "SR"

#: Seconds the full example-ring build may take (measured 6.35 s on 2026-11-07).
BUDGET_S = 30.0


@pytest.fixture(scope="module")
def example_render(built_demo: Any) -> Path:
    """The built demo's render, read and never written."""
    return Path(built_demo.build_dir)


@pytest.fixture(scope="module")
def members(example_render: Path) -> sdk.ModelMembers:
    """The ring's members; holds no response matrix."""
    return sdk.model_members(RING, example_render)


def _asked(members: sdk.ModelMembers) -> list[str]:
    """Every corrector setpoint, asked in reverse ring order.

    Reversed so the columns cannot follow ring order or the default
    horizontal-then-vertical order by accident.
    """
    setpoints = [m.setpoint_addr for m in members.correctors if m.setpoint_addr is not None]
    return list(reversed(setpoints))


def test_the_full_example_orm_has_one_row_per_bpm_plane_and_one_column_per_asked_corrector(
    example_render, members
):
    asked = _asked(members)
    assert asked, "the example ring wires no corrector"

    orm = sdk.model_response_matrix(RING, asked, render_root=example_render)

    expected_rows = [m.position_addr for m in members.bpms]
    assert orm.model == RING
    assert orm.unsolved == {}
    assert orm.rows == expected_rows
    assert orm.columns == asked
    assert len(orm.matrix) == len(expected_rows)
    assert all(len(row) == len(asked) for row in orm.matrix)

    # Each BPM contributes its x row, then its y row, by lattice index.
    pairs = list(zip(members.bpms[0::2], members.bpms[1::2], strict=True))
    for x_row, y_row in pairs:
        assert x_row.element == y_row.element
        assert (x_row.plane, y_row.plane) == ("x", "y")
    indices = [x_row.index for x_row, _ in pairs]
    assert indices == sorted(indices)


@pytest.mark.wallclock
def test_the_full_example_orm_builds_within_its_budget(example_render, members):
    asked = _asked(members)

    started = time.perf_counter()
    orm = sdk.model_response_matrix(RING, asked, render_root=example_render)
    elapsed = time.perf_counter() - started

    assert orm.unsolved == {}
    assert len(orm.columns) == len(asked)
    assert elapsed < BUDGET_S, f"the example-ring ORM took {elapsed:.2f} s (budget {BUDGET_S} s)"
