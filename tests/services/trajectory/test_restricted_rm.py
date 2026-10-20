"""The build-time model matrix restricted to the current selection.

Every test builds the shared tool from a render carrying the vendored golden
case, the way the service does, then restricts it by BPM and corrector names.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from osprey.services.trajectory import step
from osprey.services.trajectory.step import (
    SELECTION_UNKNOWN,
    STEP_FAILED,
    RmSummary,
    StepRefused,
    TrajectoryTool,
    build_tool,
    family_index,
    per_call_settings,
    restricted_tool,
    rm_summary,
)

GOLDENS = Path(__file__).parent / "data" / "trajectory_goldens.npz"
CASE = "example_sr_tikhonov"
MODEL = "SR"


@pytest.fixture(scope="module")
def goldens() -> Any:
    with np.load(GOLDENS, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _case(goldens: Any, field: str) -> Any:
    return goldens[f"{CASE}/{field}"]


@pytest.fixture
def shared(tmp_path: Path, goldens: Any) -> TrajectoryTool:
    """The shared tool built from a render carrying the golden case."""
    root = tmp_path / "build"
    view = root / "data" / "pyaml" / MODEL
    view.mkdir(parents=True)
    rm = {
        "type": "pyaml.tuning_tools.orbit_response_matrix_data",
        "matrix": _case(goldens, "matrix").tolist(),
        "observable_names": _case(goldens, "observable_names").tolist(),
        "variable_names": _case(goldens, "variable_names").tolist(),
        "observable_planes": _case(goldens, "observable_planes").tolist(),
        "variable_planes": _case(goldens, "variable_planes").tolist(),
    }
    rows = _case(goldens, "bpm_index").tolist()
    hcm = _case(goldens, "variable_names").tolist()[:72]
    index = {
        "bpm_index": [*rows, *rows],
        "hcorr_index": _case(goldens, "hcorr_index").tolist(),
        "vcorr_index": _case(goldens, "vcorr_index").tolist(),
        "unsolved": {hcm[5]: "the kick loses the particle"},
    }
    # The golden column is not zero; an unsolved corrector's is, so zero it.
    matrix = rm["matrix"]
    for row in matrix:
        row[5] = 0.0
    (view / "trajectory_rm.json").write_text(json.dumps(rm), encoding="utf-8")
    (view / "trajectory_rm.index.json").write_text(json.dumps(index), encoding="utf-8")
    measurement = {
        "alpha": float(_case(goldens, "alpha")),
        "n_discard": int(_case(goldens, "n_discard")),
        "gain": float(_case(goldens, "gain")),
    }
    facility = {"models": [{"name": MODEL, "measurement": measurement}]}
    (root / "facility.json").write_text(json.dumps(facility), encoding="utf-8")
    tool = build_tool(root, MODEL)
    assert isinstance(tool, TrajectoryTool), tool
    return tool


def _snapshot(tool: TrajectoryTool) -> Any:
    data = tool.tool.response_matrix
    return (
        np.asarray(data.matrix).copy(),
        list(data.observable_names),
        list(data.variable_names),
        tool.layout,
        tool.bpm_index,
        tool.hcorr_index,
        tool.vcorr_index,
        dict(tool.unsolved),
        tool.tool._settings,
    )


def _same(a: Any, b: Any) -> None:
    assert np.array_equal(a[0], b[0])
    assert a[1:] == b[1:]


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    step._RESTRICTED.clear()


# -- dimensions ----------------------------------------------------------------


def test_restricting_then_restoring_the_full_set_yields_the_original_dimensions(
    shared: TrajectoryTool,
) -> None:
    layout = shared.layout
    bpms = list(layout.bpm_names_h)
    full = rm_summary(shared)
    assert full == RmSummary(source="model", rows=144, h_cols=72, v_cols=72)

    excluded = {bpms[3], bpms[40], layout.hcorr_names[7], layout.vcorr_names[0]}
    restricted = restricted_tool(shared, excluded)
    assert rm_summary(restricted) == RmSummary(source="model", rows=140, h_cols=71, v_cols=71)
    assert restricted.layout.n_bpm == 70
    assert bpms[3] not in restricted.layout.bpm_names_h
    assert bpms[3].replace(":X", ":Y") not in restricted.layout.bpm_names_v
    assert layout.hcorr_names[7] not in restricted.layout.hcorr_names
    assert layout.vcorr_names[0] not in restricted.layout.vcorr_names

    restored = restricted_tool(shared, set())
    assert rm_summary(restored) == full
    assert restored.layout == layout
    assert restored.bpm_index == shared.bpm_index
    assert restored.hcorr_index == shared.hcorr_index
    assert restored.vcorr_index == shared.vcorr_index


def test_the_summary_publishes_source_rows_and_columns(shared: TrajectoryTool) -> None:
    assert rm_summary(shared).as_dict() == {
        "source": "model",
        "rows": 144,
        "h_cols": 72,
        "v_cols": 72,
    }


def test_the_restricted_sidecar_is_sliced_with_the_matrix(shared: TrajectoryTool) -> None:
    layout = shared.layout
    drop = layout.bpm_names_h[10]
    hcm = layout.hcorr_names[5]  # the unsolved one
    restricted = restricted_tool(shared, {drop, hcm, layout.vcorr_names[2]})
    n = 72
    keep = [i for i in range(n) if i != 10]
    assert restricted.bpm_index == tuple(
        shared.bpm_index[i] for i in [*keep, *(n + i for i in keep)]
    )
    assert restricted.hcorr_index == tuple(v for i, v in enumerate(shared.hcorr_index) if i != 5)
    assert restricted.vcorr_index == tuple(v for i, v in enumerate(shared.vcorr_index) if i != 2)
    assert dict(restricted.unsolved) == {}
    assert dict(restricted_tool(shared, {drop}).unsolved) == dict(shared.unsolved)
    # The restricted matrix is the shared one with the rows and columns dropped.
    full = np.asarray(shared.tool.response_matrix.matrix)
    rows = [*keep, *(n + i for i in keep)]
    cols = [c for c in range(144) if c not in (5, 72 + 2)]
    assert np.array_equal(np.asarray(restricted.tool.response_matrix.matrix), full[rows][:, cols])
    assert restricted.model == shared.model


def test_the_restricted_tool_is_built_with_neutral_defaults(shared: TrajectoryTool) -> None:
    restricted = restricted_tool(shared, {shared.layout.bpm_names_h[0]})
    settings = restricted.tool._settings
    assert (settings.alpha, settings.n_discard, settings.gain) == (0.0, 0, 1.0)
    # The configured values travel beside it, to be passed per call.
    assert (restricted.alpha, restricted.n_discard, restricted.gain) == (
        shared.alpha,
        shared.n_discard,
        shared.gain,
    )


# -- the shared tool is never touched -------------------------------------------


def test_the_shared_tool_is_unchanged_after_a_restricted_solve(
    shared: TrajectoryTool, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _snapshot(shared)
    loads: list[Any] = []
    monkeypatch.setattr(type(shared.tool), "load", lambda self, path: loads.append(path))
    layout = shared.layout
    excluded = {layout.bpm_names_h[1], layout.bpm_names_h[50], layout.hcorr_names[0]}
    restricted = restricted_tool(shared, excluded)
    assert restricted.tool is not shared.tool
    x = [0.001 * math.sin(i) for i in range(restricted.layout.n_bpm)]
    y = [0.001 * math.cos(i) for i in range(restricted.layout.n_bpm)]
    result = restricted.tool.correct(
        x,
        y,
        None,
        bpm_index=list(restricted.bpm_index),
        hcorr_index=list(restricted.hcorr_index),
        vcorr_index=list(restricted.vcorr_index),
        **per_call_settings(restricted, alpha=1e-3, n_discard=4, gain=0.5),
    )
    assert len(result.names) == 71 + 72
    assert loads == []
    _same(_snapshot(shared), before)


# -- the cache -----------------------------------------------------------------


def test_a_restriction_is_cached_by_exclusion_set_and_matrix(shared: TrajectoryTool) -> None:
    name = shared.layout.bpm_names_h[2]
    first = restricted_tool(shared, {name})
    assert restricted_tool(shared, frozenset([name])) is first
    assert restricted_tool(shared, [name, name]) is first
    other = restricted_tool(shared, {shared.layout.bpm_names_h[3]})
    assert other is not first


def test_the_full_selection_is_the_shared_tool(shared: TrajectoryTool) -> None:
    assert restricted_tool(shared, ()) is shared


# -- refusals ------------------------------------------------------------------


def test_a_selection_outside_the_build_is_refused_by_name(shared: TrajectoryTool) -> None:
    with pytest.raises(StepRefused) as caught:
        restricted_tool(shared, {"NOPE:BPM", shared.layout.bpm_names_h[0], "ALSO:NOT"})
    assert caught.value.code == SELECTION_UNKNOWN
    assert caught.value.status == 409
    assert "ALSO:NOT" in caught.value.detail
    assert "NOPE:BPM" in caught.value.detail
    assert shared.layout.bpm_names_h[0] not in caught.value.detail


def test_a_y_row_name_is_no_bpm_name(shared: TrajectoryTool) -> None:
    with pytest.raises(StepRefused) as caught:
        restricted_tool(shared, {shared.layout.bpm_names_v[0]})
    assert caught.value.code == SELECTION_UNKNOWN


def test_a_restriction_with_no_live_corrector_is_refused(shared: TrajectoryTool) -> None:
    layout = shared.layout
    with pytest.raises(StepRefused) as caught:
        restricted_tool(shared, {*layout.hcorr_names, *layout.vcorr_names})
    assert caught.value.code == STEP_FAILED
    assert MODEL in caught.value.detail


# -- per-call settings ---------------------------------------------------------


def test_a_per_call_cutoff_larger_than_the_restricted_rank_is_clamped(
    shared: TrajectoryTool,
) -> None:
    bpms = shared.layout.bpm_names_h
    restricted = restricted_tool(shared, {bpms[0], bpms[1]})
    assert restricted.layout.L == 140
    settings = per_call_settings(restricted, alpha=2e-4, n_discard=143, gain=1.0)
    assert settings == {"alpha": 2e-4, "n_discard": 140, "gain": 1.0}
    inverse = restricted.tool.pseudo_inverse(**settings)
    assert int(inverse.n_discard) == 140
    # A cutoff within the rank passes through unchanged.
    assert per_call_settings(restricted, alpha=0.0, n_discard=3, gain=0.7)["n_discard"] == 3
    assert per_call_settings(restricted, alpha=0.0, n_discard=-2, gain=0.7)["n_discard"] == 0


# -- family index --------------------------------------------------------------


def test_excluded_indices_remain_gaps_in_the_family_index(shared: TrajectoryTool) -> None:
    layout = shared.layout
    index = family_index(shared)
    assert index["bpm"][layout.bpm_names_h[0]] == 1
    assert index["bpm"][layout.bpm_names_h[71]] == 72
    assert index["hcorr"][layout.hcorr_names[3]] == 4
    assert index["vcorr"][layout.vcorr_names[0]] == 1

    restricted = restricted_tool(
        shared, {layout.bpm_names_h[1], layout.bpm_names_h[4], layout.vcorr_names[0]}
    )
    kept = [index["bpm"][name] for name in restricted.layout.bpm_names_h]
    assert kept[:4] == [1, 3, 4, 6]
    assert 2 not in kept and 5 not in kept
    assert [index["vcorr"][name] for name in restricted.layout.vcorr_names][:2] == [2, 3]
