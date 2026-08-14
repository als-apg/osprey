"""Coverage for the figure spec models.

The figure JSON is read by two consumers that cannot be type-checked against
this module -- the panel's ``figure-renderer.js`` and the MCP ``get_run_figure``
projection -- so these tests pin the things a rename or a stray default would
break silently: the snake_case field names, the ``kind`` discriminator, the
`model_dump()` round trip, and the guarantee that what comes out of a figure is
valid JSON.

Pure pydantic -- no bluesky import, so this runs in the fast import-clean lane.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from osprey.services.bluesky_bridge.figure import (
    FIGURE_REASONS,
    REASON_NO_RENDER,
    REASON_PARAMS_MISMATCH,
    REASON_PLAN_IDENTITY_UNAVAILABLE,
    REASON_RENDER_FAILED,
    REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS,
    REASON_SOURCE_UNAVAILABLE,
    BarsMark,
    Figure,
    HeatmapMark,
    LinesMark,
    Panel,
    Point,
    Series,
)

# The bridge's import boundary (`app._BRIDGE_ONLY_MODULES`), restated here so
# this test file stays importable without pulling in `app.py`.
_BRIDGE_ONLY_MODULES = ("bluesky", "ophyd", "ophyd_async", "tiled")


def _lines_panel() -> Panel:
    return Panel(
        title="BPM 3 vs corrector 1",
        x_label="corrector setpoint",
        x_units="A",
        y_label="orbit",
        y_units="mm",
        annotations=["decimated from 5000 points"],
        mark=LinesMark(
            series=[
                Series(
                    label="bpm3",
                    points=[Point(x=0.0, y=1.5), Point(x=1.0, y=None), Point(x=2.0, y=-0.5)],
                    decimated=True,
                    source_points=5000,
                )
            ],
            decimated=True,
            source_points=5000,
        ),
    )


def _heatmap_panel() -> Panel:
    return Panel(
        title="response matrix",
        x_label="corrector",
        y_label="BPM",
        mark=HeatmapMark(
            x_labels=["HCM1", "HCM2"],
            y_labels=["BPM1", "BPM2", "BPM3"],
            values=[[1.0, 2.0], [3.0, None], [5.0, 6.0]],
            value_label="slope",
            value_units="mm/A",
            source_points=6,
        ),
    )


def _bars_panel() -> Panel:
    return Panel(
        title="column anomaly",
        mark=BarsMark(
            label="anomaly",
            categories=["HCM1", "HCM2"],
            values=[0.1, None],
            source_points=2,
        ),
    )


def _full_figure() -> Figure:
    return Figure(
        panels=[_lines_panel(), _heatmap_panel(), _bars_panel()],
        partial=False,
        source="tiled",
    )


# --- Shape and round trip ----------------------------------------------------


def test_full_figure_round_trips_through_model_dump() -> None:
    figure = _full_figure()
    assert Figure.model_validate(figure.model_dump()) == figure


def test_dumped_figure_is_json_serializable() -> None:
    # `json.dumps` is what Starlette does to the route's return value; the
    # browser's JSON.parse rejects the NaN/Infinity literals Python would emit.
    payload = json.dumps(_full_figure().model_dump(), allow_nan=False)
    assert '"kind": "lines"' in payload


def test_dumped_field_names_are_the_wire_contract() -> None:
    dumped = _full_figure().model_dump()
    assert set(dumped) == {"panels", "partial", "source", "reason"}
    assert set(dumped["panels"][0]) == {
        "title",
        "x_label",
        "y_label",
        "x_units",
        "y_units",
        "annotations",
        "mark",
    }
    assert set(dumped["panels"][0]["mark"]["series"][0]) == {
        "label",
        "points",
        "decimated",
        "source_points",
    }
    assert set(dumped["panels"][1]["mark"]) == {
        "kind",
        "x_labels",
        "y_labels",
        "values",
        "value_label",
        "value_units",
        "decimated",
        "source_points",
    }
    assert set(dumped["panels"][2]["mark"]) == {
        "kind",
        "label",
        "categories",
        "values",
        "decimated",
        "source_points",
    }


@pytest.mark.parametrize(
    ("panel_factory", "expected_kind", "expected_type"),
    [
        (_lines_panel, "lines", LinesMark),
        (_heatmap_panel, "heatmap", HeatmapMark),
        (_bars_panel, "bars", BarsMark),
    ],
)
def test_mark_kind_discriminates_on_revalidation(
    panel_factory: object, expected_kind: str, expected_type: type
) -> None:
    panel = panel_factory()  # type: ignore[operator]
    dumped = panel.model_dump()
    assert dumped["mark"]["kind"] == expected_kind
    assert isinstance(Panel.model_validate(dumped).mark, expected_type)


def test_unknown_mark_kind_is_rejected() -> None:
    dumped = _lines_panel().model_dump()
    dumped["mark"]["kind"] = "table"
    with pytest.raises(ValidationError):
        Panel.model_validate(dumped)


def test_extra_fields_are_forbidden() -> None:
    dumped = _lines_panel().model_dump()
    dumped["y_labl"] = "typo"
    with pytest.raises(ValidationError):
        Panel.model_validate(dumped)


# --- Honesty fields ----------------------------------------------------------


def test_partial_and_source_are_required() -> None:
    with pytest.raises(ValidationError):
        Figure.model_validate({"panels": []})
    with pytest.raises(ValidationError):
        Figure.model_validate({"panels": [], "partial": True})


def test_source_is_restricted_to_the_two_stores() -> None:
    assert Figure(panels=[], partial=True, source="live").source == "live"
    with pytest.raises(ValidationError):
        Figure.model_validate({"panels": [], "partial": True, "source": "cache"})


def test_reason_defaults_to_none_and_survives_a_round_trip() -> None:
    figure = Figure(panels=[], partial=False, source="live")
    assert figure.reason is None

    defaulted = Figure(panels=[], partial=False, source="live", reason=REASON_NO_RENDER)
    assert Figure.model_validate(defaulted.model_dump()).reason == REASON_NO_RENDER


def test_reason_vocabulary_is_complete_and_stable() -> None:
    assert FIGURE_REASONS == {
        "no_render",
        "render_failed",
        "render_not_supported_for_session_plans",
        "params_mismatch",
        "plan_identity_unavailable",
        "source_unavailable",
    }
    assert {
        REASON_NO_RENDER,
        REASON_RENDER_FAILED,
        REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS,
        REASON_PARAMS_MISMATCH,
        REASON_PLAN_IDENTITY_UNAVAILABLE,
        REASON_SOURCE_UNAVAILABLE,
    } == FIGURE_REASONS


def test_every_mark_carries_decimation_provenance() -> None:
    lines = LinesMark(source_points=10)
    heatmap = HeatmapMark(source_points=10)
    bars = BarsMark(source_points=10)
    for mark in (lines, heatmap, bars):
        assert mark.decimated is False
        assert mark.source_points == 10

    with pytest.raises(ValidationError):
        LinesMark()  # type: ignore[call-arg]  # source_points has no default: no silent zero


# --- Numeric normalization ---------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_y_becomes_a_gap(bad: float) -> None:
    assert Point(x=1.0, y=bad).y is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_heatmap_and_bar_values_become_gaps(bad: float) -> None:
    heatmap = HeatmapMark(values=[[bad, 1.0]], source_points=2)
    assert heatmap.values == [[None, 1.0]]
    bars = BarsMark(categories=["a"], values=[bad], source_points=1)
    assert bars.values == [None]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_x_is_a_validation_error(bad: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        Point(x=bad)


def test_numpy_scalars_are_coerced_to_plain_floats() -> None:
    point = Point(x=np.float32(1.5), y=np.float64(2.5))
    assert type(point.x) is float
    assert type(point.y) is float
    assert (point.x, point.y) == (1.5, 2.5)


def test_numpy_nan_is_a_gap_not_a_nan() -> None:
    # Renders compute with numpy; np.nan reaching the wire would be invalid JSON.
    point = Point(x=1.0, y=np.float64("nan"))
    assert point.y is None
    heatmap = HeatmapMark(values=[[np.float32("nan")]], source_points=1)
    assert heatmap.values == [[None]]


def test_ints_are_accepted_as_coordinates() -> None:
    point = Point(x=3, y=4)
    assert (point.x, point.y) == (3.0, 4.0)


def test_non_numeric_values_still_fail_validation() -> None:
    with pytest.raises(ValidationError):
        Point(x="over there")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Point(x=1.0, y=object())  # type: ignore[arg-type]


def test_nan_string_is_normalized_rather_than_parsed_into_a_nan() -> None:
    assert Point(x=1.0, y="nan").y is None  # type: ignore[arg-type]
    assert math.isclose(Point(x=1.0, y="2.5").y or 0.0, 2.5)  # type: ignore[arg-type]


# --- Import discipline -------------------------------------------------------


def test_importing_figure_does_not_import_the_runengine_stack() -> None:
    # Must run in a fresh interpreter: other tests in this suite legitimately
    # import bluesky, so an in-process `sys.modules` check would depend on test
    # order rather than on what `figure.py` imports.
    failures: dict[str, str] = {}
    for module in _BRIDGE_ONLY_MODULES:
        code = (
            "import osprey.services.bluesky_bridge.figure, sys; "
            f"assert {module!r} not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if result.returncode != 0:
            failures[module] = result.stderr

    assert not failures, "\n".join(
        f"importing osprey.services.bluesky_bridge.figure leaked a top-level "
        f"import of {module!r}:\n{stderr}"
        for module, stderr in failures.items()
    )
