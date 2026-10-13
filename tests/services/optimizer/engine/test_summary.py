"""Tests for compute_result_summary."""

from __future__ import annotations

import json
from typing import Any

import polars as pl
import pytest

from osprey.services.optimizer.api.summary import (
    SummaryContext,
    _generate_text_summary,
    _NarrativeInputs,
    compute_result_summary,
)
from osprey.services.optimizer.models.api import ResultSummary
from osprey.services.optimizer.models.types import Direction


def _var_meta(
    description: str = "Var 1", units: str = "V", step_size: float = 0.1
) -> dict[str, object]:
    return {"description": description, "units": units, "step_size": step_size}


def _obs_meta(
    description: str = "Objective",
    units: str = "nC",
    direction: str = "MAXIMIZE",
    target_value: float | None = 5.0,
) -> dict[str, object]:
    return {
        "description": description,
        "units": units,
        "direction": direction,
        "target_value": target_value,
    }


def _make_data(
    obj_values: list[float | None], var_values: list[float] | None = None
) -> pl.DataFrame:
    n = len(obj_values)
    var = var_values if var_values is not None else [float(i) for i in range(n)]
    return pl.DataFrame({"var1": var, "obj": obj_values})


def _summary(data: pl.DataFrame, **overrides: Any) -> ResultSummary:
    """compute_result_summary over the standard single-var (``var1`` / ``obj``) context."""
    ctx: dict[str, Any] = {
        "objective_name": "obj",
        "variable_names": ["var1"],
        "direction": Direction.MAXIMIZE,
    }
    return compute_result_summary(data=data, context=SummaryContext(**{**ctx, **overrides}))


class TestComputeResultSummary:
    def test_basic_sorting_and_deltas(self) -> None:
        """Points sort best-first; delta_vs_initial covers every value key while
        delta_vs_target only covers the keys a target was given for."""
        data = _make_data([0.5, 0.9, 0.7])
        result = _summary(data, initial_point={"var1": 0.0, "obj": 0.5}, target_value={"obj": 1.0})
        assert len(result.best_points) == 3
        best = result.best_points[0]
        # Best row is (var1=1.0, obj=0.9); exact delta maps, not just spot keys.
        assert best.values == pytest.approx({"var1": 1.0, "obj": 0.9})
        assert best.delta_vs_initial == pytest.approx({"var1": 1.0, "obj": 0.4})
        assert best.delta_vs_target == pytest.approx({"obj": -0.1})

    @pytest.mark.parametrize(
        ("obj_values", "direction", "expected_objs"),
        [
            pytest.param(
                [float(i) for i in range(10)],
                Direction.MAXIMIZE,
                [9.0, 8.0, 7.0, 6.0, 5.0],
                id="top_n_caps_at_5",
            ),
            pytest.param([0.5, 0.9], Direction.MAXIMIZE, [0.9, 0.5], id="fewer_than_n"),
            pytest.param(
                [0.5, None, 0.9], Direction.MAXIMIZE, [0.9, 0.5], id="null_objective_rows_filtered"
            ),
            pytest.param([None, None, None], Direction.MAXIMIZE, [], id="all_null_objective"),
            pytest.param(
                [0.5, 0.1, 0.9], Direction.MINIMIZE, [0.1, 0.5, 0.9], id="minimize_sorts_ascending"
            ),
        ],
    )
    def test_best_points_selection(
        self,
        obj_values: list[float | None],
        direction: Direction,
        expected_objs: list[float],
    ) -> None:
        """Best points are sorted by direction, capped at TOP_N, with null rows filtered."""
        result = _summary(_make_data(obj_values), direction=direction)
        assert [p.values["obj"] for p in result.best_points] == expected_objs

    def test_empty_data(self) -> None:
        data = pl.DataFrame({"var1": [], "obj": []})
        result = _summary(data, initial_point={"var1": 1.0, "obj": 0.5})
        assert result.best_points == []
        assert result.initial_point == {"var1": 1.0, "obj": 0.5}

    @pytest.mark.parametrize(
        ("initial_point", "expect_delta_initial_none"),
        [
            pytest.param(None, True, id="no_initial_point"),
            pytest.param({"var1": 0.0, "obj": 0.3}, False, id="no_target_value"),
        ],
    )
    def test_deltas_absent_without_reference(
        self,
        initial_point: dict[str, float] | None,
        expect_delta_initial_none: bool,
    ) -> None:
        """Without a target_value there is no delta_vs_target; delta_vs_initial
        exists only when an initial_point was supplied."""
        result = _summary(_make_data([0.5]), initial_point=initial_point)
        best = result.best_points[0]
        assert best.delta_vs_target is None
        assert (best.delta_vs_initial is None) is expect_delta_initial_none

    @pytest.mark.parametrize(
        ("direction_kwargs", "expected_exc", "match"),
        [
            pytest.param({}, TypeError, None, id="missing_direction"),
            pytest.param(
                {"direction": "INVALID"}, ValueError, "Invalid direction", id="invalid_direction"
            ),
        ],
    )
    def test_bad_direction_raises(
        self, direction_kwargs: dict[str, Any], expected_exc: type[Exception], match: str | None
    ) -> None:
        with pytest.raises(expected_exc, match=match):
            SummaryContext(objective_name="obj", variable_names=["var1"], **direction_kwargs)


class TestNonFiniteValuesAreJsonSafe:
    """A failed read surfaces as NaN; the summary must stay JSON-serializable.

    Starlette renders responses with ``json.dumps(allow_nan=False)``, so any NaN/Inf
    in the body would make the status endpoint return 500 — and the result would be
    permanently unreadable. Each test asserts strict-JSON serializability directly.
    """

    @staticmethod
    def _assert_json_safe(result: object) -> None:
        # allow_nan=False mirrors Starlette's JSONResponse.render — raises on NaN/Inf.
        json.dumps(result.model_dump(), allow_nan=False)  # type: ignore[attr-defined]

    def test_nan_objective_rows_filtered_like_nulls(self) -> None:
        data = _make_data([0.5, float("nan"), 0.9])
        result = _summary(data)
        assert len(result.best_points) == 2
        assert result.best_points[0].values["obj"] == 0.9
        self._assert_json_safe(result)

    def test_nan_in_initial_point_is_scrubbed(self) -> None:
        data = _make_data([0.5, 0.9])
        result = _summary(
            data,
            initial_point={"var1": 0.0, "obj": float("nan")},
            target_value={"obj": float("inf")},
        )
        # The non-finite objective/target entries are dropped, not nulled.
        assert "obj" not in (result.initial_point or {})
        assert "obj" not in (result.target_value or {})
        assert result.initial_objective_value is None
        self._assert_json_safe(result)

    def test_all_nan_objective_with_nan_initial_serializes(self) -> None:
        # The unreadable-objective case: every value NaN, including the initial read.
        data = _make_data([float("nan"), float("nan")])
        result = _summary(data, initial_point={"var1": 1.0, "obj": float("nan")})
        assert result.best_points == []
        assert result.best_objective_value is None
        self._assert_json_safe(result)


class TestEnrichedSummary:
    """Tests for enriched summary fields: annotations, context, text, tied points."""

    def _enriched_context(self, **overrides: object) -> SummaryContext:
        defaults: dict[str, object] = {
            "objective_name": "obj",
            "variable_names": ["var1"],
            "direction": Direction.MAXIMIZE,
            "initial_point": {"var1": 0.0, "obj": 0.3},
            "target_value": {"obj": 5.0},
            "variable_metadata": {"var1": _var_meta()},
            "observable_metadata": {"obj": _obs_meta()},
        }
        defaults.update(overrides)
        return SummaryContext(**defaults)  # type: ignore[arg-type]

    def test_annotated_values_present(self) -> None:
        data = _make_data([0.5, 0.9], var_values=[1.0, 2.0])
        result = compute_result_summary(data=data, context=self._enriched_context())

        ann = result.best_points[0].annotated_values
        assert ann is not None
        assert "var1" in ann
        assert ann["var1"].description == "Var 1"
        assert ann["var1"].units == "V"
        assert ann["obj"].description == "Objective"
        assert ann["obj"].units == "nC"
        # Best point (var1=2.0, obj=0.9) against the initial point (0.0, 0.3).
        assert ann["var1"].delta_vs_initial == pytest.approx(2.0)
        assert ann["obj"].delta_vs_initial == pytest.approx(0.6)

    def test_annotated_values_rounding(self) -> None:
        data = _make_data([0.12345], var_values=[3.14159])
        ctx = self._enriched_context(variable_metadata={"var1": _var_meta(step_size=0.01)})
        result = compute_result_summary(data=data, context=ctx)

        ann = result.best_points[0].annotated_values
        assert ann is not None
        assert ann["var1"].value == 3.14

    @pytest.mark.parametrize(
        ("obj_values", "expected_fields"),
        [
            pytest.param(
                [0.3, 0.5, 0.9],
                {
                    "objective_name": "obj",
                    "objective_description": "Objective",
                    "direction": "MAXIMIZE",
                    "total_iterations": 3,
                    "best_objective_value": 0.9,
                    "initial_objective_value": 0.3,
                    "improved_over_initial": True,
                    # [0.3, 0.5, 0.9] is tie-free, so no tie bookkeeping is emitted.
                    "tied_count": None,
                    "tied_note": None,
                },
                id="context_fields",
            ),
            pytest.param([0.1], {"improved_over_initial": False}, id="no_improvement"),
            pytest.param(
                [2.5],
                {
                    "target_value_scalar": 5.0,
                    "target_units": "nC",
                    "target_achievement_pct": pytest.approx(50.0),
                },
                id="target_achievement_pct",
            ),
        ],
    )
    def test_context_fields(
        self, obj_values: list[float | None], expected_fields: dict[str, Any]
    ) -> None:
        data = _make_data(obj_values)
        result = compute_result_summary(data=data, context=self._enriched_context())

        for field, expected in expected_fields.items():
            assert getattr(result, field) == expected

    @pytest.mark.parametrize(
        ("obj_values", "var_values", "expected_substrings"),
        [
            pytest.param(
                [0.5, 0.9],
                [1.0, 2.0],
                ["Objective", "MAXIMIZE"],
                id="describes_objective_and_direction",
            ),
            pytest.param(
                [0.5, 0.9, 0.7, 0.6, 0.8], None, ["5 iterations"], id="reports_iteration_count"
            ),
        ],
    )
    def test_text_summary(
        self,
        obj_values: list[float | None],
        var_values: list[float] | None,
        expected_substrings: list[str],
    ) -> None:
        data = _make_data(obj_values, var_values)
        result = compute_result_summary(data=data, context=self._enriched_context())

        assert result.text_summary is not None
        for substring in expected_substrings:
            assert substring in result.text_summary

    @pytest.mark.parametrize(
        ("obj_values", "var_values", "obs_meta_override", "expected_tied", "expected_note"),
        [
            pytest.param(
                [0.9, 0.9, 0.9, 0.5],
                [1.0, 2.0, 3.0, 4.0],
                {"obj": _obs_meta()},
                3,
                "3 points tied at objective value 0.9; showing first.",
                id="exact_value_ties",
            ),
            pytest.param(
                [0.90, 0.91, 0.10],
                [0.0, 1.0, 2.0],
                {"obj": {"step_size": 0.05, "units": "nC"}},
                2,
                "2 points tied at objective value 0.91; showing first.",
                id="within_step_ties",
            ),
        ],
    )
    def test_tied_points_collapsed(
        self,
        obj_values: list[float | None],
        var_values: list[float],
        obs_meta_override: dict[str, object],
        expected_tied: int,
        expected_note: str,
    ) -> None:
        """Top points that are equal — exactly, or within the objective's step —
        collapse to a single best point with a tie note."""
        data = _make_data(obj_values, var_values)
        ctx = self._enriched_context(observable_metadata=obs_meta_override)
        result = compute_result_summary(data=data, context=ctx)

        assert result.tied_count == expected_tied
        assert result.tied_note == expected_note
        assert len(result.best_points) == 1

    def test_no_metadata_still_works(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without metadata, annotated_values is None but the rest still works — and
        the absent observable_metadata surfaces a warning, not a silent degrade."""
        data = _make_data([0.5, 0.9])
        ctx = SummaryContext(
            objective_name="obj",
            variable_names=["var1"],
            direction=Direction.MAXIMIZE,
            initial_point={"var1": 0.0, "obj": 0.3},
        )
        with caplog.at_level("WARNING", logger="osprey.services.optimizer.api.summary"):
            result = compute_result_summary(data=data, context=ctx)

        assert result.best_points[0].annotated_values is None
        assert result.best_points[0].values["obj"] == 0.9
        assert result.text_summary is not None
        assert any(
            "observable_metadata is None" in r.message and "'obj'" in r.message
            for r in caplog.records
        )

    def test_direction_from_context(self) -> None:
        """Direction in context controls sorting."""
        data = _make_data([0.5, 0.1, 0.9])
        ctx = self._enriched_context(direction="MINIMIZE")
        result = compute_result_summary(data=data, context=ctx)

        assert result.direction == "MINIMIZE"
        assert result.best_points[0].values["obj"] == 0.1

    def test_empty_data_context_fields(self) -> None:
        data = pl.DataFrame({"var1": [], "obj": []})
        result = compute_result_summary(data=data, context=self._enriched_context())

        assert result.objective_name == "obj"
        assert result.direction == "MAXIMIZE"
        assert result.total_iterations == 0
        assert result.best_objective_value is None

    def test_build_annotated_key_missing_from_initial(self) -> None:
        """A value key absent from the initial map yields delta None."""
        data = pl.DataFrame({"var1": [0.0, 1.0], "var2": [5.0, 6.0], "obj": [0.5, 0.9]})
        result = compute_result_summary(
            data=data,
            context=SummaryContext(
                objective_name="obj",
                variable_names=["var1", "var2"],
                direction=Direction.MAXIMIZE,
                # initial provides only var1 -> var2/obj are present in values
                # but absent from initial, exercising the false branch of `key in initial`.
                initial_point={"var1": 0.0},
                variable_metadata={
                    "var1": {"step_size": 0.1, "units": "V"},
                    "var2": {"step_size": 0.1, "units": "V"},
                },
                observable_metadata={"obj": {"step_size": 0.0}},
            ),
        )
        best = result.best_points[0]
        assert best.annotated_values is not None
        assert best.annotated_values["var2"].delta_vs_initial is None
        assert best.annotated_values["var1"].delta_vs_initial is not None


def test_generate_text_summary_with_no_best_obj() -> None:
    """best_obj None skips the 'Best:' line."""
    text = _generate_text_summary(
        _NarrativeInputs(
            context=SummaryContext(
                objective_name="obj",
                variable_names=["var1"],
                direction=Direction.MAXIMIZE,
            ),
            obj_desc="obj",
            best_obj=None,
            initial_obj=None,
            total_iterations=3,
            target_val=None,
            target_units="",
            achievement_pct=None,
            improved=None,
            best_point_annotated=None,
        )
    )
    assert "Best:" not in text
    assert "3 iterations evaluated." in text


def test_generate_text_summary_header_renders_direction_value() -> None:
    """A Direction member renders as its value ('MAXIMIZE') in the header, not
    'Direction.MAXIMIZE' (py3.11+ f-string behavior for str-mixin enums)."""
    text = _generate_text_summary(
        _NarrativeInputs(
            context=SummaryContext(
                objective_name="obj",
                variable_names=["var1"],
                direction=Direction.MAXIMIZE,
            ),
            obj_desc="obj",
            best_obj=None,
            initial_obj=None,
            total_iterations=1,
            target_val=None,
            target_units="",
            achievement_pct=None,
            improved=None,
            best_point_annotated=None,
        )
    )
    assert 'Optimized "obj" (MAXIMIZE).' in text
