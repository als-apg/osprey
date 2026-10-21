"""The series primitives import on their own, beside scenario application."""

from __future__ import annotations

import importlib


def test_the_series_module_imports_with_its_primitives() -> None:
    series = importlib.import_module("osprey_connectors.simulation.series")

    assert callable(series.keyed_normals)
    assert callable(series.wander)
    assert callable(series.clamp)


def test_the_series_module_carries_no_expression_helper() -> None:
    series = importlib.import_module("osprey_connectors.simulation.series")

    assert not hasattr(series, "ref_value")
    assert not hasattr(series, "ExpressionError")


def test_scenario_application_imports() -> None:
    apply = importlib.import_module("osprey.simulation.apply")

    assert callable(apply.apply_scenarios)
