"""What a write pass of the served composite posts, decided in process.

:func:`~osprey.services.virtual_accelerator.serving.publication.same_value`
says whether two values would reach a client as the same bytes, and
:func:`~osprey.services.virtual_accelerator.serving.publication.changed_outputs`
narrows a pass's values to those a client has not received yet. The identity
case runs the built control-assistant composite and proves that posting only
the changes leaves every client holding what a full post would have given it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from osprey.services.virtual_accelerator.serving.publication import (
    changed_outputs,
    same_value,
)
from osprey.services.virtual_accelerator.serving.runner_config import periodic_addresses
from tests._builds import BuiltProject

# -- same_value ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        pytest.param(0.0, -0.0, False, id="signed-zero"),
        pytest.param(-0.0, -0.0, True, id="negative-zero"),
        pytest.param(math.nan, math.nan, True, id="nan"),
        pytest.param(math.nan, 0.0, False, id="nan-and-number"),
        pytest.param(1.5, 1.5, True, id="equal-floats"),
        pytest.param(1.5, 1.25, False, id="different-floats"),
        pytest.param(True, 1, False, id="bool-is-not-int"),
        pytest.param(1, 1.0, False, id="int-is-not-float"),
        pytest.param(7, 7, True, id="equal-ints"),
        pytest.param("ok", "ok", True, id="equal-strings"),
        pytest.param("ok", "fault", False, id="different-strings"),
    ],
)
def test_same_value_holds_only_for_values_that_pack_alike(a: Any, b: Any, same: bool) -> None:
    assert same_value(a, b) is same


def test_equal_arrays_of_different_dtype_are_not_the_same_value() -> None:
    assert same_value(np.array([1.0, 2.0]), np.array([1.0, 2.0], dtype=np.float32)) is False
    assert same_value(np.array([1, 2]), np.array([1.0, 2.0])) is False


def test_arrays_of_different_shape_are_not_the_same_value() -> None:
    assert same_value(np.zeros(4), np.zeros((2, 2))) is False


def test_arrays_with_nan_at_the_same_index_are_the_same_value() -> None:
    a = np.array([0.1, math.nan, 0.3])
    assert same_value(a, a.copy()) is True
    assert same_value(a, np.array([0.1, 0.2, 0.3])) is False


def test_a_trace_differing_in_one_sample_is_not_the_same_value() -> None:
    trace = np.linspace(-1.0, 1.0, 500, dtype=np.float64)
    moved = trace.copy()
    moved[317] += 1e-12

    assert same_value(trace, trace.copy()) is True
    assert same_value(trace, moved) is False


# -- changed_outputs -----------------------------------------------------------


def test_a_name_never_served_is_posted() -> None:
    posted = changed_outputs({"A": 1.0}, {}, frozenset(), frozenset(), frozenset())
    assert posted == {"A": 1.0}


def test_a_written_name_with_an_unchanged_value_is_posted() -> None:
    posted = changed_outputs({"SP": 2.0}, {"SP": 2.0}, frozenset({"SP"}), frozenset(), frozenset())
    assert posted == {"SP": 2.0}


def test_an_undefined_condition_that_flips_is_posted_both_ways() -> None:
    served = {"RB": 3.0}

    became_udf = changed_outputs({"RB": 3.0}, served, frozenset(), frozenset({"RB"}), frozenset())
    recovered = changed_outputs({"RB": 3.0}, served, frozenset(), frozenset(), frozenset({"RB"}))

    assert became_udf == {"RB": 3.0}
    assert recovered == {"RB": 3.0}


def test_an_unchanged_name_is_left_out_and_order_is_kept() -> None:
    out = {"C": 3.0, "A": 1.0, "B": 2.5}
    served = {"A": 1.0, "B": 2.0, "C": 3.0}

    posted = changed_outputs(out, served, frozenset({"C"}), frozenset(), frozenset())

    assert list(posted) == ["C", "B"]


def test_a_none_value_passes_through() -> None:
    posted = changed_outputs({"A": None}, {"A": None}, frozenset(), frozenset(), frozenset())
    assert posted == {"A": None}


# -- the built composite -------------------------------------------------------


class _Clock:
    """A fixed instant the test advances by hand."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _undefined(composite: Any, names: list[str]) -> frozenset[str]:
    return frozenset(
        name
        for name, entry in composite.output_severity(names).items()
        if entry.get("condition") == "udf"
    )


def test_publishing_only_changes_serves_what_a_full_publish_serves(
    built_control_assistant: BuiltProject,
) -> None:
    pytest.importorskip("at")
    from osprey_connectors.simulation.composite import Composite
    from osprey_connectors.simulation.view import SimulatorView

    view = SimulatorView.of_render(built_control_assistant.build_dir)
    clock = _Clock(1_760_000_000.0)
    composite = Composite(view, model_log=False, clock=clock)
    periodic = periodic_addresses(view)
    roster = [name for name in composite.supported_variables if name not in periodic]
    correctors = sorted(
        binding.address
        for binding in view.bindings()
        if binding.role == "setpoint"
        and binding.direction == "write"
        and binding.plane is not None
        and view.channel(binding.address).writable
        and view.channel(binding.address).value_type == "float"
    )
    assert correctors, "the view states no writable float corrector setpoint"
    corrector = correctors[0]
    low, high = view.channel(corrector).value_range or (-1.0, 1.0)

    served: dict[str, Any] = dict(composite.get(roster))
    udf_served = _undefined(composite, roster)
    for step in range(10):
        clock.now += 0.5
        composite.set({corrector: low + (high - low) * (step + 1) / 12})
        out = composite.get(roster)
        udf_now = _undefined(composite, roster)

        posted = changed_outputs(out, served, frozenset({corrector}), udf_now, udf_served)
        served.update(posted)
        udf_served = (udf_served - set(posted)) | (udf_now & set(posted))

        different = [name for name in roster if not same_value(served[name], out[name])]
        assert different == [], f"pass {step}: a client would hold stale {different[:5]}"
        assert udf_served == udf_now
        assert corrector in posted
        assert len(posted) < len(out), f"pass {step} posted every channel it read"
