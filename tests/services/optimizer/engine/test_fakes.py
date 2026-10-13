"""Tests for FakeInjectorEnvironment — the app_tests-fake stand-in for
ALSInjectorSimulationEnvironment used by the decoupled app_tests suite.
"""

import pytest

from osprey.services.optimizer.models.exceptions import VariableOutOfBoundsError

from ._fakes import FakeInjectorEnvironment
from ._injector_data import VARIABLES


def test_fake_matches_real_als_names_bounds_and_evaluate_behavior() -> None:
    """Single fake-drift guard: names/bounds mirror VARIABLES, and evaluate()
    is deterministic, non-negative, and varies across inputs."""
    env = FakeInjectorEnvironment()

    assert set(env.variables) == set(VARIABLES)
    for name in ("charge_ltb", "charge_gtl", "linac_efficiency"):
        assert name in env.observables
        assert name in env.available_objectives
    assert env.default_objective == "charge_ltb"

    for name, meta in VARIABLES.items():
        assert env.variables[name].min_value == meta.min_value
        assert env.variables[name].max_value == meta.max_value

    names = list(env.variables)
    lo = {n: VARIABLES[n].min_value for n in names}
    hi = {n: VARIABLES[n].max_value for n in names}
    a = env.evaluate(lo, ["charge_ltb"])
    b = env.evaluate(lo, ["charge_ltb"])
    c = env.evaluate(hi, ["charge_ltb"])
    assert a["charge_ltb"].value is not None
    assert a["charge_ltb"].value >= 0.0  # tests assert charge >= 0
    assert a["charge_ltb"].value == b["charge_ltb"].value  # deterministic
    assert a["charge_ltb"].value != c["charge_ltb"].value  # non-constant across inputs


_FIRST = next(iter(VARIABLES))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        pytest.param({"not_a_real_var": 1.0}, "Unknown variable", id="unknown_variable"),
        pytest.param(
            {_FIRST: VARIABLES[_FIRST].max_value + 1000.0}, "outside bounds", id="out_of_bounds"
        ),
    ],
)
def test_fake_rejects_bad_variables(values: dict[str, float], match: str) -> None:
    """Matches ALSInjectorEnvironment.set_variables: same exception type
    and message shape, so tests driving Evaluator error-propagation see
    identical behavior against the fake and the real env."""
    with pytest.raises(VariableOutOfBoundsError, match=match):
        FakeInjectorEnvironment().set_variables(values)
