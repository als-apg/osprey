"""Tests for services/draft_resolution.py — submit-time draft resolution.

Canonical home for ``resolve_draft_config`` (issue #2's two-phase job
contract: a persisted draft is fully resolved against live readbacks before
the worker ever sees it) and the relocated ``read_required_variables``.
"""

from __future__ import annotations

import pytest

from osprey.services.optimizer.models.config import OptimizationConfig, VariableOverride
from osprey.services.optimizer.models.exceptions import ConnectionError as ALSConnectionError
from osprey.services.optimizer.services.draft_resolution import resolve_draft_config

from ._fakes import FakeInjectorEnvironment
from .conftest import make_disconnected_env

# A variable with round bounds [0.0, 10.0] and step 0.01 - lets the window-override
# test assert exact +/-1.0 bounds around a current readback with no grid-snap noise.
_WINDOW_VAR = "LN______MASTPH_AC00"


@pytest.fixture
def fake_env_disconnected() -> FakeInjectorEnvironment:
    return make_disconnected_env()


def test_resolves_all_variables_when_selection_unset(fake_env: FakeInjectorEnvironment) -> None:
    cfg = OptimizationConfig(environment_name="fake", control_target="va", objective_name="obj")
    with fake_env:
        resolve_draft_config(cfg, fake_env)
    assert cfg.variables == list(fake_env.variables.keys())
    assert cfg.variable_overrides is not None
    for name in cfg.variables:
        ov = cfg.variable_overrides[name]
        meta = fake_env.variables[name]
        # No override -> the variable's own full safety range on its own step grid:
        # min anchors the grid at safety_min, max never exceeds safety_max.
        assert ov.step_size == meta.step_size
        assert ov.min == meta.min_value
        assert ov.max is not None and ov.min <= ov.max <= meta.max_value
    cfg.require_resolved()  # the produced config satisfies the contract


def test_window_override_centers_on_current_readback(fake_env: FakeInjectorEnvironment) -> None:
    fake_env.set_variables({_WINDOW_VAR: 5.0})
    cfg = OptimizationConfig(
        environment_name="fake",
        control_target="va",
        objective_name="obj",
        variables=[_WINDOW_VAR],
        variable_overrides={_WINDOW_VAR: VariableOverride(window_abs=1.0)},
    )
    with fake_env:
        resolve_draft_config(cfg, fake_env)
    assert cfg.variable_overrides is not None
    ov = cfg.variable_overrides[_WINDOW_VAR]
    assert ov.min == pytest.approx(4.0)
    assert ov.max == pytest.approx(6.0)
    assert ov.window_abs is None  # materialized to absolute bounds


def test_unknown_variable_raises_value_error(fake_env: FakeInjectorEnvironment) -> None:
    cfg = OptimizationConfig(
        environment_name="fake", control_target="va", objective_name="obj", variables=["nope"]
    )
    with fake_env, pytest.raises(ValueError, match="nope"):
        resolve_draft_config(cfg, fake_env)


def test_unreadable_active_readback_raises_connection_error(
    fake_env_disconnected: FakeInjectorEnvironment,
) -> None:
    cfg = OptimizationConfig(environment_name="fake", control_target="va", objective_name="obj")
    with fake_env_disconnected, pytest.raises(ALSConnectionError):
        resolve_draft_config(cfg, fake_env_disconnected)
