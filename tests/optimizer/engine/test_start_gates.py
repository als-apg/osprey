"""Tests for api/start_gates.py — start-time verification gates (issue #2).

Canonical home for ``run_start_gates``: the machine-settings check and drift
guard that run between the environment-selection guard and the worker spawn
in ``routers/jobs.py::start_job``.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from osprey_optimizer.api.start_gates import run_start_gates
from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.models.exceptions import MachineSettingsError
from osprey_optimizer.models.protocols import PVReading
from osprey_optimizer.services.draft_resolution import resolve_draft_config

from ._fakes import FakeInjectorEnvironment
from ._injector_data import VARIABLES


def _resolved_config(fake_env: FakeInjectorEnvironment) -> OptimizationConfig:
    # FakeInjectorEnvironment defaults every unset readback to 0.0, which lies
    # outside the safety range of most of the ~49 real injector variables (e.g.
    # EG______BIAS___AC01 is [40, 60]). With no `variables=` restriction, all of
    # them resolve active and default-None overrides resolve to each variable's
    # full safety range -- so seed every setpoint to its own `min_value` (always
    # inside its own bounds) to get a config that legitimately passes the drift
    # check, rather than one that "passes" only because the fake omits most vars.
    fake_env.set_variables({name: meta.min_value for name, meta in VARIABLES.items()})
    cfg = OptimizationConfig(environment_name="fake", control_target="va", objective_name="obj")
    with fake_env:
        resolve_draft_config(cfg, fake_env)
    return cfg


def test_unresolved_config_is_409(fake_env: FakeInjectorEnvironment) -> None:
    cfg = OptimizationConfig(environment_name="fake", control_target="va", objective_name="obj")
    with pytest.raises(HTTPException) as excinfo:
        run_start_gates(cfg, fake_env, env_name="fake")
    assert excinfo.value.status_code == 409


def test_passing_gates_return_none(fake_env: FakeInjectorEnvironment) -> None:
    run_start_gates(_resolved_config(fake_env), fake_env, env_name="fake")


def test_machine_check_failure_is_412_with_all_failures(fake_env: FakeInjectorEnvironment) -> None:
    cfg = _resolved_config(fake_env)

    def _refuse() -> None:
        raise MachineSettingsError("no", failures=["GATE:A: got 0, expected == 1 (gate A)"])

    fake_env.check_machine_settings = _refuse  # type: ignore[method-assign]
    with pytest.raises(HTTPException) as excinfo:
        run_start_gates(cfg, fake_env, env_name="fake")
    assert excinfo.value.status_code == 412
    assert "GATE:A" in excinfo.value.detail


def test_drift_outside_resolved_bounds_is_412(fake_env: FakeInjectorEnvironment) -> None:
    cfg = _resolved_config(fake_env)
    assert cfg.variables is not None
    assert cfg.variable_overrides is not None
    first = cfg.variables[0]
    # Shrink the stored window so the fake env's current value now lies outside it.
    # (The fake's current readback for `first` is 0.0, which equals the resolved
    # `min` here, not `max` -- assigning min=max leaves the window at the old
    # `max`, which the current value does not satisfy.)
    cfg.variable_overrides[first].min = cfg.variable_overrides[first].max
    with pytest.raises(HTTPException) as excinfo:
        run_start_gates(cfg, fake_env, env_name="fake")
    assert excinfo.value.status_code == 412
    assert first in excinfo.value.detail


def test_unreadable_setpoint_is_412(fake_env: FakeInjectorEnvironment) -> None:
    cfg = _resolved_config(fake_env)
    assert cfg.variables is not None
    first = cfg.variables[0]
    # Resolve against the normal (connected) fake, then make every setpoint dead:
    # the drift check cannot compare a value it could not read, so a run must not
    # start on a disconnected PV.
    fake_env.get_setpoints = lambda variable_names: {  # type: ignore[method-assign]
        n: PVReading(value=None, connected=False) for n in variable_names
    }
    with pytest.raises(HTTPException) as excinfo:
        run_start_gates(cfg, fake_env, env_name="fake")
    assert excinfo.value.status_code == 412
    assert f"{first}: setpoint unavailable" in excinfo.value.detail


def test_both_gates_failing_are_aggregated_into_one_412(fake_env: FakeInjectorEnvironment) -> None:
    # Both gates trip on the same config: the machine-settings failure must not
    # short-circuit the drift check -- a single 412 reports every violation.
    cfg = _resolved_config(fake_env)
    assert cfg.variables is not None
    assert cfg.variable_overrides is not None
    first = cfg.variables[0]

    def _refuse() -> None:
        raise MachineSettingsError("no", failures=["GATE:A: locked out"])

    fake_env.check_machine_settings = _refuse  # type: ignore[method-assign]
    cfg.variable_overrides[first].min = cfg.variable_overrides[first].max
    with pytest.raises(HTTPException) as excinfo:
        run_start_gates(cfg, fake_env, env_name="fake")
    assert excinfo.value.status_code == 412
    assert "GATE:A" in excinfo.value.detail
    assert first in excinfo.value.detail
    assert "; " in excinfo.value.detail
