"""The worker's run and its write authorization: the budget asked for, and what an end writes.

The fake environment stands in for the machine and records every write. The
run's :class:`RunAuthorization` is replaced by a recorder, so these rows see
exactly what the worker asks for (the channels, the steps, the completion) and
what it writes when the authorization ends, is never had, or the run is
cancelled. What the connector child does with a grant is
``test_write_authorization_run.py``'s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from osprey_optimizer.environments.write_authority import current_write_authority
from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.models.enums import AlgorithmType
from osprey_optimizer.models.exceptions import AuthorizationEndedError, CancellationError
from osprey_optimizer.services import execution
from osprey_optimizer.services.draft_resolution import resolve_draft_config
from osprey_optimizer.services.execution import ExecutionService
from osprey_optimizer.services.write_authorization import WritePlan

VARIABLES = ["LN______MASTPH_AC00", "LN______AS2_PH_AC00"]
N_INITIAL = 3
N_ITERATIONS = 2


class Recorder:
    """A run's write authorization that records the plan and grants it, or ends at a write."""

    plans: list[WritePlan]
    released: int
    authorities: list[Any]
    refuse: AuthorizationEndedError | None = None

    def __init__(self, authority: Any, **_kwargs: Any) -> None:
        self._authority = authority
        type(self).authorities.append(authority)

    def acquire(self, plan: WritePlan) -> None:
        type(self).plans.append(plan)
        if type(self).refuse is not None:
            raise type(self).refuse
        self._authority.authorization = "f" * 32

    def release(self) -> None:
        type(self).released += 1
        self._authority.end("released")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[Recorder]:
    recording = type("RecordingAuthorization", (Recorder,), {})
    recording.plans, recording.released, recording.authorities = [], 0, []
    monkeypatch.setattr(execution, "RunAuthorization", recording)
    return recording


@pytest.fixture
def service(mock_redis: MagicMock, tmp_path: Path) -> ExecutionService:
    del mock_redis  # wanted for the Redis it patches
    return ExecutionService(results_dir=tmp_path / "results", control_key="auth-job")


@pytest.fixture
def writes(mock_plugin_env: Any, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, float]]:
    """Every write the run makes, in order."""
    recorded: list[dict[str, float]] = []
    real_set = mock_plugin_env.set_variables

    def recording_set(values: dict[str, float], **kwargs: Any) -> None:
        recorded.append(dict(values))
        real_set(values, **kwargs)

    monkeypatch.setattr(mock_plugin_env, "set_variables", recording_set)
    return recorded


def _config(env: Any, **overrides: Any) -> OptimizationConfig:
    config = OptimizationConfig(
        environment_name="test_plugin",
        control_target="va",
        objective_name="charge_ltb",
        algorithm=AlgorithmType.RANDOM,
        variables=VARIABLES,
        n_initial_samples=N_INITIAL,
        n_iterations=N_ITERATIONS,
        **overrides,
    )
    with env:
        resolve_draft_config(config, env)
    return config


# -- what is asked for ----------------------------------------------------------


@pytest.mark.parametrize(
    ("on_complete", "completion", "completion_steps"),
    [
        pytest.param("apply_best", "apply_best", 1, id="apply_best"),
        pytest.param("restore", "restore", 1, id="restore"),
        pytest.param("hold", "none", 0, id="hold"),
    ],
)
def test_the_budget_is_the_resolved_evaluations_plus_the_completion_write(
    service: ExecutionService,
    mock_plugin_env: Any,
    writes: list[dict[str, float]],
    recorder: type[Recorder],
    on_complete: str,
    completion: str,
    completion_steps: int,
) -> None:
    service.start_optimization(_config(mock_plugin_env, on_complete=on_complete))

    [plan] = recorder.plans
    evaluations = N_INITIAL + N_ITERATIONS
    assert plan.evaluations == evaluations
    assert plan.completion == completion
    assert plan.steps == evaluations + completion_steps
    # Exactly what the budget pays for was written: one write per evaluation
    # and one for the completion.
    assert len(writes) == plan.steps
    assert plan.target == "va"
    # What the run is, and the job id the operator correlates it by.
    assert plan.label == "test_plugin optimization (auth-job)"
    metadata = mock_plugin_env.variables
    assert set(plan.channels) == {metadata[name].setpoint_pv for name in VARIABLES}
    overrides = _config(mock_plugin_env).variable_overrides or {}
    for name in VARIABLES:
        low, high = plan.channels[metadata[name].setpoint_pv]
        assert overrides[name].min <= low <= high <= overrides[name].max
        assert metadata[name].min_value <= low and high <= metadata[name].max_value
    assert recorder.released == 1


def test_every_write_of_the_run_goes_out_under_the_authorization_and_none_after(
    service: ExecutionService,
    mock_plugin_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    recorder: type[Recorder],
) -> None:
    seen: list[tuple[str | None, bool, str | None]] = []
    real_set = mock_plugin_env.set_variables

    def observing_set(values: dict[str, float], **kwargs: Any) -> None:
        authority = current_write_authority()
        assert authority is not None
        seen.append((authority.authorization, authority.required, authority.ended))
        real_set(values, **kwargs)

    monkeypatch.setattr(mock_plugin_env, "set_variables", observing_set)
    service.start_optimization(_config(mock_plugin_env))
    assert seen and set(seen) == {("f" * 32, True, None)}
    [authority] = recorder.authorities
    assert authority.ended == "released"


def test_a_readonly_run_asks_for_nothing(
    service: ExecutionService, mock_plugin_env: Any, recorder: type[Recorder]
) -> None:
    service.start_optimization(
        _config(mock_plugin_env, execution_mode="readonly", on_complete="hold")
    )
    assert recorder.plans == []
    [authority] = recorder.authorities
    assert authority.required is False


# -- what an end writes -----------------------------------------------------------


def test_a_run_never_granted_writes_nothing(
    service: ExecutionService,
    mock_plugin_env: Any,
    writes: list[dict[str, float]],
    recorder: type[Recorder],
) -> None:
    recorder.refuse = AuthorizationEndedError("owner_unknown")
    with pytest.raises(AuthorizationEndedError) as caught:
        service.start_optimization(_config(mock_plugin_env, on_complete="restore"))
    assert caught.value.reason == "owner_unknown"
    assert writes == []
    assert recorder.released == 1


@pytest.mark.parametrize("on_complete", ["apply_best", "restore"])
def test_an_authorization_that_ends_mid_run_ends_it_with_no_completion_write(
    service: ExecutionService,
    mock_plugin_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    recorder: type[Recorder],
    on_complete: str,
) -> None:
    """Budget spent, a stop, a lapsed lease: the run stops there, saves what it
    measured, and neither applies its best point nor restores."""
    recorded: list[dict[str, float]] = []
    real_set = mock_plugin_env.set_variables

    def ending_set(values: dict[str, float], **kwargs: Any) -> None:
        if len(recorded) == 2:
            authority = current_write_authority()
            assert authority is not None
            authority.end("auth_exhausted")
            raise AuthorizationEndedError("auth_exhausted")
        recorded.append(dict(values))
        real_set(values, **kwargs)

    monkeypatch.setattr(mock_plugin_env, "set_variables", ending_set)
    save_partial = MagicMock(return_value=None)
    monkeypatch.setattr(service, "_save_partial_results", save_partial)
    with pytest.raises(AuthorizationEndedError):
        service.start_optimization(_config(mock_plugin_env, on_complete=on_complete))
    assert len(recorded) == 2
    # What it measured so far is kept as the run's record, as on a cancel.
    save_partial.assert_called_once()
    assert recorder.released == 1


def test_a_cancel_releases_the_authorization_and_writes_nothing_after(
    service: ExecutionService,
    mock_plugin_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    recorder: type[Recorder],
) -> None:
    recorded: list[dict[str, float]] = []
    real_set = mock_plugin_env.set_variables

    def cancelling_set(values: dict[str, float], **kwargs: Any) -> None:
        recorded.append(dict(values))
        real_set(values, **kwargs)
        if len(recorded) == 2:
            raise CancellationError("cancelled by operator")

    monkeypatch.setattr(mock_plugin_env, "set_variables", cancelling_set)
    with pytest.raises(CancellationError):
        service.start_optimization(_config(mock_plugin_env, on_complete="restore"))
    assert len(recorded) == 2
    assert recorder.released == 1
    [authority] = recorder.authorities
    assert authority.ended == "released"


# -- the plan's ranges --------------------------------------------------------------


class _Limit:
    def __init__(self, low: float | None, high: float | None, *, writable: bool = True) -> None:
        self.min_value, self.max_value, self.writable = low, high, writable


def _env(**ranges: tuple[float, float]) -> Any:
    from osprey_optimizer.models.protocols import VariableMetadata

    env = MagicMock()
    env.variables = {
        name: VariableMetadata(
            min_value=low, max_value=high, readback_pv=f"{name}:RB", setpoint_pv=f"{name}:SP"
        )
        for name, (low, high) in ranges.items()
    }
    return env


def test_a_range_is_the_window_narrowed_to_the_environment_and_the_limits_database() -> None:
    from osprey_optimizer.services.write_authorization import write_plan

    plan = write_plan(
        _env(x=(-3.0, 3.0), y=(-10.0, 10.0), z=(0.0, 5.0)),
        {"x": (-5.0, 5.0), "y": (-4.0, 4.0)},
        target="standin",
        evaluations=7,
        on_complete="hold",
        fixed={"z": 2.5},
        limits={"y:SP": _Limit(-1.0, None), "z:SP": _Limit(0.0, 5.0)},
    )
    assert dict(plan.channels) == {"x:SP": (-3.0, 3.0), "y:SP": (-1.0, 4.0), "z:SP": (2.5, 2.5)}
    assert (plan.steps, plan.completion) == (7, "none")


@pytest.mark.parametrize(
    "limits",
    [
        pytest.param({"x:SP": _Limit(None, None, writable=False)}, id="read_only"),
        pytest.param({"x:SP": _Limit(4.0, 6.0)}, id="outside_the_window"),
    ],
)
def test_a_channel_the_limits_database_forbids_is_never_asked_for(limits: dict[str, Any]) -> None:
    from osprey_optimizer.services.write_authorization import write_plan

    with pytest.raises(ValueError, match="x:SP"):
        write_plan(
            _env(x=(-3.0, 3.0)),
            {"x": (-1.0, 1.0)},
            target="standin",
            evaluations=1,
            on_complete="apply_best",
            limits=limits,
        )
