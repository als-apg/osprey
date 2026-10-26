"""What a run leaves the machine at.

A finished run leaves what ``on_complete`` says. A cancelled or failed run
leaves the machine where it is: no write follows a stop, and restoring the
starting setpoints is the operator's call, through the results restore
endpoint. The fake environment stands in for the machine. The patched engine
moves it off its starting values the way a search would, then cancels, fails or
finishes.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.models.enums import AlgorithmType
from osprey_optimizer.models.exceptions import CancellationError, RestoreFailedError
from osprey_optimizer.services.execution import ExecutionService

_EXECUTION = "osprey_optimizer.services.execution"


@pytest.fixture
def exec_service(mock_redis: MagicMock, tmp_path: Path) -> ExecutionService:
    del mock_redis  # wanted for the Redis it patches, not its value
    return ExecutionService(results_dir=tmp_path / "results", control_key="outcome-job")


@pytest.fixture
def machine(mock_plugin_env: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The fake env seeded to its midpoints, with every write recorded."""
    env = mock_plugin_env
    names = list(env.variables)[:2]
    start = {
        n: (env.variables[n].min_value + env.variables[n].max_value) / 2.0 for n in env.variables
    }
    with env:
        env.set_variables(start)
    writes: list[dict[str, float]] = []
    real_set = env.set_variables

    def recording_set(values: dict[str, float], **kwargs: Any) -> None:
        writes.append(dict(values))
        real_set(values, **kwargs)

    monkeypatch.setattr(env, "set_variables", recording_set)
    return {"env": env, "names": names, "start": {n: start[n] for n in names}, "writes": writes}


def _config(machine: dict[str, Any], **overrides: Any) -> OptimizationConfig:
    fields: dict[str, Any] = {
        "environment_name": "test_plugin",
        "control_target": "va",
        "objective_name": "charge_ltb",
        "algorithm": AlgorithmType.RANDOM,
        "variables": machine["names"],
        "n_iterations": 1,
    }
    fields.update(overrides)
    return OptimizationConfig(**fields)


def _search_then(machine: dict[str, Any], outcome: BaseException | None) -> Any:
    """An engine run that moves the machine to its lower bounds, then ``outcome``."""

    def run(_self: Any) -> Any:
        env = machine["env"]
        env.set_variables({n: env.variables[n].min_value for n in machine["names"]})
        if outcome is not None:
            raise outcome
        return MagicMock()

    return run


def _setpoints(machine: dict[str, Any]) -> dict[str, float]:
    return {n: machine["env"]._setpoints[n] for n in machine["names"]}


@pytest.mark.parametrize(
    "outcome",
    [CancellationError("cancelled by operator"), RuntimeError("generator crashed")],
    ids=["cancel", "failure"],
)
def test_a_stopped_run_leaves_the_machine_where_it_is(
    exec_service: ExecutionService,
    machine: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    outcome,
) -> None:
    """No write follows a stop: restoring is the operator's call, not the worker's."""
    monkeypatch.setattr(f"{_EXECUTION}.OptimizationEngine.run", _search_then(machine, outcome))
    with pytest.raises(type(outcome)):
        exec_service.start_optimization(_config(machine))
    lower = {n: machine["env"].variables[n].min_value for n in machine["names"]}
    assert _setpoints(machine) == pytest.approx(lower)
    # The search's move is the only write: none before it, none after the stop.
    assert machine["writes"] == [pytest.approx(lower)]


def test_the_start_is_the_setpoints_and_measuring_it_writes_nothing(
    exec_service: ExecutionService, machine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readbacks that sit off their setpoints (a calibration offset) must not leak
    into the starting point a restore writes back, and must not be written."""
    env = machine["env"]
    real_get_variables = env.get_variables

    def offset_readbacks(names: list[str]) -> Any:
        readings = real_get_variables(names)
        return {n: r.model_copy(update={"value": r.value + 0.5}) for n, r in readings.items()}

    monkeypatch.setattr(env, "get_variables", offset_readbacks)

    initial = exec_service._measure_initial_state(_config(machine))

    assert initial.start_values == pytest.approx(machine["start"])
    assert {n: initial.initial_point[n] for n in machine["names"]} == pytest.approx(
        machine["start"]
    )
    assert machine["writes"] == []


@pytest.mark.parametrize("control", ["cancel", "pause"])
def test_a_pause_or_cancel_sent_while_the_worker_spawned_is_honoured_before_anything_runs(
    exec_service: ExecutionService,
    machine: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    """The token is honoured as found at start, never reset: a pause blocks there
    (here: until a cancel arrives) and a cancel stops the run before it reads or
    writes anything."""
    token = MagicMock()
    if control == "cancel":
        token.check_cancellation.side_effect = CancellationError("cancelled during spawn")
    else:
        token.wait_if_paused.side_effect = CancellationError("cancelled while paused")
    monkeypatch.setattr(exec_service, "_get_cancellation_token", lambda: token)
    measure = MagicMock()
    monkeypatch.setattr(ExecutionService, "_measure_initial_state", measure)

    with pytest.raises(CancellationError):
        exec_service.start_optimization(_config(machine))

    token.reset.assert_not_called()
    measure.assert_not_called()
    assert machine["writes"] == []


def test_a_readonly_run_restores_nothing_on_completion(
    exec_service: ExecutionService, machine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{_EXECUTION}.OptimizationEngine.run", _search_then(machine, None))
    monkeypatch.setattr(f"{_EXECUTION}.process_results", MagicMock(return_value=None))
    exec_service.start_optimization(
        _config(machine, execution_mode="readonly", on_complete="restore")
    )
    lower = {n: machine["env"].variables[n].min_value for n in machine["names"]}
    assert machine["writes"] == [pytest.approx(lower)]


def test_a_failed_restore_on_completion_makes_the_job_an_error(
    exec_service: ExecutionService, machine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{_EXECUTION}.OptimizationEngine.run", _search_then(machine, None))
    monkeypatch.setattr(f"{_EXECUTION}.process_results", MagicMock(return_value=None))
    env = machine["env"]
    real_set = env.set_variables

    def fail_on_restore(values: dict[str, float], **kwargs: Any) -> None:
        if values == pytest.approx(machine["start"]):
            raise ConnectionError("gateway down")
        real_set(values, **kwargs)

    monkeypatch.setattr(env, "set_variables", fail_on_restore)
    with pytest.raises(RestoreFailedError, match="gateway down") as caught:
        exec_service.start_optimization(_config(machine, on_complete="restore"))
    assert "finished" in str(caught.value)
    assert isinstance(caught.value.__cause__, ConnectionError)


@pytest.mark.parametrize(
    ("on_complete", "applies_best", "restores"),
    [("apply_best", True, False), ("restore", False, True), ("hold", False, False)],
)
def test_on_complete(
    exec_service: ExecutionService,
    machine: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    on_complete: str,
    applies_best: bool,
    restores: bool,
) -> None:
    monkeypatch.setattr(f"{_EXECUTION}.OptimizationEngine.run", _search_then(machine, None))
    process = MagicMock(return_value=None)
    monkeypatch.setattr(f"{_EXECUTION}.process_results", process)
    exec_service.start_optimization(_config(machine, on_complete=on_complete))
    assert process.call_args.kwargs["apply_best_point"] is applies_best
    restored = _setpoints(machine) == pytest.approx(machine["start"])
    assert restored is restores
