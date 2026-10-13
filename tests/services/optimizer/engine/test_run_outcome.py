"""What a run leaves the machine at: ``on_abort`` and ``on_complete``.

The fake environment stands in for the machine. The patched engine moves it off
its starting values the way a search would, then cancels, fails or finishes.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from osprey.services.optimizer.models.config import OptimizationConfig
from osprey.services.optimizer.models.enums import AlgorithmType
from osprey.services.optimizer.models.exceptions import CancellationError, RestoreFailedError
from osprey.services.optimizer.services.execution import ExecutionService

_EXECUTION = "osprey.services.optimizer.services.execution"


@pytest.fixture
def exec_service(mock_redis: None, tmp_path: Path) -> ExecutionService:
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
def test_an_aborted_run_restores_its_starting_values_by_default(
    exec_service: ExecutionService,
    machine: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    outcome,
) -> None:
    monkeypatch.setattr(f"{_EXECUTION}.OptimizationEngine.run", _search_then(machine, outcome))
    with pytest.raises(type(outcome)):
        exec_service.start_optimization(_config(machine))
    assert _setpoints(machine) == pytest.approx(machine["start"])
    assert machine["writes"][-1] == pytest.approx(machine["start"])


@pytest.mark.parametrize(
    "outcome", [CancellationError("cancelled"), RuntimeError("crashed")], ids=["cancel", "failure"]
)
def test_hold_leaves_the_last_point_the_run_tried(
    exec_service: ExecutionService,
    machine: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    outcome,
) -> None:
    monkeypatch.setattr(f"{_EXECUTION}.OptimizationEngine.run", _search_then(machine, outcome))
    with pytest.raises(type(outcome)):
        exec_service.start_optimization(_config(machine, on_abort="hold"))
    lower = {n: machine["env"].variables[n].min_value for n in machine["names"]}
    assert _setpoints(machine) == pytest.approx(lower)


def test_a_readonly_run_restores_nothing(
    exec_service: ExecutionService, machine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        f"{_EXECUTION}.OptimizationEngine.run",
        _search_then(machine, CancellationError("cancelled")),
    )
    with pytest.raises(CancellationError):
        exec_service.start_optimization(_config(machine, execution_mode="readonly"))
    # The only writes are the measurement's re-set of the start and the search's move.
    assert not any(w == pytest.approx(machine["start"]) for w in machine["writes"][2:])


def test_a_failed_restore_makes_the_job_an_error_naming_both(
    exec_service: ExecutionService, machine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        f"{_EXECUTION}.OptimizationEngine.run",
        _search_then(machine, CancellationError("cancelled")),
    )
    real_restore = ExecutionService._restore_initial_values

    def failing_restore(self: ExecutionService, *args: Any) -> None:
        raise ConnectionError("gateway down")

    monkeypatch.setattr(ExecutionService, "_restore_initial_values", failing_restore)
    with pytest.raises(RestoreFailedError) as caught:
        exec_service.start_optimization(_config(machine))
    assert "cancellation" in str(caught.value)
    assert "gateway down" in str(caught.value)
    assert isinstance(caught.value.__cause__, CancellationError)
    assert real_restore is not failing_restore


def test_a_failure_before_the_start_was_measured_restores_nothing(
    exec_service: ExecutionService, machine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_measure(*_args: Any) -> None:
        raise RuntimeError("cannot read the machine")

    monkeypatch.setattr(ExecutionService, "_measure_initial_state", broken_measure)
    restore = MagicMock()
    monkeypatch.setattr(ExecutionService, "_restore_initial_values", restore)
    with pytest.raises(RuntimeError, match="cannot read"):
        exec_service.start_optimization(_config(machine))
    restore.assert_not_called()


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
