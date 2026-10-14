"""A whole optimization through a file-declared ConnectorEnvironment.

The pieces are tested apart elsewhere; this runs them together — draft
resolution, bounds, the Xopt backend, readback verification and the ``rms``
reduction — against a linear orbit model standing in for the machine. It is the
path an environment file takes in a real deployment, short of the connector pool.
"""

import math
from typing import Any

import pytest

from osprey.services.optimizer.environments.connector_env import ConnectorEnvironment
from osprey.services.optimizer.environments.spec import EnvironmentFile
from osprey.services.optimizer.models.config import OptimizationConfig
from osprey.services.optimizer.models.protocols import PVReading
from osprey.services.optimizer.services.draft_resolution import resolve_draft_config
from osprey.services.optimizer.services.orchestration import OptimizationEngine

N_BPM = 6
#: Orbit response (m per A) of each BPM to each corrector, plus a fixed error.
RESPONSE = {
    "VCM:01": [1e-6 * math.sin(i + 1) for i in range(N_BPM)],
    "VCM:02": [1e-6 * math.cos(2 * i + 1) for i in range(N_BPM)],
}
ERROR = [3e-6 * math.sin(3 * i + 0.5) for i in range(N_BPM)]

SPEC = EnvironmentFile.model_validate(
    {
        "name": "orbit",
        "variables": {
            f"{c}:SP": {"readback": f"{c}:RB", "min": -12, "max": 12, "step_size": 0.01}
            for c in RESPONSE
        },
        "observables": {
            "orbit_rms": {
                "reduce": "rms",
                "channels": [f"BPM:{{1..{N_BPM}}}:Y"],
                "direction": "MINIMIZE",
            },
        },
        "default_objective": "orbit_rms",
    }
)


class OrbitMachine:
    """Setpoints in, BPM readings out: y = error + sum(response * kick)."""

    def __init__(self, start: float = 8.0) -> None:
        self.setpoints = {f"{c}:SP": start for c in RESPONSE}
        self.writes = 0

    def _bpms(self) -> dict[str, float]:
        return {
            f"BPM:{i + 1}:Y": ERROR[i]
            + sum(RESPONSE[c][i] * self.setpoints[f"{c}:SP"] for c in RESPONSE)
            for i in range(N_BPM)
        }

    def read_many(self, channel_names: list[str]) -> dict[str, PVReading]:
        values = {
            **self.setpoints,
            **{k.replace(":SP", ":RB"): v for k, v in self.setpoints.items()},
            **self._bpms(),
        }
        return {n: PVReading(value=values[n]) for n in channel_names}

    def read_one(self, channel: str) -> PVReading:
        return self.read_many([channel])[channel]

    def set_values(self, channel_inputs: dict[str, float]) -> dict[str, float]:
        self.writes += 1
        self.setpoints.update(channel_inputs)
        return dict(channel_inputs)

    def check_connection(self, pv_name: str | None = None) -> bool:
        del pv_name  # the simulated machine is always reachable
        return True

    def rms(self) -> float:
        return math.sqrt(sum(v * v for v in self._bpms().values()) / N_BPM)


@pytest.mark.parametrize("algorithm", ["random", "upper_confidence_bound"])
def test_an_environment_file_runs_an_optimization_end_to_end(algorithm: str) -> None:
    machine = OrbitMachine()
    initial_rms = machine.rms()

    def factory(_name: str, env_config: dict[str, Any]) -> ConnectorEnvironment:
        return ConnectorEnvironment(
            {**env_config, "spec": SPEC},
            interface_factory=lambda target, mode: machine,
            limits_lookup=lambda target: None,
        )

    config = OptimizationConfig(
        environment_name="orbit",
        control_target="va",
        objective_name="orbit_rms",
        algorithm=algorithm,
        n_initial_samples=3,
        n_iterations=6,
        variables=[f"{c}:SP" for c in RESPONSE],
    )
    env = factory("orbit", {"control_target": "va"})
    with env:
        resolve_draft_config(config, env)
    result = OptimizationEngine(config, environment_factory=factory).run()

    data = result.data
    assert data.height == 9  # 3 initial + 6 iterations
    assert data["orbit_rms"].null_count() == 0
    # Every evaluation is an rms the model can reproduce, and the best beats the start.
    assert min(data["orbit_rms"].to_list()) <= initial_rms
    # The best point is the row with the smallest rms.
    best_row = data.row(data["orbit_rms"].arg_min(), named=True)
    assert result.best_point == pytest.approx({name: best_row[name] for name in result.best_point})
    # Every suggestion stayed on the step grid and inside the resolved bounds.
    for name, override in (config.variable_overrides or {}).items():
        for value in data[name].to_list():
            assert override.min - 1e-9 <= value <= override.max + 1e-9
            assert abs(round(value / 0.01) * 0.01 - value) < 1e-9
    assert machine.writes >= 9
