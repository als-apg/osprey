"""App-test-local fake environment (app + models only — no sibling imports).

Replaces `ALSInjectorSimulationEnvironment` for tests that drive app
orchestration and need only *an* environment, not real injector physics. Exposes
the real injector variable/observable NAMES and bounds (via _injector_data) so
existing test bodies and shared fixtures work unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from osprey.services.optimizer.models.enums import EnvironmentType
from osprey.services.optimizer.models.exceptions import VariableOutOfBoundsError
from osprey.services.optimizer.models.protocols import (
    EnvironmentProtocol,
    ObservableMetadata,
    PVObservable,
    PVReading,
    VariableMetadata,
)

from ._injector_data import OBSERVABLES, VARIABLES


class FakeInjectorEnvironment(EnvironmentProtocol):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._setpoints: dict[str, float] = {}
        self._check_connection_result: bool | None = True

    @property
    def environment_type(self) -> EnvironmentType:
        return EnvironmentType.SIMULATION

    @property
    def variables(self) -> dict[str, VariableMetadata]:
        return VARIABLES

    @property
    def observables(self) -> dict[str, ObservableMetadata]:
        return OBSERVABLES

    @property
    def available_objectives(self) -> list[str]:
        return list(OBSERVABLES)

    @property
    def default_objective(self) -> str:
        return "charge_ltb"

    def get_variables(self, variable_names: list[str]) -> dict[str, PVReading]:
        return {n: PVReading(value=self._setpoints.get(n, 0.0)) for n in variable_names}

    def set_variables(
        self,
        variable_inputs: dict[str, float],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> None:
        # Bounds/unknown-name validation mirrors
        # ALSInjectorEnvironment.set_variables (same exception type and
        # message shape) so tests driving Evaluator error-propagation behave
        # the same against the fake as against the real env.
        for name, value in variable_inputs.items():
            if name not in VARIABLES:
                raise VariableOutOfBoundsError(f"Unknown variable: {name}")
            metadata = VARIABLES[name]
            if not math.isfinite(value):
                raise VariableOutOfBoundsError(f"Variable {name} value {value} is not finite")
            if value < metadata.min_value or value > metadata.max_value:
                raise VariableOutOfBoundsError(
                    f"Variable {name} value {value} is outside bounds [{metadata.min_value}, {metadata.max_value}]"
                )
        for name, value in variable_inputs.items():
            self._setpoints[name] = value

    def _observable_value(self, name: str) -> float:
        # Deterministic, strictly positive, monotone in each variable, and
        # non-constant across the space — gives BO signal AND satisfies the
        # `charge >= 0` assertions. `score` is the mean normalized position in
        # [0, 1]; lower-bound corner -> 0.1*seed, upper-bound corner -> seed,
        # so evaluate(lo) != evaluate(hi). Per-observable `seed` (stable, NOT
        # Python's randomized hash) keeps charge_ltb != charge_gtl != linac_efficiency
        # and identical across processes/runs.
        if not self._setpoints:
            score = 0.5
        else:
            fracs = []
            for var, val in self._setpoints.items():
                meta = VARIABLES[var]
                span = (meta.max_value - meta.min_value) or 1.0
                fracs.append((val - meta.min_value) / span)  # 0..1
            score = sum(fracs) / len(fracs)
        score = max(0.0, min(1.0, score))
        seed = 50 + (sum(ord(c) for c in name) % 50)  # 50..99, deterministic per name
        return round(seed * (0.1 + 0.9 * score), 6)

    def get_observables(
        self,
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        return {n: PVReading(value=self._observable_value(n)) for n in observable_names}

    def evaluate(
        self,
        variable_inputs: dict[str, float],
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        self.set_variables(variable_inputs, check_cancellation=check_cancellation)
        return self.get_observables(observable_names, check_cancellation=check_cancellation)

    def read_pvs(self, names: list[str]) -> dict[str, PVReading]:
        return {n: PVReading(value=0.0, connected=True) for n in names}

    def check_connection(self) -> bool | None:
        return self._check_connection_result

    def __enter__(self) -> FakeInjectorEnvironment:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


class MockEnvironment(EnvironmentProtocol):
    """Lightweight, fully controllable endpoint-test double.

    Distinct from ``FakeInjectorEnvironment``: this uses a tiny, fixed data
    model (a single ``var1`` variable with round ``[0, 10]`` bounds, and
    ``obs1``/``obs_faulted`` observables returning fixed values) so API-router
    tests can assert exact numbers and exercise every ``read_pvs`` PVReading
    shape (waveform, enum, disconnected, NaN, scalar). ``FakeInjectorEnvironment``
    is the high-fidelity injector sim (49 real variables, computed observables) used
    by orchestration/e2e tests; use that when a test needs realistic physics,
    this when a test needs a small, predictable environment to drive HTTP
    endpoints.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._check_connection_result: bool | None = True
        self._config = config or {}

    @property
    def environment_type(self) -> EnvironmentType:
        return EnvironmentType.SIMULATION

    @property
    def variables(self) -> dict[str, VariableMetadata]:
        return {
            "var1": VariableMetadata(
                min_value=0.0,
                max_value=10.0,
                units="V",
                description="Var 1",
                readback_pv="V1:RBV",
                setpoint_pv="V1:SP",
            )
        }

    @property
    def observables(self) -> dict[str, ObservableMetadata]:
        return {
            "obs1": PVObservable(
                description="Observable 1",
                units="mA",
                min_value=0.0,
                max_value=100.0,
                readback_pv="OBS1:RBV",
            ),
            "obs_faulted": PVObservable(
                description="Faulted observable",
                units="mA",
                readback_pv="OBS_FAULTED:RBV",
            ),
        }

    @property
    def available_objectives(self) -> list[str]:
        return list(self.observables.keys())

    @property
    def default_objective(self) -> str:
        return "obs1"

    def get_variables(self, variable_names: list[str]) -> dict[str, PVReading]:
        return {n: PVReading(value=5.0) for n in variable_names}

    def set_variables(
        self,
        variable_inputs: dict[str, float],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> None:
        pass

    def evaluate(
        self,
        variable_inputs: dict[str, float],
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        self.set_variables(variable_inputs, check_cancellation=check_cancellation)
        return self.get_observables(
            observable_names,
            check_cancellation=check_cancellation,
        )

    def get_observables(
        self,
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        return {
            n: PVReading(value=None) if n == "obs_faulted" else PVReading(value=10.0)
            for n in observable_names
        }

    def read_pvs(self, names: list[str]) -> dict[str, PVReading]:
        # Test fixture covering the native-type + metadata shapes:
        #   SR:DCCT / others → scalar 487.2 (mA);  FAULT:NAN → NaN surfaced as None;
        #   MISSING:* → disconnected;  SR:BPM:WF → waveform;  SR:MODE → enum.
        out: dict[str, PVReading] = {}
        for n in names:
            if n.startswith("MISSING:"):
                out[n] = PVReading(value=None, connected=False)
            elif n == "FAULT:NAN":
                out[n] = PVReading(value=None, connected=True)
            elif n == "SR:BPM:WF":
                out[n] = PVReading(value=[0.1, -0.3, 0.5], connected=True, count=3, units="mm")
            elif n == "SR:MODE":
                out[n] = PVReading(value=1, connected=True, enum_strings=["Off", "User"])
            else:
                out[n] = PVReading(value=487.2, connected=True, units="mA", severity=0)
        return out

    def check_connection(self) -> bool | None:
        return self._check_connection_result

    def __enter__(self) -> MockEnvironment:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class EchoEnv:
    """Minimal environment returning a fixed objective dict from evaluate().

    Shared test double (was a file-local underscore-prefixed class duplicated
    across test_evaluation, test_runner, and test_fixed_features).

    Faithful to the env contract: returns exactly the requested observables
    (extras in the fixed result are not leaked) and fails loudly if a test
    requests a name the fixed result does not define — a silent full-dict
    echo would let contract misuse pass unnoticed.
    """

    def __init__(self, result: dict[str, PVReading]) -> None:
        self._result = result
        self.last_inputs: dict[str, float] | None = None
        self.last_observable_names: list[str] | None = None

    def evaluate(
        self,
        variable_inputs: dict[str, float],
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        self.last_inputs = variable_inputs
        self.last_observable_names = observable_names
        missing = [name for name in observable_names if name not in self._result]
        if missing:
            raise AssertionError(f"EchoEnv asked for observables it does not define: {missing}")
        return {name: self._result[name] for name in observable_names}
