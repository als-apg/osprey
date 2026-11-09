"""The tuning simulator's injector physics as an in-process environment.

No EPICS IOC, so nothing announces the facility PV names it models on any network:
the engine calls the physics directly. Test and benchmark use only; the names
and ranges are one real injector's, which is why this is not shipped code.

Built against whichever engine package it is given, so one definition serves
OSPREY's port and tuning_scripts itself in the parity benchmark.
"""

from __future__ import annotations

import importlib
from typing import Any

from .physics.constants import GTL_CHARGE_PV, LTB_CHARGE_PV, VARIABLES
from .physics.simulator import InjectorSimulator

#: The IOC's starting values (tuning_scripts_simulator/ioc.py ``_INITIAL_VALUES``).
INITIAL_VALUES: dict[str, float] = {
    "LN______MASTPH_AC00": 5.0,
    "LN______AS2_PH_AC00": 5.0,
    "EG______PHASE__AC02": 0.0,
    "GTL_____SHB1_PHAC00": 180.0,
    "GTL_____SHB2_PHAC00": 180.0,
}

#: tuning_scripts' phase_tuning_sim template's variables: the five RF phases.
PHASE_VARIABLES = tuple(INITIAL_VALUES)

#: The injector plugin's step sizes for them (tuning_scripts_als_plugin/constants.py).
STEP_SIZES: dict[str, float] = {
    "LN______MASTPH_AC00": 0.01,
    "LN______AS2_PH_AC00": 0.01,
    "EG______PHASE__AC02": 0.6,
    "GTL_____SHB1_PHAC00": 0.01,
    "GTL_____SHB2_PHAC00": 0.01,
}

#: Observable name -> the physics' channel for it.
OBSERVABLES = {"charge_ltb": LTB_CHARGE_PV, "charge_gtl": GTL_CHARGE_PV}


def make_environment_class(models_package: str) -> type:
    """A ``SimulatedInjectorEnvironment`` built on ``models_package``'s protocol types."""
    protocols = importlib.import_module(f"{models_package}.protocols")
    enums = importlib.import_module(f"{models_package}.enums")
    types_mod = importlib.import_module(f"{models_package}.types")
    exceptions = importlib.import_module(f"{models_package}.exceptions")

    class SimulatedInjectorEnvironment:
        def __init__(self, config: dict[str, Any]) -> None:
            self._sim = InjectorSimulator(
                seed=int(config.get("sim_seed", 42)), mode=config.get("sim_mode", "legacy")
            )
            self._values = dict(INITIAL_VALUES)
            self._sim.set_variables(self._values)

        @property
        def environment_type(self) -> Any:
            return enums.EnvironmentType.SIMULATION

        @property
        def variables(self) -> dict[str, Any]:
            return {
                name: protocols.VariableMetadata(
                    min_value=VARIABLES[name].min_value,
                    max_value=VARIABLES[name].max_value,
                    readback_pv=name,
                    setpoint_pv=name,
                    step_size=STEP_SIZES[name],
                )
                for name in PHASE_VARIABLES
            }

        @property
        def observables(self) -> dict[str, Any]:
            return {
                name: protocols.PVObservable(readback_pv=pv, direction=types_mod.Direction.MAXIMIZE)
                for name, pv in OBSERVABLES.items()
            }

        @property
        def available_objectives(self) -> list[str]:
            return list(OBSERVABLES)

        @property
        def default_objective(self) -> str:
            return "charge_ltb"

        def get_variables(self, variable_names: list[str]) -> dict[str, Any]:
            return {n: protocols.PVReading(value=self._values[n]) for n in variable_names}

        def get_setpoints(self, variable_names: list[str]) -> dict[str, Any]:
            return self.get_variables(variable_names)

        def set_variables(
            self,
            variable_inputs: dict[str, float],
            *,
            check_cancellation: Any = None,
        ) -> None:
            if check_cancellation is not None:
                check_cancellation()
            for name, value in variable_inputs.items():
                meta = VARIABLES[name]
                if not meta.min_value <= value <= meta.max_value:
                    raise exceptions.VariableOutOfBoundsError(f"{name}={value} outside bounds")
            self._values.update(variable_inputs)
            self._sim.set_variables(self._values)

        def get_observables(
            self,
            observable_names: list[str],
            *,
            check_cancellation: Any = None,
        ) -> dict[str, Any]:
            if check_cancellation is not None:
                check_cancellation()
            return {
                n: protocols.PVReading(value=self._sim.get_observable(OBSERVABLES[n]))
                for n in observable_names
            }

        def evaluate(
            self,
            variable_inputs: dict[str, float],
            observable_names: list[str],
            *,
            check_cancellation: Any = None,
        ) -> dict[str, Any]:
            if check_cancellation is not None:
                check_cancellation()
            self.set_variables(variable_inputs)
            return self.get_observables(observable_names)

        def read_pvs(self, names: list[str]) -> dict[str, Any]:
            return {n: protocols.PVReading(value=self._values.get(n)) for n in names}

        def check_connection(self) -> bool:
            return True

        def check_machine_settings(self) -> None:
            return None

        def __enter__(self) -> SimulatedInjectorEnvironment:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    return SimulatedInjectorEnvironment
