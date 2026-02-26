"""Simulation Backend Protocol.

Defines the unified interface for simulation backends used with soft IOCs.
Backends can be chained together using ChainedBackend for composition.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SimulationBackend(Protocol):
    """Protocol for simulation backends.

    All custom simulation backends must implement this interface to work with
    generated soft IOCs. The Protocol uses duck typing - no inheritance required.

    Backends can be composed using ChainedBackend, which calls backends in order.
    For on_write(), return None to delegate to the next backend in the chain.
    For step(), all backends run and results are merged (last wins).

    Example implementation:

        class PyATBackend:
            '''Physics simulation using pyAT.'''

            def __init__(self, lattice_file: str, energy: float = 1.9):
                self.lattice = load_lattice(lattice_file)
                self.energy = energy

            def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
                '''Initialize PVs from lattice model.'''
                return {pv['name']: self._get_initial_value(pv) for pv in pv_definitions}

            def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
                '''Handle setpoint write - recalculate physics.'''
                if not self._handles_pv(pv_name):
                    return None  # Delegate to next backend
                self._update_lattice(pv_name, value)
                return self._get_dependent_updates()

            def step(self, dt: float) -> dict[str, Any]:
                '''Advance simulation by dt seconds.'''
                return self._evolve_lattice(dt)

    Configuration (config.yml):

        simulation:
          backends:
            - type: "mock_style"                    # Base backend (provides defaults)
            - module_path: "my_project.simulation"  # Override (wins conflicts)
              class_name: "PyATBackend"
              params:
                lattice_file: "data/als_sr.mat"
                energy: 1.9
    """

    def initialize(self, pv_definitions: list[dict]) -> dict[str, Any]:
        """Initialize the backend and return initial PV values.

        Called once at IOC startup to set initial values for all PVs.

        Args:
            pv_definitions: List of PV definition dicts, each containing:
                - name: EPICS PV name (e.g., 'MAG:QUAD1:CURRENT:SP')
                - type: PV type ('float', 'int', 'enum', 'string', etc.)

        Returns:
            Dict mapping PV names to their initial values.
            Only include PVs that should be set to non-default values.
        """
        ...

    def on_write(self, pv_name: str, value: Any) -> dict[str, Any] | None:
        """Handle a write to a PV and return any cascading updates.

        Called when a client writes to a setpoint PV. Use this to:
        - Track the new setpoint value
        - Update paired readback PVs
        - Trigger physics calculations

        For chained backends, return None to delegate handling to the next
        backend in the chain. Return a dict (even empty) to handle the write.

        Args:
            pv_name: Name of the PV that was written to
            value: The new value that was written

        Returns:
            Dict mapping PV names to updated values, or None to delegate
            to the next backend in the chain.
        """
        ...

    def step(self, dt: float) -> dict[str, Any]:
        """Advance the simulation by a time step.

        Called periodically (at the configured update_rate) to evolve
        the simulation state. Use this for:
        - Time-dependent physics (decay, drift, oscillations)
        - Slow responses to setpoint changes
        - Adding realistic noise

        In chained backends, all backends run step() and results are merged
        with later backends winning on conflicts.

        Args:
            dt: Time elapsed since last step, in seconds

        Returns:
            Dict mapping PV names to their new values.
            Only include PVs whose values have changed.
        """
        ...
