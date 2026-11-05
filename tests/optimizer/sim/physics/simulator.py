"""Injector Simulator Entry Point.

This module provides the main entry point for the simulator, allowing selection
between the legacy mock implementation and the physics-based beam dynamics engine.
"""

from typing import TYPE_CHECKING, Protocol

import numpy as np

from .beam_dynamics import (
    AcceptanceCriteria,
    BeamElement,
    Drift,
    RFCavity,
    compute_acceptance,
    generate_bunch,
    mean_energy,
    phase_acceptance_mask,
    rms_bunch_length,
    rms_energy_spread,
    track_beam,
)
from .constants import (
    E_LINAC_MAX,
    E_SBUNCHER_MAX,
    ENERGY_ACCEPTANCE,
    F_SBAND,
    F_SHB1,
    F_SHB2,
    GTL_CHARGE_PV,
    INJECTION_BUCKET_PV,
    L_BETWEEN_LINACS,
    L_BUNCHER_TO_LINAC,
    L_GUN_TO_SHB1,
    L_LINAC_SECTION,
    L_SBUNCHER,
    L_SHB1_TO_SHB2,
    L_SHB2_TO_BUNCHER,
    LTB_CHARGE_PV,
    N_MACROPARTICLES,
    PHASE_ACCEPTANCE_DEG,
    PULSE_FWHM,
    V_SHB_MAX,
    VARIABLES,
    W_GUN,
)

VariableInputs = dict[str, float]


class SimulatorProtocol(Protocol):
    """Protocol for simulator implementations."""

    def set_variables(self, values: VariableInputs) -> None: ...
    def get_observable(self, name: str) -> float: ...


if TYPE_CHECKING:
    from .legacy_simulator import LegacyInjectorSimulator


class SimulatorNotInitializedError(Exception):
    """Simulator variables not initialized before use."""

    pass


class InjectorSimulator:
    """Main Simulator class that delegates to specific implementations based on configuration."""

    def __init__(self, seed: int = 42, mode: str = "legacy") -> None:
        """Initialize the simulator.

        Args:
            seed: Random seed for reproducibility.
            mode: Simulation mode. Options: "legacy" (mock function), "physics" (beam dynamics).
        """
        self.mode = mode
        self.seed = seed
        self._delegate: SimulatorProtocol

        if self.mode == "legacy":
            from .legacy_simulator import LegacyInjectorSimulator

            self._delegate = LegacyInjectorSimulator(seed=seed)
        elif self.mode == "physics":
            self._delegate = PhysicsInjectorSimulator(seed=seed)
        else:
            raise ValueError(f"Unknown simulator mode: {mode}")

    @property
    def legacy_delegate(self) -> "LegacyInjectorSimulator":
        """The delegate as its concrete legacy type, for tests and diagnostics.

        Raises:
            TypeError: If the simulator is not running the legacy delegate.
        """
        from .legacy_simulator import LegacyInjectorSimulator

        if not isinstance(self._delegate, LegacyInjectorSimulator):
            raise TypeError(f"Delegate is not the legacy simulator (mode={self.mode!r})")
        return self._delegate

    def set_variables(self, values: VariableInputs) -> None:
        """Store the current variable settings."""
        self._delegate.set_variables(values)

    def get_observable(self, name: str) -> float:
        """Get a single observable value.

        Args:
            name: Observable name to retrieve

        Returns:
            Observable value as float
        """
        return self._delegate.get_observable(name)


class PhysicsInjectorSimulator:
    """Physics-based simulator using the Beam Dynamics Engine."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.current_values: VariableInputs = {}
        self.injection_counter: int = 0

    def set_variables(self, values: VariableInputs) -> None:
        """Store the current variable settings."""
        self.current_values.update(values)

    def _normalize_to_10(self, name: str, value: float) -> float:
        """Normalize a variable value to [0, 10] range."""
        if name not in VARIABLES:
            return 5.0

        meta = VARIABLES[name]
        # Avoid division by zero
        if meta.max_value == meta.min_value:
            return 0.0

        norm_01 = (value - meta.min_value) / (meta.max_value - meta.min_value)
        return norm_01 * 10.0

    def map_controls(self) -> dict[str, float]:
        """Map normalized controls to physical parameters."""
        params = {
            "gun_delay": 0.0,
            "shb1_phase": 0.0,
            "shb2_phase": 0.0,
            "linac_phase": 0.0,
            "shb1_voltage": V_SHB_MAX,  # Assume max for now
            "shb2_voltage": V_SHB_MAX,
        }

        # [0, 10] -> [-180, 180] deg
        if "GTL_____SHB1_PHAC00" in self.current_values:
            val = self.current_values["GTL_____SHB1_PHAC00"]
            norm_val = self._normalize_to_10("GTL_____SHB1_PHAC00", val)
            params["shb1_phase"] = ((norm_val - 5.0) / 5.0) * 180.0

        if "GTL_____SHB2_PHAC00" in self.current_values:
            val = self.current_values["GTL_____SHB2_PHAC00"]
            norm_val = self._normalize_to_10("GTL_____SHB2_PHAC00", val)
            params["shb2_phase"] = ((norm_val - 5.0) / 5.0) * 180.0

        if "LN______MASTPH_AC00" in self.current_values:
            val = self.current_values["LN______MASTPH_AC00"]
            # Already 0-10; normalize anyway for consistency
            norm_val = self._normalize_to_10("LN______MASTPH_AC00", val)
            params["linac_phase"] = ((norm_val - 5.0) / 5.0) * 180.0

        return params

    def build_lattice(self, params: dict[str, float]) -> list[BeamElement]:
        """Build the beamline lattice with current parameters."""
        lattice: list[BeamElement] = []

        lattice.append(Drift(length=L_GUN_TO_SHB1))

        lattice.append(
            RFCavity(freq=F_SHB1, v_eff=params["shb1_voltage"], phase_deg=params["shb1_phase"])
        )

        lattice.append(Drift(length=L_SHB1_TO_SHB2))

        lattice.append(
            RFCavity(freq=F_SHB2, v_eff=params["shb2_voltage"], phase_deg=params["shb2_phase"])
        )

        lattice.append(Drift(length=L_SHB2_TO_BUNCHER))

        # Buncher: max field, -90 deg (bunching)
        v_buncher = E_SBUNCHER_MAX * L_SBUNCHER
        lattice.append(RFCavity(freq=F_SBAND, v_eff=v_buncher, phase_deg=-90.0))

        lattice.append(Drift(length=L_BUNCHER_TO_LINAC))

        v_linac = E_LINAC_MAX * L_LINAC_SECTION
        lattice.append(RFCavity(freq=F_SBAND, v_eff=v_linac, phase_deg=params["linac_phase"]))

        lattice.append(Drift(length=L_BETWEEN_LINACS))

        # Linac section 2 reuses the section-1 phase
        lattice.append(RFCavity(freq=F_SBAND, v_eff=v_linac, phase_deg=params["linac_phase"]))

        return lattice

    def run_beam_dynamics(self) -> dict[str, float]:
        """Run the beam dynamics simulation.

        Returns:
            dict[str, float]: Dictionary of observable metrics.
        """
        params = self.map_controls()

        t, W = generate_bunch(
            n_particles=N_MACROPARTICLES,
            w_mean=W_GUN,
            t_fwhm=PULSE_FWHM,
            seed=self.seed + self.injection_counter,  # Vary seed per injection
        )
        initial_charge = 1.0  # Normalized initial charge

        t += params["gun_delay"]

        lattice = self.build_lattice(params)

        t_final, W_final = track_beam(t, W, lattice)

        # GTL Charge: Assume 100% transmission from Gun
        gtl_charge = initial_charge

        bunch_length = rms_bunch_length(t_final)
        energy_spread = rms_energy_spread(W_final)

        # Target energy from phase-accepted particles only: the global mean includes
        # decelerated/lost particles, so it would misrepresent the useful beam.
        mask_finite = np.isfinite(t_final) & np.isfinite(W_final) & (W_final > 0)
        mean_E = 0.0

        if np.any(mask_finite):
            t_valid = t_final[mask_finite]
            W_valid = W_final[mask_finite]

            phase_mask = phase_acceptance_mask(t_valid, F_SBAND, PHASE_ACCEPTANCE_DEG)

            mean_E = (
                float(np.mean(W_valid[phase_mask])) if np.any(phase_mask) else mean_energy(W_final)
            )
        else:
            mean_E = 0.0

        # LTB Charge: Fraction of particles within acceptance
        acceptance_criteria = AcceptanceCriteria(
            freq=F_SBAND,
            phase_window_deg=PHASE_ACCEPTANCE_DEG,
            w_target=mean_E,
            w_window_fraction=ENERGY_ACCEPTANCE,
        )
        transmission, _ = compute_acceptance(
            t_final,
            W_final,
            criteria=acceptance_criteria,
        )

        ltb_charge = gtl_charge * transmission

        return {
            "gtl_charge": gtl_charge,
            "ltb_charge": ltb_charge,
            "transmission": transmission,
            "bunch_length": bunch_length,
            "energy_spread": energy_spread,
            "mean_energy": mean_E,
        }

    def get_observable(self, name: str) -> float:
        """Get a single observable value."""
        self.injection_counter += 1

        if name == INJECTION_BUCKET_PV:
            return float(self.injection_counter)

        # Re-run every time: the seed varies with injection_counter, so results can't be cached.
        results = self.run_beam_dynamics()

        result_mapping = {
            GTL_CHARGE_PV: "gtl_charge",
            LTB_CHARGE_PV: "ltb_charge",
            "transmission": "transmission",
            "bunch_length": "bunch_length",
            "energy_spread": "energy_spread",
            "mean_energy": "mean_energy",
        }

        if name in result_mapping:
            return results[result_mapping[name]]

        # If asking for a variable readback, return the setpoint (perfect control)
        if name in self.current_values:
            return self.current_values[name]

        return 0.0
