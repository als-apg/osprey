"""Legacy Mock simulator for ALS injector optimization testing."""

import numpy as np
import numpy.typing as npt

from .constants import (
    GTL_CHARGE_PV,
    INJECTION_BUCKET_PV,
    LTB_CHARGE_PV,
    VARIABLES,
)
from .simulator import SimulatorNotInitializedError

# Peaks in 4D/6D space (normalized coordinates [0, 1])
SIMULATOR_PEAKS: list[list[float]] = [
    [0.2, 0.3, 0.4, 0.2, 0.5, 0.5],  # Peak 1 (Global max for some objectives)
    [0.8, 0.7, 0.6, 0.8, 0.5, 0.5],
    [0.2, 0.8, 0.3, 0.7, 0.5, 0.5],
    [0.7, 0.2, 0.8, 0.3, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Peak 5 (Central local max)
]

SIMULATOR_PEAK_HEIGHTS: list[float] = [0.95, 0.90, 0.85, 0.80, 0.75]
SIMULATOR_PEAK_WIDTHS: list[float] = [0.15, 0.20, 0.15, 0.25, 0.30]

SIMULATOR_BASE_EFFICIENCY: float = 0.1

GTL_BASE_CHARGE: float = 2.0  # nC
GUN_PHASE_AMPLITUDE: float = 0.5  # Fraction of charge modulation

# Noise Levels (Relative standard deviation)
BASE_NOISE_LEVEL: float = 0.02
CHARGE_NOISE_LEVEL: float = 0.03
EFFICIENCY_NOISE_LEVEL: float = 0.01

VariableInputs = dict[str, float]


class LegacyInjectorSimulator:
    """Complex mock function simulating the ALS injector behavior (Legacy implementation).

    This creates a multi-modal optimization landscape with:
    - Multiple local optima (justifying the 5 independent BO runs)
    - Nonlinear interactions between variables
    - Realistic noise to simulate measurement uncertainty
    - Physical constraints on charge values
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.variable_names: list[str] = [
            "LN______MASTPH_AC00",
            "LN______AS2_PH_AC00",
            "EG______PHASE__AC02",
            "GTL_____SHB1_PHAC00",
            "GTL_____SHB2_PHAC00",
        ]
        self.current_values: VariableInputs = {}
        self.injection_counter: int = 0

    def set_variables(self, values: VariableInputs) -> None:
        """Store the current variable settings."""
        self.current_values = values.copy()

    def _normalize_variables(self) -> np.ndarray:
        """Normalize variables to [0, 1] for easier function design."""
        vals = np.array(
            [self.current_values.get(name, 0) for name in self.variable_names], dtype=np.float64
        )
        mins = np.array(
            [VARIABLES[name].min_value for name in self.variable_names], dtype=np.float64
        )
        maxs = np.array(
            [VARIABLES[name].max_value for name in self.variable_names], dtype=np.float64
        )
        return (vals - mins) / (maxs - mins)

    def _efficiency_function(self, x: npt.NDArray[np.float64]) -> float:
        """Complex multi-modal efficiency function.

        Creates 5 distinct peaks in the 4D space, simulating disconnected
        optimization regions in the real injector.
        """
        # Use peak configurations from constants, truncated to the active
        # variable count (SIMULATOR_PEAKS is defined wider than variable_names).
        peaks = np.array(SIMULATOR_PEAKS)[:, : len(self.variable_names)]

        heights = np.array(SIMULATOR_PEAK_HEIGHTS)
        widths = np.array(SIMULATOR_PEAK_WIDTHS)

        efficiency = SIMULATOR_BASE_EFFICIENCY
        distances = np.linalg.norm(x - peaks, axis=1)
        efficiency += float((heights * np.exp(-(distances**2) / (2 * widths**2))).sum())

        # Add some nonlinear coupling terms
        efficiency += 0.05 * np.sin(10 * x[0] * x[1])
        efficiency += 0.03 * np.cos(8 * x[2] * x[3])

        # Add a gentle global trend
        efficiency -= 0.1 * np.sum((x - 0.5) ** 2)

        efficiency = np.clip(efficiency, 0.1, 0.98)

        return float(efficiency)

    def _add_measurement_noise(self, value: float, noise_level: float | None = None) -> float:
        """Add realistic measurement noise."""
        if noise_level is None:
            noise_level = BASE_NOISE_LEVEL
        noise: float = float(self.rng.normal(0, noise_level * value))
        return float(max(0, value + noise))

    def get_observable(self, name: str) -> float:
        """Get a single observable value with the mock function.

        Args:
            name: Observable name to retrieve

        Returns:
            Observable value as float

        Raises:
            SimulatorNotInitializedError: If variables not set before getting observables
        """
        # Stands in for waiting on an injection
        self.injection_counter += 1

        if name == INJECTION_BUCKET_PV:
            return self.injection_counter

        known_observables = ["linac_efficiency", GTL_CHARGE_PV, LTB_CHARGE_PV]
        if name in known_observables:
            return self._get_known_observable(name)

        if name in self.variable_names:
            if name in self.current_values:
                return self.current_values[name]
            bounds_min = VARIABLES[name].min_value
            bounds_max = VARIABLES[name].max_value
            return (bounds_min + bounds_max) / 2.0

        return 0.0

    def _get_known_observable(self, name: str) -> float:
        """Calculate value for known calculated observables."""
        if not self.current_values:
            raise SimulatorNotInitializedError(
                f"Simulator variables must be set before getting observable '{name}'. Call set_variables() first."
            )

        x = self._normalize_variables()
        efficiency = self._efficiency_function(x)

        if name == "linac_efficiency":
            return self._add_measurement_noise(efficiency, noise_level=EFFICIENCY_NOISE_LEVEL)

        if name == GTL_CHARGE_PV:
            # GTL charge: somewhat independent but affected by gun phase
            gun_phase_effect = 1.0 + GUN_PHASE_AMPLITUDE * np.sin(2 * np.pi * x[2])
            charge = GTL_BASE_CHARGE * gun_phase_effect
            return self._add_measurement_noise(charge, noise_level=CHARGE_NOISE_LEVEL)

        # Re-enters get_observable: the GTL branch adds its own noise draw (and bumps
        # injection_counter) before LTB scales it by efficiency.
        gtl_charge = self.get_observable(GTL_CHARGE_PV)
        ltb_charge = gtl_charge * efficiency
        return self._add_measurement_noise(ltb_charge, noise_level=CHARGE_NOISE_LEVEL)
