"""Beam dynamics engine for 1D longitudinal tracking.

This module handles particle state, relativistic kinematics, and element transport.
"""

from dataclasses import dataclass
from typing import Protocol, cast, overload, runtime_checkable

import numpy as np

from .constants import MIN_BETA, c, m_e_eV


@overload
def gamma(W_eV: float) -> float: ...


@overload
def gamma(W_eV: np.ndarray) -> np.ndarray: ...


def gamma(W_eV: float | np.ndarray) -> float | np.ndarray:
    """Calculate the Lorentz factor gamma from kinetic energy in eV."""
    return 1.0 + W_eV / m_e_eV


@overload
def beta(W_eV: float) -> float: ...


@overload
def beta(W_eV: np.ndarray) -> np.ndarray: ...


def beta(W_eV: float | np.ndarray) -> float | np.ndarray:
    """Calculate the relativistic velocity beta from kinetic energy in eV."""
    # Ensure W is non-negative to avoid domain errors
    W_safe = np.maximum(W_eV, 0.0)
    g = gamma(W_safe)
    return cast(float | np.ndarray, np.sqrt(1.0 - 1.0 / (g * g)))


def generate_bunch(
    n_particles: int, w_mean: float, t_fwhm: float, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Gaussian bunch of particles.

    Args:
        n_particles: Number of particles.
        w_mean: Mean kinetic energy [eV].
        t_fwhm: Full Width at Half Maximum of time distribution [s].
        seed: Random seed.

    Returns:
        Tuple of (t_array, w_array).
    """
    rng = np.random.default_rng(seed)
    sigma_t = t_fwhm / 2.355  # FWHM = 2*sqrt(2*ln(2))*sigma approx 2.355*sigma
    t = rng.normal(0, sigma_t, n_particles)
    # Minimal implementation: 0 energy spread as sigma_W is not in arguments
    w = rng.normal(w_mean, 0.0, n_particles)
    return t, w


@dataclass
class AcceptanceCriteria:
    """Parameters for acceptance calculation."""

    freq: float
    phase_window_deg: float
    w_target: float
    w_window_fraction: float


def phase_acceptance_mask(t: np.ndarray, freq: float, window_deg: float) -> np.ndarray:
    """Particles whose RF phase falls inside +/- ``window_deg`` of the crest.

    Phases are wrapped to [-pi, pi] so the window is centered on zero.
    """
    phi = (2 * np.pi * freq * t) % (2 * np.pi)
    phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    mask: np.ndarray = np.abs(phi) < np.deg2rad(window_deg)
    return mask


def compute_acceptance(
    t: np.ndarray,
    W: np.ndarray,
    criteria: AcceptanceCriteria,
) -> tuple[float, np.ndarray]:
    """Compute transmission fraction based on phase and energy acceptance.

    Args:
        t: Time array [s].
        W: Kinetic energy array [eV].
        criteria: Acceptance criteria parameters.

    Returns:
        Tuple of (transmission_fraction, alive_mask).
    """
    alive_mask = np.isfinite(t) & np.isfinite(W) & (W > 0)

    if not np.any(alive_mask):
        return 0.0, alive_mask

    t_alive = t[alive_mask]
    W_alive = W[alive_mask]

    phase_mask = phase_acceptance_mask(t_alive, criteria.freq, criteria.phase_window_deg)

    w_mask = np.abs(W_alive - criteria.w_target) / criteria.w_target < criteria.w_window_fraction

    accepted_mask_sub = phase_mask & w_mask

    final_mask = alive_mask.copy()
    final_mask[alive_mask] = accepted_mask_sub

    transmission = np.sum(accepted_mask_sub) / len(t)

    return transmission, final_mask


def rms_bunch_length(t: np.ndarray) -> float:
    """Calculate RMS bunch length in seconds."""
    mask = np.isfinite(t)
    if not np.any(mask):
        return 0.0
    return float(np.std(t[mask]))


def rms_energy_spread(W: np.ndarray) -> float:
    """Calculate RMS energy spread in eV."""
    mask = np.isfinite(W) & (W > 0)
    if not np.any(mask):
        return 0.0
    return float(np.std(W[mask]))


def mean_energy(W: np.ndarray) -> float:
    """Calculate mean energy in eV."""
    mask = np.isfinite(W) & (W > 0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(W[mask]))


def apply_drift(t: np.ndarray, W: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply drift space.

    Args:
        t: Time array [s].
        W: Kinetic energy array [eV].
        length: Drift length [m].

    Returns:
        Updated (t, W).
    """
    b = beta(W)
    # Stopped particles (beta <= MIN_BETA): time of flight is infinite.
    mask = b > MIN_BETA
    t_new = t.copy()
    t_new[mask] += length / (b[mask] * c)
    t_new[~mask] = np.inf
    return t_new, W


def apply_rf_cavity(
    t: np.ndarray, W: np.ndarray, freq: float, v_eff: float, phase_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply thin RF kick.

    Args:
        t: Time array [s].
        W: Kinetic energy array [eV].
        freq: RF frequency [Hz].
        v_eff: Effective voltage [V].
        phase_deg: Phase [deg].

    Returns:
        Updated (t, W).
    """
    # Handle infinite time (stopped particles)
    mask = np.isfinite(t)

    delta_W = np.zeros_like(W)
    if np.any(mask):
        phi = 2 * np.pi * freq * t[mask] + np.deg2rad(phase_deg)
        delta_W[mask] = v_eff * np.cos(phi)

    return t, W + delta_W


@runtime_checkable
class BeamElement(Protocol):
    """Protocol for beamline elements."""

    def apply(self, t: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply element effect to beam."""
        ...


@dataclass
class Drift:
    """Drift space element."""

    length: float

    def apply(self, t: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return apply_drift(t, W, self.length)


@dataclass
class RFCavity:
    """RF Cavity element."""

    freq: float
    v_eff: float
    phase_deg: float

    def apply(self, t: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return apply_rf_cavity(t, W, self.freq, self.v_eff, self.phase_deg)


def track_beam(
    t_initial: np.ndarray, W_initial: np.ndarray, lattice: list[BeamElement]
) -> tuple[np.ndarray, np.ndarray]:
    """Track beam through a lattice of elements.

    Args:
        t_initial: Initial time array [s].
        W_initial: Initial kinetic energy array [eV].
        lattice: List of beamline elements.

    Returns:
        Final (t, W).
    """
    t, W = t_initial.copy(), W_initial.copy()
    for element in lattice:
        t, W = element.apply(t, W)
    return t, W
