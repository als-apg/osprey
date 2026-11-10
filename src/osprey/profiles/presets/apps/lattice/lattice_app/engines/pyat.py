"""The pyAT engine: every analysis, computed with the Accelerator Toolbox.

The model's ``engine_settings`` decide how pyAT integrates and differentiates:

``bend_pass``
    The pass method of every dipole, e.g. ``ExactSectorBendPass``.
``pass_methods``
    Pass methods by element class, e.g. ``{Quadrupole: ExactMultipolePass}``.
``dp_step``, ``xy_step``
    The finite-difference steps of the optics and chromaticity (pyAT's ``DPStep`` and
    ``XYStep``); pyAT's defaults when omitted.
``use``
    The beamline of an elegant or MAD-X deck, or the variable of a ``.mat`` file.
``six_d``
    ``true`` keeps cavities and radiation on. They are off by default.

Run as ``python -m lattice_app.engines.pyat <job.json> <out.json>``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import at
import numpy as np

from lattice_app.engines import run_engine

Job = Mapping[str, Any]


def load_ring(job: Job) -> at.Lattice:
    """The job's deck as a ring, with the model's engine settings applied."""
    settings = job.get("engine_settings") or {}
    options: dict[str, Any] = {}
    if settings.get("use"):
        options["use"] = settings["use"]
    if job.get("energy_gev"):
        options["energy"] = float(job["energy_gev"]) * 1.0e9
    elif str(job["deck"]).lower().endswith(".lte"):
        # An elegant deck carries no energy, and pyAT would assume 1 GeV without saying so.
        raise ValueError("an elegant deck names no energy: set the model's energies_gev")
    ring = at.load_lattice(job["deck"], **options)
    if ring.is_6d and not settings.get("six_d"):
        ring = ring.disable_6d(copy=True)
    by_class = dict(settings.get("pass_methods") or {})
    bend_pass = settings.get("bend_pass")
    for element in ring:
        if bend_pass and isinstance(element, at.Dipole):
            element.PassMethod = bend_pass
        if (pass_method := by_class.get(type(element).__name__)) is not None:
            element.PassMethod = pass_method
    return ring


def steps(job: Job) -> dict[str, float]:
    """The finite-difference steps the model sets, as pyAT's keywords."""
    settings = job.get("engine_settings") or {}
    named = {"DPStep": settings.get("dp_step"), "XYStep": settings.get("xy_step")}
    return {key: float(value) for key, value in named.items() if value is not None}


def survives(ring: at.Lattice, particles: np.ndarray, nturns: int) -> np.ndarray:
    """Which of *particles* (one per column) are still in the ring after *nturns*."""
    out = ring.track(np.asfortranarray(particles), nturns=nturns)
    out = out[0] if isinstance(out, tuple) else out
    alive: np.ndarray = np.all(np.isfinite(out[:, :, -1, -1]), axis=0)
    return alive


def optics(job: Job) -> dict[str, Any]:
    """Tunes, chromaticities and the Twiss functions around the ring."""
    ring = load_ring(job)
    _, ringdata, data = ring.get_optics(refpts=range(len(ring) + 1), get_chrom=True, **steps(job))
    tunes = data.mu[-1] / (2 * np.pi)
    s = data.s_pos
    lengths = [float(getattr(element, "Length", 0.0)) for element in ring]
    return {
        "scalars": {
            "tune_x": float(tunes[0]),
            "tune_y": float(tunes[1]),
            "chromaticity_x": float(ringdata.chromaticity[0]),
            "chromaticity_y": float(ringdata.chromaticity[1]),
            "circumference_m": float(s[-1]),
            "energy_gev": float(ring.energy) / 1.0e9,
            "beta_x_max": float(np.max(data.beta[:, 0])),
            "beta_y_max": float(np.max(data.beta[:, 1])),
            "eta_x_max": float(np.max(np.abs(data.dispersion[:, 0]))),
        },
        "units": {
            "circumference_m": "m",
            "energy_gev": "GeV",
            "beta_x_max": "m",
            "beta_y_max": "m",
            "eta_x_max": "m",
        },
        "tables": {
            "twiss": {
                "columns": {
                    "s": s.tolist(),
                    "beta_x": data.beta[:, 0].tolist(),
                    "beta_y": data.beta[:, 1].tolist(),
                    "alpha_x": data.alpha[:, 0].tolist(),
                    "alpha_y": data.alpha[:, 1].tolist(),
                    "eta_x": data.dispersion[:, 0].tolist(),
                    "eta_y": data.dispersion[:, 2].tolist(),
                    "mu_x": data.mu[:, 0].tolist(),
                    "mu_y": data.mu[:, 1].tolist(),
                },
                "units": {"s": "m", "beta_x": "m", "beta_y": "m", "eta_x": "m", "eta_y": "m"},
            },
            "elements": {
                "columns": {
                    "name": [str(element.FamName) for element in ring],
                    "kind": [type(element).__name__ for element in ring],
                    "s": s[:-1].tolist(),
                    "length": lengths,
                },
                "units": {"s": "m", "length": "m"},
            },
        },
    }


def chromaticity(job: Job) -> dict[str, Any]:
    """The tunes against momentum offset, and the chromatic orders fitted to them."""
    ring = load_ring(job)
    settings = job["settings"]
    offsets = np.linspace(
        float(settings["dp_min_pct"]), float(settings["dp_max_pct"]), int(settings["n_steps"])
    )
    tunes = np.full((len(offsets), 2), np.nan)
    for index, offset in enumerate(offsets):
        try:
            _, ringdata, _ = ring.get_optics(dp=offset / 100.0, **steps(job))
        except (at.AtError, np.linalg.LinAlgError, ValueError):
            continue  # no stable optics at this offset: the curve has a gap there
        tunes[index] = ringdata.tune
    scalars: dict[str, float] = {}
    stable = np.all(np.isfinite(tunes), axis=1)
    if np.count_nonzero(stable) >= 4:
        for plane, column in (("x", 0), ("y", 1)):
            fitted = np.polyfit(offsets[stable] / 100.0, tunes[stable, column], 3)
            scalars[f"chromaticity_{plane}"] = float(fitted[2])
            scalars[f"chromaticity2_{plane}"] = float(fitted[1])
            scalars[f"chromaticity3_{plane}"] = float(fitted[0])
    return {
        "scalars": scalars,
        "tables": {
            "tune_vs_dp": {
                "columns": {
                    "dp_pct": offsets.tolist(),
                    "tune_x": tunes[:, 0].tolist(),
                    "tune_y": tunes[:, 1].tolist(),
                },
                "units": {"dp_pct": "%"},
            }
        },
    }


def da(job: Job) -> dict[str, Any]:
    """The dynamic aperture at the start of the ring: the largest stable amplitude per angle."""
    ring = load_ring(job)
    settings = job["settings"]
    nturns = int(settings["nturns"])
    angles = np.linspace(0.0, np.pi, int(settings["n_angles"]))
    low = np.zeros(len(angles))
    high = np.full(len(angles), float(settings["amp_max_mm"]) * 1.0e-3)
    for _ in range(int(settings["n_bisect"])):
        mid = (low + high) / 2
        particles = np.zeros((6, len(angles)))
        particles[0] = mid * np.cos(angles)
        particles[2] = mid * np.sin(angles)
        stable = survives(ring, particles, nturns)
        low, high = np.where(stable, mid, low), np.where(stable, high, mid)
    x, y = low * np.cos(angles) * 1.0e3, low * np.sin(angles) * 1.0e3
    # Shoelace over the half-plane's boundary, closed through the origin.
    xs, ys = np.append(x, 0.0), np.append(y, 0.0)
    area = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
    return {
        "scalars": {
            "area_mm2": area,
            "x_max_mm": float(np.max(x)),
            "x_min_mm": float(np.min(x)),
            "y_max_mm": float(np.max(y)),
        },
        "units": {"area_mm2": "mm^2", "x_max_mm": "mm", "x_min_mm": "mm", "y_max_mm": "mm"},
        "tables": {
            "aperture": {
                "columns": {
                    "angle_deg": np.degrees(angles).tolist(),
                    "x_mm": x.tolist(),
                    "y_mm": y.tolist(),
                },
                "units": {"angle_deg": "deg", "x_mm": "mm", "y_mm": "mm"},
            }
        },
    }


def lma(job: Job) -> dict[str, Any]:
    """The local momentum aperture: the largest stable momentum kick at points round the ring."""
    ring = load_ring(job)
    settings = job["settings"]
    nturns = int(settings["nturns"])
    limit = float(settings["dp_max_pct"]) / 100.0
    refpts = np.unique(np.linspace(0, len(ring) - 1, int(settings["n_refpts"])).astype(int))
    positions = ring.get_s_pos(refpts)
    plus, minus = [], []
    for refpt in refpts:
        turn = ring.rotate(int(refpt))
        orbit, _ = turn.find_orbit()
        low, high = np.zeros(2), np.full(2, limit)
        signs = np.array([1.0, -1.0])
        for _ in range(int(settings["n_bisect"])):
            mid = (low + high) / 2
            particles = np.tile(orbit.reshape(6, 1), (1, 2))
            particles[4] += signs * mid
            stable = survives(turn, particles, nturns)
            low, high = np.where(stable, mid, low), np.where(stable, high, mid)
        plus.append(float(low[0]) * 100.0)
        minus.append(-float(low[1]) * 100.0)
    return {
        "scalars": {"dp_plus_min_pct": min(plus), "dp_minus_max_pct": max(minus)},
        "units": {"dp_plus_min_pct": "%", "dp_minus_max_pct": "%"},
        "tables": {
            "aperture": {
                "columns": {
                    "s": np.asarray(positions).tolist(),
                    "dp_plus_pct": plus,
                    "dp_minus_pct": minus,
                },
                "units": {"s": "m", "dp_plus_pct": "%", "dp_minus_pct": "%"},
            }
        },
    }


def peak_tune(signal: np.ndarray) -> float:
    """The frequency of a complex turn-by-turn signal's strongest line, in ``[0, 1)``."""
    turns = len(signal)
    spectrum = np.abs(np.fft.fft((signal - np.mean(signal)) * np.hanning(turns)))
    peak = int(np.argmax(spectrum))
    before, at_peak, after = (spectrum[(peak + shift) % turns] for shift in (-1, 0, 1))
    # A Hann-windowed line's neighbours place it between the bins.
    if after >= before:
        shift = (2 * after - at_peak) / (at_peak + after)
    else:
        shift = -(2 * before - at_peak) / (at_peak + before)
    return float(((peak + shift) / turns) % 1.0)


def footprint(job: Job) -> dict[str, Any]:
    """The tunes over a grid of amplitudes, and how far each drifts between two halves."""
    ring = load_ring(job)
    settings = job["settings"]
    half = int(settings["n_half"])
    count = int(settings["n_amp"])
    x = np.linspace(0.0, float(settings["x_max_mm"]), count + 1)[1:] * 1.0e-3
    y = np.linspace(0.0, float(settings["y_max_mm"]), count + 1)[1:] * 1.0e-3
    grid_x, grid_y = (axis.ravel() for axis in np.meshgrid(x, y))
    start, _, _ = ring.get_optics()
    orbit = np.asarray(start.closed_orbit).reshape(6)
    particles = np.tile(orbit.reshape(6, 1), (1, len(grid_x)))
    particles[0] += grid_x
    particles[2] += grid_y
    out = ring.track(np.asfortranarray(particles), nturns=2 * half)
    out = out[0] if isinstance(out, tuple) else out
    turns = out[:, :, -1, :] - orbit.reshape(6, 1, 1)
    tunes = np.full((len(grid_x), 2, 2), np.nan)  # particle, half, plane
    for index in range(len(grid_x)):
        if not np.all(np.isfinite(turns[:, index, :])):
            continue
        for plane, (position, angle) in enumerate(((0, 1), (2, 3))):
            alpha, beta = float(start.alpha[plane]), float(start.beta[plane])
            # In normalised coordinates the motion is a rotation, so its line is one-sided
            # and the tune is read over the whole unit interval.
            signal = turns[position, index] - 1j * (
                alpha * turns[position, index] + beta * turns[angle, index]
            )
            tunes[index, 0, plane] = peak_tune(signal[:half])
            tunes[index, 1, plane] = peak_tune(signal[half:])
    drift = tunes[:, 1, :] - tunes[:, 0, :]
    drift = (drift + 0.5) % 1.0 - 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        diffusion = np.log10(np.hypot(drift[:, 0], drift[:, 1]))
    diffusion = np.where(np.isneginf(diffusion), -16.0, diffusion)
    lost = int(np.count_nonzero(~np.isfinite(tunes[:, 0, 0])))
    return {
        "scalars": {"particles": len(grid_x), "lost": lost},
        "tables": {
            "tunes": {
                "columns": {
                    "x_mm": (grid_x * 1.0e3).tolist(),
                    "y_mm": (grid_y * 1.0e3).tolist(),
                    "tune_x": tunes[:, 0, 0].tolist(),
                    "tune_y": tunes[:, 0, 1].tolist(),
                    "diffusion": diffusion.tolist(),
                },
                "units": {"x_mm": "mm", "y_mm": "mm"},
            }
        },
    }


ANALYSES = {
    "optics": optics,
    "chromaticity": chromaticity,
    "da": da,
    "lma": lma,
    "footprint": footprint,
}

if __name__ == "__main__":
    sys.exit(run_engine(ANALYSES))
