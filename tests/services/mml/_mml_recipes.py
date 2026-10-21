"""The Middle Layer's model-mode arithmetic, ported line for line onto pyAT's tracker.

The Middle Layer answers its model questions through Accelerator Toolbox 2.0's
Matlab functions (``simulators/at2.0/atmat``) and its own model wrappers
(``mml/simulators/at``). Every function here is one of those, ported with its
own step sizes, its own finite-difference layout and its own convergence rule,
over ``at.lattice_pass`` -- the same tracking kernels the Matlab functions call
through ``linepass``. pyAT's own solvers take different steps and converge
differently, which leaves answers a few parts in 1e-8 apart where a finite
difference divides by a small step; a port repeats the Middle Layer's steps, so
the only difference left is floating-point rounding.

Positions are 0-based deck indices; ``len(lattice)`` is the lattice's end. Every
function works on the lattice it is handed and leaves it as it found it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import at
import numpy as np

#: AT's speed of light, the constant every RF formula of the Middle Layer uses.
C_LIGHT = 299792458.0

#: ``findm44``/``findm66``'s default transverse and momentum step.
MATRIX_STEP = 6.055454452393343e-06

#: ``findorbit4``/``findsyncorbit``/``findorbit6``'s Jacobian step.
ORBIT_STEP = 1e-6

#: ``findorbit4``/``findsyncorbit``/``findorbit6``'s convergence threshold on the
#: norm of the Newton update.
ORBIT_CONVERGENCE = 1e-12

#: The Newton solvers' iteration cap.
ORBIT_ITERATIONS = 20

#: ``mcf``'s momentum step.
MCF_STEP = 1e-6

#: ``tunechrom``'s default momentum step for the chromaticity.
TUNECHROM_STEP = 1e-8

#: ``locoresponsematrix``'s momentum step for the dispersion of its Linear calculator.
LOCO_DISPERSION_STEP = 1e-5


def track(lattice: Sequence[Any], rin: np.ndarray, refpts: Sequence[int] | int | None = None):
    """``linepass``: track the columns of ``rin`` once, reading them at ``refpts``.

    Returns:
        ``(6, particles, len(refpts))``; ``refpts=None`` reads the lattice's end.
    """
    where = [len(lattice)] if refpts is None else list(np.atleast_1d(refpts))
    start = np.asfortranarray(np.array(rin, dtype=float).reshape(6, -1))
    out = at.lattice_pass(lattice, start.copy(order="F"), refpts=where)
    return out[:, :, :, 0]


def circumference(lattice: Sequence[Any]) -> float:
    """``findspos(RING, length(RING)+1)``: the sum of the element lengths."""
    return float(sum(float(getattr(element, "Length", 0.0)) for element in lattice))


def cavities(lattice: Sequence[Any]) -> list[int]:
    """``findcells(THERING, 'Frequency')``: every element carrying an RF frequency."""
    return [index for index, element in enumerate(lattice) if hasattr(element, "Frequency")]


def set_cavity(lattice: Sequence[Any], state: str) -> None:
    """``setcavity``: ``'On'`` passes every cavity through ``CavityPass``; ``'Off'``
    through ``IdentityPass`` (no length) or ``DriftPass``."""
    for index in cavities(lattice):
        element = lattice[index]
        if state == "On":
            element.PassMethod = "CavityPass"
        elif state == "Off":
            length = float(getattr(element, "Length", 0.0))
            element.PassMethod = "IdentityPass" if length == 0 else "DriftPass"
        else:
            element.PassMethod = state


def _newton(start: np.ndarray, step: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """The Newton loop every AT 2.0 orbit finder shares: ``Ri += (I - J) \\ r``,
    stopped once the update's norm is not above ``ORBIT_CONVERGENCE``."""
    guess = start.copy()
    for _ in range(ORBIT_ITERATIONS):
        following = step(guess)
        change = float(np.linalg.norm(following - guess))
        guess = following
        if not change > ORBIT_CONVERGENCE:
            break
    return guess


def findorbit4(lattice: Sequence[Any], dp: float, refpts: Sequence[int] | None = None):
    """``findorbit4(RING, dP, REFPTS)``: the 4D closed orbit at fixed momentum.

    Returns:
        The fixed point (6,) and, with ``refpts``, the orbit there as ``(6, n)``.
    """

    def step(ri: np.ndarray) -> np.ndarray:
        rin = np.tile(ri[:, None], (1, 5))
        rin[:4, :4] += ORBIT_STEP * np.eye(4)
        rout = track(lattice, rin)[:, :, 0]
        jac = (rout[:4, :4] - rout[:4, [4] * 4]) / ORBIT_STEP
        update = np.linalg.solve(np.eye(4) - jac, rout[:4, 4] - ri[:4])
        return ri + np.concatenate([update, [0.0, 0.0]])

    start = np.zeros(6)
    start[4] = dp
    fixed = _newton(start, step)
    if refpts is None:
        return fixed, None
    return fixed, track(lattice, fixed, refpts)[:, 0, :]


def findsyncorbit(lattice: Sequence[Any], dct: float, refpts: Sequence[int] | None = None):
    """``findsyncorbit(RING, dCT, REFPTS)``: the closed orbit whose path length
    changes by ``dct`` per turn, solved over ``(x, x', y, y', dp)``."""
    theta = np.array([0.0, 0.0, 0.0, 0.0, dct])

    def step(ri: np.ndarray) -> np.ndarray:
        rin = np.tile(ri[:, None], (1, 6))
        rin[:5, :5] += ORBIT_STEP * np.eye(5)
        rout = track(lattice, rin)[:, :, 0]
        rows = [0, 1, 2, 3, 5]
        jac = (rout[np.ix_(rows, range(5))] - rout[rows][:, [5] * 5]) / ORBIT_STEP
        rhs = rout[rows, 5] - np.concatenate([ri[:4], [0.0]]) - theta
        update = np.linalg.solve(np.diag([1.0, 1.0, 1.0, 1.0, 0.0]) - jac, rhs)
        return ri + np.concatenate([update, [0.0]])

    fixed = _newton(np.zeros(6), step)
    if refpts is None:
        return fixed, None
    return fixed, track(lattice, fixed, refpts)[:, 0, :]


def findorbit6(lattice: Sequence[Any], refpts: Sequence[int] | None = None):
    """``findorbit6(RING, REFPTS)``: the 6D closed orbit, the first cavity's
    frequency and harmonic number setting the revolution time."""
    index = cavities(lattice)
    if not index:
        raise ValueError("findorbit6: The lattice does not have Cavity element")
    cavity = lattice[index[0]]
    period = circumference(lattice) / C_LIGHT
    theta = np.zeros(6)
    theta[5] = C_LIGHT * (float(cavity.HarmNumber) / float(cavity.Frequency) - period)

    def step(ri: np.ndarray) -> np.ndarray:
        rin = np.tile(ri[:, None], (1, 7))
        rin[:, :6] += ORBIT_STEP * np.eye(6)
        rout = track(lattice, rin)[:, :, 0]
        jac = (rout[:, :6] - rout[:, [6] * 6]) / ORBIT_STEP
        return ri + np.linalg.solve(np.eye(6) - jac, rout[:, 6] - ri - theta)

    fixed = _newton(np.zeros(6), step)
    if refpts is None:
        return fixed, None
    return fixed, track(lattice, fixed, refpts)[:, 0, :]


def findm44(lattice: Sequence[Any], dp: float, refpts: Sequence[int] | None = None):
    """``findm44(RING, dP, REFPTS)``: the one-turn 4x4 matrix by central
    differences of ``MATRIX_STEP`` about the 4D closed orbit.

    Returns:
        ``M44`` and, with ``refpts``, the ``(4, 4, n)`` matrices from the start
        to each point, and the closed orbit ``(6, n)`` there.
    """
    fixed, _ = findorbit4(lattice, dp)
    fixed[4], fixed[5] = dp, 0.0
    points = [len(lattice)] if refpts is None else sorted(set(refpts) | {len(lattice)})
    rin = np.tile(fixed[:, None], (1, 9))
    rin[:4, :4] += 0.5 * MATRIX_STEP * np.eye(4)
    rin[:4, 4:8] -= 0.5 * MATRIX_STEP * np.eye(4)
    rout = track(lattice, rin, points)
    stack = (rout[:4, 0:4, :] - rout[:4, 4:8, :]) / MATRIX_STEP
    m44 = stack[:, :, points.index(len(lattice))]
    if refpts is None:
        return m44, None, None
    where = [points.index(position) for position in refpts]
    return m44, stack[:, :, where], rout[:, 8, where]


def findm66(lattice: Sequence[Any]) -> np.ndarray:
    """``findm66(RING)``: the one-turn 6x6 matrix by central differences about
    the 6D closed orbit."""
    fixed, _ = findorbit6(lattice)
    rin = np.tile(fixed[:, None], (1, 13))
    delta = np.diag([0.5 * MATRIX_STEP] * 6)
    rin[:, 0:6] += delta
    rin[:, 6:12] -= delta
    rout = track(lattice, rin)[:, :, 0]
    return (rout[:, 0:6] - rout[:, 6:12]) / MATRIX_STEP


def getnusympmat(matrix: np.ndarray) -> np.ndarray:
    """``getnusympmat(M)``: the fractional tunes of the upper 4x4 block by the
    characteristic-polynomial formula, each folded by the sign of ``M(j, j+1)``."""
    m = np.array(matrix, dtype=float)[:4, :4]
    detp = np.linalg.det(m - np.eye(4))
    detm = np.linalg.det(m + np.eye(4))
    b = (detp - detm) / 16.0
    c = (detp + detm) / 8.0 - 1.0
    th = (m[0, 0] + m[1, 1]) / 2.0
    tv = (m[2, 2] + m[3, 3]) / 2.0
    b2mc = b * b - c
    if b2mc < 0.0:
        return np.array([-1.0, -1.0])
    sgn = 1.0 if th > tv else -1.0
    nu = np.array(
        [
            math.acos(sgn * math.sqrt(b2mc) - b) / (2.0 * math.pi),
            math.acos(-b - sgn * math.sqrt(b2mc)) / (2.0 * math.pi),
        ]
    )
    for plane in range(2):
        j = 2 * plane
        if m[j, j + 1] < 0.0:
            nu[plane] = 1.0 - nu[plane]
    return nu


def tunechrom(lattice: Sequence[Any], dp: float) -> np.ndarray:
    """``tunechrom(RING, dP)``: ``acos`` of the half-traces of ``findm44``, in [0, 0.5]."""
    m44, _, _ = findm44(lattice, dp)
    cosines = np.array([(m44[0, 0] + m44[1, 1]) / 2.0, (m44[2, 2] + m44[3, 3]) / 2.0])
    return np.arccos(cosines) / (2.0 * math.pi)


def tunechrom_chromaticity(lattice: Sequence[Any], dp: float = 0.0) -> np.ndarray:
    """``[~, chrom] = tunechrom(RING, dP, 'chrom')``: a one-sided difference over
    ``TUNECHROM_STEP``."""
    return (tunechrom(lattice, dp + TUNECHROM_STEP) - tunechrom(lattice, dp)) / TUNECHROM_STEP


def twissring_tune(lattice: Sequence[Any]) -> np.ndarray:
    """``rem(tune, 1)`` of ``twissring(RING, 0, ...)``: the phase of ``findm44``'s
    one-turn matrix, its sine signed by ``M(1,2)``."""
    m44, _, _ = findm44(lattice, 0.0)
    tunes = []
    for j in (0, 2):
        cos_mu = (m44[j, j] + m44[j + 1, j + 1]) / 2.0
        sin_mu = math.copysign(
            math.sqrt(-m44[j, j + 1] * m44[j + 1, j] - (m44[j, j] - m44[j + 1, j + 1]) ** 2 / 4.0),
            m44[j, j + 1],
        )
        tunes.append(math.atan2(sin_mu, cos_mu) / (2.0 * math.pi) % 1.0)
    return np.array(tunes)


def mcf(lattice: Sequence[Any], dp0: float = 0.0) -> float:
    """``mcf(RING)``: the path-length change of one turn from two fixed points
    ``MCF_STEP`` apart in momentum, per unit momentum and circumference."""
    fp0, _ = findorbit4(lattice, dp0)
    fp, _ = findorbit4(lattice, dp0 + MCF_STEP)
    x0dp = fp.copy()
    x0dp[4], x0dp[5] = MCF_STEP, 0.0
    x0 = np.concatenate([fp0[:4], [0.0, 0.0]])
    out = track(lattice, np.column_stack([x0, x0dp]))[:, :, 0]
    return float((out[5, 1] - out[5, 0]) / (MCF_STEP * circumference(lattice)))


def modeltune(lattice: Sequence[Any]) -> np.ndarray:
    """``modeltune``: ``getnusympmat(findm66)`` with the cavities switched on,
    ``twissring``'s tunes on a lattice with none."""
    index = cavities(lattice)
    if not index:
        return twissring_tune(lattice)
    held = [lattice[i].PassMethod for i in index]
    try:
        set_cavity(lattice, "On")
        return getnusympmat(findm66(lattice))
    finally:
        for i, method in zip(index, held, strict=True):
            lattice[i].PassMethod = method


def modelchro(
    lattice: Sequence[Any], delta_rf_hz: float = 1.0, *, hardware_step: float | None = None
) -> np.ndarray:
    """``modelchro(DeltaRF, 'Physics')``.

    With a cavity: the cavities switched on, ``getnusympmat(findm66)`` at the
    first cavity's frequency and ``delta_rf_hz`` above it (one-sided), scaled by
    ``-mcf * RF0`` with ``mcf`` taken on the lattice as ``modelchro`` holds it --
    cavities on -- at every call. ``modelchro(..., 'Hardware')`` answers the
    tune change per ``hardware_step``, the RF step in the RF family's hardware
    unit, with no momentum compaction. Without a cavity: ``tunechrom``'s 4D
    chromaticity, in physics units whatever was asked.
    """
    index = cavities(lattice)
    if not index:
        return tunechrom_chromaticity(lattice)
    held = [lattice[i].PassMethod for i in index]
    frequencies = [float(lattice[i].Frequency) for i in index]
    try:
        set_cavity(lattice, "On")
        before = getnusympmat(findm66(lattice))
        rf0 = frequencies[0]
        for i in index:
            lattice[i].Frequency = rf0 + delta_rf_hz
        after = getnusympmat(findm66(lattice))
        for i in index:
            lattice[i].Frequency = rf0
        if hardware_step is not None:
            return (after - before) / hardware_step
        compaction = mcf(lattice)
        return (after - before) / delta_rf_hz * (-compaction * rf0)
    finally:
        for i, method, frequency in zip(index, held, frequencies, strict=True):
            lattice[i].PassMethod = method
            lattice[i].Frequency = frequency


def findelemm44(element: Any, orbit: np.ndarray) -> np.ndarray:
    """``findelemm44(ELEM, PassMethod, R0)``: one element's 4x4 matrix by central
    differences of ``MATRIX_STEP`` about ``orbit``."""
    rin = np.tile(np.asarray(orbit, dtype=float)[:, None], (1, 8))
    rin[:4, 0:4] += 0.5 * MATRIX_STEP * np.eye(4)
    rin[:4, 4:8] -= 0.5 * MATRIX_STEP * np.eye(4)
    rout = track([element], rin)[:, :, 0]
    return (rout[:4, 0:4] - rout[:4, 4:8]) / MATRIX_STEP


def loco_linear(
    lattice: Sequence[Any],
    correctors: Sequence[tuple[int, int]],
    monitors: Sequence[int],
    compaction: float,
) -> np.ndarray:
    """``locoresponsematrix``'s Linear calculator at fixed path length, per radian.

    Each corrector ``(position, plane)`` (plane 0 horizontal, 1 vertical) kicks
    one radian, split half before and half after its element's matrix; the
    orbit it closes is propagated by ``findm44``'s matrices, and the path
    length a kick adds is taken off with ``locoresponsematrix``'s one-sided
    ``LOCO_DISPERSION_STEP`` dispersion: ``theta * (eta_in + eta_out) / 2 *
    eta_bpm / (L0 * mcf)``, ``eta_in``/``eta_out`` the horizontal dispersion at
    the corrector's entrance and exit.

    Returns:
        ``(correctors, monitors, 2)``: the ``x`` and ``y`` response.
    """
    end = len(lattice)
    everywhere = list(range(end + 1))
    m44, stack, orbit = findm44(lattice, 0.0, everywhere)
    _, shifted = findorbit4(lattice, LOCO_DISPERSION_STEP, everywhere)
    eta = (shifted[:4] - orbit[:4]) / LOCO_DISPERSION_STEP
    length = circumference(lattice)
    identity = np.eye(4)
    columns = []
    for position, plane in correctors:
        theta = np.zeros(4)
        theta[1 if plane == 0 else 3] = 1.0
        own = findelemm44(lattice[position], np.concatenate([orbit[:4, position], [0.0, 0.0]]))
        before = stack[:, :, position]
        inverse = np.linalg.inv(before)
        turn = before @ m44 @ inverse
        entrance = (
            np.linalg.inv(identity - turn) @ turn @ (identity + np.linalg.inv(own)) @ theta / 2.0
        )
        exit_ = theta / 2.0 + own @ (entrance + theta / 2.0)
        start = np.linalg.inv(stack[:, :, position + 1]) @ exit_
        column = np.array(
            [stack[[0, 2], :, bpm] @ (start if bpm > position else m44 @ start) for bpm in monitors]
        )
        path = (
            theta[1]
            * (eta[0, position] + eta[0, position + 1])
            * eta[[0, 2]][:, monitors]
            / length
            / compaction
            / 2.0
        )
        columns.append(column - path.T)
    return np.array(columns)


def loco_transport_full(
    line: Sequence[Any],
    start: np.ndarray,
    correctors: Sequence[tuple[int, int]],
    kicks: Sequence[float],
    monitors: Sequence[int],
) -> np.ndarray:
    """``locoresponsematrix``'s Full calculator on a transport line, bidirectional.

    Each corrector's ``KickAngle`` is moved by ``+kick/2`` and ``-kick/2`` on its
    plane, the launch orbit ``start`` is tracked to the monitors each time, and
    the difference is divided by the kick, as ``measbpmresp`` divides it.

    Returns:
        ``(correctors, monitors, 2)``: the ``x`` and ``y`` response.
    """
    columns = []
    for (position, plane), kick in zip(correctors, kicks, strict=True):
        element = line[position]
        held = np.array(element.KickAngle, dtype=float)
        arms = []
        try:
            for sign in (1.0, -1.0):
                moved = held.copy()
                moved[plane] += sign * kick / 2.0
                element.KickAngle = moved
                arms.append(track(line, start, monitors)[:, 0, :])
        finally:
            element.KickAngle = held
        columns.append(((arms[0] - arms[1])[[0, 2]] / kick).T)
    return np.array(columns)


def twissline_dispersion(
    line: Sequence[Any], twiss_in: dict[str, Any], refpts: Sequence[int]
) -> np.ndarray:
    """``twissline(LINE, 0, TWISSDATAIN, REFPTS, 'Chrom')``'s dispersion: the launch
    orbit tracked at momentum 0 and at ``TUNECHROM_STEP``, the second launched off
    by the input dispersion times the step.

    Returns:
        ``(4, len(refpts))``.
    """
    launch = np.zeros(6)
    launch[:4] = np.ravel(twiss_in["ClosedOrbit"]).astype(float)
    shifted = launch.copy()
    shifted[:4] += np.ravel(twiss_in["Dispersion"]).astype(float) * TUNECHROM_STEP
    shifted[4] = TUNECHROM_STEP
    on = track(line, launch, refpts)[:, 0, :]
    off = track(line, shifted, refpts)[:, 0, :]
    return (off[:4] - on[:4]) / TUNECHROM_STEP
