#!/usr/bin/env python3
"""Derive QF/QD/QFA/DIPOLE current bands from the real ALS-U AR ring.

This is the sole re-derivation path for the quad/dipole ``:CURRENT:SP``
min/max bands in ``channel_limits.json`` -- hand-edited bands are not
acceptable. It sweeps each device's current on the real ring (built by
``osprey.services.virtual_accelerator.lattice.build_ring``) and locates the
one-turn-map stability edge in both directions.

Edge rule (QF/QD/QFA and, informationally, DIPOLE)
---------------------------------------------------
Starting from nominal current (``StrengthMap.i_nom``), step outward in each
direction at a fixed 0.25% of nominal per step. At every step the ring's
one-turn matrix is computed (``at.find_m44``, mirroring
``lattice.solve.solve_orbit``'s guard) and reduced to a single scalar, the
max-plane transverse trace ``max(|m44[0,0]+m44[1,1]|, |m44[2,2]+m44[3,3]|)``.
The sweep stops at the *first* of two conditions to trigger -- which is, by
construction, the tighter of the two:

  (a) the max-plane trace reaches ``TRACE_EDGE`` (1.8, a safety margin below
      the hard |trace| >= 2.0 instability guard) -- the edge current is the
      linear interpolation, on current, between the last sub-threshold
      sample and this one;
  (b) `find_m44` returns a non-finite one-turn matrix -- the edge current is
      the *last finite* sample (one step back). This governs whenever a
      sweep diverges without ever crossing the trace threshold (observed for
      positive-trim DIPOLE: PolynomB[0] blows up before |trace| reaches 1.8).

Each direction sweeps independently from a freshly restored element state
(``snapshot_element``/``restore_element``) so sweeps never interact.  A
sweep that reaches neither condition within ``MAX_DEVIATION_FRACTION`` of
nominal (a generous safety valve, not a physical bound -- see module
constant) is a hard error: it means the valve is set too tight for that
device, not a valid result.

DIPOLE policy vs. derived boundary
----------------------------------
DIPOLE's *committed* band is not the derived stability edge -- it is the
fixed +/-0.5% policy trim window around nominal (``DIPOLE_POLICY_FRACTION``).
The derived boundary above is still computed for every DIPOLE device and
used only to assert an invariant: the policy window must sit strictly
inside the derived stability boundary with at least 2x headroom on the
tighter side (``DIPOLE_HEADROOM_FACTOR``). Violating this invariant is a
hard error -- it would mean the policy window itself is not a safe
operating range on the real ring.

Unipolar-supply floor (QF/QD/QFA)
---------------------------------
The QF/QD/QFA committed band minimum is floored at 0 A
(``UNIPOLAR_FLOOR_FAMILIES``). This is a supply-polarity policy decision --
these families are wired to unipolar power supplies that cannot deliver
negative current -- not a claim about ring stability. The real ring stays
stable under polarity reversal for at least the QD family (its derived lower
edge is negative, around -365 A near a ~285.8 A nominal); the floor discards
that negative headroom rather than exposing an unreachable setpoint. Only
devices whose derived lower edge is actually negative are affected; when the
floor governs, the device's edge reason is reported as ``"unipolar_floor"``.

CLI
---
Default (no flags): derive bands for every QF/QD/QFA/DIPOLE device in the
manifest's pyat-coupled partition, emit ``{address: [min, max]}`` JSON to
stdout (or ``--output``).

``--check``: derive QF01/QD01/QFA01/DIPOLE01 only, assert the edge-rule,
policy-headroom, and unipolar-floor invariants, print the same JSON, exit 0.
A fast smoke gate, not a substitute for a full run.

``--verify CHANNEL_LIMITS_JSON``: re-derive every device and compare against
the ``min_value``/``max_value`` already committed in the given
``channel_limits.json`` (see ``src/osprey/templates/apps/control_assistant/
data/channel_limits.json``), within ``--tol`` relative tolerance. Nonzero
exit on any mismatch or missing entry -- this is Task 3.2's regression gate
against silent drift between this script and the committed file.

Always run with the worktree's own interpreter: ``.venv/bin/python
scripts/va/derive_bands.py ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import at
import numpy as np

from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.lattice.inventory import pyat_coupled_device_ids
from osprey.services.virtual_accelerator.lattice.strengths import (
    StrengthMap,
    current_address,
    restore_element,
    snapshot_element,
)

# Sweep step, as a fraction of nominal current, per the plan's fixed 0.25%.
STEP_FRACTION = 0.0025

# Max-plane |trace| that marks the sweep edge -- a margin below the hard
# |trace| >= 2.0 instability guard in lattice.solve.solve_orbit.
TRACE_EDGE = 1.8

# Safety-valve sweep bound, as a fraction of nominal current, in EACH
# direction. Not a physical limit -- some devices (observed: QD family, on
# the current-reversing/negative-fraction side) need to sweep well past
# nominal before the trace threshold trips, because the K = K_baked *
# (I / I_nom) model passes through a sign flip. Calibrated empirically
# against every QF/QD/QFA device (worst case observed: QD, ~2.285x) with
# headroom; a sweep that still finds neither edge condition within this
# bound is treated as a script/data anomaly, not silently accepted.
MAX_DEVIATION_FRACTION = 3.0

# DIPOLE policy trim window (committed band) and required headroom of the
# derived stability boundary over that window.
DIPOLE_POLICY_FRACTION = 0.005
DIPOLE_HEADROOM_FACTOR = 2.0

# Families this script derives bands for.
DERIVED_FAMILIES = ("QF", "QD", "QFA", "DIPOLE")

# Families whose committed band minimum is floored at 0 A -- a unipolar-
# supply policy decision (see module docstring), not a stability result.
UNIPOLAR_FLOOR_FAMILIES = ("QF", "QD", "QFA")
UNIPOLAR_FLOOR = 0.0

# Default relative tolerance for --verify's committed-vs-derived comparison.
DEFAULT_VERIFY_TOL = 1e-6

CHECK_DEVICES = (("QF", "01"), ("QD", "01"), ("QFA", "01"), ("DIPOLE", "01"))


@dataclass(frozen=True)
class DirectionEdge:
    """One sweep direction's result: the edge current and how it was found."""

    current: float
    reason: str  # "trace_crossing" or "last_finite"
    trace_at_edge: float


@dataclass(frozen=True)
class DeviceBand:
    """Full per-device derivation result."""

    address: str
    i_nom: float
    lo_edge: DirectionEdge
    hi_edge: DirectionEdge
    band: tuple[
        float, float
    ]  # committed [min, max] (== derived edges except for DIPOLE/unipolar floor)
    band_lo_reason: (
        str  # lo_edge.reason, "unipolar_floor", or the DIPOLE policy's own lo_edge.reason
    )


def _max_plane_trace(m44: np.ndarray) -> float | None:
    """Max-plane |trace| of a one-turn matrix, or None if it isn't finite."""
    if not np.all(np.isfinite(m44)):
        return None
    trace_x = float(m44[0, 0] + m44[1, 1])
    trace_y = float(m44[2, 2] + m44[3, 3])
    return max(abs(trace_x), abs(trace_y))


def _solve_trace(ring: at.Lattice) -> float | None:
    """Max-plane |trace| of `ring`'s one-turn matrix (None if non-finite).

    Mirrors `lattice.solve.solve_orbit`'s `find_m44` guard, but returns the
    scalar trace (or None) instead of raising -- this script needs to keep
    sweeping past the finite/non-finite boundary, not stop at it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        m44 = at.find_m44(ring)[0]
    return _max_plane_trace(m44)


def _find_element(ring: at.Lattice, family: str, device_id: str) -> at.Element:
    fam_name = f"{family}{device_id}"
    element = next((el for el in ring if el.FamName == fam_name), None)
    if element is None:
        raise ValueError(f"no ring element named {fam_name!r}")
    return element


def _sweep_direction(
    ring: at.Lattice,
    strength_map: StrengthMap,
    family: str,
    device_id: str,
    i_nom: float,
    direction: int,
    baseline_trace: float,
    *,
    step_fraction: float,
    max_deviation_fraction: float,
) -> DirectionEdge:
    """Sweep one direction from nominal, stopping at the first edge trigger.

    Returns the edge -- whichever of the trace-crossing or last-finite rules
    fires first, which is by construction the tighter of the two.

    Raises:
        RuntimeError: if neither rule fires within `max_deviation_fraction`
            of nominal -- the safety valve is too tight for this device.
    """
    last_current, last_trace = i_nom, baseline_trace
    n = 1
    while step_fraction * n <= max_deviation_fraction:
        current = i_nom + direction * step_fraction * n * i_nom
        strength_map.apply(ring, family, device_id, current)
        trace = _solve_trace(ring)

        if trace is None:
            return DirectionEdge(
                current=last_current, reason="last_finite", trace_at_edge=last_trace
            )

        if trace >= TRACE_EDGE:
            frac = (TRACE_EDGE - last_trace) / (trace - last_trace)
            edge_current = last_current + frac * (current - last_current)
            return DirectionEdge(
                current=edge_current, reason="trace_crossing", trace_at_edge=TRACE_EDGE
            )

        last_current, last_trace = current, trace
        n += 1

    raise RuntimeError(
        f"{family}{device_id}: sweep direction {direction:+d} found neither a trace crossing nor "
        f"divergence within {max_deviation_fraction:.2f}x nominal -- MAX_DEVIATION_FRACTION is too "
        "tight for this device"
    )


def derive_device_band(
    ring: at.Lattice,
    strength_map: StrengthMap,
    family: str,
    device_id: str,
    baseline_trace: float,
    *,
    step_fraction: float = STEP_FRACTION,
    max_deviation_fraction: float = MAX_DEVIATION_FRACTION,
) -> DeviceBand:
    """Derive one device's band. Raises AssertionError for a DIPOLE headroom violation."""
    address = current_address(family, device_id)
    i_nom = strength_map.i_nom(address)
    element = _find_element(ring, family, device_id)
    original_state = snapshot_element(element)

    try:
        hi_edge = _sweep_direction(
            ring,
            strength_map,
            family,
            device_id,
            i_nom,
            +1,
            baseline_trace,
            step_fraction=step_fraction,
            max_deviation_fraction=max_deviation_fraction,
        )
        restore_element(element, original_state)

        lo_edge = _sweep_direction(
            ring,
            strength_map,
            family,
            device_id,
            i_nom,
            -1,
            baseline_trace,
            step_fraction=step_fraction,
            max_deviation_fraction=max_deviation_fraction,
        )
    finally:
        restore_element(element, original_state)

    if family == "DIPOLE":
        policy_half = i_nom * DIPOLE_POLICY_FRACTION
        derived_half_lo = i_nom - lo_edge.current
        derived_half_hi = hi_edge.current - i_nom
        min_derived_half = min(derived_half_lo, derived_half_hi)
        required_half = DIPOLE_HEADROOM_FACTOR * policy_half
        if min_derived_half < required_half:
            raise AssertionError(
                f"{address}: derived stability half-width {min_derived_half:.4f} A does not clear "
                f"{DIPOLE_HEADROOM_FACTOR}x the +/-{DIPOLE_POLICY_FRACTION * 100:.2f}% policy "
                f"half-width ({required_half:.4f} A) -- lo_edge={lo_edge}, hi_edge={hi_edge}"
            )
        band = (i_nom - policy_half, i_nom + policy_half)
        band_lo_reason = lo_edge.reason
    else:
        lo = lo_edge.current
        band_lo_reason = lo_edge.reason
        if family in UNIPOLAR_FLOOR_FAMILIES and lo < UNIPOLAR_FLOOR:
            lo = UNIPOLAR_FLOOR
            band_lo_reason = "unipolar_floor"
        band = (lo, hi_edge.current)

    return DeviceBand(
        address=address,
        i_nom=i_nom,
        lo_edge=lo_edge,
        hi_edge=hi_edge,
        band=band,
        band_lo_reason=band_lo_reason,
    )


def _all_devices() -> list[tuple[str, str]]:
    """Every (family, device_id) pair this script derives bands for."""
    inventory = pyat_coupled_device_ids()
    return [
        (family, device_id)
        for family in DERIVED_FAMILIES
        for device_id in inventory.get(family, [])
    ]


def derive_bands(
    ring: at.Lattice,
    strength_map: StrengthMap,
    devices: list[tuple[str, str]],
    *,
    step_fraction: float = STEP_FRACTION,
    max_deviation_fraction: float = MAX_DEVIATION_FRACTION,
) -> dict[str, DeviceBand]:
    """Derive bands for every (family, device_id) pair in `devices`.

    Raises:
        RuntimeError: propagated from a sweep that never finds an edge (see
            `_sweep_direction`).
        AssertionError: a DIPOLE device's policy window fails the headroom
            invariant against its own derived boundary (see
            `derive_device_band`).
    """
    baseline_trace = _solve_trace(ring)
    if baseline_trace is None or baseline_trace >= TRACE_EDGE:
        raise RuntimeError(
            f"nominal ring configuration is not stable enough to sweep from (baseline trace "
            f"{baseline_trace!r}, threshold {TRACE_EDGE}) -- check build_ring()/StrengthMap wiring"
        )

    results: dict[str, DeviceBand] = {}
    for family, device_id in devices:
        results[current_address(family, device_id)] = derive_device_band(
            ring,
            strength_map,
            family,
            device_id,
            baseline_trace,
            step_fraction=step_fraction,
            max_deviation_fraction=max_deviation_fraction,
        )
    return results


def _bands_to_json(results: dict[str, DeviceBand]) -> dict[str, list[float]]:
    return {address: [band.band[0], band.band[1]] for address, band in sorted(results.items())}


def _run_check(ring: at.Lattice, strength_map: StrengthMap, **sweep_kwargs) -> int:
    results = derive_bands(ring, strength_map, list(CHECK_DEVICES), **sweep_kwargs)

    for address, device_band in results.items():
        lo, hi = device_band.band
        assert lo < device_band.i_nom < hi, (
            f"{address}: nominal current not inside derived band [{lo}, {hi}]"
        )

    # Unipolar-floor invariant: QD01's real stability edge is negative, so the
    # floor must govern and the committed min must be exactly 0.0 A.
    qd01_band = results[current_address("QD", "01")]
    assert qd01_band.band[0] == 0.0, (
        f"QD01: expected unipolar floor at 0.0 A, got {qd01_band.band[0]}"
    )
    assert qd01_band.band_lo_reason == "unipolar_floor", (
        f"QD01: expected band_lo_reason 'unipolar_floor', got {qd01_band.band_lo_reason!r}"
    )

    print(json.dumps(_bands_to_json(results), indent=2))
    for address, device_band in results.items():
        lo_edge, hi_edge = device_band.lo_edge, device_band.hi_edge
        print(
            f"# {address}: i_nom={device_band.i_nom:.4f}  "
            f"lo_derived={lo_edge.current:.4f} ({lo_edge.reason}, trace={lo_edge.trace_at_edge:.4f})  "
            f"hi={hi_edge.current:.4f} ({hi_edge.reason}, trace={hi_edge.trace_at_edge:.4f})  "
            f"committed_lo={device_band.band[0]:.4f} ({device_band.band_lo_reason})",
            file=sys.stderr,
        )
    print(
        "CHECK OK: edge-rule, DIPOLE policy-headroom, and unipolar-floor invariants hold",
        file=sys.stderr,
    )
    return 0


def _run_derive_all(
    ring: at.Lattice, strength_map: StrengthMap, output: str | None, **sweep_kwargs
) -> int:
    results = derive_bands(ring, strength_map, _all_devices(), **sweep_kwargs)
    text = json.dumps(_bands_to_json(results), indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n")
    else:
        print(text)
    return 0


def _run_verify(
    ring: at.Lattice, strength_map: StrengthMap, committed_path: Path, *, tol: float, **sweep_kwargs
) -> int:
    committed = json.loads(committed_path.read_text())
    results = derive_bands(ring, strength_map, _all_devices(), **sweep_kwargs)

    missing: list[str] = []
    mismatches: list[tuple[str, object, object, float, float]] = []
    for address, device_band in sorted(results.items()):
        entry = committed.get(address)
        derived_lo, derived_hi = device_band.band
        if not entry or "min_value" not in entry or "max_value" not in entry:
            missing.append(address)
            continue
        committed_lo, committed_hi = entry["min_value"], entry["max_value"]
        if not math.isclose(committed_lo, derived_lo, rel_tol=tol) or not math.isclose(
            committed_hi, derived_hi, rel_tol=tol
        ):
            mismatches.append((address, committed_lo, committed_hi, derived_lo, derived_hi))

    if missing:
        print(f"MISSING from {committed_path}: {len(missing)}", file=sys.stderr)
        for address in missing:
            print(f"  - {address}", file=sys.stderr)
    if mismatches:
        print(f"MISMATCH ({len(mismatches)}, tol={tol}):", file=sys.stderr)
        for address, committed_lo, committed_hi, derived_lo, derived_hi in mismatches:
            print(
                f"  - {address}: committed=[{committed_lo}, {committed_hi}] "
                f"derived=[{derived_lo:.6f}, {derived_hi:.6f}]",
                file=sys.stderr,
            )

    if missing or mismatches:
        print("FAIL: committed bands do not match derivation.", file=sys.stderr)
        return 1

    print(f"OK: {len(results)} devices match committed bands in {committed_path}")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive QF/QD/QFA/DIPOLE current bands from the real ALS-U AR ring.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Self-check: derive QF01/QD01/QFA01/DIPOLE01, assert invariants, exit 0.",
    )
    mode.add_argument(
        "--verify",
        metavar="CHANNEL_LIMITS_JSON",
        help="Re-derive all devices and compare against a committed channel_limits.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write JSON to FILE instead of stdout (default mode only).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_VERIFY_TOL,
        help=f"Relative tolerance for --verify's comparison (default {DEFAULT_VERIFY_TOL}).",
    )
    parser.add_argument(
        "--step-fraction",
        type=float,
        default=STEP_FRACTION,
        help=f"Sweep step as a fraction of nominal current (default {STEP_FRACTION}).",
    )
    parser.add_argument(
        "--max-deviation-fraction",
        type=float,
        default=MAX_DEVIATION_FRACTION,
        help=f"Sweep safety-valve bound as a fraction of nominal current (default {MAX_DEVIATION_FRACTION}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    ring = build_ring()
    strength_map = StrengthMap(ring)
    sweep_kwargs = {
        "step_fraction": args.step_fraction,
        "max_deviation_fraction": args.max_deviation_fraction,
    }

    if args.check:
        return _run_check(ring, strength_map, **sweep_kwargs)
    if args.verify:
        return _run_verify(ring, strength_map, Path(args.verify), tol=args.tol, **sweep_kwargs)
    return _run_derive_all(ring, strength_map, args.output, **sweep_kwargs)


if __name__ == "__main__":
    sys.exit(main())
