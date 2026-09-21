#!/usr/bin/env python3
"""Derive magnet current bands from the ring a served tree carries.

This is the sole re-derivation path for the magnet ``:CURRENT:SP`` min/max
bands in ``channel_limits.json`` -- hand-edited bands are not acceptable. It
sweeps each device's current on the real ring (built by
``osprey.services.virtual_accelerator.lattice.build_ring``) and locates the
one-turn-map stability edge in both directions.

Everything about the accelerator comes out of the served tree: the lattice
file, and beside it the bindings document that says which lattice attribute
each address drives, through which calibration, at which nominal. This script
names no family, no element and no calibration constant of its own -- point it
at a data root with ``--data-root`` and it derives bands for whatever that
tree's bindings declare writable onto the focusing component of a magnet --
the component the edge rule's one-turn matrix actually moves with (see
``FOCUSING_INDEX``). The one policy a band still carries -- the unipolar-supply
floor -- is read off the same tree and is CLI-overridable.

A write here goes through the model's own variable for the binding, the one
the serving path builds (``model.bindings.build_action_variables``), so a
current becomes a physics value by exactly one conversion.

Two gates decide what is swept, and a device passes both or is not swept. By
kind: a hardware setpoint written onto a magnet's polynomial component
(``SWEPT_KIND``), which leaves out a monitor, an rf frequency and the energy
knob. By component: the focusing one (``FOCUSING_INDEX``), the component the
one-turn matrix moves with, which leaves out a sextupole bound to its own
component and a bend bound to the dipole component. A bend is out on the
physics whichever way a tree binds it -- its setpoint is the ring's energy
knob rather than a free strength, so there is no free strength to find an edge
for. A family the sweep does not reach is banded by whatever produced the rest
of its tree's bands: for a tree emitted from a Middle Layer export, that
export's own stated setpoint range.

Edge rule
---------
Starting from the binding's nominal current, step outward in each direction at
a fixed 0.25% of nominal per step. At every step the ring's one-turn matrix is
computed (``at.find_m66``, the matrix ``lume_pyat.solve.solve_orbit``'s guard
reads) and reduced to a single scalar, the max-plane transverse trace
``max(|m[0,0]+m[1,1]|, |m[2,2]+m[3,3]|)``. The sweep stops at the *first* of
two conditions to trigger -- which is, by construction, the tighter of the two:

  (a) the max-plane trace reaches ``TRACE_EDGE`` (1.8, a safety margin below
      the hard |trace| >= 2.0 instability guard) -- the edge current is the
      linear interpolation, on current, between the last sub-threshold
      sample and this one;
  (b) `find_m66` returns a non-finite one-turn matrix -- the edge current is
      the *last finite* sample (one step back). This governs whenever a
      sweep diverges without ever crossing the trace threshold (observed for
      a positive bend trim: PolynomB[0] blows up before |trace| reaches 1.8).

Each direction sweeps independently from a freshly restored element state
(``snapshot_element``/``restore_element``) so sweeps never interact.  A
sweep that reaches neither condition within ``MAX_DEVIATION_FRACTION`` of
nominal (a generous safety valve, not a physical bound -- see module
constant) is a hard error: it means the valve is set too tight for that
device, not a valid result.

Unipolar-supply floor
---------------------
A band minimum is floored at 0 A for a family the tree states runs on a
unipolar supply. The tree states it by what the export measured: of the
nominal currents it carries for that family, at least one is above zero and
none is below it. A supply able to deliver negative current is used that way
somewhere, so a family whose stated nominals never go negative is read as one
that cannot, and its band stops at zero. A family with mixed signs keeps its
derived lower edge, and so does one parked entirely at zero: a device at its
neutral point reads the same on either polarity and states nothing.

The floor is a supply-polarity statement, not a claim about ring stability: a
ring can stay stable with a magnet's current reversed (the derived lower edge
comes out negative), and the floor discards that headroom rather than
committing a setpoint the supply cannot reach. Only devices whose derived
lower edge is actually negative are affected; when the floor governs, the
device's edge reason is reported as ``"unipolar_floor"``.

It can only land on a family this script sweeps -- a focusing strength
setpoint -- so it never reaches a corrector, whose kick binding is not swept
at all. Within that reach the evidence has limits, so a run reports the
decision it reached for every swept family and the stated nominals that
decided it, each of these included:

  * a bipolar family parked entirely non-negative is indistinguishable from a
    unipolar one by this evidence, and is floored;
  * one stated nominal of round-off size below zero drops the floor from the
    whole family;
  * the rule reads the nominals the tree states and nothing else, so a family
    the export stated one nominal for is decided on that one.

``--unipolar-floor-families`` names the floored families outright, for a tree
whose stated nominals do not say what its supplies really are.

CLI
---
Default (no flags): derive bands for every focusing-bound device the tree's
bindings declare, emit ``{address: [min, max]}`` JSON to stdout (or
``--output``).

``--check``: derive the first device of each swept family only, assert the
edge-rule and unipolar-floor invariants, print the same JSON, exit 0. A fast
smoke gate, not a substitute for a full run.

``--verify CHANNEL_LIMITS_JSON``: re-derive every device and compare against
the ``min_value``/``max_value`` already committed in the given
``channel_limits.json``, within ``--tol`` relative tolerance. Nonzero exit on
any mismatch or missing entry -- the regression gate against silent drift
between this script and the committed file.

Every mode prints each swept family's floor decision to stderr, so what the
run did about polarity is on the record beside whatever else it reported. The
floor is applied in all three, and a mode that applied it silently would leave
the reader of a comparison guessing which edges it governed.

Always run with the worktree's own interpreter: ``.venv/bin/python
scripts/va/derive_bands.py ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import at
import numpy as np
from lume_pyat.simulator import PyATSimulator, restore_element, snapshot_element

from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS, ManifestPaths
from osprey.services.virtual_accelerator.model.bindings import build_action_variables

# Sweep step, as a fraction of nominal current, per the plan's fixed 0.25%.
STEP_FRACTION = 0.0025

# Max-plane |trace| that marks the sweep edge -- a margin below the hard
# |trace| >= 2.0 instability guard in lume_pyat.solve.solve_orbit.
TRACE_EDGE = 1.8

# Safety-valve sweep bound, as a fraction of nominal current, in EACH
# direction. Not a physical limit -- a device can need to sweep well past
# nominal before the trace threshold trips, because its calibration passes
# through a sign flip on the way. Calibrated empirically against every
# strength device of the demo ring (worst case observed: its defocusing
# quadrupole family, ~2.285x) with headroom; a sweep that still finds neither
# edge condition within this bound is treated as a script/data anomaly, not
# silently accepted.
MAX_DEVIATION_FRACTION = 3.0

# The bindings kind this script sweeps: a hardware setpoint written onto a
# magnet's polynomial component. A kick is swept by `orbit_response` instead,
# an rf frequency and the energy knob move no strength, and a monitor is read
# only.
SWEPT_KIND = "strength"

# The polynomial component the edge rule is defined for. The rule reads the
# one-turn matrix, which is linear optics: the quadrupole component is what
# moves it. A sextupole component does not appear in that matrix at all, so a
# sweep of one finds no trace crossing and no divergence however far it runs --
# the safety valve trips and reports a device the metric cannot band, which is
# a property of the metric rather than of the magnet. A bend bound to the
# dipole component is out by this gate too, on a tree that binds it as a
# strength rather than as the energy knob. Bands for those families come from
# the facility's own exported setpoint range instead.
FOCUSING_INDEX = 1

# The band minimum a unipolar supply's family commits (see module docstring).
UNIPOLAR_FLOOR = 0.0

# Default relative tolerance for --verify's committed-vs-derived comparison.
DEFAULT_VERIFY_TOL = 1e-6


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
    family: str
    i_nom: float
    lo_edge: DirectionEdge
    hi_edge: DirectionEdge
    band: tuple[float, float]  # committed [min, max] (== derived edges, up to the floor)
    band_lo_reason: str  # lo_edge.reason, or "unipolar_floor"


@dataclass(frozen=True)
class FloorDecision:
    """Whether a family's band minimum is floored, and the evidence for it."""

    family: str
    floored: bool
    stated: int  # devices of the family the tree states a nominal for
    lowest: float | None
    highest: float | None
    reason: str


def nominals_by_family(document: BindingsDocument) -> dict[str, list[float]]:
    """Every nominal current ``document`` states, grouped by family.

    A binding that states none contributes nothing: a missing nominal is an
    absence of evidence, not a value to read a polarity from.
    """
    nominals: dict[str, list[float]] = {}
    for binding in document.bindings:
        if binding.nominal is not None:
            nominals.setdefault(binding.family, []).append(float(binding.nominal))
    return nominals


def unipolar_families(nominals: Mapping[str, Sequence[float]]) -> frozenset[str]:
    """The families whose stated nominals say their supplies run one polarity.

    A family qualifies when the tree states it runs at current at all -- one
    nominal above zero -- and states none below zero (see module docstring).
    """
    return frozenset(
        family
        for family, values in nominals.items()
        if any(value > 0.0 for value in values) and all(value >= 0.0 for value in values)
    )


def unipolar_floor_families(document: BindingsDocument) -> frozenset[str]:
    """The families of ``document`` whose supplies can deliver only one polarity."""
    return unipolar_families(nominals_by_family(document))


def floor_decisions(
    nominals: Mapping[str, Sequence[float]],
    families: Iterable[str],
    floored: frozenset[str],
) -> list[FloorDecision]:
    """The floor decision reached for each family, with what decided it.

    ``floored`` is the set actually in force, so a set named on the command
    line is reported as the run applied it rather than as the tree's nominals
    would have had it.
    """
    decisions: list[FloorDecision] = []
    for family in sorted(set(families)):
        values = list(nominals.get(family, ()))
        if family in floored:
            reason = "floored at 0 A"
        elif not values:
            reason = "not floored: the tree states no nominal for it"
        elif any(value < 0.0 for value in values):
            reason = "not floored: a stated nominal is below zero"
        elif not any(value > 0.0 for value in values):
            reason = "not floored: every stated nominal is zero"
        else:
            reason = "not floored: named on the command line as bipolar"
        decisions.append(
            FloorDecision(
                family=family,
                floored=family in floored,
                stated=len(values),
                lowest=min(values) if values else None,
                highest=max(values) if values else None,
                reason=reason,
            )
        )
    return decisions


def format_floor_decisions(decisions: Iterable[FloorDecision]) -> list[str]:
    """One reportable line per decision, naming the evidence behind it."""
    lines = []
    for decision in decisions:
        if decision.stated:
            evidence = (
                f"{decision.stated} stated nominal(s), "
                f"{decision.lowest:.6g} .. {decision.highest:.6g}"
            )
        else:
            evidence = "no stated nominal"
        lines.append(f"# {decision.family}: {decision.reason}  ({evidence})")
    return lines


class Sweeper:
    """The tree's ring, and the model variable each swept address writes through.

    One place holds the ring, the simulator that resolves element names on it,
    and the variables built from the tree's own bindings -- so a sweep writes a
    current exactly the way the serving path does, through the binding's
    calibration and onto the slices it names.
    """

    def __init__(self, paths: ManifestPaths) -> None:
        """Load the tree's bindings and build its ring and variables.

        Args:
            paths: The served tree to sweep, holding the saved lattice and the
                bindings derived against it.

        Raises:
            FileNotFoundError: the tree carries no lattice or no bindings.
            BindingsError: the bindings are refused by their own schema, or
                describe a lattice other than the one the tree carries.
        """
        self.document: BindingsDocument = load_bindings(paths.va_bindings)
        self.simulator = PyATSimulator(build_ring(paths))
        strengths = [binding for binding in self.document.bindings if binding.kind == SWEPT_KIND]
        #: Every family bound to a strength, swept or not, for the message a
        #: `--families` request that names an unsweepable one gets.
        self.strength_families = frozenset(binding.family for binding in strengths)
        self.swept_bindings: tuple[Binding, ...] = tuple(
            binding for binding in strengths if binding.index == FOCUSING_INDEX
        )
        self._by_address = {binding.setpoint_address: binding for binding in self.swept_bindings}
        factories = build_action_variables(self.document)
        self._variables = {
            address: factories[address]({}, name=address, default_value=self.nominal(address))
            for address in self._by_address
        }

    @property
    def ring(self) -> at.Lattice:
        return self.simulator.lattice

    def binding(self, address: str) -> Binding:
        return self._by_address[address]

    def nominal(self, address: str) -> float:
        """The nominal hardware current the binding declares for ``address``.

        Raises:
            ValueError: the binding carries none, so the sweep has no
                baseline to start from.
        """
        nominal = self._by_address[address].nominal
        if nominal is None:
            raise ValueError(
                f"binding {address!r} carries no nominal current, so there is no baseline to "
                "sweep from -- re-export the tree's bindings with the device's nominal"
            )
        return float(nominal)

    def apply(self, address: str, current: float) -> None:
        """Write ``current`` onto the ring through ``address``'s own variable.

        ``_set`` is the write a variable driven against a simulator directly
        performs (see ``lume_pyat.actions``); the serving path reaches the very
        same call through ``LUMEPyATModel``, which this script has no served
        manifest to build.
        """
        self._variables[address]._set(self.simulator, current)

    def elements(self, address: str) -> list[at.Element]:
        """Every element a write to ``address`` touches, for snapshot/restore."""
        return [
            self.simulator.element(slice_.element) for slice_ in self._by_address[address].slices
        ]


def _max_plane_trace(one_turn: np.ndarray) -> float | None:
    """Max-plane transverse |trace| of a one-turn matrix, or None if non-finite.

    Reads the transverse 4x4 block, so the same scalar comes out of a 4D
    ring's map and the 6D map of a ring whose cavity is enabled.
    """
    if not np.all(np.isfinite(one_turn)):
        return None
    trace_x = float(one_turn[0, 0] + one_turn[1, 1])
    trace_y = float(one_turn[2, 2] + one_turn[3, 3])
    return max(abs(trace_x), abs(trace_y))


def _solve_trace(ring: at.Lattice) -> float | None:
    """Max-plane transverse |trace| of `ring`'s one-turn matrix (None if non-finite).

    Reads the matrix the way `lume_pyat.solve.solve_orbit`'s guard does --
    `at.find_m66`, which solves a 4D and a 6D ring alike, where `find_m44`
    refuses a ring with an enabled cavity and `build_ring` enables one. What
    differs is the verdict: that guard tests the eigenvalues and raises, while
    this returns the scalar trace (or None), because the sweep has to keep
    going past the finite/non-finite boundary rather than stop at it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=at.AtWarning)
        one_turn = at.find_m66(ring)[0]
    return _max_plane_trace(one_turn)


def _sweep_direction(
    sweeper: Sweeper,
    address: str,
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
        sweeper.apply(address, current)
        trace = _solve_trace(sweeper.ring)

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
        f"{address}: sweep direction {direction:+d} found neither a trace crossing nor "
        f"divergence within {max_deviation_fraction:.2f}x nominal -- MAX_DEVIATION_FRACTION is too "
        "tight for this device"
    )


def derive_device_band(
    sweeper: Sweeper,
    address: str,
    baseline_trace: float,
    floored_families: frozenset[str],
    *,
    step_fraction: float = STEP_FRACTION,
    max_deviation_fraction: float = MAX_DEVIATION_FRACTION,
) -> DeviceBand:
    """Derive one device's band, floored at zero where its family is unipolar."""
    family = sweeper.binding(address).family
    i_nom = sweeper.nominal(address)
    elements = sweeper.elements(address)
    original_states = [snapshot_element(element) for element in elements]

    def restore() -> None:
        for element, state in zip(elements, original_states, strict=True):
            restore_element(element, state)

    try:
        hi_edge = _sweep_direction(
            sweeper,
            address,
            i_nom,
            +1,
            baseline_trace,
            step_fraction=step_fraction,
            max_deviation_fraction=max_deviation_fraction,
        )
        restore()

        lo_edge = _sweep_direction(
            sweeper,
            address,
            i_nom,
            -1,
            baseline_trace,
            step_fraction=step_fraction,
            max_deviation_fraction=max_deviation_fraction,
        )
    finally:
        restore()

    lo = lo_edge.current
    band_lo_reason = lo_edge.reason
    if family in floored_families and lo < UNIPOLAR_FLOOR:
        lo = UNIPOLAR_FLOOR
        band_lo_reason = "unipolar_floor"
    band = (lo, hi_edge.current)

    return DeviceBand(
        address=address,
        family=family,
        i_nom=i_nom,
        lo_edge=lo_edge,
        hi_edge=hi_edge,
        band=band,
        band_lo_reason=band_lo_reason,
    )


def _selected_addresses(sweeper: Sweeper, families: tuple[str, ...]) -> list[str]:
    """Every swept address of the tree, narrowed to ``families`` when given.

    Raises:
        ValueError: a requested family has no device the edge rule can band --
            either it binds no strength setpoint at all (a misspelling, or a
            family the export did not couple) or it binds one the one-turn
            matrix does not see (see ``FOCUSING_INDEX``). The message says
            which of the two it is, because the fix differs.
    """
    bindings = sweeper.swept_bindings
    if not families:
        return [binding.setpoint_address for binding in bindings]

    sweepable = {binding.family for binding in bindings}
    unbound = [family for family in families if family not in sweeper.strength_families]
    unsweepable = [
        family
        for family in families
        if family in sweeper.strength_families and family not in sweepable
    ]
    if unbound:
        raise ValueError(
            f"no strength binding for {', '.join(unbound)} in this tree; it binds "
            f"{', '.join(sorted(sweeper.strength_families)) or '(no strength family)'}"
        )
    if unsweepable:
        raise ValueError(
            f"{', '.join(unsweepable)} is not bound to the focusing component "
            f"(PolynomB[{FOCUSING_INDEX}]), so the one-turn matrix does not move with it and "
            "it has no stability edge to derive; band it from the facility's exported "
            "setpoint range instead"
        )
    wanted = frozenset(families)
    return [binding.setpoint_address for binding in bindings if binding.family in wanted]


def _first_per_family(sweeper: Sweeper, addresses: list[str]) -> list[str]:
    """The first address of each family in ``addresses``, in document order."""
    seen: set[str] = set()
    first: list[str] = []
    for address in addresses:
        family = sweeper.binding(address).family
        if family not in seen:
            seen.add(family)
            first.append(address)
    return first


def derive_bands(
    sweeper: Sweeper,
    addresses: list[str],
    floored_families: frozenset[str],
    *,
    step_fraction: float = STEP_FRACTION,
    max_deviation_fraction: float = MAX_DEVIATION_FRACTION,
) -> dict[str, DeviceBand]:
    """Derive bands for every address in `addresses`.

    Raises:
        RuntimeError: propagated from a sweep that never finds an edge (see
            `_sweep_direction`), or the nominal ring is already too close to
            the stability edge to sweep from.
    """
    baseline_trace = _solve_trace(sweeper.ring)
    if baseline_trace is None or baseline_trace >= TRACE_EDGE:
        raise RuntimeError(
            f"nominal ring configuration is not stable enough to sweep from (baseline trace "
            f"{baseline_trace!r}, threshold {TRACE_EDGE}) -- check the tree's lattice and bindings"
        )

    results: dict[str, DeviceBand] = {}
    for address in addresses:
        results[address] = derive_device_band(
            sweeper,
            address,
            baseline_trace,
            floored_families,
            step_fraction=step_fraction,
            max_deviation_fraction=max_deviation_fraction,
        )
    return results


def _bands_to_json(results: dict[str, DeviceBand]) -> dict[str, list[float]]:
    return {address: [band.band[0], band.band[1]] for address, band in sorted(results.items())}


def _report_floor_decisions(
    sweeper: Sweeper, addresses: list[str], floored_families: frozenset[str]
) -> None:
    """Print, to stderr, the floor decision every swept family got."""
    decisions = floor_decisions(
        nominals_by_family(sweeper.document),
        (sweeper.binding(address).family for address in addresses),
        floored_families,
    )
    for line in format_floor_decisions(decisions):
        print(line, file=sys.stderr)


def _run_check(
    sweeper: Sweeper,
    addresses: list[str],
    floored_families: frozenset[str],
    **sweep_kwargs,
) -> int:
    results = derive_bands(
        sweeper, _first_per_family(sweeper, addresses), floored_families, **sweep_kwargs
    )

    for address, device_band in results.items():
        lo, hi = device_band.band
        assert lo < device_band.i_nom < hi, (
            f"{address}: nominal current not inside derived band [{lo}, {hi}]"
        )
        if device_band.band_lo_reason == "unipolar_floor":
            # Where the floor governs, it governs exactly: the committed
            # minimum is the floor and nothing else.
            assert device_band.band[0] == UNIPOLAR_FLOOR, (
                f"{address}: floored band minimum is {device_band.band[0]}, not {UNIPOLAR_FLOOR}"
            )
            assert device_band.lo_edge.current < UNIPOLAR_FLOOR, (
                f"{address}: reported as floored, but its derived lower edge "
                f"{device_band.lo_edge.current} is not below {UNIPOLAR_FLOOR}"
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
    _report_floor_decisions(sweeper, addresses, floored_families)
    print(
        "CHECK OK: the edge-rule and unipolar-floor invariants hold",
        file=sys.stderr,
    )
    return 0


def _run_derive_all(
    sweeper: Sweeper,
    addresses: list[str],
    floored_families: frozenset[str],
    output: str | None,
    **sweep_kwargs,
) -> int:
    results = derive_bands(sweeper, addresses, floored_families, **sweep_kwargs)
    _report_floor_decisions(sweeper, addresses, floored_families)
    text = json.dumps(_bands_to_json(results), indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n")
    else:
        print(text)
    return 0


def _run_verify(
    sweeper: Sweeper,
    addresses: list[str],
    floored_families: frozenset[str],
    committed_path: Path,
    *,
    tol: float,
    **sweep_kwargs,
) -> int:
    committed = json.loads(committed_path.read_text())
    results = derive_bands(sweeper, addresses, floored_families, **sweep_kwargs)
    _report_floor_decisions(sweeper, addresses, floored_families)

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


def _families(value: str) -> tuple[str, ...]:
    """Parse a comma-separated family list; an empty string means every family."""
    return tuple(token.strip() for token in value.split(",") if token.strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive magnet current bands from the ring a served tree carries.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Self-check: derive one device per family, assert invariants, exit 0.",
    )
    mode.add_argument(
        "--verify",
        metavar="CHANNEL_LIMITS_JSON",
        help="Re-derive all devices and compare against a committed channel_limits.json.",
    )
    parser.add_argument(
        "--data-root",
        metavar="DIR",
        type=Path,
        help="Served data tree to sweep (default: the bundled control-assistant tree).",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write JSON to FILE instead of stdout (default mode only).",
    )
    parser.add_argument(
        "--families",
        metavar="LIST",
        type=_families,
        default=(),
        help="Comma-separated families to derive (default: every focusing-bound family).",
    )
    parser.add_argument(
        "--unipolar-floor-families",
        metavar="LIST",
        type=_families,
        default=None,
        help=(
            "Comma-separated families whose committed band minimum is floored at "
            f"{UNIPOLAR_FLOOR} A (default: the families the tree states only "
            "non-negative nominals for)."
        ),
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

    paths = PACKAGE_PATHS if args.data_root is None else ManifestPaths(data_root=args.data_root)
    sweeper = Sweeper(paths)
    addresses = _selected_addresses(sweeper, args.families)
    floored_families = (
        unipolar_floor_families(sweeper.document)
        if args.unipolar_floor_families is None
        else frozenset(args.unipolar_floor_families)
    )
    sweep_kwargs = {
        "step_fraction": args.step_fraction,
        "max_deviation_fraction": args.max_deviation_fraction,
    }

    if args.check:
        return _run_check(sweeper, addresses, floored_families, **sweep_kwargs)
    if args.verify:
        return _run_verify(
            sweeper, addresses, floored_families, Path(args.verify), tol=args.tol, **sweep_kwargs
        )
    return _run_derive_all(sweeper, addresses, floored_families, args.output, **sweep_kwargs)


if __name__ == "__main__":
    sys.exit(main())
