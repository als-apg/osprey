"""Physics bridge: synchronous PyAT orbit recompute for SR magnet setpoint writes.

Wires partition (a) (pyat-coupled) SR magnet SP writes into the SR lattice
built by ``lattice.build_ring()``: writing a corrector, quadrupole, dipole, or
sextupole current updates that element's strength on a single persistent
lattice instance, re-solves the closed orbit, and makes every BPM POSITION
reading available before the write call returns (FR3/SC3: the recompute
happens synchronously in the write handler itself, never on a
polling/heartbeat tick).

This module fulfills the ``on_pyat_setpoint`` callback contract that
``ioc.records.build_records()`` exposes (see that module's docstring):
``PhysicsBridge.on_setpoint`` is passed as ``on_pyat_setpoint``, and
``PhysicsBridge.bind()`` wires the resulting ``IOCRecords.pyat_coupled`` BPM
records so they receive the recomputed positions via ``.set()``.

Current-to-strength calibration for every magnet/corrector family is owned by
:class:`~osprey.services.virtual_accelerator.lattice.strengths.StrengthMap`
(see that module's docstring for the per-family formulas); this bridge only
looks up a device's commanded current -> physical strength through it.
"""

from __future__ import annotations

from typing import Any

import at
import numpy as np

from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.lattice.errors import (
    apply_misalignment,
    bpm_read,
    magnet_cal,
)
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError, solve_orbit
from osprey.services.virtual_accelerator.lattice.strengths import (
    ElementState,
    StrengthMap,
    restore_element,
    snapshot_element,
)

_CURRENT_FIELD = "CURRENT"
_BPM_SYSTEM_FAMILY = ("DIAG", "BPM")
_BPM_FIELD = "POSITION"

# `bpm_read`'s full keyword-argument set at identity (no-op) values -- a BPM
# with no seeded error reads the true orbit position exactly. Fault dicts
# passed into PhysicsBridge only need to name the fields they perturb; the
# rest fall back to this identity.
_IDENTITY_BPM_ERROR: dict[str, float] = {
    "offset_x": 0.0,
    "offset_y": 0.0,
    "gain_x": 1.0,
    "gain_y": 1.0,
    "polarity_x": 1.0,
    "polarity_y": 1.0,
    "roll": 0.0,
    "cal_x": 0.0,
    "cal_y": 0.0,
    "noise_x": 0.0,
    "noise_y": 0.0,
}


class UnknownDeviceError(ValueError):
    """Raised when a pyat-coupled address doesn't map to a known lattice element."""


def _parse_pyat_coupled_address(address: str) -> tuple[str, str, str, str]:
    """Split a manifest address into (system, family, device, field).

    e.g. "SR:MAG:HCM:05:CURRENT:SP" -> ("MAG", "HCM", "05", "CURRENT")
    """
    parts = address.split(":")
    if len(parts) != 6:
        raise UnknownDeviceError(f"not a 6-level manifest address: {address!r}")
    _ring, system, family, device, field, _subfield = parts
    return system, family, device, field


def _bpm_address(device: str, axis: str) -> str:
    ring, system, family = "SR", *_BPM_SYSTEM_FAMILY
    return f"{ring}:{system}:{family}:{device}:{_BPM_FIELD}:{axis}"


class PhysicsBridge:
    """Owns the persistent SR lattice instance and the pyat-coupled write path.

    A single `PhysicsBridge` instance owns one `at.Lattice` for the lifetime of
    the IOC process -- every SP write mutates that same lattice in place
    (never rebuilds it), so sequential writes compose exactly like their
    physical counterparts would: writing a device twice is idempotent (last
    value wins, not cumulative), and writing two independent devices in
    either order reaches the same final state (SC3).
    """

    def __init__(
        self,
        *,
        element_misalignments: dict[str, dict[str, float]] | None = None,
        bpm_errors: dict[str, dict[str, float]] | None = None,
        corrector_gains: dict[str, dict[str, float]] | None = None,
        rng_seed: int | None = None,
    ) -> None:
        """Build the ring and, optionally, seed FR3/FR4 physics faults on it.

        Args:
            element_misalignments: fam_name (e.g. "QF07", "DIPOLE03") -> kwargs
                for `errors.apply_misalignment` (`dx`/`dy`/`roll`, all optional).
                Applied once, in `__init__`, after the ring is built and before
                the nominal orbit is solved -- an element absent from this dict
                keeps AT's default (unmisaligned) T1/T2/R1/R2.
            bpm_errors: BPM fam_name (e.g. "BPM01") -> a partial override of
                `errors.bpm_read`'s keyword args; missing fields fall back to
                identity (see `_IDENTITY_BPM_ERROR`). A BPM absent from this
                dict reads with identity error (i.e. exactly its true position).
            corrector_gains: magnet fam_name (e.g. "HCM01", "QF07") -> a
                partial override of `errors.magnet_cal`'s `factor`/`offset`;
                missing fields default to `factor=1.0, offset=0.0` (identity).
            rng_seed: seed for the `numpy.random.Generator` BPM readout noise
                is drawn from. `None` seeds from OS entropy (non-reproducible),
                matching `numpy.random.default_rng`'s own default.

        Raises:
            SystemExit: a seeded misalignment leaves the ring without a stable
                closed orbit (FR12) -- turns an opaque boot crash into a
                diagnosable one naming the seeded elements and magnitudes.
        """
        self._ring: at.Lattice = build_ring()
        self._index_by_famname: dict[str, int] = {el.FamName: i for i, el in enumerate(self._ring)}
        self._strength_map = StrengthMap(self._ring)
        self._bpm_positions: dict[str, float] = {}
        self._bpm_device_ids: list[str] = []
        self._bpm_readback_records: dict[str, Any] = {}
        self._rng = np.random.default_rng(rng_seed)
        self._bpm_error_state: dict[str, dict[str, float]] = dict(bpm_errors or {})
        self._magnet_cal_state: dict[str, dict[str, float]] = dict(corrector_gains or {})

        for fam_name, misalign in (element_misalignments or {}).items():
            idx = self._element_index(fam_name)  # validates before mutating anything
            apply_misalignment(self._ring[idx], **misalign)

        try:
            self._solve_orbit()  # establish the nominal closed orbit at construction
        except OrbitSolveError as exc:
            raise SystemExit(
                f"FATAL: seeded misalignments {element_misalignments!r} left the SR "
                f"lattice without a stable closed orbit at boot ({exc}); reduce the "
                "misalignment magnitude or remove the fault"
            ) from exc

    def bind(self, pyat_coupled_records: dict[str, Any]) -> None:
        """Wire the BPM POSITION readback records this bridge should push into.

        Args:
            pyat_coupled_records: the `IOCRecords.pyat_coupled` dict from
                `ioc.records.build_records()` -- contains every partition (a)
                record (both the SR magnet SP writables and the SR BPM
                POSITION readbacks) keyed by address. Only the BPM entries are
                retained; magnet SP records are driven by softioc directly,
                not by this bridge.
        """
        ring, system, family = "SR", *_BPM_SYSTEM_FAMILY
        prefix = f"{ring}:{system}:{family}:"
        self._bpm_readback_records = {
            address: rec
            for address, rec in pyat_coupled_records.items()
            if address.startswith(prefix) and address.endswith((":X", ":Y"))
        }
        self._push_bpm_readbacks()

    def on_setpoint(self, address: str, value: float) -> None:
        """`on_pyat_setpoint` callback: apply one SP write and push BPM readbacks.

        Args:
            address: the manifest address of the SP channel that was written,
                e.g. "SR:MAG:HCM:05:CURRENT:SP".
            value: the new current, in Amps. Absolute, not a delta -- writing
                the same address twice with different values is idempotent
                (the second write fully determines the element's strength).

        Raises:
            UnknownDeviceError: if address doesn't map to a corrector, magnet,
                or sextupole element in the lattice.
            OrbitSolveError: if the resulting lattice has no stable closed
                orbit -- the write is rolled back (the element's prior
                strength is restored) before this is raised, so a rejected
                write never leaves the lattice in a broken state.
        """
        system, family, device, field = _parse_pyat_coupled_address(address)
        if system != "MAG":
            raise UnknownDeviceError(
                f"expected a MAG setpoint, got system={system!r} in {address!r}"
            )
        if field != _CURRENT_FIELD:
            raise UnknownDeviceError(
                f"expected a {_CURRENT_FIELD} setpoint, got field={field!r} in {address!r}"
            )
        fam_name = f"{family}{device}"
        idx = self._element_index(fam_name)  # validates before mutating anything
        previous_state: ElementState = snapshot_element(self._ring[idx])

        self._apply_current(family, device, fam_name, value)
        try:
            self._solve_orbit()
        except OrbitSolveError:
            restore_element(self._ring[idx], previous_state)
            raise
        self._push_bpm_readbacks()

    def bpm_positions(self) -> dict[str, float]:
        """Return the most recently solved BPM POSITION readings, keyed by address.

        Available independent of `bind()` -- this is the physics-only view
        used by tests and by any consumer that doesn't need live IOC records.
        """
        return dict(self._bpm_positions)

    # -- internals ---------------------------------------------------------

    def _element_index(self, fam_name: str) -> int:
        idx = self._index_by_famname.get(fam_name)
        if idx is None:
            raise UnknownDeviceError(f"no lattice element named {fam_name!r}")
        return idx

    def _apply_current(self, family: str, device_id: str, fam_name: str, value: float) -> None:
        # A seeded calibration error (gain/polarity/offset) acts on the
        # commanded current before it's converted to physical strength -- a
        # miscalibrated magnet's *field* differs from its setpoint, not the
        # other way around. Identity (factor=1, offset=0) if unseeded.
        cal = self._magnet_cal_state.get(fam_name, {})
        value = magnet_cal(value, factor=cal.get("factor", 1.0), offset=cal.get("offset", 0.0))

        try:
            self._strength_map.apply(self._ring, family, device_id, value)
        except ValueError as exc:
            raise UnknownDeviceError(
                f"family {family!r} (device {fam_name!r}) is not pyat-coupled: {exc}"
            ) from exc

    def _solve_orbit(self) -> None:
        # pyAT's find_m44/find_orbit4 usually signal an unstable/non-converged
        # solve by value (NaN), which `solve_orbit` already guards against by
        # raising OrbitSolveError -- but in rare configurations find_m44/
        # find_orbit4 themselves raise instead of returning a non-finite
        # orbit: `at.AtError` on a pyAT-detected failure, or a plain
        # `numpy.linalg.LinAlgError`/`ValueError` on LAPACK/pyAT-internal
        # non-convergence. All three are the same failure from this bridge's
        # perspective (no orbit can be trusted), so they are folded into
        # OrbitSolveError here, ahead of both the constructor's and
        # `on_setpoint`'s single `except OrbitSolveError` rollback paths --
        # any exception besides OrbitSolveError itself must not escape this
        # method uncaught, or a failed write would skip the rollback path.
        try:
            orbit_at_monitors = solve_orbit(self._ring)
        except OrbitSolveError:
            raise
        except (at.AtError, np.linalg.LinAlgError, ValueError) as exc:
            raise OrbitSolveError(f"closed orbit solve raised {type(exc).__name__}: {exc}") from exc

        monitor_indices = self._ring.get_refpts(at.Monitor)
        positions: dict[str, float] = {}
        device_ids: list[str] = []
        for row, el_idx in enumerate(monitor_indices):
            fam_name = self._ring[el_idx].FamName  # e.g. "BPM05"
            device = fam_name[len("BPM") :]
            device_ids.append(device)
            positions[_bpm_address(device, "X")] = float(orbit_at_monitors[row, 0])
            positions[_bpm_address(device, "Y")] = float(orbit_at_monitors[row, 2])

        self._bpm_positions = positions
        self._bpm_device_ids = device_ids

    def _push_bpm_readbacks(self) -> None:
        """Push each BPM's seeded-error *reading* into its bound RB record.

        `_bpm_positions` (the physics truth `bpm_positions()` exposes) is
        never touched here -- only the values pushed into IOC records run
        through `bpm_read`, per FR3's "errors apply on the reading, not the
        truth" contract.
        """
        for device in self._bpm_device_ids:
            true_x = self._bpm_positions[_bpm_address(device, "X")]
            true_y = self._bpm_positions[_bpm_address(device, "Y")]
            state = {**_IDENTITY_BPM_ERROR, **self._bpm_error_state.get(f"BPM{device}", {})}
            reading_x, reading_y = bpm_read(true_x, true_y, rng=self._rng, **state)

            x_rec = self._bpm_readback_records.get(_bpm_address(device, "X"))
            if x_rec is not None:
                x_rec.set(reading_x)
            y_rec = self._bpm_readback_records.get(_bpm_address(device, "Y"))
            if y_rec is not None:
                y_rec.set(reading_y)


__all__ = [
    "PhysicsBridge",
    "UnknownDeviceError",
    "OrbitSolveError",
]
