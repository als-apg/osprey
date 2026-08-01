"""The ALS-U AR pyAT ring expressed as a LUME model.

:class:`PyATRingModel` owns one persistent ``at.Lattice`` and serves the
pyat-coupled channel partition through the ``LUMEModel`` contract: magnet
setpoints in (``_set``), BPM positions out (``_get``). It is the in-tree
seed of a standalone ``lume-pyat`` package, so it imports nothing from
``ioc`` and never touches EPICS -- and, unlike the IOC bridge that wraps
it, it never calls ``SystemExit``: killing the host process is the
serving layer's decision, not the model's.

Two pieces of state live here, both committed only after a successful
solve:

- **retained inputs** -- the last value written to each setpoint, seeded
  from the ``machine.json`` nominals the catalog carries as
  ``default_value``. These are post-calibration *physical* currents: a
  seeded magnet calibration error acts on the commanded current before it
  reaches this model, so what a caller passes to ``set()`` is exactly what
  is retained and exactly what is converted to strength.
- **cached outputs** -- the BPM positions from the last successful solve.
  This is the physics truth; BPM readout errors are a serving-layer
  concern applied to the *reading*, never to the truth.

``_set`` is atomic across an arbitrary number of setpoints: every element
it is about to mutate is snapshotted first, and a solve that trips
:class:`~osprey.services.virtual_accelerator.lattice.solve.OrbitSolveError`
restores all of them and leaves the retained inputs and cached outputs
exactly as they were. It solves exactly once per call regardless of how
many setpoints it was handed.

Synchronous by construction -- no threads, no locks, no event loop. The
IOC drives ``_set`` from the softioc dispatcher thread, where a lock would
be a deadlock waiting to happen and a background solve would break the
"BPMs are current before the write returns" contract.
"""

from __future__ import annotations

from typing import Any

import at
import numpy as np
from lume.model import LUMEModel, Variable

from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.lattice.errors import apply_misalignment
from osprey.services.virtual_accelerator.lattice.solve import (
    OrbitSolveError,
    monitor_xy,
    solve_orbit,
)
from osprey.services.virtual_accelerator.lattice.strengths import (
    ElementState,
    StrengthMap,
    restore_element,
    snapshot_element,
)
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    build_manifest,
)
from osprey.services.virtual_accelerator.model.catalog import build_variable_catalog

_SETPOINT_SUBFIELD = "SP"


class UnknownDeviceError(ValueError):
    """Raised when an address doesn't map to a known lattice element."""


class PyATRingModel(LUMEModel):
    """LUME model over a single persistent ALS-U AR ``at.Lattice``.

    One instance owns one lattice for its whole lifetime -- every write
    mutates that same lattice in place (never rebuilds it), so sequential
    writes compose exactly like their physical counterparts would: writing
    a device twice is idempotent (last value wins, not cumulative), and
    writing two independent devices in either order reaches the same final
    state.
    """

    def __init__(
        self,
        *,
        element_misalignments: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Build the ring, seed optional misalignments, and solve the nominal orbit.

        Args:
            element_misalignments: fam_name (e.g. ``"QF07"``, ``"DIPOLE03"``)
                -> kwargs for
                :func:`~osprey.services.virtual_accelerator.lattice.errors.apply_misalignment`
                (``dx``/``dy``/``roll``, all optional). Applied once, here,
                after the ring is built and before the nominal orbit is
                solved -- an element absent from this dict keeps AT's default
                (unmisaligned) T1/T2/R1/R2. Every fam_name is validated
                against the ring before any element is mutated.

        Raises:
            UnknownDeviceError: a misalignment names an element the ring
                does not have.
            OrbitSolveError: the seeded misalignments leave the ring without
                a stable closed orbit -- the message names the seeded
                elements and their magnitudes so an otherwise opaque boot
                failure is diagnosable. Deliberately *not* ``SystemExit``:
                whether an unusable model should end the process is the
                caller's call, not this class's.
        """
        self._ring: at.Lattice = build_ring()
        self._index_by_famname: dict[str, int] = {el.FamName: i for i, el in enumerate(self._ring)}
        self._strength_map = StrengthMap(self._ring)

        # One manifest build, shared with the catalog: it costs ~70 ms and this
        # constructor needs it twice over (catalog + device lookup).
        channels = build_manifest()["channels"]

        # Built once: LUMEModel.get/set consult supported_variables on every
        # call, so the catalog must never be rebuilt per access.
        self._catalog = build_variable_catalog(channels)

        # Address -> lattice device, and monitor FamName -> its X/Y output
        # addresses. Both come straight off the manifest entries (which carry
        # `family` and `device`), so no address grammar is re-parsed here.
        self._device_by_address: dict[str, tuple[str, str]] = {}
        self._bpm_addresses: dict[str, dict[str, str]] = {}
        for channel in channels:
            if channel["partition"] != PARTITION_PYAT_COUPLED:
                continue
            address = channel["address"]
            if address not in self._catalog:
                continue  # the :RB setpoint echo -- a serving-layer concern
            if channel["subfield"] == _SETPOINT_SUBFIELD:
                self._device_by_address[address] = (channel["family"], channel["device"])
            else:
                fam_name = f"{channel['family']}{channel['device']}"
                self._bpm_addresses.setdefault(fam_name, {})[channel["subfield"]] = address

        self._nominals: dict[str, float] = {
            address: float(self._catalog[address].default_value)
            for address in self._device_by_address
        }
        self._inputs: dict[str, float] = dict(self._nominals)
        self._outputs: dict[str, float] = {}

        for fam_name in element_misalignments or {}:
            self._element_index(fam_name)  # validate every name before mutating anything
        for fam_name, misalign in (element_misalignments or {}).items():
            apply_misalignment(self._ring[self._element_index(fam_name)], **misalign)

        try:
            self._outputs = self._solve()  # establish the nominal closed orbit
        except OrbitSolveError as exc:
            raise OrbitSolveError(
                f"seeded misalignments {element_misalignments!r} left the SR "
                f"lattice without a stable closed orbit at boot ({exc}); reduce the "
                "misalignment magnitude or remove the fault"
            ) from exc

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """The model's variable catalog, keyed by full channel address.

        The same object on every access -- ``LUMEModel.get``/``set`` consult
        this per call, and callers may legitimately hold onto it.
        """
        return self._catalog

    def _get(self, names: list[str]) -> dict[str, Any]:
        """Return one value per requested name: retained input or cached output.

        Raises:
            UnknownDeviceError: any name that is not a model variable.
        """
        values: dict[str, Any] = {}
        for name in names:
            if name in self._inputs:
                values[name] = self._inputs[name]
            elif name in self._outputs:
                values[name] = self._outputs[name]
            else:
                raise UnknownDeviceError(f"{name!r} is not a variable of this model")
        return values

    def _set(self, values: dict[str, Any]) -> None:
        """Apply setpoints atomically and re-solve the closed orbit exactly once.

        Args:
            values: address -> current, in Amps. Absolute, not a delta.
                Post-calibration: a seeded magnet calibration error is
                applied by the caller, before the current gets here.

        Raises:
            UnknownDeviceError: a name is not a settable input of this model,
                or its family/device has no matching lattice element.
            OrbitSolveError: the combined write leaves the lattice without a
                stable closed orbit. Every mutated element is restored and
                neither the retained inputs nor the cached outputs are
                touched, so a rejected write is a complete no-op.
        """
        for name in values:
            if name not in self._catalog:
                raise UnknownDeviceError(f"{name!r} is not a variable of this model")
            if name not in self._device_by_address:
                raise UnknownDeviceError(f"{name!r} is not a settable input of this model")

        snapshots: list[tuple[int, ElementState]] = []
        for name in values:
            family, device_id = self._device_by_address[name]
            idx = self._element_index(f"{family}{device_id}")
            snapshots.append((idx, snapshot_element(self._ring[idx])))

        try:
            for name, value in values.items():
                self._apply_current(name, value)
            new_outputs = self._solve()
        except Exception:
            for idx, state in snapshots:
                restore_element(self._ring[idx], state)
            raise

        # Commit only now: a partially applied write must never be visible.
        self._inputs.update(values)
        self._outputs = new_outputs

    def reset(self) -> None:
        """Restore every setpoint to its ``machine.json`` nominal.

        Re-applies the nominal currents to the *existing* lattice rather
        than rebuilding it, so construction-time misalignments (and any
        other ring state seeded at ``__init__``) survive a reset -- this
        undoes writes, not faults.

        Raises:
            OrbitSolveError: if the nominal configuration itself has no
                stable closed orbit, which can only happen when the seeded
                faults alone destabilize the ring.
        """
        for address in self._device_by_address:
            self._apply_current(address, self._nominals[address])
        self._outputs = self._solve()
        self._inputs = dict(self._nominals)

    # -- internals ---------------------------------------------------------

    def _element_index(self, fam_name: str) -> int:
        idx = self._index_by_famname.get(fam_name)
        if idx is None:
            raise UnknownDeviceError(f"no lattice element named {fam_name!r}")
        return idx

    def _apply_current(self, address: str, value: float) -> None:
        family, device_id = self._device_by_address[address]
        fam_name = f"{family}{device_id}"
        try:
            self._strength_map.apply(self._ring, family, device_id, value)
        except ValueError as exc:
            # StrengthMap.apply signals an unrecognized family or a missing
            # element with a bare ValueError; name it for what it is. This
            # conversion is why the solve's own broad `except ValueError`
            # (below) must stay scoped to the solve call alone --
            # UnknownDeviceError *is* a ValueError, and a wider catch would
            # silently relabel "unknown device" as "unstable orbit".
            raise UnknownDeviceError(
                f"family {family!r} (device {fam_name!r}) is not pyat-coupled: {exc}"
            ) from exc

    def _solve(self) -> dict[str, float]:
        """Solve the closed orbit once and return the BPM readings it implies.

        Raises:
            OrbitSolveError: the ring has no trustworthy closed orbit.
        """
        # pyAT's find_m44/find_orbit4 usually signal an unstable/non-converged
        # solve by value (NaN), which `solve_orbit` already guards against by
        # raising OrbitSolveError -- but in rare configurations they raise
        # instead: `at.AtError` on a pyAT-detected failure, or a plain
        # `numpy.linalg.LinAlgError`/`ValueError` on LAPACK/pyAT-internal
        # non-convergence. All three are the same failure from here (no orbit
        # can be trusted), so they are folded into OrbitSolveError, keeping
        # the single `except OrbitSolveError` rollback path sufficient.
        try:
            orbit_at_monitors = solve_orbit(self._ring)
        except (at.AtError, np.linalg.LinAlgError, ValueError) as exc:
            raise OrbitSolveError(f"closed orbit solve raised {type(exc).__name__}: {exc}") from exc

        outputs: dict[str, float] = {}
        for fam_name, x, y in monitor_xy(self._ring, orbit_at_monitors):
            axes = self._bpm_addresses[fam_name]
            outputs[axes["X"]] = x
            outputs[axes["Y"]] = y
        return outputs


__all__ = [
    "PyATRingModel",
    "UnknownDeviceError",
    "OrbitSolveError",
]
