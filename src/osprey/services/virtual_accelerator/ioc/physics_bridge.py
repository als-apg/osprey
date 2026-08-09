"""Physics bridge: synchronous PyAT orbit recompute for SR magnet setpoint writes.

Wires partition (a) (pyat-coupled) SR magnet SP writes into the SR lattice
built by ``lattice.build_ring()``: writing a corrector, quadrupole, dipole, or
sextupole current updates that element's strength on a single persistent
lattice instance, re-solves the closed orbit, and makes every BPM POSITION
reading available before the write call returns (FR3/SC3: the recompute
happens synchronously in the write handler itself, never on a
polling/heartbeat tick).

This module fulfills the ``on_pyat_setpoint`` callback contract that
``serving.pvdb.build_serving_pvdb()`` exposes (see that module's docstring):
``PhysicsBridge.on_setpoint`` is passed as ``on_pyat_setpoint``, and
``PhysicsBridge.bind()`` wires the resulting ``ServingRecords.pyat_coupled``
BPM records so they receive the recomputed positions via ``.set()``.

The ring itself lives behind a :class:`~lume.model.LUMEModel` -- by default
:class:`~osprey.services.virtual_accelerator.model.pyat.PyATRingModel`, which
owns the lattice, the current->strength calibration
(:class:`~osprey.services.virtual_accelerator.lattice.strengths.StrengthMap`,
see that module's docstring for the per-family formulas), the atomic
apply-and-solve, and its rollback. This bridge is the *serving* half: address
grammar, seeded magnet calibration and BPM readout errors, and the served
record wiring. Everything the model owns is reached through its public
``set()``/``get()``, so a different backend (a surrogate, Cheetah, Bmad) can be
injected through ``model=`` without this module changing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from osprey.services.virtual_accelerator.lattice.errors import bpm_read, magnet_cal
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel, UnknownDeviceError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.model import LUMEModel

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
    """Serves the pyat-coupled write path from a `LUMEModel` physics backend.

    A single `PhysicsBridge` instance holds one model for the lifetime of the
    IOC process, and that model owns one lattice -- every SP write mutates that
    same lattice in place (never rebuilds it), so sequential writes compose
    exactly like their physical counterparts would: writing a device twice is
    idempotent (last value wins, not cumulative), and writing two independent
    devices in either order reaches the same final state (SC3).
    """

    def __init__(
        self,
        *,
        model: LUMEModel | None = None,
        element_misalignments: dict[str, dict[str, float]] | None = None,
        bpm_errors: dict[str, dict[str, float]] | None = None,
        corrector_gains: dict[str, dict[str, float]] | None = None,
        rng_seed: int | None = None,
    ) -> None:
        """Attach a physics model and, optionally, seed FR3/FR4 serving faults.

        Args:
            model: the physics backend to serve. `None` (the default)
                constructs a `PyATRingModel`, forwarding `element_misalignments`
                to it. Supplying a model makes the backend pluggable -- a
                surrogate, Cheetah or Bmad model implementing the same
                `LUMEModel` contract serves through this bridge unchanged.
            element_misalignments: fam_name (e.g. "QF07", "DIPOLE03") -> kwargs
                for `errors.apply_misalignment` (`dx`/`dy`/`roll`, all optional),
                seeded on the ring the default model builds. Mutually exclusive
                with `model`: a caller supplying its own model is responsible for
                that model's ring state.
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
            ValueError: both `model` and `element_misalignments` were given.
            UnknownDeviceError: a seeded misalignment names an element the ring
                does not have -- propagates from the model unchanged.
            SystemExit: a seeded misalignment leaves the ring without a stable
                closed orbit (FR12) -- turns an opaque boot crash into a
                diagnosable one naming the seeded elements and magnitudes. Only
                the default-construction path converts the model's
                `OrbitSolveError` this way: ending the process is the serving
                layer's decision, which is why the model itself never does it.
        """
        if model is not None and element_misalignments is not None:
            raise ValueError(
                "pass either model= or element_misalignments=, not both: seeding a ring "
                "is the responsibility of whoever built the model"
            )

        self._bpm_positions: dict[str, float] = {}
        self._bpm_readback_records: dict[str, Any] = {}
        self._rng = np.random.default_rng(rng_seed)
        self._bpm_error_state: dict[str, dict[str, float]] = dict(bpm_errors or {})
        self._magnet_cal_state: dict[str, dict[str, float]] = dict(corrector_gains or {})

        if model is None:
            try:
                model = PyATRingModel(element_misalignments=element_misalignments)
            except OrbitSolveError as exc:
                # Deliberately OrbitSolveError only: an UnknownDeviceError from an
                # unknown misaligned fam_name must reach the caller as itself.
                raise SystemExit(
                    f"FATAL: seeded misalignments {element_misalignments!r} left the SR "
                    f"lattice without a stable closed orbit at boot ({exc}); reduce the "
                    "misalignment magnitude or remove the fault"
                ) from exc
        self._model = model

        # The model's read-only variables are exactly the BPM position outputs.
        # Sorted device order matches the ring order `monitor_xy` walks, so the
        # readout-noise draw sequence is unchanged.
        self._bpm_output_addresses: list[str] = sorted(
            address for address, variable in model.supported_variables.items() if variable.read_only
        )
        self._bpm_device_ids: list[str] = sorted(
            {address.split(":")[3] for address in self._bpm_output_addresses}
        )
        self._refresh_bpm_positions()

    def bind(self, pyat_coupled_records: dict[str, Any]) -> None:
        """Wire the BPM POSITION readback records this bridge should push into.

        Args:
            pyat_coupled_records: the `ServingRecords.pyat_coupled` dict from
                `serving.pvdb.build_serving_pvdb()` -- contains every partition
                (a) record (both the SR magnet SP writables and the SR BPM
                POSITION readbacks) keyed by address. Only the BPM entries are
                retained; magnet SP records are driven by the serving write
                path directly, not by this bridge.
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
        if address not in self._model.supported_variables:
            raise UnknownDeviceError(f"no lattice element named {fam_name!r}")

        # A seeded calibration error (gain/polarity/offset) acts on the
        # commanded current before it's converted to physical strength -- a
        # miscalibrated magnet's *field* differs from its setpoint, not the
        # other way around. Identity (factor=1, offset=0) if unseeded. The
        # model therefore retains the post-calibration *physical* current.
        cal = self._magnet_cal_state.get(fam_name, {})
        value = magnet_cal(value, factor=cal.get("factor", 1.0), offset=cal.get("offset", 0.0))

        # Public set(), not _set(): lume's own read-only and type validation
        # stays on the write path. The model applies, solves once, and rolls
        # the element back itself if the orbit is lost, re-raising
        # OrbitSolveError -- so a rejected write is still a complete no-op here.
        self._model.set({address: value})
        self._refresh_bpm_positions()
        self._push_bpm_readbacks()

    def bpm_positions(self) -> dict[str, float]:
        """Return the most recently solved BPM POSITION readings, keyed by address.

        Available independent of `bind()` -- this is the physics-only view
        used by tests and by any consumer that doesn't need live IOC records.
        """
        return dict(self._bpm_positions)

    # -- internals ---------------------------------------------------------

    def _refresh_bpm_positions(self) -> None:
        """Re-read the model's BPM truth into `_bpm_positions`.

        Public `get()`, not `_get()`, to keep the read path symmetric with the
        write path: lume validates the returned values against the catalog on
        the way out. That is cheap here -- BPM outputs carry `value_range=None`,
        so the check is name/type only.
        """
        self._bpm_positions = dict(self._model.get(self._bpm_output_addresses))

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
