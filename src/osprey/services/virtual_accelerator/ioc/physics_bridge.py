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
grammar, applying the model's fault state (magnet calibration on the way in,
BPM readout errors on the way out), the readout-noise draws, and the served
record wiring. Everything the model owns is reached through its public
``set()``/``get()``, so a different backend (a surrogate, Cheetah, Bmad) can be
injected through ``model=`` for the orbit itself.

The bridge keeps no fault state of its own. Fault seeds live in the model as
writable variables, and the bridge reads the current values back each time it
serves -- so a fault written to the model applies on the next write or push.
That read is where the bridge is bound to the bundled demo model rather than
to the ``LUMEModel`` contract: it looks the faults up under the names the
bundled model declares (``BPMnn.<field>``, ``<fam>.cal_factor``/
``.cal_offset``), so a backend that declares them under other names, or not at
all, serves an unfaulted machine. Which records are readings and which are
setpoints is not the bridge's call either: readings are the model's read-only
scalar variables, and setpoints are whatever the manifest declared (see
:meth:`PhysicsBridge.bind`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from lume.variables import NDVariable

from osprey.services.virtual_accelerator.lattice.errors import bpm_read, magnet_cal
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.model.fault_bounds import BPM_ERROR_FIELD_BOUNDS
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel, UnknownDeviceError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from lume.model import LUMEModel

# The serving layer's convention (see serving/pvdb.py, serving/write_path.py):
# a module logger, never a print. `entrypoint.main()` calls
# `configure_logging()` before it builds this bridge, so these records reach
# the container's stderr alongside the rest of the serving layer's, while an
# import of this module from a library path still configures nothing.
LOG = logging.getLogger(__name__)

_CURRENT_FIELD = "CURRENT"
_BPM_SYSTEM_FAMILY = ("DIAG", "BPM")
_BPM_FIELD = "POSITION"

# `bpm_read`'s full keyword-argument set at identity (no-op) values -- a BPM
# with no fault reads the true orbit position exactly. The model supplies the
# fields `BPM_ERROR_FIELD_BOUNDS` names, as `BPMnn.<field>`; a field the model
# declares no variable for keeps this identity. `cal_x`/`cal_y` are carried by
# no model variable, so they always do.
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

# `magnet_cal`'s keyword -> the `<fam>.<field>` model variable that carries
# it, and the identity a magnet whose model declares no such variable keeps.
_MAGNET_CAL_FIELDS: dict[str, str] = {"factor": "cal_factor", "offset": "cal_offset"}
_IDENTITY_MAGNET_CAL: dict[str, float] = {"factor": 1.0, "offset": 0.0}

# The `<fam>.<field>` suffixes that make a changed model variable a magnet
# calibration -- the only names `refresh()` has to re-apply a setpoint for.
_MAGNET_CAL_VARIABLES: frozenset[str] = frozenset(_MAGNET_CAL_FIELDS.values())


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
        rng_seed: int | None = None,
    ) -> None:
        """Attach a physics model and seed the readout-noise generator.

        BPM readout errors and magnet calibration are not arguments here:
        they are the model's state (e.g. `PyATRingModel(bpm_errors=...,
        corrector_gains=...)`), read back from it each time the bridge serves.

        Args:
            model: the physics backend to serve. `None` (the default)
                constructs a `PyATRingModel`, forwarding `element_misalignments`
                to it. A different `LUMEModel` (a surrogate, Cheetah, Bmad)
                serves its orbit through this bridge; its faults apply only
                where it declares them under the bundled model's names (see
                the module docstring).
            element_misalignments: fam_name (e.g. "QF07", "DIPOLE03") -> kwargs
                for `errors.apply_misalignment` (`dx`/`dy`/`roll`, all optional),
                seeded on the ring the default model builds. Mutually exclusive
                with `model`: a caller supplying its own model is responsible for
                that model's ring state.
            rng_seed: seed for the `numpy.random.Generator` BPM readout noise
                is drawn from; the noise sigma is the model's `BPMnn.noise_x`/
                `noise_y`. `None` seeds from OS entropy (non-reproducible),
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
        self._setpoint_records: dict[str, Any] = {}
        self._setpoint_addresses: dict[str, str] = {}
        self._rng = np.random.default_rng(rng_seed)

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

        # The model's read-only scalar variables are its per-address readings
        # (on the bundled ring, the BPM positions). Its read-only arrays -- the
        # optics -- are model-only and never served, so shape tells the two
        # apart; the address text is never read. Sorted order matches the ring
        # order `monitor_xy` walks, so the readout-noise draw sequence follows
        # the ring.
        self._bpm_output_addresses: list[str] = sorted(
            name
            for name, variable in model.supported_variables.items()
            if variable.read_only and not isinstance(variable, NDVariable)
        )
        self._bpm_device_ids: list[str] = sorted(
            {address.split(":")[3] for address in self._bpm_output_addresses}
        )
        # Per served device, `bpm_read` keyword -> the model variable holding
        # it, for the fault fields this model declares; the rest stay identity.
        # Flattened once, so each push reads every BPM fault in one `get()`.
        self._bpm_fault_names: dict[str, dict[str, str]] = {
            device: {
                field: f"BPM{device}.{field}"
                for field in BPM_ERROR_FIELD_BOUNDS
                if f"BPM{device}.{field}" in model.supported_variables
            }
            for device in self._bpm_device_ids
        }
        self._bpm_fault_reads: list[str] = [
            name for fields in self._bpm_fault_names.values() for name in fields.values()
        ]
        self._refresh_bpm_positions()

    def bind(
        self,
        pyat_coupled_records: dict[str, Any],
        *,
        physics_setpoints: frozenset[str] = frozenset(),
    ) -> None:
        """Wire the served records: push into the readbacks, retain the setpoints.

        Replaces any earlier binding and pushes the current BPM readings at
        once, so a freshly bound record never serves its boot value.

        Which record is which is decided by membership, never by reading the
        address text: a record is a reading when the model declares a
        read-only scalar under its address, and a setpoint when the manifest
        declared it one. Every other record is ignored.

        Args:
            pyat_coupled_records: the `ServingRecords.pyat_coupled` dict from
                `serving.pvdb.build_serving_pvdb()` -- every partition (a)
                record keyed by address, readings and setpoints alike.
            physics_setpoints: `ServingRecords.physics_setpoints` -- the
                addresses the manifest declared as setpoints inside that
                partition. Those records are retained, keyed by address, as
                the source of the currently commanded values; the bridge
                never writes them -- the serving write path drives them.
                Empty (the default) retains none, which is enough for a
                caller that only needs the readings pushed.
        """
        self._bpm_readback_records = {
            address: rec
            for address, rec in pyat_coupled_records.items()
            if address in self._bpm_output_addresses
        }
        self._setpoint_records = {
            address: rec
            for address, rec in pyat_coupled_records.items()
            if address in physics_setpoints
        }
        # fam_name -> its CURRENT setpoint address, so `refresh()` can find
        # the commanded current of a magnet the model named by fam_name.
        # Built once here rather than searched per refresh, and restricted to
        # the CURRENT field for the same reason `on_setpoint` refuses any
        # other: a magnet's strength follows its current, nothing else.
        self._setpoint_addresses = {}
        for address in self._setpoint_records:
            try:
                _system, family, device, field = _parse_pyat_coupled_address(address)
            except UnknownDeviceError:
                # The fam_name lookup is the bundled tree's spelling (module
                # docstring). A setpoint spelled otherwise is still retained;
                # it just cannot be re-applied by name on a calibration change.
                continue
            if field == _CURRENT_FIELD:
                self._setpoint_addresses[f"{family}{device}"] = address
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

        # A calibration error (gain/polarity/offset) acts on the commanded
        # current before it's converted to physical strength -- a
        # miscalibrated magnet's *field* differs from its setpoint, not the
        # other way around. Read from the model at every write, so a
        # calibration written to the model applies to the next setpoint. The
        # model therefore retains the post-calibration *physical* current.
        value = magnet_cal(value, **self._magnet_calibration(fam_name))

        # Public set(), not _set(): lume's own read-only and type validation
        # stays on the write path. The model applies, solves once, and rolls
        # the element back itself if the orbit is lost, re-raising
        # OrbitSolveError -- so a rejected write is still a complete no-op here.
        self._model.set({address: value})
        self._refresh_bpm_positions()
        self._push_bpm_readbacks()

    def refresh(self, changed: Iterable[str]) -> None:
        """Re-serve the ring after a model-only write of `changed` variables.

        The model surface writes fault variables straight into the model,
        never through a served setpoint, so nothing on the write path re-runs
        afterwards. A magnet whose calibration changed is now delivering the
        wrong current for what it was last commanded, and every BPM reading
        is stale -- this puts both right, and is the whole of what a model
        write has to do to become visible.

        Called on the run loop's thread, after the write has landed, by
        `serving.model_surface.ModelSurface` (its `refresh=` argument).

        Args:
            changed: the model variable names just written, e.g.
                `["QF07.cal_factor", "BPM01.offset_x", "stuck_setpoints"]`.
                Names the bridge serves nothing for are ignored; an empty
                `changed` still refreshes and pushes the readings, because a
                reset restores state this bridge does not track.

        The commanded currents are re-applied in a single `set()` -- one
        closed-orbit solve however many magnets a family-wide calibration
        change touches -- and the BPM readings are refreshed and pushed
        exactly once, which keeps a seeded run's readout-noise draw sequence
        fixed. The setpoint records are read, never written: they carry what
        an operator commanded, and a calibration change is not a magnet move.

        Raises:
            OrbitSolveError: the re-applied currents leave the ring without a
                stable closed orbit. The model rolls the whole batch back
                before re-raising, so the ring keeps the strengths it had.
        """
        batch: dict[str, float] = {}
        for fam_name in self._recalibrated_families(changed):
            address = self._setpoint_addresses.get(fam_name)
            if address is None:
                # Nothing commanded this magnet through a served record, so
                # there is no current to re-apply: the new calibration takes
                # effect on the next setpoint write.
                continue
            commanded = self._setpoint_records[address].get()
            batch[address] = magnet_cal(commanded, **self._magnet_calibration(fam_name))

        if batch:
            self._model.set(batch)
        self._refresh_bpm_positions()
        self._push_bpm_readbacks()

    def bpm_positions(self) -> dict[str, float]:
        """Return the most recently solved BPM POSITION readings, keyed by address.

        Available independent of `bind()` -- this is the physics-only view
        used by tests and by any consumer that doesn't need live IOC records.
        """
        return dict(self._bpm_positions)

    # -- internals ---------------------------------------------------------

    def _recalibrated_families(self, changed: Iterable[str]) -> list[str]:
        """The fam_names among `changed` whose magnet calibration was written.

        Sorted, so a batch built from them is the same batch whatever order
        the surface reports its writes in.
        """
        families = {
            fam_name
            for fam_name, _dot, field in (name.rpartition(".") for name in changed)
            if fam_name and field in _MAGNET_CAL_VARIABLES
        }
        return sorted(families)

    def _magnet_calibration(self, fam_name: str) -> dict[str, float]:
        """`magnet_cal`'s `factor`/`offset` for `fam_name`, as the model holds them now.

        One `get()` for the calibration variables the model declares for this
        magnet; a field it declares none for is identity.
        """
        names = {
            keyword: f"{fam_name}.{field}"
            for keyword, field in _MAGNET_CAL_FIELDS.items()
            if f"{fam_name}.{field}" in self._model.supported_variables
        }
        values = self._model.get(list(names.values())) if names else {}
        return {
            **_IDENTITY_MAGNET_CAL,
            **{keyword: values[name] for keyword, name in names.items()},
        }

    def _bpm_read_faults(self) -> dict[str, dict[str, float]]:
        """Each served device's `bpm_read` fault keywords, as the model holds them now.

        One `get()` for every BPM fault variable the model declares, merged
        over identity per device, keyed by device id.
        """
        values = self._model.get(self._bpm_fault_reads) if self._bpm_fault_reads else {}
        return {
            device: {
                **_IDENTITY_BPM_ERROR,
                **{field: values[name] for field, name in fields.items()},
            }
            for device, fields in self._bpm_fault_names.items()
        }

    def _refresh_bpm_positions(self) -> None:
        """Re-read the model's BPM truth into `_bpm_positions`.

        Public `get()`, not `_get()`, to keep the read path symmetric with the
        write path: lume validates the returned values against the catalog on
        the way out. That is cheap here -- BPM outputs carry `value_range=None`,
        so the check is name/type only.
        """
        self._bpm_positions = dict(self._model.get(self._bpm_output_addresses))

    def _push_bpm_readbacks(self) -> None:
        """Push each BPM's faulted *reading* into its bound RB record.

        `_bpm_positions` (the physics truth `bpm_positions()` exposes) is
        never touched here -- only the values pushed into IOC records run
        through `bpm_read`, per FR3's "errors apply on the reading, not the
        truth" contract. The fault fields are the model's current values.
        Every served device gets exactly one `bpm_read`, in sorted device
        order, whether or not it is bound or faulted: each call draws both
        noise axes, so this keeps a seeded run's draw sequence fixed.
        """
        faults = self._bpm_read_faults()
        for device in self._bpm_device_ids:
            true_x = self._bpm_positions[_bpm_address(device, "X")]
            true_y = self._bpm_positions[_bpm_address(device, "Y")]
            reading_x, reading_y = bpm_read(true_x, true_y, rng=self._rng, **faults[device])

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
