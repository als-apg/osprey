"""Physics bridge: synchronous orbit recompute for coupled setpoint writes.

Wires the coupled setpoint writes of the served namespace into the lattice the
served tree describes: writing a bound device applies the commanded hardware
value to the model, re-solves the closed orbit, and makes every monitor
reading available before the write call returns (FR3/SC3: the recompute
happens synchronously in the write handler itself, never on a
polling/heartbeat tick).

This module fulfills the ``on_pyat_setpoint`` callback contract that
``serving.pvdb.build_serving_pvdb()`` exposes (see that module's docstring):
``PhysicsBridge.on_setpoint`` is passed as ``on_pyat_setpoint``, and
``PhysicsBridge.bind()`` wires the resulting ``ServingRecords.pyat_coupled``
monitor records so they receive the recomputed readings via ``.set()``.

The ring itself lives behind a :class:`~lume.model.LUMEModel` -- by default
:class:`~osprey.services.virtual_accelerator.model.pyat.PyATRingModel`, which
owns the lattice, the hardware->physics calibration each binding declares (see
:mod:`~osprey.services.virtual_accelerator.model.variables` for the conversion
each variable kind applies), the atomic apply-and-solve, and its rollback.

This bridge is the *serving* half: applying the model's fault state (magnet
calibration on the way in, monitor readout errors on the way out), the
readout-noise draws, and the served record wiring. **It resolves nothing from
an address.** Which element a write drives, which monitor an address reads and
on which transverse axis all come from the model's variable catalog -- the
served ``va_bindings.json`` resolved by address, one variable per binding --
so a facility whose addresses are not six colon-separated levels, whose
monitors are not called ``BPM`` and whose currents are not spelled ``CURRENT``
needs no case of its own here. Everything the model owns is reached through
its public ``set()``/``get()``, so a different backend (a surrogate, Cheetah,
Bmad) can be injected through ``model=`` without this module changing.

The bridge keeps no fault state of its own. Fault seeds live in the model as
writable variables and are read back each time the bridge serves, so a fault
written to the model applies on the next write or push. That read is the one
place the bridge expects more of a backend than the ``LUMEModel`` contract: it
looks the faults up under the names the served model declares
(``<element>.<field>``), so a backend that declares them under other names, or
not at all, serves an unfaulted machine.

A monitor also moves on its own. The machine file declares how each channel's
reading wanders and scatters around its level (``texture``, relative
``noise``, ``noise_abs``), and the archived history of that channel is
synthesized from the same declaration. The bridge applies it to the model's
truth through a :class:`MachineMotion` -- the simulation engine, whose
arithmetic is the synthesis's own -- so the present a recorder samples and the
past it was seeded with are one description of the machine. The motion is beam
motion: it is added to the truth before the readout faults, so a monitor's
offsets and gains apply to the moving beam exactly as they apply to the
synthesized history. The truth itself stays motion-free. Because the motion is
a function of time, :meth:`PhysicsBridge.tick` re-serves every reading from the
last solved orbit on each telemetry tick without solving it again; only a write
solves.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from lume.variables import NDVariable

from osprey.services.virtual_accelerator.lattice.errors import bpm_read, magnet_cal
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.model.fault_bounds import (
    BPM_ERROR_FIELDS,
    BPM_ERROR_IDENTITY,
    CORRECTOR_GAIN_FIELDS,
    MAGNET_CAL_IDENTITY,
)
from osprey.services.virtual_accelerator.model.pyat import FAULT_SEPARATOR, UnknownDeviceError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable

    from lume.model import LUMEModel

# The serving layer's convention (see serving/pvdb.py, serving/write_path.py):
# a module logger, never a print. `entrypoint.main()` calls
# `configure_logging()` before it builds this bridge, so these records reach
# the container's stderr alongside the rest of the serving layer's, while an
# import of this module from a library path still configures nothing.
LOG = logging.getLogger(__name__)

# The two transverse axes a monitor binding may read, in the bindings schema's
# own spelling (`bindings.ATTRIBUTES_BY_KIND["monitor"]`). They are the axes of
# the readout model rather than a facility's vocabulary: `bpm_read` mixes the
# two planes through the monitor's roll, so a reading is only defined as a
# pair. What an *address* spells its axis is the facility's business and is
# never read here.
_AXIS_X = "x"
_AXIS_Y = "y"
_AXES = (_AXIS_X, _AXIS_Y)

# `bpm_read`'s full keyword-argument set at identity (no-op) values -- a
# monitor with no fault reads the true orbit position exactly. The model
# supplies the fields it declares a variable for, as `<element>.<field>`; a
# field it declares none for keeps this identity, and `cal_x`/`cal_y` are
# carried by no model variable, so they always do.
_IDENTITY_BPM_ERROR: dict[str, float] = {**BPM_ERROR_IDENTITY, "cal_x": 0.0, "cal_y": 0.0}

# `magnet_cal`'s keyword -> the `<element>.<field>` model variable carrying
# it, and the identity a magnet whose model declares no such variable keeps.
_MAGNET_CAL_FIELDS: dict[str, str] = dict(CORRECTOR_GAIN_FIELDS)
_IDENTITY_MAGNET_CAL: dict[str, float] = {
    keyword: MAGNET_CAL_IDENTITY[field] for keyword, field in _MAGNET_CAL_FIELDS.items()
}

# The `<element>.<field>` suffixes that make a changed model variable a magnet
# calibration -- the only names `refresh()` has to re-apply a setpoint for.
_MAGNET_CAL_VARIABLES: frozenset[str] = frozenset(_MAGNET_CAL_FIELDS.values())


class MachineMotion(Protocol):
    """How a channel's reading moves around the level the model computes.

    :class:`~osprey.simulation.engine.SimulationEngine` is the implementation:
    it holds the machine file's per-channel ``texture``/``noise``/``noise_abs``
    and applies them with the same arithmetic synthesis uses.
    """

    def has_motion(self, pv: str) -> bool:
        """Whether ``measure`` would move a value of ``pv``."""
        ...

    def measure(self, pv: str, value: float, t_abs_s: float) -> float:
        """``value`` with ``pv``'s declared motion at absolute epoch ``t_abs_s``."""
        ...


def _nothing_is_stuck() -> frozenset[str]:
    """No setpoint is stuck: what a bridge with no serving path around it sees."""
    return frozenset()


def _fault_name(element: str, field: str) -> str:
    """The model variable one device's one fault field is declared under."""
    return f"{element}{FAULT_SEPARATOR}{field}"


def _bound_element(variable: Any) -> str | None:
    """Return the element a bound variable's device is named after.

    Two spellings, because a variable states its binding in the shape its own
    kind needs: a monitor sits at one element (``element_name``), and a
    setpoint is written to one or more slices of a lattice field
    (``bindings``, the element read back first). A lattice-level knob -- the
    ring energy -- binds no element at all and answers ``None``.

    The model's faults are keyed by this name, which is the deck's own
    ``FamName`` as the bindings document carries it and as ``VA_BPM_ERRORS``
    and ``VA_CORR_GAIN`` name their devices. Nothing here reassembles a device
    name out of an address.

    Args:
        variable: one entry of the model's ``supported_variables``.

    Returns:
        The element name, or ``None`` for a variable that binds no element.
    """
    element = getattr(variable, "element_name", None)
    if element is not None:
        return str(element)
    slices = getattr(variable, "bindings", ())
    return str(slices[0].element_name) if slices else None


class PhysicsBridge:
    """Serves the coupled write path from a `LUMEModel` physics backend.

    A single `PhysicsBridge` instance holds one model for the lifetime of the
    IOC process, and that model owns one lattice -- every SP write mutates that
    same lattice in place (never rebuilds it), so sequential writes compose
    exactly like their physical counterparts would: writing a device twice is
    idempotent (last value wins, not cumulative), and writing two independent
    devices in either order reaches the same final state (SC3).
    """

    def __init__(
        self,
        model: LUMEModel,
        *,
        rng_seed: int | None = None,
        motion: MachineMotion | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Attach a physics model and seed the readout-noise generator.

        Monitor readout errors and magnet calibration are not arguments here:
        they are the model's own state (`PyATRingModel(bpm_errors=...,
        corrector_gains=...)`), read back from it each time the bridge serves.
        Offsets and noise are therefore in the unit the monitor publishes, not
        in metres: a reading reaches this bridge already mapped through the
        binding's `monitor_inverse`, which is where a facility's
        metre-to-millimetre step lives.

        Args:
            model: the physics backend to serve, already built from the served
                tree. Required: resolving a tree into a lattice and a variable
                catalog is the model's own job, and there is no tree here to
                resolve -- which is also what keeps the backend pluggable,
                since a surrogate, Cheetah or Bmad model implementing the same
                `LUMEModel` contract serves through this bridge unchanged.
            rng_seed: seed for the `numpy.random.Generator` monitor readout
                noise is drawn from; the noise width is the model's own
                `<element>.noise_x`/`noise_y`. `None` seeds from OS entropy
                (non-reproducible), matching `numpy.random.default_rng`'s own
                default.
            motion: the machine file's declared motion per monitor address,
                applied to the truth before the readout faults each time a
                reading is served. A monitor it declares no motion for, and
                every monitor when this is `None` (the default), is served
                exactly as the model and its faults make it. The relative
                noise it applies is the machine file's own number and is not
                scaled by the telemetry noise level the static channels use.
            clock: absolute epoch seconds at which a served reading is taken,
                the time the motion is evaluated at. `time.time` by default.

        Raises:
            ValueError: the model publishes a read-only scalar that states
                no element or no transverse axis, or two that read the same
                axis at one element. Either way its reading could not be
                served -- the first has no monitor to key an error model or a
                record by, the second would drop a served reading silently --
                and a monitor whose value never moves is exactly the failure a
                stand-in target must not have.
        """
        self._bpm_positions: dict[str, float] = {}
        self._bpm_readback_records: dict[str, Any] = {}
        self._setpoint_records: dict[str, Any] = {}
        self._setpoint_addresses: dict[str, list[str]] = {}
        self._stuck = _nothing_is_stuck
        self._rng = np.random.default_rng(rng_seed)
        self._model = model
        self._motion = motion
        self._clock = clock

        # The model's read-only *scalar* variables are exactly the monitor
        # readings the bindings document publishes. Its read-only arrays --
        # the optics -- are model-only and never served, so shape tells the
        # two apart and no name is read to do it. Grouped per element, because
        # a reading is a pair: `bpm_read` mixes the planes through the
        # monitor's roll, and a fault is one device's, not one axis's.
        self._monitors: dict[str, dict[str, str]] = {}
        for address, variable in sorted(model.supported_variables.items()):
            if not variable.read_only or isinstance(variable, NDVariable):
                continue
            element = _bound_element(variable)
            axis = getattr(variable, "axis", None)
            if element is None or axis not in _AXES:
                raise ValueError(
                    f"the read-only variable {address!r} states element={element!r} "
                    f"axis={axis!r}: a monitor this bridge can serve names the element "
                    f"it sits at and one of the axes {_AXES}"
                )
            axes = self._monitors.setdefault(element, {})
            if axis in axes:
                raise ValueError(
                    f"{address!r} and {axes[axis]!r} both read the {axis!r} axis at "
                    f"element {element!r}; one of the two readings could only be dropped"
                )
            axes[axis] = address

        # Sorted, so the model is asked for its truth in one stable order and
        # the readout-noise draw sequence does not depend on dict order.
        self._bpm_output_addresses: list[str] = sorted(
            address for axes in self._monitors.values() for address in axes.values()
        )
        # Per served monitor, `bpm_read` keyword -> the model variable holding
        # it, for the fault fields this model declares; the rest stay
        # identity. Flattened once, so each push reads every fault in one
        # `get()`.
        self._bpm_fault_names: dict[str, dict[str, str]] = {
            element: {
                field: name
                for field in BPM_ERROR_FIELDS
                if (name := _fault_name(element, field)) in model.supported_variables
            }
            for element in self._monitors
        }
        self._bpm_fault_reads: list[str] = [
            name for fields in self._bpm_fault_names.values() for name in fields.values()
        ]
        # The served readings the machine file gives a motion of their own.
        # Decided once: what a channel declares is fixed for the machine file's
        # lifetime, and every other reading is served with no motion at all.
        self._moving: frozenset[str] = (
            frozenset()
            if motion is None
            else frozenset(a for a in self._bpm_output_addresses if motion.has_motion(a))
        )
        self._refresh_bpm_positions()

    def bind(
        self,
        pyat_coupled_records: dict[str, Any],
        *,
        physics_setpoints: frozenset[str] = frozenset(),
    ) -> None:
        """Wire the served records: push into the readbacks, retain the setpoints.

        Replaces any earlier binding and pushes the current readings at once,
        so a freshly bound record never serves its boot value.

        Which record is which is decided by membership, never by reading the
        address text: a record is a reading when the model publishes one on
        its address, and a setpoint when the manifest declared it one. Every
        other record is ignored.

        Args:
            pyat_coupled_records: the `ServingRecords.pyat_coupled` dict from
                `serving.pvdb.build_serving_pvdb()` -- contains every coupled
                record (both the setpoint writables and the monitor readbacks)
                keyed by address.
            physics_setpoints: `ServingRecords.physics_setpoints` -- the
                addresses the manifest declared as setpoints inside that
                partition. Those records are retained, keyed by address, as
                the source of the currently commanded values; the bridge never
                writes them -- the serving write path drives them. Empty (the
                default) retains none, which is enough for a caller that only
                needs the readings pushed.
        """
        served = frozenset(self._bpm_output_addresses)
        self._bpm_readback_records = {
            address: rec for address, rec in pyat_coupled_records.items() if address in served
        }
        self._setpoint_records = {
            address: rec
            for address, rec in pyat_coupled_records.items()
            if address in physics_setpoints
        }
        # Element -> the retained setpoints driving it, so `refresh()` can
        # re-command a magnet whose calibration moved. Built from the model's
        # own bindings rather than from the address text, and in sorted order
        # so a re-applied batch is the same batch on every boot.
        self._setpoint_addresses = {}
        variables = self._model.supported_variables
        for address in sorted(self._setpoint_records):
            variable = variables.get(address)
            element = None if variable is None else _bound_element(variable)
            if element is not None:
                self._setpoint_addresses.setdefault(element, []).append(address)
        self._push_bpm_readbacks()

    def on_setpoint(self, address: str, value: float) -> None:
        """`on_pyat_setpoint` callback: apply one SP write and push readbacks.

        The hook applies the write and nothing else around it: one write is
        one model batch, one solve and one rollback. It converts nothing --
        the hardware-to-physics calibration the facility exported lives on the
        model variable the binding built -- and it publishes no setpoint
        readback, which is the write path's to serve per the binding's own
        readback rule.

        Args:
            address: the manifest address of the setpoint channel that was
                written.
            value: the new value, in the hardware unit the facility states for
                this channel. Absolute, not a delta -- writing the same
                address twice with different values is idempotent (the second
                write fully determines the element's strength).

        Raises:
            UnknownDeviceError: the address is not one the model drives -- it
                binds no lattice element, or it is a monitor reading rather
                than a setpoint.
            OrbitSolveError: if the resulting lattice has no stable closed
                orbit -- the write is rolled back (the element's prior
                strength is restored) before this is raised, so a rejected
                write never leaves the lattice in a broken state.
        """
        variable = self._model.supported_variables.get(address)
        if variable is None or variable.read_only:
            raise UnknownDeviceError(
                f"{address!r} is not a setpoint this model drives; the served bindings "
                "document binds no writable variable to it"
            )

        # A calibration error (gain/polarity/offset) acts on the commanded
        # value before the model converts it to physical strength -- a
        # miscalibrated magnet's *field* differs from its setpoint, not the
        # other way around. This is the fault, not the facility's own
        # calibration: that one belongs to the variable and is applied inside
        # the model, which is why the value here stays in the hardware unit
        # the client wrote and the bridge never touches the binding's curve.
        # Read from the model at every write, so a calibration written to the
        # model applies to the next setpoint. Identity where the model
        # declares no calibration for this element -- a cavity, say.
        element = _bound_element(variable)
        value = magnet_cal(value, **self._magnet_calibration(element))

        # Public set(), not _set(): lume's own read-only and type validation
        # stays on the write path. The model applies, solves once, and rolls
        # the element back itself if the orbit is lost, re-raising
        # OrbitSolveError -- so a rejected write is still a complete no-op here.
        self._model.set({address: value})
        self._refresh_bpm_positions()
        self._push_bpm_readbacks()

    def follow_stuck_setpoints(self, stuck: Callable[[], frozenset[str]]) -> None:
        """Read the stuck set from ``stuck`` whenever :meth:`refresh` runs.

        Which setpoints are stuck is the serving write path's state, not this
        bridge's: the path decides what a write to one does and replaces the
        set whole when it changes. So the set is read at the moment it is
        needed rather than copied here, and a bridge nobody wires up sees none
        stuck -- which is right for one serving no write path at all.

        Wired after the write path exists, which is after this bridge does:
        the bridge has to be bound before the server copies its record specs,
        and the write path is built with the server.

        Args:
            stuck: returns the addresses stuck right now. Called on the run
                loop's thread, from :meth:`refresh` alone.
        """
        self._stuck = stuck

    def refresh(self, changed: Iterable[str]) -> None:
        """Re-serve the ring after a model-only write of `changed` variables.

        The model surface writes fault variables straight into the model,
        never through a served setpoint, so nothing on the write path re-runs
        afterwards. A magnet whose calibration changed is now delivering the
        wrong value for what it was last commanded, and every monitor reading
        is stale -- this puts both right, and is the whole of what a model
        write has to do to become visible.

        Called on the run loop's thread, after the write has landed.

        The commanded values are re-applied in a single `set()` -- one
        closed-orbit solve however many magnets a calibration change
        touches -- and the readings are refreshed and pushed exactly once,
        which keeps a seeded run's readout-noise draw sequence fixed. The
        setpoint records are read, never written: they carry what an operator
        commanded, and a calibration change is not a magnet move. A setpoint
        stuck right now is left alone -- see `follow_stuck_setpoints`.

        Args:
            changed: the model variable names just written. Names this bridge
                serves nothing for are ignored; an empty `changed` still
                refreshes and pushes the readings, because a reset restores
                state this bridge does not track.

        Raises:
            OrbitSolveError: the re-applied values leave the ring without a
                stable closed orbit. The model rolls the whole batch back
                before re-raising, so the ring keeps the strengths it had.
        """
        # Read once, so every address in this batch is judged against one set
        # and a swap part-way through cannot split the batch between two.
        stuck = self._stuck()
        batch: dict[str, float] = {}
        for element in self._recalibrated_elements(changed):
            calibration = self._magnet_calibration(element)
            for address in self._setpoint_addresses.get(element, ()):
                if address in stuck:
                    # A stuck setpoint records what was written to it and
                    # hands the model nothing, so its record holds a value the
                    # ring never took. Re-commanding from it would move the
                    # magnet to a value that was refused, which is the one
                    # thing being stuck means it cannot do.
                    continue
                # Nothing commanded through a served record has no value to
                # re-apply: the new calibration takes effect on the next write.
                batch[address] = magnet_cal(self._setpoint_records[address].get(), **calibration)

        if batch:
            self._model.set(batch)
        self._refresh_bpm_positions()
        self._push_bpm_readbacks()

    @property
    def moves(self) -> bool:
        """Whether any served reading carries a declared motion.

        False means every reading is a function of the model's state alone,
        so re-serving it between writes changes nothing and :meth:`tick` has
        no work worth scheduling.
        """
        return bool(self._moving)

    def tick(self) -> None:
        """Re-serve every monitor reading at the current time.

        The orbit is the one the last write or refresh solved: nothing is
        re-applied to the model and nothing is solved, so a tick can neither
        move a magnet nor fail a solve, and a stuck or rolled-back setpoint is
        exactly as the write path left it. What changes between two ticks is
        the machine file's motion, evaluated at the tick's time, and the
        readout noise draw. The faults are read back from the model as on any
        push, so this runs where every other model access does -- on the run
        loop's thread.
        """
        self._push_bpm_readbacks()

    def bpm_positions(self) -> dict[str, float]:
        """Return the most recently solved monitor readings, keyed by address.

        The model's truth, in the unit each monitor publishes (the binding's
        `monitor_inverse` is applied by the variable, on the way out of the
        model) and with no seeded readout error applied. Available independent
        of `bind()` -- this is the physics-only view used by tests and by any
        consumer that doesn't need live IOC records.
        """
        return dict(self._bpm_positions)

    # -- internals ---------------------------------------------------------

    def _recalibrated_elements(self, changed: Iterable[str]) -> list[str]:
        """The elements among `changed` whose magnet calibration was written.

        Sorted, so a batch built from them is the same batch whatever order
        the surface reports its writes in.
        """
        elements = {
            element
            for element, _dot, field in (name.rpartition(FAULT_SEPARATOR) for name in changed)
            if element and field in _MAGNET_CAL_VARIABLES
        }
        return sorted(elements)

    def _magnet_calibration(self, element: str | None) -> dict[str, float]:
        """`magnet_cal`'s `factor`/`offset` for `element`, as the model holds them now.

        One `get()` for the calibration variables the model declares for this
        element; a field it declares none for is identity, and so is a
        variable that binds no element at all.
        """
        if element is None:
            return dict(_IDENTITY_MAGNET_CAL)
        declared = self._model.supported_variables
        names = {
            keyword: name
            for keyword, field in _MAGNET_CAL_FIELDS.items()
            if (name := _fault_name(element, field)) in declared
        }
        values = self._model.get(list(names.values())) if names else {}
        return {
            **_IDENTITY_MAGNET_CAL,
            **{keyword: values[name] for keyword, name in names.items()},
        }

    def _bpm_read_faults(self) -> dict[str, dict[str, float]]:
        """Each served monitor's `bpm_read` fault keywords, as the model holds them now.

        One `get()` for every reading-error variable the model declares,
        merged over identity per monitor, keyed by element.
        """
        values = self._model.get(self._bpm_fault_reads) if self._bpm_fault_reads else {}
        return {
            element: {
                **_IDENTITY_BPM_ERROR,
                **{field: values[name] for field, name in fields.items()},
            }
            for element, fields in self._bpm_fault_names.items()
        }

    def _refresh_bpm_positions(self) -> None:
        """Re-read the model's monitor truth into `_bpm_positions`.

        Public `get()`, not `_get()`, to keep the read path symmetric with the
        write path: lume validates the returned values against the catalog on
        the way out. That is cheap here -- monitor outputs carry
        `value_range=None`, so the check is name/type only.
        """
        self._bpm_positions = dict(self._model.get(self._bpm_output_addresses))

    def _push_bpm_readbacks(self) -> None:
        """Push each monitor's faulted *reading* into its bound RB record.

        `_bpm_positions` (the physics truth `bpm_positions()` exposes) is
        never touched here -- only the values pushed into IOC records run
        through `bpm_read`, per FR3's "errors apply on the reading, not the
        truth" contract. Both are in the monitor's published unit, so a seeded
        offset is in the unit the facility's own device database states it in.
        The fault fields are the model's current values. Every served monitor
        gets exactly one `bpm_read`, in sorted element order, whether or not
        it is bound or faulted: each call draws both axes, so this keeps a
        seeded run's draw sequence fixed.

        A monitor the document binds on one plane only reads that plane as
        though the other were exactly on axis: the unbound plane has no served
        truth to mix in, so a roll seeded on such a monitor rotates against
        zero.

        The machine file's motion is added to the truth first, at one time
        for the whole push, so it is beam motion the faults then read: an
        offset is subtracted from the moving beam, as the seeded history
        subtracts it from the synthesized one.
        """
        faults = self._bpm_read_faults()
        now = self._clock() if self._moving else 0.0
        for element in sorted(self._monitors):
            axes = self._monitors[element]
            x_address = axes.get(_AXIS_X)
            y_address = axes.get(_AXIS_Y)
            beam_x = self._beam(x_address, now)
            beam_y = self._beam(y_address, now)
            reading_x, reading_y = bpm_read(beam_x, beam_y, rng=self._rng, **faults[element])

            for address, reading in ((x_address, reading_x), (y_address, reading_y)):
                if address is None:
                    continue
                record = self._bpm_readback_records.get(address)
                if record is not None:
                    record.set(reading)

    def _beam(self, address: str | None, now: float) -> float:
        """The beam position a monitor reads at `now`: truth plus declared motion.

        An unbound plane is exactly on axis, and a reading the machine file
        declares no motion for is the truth itself.
        """
        if address is None:
            return 0.0
        truth = self._bpm_positions[address]
        if self._motion is None or address not in self._moving:
            return truth
        return self._motion.measure(address, truth, now)


__all__ = [
    "MachineMotion",
    "OrbitSolveError",
    "PhysicsBridge",
    "UnknownDeviceError",
]
