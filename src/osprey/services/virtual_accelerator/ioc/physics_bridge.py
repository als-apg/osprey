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

This bridge is the *serving* half: the seeded readout and calibration faults,
and the served record wiring. **It resolves nothing from an address.** Which
element a write drives, which monitor an address reads and on which transverse
axis all come from the model's variable catalog -- the served
``va_bindings.json`` resolved by address, one variable per binding -- so a
facility whose addresses are not six colon-separated levels, whose monitors
are not called ``BPM`` and whose currents are not spelled ``CURRENT`` needs no
case of its own here. Everything the model owns is reached through its public
``set()``/``get()``, so a different backend (a surrogate, Cheetah, Bmad) can
be injected through ``model=`` without this module changing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from osprey.services.virtual_accelerator.lattice.errors import bpm_read, magnet_cal
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.model.pyat import UnknownDeviceError

if TYPE_CHECKING:  # pragma: no cover - typing only
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
# monitor with no seeded error reads the true orbit position exactly. Fault
# dicts passed into PhysicsBridge only need to name the fields they perturb;
# the rest fall back to this identity.
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


def _bound_element(variable: Any) -> str | None:
    """Return the element a bound variable's device is named after.

    Two spellings, because a variable states its binding in the shape its own
    kind needs: a monitor sits at one element (``element_name``), and a
    setpoint is written to one or more slices of a lattice field
    (``bindings``, the element read back first). A lattice-level knob -- the
    ring energy -- binds no element at all and answers ``None``.

    The seeded faults are keyed by this name, which is the deck's own
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
        bpm_errors: dict[str, dict[str, float]] | None = None,
        corrector_gains: dict[str, dict[str, float]] | None = None,
        rng_seed: int | None = None,
    ) -> None:
        """Attach a physics model and, optionally, seed FR3/FR4 serving faults.

        Args:
            model: the physics backend to serve, already built from the served
                tree. Required: resolving a tree into a lattice and a variable
                catalog is the model's own job, and there is no tree here to
                resolve -- which is also what keeps the backend pluggable,
                since a surrogate, Cheetah or Bmad model implementing the same
                `LUMEModel` contract serves through this bridge unchanged.
            bpm_errors: monitor element name, as the deck spells it and as the
                bindings document carries it -> a partial override of
                `errors.bpm_read`'s keyword args; missing fields fall back to
                identity (see
                `_IDENTITY_BPM_ERROR`). A monitor absent from this dict reads
                with identity error (i.e. exactly its true position). Offsets
                and noise are in the unit the monitor publishes, not in
                metres: a reading reaches this bridge already mapped through
                the binding's `monitor_inverse`, which is where a facility's
                metre-to-millimetre step lives. An element name the lattice
                carries no monitor at is not fatal -- it perturbs nothing, and
                warns once at construction so the typo is visible.
            corrector_gains: setpoint element name, spelled the same way -> a
                partial override of `errors.magnet_cal`'s `factor`/`offset`;
                missing fields default to `factor=1.0, offset=0.0` (identity).
            rng_seed: seed for the `numpy.random.Generator` monitor readout
                noise is drawn from. `None` seeds from OS entropy
                (non-reproducible), matching `numpy.random.default_rng`'s own
                default.

        Raises:
            ValueError: the model publishes a read-only variable that states
                no element or no transverse axis, or two that read the same
                axis at one element. Either way its reading could not be
                served -- the first has no monitor to key an error model or a
                record by, the second would drop a served reading silently --
                and a monitor whose value never moves is exactly the failure a
                stand-in target must not have.
        """
        self._bpm_positions: dict[str, float] = {}
        self._bpm_readback_records: dict[str, Any] = {}
        self._rng = np.random.default_rng(rng_seed)
        self._bpm_error_state: dict[str, dict[str, float]] = dict(bpm_errors or {})
        self._magnet_cal_state: dict[str, dict[str, float]] = dict(corrector_gains or {})
        self._model = model

        # The model's read-only variables are exactly the monitor readings the
        # bindings document publishes. Grouped per element, because a reading
        # is a pair: `bpm_read` mixes the planes through the monitor's roll,
        # and a seeded error is one device's, not one axis's.
        self._monitors: dict[str, dict[str, str]] = {}
        for address, variable in sorted(model.supported_variables.items()):
            if not variable.read_only:
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
        self._warn_unknown_bpm_error_ids()
        self._refresh_bpm_positions()

    def bind(self, pyat_coupled_records: dict[str, Any]) -> None:
        """Wire the monitor readback records this bridge should push into.

        Args:
            pyat_coupled_records: the `ServingRecords.pyat_coupled` dict from
                `serving.pvdb.build_serving_pvdb()` -- contains every coupled
                record (both the setpoint writables and the monitor readbacks)
                keyed by address. Only the records on an address the bindings
                document publishes a monitor reading on are retained; setpoint
                records are driven by the serving write path directly, not by
                this bridge.
        """
        served = frozenset(self._bpm_output_addresses)
        self._bpm_readback_records = {
            address: rec for address, rec in pyat_coupled_records.items() if address in served
        }
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

        # A seeded calibration error (gain/polarity/offset) acts on the
        # commanded value before the model converts it to physical strength --
        # a miscalibrated magnet's *field* differs from its setpoint, not the
        # other way around. This is the seeded FR4 fault, not the facility's
        # own calibration: that one belongs to the variable and is applied
        # inside the model, which is why the value here stays in the hardware
        # unit the client wrote and the bridge never touches the binding's
        # curve. Identity (factor=1, offset=0) if unseeded.
        element = _bound_element(variable)
        cal = self._magnet_cal_state.get(element, {}) if element is not None else {}
        value = magnet_cal(value, factor=cal.get("factor", 1.0), offset=cal.get("offset", 0.0))

        # Public set(), not _set(): lume's own read-only and type validation
        # stays on the write path. The model applies, solves once, and rolls
        # the element back itself if the orbit is lost, re-raising
        # OrbitSolveError -- so a rejected write is still a complete no-op here.
        self._model.set({address: value})
        self._refresh_bpm_positions()
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

    def _warn_unknown_bpm_error_ids(self) -> None:
        """Warn once per seeded element name this lattice has no monitor at.

        `_push_bpm_readbacks` merges the seeded state per *served* monitor, so
        a name the document binds no monitor to is dropped by that `.get()`
        and perturbs nothing. Silently: a typo'd device would serve a
        perfectly unperturbed machine while looking configured, which is
        exactly the failure a stand-in target must not have. Sorted, so the
        boot log reads the same on every start.
        """
        for element in sorted(set(self._bpm_error_state) - set(self._monitors)):
            LOG.warning(
                "VA_BPM_ERRORS names %s, which this lattice has no monitor at; "
                "its readout errors apply to nothing (on the live stand-in the "
                "same value arrives through VA_STANDIN_BPM_ERRORS)",
                element,
            )

    def _refresh_bpm_positions(self) -> None:
        """Re-read the model's monitor truth into `_bpm_positions`.

        Public `get()`, not `_get()`, to keep the read path symmetric with the
        write path: lume validates the returned values against the catalog on
        the way out. That is cheap here -- monitor outputs carry
        `value_range=None`, so the check is name/type only.
        """
        self._bpm_positions = dict(self._model.get(self._bpm_output_addresses))

    def _push_bpm_readbacks(self) -> None:
        """Push each monitor's seeded-error *reading* into its bound RB record.

        `_bpm_positions` (the physics truth `bpm_positions()` exposes) is
        never touched here -- only the values pushed into IOC records run
        through `bpm_read`, per FR3's "errors apply on the reading, not the
        truth" contract. Both are in the monitor's published unit, so a seeded
        offset is in the unit the facility's own device database states it in.

        A monitor the document binds on one plane only reads that plane as
        though the other were exactly on axis: the unbound plane has no served
        truth to mix in, so a roll seeded on such a monitor rotates against
        zero.
        """
        for element in sorted(self._monitors):
            axes = self._monitors[element]
            x_address = axes.get(_AXIS_X)
            y_address = axes.get(_AXIS_Y)
            true_x = self._bpm_positions[x_address] if x_address is not None else 0.0
            true_y = self._bpm_positions[y_address] if y_address is not None else 0.0
            state = {**_IDENTITY_BPM_ERROR, **self._bpm_error_state.get(element, {})}
            reading_x, reading_y = bpm_read(true_x, true_y, rng=self._rng, **state)

            for address, reading in ((x_address, reading_x), (y_address, reading_y)):
                if address is None:
                    continue
                record = self._bpm_readback_records.get(address)
                if record is not None:
                    record.set(reading)


__all__ = [
    "OrbitSolveError",
    "PhysicsBridge",
    "UnknownDeviceError",
]
