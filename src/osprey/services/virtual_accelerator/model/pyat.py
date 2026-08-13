"""The ALS-U AR ring as a ``lume-pyat`` model.

:class:`PyATRingModel` is the facility adapter, and only that. Everything
generic about serving a pyAT ring through the LUME contract -- owning one
persistent lattice, atomic multi-variable writes, one solve per batch,
rollback on a lost closed orbit, retained inputs and cached outputs --
belongs to :class:`~lume_pyat.model.LUMEPyATModel` and is inherited rather
than restated here. What is left is the four facility-specific facts that
class cannot know:

- which lattice to drive (:func:`~osprey.services.virtual_accelerator.lattice.build_ring`),
- how a commanded current becomes a strength
  (:class:`~osprey.services.virtual_accelerator.lattice.strengths.StrengthMap`),
- which variables exist and what each is bound to
  (:func:`~osprey.services.virtual_accelerator.model.bindings.build_action_variables`),
- and how a boot failure should read.

The ring, the strength map and the variables are built as one set, in that
order: the map bakes its strength baseline from the very lattice the
simulator goes on to mutate, so a variable's conversion is always relative
to the ring it writes into. Building the map from a second ``build_ring()``
would give a baseline that merely *happens* to match.

Declared defaults are recorded as the retained input values at construction
but are not written to the lattice, so the ring boots in exactly the state
``build_ring()`` produced -- the ``machine.json`` nominals the catalog
carries as ``default_value`` *are* that state. :meth:`reset` writes them
back through the transformer, which is what makes a reset undo writes
without disturbing construction-time faults.

This module imports nothing from ``ioc`` and never touches EPICS, and -- like
the base class -- never raises ``SystemExit``: killing the host process is
the serving layer's decision.
"""

from __future__ import annotations

from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator

from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.lattice.strengths import StrengthMap
from osprey.services.virtual_accelerator.model.bindings import build_action_variables

# An alias, never a subclass. ``CurrentSetpointVariable`` raises
# ``UnknownElementError`` from inside the write path, and importing this
# module from there to raise a distinct class would close an import cycle
# (pyat -> bindings -> variables -> pyat). Aliasing is what keeps an existing
# ``except UnknownDeviceError`` catching every lookup failure the model can
# produce, wherever in the stack it was raised.
UnknownDeviceError = UnknownElementError


class PyATRingModel(LUMEPyATModel):
    """LUME model over a single persistent ALS-U AR ``at.Lattice``.

    Magnet and corrector setpoints in (Amps, absolute and post-calibration),
    BPM positions out (meters). One instance owns one lattice for its whole
    lifetime -- every write mutates that same lattice in place -- so
    sequential writes compose exactly like their physical counterparts
    would, and a fault seeded at construction survives every later write and
    every :meth:`reset`.
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
                :func:`~lume_pyat.utils.apply_misalignment` (``dx``/``dy``/
                ``roll``, all optional). Applied once, here, after the ring is
                built and before the nominal orbit is solved -- an element
                absent from this dict keeps AT's default (unmisaligned)
                T1/T2/R1/R2. Every fam_name is validated against the ring
                before any element is mutated.

        Raises:
            UnknownDeviceError: a misalignment names an element the ring does
                not have, or a variable binds one.
            OrbitSolveError: the seeded misalignments leave the ring without a
                stable closed orbit -- the message names the seeded elements
                and their magnitudes so an otherwise opaque boot failure is
                diagnosable. Deliberately *not* ``SystemExit``: whether an
                unusable model should end the process is the caller's call.
        """
        ring = build_ring()
        strength_map = StrengthMap(ring)
        variables = build_action_variables(strength_map=strength_map)

        try:
            super().__init__(
                simulator=PyATSimulator(ring, element_misalignments=element_misalignments),
                action_variables=list(variables.values()),
            )
        except OrbitSolveError as exc:
            raise OrbitSolveError(
                f"seeded misalignments {element_misalignments!r} left the SR "
                f"lattice without a stable closed orbit at boot ({exc}); reduce the "
                "misalignment magnitude or remove the fault"
            ) from exc


__all__ = [
    "PyATRingModel",
    "UnknownDeviceError",
    "OrbitSolveError",
]
