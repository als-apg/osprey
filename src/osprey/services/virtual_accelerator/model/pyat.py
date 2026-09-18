"""The served ring as a ``lume-pyat`` model.

:class:`PyATRingModel` is the facility adapter, and only that. Everything
generic about serving a pyAT ring through the LUME contract -- owning one
persistent lattice, atomic multi-variable writes, one solve per batch,
rollback on a lost closed orbit, retained inputs and cached outputs --
belongs to :class:`~lume_pyat.model.LUMEPyATModel` and is inherited rather
than restated here. What is left is the three facility-specific facts that
class cannot know, and all three are read out of the tree this model is
served:

- which lattice to drive, and whether it is still the one the bindings were
  derived against
  (:func:`~osprey.services.virtual_accelerator.lattice.build_ring`),
- which variables exist, what each is bound to and how a commanded hardware
  value becomes a physics one
  (:func:`~osprey.services.virtual_accelerator.model.bindings.build_action_variables`
  over the served ``va_bindings.json``, through
  :func:`~osprey.services.virtual_accelerator.model.catalog.build_variable_catalog`),
- and how a boot failure should read.

**One served tree, named once.** ``data_dir`` is a facility's ``data/``
directory and every file this model reads is resolved against it through
:class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`:
the lattice, the bindings, the ``machine.json`` nominals and the
``channel_limits.json`` bands. There is no default and no fallback -- a
standalone demo names the packaged tree explicitly, like any other -- because
the alternative is serving the framework's own demo nominals and bands behind
a facility's addresses, and the band a nominal is weighed against is the one
thing a facility must recognise as its own.

**The channel list is passed in, not resolved here.** It is the set of
channels the deployment resolved and the IOC is serving on, and the model has
to be on exactly that namespace: resolving a second one would let the two
drift apart silently. The tree is not asked for it either, because an emitted
tree carries no manifest at all -- the channel set is derived at build time
from the facility's databases.

**The energy knob has to be coupled, and that happens here.** The catalog
hands out one variable per address, so nothing inside it can see the finished
set; :func:`~osprey.services.virtual_accelerator.model.bindings.couple_energy_knob`
adopts every rigidity-scaled setpoint into the knob once they all exist,
which is after the catalog is built and before the model takes the variables
over. Skipping it would leave a knob that writes electron-volts and rescales
nothing, with the fields a rescale touches outside the model's rollback.

Declared defaults are recorded as the retained input values at construction
but are not written to the lattice, so the ring boots in exactly the state
``build_ring()`` produced -- the served ``machine.json`` nominals the catalog
carries as ``default_value`` *are* that state. :meth:`reset` writes them back
through the calibrations, which is what makes a reset undo writes without
disturbing construction-time faults.

This module imports nothing from ``ioc`` and never touches EPICS, and -- like
the base class -- never raises ``SystemExit``: killing the host process is
the serving layer's decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator

from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.bindings import (
    build_action_variables,
    couple_energy_knob,
)
from osprey.services.virtual_accelerator.model.catalog import build_variable_catalog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

# An alias, never a subclass. The backend raises ``UnknownElementError``
# itself, from every element lookup it performs -- when it adopts the
# variables, when a misalignment names an element, and on the write path --
# so a distinct class here could only ever cover failures raised on this side
# of the boundary. Aliasing is what keeps an existing ``except
# UnknownDeviceError`` catching every lookup failure the model can produce,
# wherever in the stack it was raised.
UnknownDeviceError = UnknownElementError


class PyATRingModel(LUMEPyATModel):
    """A LUME model over the single persistent ``at.Lattice`` a tree serves.

    Hardware setpoints in, monitor readings out, both in the units the
    facility's own calibrations state -- the conversions live on the variables
    (:mod:`~osprey.services.virtual_accelerator.model.variables`), which read
    them from the served bindings document. One instance owns one lattice for
    its whole lifetime -- every write mutates that same lattice in place -- so
    sequential writes compose exactly like their physical counterparts would,
    and a fault seeded at construction survives every later write and every
    :meth:`reset`.
    """

    def __init__(
        self,
        data_dir: Path,
        channels: list[dict],
        *,
        element_misalignments: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Build the ring from a served tree and adopt the variables it binds.

        Args:
            data_dir: The facility ``data/`` directory to serve: the lattice
                and the bindings under its ``simulation/``, the scenario seed
                and the write bands where
                :class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`
                resolves them. Required, and never defaulted.
            channels: The served manifest's channel list, exactly as
                :func:`~osprey.services.virtual_accelerator.manifest.loaders.load_manifest_file`
                returns it -- the namespace the deployment resolved, which the
                IOC beside this model is serving on.
            element_misalignments: Element ``FamName`` -> kwargs for
                :func:`~lume_pyat.utils.apply_misalignment` (``dx``/``dy``/
                ``roll``, all optional). Applied once, here, after the ring is
                built and before the nominal orbit is solved -- an element
                absent from this dict keeps AT's default (unmisaligned)
                T1/T2/R1/R2. Every name is validated against the ring before
                any element is mutated.

        Raises:
            FileNotFoundError: The tree carries no lattice, no bindings, no
                ``machine.json`` or no ``channel_limits.json``; the message
                names the file that is missing.
            BindingsError: The served bindings document is refused by its own
                schema, the lattice is not the file those bindings were
                derived against, or a bound element name is not carried by
                exactly one element.
            ValueError: A served ``machine.json`` nominal falls outside its
                ``channel_limits.json`` band -- the boot refusal that makes
                the model's declared state the facility's own.
            UnknownDeviceError: A misalignment names an element the ring does
                not have, or a variable binds one.
            OrbitSolveError: The seeded misalignments leave the ring without a
                stable closed orbit -- the message names the seeded elements
                and their magnitudes so an otherwise opaque boot failure is
                diagnosable. Deliberately *not* ``SystemExit``: whether an
                unusable model should end the process is the caller's call.
        """
        paths = ManifestPaths(data_root=data_dir)
        # Read for the bindings alone. ``build_ring`` reads the same file and
        # owns every check that the two files still describe one accelerator,
        # so what comes back here needs no validation of its own -- and a
        # document this read accepts is the document that ring was checked
        # against.
        document = load_bindings(paths.va_bindings)
        ring = build_ring(paths)
        catalog = build_variable_catalog(paths, channels, build_action_variables(document))
        # Before the model takes the variables over: the knob can only adopt
        # the setpoints it rescales once the whole catalog exists.
        couple_energy_knob(catalog)

        try:
            super().__init__(
                simulator=PyATSimulator(ring, element_misalignments=element_misalignments),
                action_variables=list(catalog.values()),
            )
        except OrbitSolveError as exc:
            raise OrbitSolveError(
                f"seeded misalignments {element_misalignments!r} left the lattice "
                f"{paths.lattice_json} without a stable closed orbit at boot ({exc}); "
                "reduce the misalignment magnitude or remove the fault"
            ) from exc


__all__ = [
    "PyATRingModel",
    "UnknownDeviceError",
    "OrbitSolveError",
]
