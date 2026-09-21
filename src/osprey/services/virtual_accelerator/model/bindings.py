"""Bind the model's variable catalog to the lattice the bindings document names.

:func:`~osprey.services.virtual_accelerator.model.catalog.build_variable_catalog`
derives what every model variable *is* -- its address, nominal, unit,
declared range and validation config -- from the served manifest and the
facility databases beside it. What it cannot know is what each variable is
*bound to*: the lattice element a channel drives, the attribute and
component the write lands on, how a split device shares one setpoint, and
whether the physics value moves with the beam rigidity. Those facts are read
here, one binding at a time, out of ``va_bindings.json``.

**Nothing in this module derives a facility fact from an address.** The
element name is the deck's own ``FamName`` as the exporter read it off the
ring, the monitor's axis is the binding's attribute, and the polynomial
component is the binding's index -- so a facility whose elements are not
named ``{family}{device}`` and whose position subfields are not spelled
``X``/``Y`` needs no case of its own here. The older path concatenated a
family and a device number into an element name and mapped a subfield to an
axis through a table; both were ALS-U conventions masquerading as model
logic, and both are gone.

**Factories, not variables.** Each binding becomes a *factory* keyed by the
address it claims, and the catalog calls it with the ``ScalarVariable``
fields it derived. That is what keeps parity structural: there is one
derivation path for a variable's name, nominal, unit and range, and a bound
catalog can differ from a declared one only in the binding itself. Building
variables here instead would put a second construction loop beside the
catalog's, to be kept in step with ``machine.json`` by hand.

**A binding is never skipped.** Every binding yields a factory, and a
factory that cannot build its variable raises. An address the document binds
but this module dropped would reach the catalog as a plain
``ScalarVariable``: a channel that accepts writes, reads them back
unchanged, and moves nothing on the lattice. The refusals that matter come
earlier or later, and neither is restated here --
:func:`~osprey.services.virtual_accelerator.lattice.ring.build_ring` refuses
a bound element name the served lattice does not carry exactly once, naming
the family, and ``LUMEPyATModel`` validates every element, attribute and
index when it adopts the variables. A slice weight is any finite non-zero
factor and the served base validates it, so nothing here narrows it: a series
string gives each magnet its fixed factor and a split device shares ``1/n``.

**The energy knob has to be coupled.** The knob rescales every
rigidity-scaled setpoint when the ring energy moves, and it can only adopt
them once they all exist -- which is after the catalog is built, not while
its factories are being handed out. :func:`couple_energy_knob` is that step,
and a caller that skips it gets a knob which writes the ring energy and
rescales nothing.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from lume_pyat.actions import ElementBinding, PyATWritableScalarVariable

from osprey.services.virtual_accelerator.model.variables import (
    EnergyVariable,
    KickVariable,
    MonitorVariable,
    RFVariable,
    StrengthVariable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

    from lume.variables import ScalarVariable

    from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument

    #: What the catalog calls per address: ``factory(channel, **scalar_kwargs)``.
    VariableFactory = Callable[..., ScalarVariable]

#: The class each writable element kind is implemented by. The three name
#: what a kind writes and nothing more -- a weight is any finite non-zero
#: factor for all of them -- which is why the construction below is one
#: function rather than three.
_SETPOINT_CLASS: dict[str, type[PyATWritableScalarVariable]] = {
    "strength": StrengthVariable,
    "kick": KickVariable,
    "rf": RFVariable,
}


def _element_bindings(binding: Binding) -> list[ElementBinding]:
    """Turn a binding's slices into the element bindings lume-pyat writes to.

    One per slice, in document order, all on the same attribute and component
    -- a split device is one setpoint over several elements of one lattice
    field. The first slice is the one a read comes from, which the schema
    pins to the binding's own ``element``.
    """
    return [
        ElementBinding(
            element_name=slice_.element,
            attribute=binding.attribute,
            index=binding.index,
            weight=slice_.weight,
        )
        for slice_ in binding.slices
    ]


def _setpoint_variable(
    binding: Binding,
    deck_energy_gev: float,
    channel: dict,
    **scalar_kwargs: object,
) -> PyATWritableScalarVariable:
    """Build one hardware setpoint: a strength, a kick or the rf frequency.

    Args:
        binding: the document's binding for this address.
        deck_energy_gev: the document's ``energy_gev`` -- the energy the
            calibration was sampled at and the lattice file is built for.
        channel: the manifest channel the catalog derived its fields from.
            Read for nothing: everything this variable is bound to comes from
            ``binding``, and the address the two agree on is the catalog's key.
        **scalar_kwargs: the ``ScalarVariable`` fields the catalog derived.

    Raises:
        pydantic.ValidationError: the slices do not follow the kind's
            convention -- a strength or frequency not replicated to every
            slice in full, a kick not shared equally between them.
    """
    return _SETPOINT_CLASS[binding.kind](
        bindings=_element_bindings(binding),
        calibration=binding.calibration,
        monitor_inverse=binding.monitor_inverse,
        energy_scaling=binding.energy_scaling,
        deck_energy_gev=deck_energy_gev,
        **scalar_kwargs,
    )


def _monitor_variable(
    binding: Binding,
    channel: dict,
    **scalar_kwargs: object,
) -> MonitorVariable:
    """Build one orbit reading, on the axis and monitor the binding names.

    Args:
        binding: the document's binding for this address.
        channel: as for :func:`_setpoint_variable`, read for nothing.
        **scalar_kwargs: the ``ScalarVariable`` fields the catalog derived.
    """
    return MonitorVariable(
        element_name=binding.element,
        axis=binding.attribute,
        monitor_inverse=binding.monitor_inverse,
        **scalar_kwargs,
    )


def _energy_variable(
    binding: Binding,
    deck_energy_gev: float,
    channel: dict,
    **scalar_kwargs: object,
) -> EnergyVariable:
    """Build the ring's energy knob from the bend's own energy table.

    Args:
        binding: the document's energy binding -- the one it may carry.
        deck_energy_gev: the energy the lattice file is built at, which the
            knob holds the ring at while the bend sits at its nominal.
        channel: as for :func:`_setpoint_variable`, read for nothing.
        **scalar_kwargs: the ``ScalarVariable`` fields the catalog derived.

    Raises:
        pydantic.ValidationError: the table maps the nominal setpoint to
            zero, or the deck energy is not a positive number of GeV -- both
            leave every energy the knob writes undefined.
    """
    return EnergyVariable(
        energy_table=binding.energy_table,
        nominal=binding.nominal,
        deck_energy_gev=deck_energy_gev,
        **scalar_kwargs,
    )


def _factory(binding: Binding, deck_energy_gev: float) -> VariableFactory:
    """Choose the factory one binding's kind is built by."""
    if binding.kind == "monitor":
        return partial(_monitor_variable, binding)
    if binding.kind == "energy":
        return partial(_energy_variable, binding, deck_energy_gev)
    return partial(_setpoint_variable, binding, deck_energy_gev)


def build_action_variables(document: BindingsDocument) -> dict[str, VariableFactory]:
    """One variable factory per bound address, for the catalog to call.

    The mapping :func:`~osprey.services.virtual_accelerator.model.catalog.build_variable_catalog`
    takes as ``action_variables``: every binding of the document becomes the
    factory for the address it claims, and each factory is called as
    ``factory(channel, **scalar_kwargs)`` with the whole manifest channel and
    the catalog's derived fields.

    The key is the binding's ``setpoint_address`` throughout, which is the
    address written for a writable kind and the address published for a
    monitor -- a read-only binding has no setpoint, so the schema puts its own
    address there and leaves ``readback_address`` null. The document refuses
    two bindings claiming one address, so the mapping is one factory per
    binding with nothing collapsed.

    Args:
        document: the served ``va_bindings.json``, already parsed. A caller
            holding a ring built by
            :func:`~osprey.services.virtual_accelerator.lattice.ring.build_ring`
            has read the same file: that function loads the document itself,
            so the two reads are of one tree and the digest check it performs
            covers this one too.

    Returns:
        Factory per address, in document order. The order is the order the
        energy knob adopts its rescaled variables in.
    """
    return {
        binding.setpoint_address: _factory(binding, document.energy_gev)
        for binding in document.bindings
    }


def couple_energy_knob(catalog: Mapping[str, ScalarVariable]) -> tuple[str, ...]:
    """Adopt the catalog's rigidity-scaled setpoints into its energy knob.

    **Call this on the finished catalog, before the model adopts it.** An
    energy write moves the ring energy and rescales every field whose
    ``energy_scaling`` is ``brho``, so that a magnet left at a fixed current
    ends where its control system would put it. The knob learns which
    variables those are here rather than at construction, because it is one
    binding among a facility's thousands and nothing can hand its constructor
    a catalog that is still being built. Skipping the call leaves a knob that
    writes electron-volts and rescales nothing -- and leaves the fields a
    rescale would have touched outside the model's rollback, since the knob
    reports its adopted variables' snapshot targets as its own.

    Args:
        catalog: the built catalog, as ``build_variable_catalog`` returns it.
            The knob and the variables it rescales are both found in here, so
            a caller does not have to know which families move with the
            energy -- or whether this facility has an energy knob at all.

    Returns:
        The names adopted, in catalog order; empty when the catalog carries no
        energy knob, which is a facility whose bends are not a control
        channel rather than a mistake.
    """
    knob = next((entry for entry in catalog.values() if isinstance(entry, EnergyVariable)), None)
    if knob is None:
        return ()
    # One knob per document: the schema refuses a second energy binding, so
    # the first is the only one.
    return knob.couple(
        entry for entry in catalog.values() if isinstance(entry, PyATWritableScalarVariable)
    )


__all__ = ["build_action_variables", "couple_energy_knob"]
