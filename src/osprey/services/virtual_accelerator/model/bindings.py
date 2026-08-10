"""Bind the model's variable catalog to ALS-U AR lattice elements.

:func:`~osprey.services.virtual_accelerator.model.catalog.build_variable_catalog`
derives what every model variable *is* -- its address, nominal, unit,
declared range and validation config -- from the manifest and the facility
databases. What it cannot know is what each variable is *bound to*, because
that is a fact about this facility's naming: the ring element a channel
drives is its ``{family}{device}`` flat name, the attribute a write lands in
is fixed per family, and a BPM channel's ``:X``/``:Y`` subfield names a
transverse axis. Those three derivations live here, and only here.

They reach the catalog through its ``input_factory``/``output_factory``
seam, so this module adds a binding to each variable rather than rebuilding
the catalog's fields alongside it. That is what makes parity structural:
there is one derivation path for a variable's address, nominal, unit and
range, and a bound catalog can differ from a plain one only in the binding
itself. A second construction loop here would be a second thing to keep in
step with ``machine.json``.

The flat-name concatenation is a facility convention, not a lume-pyat one:
``lume_pyat`` treats ``element_name`` as an opaque ``FamName`` and parses
nothing out of it. Every element name this module hands out has to address
exactly one element of the ring -- ``PyATSimulator`` rejects an ambiguous
binding at construction -- which the ALS-U AR ring satisfies because each
addressable device carries its own unique flat name.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from lume_pyat.actions import PyATReadOnlyScalarVariable

from .catalog import build_variable_catalog
from .variables import DECLARED_ATTRIBUTE, CurrentSetpointVariable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.variables import ScalarVariable

    from osprey.services.virtual_accelerator.lattice.strengths import StrengthMap

# The transverse axis each BPM position subfield reads.
_AXIS_BY_SUBFIELD = {"X": "x", "Y": "y"}


def _setpoint_variable(
    channel: dict,
    *,
    strength_map: StrengthMap,
    **scalar_kwargs: Any,
) -> CurrentSetpointVariable:
    """Build one magnet/corrector setpoint bound to its ring element."""
    family = channel["family"]
    device_id = channel["device"]
    return CurrentSetpointVariable(
        strength_map=strength_map,
        family=family,
        device_id=device_id,
        element_name=f"{family}{device_id}",
        # `.get`, not `[]`: a family with no strength formula is the
        # variable's own error to name (it raises for exactly that, as a
        # ValidationError), and a KeyError from here would pre-empt it.
        attribute=DECLARED_ATTRIBUTE.get(family),
        **scalar_kwargs,
    )


def _monitor_variable(channel: dict, **scalar_kwargs: Any) -> PyATReadOnlyScalarVariable:
    """Build one BPM position reading bound to its monitor and axis."""
    return PyATReadOnlyScalarVariable(
        element_name=f"{channel['family']}{channel['device']}",
        # Same reasoning as the declared attribute above: an unmapped
        # subfield surfaces as the variable rejecting the axis.
        axis=_AXIS_BY_SUBFIELD.get(channel["subfield"]),
        **scalar_kwargs,
    )


def build_action_variables(
    channels: list[dict] | None = None,
    *,
    strength_map: StrengthMap,
) -> dict[str, ScalarVariable]:
    """Build the catalog as lattice-bound action variables, keyed by address.

    Same keys, same declared fields and same counts as
    :func:`~osprey.services.virtual_accelerator.model.catalog.build_variable_catalog`
    -- it *is* that function, called with factories that bind each variable
    to an element. Setpoints become
    :class:`~osprey.services.virtual_accelerator.model.variables.CurrentSetpointVariable`
    (Amps in, strength onto the lattice); BPM readings become
    ``PyATReadOnlyScalarVariable`` (one transverse coordinate off the last
    solve).

    Args:
        channels: the manifest's ``channels`` list; ``None`` builds the
            manifest. A caller that already has one should pass it -- see
            ``build_variable_catalog``.
        strength_map: the map every setpoint converts its current through.
            Shared by all of them and baked from the ring they will write
            to, so the ring, the map and these variables are one set.

    Raises:
        pydantic.ValidationError: a channel names a family with no strength
            formula, or a subfield that is not a transverse axis -- both
            mistakes in the *definition* of a variable, so they surface here
            rather than at the first write or read.
    """
    return build_variable_catalog(
        channels,
        input_factory=partial(_setpoint_variable, strength_map=strength_map),
        output_factory=_monitor_variable,
    )


__all__ = ["build_action_variables"]
