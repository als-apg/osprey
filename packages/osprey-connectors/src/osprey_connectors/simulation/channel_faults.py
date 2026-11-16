"""The faults of a simulated channel itself, applied by the composite's channel layer.

A scenario's ``channel_faults`` maps an address to one
:data:`~osprey_connectors.simulation.records.ChannelFaultKind`, each a key of
:data:`CHANNEL_FAULTS`. A fault acts on the channel whether a physics model
wires it or the texture holds it, while its scenario is active:

* ``stuck`` faults a setpoint. A write is accepted and held, and forwarded to
  nothing; the setpoint reads the value last written, its start value before
  any write. Its readback shows the machine where it was: a wired readback
  the model's reading, a texture readback its held value, with no echo.
* ``frozen`` faults a reading (a readback or a ``none`` channel). It reads
  the value it served when the fault became active, with no motion and no
  model update.
* ``disconnected`` faults any channel. A float reading reads not-a-number, a
  reading of any other type the value it served when the fault became
  active, and the reading reports the ``udf`` condition. A disconnected
  setpoint holds its writes as ``stuck`` does and reads its demand; its
  readback is the disconnected reading. A setpoint with no readback of its
  own reads its demand and reports the condition itself.

No fault withholds a reply: a read always answers.

The module imports only the standard library, the package's model base and the
view's records, so the build's validator and the composite share one table.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from osprey_connectors.model import FrozenModel
from osprey_connectors.simulation.records import ChannelFaultKind, ChannelRole

__all__ = [
    "CHANNEL_FAULTS",
    "DISCONNECTED",
    "FROZEN",
    "STUCK",
    "ChannelFault",
]

#: Holds a setpoint's writes.
STUCK: Final = "stuck"

#: Holds a reading at the value it served when the fault became active.
FROZEN: Final = "frozen"

#: Reads not-a-number, or the activation value, with the ``udf`` condition.
DISCONNECTED: Final = "disconnected"

_FLOAT = "float"
_UNDEFINED = "udf"


def _snapshot(value_type: str, snapshot: object) -> object:
    del value_type
    return snapshot


def _not_a_number(value_type: str, snapshot: object) -> object:
    return math.nan if value_type == _FLOAT else snapshot


class ChannelFault(FrozenModel):
    """One channel fault's semantics.

    Attributes:
        roles: The facility channel roles the fault may name.
        write: Whether a write to a faulted setpoint is held and forwarded to
            nothing.
        read: What a faulted reading answers, given its ``value_type`` and the
            value it served when the fault became active; ``None`` when the
            fault changes no reading.
        severity: The condition a faulted reading reports, or ``None``.
    """

    roles: frozenset[ChannelRole]
    write: bool
    read: Callable[[str, object], object] | None
    severity: str | None


#: Every channel fault, by its word: one entry per
#: :data:`~osprey_connectors.simulation.records.ChannelFaultKind`, in its order.
CHANNEL_FAULTS: Mapping[ChannelFaultKind, ChannelFault] = MappingProxyType(
    {
        STUCK: ChannelFault(roles=frozenset({"setpoint"}), write=True, read=None, severity=None),
        FROZEN: ChannelFault(
            roles=frozenset({"readback", "none"}), write=False, read=_snapshot, severity=None
        ),
        DISCONNECTED: ChannelFault(
            roles=frozenset({"setpoint", "readback", "none"}),
            write=True,
            read=_not_a_number,
            severity=_UNDEFINED,
        ),
    }
)
