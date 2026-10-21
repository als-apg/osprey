"""The faults of a simulated channel itself, applied by the composite's channel layer.

A scenario's ``channel_faults`` maps an address to one word of
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

The module imports only the standard library, so the build's validator and
the composite share one table.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "CHANNEL_FAULTS",
    "DISCONNECTED",
    "FROZEN",
    "STUCK",
    "ChannelFault",
]

#: Holds a setpoint's writes.
STUCK = "stuck"

#: Holds a reading at the value it served when the fault became active.
FROZEN = "frozen"

#: Reads not-a-number, or the activation value, with the ``udf`` condition.
DISCONNECTED = "disconnected"

_SETPOINT = "setpoint"
_READBACK = "readback"
_NONE = "none"
_FLOAT = "float"
_UNDEFINED = "udf"


def _snapshot(value_type: str, snapshot: Any) -> Any:
    del value_type
    return snapshot


def _not_a_number(value_type: str, snapshot: Any) -> Any:
    return math.nan if value_type == _FLOAT else snapshot


@dataclass(frozen=True)
class ChannelFault:
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

    roles: frozenset[str]
    write: bool
    read: Callable[[str, Any], Any] | None
    severity: str | None


#: Every channel fault, by its word.
CHANNEL_FAULTS: Mapping[str, ChannelFault] = MappingProxyType(
    {
        STUCK: ChannelFault(roles=frozenset({_SETPOINT}), write=True, read=None, severity=None),
        FROZEN: ChannelFault(
            roles=frozenset({_READBACK, _NONE}), write=False, read=_snapshot, severity=None
        ),
        DISCONNECTED: ChannelFault(
            roles=frozenset({_SETPOINT, _READBACK, _NONE}),
            write=True,
            read=_not_a_number,
            severity=_UNDEFINED,
        ),
    }
)
