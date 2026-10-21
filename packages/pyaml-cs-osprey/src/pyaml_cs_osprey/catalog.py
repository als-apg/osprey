"""Parse ``pyaml-cs-oa`` channel references.

A pyAML configuration names each control-system channel with a short reference string.
The grammar, shared with ``pyaml-cs-oa``, is::

    ADDR[unit]            read-write: one address for both directions
    (ADDR)[unit]          write-only
    (RB, SP)[unit]        read-write: readback address RB, setpoint address SP
    ...@i[unit]           any of the above, addressing element ``i`` of an array channel

The ``[unit]`` suffix is optional; a missing unit is ``''``. Whitespace between parts is
ignored. An indexed reference parses (so the control system can refuse it by name) and
sets :attr:`ChannelReference.index`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pyaml.common.exception import PyAMLException

__all__ = ["ChannelReference", "parse_reference"]

Mode = Literal["w", "rw"]

_ADDR = r"[^\s()\[\],@]+"
_REFERENCE_RE = re.compile(
    rf"""
    (?:
        \(\s*(?P<rb>{_ADDR})\s*,\s*(?P<sp>{_ADDR})\s*\)    # (RB, SP)
      | \(\s*(?P<wo>{_ADDR})\s*\)                          # (ADDR)
      | (?P<bare>{_ADDR})                                  # ADDR
    )
    (?:\s*@\s*(?P<index>\d+))?
    (?:\s*\[\s*(?P<unit>[^\[\]]*?)\s*\])?
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class ChannelReference:
    """One parsed channel reference.

    Attributes:
        text: The reference as written, surrounding whitespace removed; excluded from
            equality, so two spellings of one reference compare equal.
        address: The address this reference writes; a bare reference also reads it.
        readback: The separate readback address of a read-write pair, else ``None``.
        mode: ``"w"`` for a write-only reference, else ``"rw"``.
        unit: The unit suffix, ``''`` when absent.
        index: The array element index, ``None`` for a scalar reference.
    """

    text: str = field(compare=False)
    address: str
    readback: str | None
    mode: Mode
    unit: str
    index: int | None


def parse_reference(text: object) -> ChannelReference:
    """Parse a ``pyaml-cs-oa`` channel reference.

    Args:
        text: The reference string from a pyAML configuration; anything else is refused.

    Returns:
        The parsed reference.

    Raises:
        PyAMLException: The text does not follow the grammar; the message quotes it.
    """
    if not isinstance(text, str):
        raise PyAMLException(f"channel reference must be a string, got {text!r}")
    stripped = text.strip()
    match = _REFERENCE_RE.fullmatch(stripped)
    if match is None:
        raise PyAMLException(
            f"malformed channel reference {text!r}: expected ADDR[unit], (ADDR)[unit] "
            "or (RB, SP)[unit], optionally with @index before the unit"
        )
    unit = match["unit"] or ""
    index = int(match["index"]) if match["index"] is not None else None
    address: str
    readback: str | None = None
    mode: Mode
    if match["sp"] is not None:
        address, readback, mode = match["sp"], match["rb"], "rw"
    elif match["wo"] is not None:
        address, mode = match["wo"], "w"
    else:
        address, mode = match["bare"], "rw"
    return ChannelReference(stripped, address, readback, mode, unit, index)
