"""Pair each settable channel with the readback that reports it.

A plan that drives a setpoint wants to observe the value the machine actually
took, not the value it was asked to take. Where the facility publishes a
readback beside the setpoint, that is the channel to read; where it does not,
the worker reads the setpoint back and reports what it can.

The rule is one line of address grammar -- the final colon-separated token
``SP`` becomes ``RB`` -- and one membership test: the candidate is adopted only
when the roster already holds that address *with read direction*. The grammar
alone would be a guess; the roster is the authority on what exists, so a
setpoint whose sibling nobody enumerated stays unpaired rather than pointing a
plan at an address that is not there. That test also declines a ``:RB`` sibling
the source called settable, which is a corpus that has drifted rather than a
readback.

Address grammar lives in exactly two places in this package: here, and
:mod:`osprey.channel_roster.database`'s direction fallback. Membership never
comes from grammar anywhere -- the readers enumerate it, and this module only
consults what they returned. Hence the input is a plain sequence of records:
pairing is one heuristic applied identically to both sources, and importing
either reader here would tie it to one of them.
"""

from __future__ import annotations

from collections.abc import Sequence

from .records import ChannelRecord

#: Final address token that marks a setpoint -- the same token
#: :mod:`osprey.channel_roster.database` derives direction from.
WRITE_SUBFIELD = "SP"

#: Final address token that marks the readback of a setpoint.
READBACK_SUBFIELD = "RB"

#: What separates an address into its tokens.
ADDRESS_SEPARATOR = ":"


def assign_readbacks(records: Sequence[ChannelRecord]) -> tuple[ChannelRecord, ...]:
    """Give every settable channel its readback, where the roster has one.

    Args:
        records: The roster's records, from either reader, in source order.

    Returns:
        The same records in the same order, with each write-direction record
        whose ``:SP`` sibling ``:RB`` is enumerated as a read channel replaced
        by a copy carrying that address as its
        :attr:`~osprey.channel_roster.records.ChannelRecord.readback`. Every
        other record -- read channels, directionless channels, setpoints with
        no enumerated readback, addresses the grammar does not read as a
        setpoint -- is returned unchanged, its readback left ``None`` so the
        worker reads the setpoint itself.
    """
    readable = {record.address for record in records if record.direction == "read"}
    return tuple(_paired(record, readable) for record in records)


def _paired(record: ChannelRecord, readable: frozenset[str] | set[str]) -> ChannelRecord:
    """Return ``record`` with its readback, or unchanged when it has none."""
    if record.direction != "write":
        return record
    candidate = _readback_address(record.address)
    if candidate is None or candidate not in readable:
        return record
    return record.with_readback(candidate)


def _readback_address(address: str) -> str | None:
    """Return the readback address the grammar names for ``address``.

    ``None`` when the grammar names none: an address carrying no separator has
    no final token to replace, and one whose final token is not
    :data:`WRITE_SUBFIELD` is not a setpoint.
    """
    prefix, separator, subfield = address.rpartition(ADDRESS_SEPARATOR)
    if not separator or subfield != WRITE_SUBFIELD:
        return None
    return prefix + separator + READBACK_SUBFIELD
