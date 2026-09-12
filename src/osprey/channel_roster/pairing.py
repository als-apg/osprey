"""Pair each settable channel with the readback that reports it.

A plan that drives a setpoint wants to observe the value the machine actually
took, not the value it was asked to take. Where the facility publishes a
readback beside the setpoint, that is the channel to read; where it does not,
the worker reads the setpoint back and reports what it can.

The rule is one naming convention plus one membership test. The convention is
the bundled demo tree's, and nobody else's: a final token of
:data:`~osprey.channel_roster.records.WRITE_SUBFIELD` becomes
:data:`~osprey.channel_roster.records.READBACK_SUBFIELD`. The membership test
is the authority: the candidate is adopted only when the roster already holds
that address *with read direction*. A facility that names its readbacks some
other way is not paired by the convention and loses nothing it had -- the
convention alone would be a guess anywhere, so a setpoint whose sibling nobody
enumerated stays unpaired rather than pointing a plan at an address that is not
there. That test also declines a readback sibling the source called settable,
which is a corpus that has drifted rather than a readback.

The tokens the convention is spelled in have one producer,
:mod:`osprey.channel_roster.records`, shared with
:mod:`osprey.channel_roster.database`'s direction fallback. Membership never
comes from a naming convention anywhere -- the readers enumerate it, and this
module only consults what they returned. Hence the input is a plain sequence of
records: pairing is one heuristic applied identically to both sources, and
importing either reader here would tie it to one of them.
"""

from __future__ import annotations

from collections.abc import Sequence

from .records import ADDRESS_SEPARATOR, READBACK_SUBFIELD, WRITE_SUBFIELD, ChannelRecord


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
    """Return ``record`` with its readback, or unchanged when it has none.

    A readback the source itself stated (the graph reader's device grouping,
    :func:`osprey.channel_roster.graph._corpus_readbacks`) is kept as it is:
    the corpus is the authority on what reports a setpoint, and the grammar
    here is only for the records it paired nothing with.
    """
    if record.direction != "write" or record.readback is not None:
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
