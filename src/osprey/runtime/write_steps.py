"""The two steps every write path shares: the target pin and the checked write.

``osprey.runtime`` (agent code in a sandbox) and the control session (a
long-lived surface such as a panel backend) reach the machine through different
front doors, but the two decisions that make a write safe are the same and live
here once:

- :func:`pin_verdict` — whether the deployment's control context still names
  the ``(target, generation)`` a writer was bound to. Pure: the caller reads the
  record (each path has its own reader and its own way of swallowing a bad one)
  and decides what an ``unavailable`` answer means for it.
- :func:`checked_write` — the local limits net, the attempt observer and the
  connector's denial contract, in that order. A refusal by the net is raised
  before the attempt is observed, so no ledger ever counts a write that could
  not have been sent. The connector call is always
  :meth:`~osprey_connectors.control_system.ControlSystemConnector.write_channel_checked`:
  one hardware door, whichever front door a write came in by.

Neither function owns a timeout of its own: the connector's call and step-read
bounds come from its configuration, and ``timeout`` is passed straight through
to the connector call.

This module is a leaf: it imports the connector layer lazily, so both
``osprey.runtime`` and ``osprey.runtime.session`` can import it at module load.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey_connectors.control_system import (
        ChannelWriteResult,
        ControlSystemConnector,
    )

__all__ = ["PinVerdict", "checked_write", "pin_verdict"]


class PinVerdict(StrEnum):
    """What the deployment's control context says about a writer's binding."""

    #: The record names the bound ``(target, generation)``.
    MATCH = "match"
    #: The record names another target or another generation — or the writer
    #: was bound to no record at all and one now exists.
    MOVED = "moved"
    #: There is no record to compare against: missing, unreadable or corrupt.
    UNAVAILABLE = "unavailable"


def pin_verdict(record: Any, key: tuple[str, int | None] | None) -> PinVerdict:
    """Compare the control context ``record`` against a writer's bound ``key``.

    Args:
        record: The deployment's control context as just read (anything with
            ``target`` and ``generation``), or ``None`` when it could not be read.
        key: The ``(target, generation)`` the writer was bound to, or ``None``
            for a writer that was bound while no record existed. A generation of
            ``None`` (an unparseable stamp) never matches a record.

    Returns:
        :attr:`PinVerdict.UNAVAILABLE` when ``record`` is ``None`` (whatever the
        key), :attr:`PinVerdict.MATCH` when the record's ``(target, generation)``
        equals ``key``, else :attr:`PinVerdict.MOVED`.

    The verdict is the comparison only. What ``unavailable`` means is the
    caller's: a write refuses on it, while a session opened without a record
    has not moved when there is still none.
    """
    if record is None:
        return PinVerdict.UNAVAILABLE
    if key is None or key[1] is None:
        return PinVerdict.MOVED
    if (record.target, record.generation) == tuple(key):
        return PinVerdict.MATCH
    return PinVerdict.MOVED


async def checked_write(
    connector: ControlSystemConnector,
    address: str,
    value: Any,
    *,
    net: Callable[..., Any] | None,
    on_attempt: Callable[[str], Any] | None,
    confirm: bool | None = None,
    timeout: float | None = None,
) -> ChannelWriteResult:
    """Write ``value`` to ``address`` through ``connector`` under the denial contract.

    Steps, in order:

    1. ``net(address, value, read_current=connector.current_value_reader())``
       when ``net`` is given — a local limits check that raises
       :class:`~osprey.errors.ChannelLimitsViolationError` as it is, before
       anything is observed or sent.
    2. ``on_attempt(address)`` when given — immediately before the connector
       call, so an attempt ledger records exactly the writes that may reach the
       control system.
    3. The connector's :meth:`write_channel_checked`. A connector-level limits
       refusal becomes ``ChannelWriteBlockedError(reason='LIMITS')`` chained
       from the violation (``__cause__``), and the result is held to
       :func:`raise_for_write_result`: ``refused`` raises
       ``ChannelWriteBlockedError``; ``failed``, ``mismatch`` and
       ``unconfirmed`` raise ``ChannelWriteFailedError``; transport errors
       propagate.

    ``confirm`` and ``timeout`` reach the connector call only when given, so an
    omitted one leaves the channel's own default in force.

    Returns:
        The connector's result, whose outcome is ``confirmed`` or ``unrequested``.
    """
    if net is not None:
        net(address, value, read_current=connector.current_value_reader())

    call_kwargs: dict[str, Any] = {}
    if confirm is not None:
        call_kwargs["confirm"] = confirm
    if timeout is not None:
        call_kwargs["timeout"] = timeout

    if on_attempt is not None:
        on_attempt(address)

    return await connector.write_channel_checked(address, value, **call_kwargs)
