"""The process's pyAML run: where a pyAML write is accepted and recorded.

A pyAML write is an ordinary checked connector write
(``osprey.runtime.write_channel`` / ``write_channels``), accepted only while a
:func:`pyaml_cs_osprey.run_tool.run_tool` call is in progress in this process;
anywhere else it is refused as :data:`OUTSIDE_A_RUN` and nothing is read or
written. The run records, in memory for the call's result, each address's value
before the first write the runtime sent or landed; a refused write records
nothing. Nothing is written back.

The run belongs to the process, not to the thread that opened it: a
``run_tool`` on any thread joins the one run, and the last one to leave ends
it. Its abort flag is a :class:`threading.Event`, so a ``SIGINT`` handler sets
it without taking :data:`_GUARD`, and every thread reads it.

A forked child starts with no run.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import osprey.runtime
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from pyaml_cs_osprey.errors import OspreyWriteRefused

#: Why a device write outside a ``run_tool`` call is refused.
OUTSIDE_A_RUN = "pyAML writes run only inside pyaml_measure"

#: Held across one write's read, write and record, and over every change of
#: :data:`_RUN`. Nothing taken under it takes it again.
_GUARD = threading.Lock()


@dataclass
class _PyamlRun:
    """The ``run_tool`` calls in progress in this process.

    Attributes:
        target: The control target whose run lock the run holds.
        moved: Each address's value before the first write the runtime sent
            or landed, in the order those writes went out. A refused write
            records nothing.
        abort: A Ctrl-C asked every tool of the run to stop.
        depth: How many ``run_tool`` calls are inside the run.
    """

    target: str
    moved: dict[str, Any] = field(default_factory=dict)
    abort: threading.Event = field(default_factory=threading.Event)
    depth: int = 0


#: The process's run, ``None`` when no ``run_tool`` call is in progress.
_RUN: _PyamlRun | None = None


@contextlib.contextmanager
def opened(target: str) -> Iterator[None]:
    """Be inside the process's run on *target* for the body, opening it when none is.

    Raises:
        RuntimeError: The process's run is on another target.
    """
    global _RUN
    with _GUARD:
        run = _RUN
        if run is None:
            run = _RUN = _PyamlRun(target=target)
        elif run.target != target:
            raise RuntimeError(
                f"a pyAML run on {run.target} is in progress; it cannot open one on {target}"
            )
        run.depth += 1
    try:
        yield
    finally:
        with _GUARD:
            run.depth -= 1
            if run.depth == 0 and _RUN is run:
                _RUN = None


def request_abort() -> bool:
    """Ask every tool of the process's run to stop; whether a run is open."""
    run = _RUN
    if run is None:
        return False
    run.abort.set()
    return True


def abort_requested() -> bool:
    """Whether a Ctrl-C asked the process's run to stop."""
    run = _RUN
    return run is not None and run.abort.is_set()


def moved() -> dict[str, Any]:
    """A copy of what the process's run moved.

    Each address whose write the runtime sent or landed, with its value before
    the first such write.
    """
    with _GUARD:
        run = _RUN
        return {} if run is None else dict(run.moved)


def write(
    addresses: Sequence[str],
    write: Callable[..., object],
    map_write: Callable[[Exception], Exception],
    *,
    map_read: Callable[[Exception, list[str]], Exception],
    confirm: bool | None,
) -> None:
    """Write ``addresses`` inside the process's run only, recording what moved.

    The addresses not yet recorded are read in one ``read_channels`` call
    before the write. Each of them is recorded with that value once the runtime
    sent or landed its write, whether or not the call then raised; a refused
    write records nothing. An escape that is not an ``Exception`` records every
    one of them, since the write may still go out, and propagates unchanged. An
    address already recorded keeps its first value.

    Args:
        addresses: Every address the write touches.
        write: The runtime write. It is called with ``confirm=confirm`` when
            ``confirm`` is given and with no keyword otherwise, so each channel
            resolves its own confirm default.
        map_write: Maps an exception ``write`` raised onto the exception raised
            in its place.
        map_read: Maps an exception the pre-write read raised, with the
            addresses asked for, onto the exception raised in its place.
        confirm: Passed to ``write`` when not ``None``.

    Raises:
        OspreyWriteRefused: No ``run_tool`` call is in progress
            (:data:`OUTSIDE_A_RUN`, naming the first address); nothing was read
            or written.
        Exception: ``map_read``'s mapping of a read failure, with nothing
            written, or ``map_write``'s mapping of a write failure.
    """
    with _GUARD:
        run = _RUN
        if run is None:
            raise OspreyWriteRefused(OUTSIDE_A_RUN, addresses[0] if addresses else None)
        missing = [address for address in dict.fromkeys(addresses) if address not in run.moved]
        pending: dict[str, Any] = {}
        if missing:
            try:
                pending = dict(zip(missing, osprey.runtime.read_channels(missing), strict=True))
            except Exception as exc:
                raise map_read(exc, list(addresses)) from exc
        raised: BaseException | None = None
        with osprey.runtime.record_writes() as records:
            try:
                write(**({} if confirm is None else {"confirm": confirm}))
            except Exception as exc:
                raised = exc
                raise map_write(exc) from exc
            except BaseException as exc:
                raised = exc
                raise
            finally:
                run.moved.update(
                    (address, value)
                    for address, value in pending.items()
                    if _moved(address, records, raised)
                )


def _moved(address: str, records: Sequence[tuple[str, str]], raised: BaseException | None) -> bool:
    """Whether ``address``'s write may have put its value on the control system.

    An escape that is not an ``Exception``, such as a ``KeyboardInterrupt``
    while the write keeps running elsewhere, lists every address, failing
    closed: the write may still go out. Otherwise the runtime's phases decide:
    ``sent`` or ``landed`` moved it, and no ``attempt`` means it never reached
    a connector. An ``attempt`` with no later phase is listed, failing closed,
    unless the call raised a refusal of that address: a
    ``ChannelWriteBlockedError``, which the runtime raises only after every
    channel of the call has reported its outcome, so a channel with no outcome
    was refused; or a ``ChannelLimitsViolationError`` naming the address, which
    a connector raises before that channel is sent. Any other exception, such
    as a transport error partway through a batch, leaves unknown whether the
    value went out.
    """
    if raised is not None and not isinstance(raised, Exception):
        return True
    phases = {phase for recorded, phase in records if recorded == address}
    if phases & {"sent", "landed"}:
        return True
    if "attempt" not in phases:
        return False
    if isinstance(raised, ChannelWriteBlockedError):
        return False
    return not (
        isinstance(raised, ChannelLimitsViolationError) and raised.channel_address == address
    )


def _reset() -> None:
    """Give a forked child no run and a guard of its own."""
    global _GUARD, _RUN
    _GUARD = threading.Lock()
    _RUN = None


os.register_at_fork(after_in_child=_reset)
