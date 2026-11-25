"""What a plan device writes, read from the device's own statement.

A plan device writes to the machine only from ``set()`` and ``trigger()``, and
states the channels those two make through ``set_writes(*args)`` and
``trigger_writes()``: pure methods, no I/O, each returning a tuple of
:class:`ChannelWrite` naming every channel the matching verb writes, in the
order it writes them. That statement is the one answer to "what does this
launch write", read here for every message a plan yields:

- the read-only preview walk tallies it per channel (:class:`WritesTally`), so
  the approval prompt lists every write a launch makes, not only its moves;
- both worker plan wrappers run the plan through :func:`held_to_declaration`,
  so a launch whose plan does not declare ``writes: true`` is refused at its
  first write, before that write reaches the RunEngine. The preview walks the
  same wrapper, so the preview and the run refuse in the same words;
- the run's write authorization (``write_grant``) surveys the same statements
  to ask for its scope, and checks each write against them as the run goes.

A device that can stop a move also states what its ``stop()`` writes, through
``stop_writes()``, read by :func:`message_may_writes`: what may be written if
the run stops mid-move. It is not a write the launch makes and is never in
:func:`message_writes`.

A device that has a writing verb but no matching statement refuses the launch
(:class:`UnstatedWriteError`): a write nobody stated is a write nobody can
approve.

Imports nothing from ophyd or bluesky: the bridge process imports this module
for its vocabulary, and the worker reads messages by attribute.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Iterable
from typing import Any, NamedTuple


class ChannelWrite(NamedTuple):
    """One write a device verb makes: the channel address and the value sent."""

    address: str
    value: Any


class UnstatedWriteError(RuntimeError):
    """A device that can write does not state the channels it writes."""


class UndeclaredWriteError(RuntimeError):
    """A launch writes while its plan does not declare ``writes: true``."""


_STATEMENTS = {"set": "set_writes", "trigger": "trigger_writes"}


def _device_name(obj: Any) -> str:
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name
    return repr(obj)


def message_writes(msg: Any) -> tuple[ChannelWrite, ...]:
    """The channel writes one plan message makes, as its device states them.

    A ``set`` on an object with a ``set`` method is that object's
    ``set_writes(*msg.args)``; a ``trigger`` on an object with a ``trigger``
    method is its ``trigger_writes()``. Any other message writes nothing, and
    so does a verb sent to an object that lacks it: the RunEngine refuses that
    message itself before anything is written.

    Raises:
        UnstatedWriteError: The object has the verb but not its statement, or
            the statement is not a tuple of :class:`ChannelWrite`.
    """
    command = getattr(msg, "command", None)
    if not isinstance(command, str) or command not in _STATEMENTS:
        return ()
    statement_name = _STATEMENTS[command]
    obj = getattr(msg, "obj", None)
    if not callable(getattr(obj, command, None)):
        return ()
    statement = getattr(obj, statement_name, None)
    refusal = UnstatedWriteError(
        f"device {_device_name(obj)!r} does not state the channels its {command} writes; "
        "the launch is refused"
    )
    if not callable(statement):
        raise refusal
    args = tuple(getattr(msg, "args", ()) or ()) if command == "set" else ()
    writes = statement(*args)
    if not isinstance(writes, tuple) or not all(isinstance(w, ChannelWrite) for w in writes):
        raise refusal
    return writes


def message_may_writes(msg: Any) -> tuple[ChannelWrite, ...]:
    """The channel writes a stop of the device one plan message moves may make.

    A ``set`` on an object with a ``stop`` method is that object's
    ``stop_writes()``: what its ``stop()`` writes when it stops a move the
    run started. The RunEngine stops only what a run moved, so any other
    message, and a ``set`` on an object without ``stop``, states none.

    Raises:
        UnstatedWriteError: The object has ``stop`` but not its statement, or
            the statement is not a tuple of :class:`ChannelWrite`.
    """
    if getattr(msg, "command", None) != "set":
        return ()
    obj = getattr(msg, "obj", None)
    if not callable(getattr(obj, "stop", None)):
        return ()
    statement = getattr(obj, "stop_writes", None)
    refusal = UnstatedWriteError(
        f"device {_device_name(obj)!r} does not state the channels its stop writes; "
        "the launch is refused"
    )
    if not callable(statement):
        raise refusal
    writes = statement()
    if not isinstance(writes, tuple) or not all(isinstance(w, ChannelWrite) for w in writes):
        raise refusal
    return writes


def _json_safe(value: Any) -> Any:
    """Reduce one move target to a plain JSON-serializable scalar.

    Targets arrive as whatever the plan computed — commonly a numpy scalar out
    of ``numpy.linspace``, which no JSON encoder accepts. ``.item()`` unwraps
    those to plain Python; anything left that is not already a JSON scalar is
    carried as its ``repr`` so an exotic target degrades to a readable string
    instead of failing the whole response at encode time.
    """
    unwrap = getattr(value, "item", None)
    if callable(unwrap):
        # a non-scalar `.item()`; fall through to repr
        with contextlib.suppress(Exception):
            value = unwrap()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


class _ChannelTally:
    __slots__ = ("count", "device", "equal", "value")

    def __init__(self, device: str, value: Any) -> None:
        self.device = device
        self.value = value
        self.equal = True
        self.count = 0


class WritesTally:
    """Every write a launch makes, one entry per channel.

    Entries are in first-write order. An entry's ``value`` is the JSON-safe
    value every write to that channel carried when they all carried an equal
    one, else ``None``. Memory per channel is constant: the first value, an
    equal flag and a count.
    """

    def __init__(self) -> None:
        self._channels: dict[str, _ChannelTally] = {}
        self._total = 0

    def add(self, device_name: str, writes: Iterable[ChannelWrite]) -> None:
        """Count ``writes``, made by the device named ``device_name``."""
        for write in writes:
            value = _json_safe(write.value)
            entry = self._channels.get(write.address)
            if entry is None:
                entry = self._channels[write.address] = _ChannelTally(device_name, value)
            elif entry.equal and entry.value != value:
                entry.equal = False
            entry.count += 1
            self._total += 1

    def entries(self) -> list[dict[str, Any]]:
        """``{"channel", "device", "value", "count"}`` per channel, first-write order."""
        return [
            {
                "channel": address,
                "device": entry.device,
                "value": entry.value if entry.equal else None,
                "count": entry.count,
            }
            for address, entry in self._channels.items()
        ]

    def total(self) -> int:
        """How many writes were counted, across every channel."""
        return self._total


def _close_quietly(plan: Any) -> None:
    close = getattr(plan, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def held_to_declaration(
    plan_name: str, declares_writes: bool | None, plan: Any
) -> Generator[Any, Any, Any]:
    """Forward ``plan`` as ``yield from plan`` would, refusing what it may not write.

    Every message is read through :func:`message_writes` before it is yielded,
    so a refusal lands before the write it names reaches the RunEngine. With
    ``declares_writes`` False, the first message that writes closes ``plan``
    and raises :class:`UndeclaredWriteError`; with ``None`` (a plan with no
    authoring metadata) only the statement itself is checked. Both refusals
    are raised out of this generator and never thrown into ``plan``, so a plan
    body cannot catch the refusal and carry on.

    Replies, thrown exceptions and ``close`` are forwarded exactly as
    ``yield from`` forwards them, and the plan's return value is this
    generator's.
    """
    iterator = iter(plan)

    def checked(message: Any) -> Any:
        try:
            writes = message_writes(message)
        except UnstatedWriteError:
            _close_quietly(iterator)
            raise
        if writes and declares_writes is False:
            _close_quietly(iterator)
            raise UndeclaredWriteError(
                f"plan {plan_name!r} does not declare writes: true and this launch writes "
                f"{writes[0].address}; state writes: true in PLAN_METADATA to run it"
            )
        return message

    try:
        message = next(iterator)
    except StopIteration as stop:
        return stop.value
    while True:
        outgoing = checked(message)
        try:
            reply = yield outgoing
        except GeneratorExit:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            raise
        except BaseException as exc:
            throw = getattr(iterator, "throw", None)
            if throw is None:
                raise
            try:
                message = throw(exc)
            except StopIteration as stop:
                return stop.value
        else:
            try:
                message = next(iterator) if reply is None else iterator.send(reply)
            except StopIteration as stop:
                return stop.value
