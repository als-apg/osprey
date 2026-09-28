"""A minimal Channel Access client over pvapy for the live VA suites.

The virtual accelerator's live tests assert what a client sees from the far
side of a real wire. The client used to be pyepics; osprey's EPICS stack is now
pvapy (``pvaccess.Channel(name, pvaccess.CA)``), so the tests drive the served
records through the same library production does.

Three properties the callers rely on:

* Every read is a fresh ``get`` over the wire -- there is no monitor cache to
  hand back a value that arrived before the last write.
* :func:`caput` waits for the server's put-callback
  (``record[block=true]``), the pvapy spelling of pyepics' ``wait=True``: it
  returns only once the server has completed the write. It is asynchronous
  underneath so it never holds the GIL against an in-process server; see it.
* Call it from the main thread (or a daemon thread that never exits). On macOS
  a joined non-main thread that touched pvapy hangs on exit; Linux, where these
  suites actually run, is unaffected, but there is no reason to rely on that.

``pvaccess`` must be imported before ``pcaspy`` in any process that hosts both:
pcaspy's extension carries its own statically linked EPICS client symbols, and
the client has to be bound to pvapy's copy. :func:`initialize` does that.
"""

from __future__ import annotations

import threading
import time
from typing import Any

#: The pvRequest of a confirming put: return only when the put-callback fires.
BLOCKING_PUT_REQUEST = "record[block=true]field(value)"

_channels: dict[str, Any] = {}


def initialize() -> Any:
    """Import pvapy (before any pcaspy import) and return the module."""
    import pvaccess

    return pvaccess


def channel(address: str, timeout: float) -> Any:
    """A cached CA channel for ``address`` whose connect/get is bounded by ``timeout``."""
    pvaccess = initialize()
    ch = _channels.get(address)
    if ch is None:
        ch = pvaccess.Channel(address, pvaccess.CA)
        _channels[address] = ch
    ch.setTimeout(timeout)
    return ch


def get_fields(address: str, request: str, timeout: float) -> dict[str, Any] | None:
    """One wire read of ``request`` as a dict, or ``None`` if it failed."""
    pvaccess = initialize()
    try:
        return channel(address, timeout).get(request).toDict()
    except pvaccess.PvaException:
        return None


def caget(address: str, timeout: float) -> Any:
    """The value on the wire now, or ``None`` if the channel did not answer.

    An enum is reported as its index, the way pyepics' ``caget`` did.
    """
    fields = get_fields(address, "field(value)", timeout)
    if fields is None:
        return None
    value = fields.get("value")
    if isinstance(value, dict) and "index" in value:
        return value["index"]
    return value


def caput(address: str, value: float, timeout: float) -> int | None:
    """Write with put-completion; ``1`` once the server completed it, else ``None``.

    The return mirrors pyepics' ``caput(..., wait=True)`` so assertions of the
    form ``caput(...) == 1`` keep their meaning.

    ``asyncPut`` rather than ``put``, and that is load-bearing: pvapy's
    synchronous ``put`` holds the GIL for the whole round trip, and the pcaspy
    server these suites share a process with is serviced from a Python thread.
    A synchronous put therefore starves the server that has to answer it --
    the circuit goes unresponsive and the put fails after ~35 s (measured in
    the linux venue). ``asyncPut`` returns at once, and waiting on an event
    here releases the GIL, so the server keeps running. ``record[block=true]``
    makes the callback fire only when the server has completed the write
    (measured: a write the driver completes 1 s later calls back after 1 s).
    """
    pvaccess = initialize()
    done = threading.Event()
    outcome: list[bool] = []

    def settled(ok: bool) -> Any:
        def callback(*_: Any) -> None:
            outcome.append(ok)
            done.set()

        return callback

    payload = pvaccess.PvObject({"value": pvaccess.DOUBLE}, {"value": float(value)})
    try:
        target = channel(address, timeout)
        target.get("field(value)")  # connect first: an unconnected asyncPut just fails
        target.asyncPut(payload, settled(True), settled(False), BLOCKING_PUT_REQUEST)
    except pvaccess.PvaException:
        return None
    if not done.wait(timeout) or not outcome[0]:
        return None
    return 1


def severity(address: str, timeout: float) -> int | None:
    """The alarm severity on the wire now, or ``None`` if unread."""
    fields = get_fields(address, "field(value,alarm)", timeout)
    if fields is None:
        return None
    return fields.get("alarm", {}).get("severity")


def control_limits(address: str, timeout: float) -> tuple[float, float] | None:
    """The ``(low, high)`` control limits of ``address``, or ``None`` if unread."""
    fields = get_fields(address, "field(value,control)", timeout)
    if fields is None:
        return None
    control = fields.get("control", {})
    return control.get("limitLow"), control.get("limitHigh")


class Monitor:
    """A CA subscription that records every value it is sent, as a context manager.

    pvapy delivers the current value as the first event, where a pyepics
    callback added after connecting did not; callers that assert "nothing
    moved" filter by value, so that first event is harmless.
    """

    def __init__(self, address: str, timeout: float) -> None:
        self.address = address
        self.values: list[Any] = []
        self._timeout = timeout
        self._channel = channel(address, timeout)
        self._name = f"va-test-monitor-{id(self)}"

    def _on_update(self, pv: Any) -> None:
        self.values.append(pv["value"])

    def __enter__(self) -> Monitor:
        self._channel.subscribe(self._name, self._on_update)
        self._channel.startMonitor("field(value)")
        # The subscription is live once its initial event has arrived; a write
        # made before that could be folded into the initial value and never
        # reach the subscriber as an event of its own.
        deadline = time.monotonic() + self._timeout
        while not self.values and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self.values:
            self.__exit__()
            raise TimeoutError(f"monitor on {self.address} never delivered its initial value")
        return self

    def __exit__(self, *exc: object) -> None:
        # Stop before unsubscribing so no update lands on a removed subscriber.
        self._channel.stopMonitor()
        self._channel.unsubscribe(self._name)


def served_probe_source(address: str, timeout: float = 1.0) -> str:
    """Python source for a child process that reports whether ``address`` answers.

    The child prints ``SERVED`` or ``NONE``, flushes, and leaves through
    ``os._exit`` so EPICS teardown at interpreter exit can never hold it up. It
    makes its one read on its main thread, which is safe on macOS too.
    """
    return (
        "import os, sys, pvaccess\n"
        f"ch = pvaccess.Channel({address!r}, pvaccess.CA)\n"
        f"ch.setTimeout({float(timeout)!r})\n"
        "try:\n"
        "    ch.get('field(value)')\n"
        "    word = 'SERVED'\n"
        "except pvaccess.PvaException:\n"
        "    word = 'NONE'\n"
        "sys.stdout.write(word)\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
