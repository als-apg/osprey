"""What a client write to a co-hosted setpoint means, without a transport.

Everything that happens between a setpoint write arriving at the server and
the echo the client is owed lives here: the drive-limit clamp, the decision
of what writing a given address *means*, the hand-off to the physics model,
and the values posted on success. None of it imports the Channel Access
server library or the serving package that hosts it, so the whole write
path is exercisable in process against a fake driver --
:mod:`~osprey.services.virtual_accelerator.serving.runner` is the shell
that binds it to pcaspy and to the run loop, and holds nothing this module
does not already decide.

Four properties of that path are contracts, not preferences.

**A rejected write moves nothing, for any reader.** Channel Access
put-completion has no failure channel -- it can only ever report success --
so a refusal is expressed by *withholding* the echo. Withholding means no
``setParam``: the parameter store is what a fresh ``caget`` reads, so a
value recorded there and merely not flushed to monitors would still be
served to a one-shot reader. The write path therefore never records a value
the model has not taken, and the optional alarm is the only trace a refusal
leaves.

**Commit before signalling.** ``callbackPV`` ends the asynchronous write and
unblocks a client waiting on put-completion. Every value that write is
supposed to have produced is committed and posted first, so a client that
unblocks and reads immediately is guaranteed to see it. And completion is
signalled on *every* outcome, including a refusal: the server library
postpones each subsequent write to a PV whose asynchronous write never
completed, so a missing ``callbackPV`` freezes that setpoint permanently.

**The model is never touched here.** A write that needs physics is handed to
the run loop through :meth:`~CohostWritePath.write`'s ``enqueue`` callback
and completed later from the loop's own thread; the model class underneath
is documented as not thread-safe, so the server thread must not reach into
it. Nothing in this module holds a model reference, which is what makes
that structural rather than a rule to remember.

**A channel's two views carry one value.** Some co-hosted addresses are
served twice -- on Channel Access from the manifest-derived database, and on
PVA because the same address is also a model variable. Both transports enter
here (:meth:`~CohostWritePath.write` from Channel Access,
:meth:`~CohostWritePath.put` from PVA), and both leave through the same
clamp, the same physics hand-off and the same publish, so the value on the
two views is one value written once and never two views drifting. What
differs between them is only how the client is told the write finished, and
that difference is real: Channel Access can report nothing but success,
while a PVA put carries an error string back to the client that issued it.
"""

from __future__ import annotations

import logging
import numbers
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from lume.model import LUMEModel

from osprey.services.virtual_accelerator.serving.pvdb import (
    SETPOINT_SUBFIELD,
    ServingRecords,
    discard_pva_post,
)

LOG = logging.getLogger(__name__)

# What writing a setpoint address does.
#
# ``PHYSICS``   the value goes to the model first; the setpoint readback and
#               the paired ``:RB`` echo carry it only once the model accepts.
# ``ECHO``      no physics: the paired ``:RB`` follows the setpoint
#               immediately, a plain value copy.
# ``LATCH``     the setpoint records the written value and nothing else
#               happens -- a stuck-setpoint fault, or a pyat-coupled setpoint
#               in a process with no lattice behind it.
MODE_PHYSICS = "physics"
MODE_ECHO = "echo"
MODE_LATCH = "latch"

#: Runner configuration this write path requires, merged over the generated
#: configuration by the runner. Every key is load-bearing:
#:
#: ``update_rate``
#:     Zero disables the batching window, so each queued write gets a
#:     ``model.set()`` of its own and one client's write is never merged with
#:     another's. Setpoint writes are physically ordered events; a batch
#:     window would apply two of them in one solve and report a settle time
#:     that belongs to neither.
#: ``echo_unconfirmed_writes``
#:     False. The runner-native write path publishes a value only once the
#:     model has taken it (this module does the same for the co-hosted PVs).
#:     True would leave a refused write's value standing on the PV.
#: ``alarm_on_refused_write``
#:     True. Put-completion cannot report failure, so an alarm is the only
#:     signal a refused write can leave for a client. It is self-clearing:
#:     the next accepted write's ``setParam`` recomputes the alarm from the
#:     value, and the served database declares no alarm limits, so that
#:     recomputation always lands on NO_ALARM.
#: ``clamp_writes``
#:     False -- because this module clamps instead, for every transport.
#:     Enforcement of a drive band exists nowhere but in a write path: the
#:     model does not enforce it and neither server rejects an out-of-band
#:     write despite publishing the band as display limits. The runner's own
#:     clamp would enforce the same band (a variable's ``value_range`` and the
#:     manifest's drive limits are the same numbers) on the same PVA puts this
#:     module now handles, which is precisely why it is off: two enforcement
#:     points on one value is a second thing to keep in step, and the one that
#:     survives is the one both transports share. See
#:     :func:`clamp_into`.
#: ``control_pvs``
#:     False. The runner claims no name the facility's channel manifest does
#:     not describe, so no RESET (or SNAPSHOT) PV is served.
RUNNER_CONFIG_POLICY: dict[str, Any] = {
    "update_rate": 0.0,
    "echo_unconfirmed_writes": False,
    "alarm_on_refused_write": True,
    "clamp_writes": False,
    "control_pvs": False,
}

#: Reported to a PVA client that put to an address the co-hosted namespace
#: does not serve as a setpoint. Channel Access expresses the same refusal by
#: returning False from the driver's write, which the server library turns
#: into a rejected put with no message of its own; a PVA put can carry one,
#: so it does.
NOT_WRITABLE = "not a writable setpoint"


class WriteDriver(Protocol):
    """The driver surface this module writes through.

    The live implementation is a Channel Access driver; the shape is small
    enough that the whole write path can be driven against a fake one.
    """

    def setParam(self, reason: str, value: Any) -> None:  # noqa: N802 - driver contract
        """Record ``value`` as the served value of ``reason``."""

    def getParam(self, reason: str) -> Any:  # noqa: N802 - driver contract
        """Return the currently served value of ``reason``."""

    def updatePV(self, reason: str) -> None:  # noqa: N802 - driver contract
        """Post a monitor event for ``reason`` alone."""

    def callbackPV(self, reason: str) -> None:  # noqa: N802 - driver contract
        """Complete the asynchronous write in flight on ``reason``."""

    def setParamStatus(  # noqa: N802 - driver contract
        self, reason: str, alarm: Any, severity: Any
    ) -> None:
        """Set ``reason``'s alarm condition without changing its value."""


@dataclass(frozen=True)
class SetpointRoute:
    """What one writable address does with a written value.

    Attributes:
        address: the setpoint's own address.
        readback: the paired ``:RB`` that echoes an accepted value, or
            ``None`` when this setpoint owes no echo (a stuck fault, a
            pyat-coupled setpoint with no paired readback in the channel
            set, or one served with no lattice behind it).
        limits: the ``(low, high)`` drive band a written value is clamped
            into, or ``None`` for an unbounded setpoint.
        mode: one of ``MODE_PHYSICS`` / ``MODE_ECHO`` / ``MODE_LATCH``.
        asyn: whether the served PV declares an asynchronous write, and so
            whether the client is blocked until ``callbackPV`` fires.
    """

    address: str
    readback: str | None
    limits: tuple[float, float] | None
    mode: str
    asyn: bool


class SetpointRoutedModel(LUMEModel):
    """A model whose pyat-coupled setpoint writes go through a hook.

    The physics bridge -- not the model -- owns the two things a setpoint
    write needs beyond the lattice write itself: the magnet calibration
    applied to the commanded current, and the push of the recomputed BPM
    readings onto their served PVs. Both have to happen on whichever thread
    owns the model, and the run loop reaches the model only through
    ``model.set()``. Wrapping the model is what puts the hook on that
    thread: every transport the runner serves -- the co-hosted Channel
    Access namespace and the runner's own PVA puts alike -- ends up calling
    the same hook, from the same thread, rather than each transport
    arranging its own physics.

    Everything else delegates. A write to a variable outside ``routed`` is
    handed to the wrapped model unchanged, as is every read and the reset.
    """

    def __init__(
        self,
        model: LUMEModel,
        *,
        on_setpoint: Callable[[str, float], None],
        routed: frozenset[str],
    ) -> None:
        """Wrap ``model``, routing writes to ``routed`` through ``on_setpoint``.

        Args:
            model: the model to wrap. Reads, resets and unrouted writes reach
                it unchanged.
            on_setpoint: called as ``on_setpoint(address, value)`` for each
                routed write. Expected to apply the write to ``model``
                itself -- this wrapper does not also write it -- and to raise
                if the model refuses it.
            routed: the addresses whose writes go through ``on_setpoint``.
        """
        self._model = model
        self._on_setpoint = on_setpoint
        self._routed = routed

    @property
    def supported_variables(self) -> dict[str, Any]:
        """The wrapped model's variables, unchanged."""
        return self._model.supported_variables

    def _get(self, names: list[str]) -> dict[str, Any]:
        return dict(self._model.get(list(names)))

    def _set(self, values: dict[str, Any]) -> None:
        """Apply ``values``, routing the hooked addresses through the hook.

        The unrouted remainder is applied in one call, preserving the run
        loop's one-``set``-per-cycle shape. An empty batch is passed through
        rather than dropped: the run loop's startup cycle carries no values
        at all, and the wrapped model treats that as "re-solve and refresh".
        """
        routed = {name: values[name] for name in values if name in self._routed}
        direct = {name: values[name] for name in values if name not in self._routed}

        # The `not routed` arm is what preserves the empty startup cycle. Not
        # `if direct or not values`: a batch that is entirely routed must not
        # also trigger a bare solve of its own, because the hook's own write
        # to the model already performs one.
        if direct or not routed:
            self._model.set(direct)

        for address, value in routed.items():
            self._on_setpoint(address, value)

    def reset(self) -> None:
        """Reset the wrapped model."""
        self._model.reset()


def physics_setpoint_addresses(records: ServingRecords) -> frozenset[str]:
    """The pyat-coupled setpoint addresses in ``records``.

    The pyat-coupled partition holds both halves of the physics coupling --
    the magnet setpoints written into the lattice and the BPM readings
    solved out of it -- so the setpoints are the writable subfield of it.
    """
    suffix = f":{SETPOINT_SUBFIELD}"
    return frozenset(address for address in records.pyat_coupled if address.endswith(suffix))


def clamp_into(value: Any, limits: tuple[float, float] | None) -> Any:
    """Clamp ``value`` into ``limits``, or return it untouched.

    Values a numeric band cannot describe -- text, enum states, booleans --
    pass through: a drive band on such a channel would have no meaning, and
    silently coercing one would be worse than ignoring it. The returned
    value is either the argument itself or a band endpoint, never a
    recomputation of one, so an accepted value is bit-exact wherever it is
    later published.
    """
    if limits is None:
        return value
    # bool is an int, and clamping a flag into a numeric band is meaningless.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return value
    low, high = limits
    clamped = min(max(value, low), high)
    if clamped != value:
        LOG.info("clamped write %s into (%s, %s) -> %s", value, low, high, clamped)
    return clamped


class CohostWritePath:
    """The write behaviour of every co-hosted setpoint in one object.

    Built once, before the server exists, from the served database and the
    facility's faults and limits; consulted on every write thereafter. It
    holds no model, no server and no driver -- the driver is passed in per
    call, so the same object serves both server threads (:meth:`write` for
    Channel Access, :meth:`put` for PVA) and the run loop thread (the
    completion each schedules) without any of them reaching into another.
    """

    def __init__(
        self,
        records: ServingRecords,
        *,
        enqueue: Callable[..., None] | None = None,
        physics_setpoints: frozenset[str] = frozenset(),
        stuck_setpoints: frozenset[str] = frozenset(),
        drive_limits: Mapping[str, tuple[float, float]] | None = None,
        refusal_alarm: tuple[Any, Any] | None = None,
        pva_post: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Derive one route per writable address.

        Args:
            records: the built serving database. Supplies the setpoint ->
                readback pairing and the served spec of each address (from
                which the asynchronous-write declaration is read).
            enqueue: the run loop's enqueue callable, invoked as
                ``enqueue(values, done=callback)``. ``None`` means there is
                no model behind this process: pyat-coupled setpoints then
                latch their written value and propagate nothing, which is
                exactly what they did with no physics hook installed.
            physics_setpoints: the pyat-coupled setpoint addresses (see
                :func:`physics_setpoint_addresses`).
            stuck_setpoints: apply-fault addresses. A stuck setpoint still
                records the value written to it, but neither the physics hook
                nor the readback echo fires, so that device's readback simply
                never moves -- identically for every reader, which is what
                makes the fault a property of the substrate rather than of
                one client's view.
            drive_limits: ``{address: (low, high)}``, the band each written
                value is clamped into. The served database publishes the same
                band as display limits; nothing else enforces it.
            refusal_alarm: ``(alarm, severity)`` raised on a setpoint whose
                write the model refused, or ``None`` to leave a refusal
                silent. Cleared by the next accepted write. Raised whichever
                transport the refused write arrived on: it is the served
                channel's condition, not one client's error report.
            pva_post: called as ``pva_post(address, value)`` for every value
                this path publishes, to carry it onto the PVA channel serving
                the same address. Addresses with no PVA channel are the
                callee's business to ignore. ``None`` publishes on Channel
                Access alone.

        Raises:
            ValueError: a setpoint that routes through the model is not
                declared an asynchronous write. Its client would then be told
                the write completed before the solve had even started, and
                would read the pre-write value back.
        """
        self._enqueue = enqueue
        self._refusal_alarm = refusal_alarm
        self._pva_post = pva_post if pva_post is not None else discard_pva_post
        self._routes: dict[str, SetpointRoute] = {}

        limits = dict(drive_limits or {})
        writable = set(records.setpoint_readbacks) | set(physics_setpoints)
        for address in sorted(writable):
            if address in stuck_setpoints:
                mode, readback = MODE_LATCH, None
            elif address in physics_setpoints:
                if enqueue is None:
                    mode, readback = MODE_LATCH, None
                else:
                    mode = MODE_PHYSICS
                    readback = records.setpoint_readbacks.get(address)
            else:
                mode, readback = MODE_ECHO, records.setpoint_readbacks[address]

            self._routes[address] = SetpointRoute(
                address=address,
                readback=readback,
                limits=limits.get(address),
                mode=mode,
                asyn=bool(records.pvdb[address].get("asyn", False)),
            )

        unblocking = sorted(
            route.address
            for route in self._routes.values()
            if route.mode == MODE_PHYSICS and not route.asyn
        )
        if unblocking:
            raise ValueError(
                "setpoints routed through the model must be declared asynchronous "
                "(build_serving_pvdb(..., async_setpoints=True)); these are not: "
                f"{unblocking[:5]}{' ...' if len(unblocking) > 5 else ''}"
            )

    @property
    def routes(self) -> dict[str, SetpointRoute]:
        """The route table, keyed by setpoint address."""
        return dict(self._routes)

    def write(self, driver: WriteDriver, reason: str, value: Any) -> bool:
        """Handle a Channel Access write to ``reason``.

        Returns:
            True if ``reason`` is a writable co-hosted setpoint and the write
            was accepted for processing, False if it is not writable -- in
            which case nothing has been recorded and nothing moves. The
            server library turns a False into a rejected put; there is no
            message to carry with it.
        """
        route = self._routes.get(reason)
        if route is None:
            # Not a setpoint: a readback, a telemetry channel or a status
            # flag. Refusing here is what keeps the served value the value
            # its own source published.
            LOG.debug("refused write to non-setpoint channel %s", reason)
            return False

        self._begin(driver, route, value, partial(self._signal_ca, driver, route))
        return True

    def put(
        self,
        driver: WriteDriver,
        reason: str,
        value: Any,
        done: Callable[[str | None], None],
    ) -> bool:
        """Handle a PVA put to ``reason``, exactly as :meth:`write` would.

        The same address is served on both transports, so a put means what a
        write means: the same clamp, the same hand-off to the model, and on
        success the same values on both views of both addresses. Only the
        completion differs. ``done`` is this transport's, and it is called
        exactly once on every outcome -- with the model's error string when
        the model refused, and with ``None`` when it did not.

        That is the one place PVA is not symmetric with Channel Access, and
        it is an improvement rather than a difference to paper over: a
        Channel Access client can only be told that its write finished, and
        learns of a refusal from the alarm this path raises, while a PVA
        client is handed the reason its put failed. Both still see the same
        thing on the channel itself, which is nothing.

        Args:
            driver: the Channel Access driver. A put publishes on both views,
                so it reaches this even though it did not arrive through it.
            reason: the address written.
            value: the requested value, already unwrapped from its PVA
                container by the caller -- nothing here knows p4p types.
            done: this put's completion, called with an error string or None.

        Returns:
            True if ``reason`` is a writable co-hosted setpoint. False if it
            is not, in which case nothing moves and ``done`` has already been
            called with :data:`NOT_WRITABLE`.
        """
        route = self._routes.get(reason)
        if route is None:
            LOG.debug("refused put to non-setpoint channel %s", reason)
            # Owed even here: a put left uncompleted blocks the client that
            # issued it until its own timeout expires.
            done(NOT_WRITABLE)
            return False

        self._begin(driver, route, value, done)
        return True

    def _begin(
        self,
        driver: WriteDriver,
        route: SetpointRoute,
        value: Any,
        signal: Callable[[str | None], None],
    ) -> None:
        """Clamp, then either hand the value to the model or publish it now.

        The whole of what a write *means* is here, and both transports enter
        it, which is what makes them agree: ``signal`` is the only thing
        either one contributes of its own.
        """
        value = clamp_into(value, route.limits)

        # The `is not None` is an invariant, not a fallback: MODE_PHYSICS is
        # only ever assigned when there is a run loop to enqueue onto. It is
        # written into the condition so the narrowing is checkable.
        if route.mode == MODE_PHYSICS and self._enqueue is not None:
            # Hand the model the *clamped* value and return. Nothing is
            # published yet: the setpoint and its echo may only carry a value
            # the model has accepted, and whether it does is not known on
            # this thread.
            self._enqueue(
                {route.address: {"value": value, "ts": time.monotonic()}},
                done=partial(self.complete, driver, route, value, signal),
            )
            return

        self._publish(driver, route, value)
        signal(None)

    def complete(
        self,
        driver: WriteDriver,
        route: SetpointRoute,
        value: Any,
        signal: Callable[[str | None], None],
        error: str | None,
    ) -> None:
        """Finish a write the model has now either taken or refused.

        Called from the run loop's thread once the cycle carrying this write
        has finished, whichever transport the write arrived on. ``error`` is
        None when the model took the value.
        """
        if error is None:
            self._publish(driver, route, value)
        else:
            LOG.info("model refused write of %s to %s: %s", value, route.address, error)
            if self._refusal_alarm is not None:
                alarm, severity = self._refusal_alarm
                driver.setParamStatus(route.address, alarm, severity)
                # Posts the alarm transition alone: the value is deliberately
                # untouched, so a monitoring client sees the refusal without
                # seeing movement.
                driver.updatePV(route.address)
        signal(error)

    def _publish(self, driver: WriteDriver, route: SetpointRoute, value: Any) -> None:
        """Commit ``value`` on the setpoint and its echo, on every view.

        Posting is per address rather than a database-wide sweep: the served
        database has thousands of entries and a sweep on each write would
        cost the whole namespace to deliver two values.
        """
        self._commit(driver, route.address, value)
        if route.readback is not None:
            self._commit(driver, route.readback, value)

    def _commit(self, driver: WriteDriver, address: str, value: Any) -> None:
        """Publish one accepted value on both views of one address.

        The value is the client's own, post-clamp: not a re-derivation of it
        and never a value read back out of the model. Channel Access first,
        because it is the authoritative view of the machine and because it is
        the view whose client is still blocked.
        """
        driver.setParam(address, value)
        driver.updatePV(address)
        try:
            self._pva_post(address, value)
        except Exception:
            # A failure on the second view must not cost the first view's
            # client its completion: `_signal_ca` runs after this, and a
            # Channel Access write whose `callbackPV` never fires postpones
            # every later write to that PV for the life of the process. The
            # authoritative value is committed above, before anything here
            # can fail, so what is lost is one PVA update and not a value.
            LOG.exception("failed to publish %s on the PVA view of %s", value, address)

    def _signal_ca(self, driver: WriteDriver, route: SetpointRoute, error: str | None) -> None:
        """Complete the Channel Access write, if this setpoint declares one.

        Always last, and on every outcome. Last, because completion is what
        unblocks a client waiting on the put and it must find the values
        already committed. On every outcome -- ``error`` is accepted and
        ignored, there being no failure channel to carry it -- because the
        server library postpones every later write to a PV whose asynchronous
        write is still in flight, so a refusal that skipped this would freeze
        the setpoint.

        Fires only for a write that arrived on Channel Access. A PVA put
        starts no asynchronous write here, and ending one that was never
        started would complete some *other* client's put.
        """
        if route.asyn:
            driver.callbackPV(route.address)


__all__ = [
    "MODE_ECHO",
    "MODE_LATCH",
    "MODE_PHYSICS",
    "NOT_WRITABLE",
    "RUNNER_CONFIG_POLICY",
    "CohostWritePath",
    "SetpointRoute",
    "SetpointRoutedModel",
    "WriteDriver",
    "clamp_into",
    "discard_pva_post",
    "physics_setpoint_addresses",
]
