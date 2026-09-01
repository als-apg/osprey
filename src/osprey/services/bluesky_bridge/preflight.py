"""What a worker plan wrapper settles before it hands a plan to the RunEngine.

Both wrappers in this package — ``qserver_startup._make_plan_function`` for a
catalog plan and ``session_upload.install_session_plan`` for a session one —
build the same connector-mediated devices in the same worker process, and both
have the same two things to say before a run starts: whether the channels it
declares are alive, and, when a declared name does not resolve, what this
worker actually holds. One copy of each lives here so the two wrappers cannot
answer an operator differently.

The reachability half: ask the connector, before anything moves.

An address that no IOC serves is invisible when the worker builds its devices —
``ConnectorSettable``/``ConnectorReadable`` declare no ophyd signals and the
device build does no I/O — so today it first surfaces *mid-run*, as a
``ConnectionError`` from the connector's read, with the RunEngine already part
way through a plan and setpoints already applied. This module is the
before-motion half of that story: given a connector and the addresses a run is
about to touch, it answers which of them do not respond, and the caller turns
that answer into a refusal instead of a partially-executed run.

Two properties are deliberate:

**The bound is the probe's own.** Each ``validate_channel`` call is wrapped in
``asyncio.wait_for(..., timeout_s)`` even though some connectors bound
themselves — the EPICS connector probes with a 2 s timeout, the DOOCS one is
unbounded — because a pre-flight gate that inherits a per-connector timeout
gives a different, and sometimes infinite, worst case per lane. The constants
here bound every lane alike, and the refusal quotes whichever of them the
verdict actually rested on (:attr:`ProbeOutcome.timeout_s` for an address that
failed a probe, :attr:`ProbeOutcome.budget_s` for one the sweep never reached).

**One retry, and a budget that scales with the sweep.** A gateway under load
can drop a first probe for a channel that is perfectly healthy, and turning
that into a refused run would be worse than the failure mode being fixed, so
everything the first pass failed is asked once more — after that pass finishes,
so the retry never competes with it. The retry is bounded exactly like the
first pass, and the sweep as a whole is bounded in wall-clock time, because
per-probe bounds alone do not bound a sweep: the case this gate exists for is
the one where a gateway or an IOC is down and *every* address fails, and a
per-address timeout paid twice over hundreds of addresses is minutes of a
blocked RunEngine. Nothing outside this module supplies that bound — the
``wait_for`` message carries no timeout and the queue has no per-item execution
limit — so it is supplied here, by :func:`sweep_budget`.

That budget cannot be one number. A fixed one either strangles a large healthy
plan or lets a large dead one hold the RunEngine for minutes, because the two
costs differ by the ratio of a probe timeout to a channel's answer latency. So
it is derived from how many addresses there are and how many of them the
connector can really have in flight, floored at :data:`PROBE_BUDGET_S` and
capped at :data:`PROBE_BUDGET_MAX_S`.

**What a cut-short sweep may claim.** An address is reported unresponsive only
on the evidence of a probe that ran and came back negative. Everything else the
budget left behind — never started, or still in flight when the budget expired
— is reported *unchecked*, separately, because "did not respond within 5 s" is
a statement about a probe that happened. Both refuse the run; they do not
pretend to be the same finding.

Pure ``asyncio`` over the connector abstraction: no Channel Access, no
``bluesky_queueserver``, nothing that only exists inside the worker container.
The coroutine is awaited on the RunEngine's own loop in production and runs
just as well on a bare event loop under pytest.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("osprey.services.bluesky_bridge.preflight")

PROBE_TIMEOUT_S = 5.0
"""Seconds a single address gets to answer one probe, on every connector.

Comfortably above the EPICS connector's own 2 s channel probe, so a reachable
channel behind a slow gateway is not failed by this bound before its connector
has had a fair attempt; short enough that a plan naming a dead address is
refused in seconds rather than after a per-address stall. Quoted verbatim by
the refusal the caller composes for the addresses that actually failed a
probe, so changing it changes what operators are told.
"""

PROBE_CONCURRENCY = 32
"""How many probes may be in flight at once.

A roster-derived plan can name hundreds of channels, and firing every probe
simultaneously turns a reachability check into a burst the gateway reads as
load — the very condition that produces false negatives. The sweep stays
concurrent, just not unbounded.

This is a ceiling, not a promise: the EPICS connector's own reads dispatch
through ``asyncio.to_thread``, so its real parallelism is whatever the default
``ThreadPoolExecutor`` allows and can sit well below this number on a small
container — six threads on a two-CPU one. :func:`_dispatch_ceiling` models
that, and the sweep's budget is costed against the smaller of the two rather
than against this constant, which would otherwise promise a parallelism the
connector never delivers.
"""

PROBE_BUDGET_S = 20.0
"""Floor on a sweep's wall-clock budget: what a small plan gets.

Four probe timeouts — room for a slow but healthy pass plus its retry over the
handful of channels an ordinary plan declares, and a bound on how long such a
run can sit in this gate when nothing answers. A plan declaring more addresses
than one dispatch batch is given proportionally more (:func:`sweep_budget`).
"""

PROBE_BUDGET_MAX_S = 90.0
"""Ceiling on a sweep's wall-clock budget, however many addresses it names.

Sized from the largest device set this project ships against the slowest
latency a healthy gateway plausibly answers with, through the dispatch
parallelism a small container actually has: 2908 addresses × 0.1 s ÷ 6 threads
≈ 49 s for the first pass, leaving some forty seconds for retries and jitter.
A healthy sweep of any roster OSPREY builds therefore finishes inside this, and
is not refused for being large.

The cost of that headroom is the other half of the trade, and it is real: when
nothing answers, a plan declaring more than a few dozen channels holds the
RunEngine here for up to a minute and a half instead of twenty seconds. That is
accepted deliberately. Refusing a *healthy* large plan because a loaded gateway
answered in 100 ms rather than 20 ms would be a false refusal of real work,
which is worse than a slow true one — and small plans, which is nearly all of
them, are unaffected either way.
"""


def _dispatch_ceiling() -> int:
    """How many probes the connector can really have in flight at once.

    The EPICS connector's reads go through ``asyncio.to_thread``, which lands
    on the interpreter's default ``ThreadPoolExecutor`` — ``min(32, cpu_count +
    4)``, six on a two-CPU container. Spelled the way CPython sizes it, because
    a budget costed against :data:`PROBE_CONCURRENCY` would assume a
    parallelism the connector cannot deliver and would then look generous while
    being five times too small.
    """
    return min(32, (os.cpu_count() or 1) + 4)


def sweep_budget(
    address_count: int,
    *,
    timeout_s: float = PROBE_TIMEOUT_S,
    concurrency: int = PROBE_CONCURRENCY,
) -> float:
    """The wall-clock budget a sweep over ``address_count`` addresses gets.

    Two passes of ``timeout_s`` per batch of addresses the connector can hold
    in flight — the modelled cost of the *worst* case, where nothing answers —
    clamped between :data:`PROBE_BUDGET_S` and :data:`PROBE_BUDGET_MAX_S`.

    Deriving it from the count is what keeps the bound from being a second
    failure mode. A budget fixed at the floor refuses healthy plans purely for
    being large (2908 addresses answering in 100 ms each, six at a time, is 49
    seconds of perfectly good pre-flight); one fixed at the ceiling makes every
    small plan's dead-gateway case cost a minute and a half. The size of the
    plan is the thing that separates those, so the size of the plan is what
    sets the budget.
    """
    parallel = max(1, min(concurrency, _dispatch_ceiling()))
    modelled = 2 * timeout_s * math.ceil(max(0, address_count) / parallel)
    return min(max(modelled, PROBE_BUDGET_S), PROBE_BUDGET_MAX_S)


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """What a pre-flight sweep found.

    The two failure populations are kept apart on purpose. Both refuse the
    run, but only one of them is evidence that a channel is unreachable, and a
    refusal that called them the same thing would tell an operator that
    channels were dead when all that is known is that nobody got to ask.

    Attributes:
        unresponsive: The addresses whose probe ran and came back negative,
            twice, in the order they were first named.
        unchecked: The addresses the sweep never got a complete answer for —
            never started, or still in flight when the budget expired — in the
            order they were first named.
        timeout_s: The bound each individual probe was given.
        budget_s: The bound the sweep as a whole was given.
        budget_exhausted: True when the sweep was cut short by ``budget_s``
            rather than finishing on its own.
    """

    unresponsive: tuple[str, ...]
    timeout_s: float
    budget_s: float
    budget_exhausted: bool = False
    unchecked: tuple[str, ...] = ()

    @property
    def all_responded(self) -> bool:
        """True when every address answered — nothing failed, nothing skipped.

        An unchecked address is not a pass. A sweep the budget cut short before
        it reached half its addresses knows nothing about that half, and
        treating silence as consent is the failure this gate exists to end.
        """
        return not self.unresponsive and not self.unchecked


def _ordered_unique(addresses: Iterable[str]) -> tuple[str, ...]:
    """The addresses to probe: blanks dropped, duplicates collapsed, order kept.

    A settable contributes both its setpoint and its readback, and several plan
    parameters may name the same channel, so the caller's list routinely repeats
    itself; probing an address twice would only double the load. Order is
    first-seen so the refusal reads in the order the plan declared things.
    """
    seen: dict[str, None] = {}
    for address in addresses:
        if isinstance(address, str) and address.strip():
            seen.setdefault(address, None)
    return tuple(seen)


async def _probe_one(connector: Any, address: str, timeout_s: float) -> bool:
    """Probe one address, bounded by ``timeout_s``.

    Any failure mode the connector can produce — a false return, a timeout, a
    raised transport error — is the same answer to this question: the address
    did not respond. Only cancellation of the probe itself propagates.
    """
    try:
        return bool(await asyncio.wait_for(connector.validate_channel(address), timeout_s))
    except TimeoutError:
        logger.debug("preflight: %s did not answer within %.3gs", address, timeout_s)
        return False
    except Exception as exc:
        logger.debug("preflight: %s probe failed: %s", address, exc)
        return False


async def probe_addresses(
    connector: Any,
    addresses: Iterable[str],
    *,
    timeout_s: float = PROBE_TIMEOUT_S,
    concurrency: int = PROBE_CONCURRENCY,
    budget_s: float | None = None,
) -> ProbeOutcome:
    """Probe every address on ``connector`` and report what did not answer.

    Runs a bounded-concurrency sweep of ``connector.validate_channel`` over the
    distinct addresses, then asks the failures once more under the same bound.
    An address is reported *unresponsive* when both attempts ran and both came
    back negative, and *unchecked* when the sweep's budget expired before it
    had a complete answer for it.

    Every address the caller named is accounted for however the sweep ends: the
    result is derived from positive evidence — which addresses answered, and
    which were actually asked — so a pass cut short reports what it never
    reached rather than losing it or asserting something about it.

    Args:
        connector: Any OSPREY control-system connector — the only thing asked
            of it is ``async validate_channel(address) -> bool``.
        addresses: Addresses a run is about to touch. Blank entries and
            duplicates are dropped; first-seen order is preserved.
        timeout_s: Seconds each individual probe is given, on both the first
            pass and the retry.
        concurrency: Maximum probes in flight, on both passes.
        budget_s: Seconds the whole sweep gets before it stops asking.
            Defaults to :func:`sweep_budget` for this address count.

    Returns:
        A :class:`ProbeOutcome`; ``all_responded`` is true when nothing failed
        and nothing was left unchecked.
    """
    targets = _ordered_unique(addresses)
    if budget_s is None:
        budget_s = sweep_budget(len(targets), timeout_s=timeout_s, concurrency=concurrency)
    if not targets:
        return ProbeOutcome(unresponsive=(), timeout_s=timeout_s, budget_s=budget_s)

    limit = asyncio.Semaphore(max(1, concurrency))
    responded: set[str] = set()
    answered: set[str] = set()

    async def bounded(address: str) -> None:
        async with limit:
            # Recorded only once the probe has come back: a probe cancelled
            # in flight by the budget produced no answer, and the refusal is
            # not entitled to say the channel failed one.
            reachable = await _probe_one(connector, address, timeout_s)
            answered.add(address)
            if reachable:
                responded.add(address)

    budget_exhausted = False
    try:
        async with asyncio.timeout(budget_s):
            await asyncio.gather(*(bounded(address) for address in targets))
            failures = [address for address in targets if address not in responded]
            if failures:
                logger.debug("preflight: retrying %d unresponsive address(es)", len(failures))
                await asyncio.gather(*(bounded(address) for address in failures))
    except TimeoutError:
        budget_exhausted = True
        logger.warning(
            "preflight: the %.3gs sweep budget expired with %d of %d address(es) unconfirmed",
            budget_s,
            len(targets) - len(responded),
            len(targets),
        )

    return ProbeOutcome(
        unresponsive=tuple(
            address for address in targets if address in answered and address not in responded
        ),
        timeout_s=timeout_s,
        budget_s=budget_s,
        budget_exhausted=budget_exhausted,
        unchecked=tuple(address for address in targets if address not in answered),
    )


_ADDRESS_ATTRIBUTES = ("_setpoint_pv", "_readback_pv", "_read_pv")
"""Where a connector-mediated device keeps the addresses it touches.

``ConnectorSettable`` holds a setpoint and a readback, ``ConnectorReadable``
one read channel, and neither exposes them publicly. Read by name rather than
by importing the device classes: that would pull ophyd-async into the import
path of every module that wants a pre-flight, for a string lookup.
"""


def _declared_addresses(declared: Mapping[str, Any]) -> list[str]:
    """Every control-system address the declared devices would touch.

    A settable contributes both halves: a plan that drives a setpoint also
    reads the readback to settle each move, so an unreachable readback fails a
    run exactly as an unreachable setpoint does. A readable contributes its one
    channel. Duplicates and blanks are left for the probe to drop.

    A device carrying none of these attributes contributes nothing rather than
    raising — in the worker every device is connector-mediated, so that case is
    a caller holding something else, and there is no address in it to probe.
    """
    addresses: list[str] = []
    for device in declared.values():
        for attribute in _ADDRESS_ATTRIBUTES:
            address = getattr(device, attribute, None)
            if isinstance(address, str):
                addresses.append(address)
    return addresses


def probe_before_motion(plan_name: str, declared: Mapping[str, Any]) -> Iterator[Any]:
    """Refuse the run, before it moves anything, if a declared address is dead.

    This is the per-*run* seam, and both worker plan wrappers ``yield from``
    it. An address no IOC serves is invisible when the worker builds its
    devices and, without this, first surfaces mid-plan as a ``ConnectionError``
    out of the connector's read — with setpoints already applied. Asking the
    same question here, one message before the plan is even constructed, turns
    that into a refusal: nothing has been sent, so nothing is half-done. It is
    deliberately not asked at enqueue time either; what a queued item needs is
    that its channels are alive *when it runs*, and the gap between the two can
    be an IOC restart.

    The probe is a coroutine and has to run on the RunEngine's own loop — the
    loop the connector was connected on. Inside a plan that is what
    ``Msg('wait_for', None, [factory])`` is for (``bluesky.plan_stubs.wait_for``
    sends the same message): the RunEngine creates the awaitable on its loop
    and hands back the finished tasks. Scheduling onto that loop from here
    directly would deadlock instead — a plan is advanced *by* the loop thread.

    Args:
        plan_name: The plan's name, as the refusal should name it.
        declared: The devices this run may resolve — the channels its params
            declare, and nothing else.

    Raises:
        ConnectionError: One or more declared addresses failed their probe, or
            were left unchecked when the sweep's budget expired. The same class
            the connector raises for an unreachable channel mid-run, so the
            failure keeps its name and only changes its timing.
    """
    addresses = _declared_addresses(declared)
    connector = next(
        (
            osprey_connector
            for osprey_connector in (
                getattr(device, "_osprey_connector", None) for device in declared.values()
            )
            if osprey_connector is not None
        ),
        None,
    )
    skipped = None
    if connector is None:
        skipped = "no declared device names an OSPREY connector"
    elif not callable(getattr(connector, "validate_channel", None)):
        # Unreachable through a real connector — `validate_channel` is abstract
        # on `ControlSystemConnector` — and checked anyway, once, because the
        # alternative is worse than a skipped gate: every probe would raise
        # `AttributeError`, every address would read as dead, and a wiring
        # mistake would come back to the operator as "the machine is down".
        skipped = f"{type(connector).__name__} exposes no validate_channel probe"
    elif not addresses:
        skipped = "no declared device exposes a control-system address"
    if skipped is not None:
        if declared:
            logger.warning(
                "preflight: running plan %r without a reachability pre-flight — %s",
                plan_name,
                skipped,
            )
        return

    from bluesky.utils import Msg

    # `RunEngine._wait_for` creates the awaitable on its own loop and returns
    # the list of finished tasks, which is what makes a non-empty answer here
    # mean "a RunEngine ran this". Contract read from bluesky 1.15.1
    # (`run_engine.py::_wait_for`), the floor this package pins; re-verify it
    # if that floor moves. Anything else driving this generator — the read-only
    # preview walk in `qserver_startup.collect_channel_moves` is the one that
    # does — consumes the message without executing it and sends `None` back.
    finished = yield Msg("wait_for", None, [lambda: probe_addresses(connector, addresses)])
    if not finished:
        logger.info(
            "preflight: no RunEngine answered the pre-flight probe for plan %r — skipping it; "
            "the caller walked this plan rather than running it, so it moves nothing",
            plan_name,
        )
        return

    outcome = finished[0].result()
    if outcome.all_responded:
        return

    from osprey.services.bluesky_bridge.queue_backend import resolve_lane_identity

    lane, target = resolve_lane_identity()
    clauses = []
    if outcome.unresponsive:
        count = len(outcome.unresponsive)
        clauses.append(
            f"{count} declared channel{_plural(count)} did not respond within "
            f"{outcome.timeout_s:g} s: {_capped_addresses(outcome.unresponsive)}"
        )
    if outcome.unchecked:
        count = len(outcome.unchecked)
        clauses.append(
            f"{count} further channel{_plural(count)} {'was' if count == 1 else 'were'} still "
            f"unchecked when the {outcome.budget_s:g} s pre-flight budget expired: "
            f"{_capped_addresses(outcome.unchecked)}"
        )
    raise ConnectionError(
        f"refusing plan {plan_name!r} before it moves anything — "
        f"lane {lane} (target {target}):\n" + "\n".join(clauses)
    )


def _plural(count: int) -> str:
    """``"s"`` unless there is exactly one of them."""
    return "" if count == 1 else "s"


def _capped_addresses(addresses: tuple[str, ...]) -> str:
    """A refusal clause's address list, capped at :data:`DEVICE_NAME_LIMIT`.

    Same reason the device list is capped: this text crosses a 0MQ hop into an
    operator's plan-submission failure, and a plan over a roster-derived device
    set can name thousands of addresses. Order is the plan's own, not sorted —
    the first names an operator reads should be the first ones the plan
    declares.
    """
    if len(addresses) <= DEVICE_NAME_LIMIT:
        return ", ".join(addresses)
    shown = ", ".join(addresses[:DEVICE_NAME_LIMIT])
    return f"{shown} (+{len(addresses) - DEVICE_NAME_LIMIT} more)"


DEVICE_NAME_LIMIT = 20
"""How many device names a wrapper's refusal lists before summarizing the rest.

These messages are interpolated straight into an error the worker ships back
over 0MQ into an operator's plan-submission failure, and a roster-derived
device set runs into the thousands — the message stays legible prose instead of
enumerating the whole namespace.
"""


def available_devices_phrase(available: Mapping[str, Any] | Iterable[str]) -> str:
    """The unresolved-device message's device list, capped for readability.

    Names the first :data:`DEVICE_NAME_LIMIT` devices in sorted order and, once
    the worker has built more than that, summarizes the rest by count and
    points at ``GET /devices`` instead of naming them all.

    ``queue._available_devices_phrase`` is the bridge-side sibling of this and
    deliberately keeps its own body: its cap also honours the caller's page
    size, and it points at the response body's own ``available_devices`` when
    the whole set fits in one page — a pointer that only exists over HTTP.
    """
    names = sorted(available)
    if len(names) <= DEVICE_NAME_LIMIT:
        return f"{names}"
    shown = names[:DEVICE_NAME_LIMIT]
    return f"{shown} (+{len(names) - len(shown)} more; full list via GET /devices)"
