"""Unit tests for the worker-side pre-flight reachability probe (task 5.1).

Host-testable on purpose: `preflight` talks to nothing but the connector
abstraction, so the whole module is exercised here against a stub connector —
no Channel Access, no IOC, no queueserver, no bluesky import. The stub's call
log is part of what is asserted, because "one retry of only the failures" is a
claim about *how many times each address is asked*, not just about the answer,
and the sweep's wall clock is asserted too — the bound on a mass failure is a
property nothing outside this module supplies.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from osprey.services.bluesky_bridge.preflight import (
    PROBE_BUDGET_MAX_S,
    PROBE_BUDGET_S,
    PROBE_CONCURRENCY,
    PROBE_TIMEOUT_S,
    ProbeOutcome,
    probe_addresses,
    sweep_budget,
)

FAST_TIMEOUT = 0.05
"""Probe bound used by the timeout tests, so a hang costs milliseconds."""


class StubConnector:
    """A connector stand-in whose `validate_channel` is scripted per address.

    Each address maps to a list of outcomes consumed one per call, so a channel
    can fail its first probe and answer the retry. An outcome is either a bool,
    an exception instance to raise, or the string ``"hang"`` for a probe that
    never returns. Addresses with no script always answer True.
    """

    def __init__(self, script: dict[str, list[object]] | None = None) -> None:
        self._script = {address: list(outcomes) for address, outcomes in (script or {}).items()}
        self.calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.cancelled: list[str] = []

    async def validate_channel(self, channel_address: str) -> bool:
        self.calls.append(channel_address)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            outcomes = self._script.get(channel_address)
            outcome: object = True
            if outcomes:
                outcome = outcomes.pop(0)
            elif outcomes is not None:
                outcome = False
            if outcome == "hang":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.append(channel_address)
                    raise
            if isinstance(outcome, BaseException):
                raise outcome
            await asyncio.sleep(0)
            return bool(outcome)
        finally:
            self.in_flight -= 1


async def test_all_addresses_respond_reports_nothing_unresponsive():
    connector = StubConnector()

    outcome = await probe_addresses(connector, ["SR:A", "SR:B", "SR:C"])

    assert outcome == ProbeOutcome(
        unresponsive=(), timeout_s=PROBE_TIMEOUT_S, budget_s=PROBE_BUDGET_S
    )
    assert outcome.all_responded is True
    assert connector.calls == ["SR:A", "SR:B", "SR:C"]


async def test_flaky_address_answering_the_retry_is_not_reported():
    connector = StubConnector({"SR:FLAKY": [False, True]})

    outcome = await probe_addresses(connector, ["SR:OK", "SR:FLAKY"])

    assert outcome.unresponsive == ()
    assert outcome.all_responded is True
    assert connector.calls.count("SR:FLAKY") == 2, "the failure gets exactly one retry"
    assert connector.calls.count("SR:OK") == 1, "only failures are retried"


async def test_dead_address_is_reported_after_the_retry():
    connector = StubConnector({"SR:DEAD": [False, False]})

    outcome = await probe_addresses(connector, ["SR:OK", "SR:DEAD", "SR:ALSO_OK"])

    assert outcome.unresponsive == ("SR:DEAD",)
    assert outcome.all_responded is False
    assert outcome.timeout_s == PROBE_TIMEOUT_S
    assert connector.calls.count("SR:DEAD") == 2


async def test_unresponsive_addresses_keep_the_declared_order():
    connector = StubConnector({"SR:D1": [False, False], "SR:D2": [False, False]})

    outcome = await probe_addresses(connector, ["SR:D2", "SR:OK", "SR:D1"])

    assert outcome.unresponsive == ("SR:D2", "SR:D1")


async def test_a_hanging_validate_is_bounded_by_the_probe_not_the_connector():
    connector = StubConnector({"SR:HUNG": ["hang", "hang"]})

    started = time.monotonic()
    outcome = await asyncio.wait_for(
        probe_addresses(connector, ["SR:OK", "SR:HUNG"], timeout_s=FAST_TIMEOUT),
        timeout=5.0,
    )
    elapsed = time.monotonic() - started

    assert outcome.unresponsive == ("SR:HUNG",)
    assert outcome.timeout_s == FAST_TIMEOUT
    # Two bounded attempts (concurrent pass + serial retry), nothing unbounded.
    assert elapsed < FAST_TIMEOUT * 10
    assert connector.cancelled == ["SR:HUNG", "SR:HUNG"], "each hung probe is cancelled"


async def test_one_hanging_address_does_not_stall_the_others():
    connector = StubConnector({"SR:HUNG": ["hang", "hang"]})

    outcome = await probe_addresses(connector, ["SR:HUNG", "SR:A", "SR:B"], timeout_s=FAST_TIMEOUT)

    assert outcome.unresponsive == ("SR:HUNG",)
    assert connector.calls.count("SR:A") == 1
    assert connector.calls.count("SR:B") == 1


async def test_a_raising_validate_counts_as_unresponsive():
    connector = StubConnector({"SR:BOOM": [RuntimeError("no route to host"), True]})

    outcome = await probe_addresses(connector, ["SR:BOOM"])

    assert outcome.unresponsive == (), "the retry answered, so the run is not refused"

    connector = StubConnector(
        {"SR:BOOM": [RuntimeError("no route to host"), ConnectionError("still gone")]}
    )
    outcome = await probe_addresses(connector, ["SR:BOOM"])

    assert outcome.unresponsive == ("SR:BOOM",)


async def test_duplicate_and_blank_addresses_are_collapsed_before_probing():
    connector = StubConnector()

    outcome = await probe_addresses(connector, ["SR:A", "SR:A", "", "   ", "SR:B", "SR:A"])

    assert outcome.all_responded is True
    assert connector.calls == ["SR:A", "SR:B"]


async def test_no_addresses_never_touches_the_connector():
    connector = StubConnector()

    outcome = await probe_addresses(connector, [])

    assert outcome == ProbeOutcome(
        unresponsive=(), timeout_s=PROBE_TIMEOUT_S, budget_s=PROBE_BUDGET_S
    )
    assert connector.calls == []


async def test_the_concurrent_pass_is_bounded():
    connector = StubConnector()
    addresses = [f"SR:CH{index:03d}" for index in range(PROBE_CONCURRENCY * 3)]

    outcome = await probe_addresses(connector, addresses, concurrency=4)

    assert outcome.all_responded is True
    assert len(connector.calls) == len(addresses)
    assert connector.max_in_flight <= 4


async def test_the_probe_bound_is_connector_agnostic_and_defaults_to_the_module_constant():
    connector = StubConnector()

    outcome = await probe_addresses(connector, ["SR:A"])

    assert outcome.timeout_s == PROBE_TIMEOUT_S
    assert PROBE_TIMEOUT_S > 2.0, "must not be tighter than the EPICS connector's own 2 s probe"


def test_probe_outcome_is_immutable():
    outcome = ProbeOutcome(
        unresponsive=("SR:DEAD",), timeout_s=PROBE_TIMEOUT_S, budget_s=PROBE_BUDGET_S
    )

    with pytest.raises(Exception):
        outcome.unresponsive = ()  # type: ignore[misc]


async def test_a_sweep_where_nothing_answers_is_bounded_by_the_budget():
    """The mass-failure case: a dead gateway must not hold the RunEngine.

    Per-probe bounds do not bound a sweep. With every address hanging, the
    first pass alone costs ``timeout_s`` per batch of ``concurrency`` and the
    retry asks all of them again — minutes of a blocked RunEngine at the roster
    sizes this gate is meant for, since nothing outside the module imposes a
    limit (the ``wait_for`` message carries no timeout and the queue has no
    per-item execution bound). The budget is that limit, and it is pinned here
    in wall-clock time rather than in arithmetic over the constants, which the
    connector's own thread-pool dispatch can invalidate.
    """
    addresses = [f"SR:CH{index:03d}" for index in range(64)]
    connector = StubConnector({address: ["hang"] * 4 for address in addresses})

    started = time.monotonic()
    outcome = await probe_addresses(
        connector, addresses, timeout_s=FAST_TIMEOUT * 10, concurrency=8, budget_s=0.3
    )
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"the sweep ran {elapsed:.2f}s against a 0.3s budget"
    assert outcome.budget_exhausted is True
    assert outcome.all_responded is False
    # Nothing here answered a probe, so nothing is claimed to have failed one.
    assert outcome.unchecked == tuple(addresses)
    assert outcome.unresponsive == ()


async def test_a_cut_short_sweep_reports_what_it_never_asked_as_unchecked():
    """ "Did not respond" is a claim about a probe that happened.

    A probe still in flight when the budget expired produced no answer at all,
    and reporting it as unresponsive would tell an operator the channel is
    dead when all that is known is that nobody finished asking.
    """
    connector = StubConnector({"SR:HANGS": ["hang", "hang"]})

    outcome = await probe_addresses(
        connector, ["SR:HANGS"], timeout_s=FAST_TIMEOUT * 10, budget_s=0.1
    )

    assert outcome.budget_exhausted is True
    assert outcome.unchecked == ("SR:HANGS",)
    assert outcome.unresponsive == ()
    assert outcome.budget_s == 0.1


async def test_a_sweep_that_finishes_reports_its_failures_as_probed():
    connector = StubConnector({"SR:DEAD": [False, False]})

    outcome = await probe_addresses(connector, ["SR:DEAD"], timeout_s=FAST_TIMEOUT)

    assert outcome.budget_exhausted is False
    assert outcome.unresponsive == ("SR:DEAD",)
    assert outcome.unchecked == ()
    assert outcome.timeout_s == FAST_TIMEOUT


async def test_the_retry_is_bounded_like_the_first_pass():
    """The retry is the mass-failure case's second helping, so it is capped too.

    A retry that fired every failure at once would be exactly the burst the
    first pass is bounded to avoid, and at a moment when the gateway is by
    definition already struggling.
    """
    addresses = [f"SR:CH{index:03d}" for index in range(24)]
    connector = StubConnector({address: [False, True] for address in addresses})

    outcome = await probe_addresses(connector, addresses, concurrency=4)

    assert outcome.all_responded is True
    assert connector.max_in_flight <= 4
    assert len(connector.calls) == 2 * len(addresses)


class SlowConnector:
    """A connector whose every channel is healthy but not fast.

    Models the condition that makes a fixed budget dangerous: a loaded gateway
    answering every search correctly, in a hundred milliseconds rather than a
    few.
    """

    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s
        self.calls: list[str] = []

    async def validate_channel(self, channel_address: str) -> bool:
        self.calls.append(channel_address)
        await asyncio.sleep(self.latency_s)
        return True


def test_the_budget_covers_a_healthy_sweep_of_the_largest_roster_shipped():
    """A big plan must not be refused for being big.

    The arithmetic the ceiling is sized against, spelled out: the demo corpus
    is 2908 channels, a loaded-but-healthy gateway answers a search in about
    100 ms, and `validate_channel` dispatches through the interpreter's default
    thread pool — six threads on a two-CPU container. A budget below the
    product of those refuses a plan whose channels are all perfectly alive,
    which is a false refusal of real work.
    """
    roster = 2908
    healthy_pass_s = roster * 0.1 / 6

    assert healthy_pass_s < PROBE_BUDGET_MAX_S, (
        f"a healthy sweep of {roster} addresses models at {healthy_pass_s:.0f}s, "
        f"past the {PROBE_BUDGET_MAX_S}s ceiling — every such plan would be refused"
    )
    assert sweep_budget(roster) == PROBE_BUDGET_MAX_S


def test_the_budget_scales_between_its_floor_and_its_ceiling():
    assert sweep_budget(1) == PROBE_BUDGET_S
    assert sweep_budget(1_000_000) == PROBE_BUDGET_MAX_S
    assert PROBE_BUDGET_S <= sweep_budget(200) <= PROBE_BUDGET_MAX_S


def test_the_floor_covers_two_full_passes_of_the_probe_timeout():
    """A small plan's budget must not cut its own retry short."""
    assert PROBE_BUDGET_S >= 4 * PROBE_TIMEOUT_S
    assert PROBE_BUDGET_MAX_S > PROBE_BUDGET_S


async def test_a_large_healthy_but_slow_sweep_is_not_refused():
    """The live half of the ceiling pin, at a scale a unit test can afford.

    Same shape as the roster case — every channel answers, none quickly — so a
    budget that did not scale with the address count would report a machine
    full of dead channels.
    """
    addresses = [f"SR:CH{index:04d}" for index in range(400)]
    connector = SlowConnector(latency_s=0.02)

    started = time.monotonic()
    outcome = await probe_addresses(connector, addresses, concurrency=8)
    elapsed = time.monotonic() - started

    assert outcome.all_responded is True
    assert outcome.budget_exhausted is False
    assert elapsed < sweep_budget(len(addresses), concurrency=8)
