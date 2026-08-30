"""The background TCP prober behind the roster's ``endpoint_tcp`` rows (FR-2).

Two properties carry most of the weight here and are tested against real
sockets rather than mocks, because both are claims about the network:

* an ``ok`` row is taken against a listener that genuinely accepts a connection;
* an ``unreachable`` row is taken against a port nothing is bound to.

The third is the honest refusal: an address-list gateway must produce
``not_applicable`` **without opening a socket at all**, so that test replaces
``asyncio.open_connection`` with a function that fails the test if it is ever
called. A row that says "not applicable" while having quietly connected would be
the exact dishonesty the mode distinction exists to prevent.

Every interval and timeout here is tiny, and the clock is injected rather than
slept through, so none of this costs the suite real time.

The last section covers the other half of the prober's job: publishing what it
measured to the state file every sweep, in the vocabulary the readers use.
"""

from __future__ import annotations

import asyncio
import copy
import socket
from datetime import UTC, datetime
from typing import Any

import pytest

from osprey.audit.posture import OSPREY_AGENT_DATA_ROOT
from osprey.mcp_server.control_system import endpoint_prober as ep
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.endpoint_prober import (
    STATE_DOWN,
    STATE_REACHED,
    STATUS_NOT_APPLICABLE,
    STATUS_OK,
    STATUS_STALE,
    STATUS_UNREACHABLE,
    EndpointProber,
)

LIVE = "live"
VA = "va"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """A monotonic clock a test advances by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _va_config(port: Any, *, use_name_server: bool = True, **extra: Any) -> dict[str, Any]:
    """A config whose only target is the virtual accelerator."""
    control_system: dict[str, Any] = {
        "type": "virtual_accelerator",
        "connector": {
            "virtual_accelerator": {
                "probe_channel": "VA:PROBE",
                "gateways": {
                    "read_only": {
                        "address": "127.0.0.1",
                        "port": port,
                        "use_name_server": use_name_server,
                    }
                },
            }
        },
    }
    control_system.update(extra)
    return {"control_system": control_system}


def _both_targets_config(va_port: Any, live_port: Any) -> dict[str, Any]:
    return {
        "control_system": {
            "type": "virtual_accelerator",
            "connector": {
                "virtual_accelerator": {
                    "probe_channel": "VA:PROBE",
                    "gateways": {
                        "read_only": {
                            "address": "127.0.0.1",
                            "port": va_port,
                            "use_name_server": True,
                        }
                    },
                },
                "epics": {
                    "probe_channel": "LIVE:PROBE",
                    "gateways": {
                        "read_only": {
                            "address": "127.0.0.1",
                            "port": live_port,
                            "use_name_server": True,
                        }
                    },
                },
            },
        }
    }


def _closed_port() -> int:
    """A port that was bound long enough to be sure nothing else has it, then freed."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


@pytest.fixture
async def listener():
    """A real TCP listener on an ephemeral port; yields its port."""

    async def _handle(_reader: Any, writer: Any) -> None:
        writer.close()

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    try:
        yield port
    finally:
        server.close()
        await server.wait_closed()


def _prober(config: dict[str, Any], **kwargs: Any) -> EndpointProber:
    kwargs.setdefault("connect_timeout_s", 0.5)
    kwargs.setdefault("interval_s", 0.05)
    return EndpointProber(config, **kwargs)


# ---------------------------------------------------------------------------
# Probe semantics
# ---------------------------------------------------------------------------


async def test_listening_name_server_reports_ok(listener):
    prober = _prober(_va_config(listener), targets=(VA,))

    await prober.sweep_once()
    row = prober.snapshot()[VA]["read_only"]

    assert row["endpoint_tcp"] == STATUS_OK
    assert row["gateway"] == f"127.0.0.1:{listener}"
    assert row["detail"] == ""
    assert row["probed_at"]


async def test_closed_port_reports_unreachable_with_detail():
    port = _closed_port()
    prober = _prober(_va_config(port), targets=(VA,))

    await prober.sweep_once()
    row = prober.snapshot()[VA]["read_only"]

    assert row["endpoint_tcp"] == STATUS_UNREACHABLE
    assert row["gateway"] == f"127.0.0.1:{port}"
    # The detail names what went wrong, not merely that something did.
    assert row["detail"]
    assert row["detail"] != ""


async def test_addr_list_mode_is_not_applicable_and_never_connects(monkeypatch, listener):
    """UDP honesty: no socket is opened for an address-list gateway.

    The endpoint points at a port that *is* listening, so a probe that ran would
    return ``ok`` and the assertion on the status alone could pass for the wrong
    reason. The monkeypatched connector is what actually proves nothing ran.
    """

    async def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "asyncio.open_connection was called for a use_name_server: false "
            "gateway; CA search there is UDP and TCP proves nothing about it."
        )

    monkeypatch.setattr(asyncio, "open_connection", _forbidden)

    prober = _prober(_va_config(listener, use_name_server=False), targets=(VA,))
    await prober.sweep_once()
    row = prober.snapshot()[VA]["read_only"]

    assert row["endpoint_tcp"] == STATUS_NOT_APPLICABLE
    assert row["gateway"] == f"127.0.0.1:{listener}"
    assert "UDP" in row["detail"]


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


async def test_old_row_reads_stale_while_the_cache_keeps_the_real_status(listener):
    clock = FakeClock()
    prober = _prober(_va_config(listener), targets=(VA,), interval_s=10.0, monotonic=clock)

    await prober.sweep_once()
    assert prober.snapshot()[VA]["read_only"]["endpoint_tcp"] == STATUS_OK

    # Exactly at the threshold is not yet stale; past it, it is.
    clock.advance(prober.staleness_threshold_s)
    assert prober.snapshot()[VA]["read_only"]["endpoint_tcp"] == STATUS_OK

    clock.advance(1.0)
    row = prober.snapshot()[VA]["read_only"]
    assert row["endpoint_tcp"] == STATUS_STALE
    assert row["last_status"] == STATUS_OK


async def test_not_applicable_never_goes_stale(listener):
    """A configuration verdict has nothing to decay: it was never measured."""
    clock = FakeClock()
    prober = _prober(
        _va_config(listener, use_name_server=False),
        targets=(VA,),
        interval_s=10.0,
        monotonic=clock,
    )

    await prober.sweep_once()
    clock.advance(prober.staleness_threshold_s * 100)

    assert prober.snapshot()[VA]["read_only"]["endpoint_tcp"] == STATUS_NOT_APPLICABLE


async def test_staleness_uses_the_configured_interval():
    prober = EndpointProber({"control_system": {"target_switch": {"probe_interval_s": 4}}})

    assert prober.probe_interval_s == 4.0
    assert prober.staleness_threshold_s == 12.0


async def test_missing_or_unusable_interval_falls_back_to_the_default():
    assert EndpointProber({}).probe_interval_s == ep.DEFAULT_PROBE_INTERVAL_S
    assert (
        EndpointProber(
            {"control_system": {"target_switch": {"probe_interval_s": "nonsense"}}}
        ).probe_interval_s
        == ep.DEFAULT_PROBE_INTERVAL_S
    )
    assert (
        EndpointProber(
            {"control_system": {"target_switch": {"probe_interval_s": 0}}}
        ).probe_interval_s
        == ep.DEFAULT_PROBE_INTERVAL_S
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_first_sweep_runs_immediately_not_after_an_interval(listener):
    """A long interval must not leave the roster row-less at startup."""
    prober = _prober(_va_config(listener), targets=(VA,), interval_s=1000.0)

    await prober.start()
    try:
        await asyncio.wait_for(prober.first_sweep_done.wait(), timeout=5.0)
        assert prober.snapshot()[VA]["read_only"]["endpoint_tcp"] == STATUS_OK
    finally:
        await prober.stop()


async def test_stop_cancels_the_task_promptly(listener):
    prober = _prober(_va_config(listener), targets=(VA,), interval_s=1000.0)

    await prober.start()
    await asyncio.wait_for(prober.first_sweep_done.wait(), timeout=5.0)
    task = prober._task
    assert task is not None

    await asyncio.wait_for(prober.stop(), timeout=5.0)

    assert task.done()
    assert prober.running is False


async def test_start_is_idempotent_and_stop_tolerates_never_started(listener):
    prober = _prober(_va_config(listener), targets=(VA,), interval_s=1000.0)

    await prober.stop()  # never started

    await prober.start()
    first = prober._task
    await prober.start()
    assert prober._task is first

    await prober.stop()
    await prober.stop()


async def test_loop_keeps_sweeping_after_the_first(listener):
    clock = FakeClock()
    prober = _prober(_va_config(listener), targets=(VA,), interval_s=0.01, monotonic=clock)

    await prober.start()
    try:
        await asyncio.wait_for(prober.first_sweep_done.wait(), timeout=5.0)
        first = prober.snapshot()[VA]["read_only"]["probed_at"]
        clock.advance(1.0)

        async def _refreshed() -> None:
            while prober.snapshot()[VA]["read_only"]["probed_at"] == first:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_refreshed(), timeout=5.0)
    finally:
        await prober.stop()


# ---------------------------------------------------------------------------
# Derivation failure and snapshot isolation
# ---------------------------------------------------------------------------


async def test_underivable_target_is_absent_and_the_other_target_survives(listener):
    """``live`` cannot be derived here — there is no non-simulated block."""
    prober = _prober(_va_config(listener), targets=(LIVE, VA))

    await prober.sweep_once()
    snapshot = prober.snapshot()

    assert LIVE not in snapshot
    assert snapshot[VA]["read_only"]["endpoint_tcp"] == STATUS_OK


async def test_target_with_no_gateways_is_absent(listener):
    config = _va_config(listener)
    config["control_system"]["connector"]["virtual_accelerator"]["gateways"] = {}
    prober = _prober(config, targets=(VA,))

    await prober.sweep_once()

    assert prober.snapshot() == {}


async def test_a_target_that_becomes_derivable_fills_in_on_a_later_sweep(listener):
    """Derivation is redone every sweep, so no restart is needed to recover."""
    config = _both_targets_config(listener, listener)
    removed = config["control_system"]["connector"].pop("epics")

    prober = _prober(config, targets=(LIVE, VA))
    await prober.sweep_once()
    assert LIVE not in prober.snapshot()

    config["control_system"]["connector"]["epics"] = removed
    await prober.sweep_once()
    assert prober.snapshot()[LIVE]["read_only"]["endpoint_tcp"] == STATUS_OK


async def test_both_targets_are_probed_in_one_sweep(listener):
    closed = _closed_port()
    prober = _prober(_both_targets_config(listener, closed), targets=(LIVE, VA))

    await prober.sweep_once()
    snapshot = prober.snapshot()

    assert snapshot[VA]["read_only"]["endpoint_tcp"] == STATUS_OK
    assert snapshot[LIVE]["read_only"]["endpoint_tcp"] == STATUS_UNREACHABLE


async def test_snapshot_is_isolated_from_the_cache(listener):
    prober = _prober(_va_config(listener), targets=(VA,))
    await prober.sweep_once()

    snapshot = prober.snapshot()
    snapshot[VA]["read_only"]["endpoint_tcp"] = "tampered"
    snapshot[VA]["injected"] = {"endpoint_tcp": "invented"}
    snapshot["ghost"] = {}

    fresh = prober.snapshot()
    assert fresh[VA]["read_only"]["endpoint_tcp"] == STATUS_OK
    assert set(fresh) == {VA}
    assert set(fresh[VA]) == {"read_only"}


# ---------------------------------------------------------------------------
# Reachability publication (FR-19)
# ---------------------------------------------------------------------------
#
# The prober is the only process that measures these endpoints and the chip that
# renders them lives in another one, so the loop publishes every sweep to the
# state file. Unconditionally: a reader ages the rows, and an unchanged row that
# simply stops being republished is indistinguishable from a prober that died.
#
# What is published is what was MEASURED. ``stale`` is a read-time verdict drawn
# from ``probed_at`` by whoever reads, never a state the prober writes down.


@pytest.fixture
def published_state(tmp_path, monkeypatch):
    """A real state record under a tmp root, so a publish can be read back.

    Stamping ``OSPREY_AGENT_DATA_ROOT`` is how a spawned session anchors the
    directory, and it is the rule that wins in ``target_state.state_dir``, so a
    test that stamps it exercises the path a deployed server actually takes.
    """
    monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, str(tmp_path))
    target_state.write_on_start(VA, {VA: {"label": "Virtual accelerator"}})
    return tmp_path


async def test_every_sweep_of_the_loop_publishes(listener, monkeypatch):
    """Three sweeps, three publishes, each measured later than the last."""
    published: list[dict[str, Any]] = []

    def _record(rows: Any, **kwargs: Any) -> bool:
        published.append(copy.deepcopy(rows))
        return True

    monkeypatch.setattr(ep.target_state, "publish_reachability", _record)

    prober = _prober(_va_config(listener), targets=(VA,), interval_s=0.01)
    await prober.start()
    try:

        async def _three() -> None:
            while len(published) < 3:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_three(), timeout=5.0)
    finally:
        await prober.stop()

    stamps = [
        datetime.fromisoformat(sweep[VA]["read_only"]["probed_at"]) for sweep in published[:3]
    ]
    assert stamps[0] < stamps[1] < stamps[2]


async def test_published_rows_speak_the_readers_vocabulary(listener):
    """``ok``/``unreachable`` are the prober's words; the file carries FR-5's."""
    closed = _closed_port()
    prober = _prober(_both_targets_config(listener, closed), targets=(LIVE, VA))

    await prober.sweep_once()
    rows = prober.reachability_rows()

    assert rows[VA]["read_only"]["state"] == STATE_REACHED
    assert rows[LIVE]["read_only"]["state"] == STATE_DOWN


async def test_not_applicable_is_published_as_itself(listener):
    """A verdict from configuration travels intact; collapsing it invents a failure."""
    prober = _prober(_va_config(listener, use_name_server=False), targets=(VA,))

    await prober.sweep_once()
    row = prober.reachability_rows()[VA]["read_only"]

    assert row["state"] == STATUS_NOT_APPLICABLE
    assert "UDP" in row["detail"]


async def test_stale_is_never_published(listener):
    """Age is the signal; the reader draws the conclusion from ``probed_at``."""
    clock = FakeClock()
    prober = _prober(_va_config(listener), targets=(VA,), interval_s=10.0, monotonic=clock)

    await prober.sweep_once()
    clock.advance(prober.staleness_threshold_s * 10)

    assert prober.snapshot()[VA]["read_only"]["endpoint_tcp"] == STATUS_STALE
    assert prober.reachability_rows()[VA]["read_only"]["state"] == STATE_REACHED


async def test_published_rows_carry_the_gateway_they_were_measured_at(listener):
    """A row that named no endpoint could be read as a claim about another one."""
    prober = _prober(_va_config(listener), targets=(VA,))

    await prober.sweep_once()
    row = prober.reachability_rows()[VA]["read_only"]

    assert row["gateway"] == f"127.0.0.1:{listener}"
    datetime.fromisoformat(row["probed_at"])


async def test_the_real_publisher_accepts_what_the_prober_produces(listener, published_state):
    """End to end through ``target_state``: the file is read back, not mocked."""
    prober = _prober(_va_config(listener), targets=(VA,), interval_s=1000.0)

    await prober.start()
    try:
        await asyncio.wait_for(prober.first_sweep_done.wait(), timeout=5.0)
    finally:
        await prober.stop()

    block = target_state.read()["reachability"]
    row = block["targets"][VA]["read_only"]

    assert row["state"] == STATE_REACHED
    assert row["gateway"] == f"127.0.0.1:{listener}"
    datetime.fromisoformat(block["published_at"])
    age_s = (datetime.now(UTC) - datetime.fromisoformat(row["probed_at"])).total_seconds()
    assert 0 <= age_s < 60


async def test_a_failed_publish_does_not_stop_the_loop(listener, monkeypatch, caplog):
    """An unwritable state directory costs a publish, never the measurements."""
    calls: list[int] = []

    def _boom(rows: Any, **kwargs: Any) -> bool:
        calls.append(1)
        raise OSError("state directory is read-only")

    monkeypatch.setattr(ep.target_state, "publish_reachability", _boom)

    prober = _prober(_va_config(listener), targets=(VA,), interval_s=0.01)

    with caplog.at_level("WARNING", logger=ep.logger.name):
        await prober.start()
        try:

            async def _twice() -> None:
                while len(calls) < 2:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(_twice(), timeout=5.0)
            assert prober.running is True
            assert prober.snapshot()[VA]["read_only"]["endpoint_tcp"] == STATUS_OK
        finally:
            await prober.stop()

    assert any("reachability" in record.getMessage().lower() for record in caplog.records)


async def test_an_empty_sweep_publishes_an_empty_block(listener):
    """A target that stops deriving must not leave its last row standing."""
    config = _va_config(listener)
    config["control_system"]["connector"]["virtual_accelerator"]["gateways"] = {}
    prober = _prober(config, targets=(VA,))

    await prober.sweep_once()

    assert prober.reachability_rows() == {}


async def test_reachability_rows_are_isolated_from_the_cache(listener):
    """The publisher passes rows through verbatim, so they must not alias state."""
    prober = _prober(_va_config(listener), targets=(VA,))
    await prober.sweep_once()

    rows = prober.reachability_rows()
    rows[VA]["read_only"]["state"] = "tampered"

    assert prober.reachability_rows()[VA]["read_only"]["state"] == STATE_REACHED
