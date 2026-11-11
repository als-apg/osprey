"""A program's write authorization over real Channel Access, through the pool.

Two soft IOCs serve the SAME record names with different seeds: one is the
deployment's ``read_only`` gateway and the other its ``write_access`` gateway,
so the IOC a write lands on says which gateway the child selected. The
machine's switch stays off throughout. The holder — this test process, which
holds the lease — asks the pool for a child under its authorization; another
child of the same target under the same login names none.

The IOCs are ``python -m epicscorelibs.ioc`` (see ``test_pool_soft_ioc.py``);
this process never loads a Channel Access client itself.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from typing import Any

import pytest
import yaml

from osprey_connectors import control_context
from osprey_connectors import write_authorization as wa
from osprey_connectors.control_system._base import WriteOutcome
from osprey_connectors.errors import ChannelWriteBlockedError
from osprey_connectors.ipc._pool import ConnectorHostPool
from osprey_connectors.write_authorization import (
    Approver,
    ChannelRange,
    Grant,
    _issue_grant,
    _parse_ledger,
    revoke_grants,
    start_lease,
)
from osprey_connectors_tests._harness._control_context_fixtures import (
    FIXTURE_IDENTITY,
    write_control_context,
)
from osprey_connectors_tests._paths import PACKAGE_SRC
from osprey_connectors_tests._write_fakes import arm_switch, turn_switch
from osprey_connectors_tests.ipc._soft_ioc import SoftIOC

#: Seeds far apart, so a value read from the wrong IOC is obvious.
SEED_READ = 1.0
SEED_WRITE = 100.0

CA_TIMEOUT_S = 5.0
FAR_FUTURE_NS = time.time_ns() + 12 * 3600 * 10**9


@pytest.fixture
def iocs(tmp_path, ioc_prefix):
    """The read gateway's IOC and the write gateway's, serving this test's prefix."""
    read = SoftIOC(tmp_path, "read", ioc_prefix, SEED_READ)
    try:
        write = SoftIOC(tmp_path, "write", ioc_prefix, SEED_WRITE)
    except BaseException:
        read.stop()
        raise
    yield read, write
    read.stop()
    write.stop()


def _section(read_port: int, write_port: int) -> dict[str, Any]:
    def endpoint(port: int) -> dict[str, Any]:
        return {"address": "127.0.0.1", "port": port, "use_name_server": False}

    return {
        "type": "epics",
        "writes": "allowed",
        "connector": {
            "epics": {
                "timeout_s": CA_TIMEOUT_S,
                "gateways": {
                    "read_only": endpoint(read_port),
                    "write_access": endpoint(write_port),
                },
            }
        },
    }


@pytest.fixture
def deployment(tmp_path, monkeypatch, iocs):
    """A pool over the two-gateway deployment, this login's switch off, and its grant writer."""
    read, write = iocs
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(PACKAGE_SRC))
    for name in (wa.AUTHORIZATION_ENV_VAR, wa.LEDGER_DIR_ENV_VAR, "OSPREY_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "agent-data"
    arm_switch(monkeypatch, root, None, stamp=False)
    write_control_context(root, target="live", owned_by=None, switch={})
    section = _section(read.port, write.port)
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump({"control_system": section}))
    return ConnectorHostPool(section, config_file=config_file)


def grant_for(channels: dict[str, ChannelRange], *, steps: int, lease: bool = True) -> Grant:
    """Approve *channels* for this login as holder; start this process's lease unless told not to."""
    control_context.invalidate_cache()
    record = control_context.read_record()
    assert record is not None and isinstance(record.switch_epoch, int)
    now = time.time_ns()
    grant = Grant(
        id=uuid.uuid4().hex,
        owner=FIXTURE_IDENTITY,
        holder=FIXTURE_IDENTITY,
        target="live",
        switch_epoch=record.switch_epoch,
        channels=channels,
        steps=steps,
        issued_at_ns=now,
        expires_at_ns=FAR_FUTURE_NS,
        lease_s=600.0,
        approver=Approver(identity="alice", surface="chip", at_ns=now),
        label="run-1",
    )
    _issue_grant(grant)
    if lease:
        start_lease(grant.id)
    return grant


def spent(grant: Grant) -> int:
    directory = wa.ledger_dir()
    assert directory is not None
    return len(_parse_ledger((directory / f"{grant.id}.ledger.json").read_text()).steps)


async def value(connector: Any, channel: str) -> Any:
    return (await connector.read_channel(channel, timeout=CA_TIMEOUT_S)).value


async def test_the_holder_writes_its_listed_channels_in_range_and_nothing_else(
    deployment, ioc_prefix
):
    sp, rb, unlisted = f"{ioc_prefix}:SP", f"{ioc_prefix}:RB", f"{ioc_prefix}:SP.DRVH"
    pool = deployment
    grant = grant_for({sp: ChannelRange(0.0, 50.0), rb: ChannelRange(0.0, 500.0)}, steps=3)
    try:
        held = await pool.connector("live", authorization=grant.id)
        other = await pool.connector("live")
        assert held.pid != other.pid
        assert held.authorization == grant.id
        assert (held.report["selected_role"], held.report["writes_enabled"]) == (
            "write_access",
            True,
        )
        assert held.report["authorization"] == grant.id
        assert (other.report["selected_role"], other.report["writes_enabled"]) == (
            "read_only",
            False,
        )
        assert other.report["authorization"] is None

        landed = await held.write_channel(sp, 5.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert landed.outcome is WriteOutcome.CONFIRMED
        assert spent(grant) == 1

        for channel, attempted, reason in (
            (sp, 60.0, "auth_range"),
            (unlisted, 1.0, "auth_channel"),
        ):
            refused = await held.write_channel(channel, attempted, confirm=True)
            assert (refused.outcome, refused.verdict_reason) == (WriteOutcome.REFUSED, reason)
        refused_batch = await held.write_multiple_channels([(sp, 6.0), (rb, 900.0)], confirm=True)
        assert [r.verdict_reason for r in refused_batch] == ["auth_range", "auth_range"]
        assert spent(grant) == 1

        batch = await held.write_multiple_channels([(sp, 6.0), (rb, 7.0)], confirm=True)
        assert [r.outcome for r in batch] == [WriteOutcome.CONFIRMED] * 2
        assert spent(grant) == 2

        await held.write_channel_checked(sp, 8.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert spent(grant) == 3
        with pytest.raises(ChannelWriteBlockedError) as caught:
            await held.write_channel_checked(sp, 9.0, confirm=True)
        assert caught.value.verdict_reason == "auth_exhausted"

        # Another process of this login, naming no authorization, is refused as before.
        not_held = await other.write_channel(sp, 9.0, confirm=True)
        assert not_held.verdict_reason == "switch_off"

        assert [await value(held, sp), await value(held, rb)] == [8.0, 7.0]
        assert await value(held, unlisted) == 0.0
        assert [await value(other, sp), await value(other, rb)] == [SEED_READ, SEED_READ]
        assert spent(grant) == 3
    finally:
        await pool.close()


async def test_the_switch_turned_on_does_not_lift_the_bounds_and_a_stop_ends_them(
    deployment, ioc_prefix
):
    sp = f"{ioc_prefix}:SP"
    pool = deployment
    grant = grant_for({sp: ChannelRange(0.0, 50.0)}, steps=10)
    try:
        held = await pool.connector("live", authorization=grant.id)
        turn_switch("live", "allowed")

        out = await held.write_channel(sp, 60.0, confirm=True)
        assert out.verdict_reason == "auth_range"
        inside = await held.write_channel(sp, 20.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert inside.outcome is WriteOutcome.CONFIRMED

        assert revoke_grants(by="alice", targets=["live"]) == (grant.id,)
        stopped = await held.write_channel(sp, 21.0, confirm=True)
        assert stopped.verdict_reason == "auth_revoked"
        assert await value(held, sp) == 20.0
        assert spent(grant) == 1
    finally:
        await pool.close()


async def test_a_child_replaced_mid_run_keeps_the_count(deployment, ioc_prefix):
    sp = f"{ioc_prefix}:SP"
    pool = deployment
    grant = grant_for({sp: ChannelRange(0.0, 50.0)}, steps=2)
    try:
        held = await pool.connector("live", authorization=grant.id)
        await held.write_channel_checked(sp, 1.5, confirm=True, timeout=CA_TIMEOUT_S)
        first = held.pid
        assert first is not None
        os.kill(first, signal.SIGKILL)
        deadline = time.monotonic() + 10.0
        while held.pid is not None and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        await held.write_channel_checked(sp, 2.5, confirm=True, timeout=CA_TIMEOUT_S)
        assert held.pid not in (None, first)
        exhausted = await held.write_channel(sp, 3.5, confirm=True)
        assert exhausted.verdict_reason == "auth_exhausted"
        assert await value(held, sp) == 2.5
        assert spent(grant) == 2
    finally:
        await pool.close()


async def test_a_handle_taken_before_the_lease_is_refused_switching_like_any_refusal(
    deployment, ioc_prefix
):
    """The key's child came up before it could write: its refusal spends nothing and is not resent."""
    sp = f"{ioc_prefix}:SP"
    pool = deployment
    grant = grant_for({sp: ChannelRange(0.0, 50.0)}, steps=1, lease=False)
    try:
        early = await pool.connector("live", authorization=grant.id)
        assert (early.report["selected_role"], early.report["writes_enabled"]) == (
            "read_only",
            False,
        )
        first = early.pid
        start_lease(grant.id)

        refused = await early.write_channel(sp, 4.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert (refused.outcome, refused.verdict_reason) == (WriteOutcome.REFUSED, "switching")
        assert early.pid == first
        assert spent(grant) == 0
    finally:
        await pool.close()


@pytest.mark.parametrize("stop", [["live"], "all"], ids=["named", "all"])
async def test_an_approved_request_ends_at_the_persons_stop(deployment, ioc_prefix, stop):
    """Request, approval, lease, a write; then the switch's own stop refuses the next one."""
    sp = f"{ioc_prefix}:SP"
    pool = deployment
    request = wa.write_request(
        "live",
        {sp: (0.0, 50.0)},
        steps=5,
        expires_at_ns=FAR_FUTURE_NS,
        lease_s=600.0,
        label="run-stop",
        completion="none",
    )
    grant = wa.approve_request(request, surface="cli")
    assert grant.id == request.id
    start_lease(grant.id)
    try:
        held = await pool.connector("live", authorization=grant.id)
        landed = await held.write_channel(sp, 11.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert landed.outcome is WriteOutcome.CONFIRMED

        control_context.set_switch(stop, "off")
        control_context.invalidate_cache()

        stopped = await held.write_channel(sp, 12.0, confirm=True)
        assert (stopped.outcome, stopped.verdict_reason) == (WriteOutcome.REFUSED, "auth_revoked")
        assert await value(held, sp) == 11.0
        assert spent(grant) == 1
    finally:
        await pool.close()
