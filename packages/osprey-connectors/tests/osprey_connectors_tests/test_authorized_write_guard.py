"""The write guard under a program's write authorization, in process.

A process that names an authorization (``OSPREY_WRITE_AUTHORIZATION``) has
every write call judged by
:func:`~osprey_connectors.write_authorization.authorized_execution_verdict` on
its channels and values, which spends the call's one step before anything is
put. These tests drive the guard of :class:`ControlSystemConnector` with
recording connectors: what is written, what is refused and why (the typed
``verdict_reason``), how many steps a call spends, and that a process naming
no authorization is judged exactly as before.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors import control_context, posture_store, session_hold
from osprey_connectors import write_authorization as wa
from osprey_connectors.control_system._base import (
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from osprey_connectors.write_authorization import (
    Approver,
    ChannelRange,
    Grant,
    _issue_grant,
    _parse_ledger,
    revoke_grants,
    start_lease,
)
from osprey_connectors_tests._harness import isolate_verdict_environment, write_control_context
from osprey_connectors_tests._harness._control_context_fixtures import FIXTURE_IDENTITY
from osprey_connectors_tests._write_fakes import RecordingConnector, config_reader, turn_switch

SECTION = {"type": "epics", "writes": "allowed"}
TARGET = "va"
ME = FIXTURE_IDENTITY
CHANNELS = {"A:SP": ChannelRange(0.0, 10.0), "B:SP": ChannelRange(-1.0, 1.0)}
FAR_FUTURE_NS = time.time_ns() + 12 * 3600 * 10**9


class BaseBatchConnector(ControlSystemConnector):
    """A connector with no batch of its own: the base class's sequential loop runs."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, Any]] = []

    async def connect(self, config: dict[str, Any]) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        raise NotImplementedError

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        raise NotImplementedError

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,  # noqa: ARG002 - the connector interface fixes this signature
        confirm: bool | None = None,  # noqa: ARG002 - the connector interface fixes this signature
    ) -> ChannelWriteResult:
        if channel_address == "LIMITED:SP":
            raise ChannelLimitsViolationError(
                channel_address=channel_address,
                value=value,
                violation_type="MAX_EXCEEDED",
                violation_reason="above the limits database's max",
            )
        self.writes.append((channel_address, value))
        return ChannelWriteResult(
            channel_address=channel_address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )

    async def subscribe(self, channel_address, callback, *, on_connection=None):
        raise NotImplementedError

    async def unsubscribe(self, subscription_id):
        raise NotImplementedError

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address) -> bool:  # noqa: ARG002 - the connector interface fixes this signature
        return True


class StrayBatchConnector(RecordingConnector):
    """A batch of its own that also writes a channel the caller never asked for."""

    async def write_multiple_channels(self, operations, timeout=None):  # noqa: ARG002 - the connector interface fixes this signature
        results = [await self.write_channel(addr, val) for addr, val in operations]
        results.append(await self.write_channel("A:SP", 1.0))
        return results


class DelegatingBatchConnector(BaseBatchConnector):
    """A batch of its own that hands the whole batch to the base class's loop."""

    async def write_multiple_channels(self, operations, timeout=None, confirm=None):
        return await super().write_multiple_channels(operations, timeout=timeout, confirm=confirm)


class ClosedRouteConnector(BaseBatchConnector):
    """A connector whose selected gateway cannot carry a write (EPICS before a respawn)."""

    def _write_route_open(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This login's host record (every switch off), the config served, no grant named."""
    data = tmp_path / "root"
    isolate_verdict_environment(monkeypatch, data)
    for name in (wa.AUTHORIZATION_ENV_VAR, wa.LEDGER_DIR_ENV_VAR, "OSPREY_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr("osprey_connectors.config.get_config_value", config_reader(SECTION))
    write_control_context(data, switch={})
    return data


def _epoch() -> int:
    control_context.invalidate_cache()
    record = control_context.read_record()
    assert record is not None and isinstance(record.switch_epoch, int)
    return record.switch_epoch


def approve(monkeypatch: pytest.MonkeyPatch, *, name: bool = True, **fields: Any) -> Grant:
    """Issue a grant to this login, start this process's lease, and name it in the environment."""
    now = time.time_ns()
    values: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "owner": ME,
        "holder": ME,
        "target": TARGET,
        "switch_epoch": _epoch(),
        "channels": CHANNELS,
        "steps": 3,
        "issued_at_ns": now,
        "expires_at_ns": FAR_FUTURE_NS,
        "lease_s": 60.0,
        "approver": Approver(identity="alice", surface="chip", at_ns=now),
        "label": "run-1",
        "revoked": None,
    }
    values.update(fields)
    grant = Grant(**values)
    _issue_grant(grant)
    start_lease(grant.id)
    if name:
        monkeypatch.setenv(wa.AUTHORIZATION_ENV_VAR, grant.id)
    return grant


def steps(grant: Grant) -> int:
    directory = wa.ledger_dir()
    assert directory is not None
    ledger = _parse_ledger((directory / f"{grant.id}.ledger.json").read_text())
    return len(ledger.steps)


def connector(cls: type[ControlSystemConnector] = BaseBatchConnector) -> Any:
    built = cls()
    built._control_target = TARGET
    return built


def reasons(results: list[ChannelWriteResult]) -> list[str | None]:
    return [result.verdict_reason for result in results]


# -- a process naming no authorization ---------------------------------------


class TestUnnamed:
    async def test_the_switch_off_refuses_with_its_typed_reason(self, monkeypatch):
        approve(monkeypatch, name=False)
        conn = connector()
        result = await conn.write_channel("A:SP", 1.0)
        assert (result.outcome, result.verdict_reason) == (WriteOutcome.REFUSED, "switch_off")
        assert conn.writes == []

    async def test_another_process_under_the_same_login_gains_nothing(self, monkeypatch):
        grant = approve(monkeypatch)
        assert (await connector().write_channel("A:SP", 1.0)).outcome is WriteOutcome.CONFIRMED
        monkeypatch.delenv(wa.AUTHORIZATION_ENV_VAR)
        other = connector()
        refused = await other.write_channel("A:SP", 1.0)
        assert refused.verdict_reason == "switch_off"
        assert other.writes == []
        assert steps(grant) == 1

    async def test_the_base_batch_is_judged_per_channel_as_before(self, monkeypatch):
        turn_switch(TARGET, "asked")
        judged: list[Any] = []
        real = posture_store.write_verdict

        def counting(*args: Any, **kwargs: Any) -> Any:
            judged.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(posture_store, "write_verdict", counting)
        conn = connector()
        results = await conn.write_multiple_channels([("A:SP", 1.0), ("C:SP", 99.0)])
        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED] * 2
        assert conn.writes == [("A:SP", 1.0), ("C:SP", 99.0)]
        assert len(judged) == 2

    async def test_a_refusal_raised_by_the_checked_write_carries_the_reason(self):
        with pytest.raises(ChannelWriteBlockedError) as caught:
            await connector().write_channel_checked("A:SP", 1.0)
        assert caught.value.verdict_reason == "switch_off"
        assert caught.value.reason == "WRITES_DISABLED"


# -- one write call -----------------------------------------------------------


class TestSingleWrites:
    async def test_a_listed_channel_in_range_is_written_with_the_switch_off(self, monkeypatch):
        grant = approve(monkeypatch)
        conn = connector()
        result = await conn.write_channel("A:SP", 5.0)
        assert result.outcome is WriteOutcome.CONFIRMED
        assert conn.writes == [("A:SP", 5.0)]
        assert steps(grant) == 1

    @pytest.mark.parametrize(
        ("channel", "value", "reason"),
        [("A:SP", 10.5, "auth_range"), ("C:SP", 1.0, "auth_channel")],
        ids=["out-of-range", "unlisted"],
    )
    async def test_a_write_outside_the_bounds_writes_nothing(
        self, monkeypatch, channel, value, reason
    ):
        grant = approve(monkeypatch)
        conn = connector()
        result = await conn.write_channel(channel, value)
        assert (result.outcome, result.verdict_reason) == (WriteOutcome.REFUSED, reason)
        assert f"[osprey-refusal: {reason}]" in (result.error_message or "")
        assert conn.writes == []
        assert steps(grant) == 0

    async def test_the_budget_ends_the_writes(self, monkeypatch):
        grant = approve(monkeypatch, steps=2)
        conn = connector()
        outcomes = [(await conn.write_channel("A:SP", float(n))).verdict_reason for n in range(3)]
        assert outcomes == [None, None, "auth_exhausted"]
        assert conn.writes == [("A:SP", 0.0), ("A:SP", 1.0)]
        assert steps(grant) == 2
        with pytest.raises(ChannelWriteBlockedError) as caught:
            await conn.write_channel_checked("A:SP", 1.0)
        assert caught.value.verdict_reason == "auth_exhausted"

    async def test_a_checked_write_is_one_step(self, monkeypatch):
        grant = approve(monkeypatch)
        await connector().write_channel_checked("A:SP", 1.0)
        assert steps(grant) == 1

    async def test_a_write_the_limits_refuse_has_spent_its_step(self, monkeypatch):
        grant = approve(monkeypatch, channels={"LIMITED:SP": ChannelRange(0.0, 10.0)})
        conn = connector()
        with pytest.raises(ChannelLimitsViolationError):
            await conn.write_channel("LIMITED:SP", 5.0)
        assert conn.writes == []
        assert steps(grant) == 1

    @pytest.mark.parametrize("position", ["asked", "allowed"])
    async def test_the_switch_turned_on_does_not_lift_the_bounds(self, monkeypatch, position):
        grant = approve(monkeypatch)
        turn_switch(TARGET, position)
        conn = connector()
        assert (await conn.write_channel("C:SP", 1.0)).verdict_reason == "auth_channel"
        assert (await conn.write_channel("A:SP", 11.0)).verdict_reason == "auth_range"
        assert (await conn.write_channel("A:SP", 1.0)).outcome is WriteOutcome.CONFIRMED
        assert conn.writes == [("A:SP", 1.0)]
        assert steps(grant) == 1

    @pytest.mark.parametrize("gesture", ["named", "all"])
    async def test_a_stop_refuses_the_next_write(self, monkeypatch, gesture):
        grant = approve(monkeypatch)
        conn = connector()
        assert (await conn.write_channel("A:SP", 1.0)).outcome is WriteOutcome.CONFIRMED
        if gesture == "named":
            revoke_grants(by="alice", targets=[TARGET])
        else:
            control_context.set_switch("all", "off")
            control_context.invalidate_cache()
        refused = await conn.write_channel("A:SP", 2.0)
        assert refused.verdict_reason == "auth_revoked"
        assert conn.writes == [("A:SP", 1.0)]
        assert steps(grant) == 1

    async def test_a_hold_refuses_first_and_spends_nothing(self, monkeypatch, tmp_path):
        grant = approve(monkeypatch)
        monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(tmp_path / "holds"))
        monkeypatch.setenv("OSPREY_SESSION_ID", "s1")
        session_hold.ensure_session("s1")
        session_hold.open_flag(
            "s1",
            flag_id="f1",
            source="watchdog:scope",
            reason="off the plan",
            prompt_ref="record:f1",
            prompt_excerpt="move it",
        )
        conn = connector()
        assert (await conn.write_channel("A:SP", 1.0)).verdict_reason == "session_held"
        assert conn.writes == []
        assert steps(grant) == 0

    async def test_a_tier_that_never_writes_is_not_lifted(self, monkeypatch):
        grant = approve(monkeypatch)
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value",
            config_reader({"type": "epics", "writes": "never"}),
        )
        assert (await connector().write_channel("A:SP", 1.0)).verdict_reason == "tier_never"
        assert steps(grant) == 0

    async def test_an_authorized_write_reads_the_record_once(self, monkeypatch):
        approve(monkeypatch)
        reads: list[Any] = []
        real = control_context.read_record_status

        def counting(*args: Any, **kwargs: Any) -> Any:
            reads.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(control_context, "read_record_status", counting)
        assert (await connector().write_channel("A:SP", 1.0)).outcome is WriteOutcome.CONFIRMED
        assert len(reads) == 1

    async def test_the_gateway_answer_names_the_authorization_without_spending(self, monkeypatch):
        grant = approve(monkeypatch)
        conn = connector()
        assert conn._writes_enabled is True
        monkeypatch.delenv(wa.AUTHORIZATION_ENV_VAR)
        assert conn._writes_enabled is False
        assert steps(grant) == 0


# -- one batch ----------------------------------------------------------------


class TestBatches:
    @pytest.mark.parametrize("cls", [BaseBatchConnector, RecordingConnector], ids=["base", "own"])
    async def test_a_batch_is_one_step(self, monkeypatch, cls):
        grant = approve(monkeypatch)
        conn = connector(cls)
        results = await conn.write_multiple_channels([("A:SP", 1.0), ("B:SP", 0.5)])
        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED] * 2
        assert conn.writes == [("A:SP", 1.0), ("B:SP", 0.5)]
        assert steps(grant) == 1

    @pytest.mark.parametrize("cls", [BaseBatchConnector, RecordingConnector], ids=["base", "own"])
    @pytest.mark.parametrize(
        ("bad", "reason"),
        [(("B:SP", 2.0), "auth_range"), (("C:SP", 0.0), "auth_channel")],
        ids=["out-of-range", "unlisted"],
    )
    async def test_one_bad_operation_refuses_the_batch_before_any_put(
        self, monkeypatch, cls, bad, reason
    ):
        grant = approve(monkeypatch)
        conn = connector(cls)
        results = await conn.write_multiple_channels([("A:SP", 1.0), bad])
        assert reasons(results) == [reason, reason]
        assert conn.writes == []
        assert steps(grant) == 0

    async def test_an_exhausted_budget_refuses_the_whole_batch(self, monkeypatch):
        grant = approve(monkeypatch, steps=1)
        conn = connector()
        await conn.write_channel("A:SP", 1.0)
        results = await conn.write_multiple_channels([("A:SP", 2.0), ("B:SP", 0.0)])
        assert reasons(results) == ["auth_exhausted", "auth_exhausted"]
        assert conn.writes == [("A:SP", 1.0)]
        assert steps(grant) == 1

    async def test_a_batch_of_its_own_writes_only_what_it_was_judged_on(self, monkeypatch):
        grant = approve(monkeypatch, steps=1)
        conn = connector(StrayBatchConnector)
        results = await conn.write_multiple_channels([("B:SP", 0.0)])
        # Under an authorization the base class's batch runs: the stray write
        # the connector's own batch would add is never made.
        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED]
        assert conn.writes == [("B:SP", 0.0)]
        assert steps(grant) == 1

    async def test_a_batch_of_its_own_runs_as_before_with_no_authorization(self):
        turn_switch(TARGET, "allowed")
        conn = connector(StrayBatchConnector)
        results = await conn.write_multiple_channels([("B:SP", 0.0)])
        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED] * 2
        assert conn.writes == [("B:SP", 0.0), ("A:SP", 1.0)]

    async def test_a_batch_delegating_to_super_is_still_one_step(self, monkeypatch):
        grant = approve(monkeypatch)
        conn = connector(DelegatingBatchConnector)
        results = await conn.write_multiple_channels([("A:SP", 1.0), ("B:SP", 0.5)])
        assert [r.outcome for r in results] == [WriteOutcome.CONFIRMED] * 2
        assert conn.writes == [("A:SP", 1.0), ("B:SP", 0.5)]
        assert steps(grant) == 1

    async def test_a_limits_refusal_inside_a_batch_is_still_one_step(self, monkeypatch):
        grant = approve(
            monkeypatch,
            channels={"A:SP": ChannelRange(0.0, 10.0), "LIMITED:SP": ChannelRange(0.0, 10.0)},
        )
        conn = connector()
        with pytest.raises(ChannelLimitsViolationError):
            await conn.write_multiple_channels([("A:SP", 1.0), ("LIMITED:SP", 1.0)])
        assert conn.writes == [("A:SP", 1.0)]
        assert steps(grant) == 1


# -- a connector whose gateway cannot carry the write ------------------------


class TestClosedRoute:
    async def test_an_allowed_write_is_refused_switching_and_spends_nothing(self, monkeypatch):
        grant = approve(monkeypatch)
        conn = connector(ClosedRouteConnector)
        result = await conn.write_channel("A:SP", 1.0)
        assert (result.outcome, result.verdict_reason) == (WriteOutcome.REFUSED, "switching")
        batch = await conn.write_multiple_channels([("A:SP", 1.0), ("B:SP", 0.0)])
        assert reasons(batch) == ["switching", "switching"]
        assert conn.writes == []
        assert steps(grant) == 0

    async def test_a_write_the_authorization_refuses_names_its_own_reason(self, monkeypatch):
        approve(monkeypatch)
        result = await connector(ClosedRouteConnector).write_channel("A:SP", 99.0)
        assert result.verdict_reason == "auth_range"
