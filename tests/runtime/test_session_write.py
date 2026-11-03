"""ControlSession.write: WC1's order and every WC6 server row, one case each.

Each case opens a real :class:`~osprey.runtime.session.ControlSession` over a
real control-context record in ``tmp_path``, with the fake connector registered
under the virtual accelerator's type. A case that reaches the connector replaces
``write_channel`` on the class, which also bypasses the guard wrapper, so the
connector's answer is exactly the one the case names. Audit records land in a
redirected ledger and are read back as JSON lines.

Common to every case: the write's entry and its in-flight marker are gone when
``write()`` returns (the done-callback ran), and the write filed exactly the
records WC7 orders — one for a refusal by the session, ``attempt`` then the
outcome for anything that reached the connector.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from osprey.mcp_server.control_system import target_eligibility, target_state
from osprey.runtime import session as session_module
from osprey.runtime.budget import WriteBudget
from osprey.runtime.session import ControlSession
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.control_system import _base as connector_base
from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
from osprey_connectors.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._harness._control_context_fixtures import _set_switch_to
from tests._audit_capture import capture_audit_bodies

#: The virtual accelerator with the write tier allowed: with its switch on (the
#: record :func:`write_record` writes by default) the write verdict allows.
VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "writes": "allowed",
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}
#: The section the refusal-wording tests serve: the same tier-allowed render, so
#: a refusal is worded from the verdict's own reason.
ARMED_SECTION = VA_SECTION
#: The same machine on a ``never`` tier (no ``writes`` key): the verdict refuses
#: every write by this login before the connector is reached.
NEVER_SECTION = {key: value for key, value in VA_SECTION.items() if key != "writes"}


def phrase(reason: str) -> str:
    """The words a refusal on the simulator is answered in."""
    return refusal_phrase(reason, "simulator")


STANDIN_ONLY_SECTION = {"type": types.LIVE_STANDIN, "connector": {types.LIVE_STANDIN: {}}}

CHANNEL = "SR:QF:01:SP"
ORIGIN = {"screen": "sr/overview", "rev": "r7"}

#: T_b = 0.2, S_b = 0.1, so D_b = 0.5; with the margin shrunk to 0.1 a write is
#: answered unknown 0.6 s after admission.
SMALL_BUDGET = WriteBudget(T=0.2, S=0.1, D=0.5, grace_s=2.5, relay_timeout_s=3.5)
MARGIN_S = 0.1


class _Connector(FakeConnector):
    """The bound connector; each case replaces ``write_channel``."""

    disconnects = 0

    async def disconnect(self):
        type(self).disconnects += 1
        await super().disconnect()


def _serve(monkeypatch, section):
    def get_config_value(path, default=None, _config_path=None):
        return section if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)


def write_record(state_dir: Path, *, target="va", generation=3, switch=None) -> None:
    """Write the record and make its switch exactly *switch* (default: *target* on)."""
    path = state_dir / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(target=target, generation=generation), path=path
    )
    control_context.invalidate_cache()
    if switch is None:
        wanted = dict.fromkeys({target} & set(types.CONTROL_TARGETS), "allowed")
    elif isinstance(switch, Mapping):
        wanted = dict(switch)
    else:
        wanted = dict.fromkeys(switch, "allowed")
    _set_switch_to(path, wanted)
    control_context.invalidate_cache()


@pytest.fixture
def audit_dir(monkeypatch):
    """The panel audit bodies this test stores in the agent record, in order."""
    return capture_audit_bodies(monkeypatch, "panel")


@pytest.fixture
def state_dir(tmp_path, monkeypatch, audit_dir):  # noqa: ARG001 - captures the audit first
    directory = tmp_path / "control_target"
    directory.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    for name in list(os.environ):
        if name.startswith("OSPREY_CONTROL_TARGET") or name.startswith("EPICS_"):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "_MARGIN_S", MARGIN_S)
    monkeypatch.setattr(connector_base, "_long_lived_session", False)
    control_context.invalidate_cache()
    yield directory
    control_context.invalidate_cache()


@pytest.fixture
def va(monkeypatch, state_dir):
    _serve(monkeypatch, VA_SECTION)
    _Connector.disconnects = 0
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, _Connector)
        yield state_dir


def connector_answers(monkeypatch, behaviour):
    """Make the bound connector's ``write_channel`` run *behaviour*; return its calls."""
    calls: list[tuple] = []

    async def write_channel(_self, address, value, **kwargs):
        calls.append((address, value, kwargs))
        return await behaviour(address, value)

    monkeypatch.setattr(_Connector, "write_channel", write_channel)
    return calls


def returns(outcome, **fields):
    async def behaviour(address, value):
        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=outcome, **fields
        )

    return behaviour


def raises(exc):
    async def behaviour(address, value):  # noqa: ARG001 - the behaviour signature
        raise exc

    return behaviour


def records(audit_dir: list[dict]) -> list[dict]:
    return list(audit_dir)


def decisions(audit_dir: list[dict]) -> list[tuple[str, str]]:
    return [(r["decision"], r["reason"]) for r in records(audit_dir)]


def markers(state_dir: Path) -> list[Path]:
    return sorted(state_dir.glob(target_state.INFLIGHT_FILE_GLOB))


async def opened(**kwargs) -> ControlSession:
    return await ControlSession.open(surface="panel", budget=SMALL_BUDGET, **kwargs)


def assert_released(session: ControlSession, state_dir: Path) -> None:
    assert session._writes == {}
    assert markers(state_dir) == []


# -- the connector rows (6-14) -----------------------------------------------

#: (row, connector behaviour, outcome, reason, message or None for "check below").
CONNECTOR_ROWS = [
    (
        "6-returned",
        returns(
            WriteOutcome.REFUSED,
            refusal_reason="WRITES_DISABLED",
            error_message="writes are disabled for this tier",
        ),
        "refused",
        "WRITES_DISABLED",
        "writes are disabled for this tier",
    ),
    (
        "6-raised",
        raises(ChannelWriteBlockedError(CHANNEL, "WRITES_DISABLED", message=" chip says no ")),
        "refused",
        "WRITES_DISABLED",
        "chip says no",
    ),
    (
        "7-raised",
        raises(
            ChannelLimitsViolationError(CHANNEL, 9.0, "MAX_VALUE", "9.0 is above the limit 5.0")
        ),
        "refused",
        "LIMITS",
        "9.0 is above the limit 5.0",
    ),
    (
        "7-returned",
        returns(WriteOutcome.REFUSED, refusal_reason="LIMITS", error_message="step too large"),
        "refused",
        "LIMITS",
        "step too large",
    ),
    (
        "8",
        returns(
            WriteOutcome.REFUSED,
            refusal_reason="VALIDATION_ERROR",
            error_message="limits could not be checked",
        ),
        "refused",
        "VALIDATION_ERROR",
        "limits could not be checked",
    ),
    (
        "9",
        returns(
            WriteOutcome.REFUSED,
            refusal_reason="CONTROL_SYSTEM_REFUSED",
            error_message="IOC access security refused",
        ),
        "refused",
        "CONTROL_SYSTEM_REFUSED",
        "IOC access security refused",
    ),
    (
        "10",
        returns(WriteOutcome.CONFIRMED, observed_value=1.5),
        "confirmed",
        "write_landed",
        "written",
    ),
    (
        "11",
        returns(WriteOutcome.MISMATCH, observed_value=1.25),
        "mismatch",
        "write_unconfirmed",
        "written; the channel holds 1.25",
    ),
    (
        "12-unconfirmed",
        returns(WriteOutcome.UNCONFIRMED, error_message="no ack"),
        "unconfirmed",
        "write_unconfirmed",
        "sent, not verified",
    ),
    (
        "12-unrequested",
        returns(WriteOutcome.UNREQUESTED),
        "unconfirmed",
        "write_unconfirmed",
        "sent, not verified",
    ),
    (
        "13",
        returns(WriteOutcome.FAILED, error_message="put failed"),
        "unknown",
        "write_unconfirmed",
        "outcome unknown — check the readback",
    ),
    (
        "14-raised",
        raises(ConnectionError("gateway gone")),
        "unknown",
        "write_unconfirmed",
        "outcome unknown — check the readback",
    ),
    (
        "14-failed-no-outcome",
        raises(ChannelWriteFailedError(CHANNEL, "FAILED")),
        "unknown",
        "write_unconfirmed",
        "outcome unknown — check the readback",
    ),
]


class TestConnectorRows:
    @pytest.mark.parametrize(
        ("row", "behaviour", "outcome", "reason", "message"),
        CONNECTOR_ROWS,
        ids=[case[0] for case in CONNECTOR_ROWS],
    )
    async def test_row(self, va, audit_dir, monkeypatch, row, behaviour, outcome, reason, message):
        calls = connector_answers(monkeypatch, behaviour)
        write_record(va)
        session = await opened()

        result = await session.write(CHANNEL, 1.5, origin=ORIGIN)

        assert (result.outcome, result.reason, result.message) == (outcome, reason, message), row
        expected_decision = "refused" if outcome == "refused" else "allowed"
        assert decisions(audit_dir) == [
            ("allowed", "write_attempt"),
            (expected_decision, reason),
        ]
        assert len(calls) == 1
        assert_released(session, va)
        await session.close()

    async def test_the_connector_call_is_bounded_by_t_b(self, va, monkeypatch):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()

        await session.write(CHANNEL, 1.0)
        await session.write(CHANNEL, 2.0, timeout=0.05)
        await session.write(CHANNEL, 3.0, timeout=30.0)

        assert [kwargs["timeout"] for _a, _v, kwargs in calls] == [0.2, 0.05, 0.2]
        await session.close()

    async def test_a_mismatch_names_the_whole_observed_value(self, va, monkeypatch):
        connector_answers(monkeypatch, returns(WriteOutcome.MISMATCH, observed_value="x" * 400))
        write_record(va)
        session = await opened()

        result = await session.write(CHANNEL, "y")

        assert result.message == "written; the channel holds " + "x" * 400
        assert result.observed_value == "x" * 400
        await session.close()

    async def test_an_empty_osprey_message_is_left_empty_for_the_adapter(
        self, va, audit_dir, monkeypatch
    ):
        connector_answers(
            monkeypatch, raises(ChannelLimitsViolationError(CHANNEL, 9.0, "MAX_VALUE", "   "))
        )
        write_record(va)
        session = await opened()

        result = await session.write(CHANNEL, 9.0)

        assert (result.outcome, result.reason, result.message) == ("refused", "LIMITS", "")
        assert decisions(audit_dir)[-1] == ("refused", "LIMITS")
        await session.close()

    async def test_a_never_tier_is_refused_by_the_verdict_before_the_connector(
        self, va, audit_dir, monkeypatch
    ):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        _serve(monkeypatch, NEVER_SECTION)
        write_record(va)
        session = await opened()

        result = await session.write(CHANNEL, 1.5, origin=ORIGIN)

        assert (result.outcome, result.reason) == ("refused", "tier_never")
        assert result.message == phrase("tier_never")
        assert decisions(audit_dir) == [("refused", "tier_never")]
        assert calls == []
        await session.close()

    async def test_the_monitors_own_refusal_is_relayed_whole_and_logged(
        self, va, audit_dir, monkeypatch, caplog
    ):
        reasons = "writes are off for the 'va' control target " + "— " * 200 + "config.yml is not"
        connector_answers(
            monkeypatch,
            raises(ChannelWriteBlockedError(CHANNEL, "WRITES_DISABLED", message=reasons)),
        )
        write_record(va)
        session = await opened()
        caplog.set_level(logging.INFO, logger="control_session")

        result = await session.write(CHANNEL, 1.5, origin=ORIGIN)

        assert (result.outcome, result.reason) == ("refused", "WRITES_DISABLED")
        assert result.message == reasons
        assert "refused by the reference monitor" in caplog.text
        assert decisions(audit_dir) == [
            ("allowed", "write_attempt"),
            ("refused", "WRITES_DISABLED"),
        ]
        await session.close()

    @pytest.mark.parametrize(
        "behaviour",
        [
            raises(ChannelWriteBlockedError(CHANNEL, "VALIDATION_ERROR", message="v" * 2000)),
            raises(ChannelWriteBlockedError(CHANNEL, "CONTROL_SYSTEM_REFUSED", message="c" * 2000)),
            raises(ChannelWriteBlockedError(CHANNEL, "LIMITS", message="l" * 2000)),
            raises(ChannelLimitsViolationError(CHANNEL, 9.0, "MAX_VALUE", "m" * 2000)),
            returns(
                WriteOutcome.REFUSED,
                refusal_reason="CONTROL_SYSTEM_REFUSED",
                error_message="r" * 2000,
            ),
        ],
        ids=["validation", "control-system", "limits-blocked", "limits-raised", "returned"],
    )
    async def test_every_relayed_refusal_text_is_carried_whole(self, va, monkeypatch, behaviour):
        connector_answers(monkeypatch, behaviour)
        write_record(va)
        session = await opened()

        result = await session.write(CHANNEL, 1.5)

        assert result.outcome == "refused"
        assert result.message == result.message[0] * 2000
        await session.close()

    async def test_a_narrowed_target_sends_the_operator_to_the_chip(self, va, monkeypatch):
        _serve(monkeypatch, ARMED_SECTION)
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va, switch=())
        session = await opened()

        result = await session.write(CHANNEL, 1.5, origin=ORIGIN)

        assert (result.outcome, result.reason) == ("refused", "switch_off")
        assert result.message == phrase("switch_off")
        assert calls == []
        await session.close()

    async def test_an_unreadable_record_is_said_to_be_unreadable(self, va, monkeypatch):
        _serve(monkeypatch, ARMED_SECTION)
        connector_answers(
            monkeypatch, raises(ChannelWriteBlockedError(CHANNEL, "WRITES_DISABLED", message="no"))
        )

        def unreadable(_section, target, **_kwargs):
            # What the verdict answers for a record it cannot read; it never raises.
            return posture_store.WriteVerdict(
                allowed=False, reason="record_unavailable", target=target, kind="simulator"
            )

        monkeypatch.setattr(posture_store, "write_verdict", unreadable)
        write_record(va)
        session = await opened()

        result = await session.write(CHANNEL, 1.5, origin=ORIGIN)

        assert (result.outcome, result.reason) == ("refused", "record_unavailable")
        assert result.message == phrase("record_unavailable")
        await session.close()

    async def test_a_read_only_run_says_the_whole_deployment_is_read_only(self, va, monkeypatch):
        _serve(monkeypatch, ARMED_SECTION)
        connector_answers(
            monkeypatch, raises(ChannelWriteBlockedError(CHANNEL, "WRITES_DISABLED", message="no"))
        )
        write_record(va)
        # A read-only run from the start: the verdict refuses it at open and at
        # the write alike, so the connector's own refusal is worded from it.
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        session = await opened()

        result = await session.write(CHANNEL, 1.5, origin=ORIGIN)

        assert (result.outcome, result.reason) == ("refused", "readonly_run")
        assert result.message == phrase("readonly_run")
        await session.close()

    async def test_a_refusal_the_session_cannot_account_for_carries_the_monitors_words(
        self, va, monkeypatch
    ):
        _serve(monkeypatch, ARMED_SECTION)
        connector_answers(
            monkeypatch, raises(ChannelWriteBlockedError(CHANNEL, "WRITES_DISABLED", message="no"))
        )
        write_record(va)
        session = await opened()

        result = await session.write(CHANNEL, 1.5, origin=ORIGIN)

        assert (result.outcome, result.reason, result.message) == (
            "refused",
            "WRITES_DISABLED",
            "no",
        )
        await session.close()

    async def test_a_held_session_is_refused_in_the_holds_words(
        self, va, audit_dir, monkeypatch, tmp_path
    ):
        from osprey_connectors import session_hold

        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        root = tmp_path / "holds"
        monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
        monkeypatch.setenv(session_module.SESSION_KEY_ENV_VAR, "alice-1")
        write_record(va)
        session = await opened()
        assert session_hold.open_flag(
            "alice-1",
            flag_id="flag-1",
            source="watchdog:limits",
            reason="a limit was refused",
            prompt_ref="turn-7",
            prompt_excerpt="set the current to 900",
            root=root,
        )

        result = await session.write(CHANNEL, 1.5, origin=ORIGIN)

        assert (result.outcome, result.reason) == ("refused", "session_held")
        assert result.message == phrase("session_held") + " Raised by: watchdog:limits."
        assert session.writes["reason"] == "session_held"
        assert decisions(audit_dir) == [("refused", "session_held")]
        assert calls == []
        await session.close()


# -- the session's own rows (3-5c) -------------------------------------------


class TestSessionRows:
    """Every refusal before the connector: one record, no connector call, nothing left."""

    async def _refused(self, session, state_dir, audit_dir, calls, outcome, reason, message):
        result = await session.write(CHANNEL, 1.0, origin=ORIGIN)

        assert (result.outcome, result.reason, result.message) == (outcome, reason, message)
        assert decisions(audit_dir) == [
            ("refused", reason),
        ]
        assert calls == []
        assert_released(session, state_dir)

    async def test_row_3_writes_stopped(self, va, audit_dir, monkeypatch):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()
        session.stop_writes()

        await self._refused(
            session, va, audit_dir, calls, "target_changed", "switching", phrase("switching")
        )
        await session.close()

    @pytest.mark.parametrize(("target", "generation"), [("va", 4), ("live", 3)])
    async def test_row_3_record_moved(self, va, audit_dir, monkeypatch, target, generation):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()
        write_record(va, target=target, generation=generation)

        await self._refused(
            session, va, audit_dir, calls, "target_changed", "switching", phrase("switching")
        )
        await session.close()

    async def test_row_3a_switch_turned_off_is_refused_in_the_verdicts_words(
        self, va, audit_dir, monkeypatch
    ):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()
        write_record(va, switch=())

        await self._refused(
            session, va, audit_dir, calls, "refused", "switch_off", phrase("switch_off")
        )
        await session.close()

    @pytest.mark.parametrize(
        ("opened_at", "now"),
        [("allowed", "asked"), ("asked", "allowed")],
        ids=["to-asked", "to-allowed"],
    )
    async def test_row_3a_position_moved_is_a_moved_binding(
        self, va, audit_dir, monkeypatch, opened_at, now
    ):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va, switch={"va": opened_at})
        session = await opened()
        assert session.writes["position"] == opened_at
        write_record(va, switch={"va": now})

        await self._refused(
            session, va, audit_dir, calls, "target_changed", "switching", phrase("switching")
        )
        assert session.writes["position"] == now
        await session.close()

    async def test_row_4_record_gone(self, va, audit_dir, monkeypatch):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()
        (va / control_context.RECORD_FILENAME).unlink()
        control_context.invalidate_cache()

        await self._refused(
            session,
            va,
            audit_dir,
            calls,
            "refused",
            "control_context_unavailable",
            "no control target record; nothing written",
        )
        await session.close()

    async def test_row_4_fallback(self, va, audit_dir, monkeypatch):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        session = await opened()
        assert session.binding.state == "fallback"

        await self._refused(
            session,
            va,
            audit_dir,
            calls,
            "refused",
            "control_context_unavailable",
            "no control target record; nothing written",
        )
        await session.close()

    async def test_row_4b_unservable(self, state_dir, audit_dir, monkeypatch):
        _serve(monkeypatch, STANDIN_ONLY_SECTION)
        write_record(state_dir, target="live")
        session = await opened()
        assert session.binding.state == "unservable"

        await self._refused(
            session,
            state_dir,
            audit_dir,
            [],
            "refused",
            "control_context_unavailable",
            "this control target cannot be served here; nothing written",
        )
        await session.close()

    async def test_row_5_switch_in_progress(self, va, audit_dir, monkeypatch):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        seen = []

        def blocking_pids(record, _reports, session_key):
            seen.append((record.target, record.generation, session_key))
            return (4242,)

        monkeypatch.setattr(control_context, "blocking_pids", blocking_pids)
        write_record(va)
        session = await opened()

        await self._refused(
            session, va, audit_dir, calls, "switch_in_progress", "switching", phrase("switching")
        )
        assert seen == [("va", 3, None)]
        await session.close()

    async def test_row_5a_attempt_not_stored(self, va, audit_dir, monkeypatch):
        from osprey.audit import writer

        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        real_record = writer.record

        def record(**fields):
            if fields.get("reason") == "write_attempt":
                return None
            return real_record(**fields)

        monkeypatch.setattr(writer, "record", record)
        write_record(va)
        session = await opened()

        await self._refused(
            session,
            va,
            audit_dir,
            calls,
            "refused",
            "audit_unavailable",
            "the audit ledger is unavailable; nothing written",
        )
        await session.close()

    async def test_row_5c_marker_not_written(self, va, audit_dir, monkeypatch, tmp_path):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()
        not_a_directory = tmp_path / "not-a-directory"
        not_a_directory.write_text("", encoding="utf-8")
        monkeypatch.setattr(target_state, "state_dir", lambda: not_a_directory)

        await self._refused(
            session,
            va,
            audit_dir,
            calls,
            "refused",
            "record_unavailable",
            phrase("record_unavailable"),
        )
        await session.close()

    async def test_row_5c_switch_gate_raises(self, va, audit_dir, monkeypatch):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))

        def blocking_pids(_record, _reports, _session_key):
            raise OSError("reports unreadable")

        monkeypatch.setattr(control_context, "blocking_pids", blocking_pids)
        write_record(va)
        session = await opened()

        await self._refused(
            session,
            va,
            audit_dir,
            calls,
            "refused",
            "record_unavailable",
            phrase("record_unavailable"),
        )
        await session.close()

    async def test_a_defect_before_the_attempt_record_is_a_refusal(
        self, va, audit_dir, monkeypatch
    ):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()

        def pin_verdict(_record, _key):
            raise RuntimeError("a defect in the pin check")

        monkeypatch.setattr(session_module, "pin_verdict", pin_verdict)

        await self._refused(
            session,
            va,
            audit_dir,
            calls,
            "refused",
            "record_unavailable",
            phrase("record_unavailable"),
        )
        await session.close()

    async def test_a_failed_marker_rename_leaves_no_temp_file(self, va, monkeypatch):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()

        def replace(_src, _dst):
            raise OSError("rename refused")

        with monkeypatch.context() as patch:
            patch.setattr(session_module.os, "replace", replace)
            result = await session.write(CHANNEL, 1.0, origin=ORIGIN)

        assert (result.outcome, result.reason) == ("refused", "record_unavailable")
        assert calls == []
        assert list(va.glob("*.tmp")) == []
        await session.close()


# -- one write at a time per channel (5b), the deadline (14), drain ----------


class _Held:
    """A connector write held in a worker thread until released."""

    def __init__(self, outcome=WriteOutcome.CONFIRMED):
        self.release = threading.Event()
        self.started = threading.Event()
        self.returned = threading.Event()
        self.outcome = outcome

    async def __call__(self, address, value):
        def hold():
            self.started.set()
            self.release.wait(10)

        try:
            await asyncio.to_thread(hold)
        finally:
            self.returned.set()
        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=self.outcome
        )


async def _until(predicate, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class TestInFlight:
    async def test_row_5b_a_busy_channel_is_refused_at_once(self, va, audit_dir, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        first = asyncio.ensure_future(session.write(f" {CHANNEL} ", 1.0))
        await _until(held.started.is_set)

        result = await asyncio.wait_for(session.write(CHANNEL, 2.0), timeout=0.1)

        assert (result.outcome, result.reason, result.message) == (
            "refused",
            "write_in_progress",
            "a write to this channel is still in progress; nothing written",
        )
        assert len(markers(va)) == 1  # the first write's, nothing for the refused one
        held.release.set()
        assert (await first).outcome == "confirmed"
        assert decisions(audit_dir) == [
            ("allowed", "write_attempt"),
            ("refused", "write_in_progress"),
            ("allowed", "write_landed"),
        ]
        assert_released(session, va)
        await session.close()

    async def test_another_channel_is_not_busy(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        first = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)
        held.release.set()

        other = await session.write("SR:QF:02:SP", 2.0)

        assert other.outcome == "confirmed"
        assert (await first).outcome == "confirmed"
        await session.close()

    async def test_the_marker_names_its_client(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        pending = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)

        (path,) = markers(va)
        marker = json.loads(path.read_text())
        assert path.name.startswith(f"exec_inflight_{os.getpid()}_")
        assert marker["surface"] == "panel"
        assert marker["pid"] == os.getpid()
        assert marker["target"] == "va"
        assert target_eligibility.busy_client(marker) == "a panel write"
        assert target_state.in_flight_executions() == [marker]

        held.release.set()
        await pending
        await session.close()

    async def test_a_caller_names_its_own_client(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened(client_label="a display write")
        pending = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)

        (path,) = markers(va)
        marker = json.loads(path.read_text())
        assert target_eligibility.busy_client(marker) == "a display write"

        held.release.set()
        await pending
        await session.close()

    async def test_row_14_deadline_overrun_keeps_the_marker_until_the_thread_returns(
        self, va, audit_dir, monkeypatch
    ):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        loop = asyncio.get_running_loop()
        started = loop.time()

        result = await session.write(CHANNEL, 1.0)

        waited = loop.time() - started
        assert (result.outcome, result.reason, result.message) == (
            "unknown",
            "write_unconfirmed",
            "outcome unknown — check the readback",
        )
        assert 0.5 <= waited < 0.5 + MARGIN_S + 0.4
        assert decisions(audit_dir) == [
            ("allowed", "write_attempt"),
            ("allowed", "write_unconfirmed"),
        ]
        # Still running: the marker and the entry stand, so the channel stays busy.
        assert len(markers(va)) == 1
        assert CHANNEL in session._writes
        busy = await session.write(CHANNEL, 2.0)
        assert busy.reason == "write_in_progress"
        (entry,) = session._writes.values()

        held.release.set()
        await entry.task

        assert held.returned.is_set()
        assert_released(session, va)
        # The late completion is logged, not re-audited.
        assert decisions(audit_dir) == [
            ("allowed", "write_attempt"),
            ("allowed", "write_unconfirmed"),
            ("refused", "write_in_progress"),
        ]
        await session.close()

    async def test_a_cancelled_waiter_does_not_cancel_the_write(self, va, audit_dir, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        waiter = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)
        (entry,) = session._writes.values()

        waiter.cancel()
        await asyncio.sleep(0.01)
        assert not entry.task.done()
        held.release.set()
        result = await entry.task

        assert result.outcome == "confirmed"
        assert decisions(audit_dir)[-1] == ("allowed", "write_landed")
        assert_released(session, va)
        await session.close()

    async def test_a_cancelled_task_leaves_its_marker(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        waiter = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)
        (entry,) = session._writes.values()

        entry.task.cancel()  # process teardown, never this module
        with pytest.raises(asyncio.CancelledError):
            await entry.task
        await waiter

        assert session._writes == {}
        assert len(markers(va)) == 1
        held.release.set()
        await _until(held.returned.is_set)
        await session.close()


class TestDrainAndClose:
    async def test_drain_waits_for_a_write_that_ends_in_time(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        pending = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)
        session.stop_writes()
        asyncio.get_running_loop().call_later(0.1, held.release.set)

        assert await session.drain(5.0) is True
        assert (await pending).outcome == "confirmed"
        assert_released(session, va)
        await session.close()

    async def test_drain_fails_when_its_own_timeout_runs_out(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        pending = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)
        session.stop_writes()

        assert await session.drain(0.05) is False

        held.release.set()
        await pending
        await session.close()

    async def test_drain_stops_at_the_admission_deadline(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        loop = asyncio.get_running_loop()
        started = loop.time()
        pending = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)

        assert await session.drain(30.0) is False

        assert loop.time() - started < 0.5 + MARGIN_S + 0.4
        held.release.set()
        await pending
        (entry,) = session._writes.values() or [None]
        if entry is not None:
            await entry.task
        await session.close()

    async def test_close_drains_to_the_admission_deadline(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        pending = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)
        asyncio.get_running_loop().call_later(0.1, held.release.set)

        await session.close()

        assert held.returned.is_set()
        assert (await pending).outcome == "confirmed"
        assert _Connector.disconnects == 1
        assert_released(session, va)

    async def test_close_after_a_failed_drain_skips_disconnect(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        result = await session.write(CHANNEL, 1.0)  # answered unknown at the deadline
        assert result.outcome == "unknown"
        (entry,) = session._writes.values()

        await session.close()

        assert _Connector.disconnects == 0
        assert len(markers(va)) == 1
        refused = await session.write("SR:QF:02:SP", 1.0)
        assert refused.outcome == "target_changed"
        held.release.set()
        await entry.task
        assert_released(session, va)

    async def test_every_close_awaits_one_shielded_task(self, va, monkeypatch):
        held = _Held()
        connector_answers(monkeypatch, held)
        write_record(va)
        session = await opened()
        pending = asyncio.ensure_future(session.write(CHANNEL, 1.0))
        await _until(held.started.is_set)

        first = asyncio.ensure_future(session.close())
        await asyncio.sleep(0.01)
        first.cancel()
        second = asyncio.ensure_future(session.close())
        await asyncio.sleep(0.01)
        assert not session._close_task.done()
        held.release.set()
        await second

        assert session._close_task.done()
        assert _Connector.disconnects == 1
        assert (await pending).outcome == "confirmed"


# -- the caller's refusals and the audit record ------------------------------


class TestRefuseAndAudit:
    @pytest.mark.parametrize(
        ("reason", "outcome", "message"),
        [
            ("switch_off", "refused", phrase("switch_off")),
            ("session_held", "refused", phrase("session_held")),
            ("switching", "target_changed", phrase("switching")),
        ],
    )
    async def test_refuse_audits_the_callers_reason(
        self, va, audit_dir, monkeypatch, reason, outcome, message
    ):
        calls = connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()

        result = session.refuse(CHANNEL, 1.0, reason, origin=ORIGIN)

        assert (result.outcome, result.reason, result.message) == (outcome, reason, message)
        assert decisions(audit_dir) == [("refused", reason)]
        assert calls == []
        assert_released(session, va)
        await session.close()

    async def test_the_record_follows_wc7(self, va, audit_dir, monkeypatch):
        from osprey.audit.call import write_stamps

        connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()

        await session.write(CHANNEL, "v" * 300, origin=ORIGIN)

        attempt, landed = records(audit_dir)
        stamps = write_stamps()
        expected = (
            f"channel={CHANNEL} screen=sr/overview rev=r7 "
            f"ca_user={stamps.get('ca_user', '-')} ca_host={stamps.get('ca_host', '-')} "
            f'value_len=300 value="{"v" * 256}"'
        )
        for record in (attempt, landed):
            assert record["surface"] == "panel"
            assert record["session"] == f"panel:{os.getpid()}"
            assert record["subject"] == "panel_write"
            assert record["posture"] == "writes"
            assert record["detail"] == expected
        assert (attempt["decision"], attempt["reason"]) == ("allowed", "write_attempt")
        assert (landed["decision"], landed["reason"]) == ("allowed", "write_landed")
        await session.close()

    async def test_an_origin_the_request_did_not_name_is_a_dash(self, va, audit_dir, monkeypatch):
        connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
        write_record(va)
        session = await opened()

        await session.write(CHANNEL, 2)
        await session.write("SR:QF:02:SP", 3, origin={"screen": "sr/overview", "rev": None})

        details = [r["detail"] for r in records(audit_dir)]
        assert " screen=- rev=- " in details[0]
        assert " screen=sr/overview rev=- " in details[2]
        assert details[0].endswith('value_len=1 value="2"')
        await session.close()


async def test_long_identifiers_never_push_the_value_out_of_the_detail(va, audit_dir, monkeypatch):
    connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
    write_record(va)
    session = await opened()

    channel = "C" * 2000
    await session.write(channel, "v" * 300, origin={"screen": "s" * 2000, "rev": "r" * 2000})

    detail = records(audit_dir)[-1]["detail"]
    assert detail.startswith(f"channel={'C' * 256} screen={'s' * 128} rev={'r' * 128} ")
    assert detail.endswith(f'value_len=300 value="{"v" * 256}"')
    await session.close()


def parse_detail(detail: str) -> dict[str, str]:
    """A session-write ``detail`` read back: whitespace-split ``key=value`` tokens."""
    return dict(token.partition("=")[::2] for token in detail.split())


@pytest.mark.parametrize(
    "value",
    [
        "1 position=allowed",
        "a=b\tscreen=forged\nrev=x",
        'nbsp\u00a0line\u2028sep "quoted" \\back',
    ],
)
async def test_a_value_cannot_forge_a_detail_token(va, audit_dir, monkeypatch, value):
    import json

    connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
    write_record(va)
    session = await opened()

    await session.write(CHANNEL, value, origin=ORIGIN)

    for record in records(audit_dir):
        tokens = record["detail"].split()
        assert [t.partition("=")[0] for t in tokens] == [
            "channel",
            "screen",
            "rev",
            "ca_user",
            "ca_host",
            "value_len",
            "value",
        ]
        parsed = parse_detail(record["detail"])
        assert parsed["screen"] == "sr/overview"
        assert parsed["value_len"] == str(len(value))
        assert json.loads(parsed["value"]) == value
    await session.close()


async def test_an_escaped_value_is_cut_whole_inside_the_detail_bound(va, audit_dir, monkeypatch):
    import json

    from osprey.audit.envelope import MAX_DETAIL_CHARS

    connector_answers(monkeypatch, returns(WriteOutcome.CONFIRMED))
    write_record(va)
    session = await opened()

    value = "\u00e9 " * 400
    await session.write("C" * 2000, value, origin={"screen": "s" * 2000, "rev": "r" * 2000})

    detail = records(audit_dir)[-1]["detail"]
    assert len(detail) <= MAX_DETAIL_CHARS
    parsed = parse_detail(detail)
    assert parsed["value_len"] == "800"
    decoded = json.loads(parsed["value"])
    assert decoded and value.startswith(decoded)
    await session.close()
