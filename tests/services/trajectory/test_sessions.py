"""Sessions on the action layer: open in the worker, take-over and release on the control path.

The gate is the real :class:`AdmitGate` on a real control-context tree and the
run lock is the real :func:`~osprey.runtime.guarded_run.lock` under a
throwaway deployment root; the connector is a stand-in whose injection-flag
read answers or fails on demand. The limits validator is stubbed, as in the
action layer's own tests.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from tests._guarded_run import reset_run_state

from osprey.runtime.guarded_run import ENV_CONTROL_IDENTITY
from osprey.services.trajectory import actions
from osprey.services.trajectory.actions import ACTION_STOPPED, ActionLayer, ActionRefused
from osprey.services.trajectory.admit import NARROWING, Admission, AdmitGate, AdmitRefused
from osprey.services.trajectory.sessions import (
    BAD_REQUEST,
    NO_SESSION,
    NOT_SESSION_HOLDER,
    OWNER_IN_BODY,
    SESSION_HELD,
    SESSION_TAKEN_OVER,
    SessionRefused,
    Sessions,
    check_body,
)
from osprey.services.trajectory.startup import TrajectoryCommands, TrajectoryStartup
from osprey_connectors import control_context, posture_store
from osprey_connectors.errors import ChannelReadFailedError
from osprey_connectors.identity import AUDIT_IDENTITY_ENV

ALICE = "alice"
BOB = "bob"
PANEL = "panel"
AGENT = "agent"
TARGET = "va"
COMMANDS = TrajectoryCommands(
    model="SR", arm=("SR:BPM:ARM",), injection_request="SR:INJ:REQ", injection_busy="SR:INJ:BUSY"
)


class _Connector:
    """Answers every read with *outcome* (raised when it is an exception)."""

    def __init__(self) -> None:
        self.outcome: Any = 0
        self.reads = 0
        self._limits_validator: Any = None

    async def read_channel(self, address: str, timeout: float | None = None) -> Any:  # noqa: ARG002
        self.reads += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Validator:
    failsafe_reason = None


@pytest.fixture(autouse=True)
def _deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_CONTEXT_DIR_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        ENV_CONTROL_IDENTITY,
        AUDIT_IDENTITY_ENV,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "deploy"
    (root / "build").mkdir(parents=True)
    (root / "build" / "config.yml").write_text("control_system: {}\n", encoding="utf-8")
    monkeypatch.chdir(root)
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()
    reset_run_state(monkeypatch)
    tree = root / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True, exist_ok=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    posture_store.invalidate_cache()
    monkeypatch.setattr(
        actions.LimitsValidator, "from_config", staticmethod(lambda **_kw: _Validator())
    )
    yield root
    posture_store.invalidate_cache()
    reset_config_cache()


def _narrow(owner: str) -> None:
    tree = Path(os.environ[posture_store.CONTROL_CONTEXT_TREE_ENV_VAR])
    control_context.write_record(
        control_context.ControlContext(target=TARGET, generation=1, posture={TARGET: "sandbox"}),
        path=tree / owner / control_context.RECORD_FILENAME,
    )
    posture_store.invalidate_cache()


def _sessions() -> tuple[Sessions, _Connector]:
    connector = _Connector()
    startup = TrajectoryStartup(
        target=TARGET, connector=connector, commands=COMMANDS, guarded_run_dir=Path("/unused")
    )
    layer = ActionLayer(startup, AdmitGate(target=TARGET, section={"writes_enabled": True}))
    return Sessions(layer), connector


async def _session_refused(coro: Any) -> SessionRefused:
    with pytest.raises(SessionRefused) as caught:
        await coro
    return caught.value


async def _noop(_admission: Admission, _stop: asyncio.Event) -> str:
    return "ran"


# -- open ----------------------------------------------------------------------


async def test_open_installs_a_holder_and_answers_client_owner_and_target() -> None:
    sessions, connector = _sessions()
    try:
        grant = await sessions.open(ALICE, PANEL)
    finally:
        await sessions.layer.close()
    assert grant.as_body() == {"client_id": grant.client_id, "owner": ALICE, "target": TARGET}
    assert grant.client_id
    assert sessions.holder is not None
    assert (sessions.holder.owner, sessions.holder.kind) == (ALICE, PANEL)
    assert sessions.holder.stop is sessions.layer.stop_event
    # Open is a worker action: it probed the target.
    assert connector.reads == 1


async def test_a_second_open_is_session_held_naming_the_door_and_owner() -> None:
    sessions, _ = _sessions()
    try:
        await sessions.open(ALICE, PANEL)
        refusal = await _session_refused(sessions.open(BOB, AGENT))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (SESSION_HELD, 409)
    assert repr(ALICE) in refusal.detail and repr(PANEL) in refusal.detail
    assert sessions.holder is not None and sessions.holder.owner == ALICE


async def test_two_queued_opens_leave_one_holder_the_second_refused_in_the_worker() -> None:
    sessions, _ = _sessions()
    try:
        first, second = await asyncio.gather(
            sessions.open(ALICE, PANEL), sessions.open(BOB, AGENT), return_exceptions=True
        )
    finally:
        await sessions.layer.close()
    assert not isinstance(first, BaseException)
    assert isinstance(second, SessionRefused) and second.code == SESSION_HELD
    assert sessions.holder is not None and sessions.holder.client_id == first.client_id


async def test_open_on_an_unreachable_target_is_refused_by_the_worker() -> None:
    sessions, connector = _sessions()
    connector.outcome = ChannelReadFailedError("SR:INJ:BUSY", "down")
    try:
        with pytest.raises(ActionRefused):
            await sessions.open(ALICE, PANEL)
    finally:
        await sessions.layer.close()
    assert sessions.holder is None


# -- request bodies -------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"owner": BOB}, OWNER_IN_BODY),
        ({"owner": BOB, "kind": AGENT}, OWNER_IN_BODY),
        ({"kind": AGENT}, BAD_REQUEST),
        ({"client_id": "x"}, BAD_REQUEST),
        ([1], BAD_REQUEST),
    ],
)
def test_any_body_key_is_400(body: Any, code: str) -> None:
    with pytest.raises(SessionRefused) as caught:
        check_body(body)
    assert (caught.value.code, caught.value.status) == (code, 400)


def test_an_empty_or_absent_body_passes() -> None:
    check_body(None)
    check_body({})


async def test_an_open_with_a_body_key_is_refused_before_the_worker() -> None:
    sessions, connector = _sessions()
    try:
        refusal = await _session_refused(sessions.open(ALICE, PANEL, {"owner": BOB}))
    finally:
        await sessions.layer.close()
    assert refusal.code == OWNER_IN_BODY
    assert connector.reads == 0
    assert sessions.holder is None


def test_take_over_and_release_refuse_body_keys() -> None:
    sessions, _ = _sessions()
    with pytest.raises(SessionRefused) as caught:
        sessions.take_over(BOB, AGENT, {"kind": PANEL})
    assert caught.value.code == BAD_REQUEST
    with pytest.raises(SessionRefused) as caught:
        sessions.release(ALICE, PANEL, "x", {"owner": ALICE})
    assert caught.value.code == OWNER_IN_BODY


# -- take-over ------------------------------------------------------------------


def test_take_over_with_no_holder_is_no_session() -> None:
    sessions, _ = _sessions()
    with pytest.raises(SessionRefused) as caught:
        sessions.take_over(BOB, AGENT)
    assert (caught.value.code, caught.value.status) == (NO_SESSION, 409)
    assert sessions.holder is None


async def test_take_over_installs_the_new_holder_with_a_clear_stop_event() -> None:
    sessions, _ = _sessions()
    try:
        old = await sessions.open(ALICE, PANEL)
        old_stop = sessions.holder.stop  # type: ignore[union-attr]
        grant = sessions.take_over(BOB, AGENT)
    finally:
        await sessions.layer.close()
    assert grant.owner == BOB and grant.client_id != old.client_id
    holder = sessions.holder
    assert holder is not None and (holder.owner, holder.kind) == (BOB, AGENT)
    assert old_stop.is_set()
    assert holder.stop is sessions.layer.stop_event and not holder.stop.is_set()


async def test_take_over_during_an_in_flight_action_raises_no_run_busy() -> None:
    sessions, _ = _sessions()
    started = asyncio.Event()

    async def runs_until_stopped(_admission: Admission, stop: asyncio.Event) -> str:
        started.set()
        await stop.wait()
        return "stopped"

    try:
        grant = await sessions.open(ALICE, PANEL)
        in_flight = asyncio.create_task(
            sessions.call(ALICE, PANEL, grant.client_id, runs_until_stopped)
        )
        await asyncio.wait_for(started.wait(), 2)
        new = sessions.take_over(BOB, AGENT)
        assert await asyncio.wait_for(in_flight, 2) == "stopped"
    finally:
        await sessions.layer.close()
    holder = sessions.holder
    assert holder is not None and holder.client_id == new.client_id and holder.owner == BOB
    assert not holder.stop.is_set()


async def test_take_over_of_an_unreachable_target_succeeds() -> None:
    sessions, connector = _sessions()
    try:
        await sessions.open(ALICE, PANEL)
        connector.outcome = ChannelReadFailedError("SR:INJ:BUSY", "down")
        reads = connector.reads
        grant = sessions.take_over(BOB, AGENT)
    finally:
        await sessions.layer.close()
    assert grant.owner == BOB
    assert connector.reads == reads


async def test_take_over_is_refused_by_admit_and_leaves_the_holder() -> None:
    sessions, _ = _sessions()
    try:
        await sessions.open(ALICE, PANEL)
        _narrow(BOB)
        with pytest.raises(AdmitRefused) as caught:
            sessions.take_over(BOB, AGENT)
    finally:
        await sessions.layer.close()
    assert caught.value.code == NARROWING
    assert sessions.holder is not None and sessions.holder.owner == ALICE
    assert not sessions.holder.stop.is_set()


async def test_the_voided_clients_queued_and_later_calls_are_session_taken_over() -> None:
    sessions, _ = _sessions()
    release = asyncio.Event()
    ran: list[str] = []

    async def slow(_admission: Admission, _stop: asyncio.Event) -> None:
        await release.wait()

    async def queued(_admission: Admission, _stop: asyncio.Event) -> None:
        ran.append("queued")

    try:
        old = await sessions.open(ALICE, PANEL)
        new = sessions.take_over(BOB, AGENT)
        # Hold the worker with the new holder's call, queue one for the voided client.
        first = asyncio.create_task(sessions.call(BOB, AGENT, new.client_id, slow))
        await asyncio.sleep(0.02)
        second = asyncio.create_task(sessions.call(ALICE, PANEL, old.client_id, queued))
        await asyncio.sleep(0.02)
        release.set()
        await first
        queued_refusal = await _session_refused(second)
        later_refusal = await _session_refused(sessions.call(ALICE, PANEL, old.client_id, queued))
        assert await sessions.call(BOB, AGENT, new.client_id, _noop) == "ran"
    finally:
        await sessions.layer.close()
    assert (queued_refusal.code, queued_refusal.status) == (SESSION_TAKEN_OVER, 409)
    assert later_refusal.code == SESSION_TAKEN_OVER
    assert ran == []


async def test_a_call_queued_before_a_take_over_is_refused_one_way_or_another() -> None:
    sessions, _ = _sessions()
    release = asyncio.Event()

    async def slow(_admission: Admission, _stop: asyncio.Event) -> None:
        await release.wait()

    try:
        old = await sessions.open(ALICE, PANEL)
        first = asyncio.create_task(sessions.call(ALICE, PANEL, old.client_id, slow))
        await asyncio.sleep(0.02)
        second = asyncio.create_task(sessions.call(ALICE, PANEL, old.client_id, _noop))
        await asyncio.sleep(0.02)
        sessions.take_over(BOB, AGENT)
        release.set()
        await first
        with pytest.raises(ActionRefused) as caught:
            await second
    finally:
        await sessions.layer.close()
    # Bound to the old holder's stop event, which the take-over set.
    assert caught.value.code == ACTION_STOPPED


# -- calls ----------------------------------------------------------------------


async def test_the_holders_call_runs() -> None:
    sessions, _ = _sessions()
    try:
        grant = await sessions.open(ALICE, PANEL)
        assert await sessions.call(ALICE, PANEL, grant.client_id, _noop) == "ran"
    finally:
        await sessions.layer.close()


@pytest.mark.parametrize(
    ("owner", "kind", "client"),
    [
        (ALICE, PANEL, "unknown"),
        (ALICE, PANEL, None),
        (BOB, PANEL, "grant"),
        (ALICE, AGENT, "grant"),
    ],
)
async def test_a_call_from_a_non_holder_or_unknown_client_is_not_session_holder(
    owner: str, kind: str, client: str | None
) -> None:
    sessions, _ = _sessions()
    try:
        grant = await sessions.open(ALICE, PANEL)
        client_id = grant.client_id if client == "grant" else client
        refusal = await _session_refused(sessions.call(owner, kind, client_id, _noop))
    finally:
        await sessions.layer.close()
    assert (refusal.code, refusal.status) == (NOT_SESSION_HOLDER, 409)


async def test_a_call_with_no_holder_is_not_session_holder() -> None:
    sessions, _ = _sessions()
    try:
        refusal = await _session_refused(sessions.call(ALICE, PANEL, "anything", _noop))
    finally:
        await sessions.layer.close()
    assert refusal.code == NOT_SESSION_HOLDER


# -- release --------------------------------------------------------------------


async def test_release_drops_the_holder_and_answers_its_record() -> None:
    sessions, _ = _sessions()
    try:
        grant = await sessions.open(ALICE, PANEL)
        record = sessions.release(ALICE, PANEL, grant.client_id)
        again = await sessions.open(BOB, AGENT)
    finally:
        await sessions.layer.close()
    assert record.as_record() == {"owner": ALICE, "target": TARGET, "client_id": grant.client_id}
    assert sessions.holder is not None and sessions.holder.client_id == again.client_id


async def test_a_narrowed_holder_can_release() -> None:
    sessions, _ = _sessions()
    try:
        grant = await sessions.open(ALICE, PANEL)
        _narrow(ALICE)
        sessions.release(ALICE, PANEL, grant.client_id)
    finally:
        await sessions.layer.close()
    assert sessions.holder is None


async def test_the_holder_of_an_unreachable_target_can_release() -> None:
    sessions, connector = _sessions()
    try:
        grant = await sessions.open(ALICE, PANEL)
        connector.outcome = ChannelReadFailedError("SR:INJ:BUSY", "down")
        reads = connector.reads
        sessions.release(ALICE, PANEL, grant.client_id)
    finally:
        await sessions.layer.close()
    assert sessions.holder is None
    assert connector.reads == reads


@pytest.mark.parametrize(("owner", "client"), [(BOB, "grant"), (ALICE, "unknown"), (ALICE, None)])
async def test_release_by_a_non_holder_or_unknown_client_is_not_session_holder(
    owner: str, client: str | None
) -> None:
    sessions, _ = _sessions()
    try:
        grant = await sessions.open(ALICE, PANEL)
        with pytest.raises(SessionRefused) as caught:
            sessions.release(owner, PANEL, grant.client_id if client == "grant" else client)
    finally:
        await sessions.layer.close()
    assert (caught.value.code, caught.value.status) == (NOT_SESSION_HOLDER, 409)
    assert sessions.holder is not None and sessions.holder.owner == ALICE


def test_release_with_no_holder_is_not_session_holder() -> None:
    sessions, _ = _sessions()
    with pytest.raises(SessionRefused) as caught:
        sessions.release(ALICE, PANEL, "anything")
    assert caught.value.code == NOT_SESSION_HOLDER


# -- state ----------------------------------------------------------------------


async def test_state_names_the_holder_and_never_the_client_id() -> None:
    sessions, _ = _sessions()
    assert sessions.state() == {"target": TARGET, "owner": None, "kind": None}
    try:
        grant = await sessions.open(ALICE, PANEL)
    finally:
        await sessions.layer.close()
    state = sessions.state()
    assert state == {"target": TARGET, "owner": ALICE, "kind": PANEL}
    assert grant.client_id not in repr(state)
