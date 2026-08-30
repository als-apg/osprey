"""The session-control reconciler: two files on disk, one connector.

The reconciler is the controls server's answer to a surface that cannot call
it. The web server writes *desired* state — a per-(session, target) posture
entry, a switch request addressed to one server PID — and this task is what
turns either of them into something that has actually happened to the
connector, through exactly the functions the agent's own tool uses.

What is pinned here, and why each one matters:

* **The gate is asked immediately before the switch**, so a refusal reaches the
  operator in the tool's own words rather than in a second vocabulary invented
  for the button.
* **Every terminus publishes**, removes the request, and files one audit
  record. A request that vanished with nothing published would leave a chip
  spinning on ``switching…`` forever, and the request file is what the route
  reads to refuse a second click — so the two must never both be missing.
* **A request is addressed, not broadcast.** A body naming another server's PID
  is dropped without acting, and one older than the TTL ends as
  ``request_expired`` rather than moving a session minutes after the gesture.
* **A narrowing waits for a running execution** and says so while it waits: the
  realignment rebuilds the connector, and rebuilding it under a run would
  retire the child that run was promised.
* **The reconciler races the agent safely.** It goes through
  ``hosts.switch()`` — the same lock, the same equality guard — so two callers
  aiming at the same target spawn one child between them.

Everything below runs against fakes: no sockets, no child processes, no real
event-loop sleeping. The reconciler's ``poll_once`` is public precisely so a
test can drive the polls itself instead of waiting a second per assertion.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from osprey.audit import writer as audit_writer
from osprey.mcp_server.control_system import session_control, target_state
from osprey.mcp_server.control_system.connector_host_manager import SwitchError
from osprey.mcp_server.control_system.server_context import ControlSystemContext
from osprey.mcp_server.control_system.tools import control_target
from osprey_connectors import session_store

pytestmark = pytest.mark.unit

SESSION_KEY = "session-abc"


# --------------------------------------------------------------------- fakes


class FakeManager:
    """A connector-host supervisor with the parts the reconciler touches.

    The switch keeps the real manager's two load-bearing properties: it is
    serialized by a lock, and a switch whose destination is already active with
    a live child returns the settled result without spawning (the equality
    guard of task 3.2). Both are what the race test is about.
    """

    def __init__(self, target: str = "live", baseline: str = "live") -> None:
        self._target = target
        self.baseline = baseline
        self._generation = 0
        self._lock = asyncio.Lock()
        self.spawns = 0
        self.switch_calls: list[str] = []
        self.respawns = 0
        self.fail_with: Exception | None = None
        self.respawn_fails_with: Exception | None = None
        self.started = True
        #: Set to make ``switch`` block once the lock is held, so a test can
        #: put one caller *inside* the critical section and start the other.
        self.entered_switch = asyncio.Event()
        self.release_switch: asyncio.Event | None = None

    def active_target(self) -> str:
        return self._target

    def active_generation(self) -> int:
        return self._generation

    def is_started(self) -> bool:
        return self.started

    def has_child(self) -> bool:
        return True

    async def switch(self, target: str, *, force: bool = False) -> dict:
        async with self._lock:
            self.switch_calls.append(target)
            self.entered_switch.set()
            if self.release_switch is not None:
                await self.release_switch.wait()
            if self.fail_with is not None:
                raise self.fail_with
            if target == self._target and not force:
                return self._result(target, self._target, changed=False)
            previous, self._target = self._target, target
            self._generation += 1
            self.spawns += 1
            return self._result(target, previous, changed=True)

    async def respawn_same_target(self) -> dict:
        async with self._lock:
            self.respawns += 1
            if self.respawn_fails_with is not None:
                raise self.respawn_fails_with
            return self._result(self._target, self._target, changed=False)

    def _result(self, target: str, previous: str, *, changed: bool) -> dict:
        return {
            "target": target,
            "previous_target": previous,
            "generation": self._generation,
            "target_changed": changed,
            "connector_type": "mock",
            "selected_role": "read_only",
            "endpoint": None,
            "probe_channel": "PROBE:CHANNEL",
            "child_pid": 4242,
            "previous_drained": True,
            "drain_timeout_s": 1.0,
        }


def install_context(manager: FakeManager, monkeypatch, raw: dict | None = None):
    """Make *manager*'s deployment the server context the reconciler reads.

    The same shape ``test_control_target_set.install_context`` builds: a real
    ``ControlSystemContext`` with its two lazily-built members supplied, so
    ``invalidate_connector`` runs its real branch against the fake manager.
    """
    from osprey.mcp_server.control_system import server_context as server_context_mod

    context = ControlSystemContext()
    context._config = type("Config", (), {"raw": raw if raw is not None else {}})()
    context._connector_hosts = manager
    monkeypatch.setattr(server_context_mod, "_registry", context)
    return context


# ------------------------------------------------------------------ fixtures


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor both files under tmp_path, the way a session child is stamped.

    Through the environment stamp rather than by patching the root resolver:
    that stamp is the one resolution rule the state file, the posture store and
    the hooks all follow, so a test that patched around it would be testing a
    path no deployment takes.
    """
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    session_store.invalidate_cache()
    target_state.state_dir().mkdir(parents=True, exist_ok=True)
    yield tmp_path
    session_store.invalidate_cache()


@pytest.fixture(autouse=True)
def started_record(state_root):
    """The state record this server publishes into. Publishers merge, not create."""
    target_state.write_on_start("live", {"live": {"label": "Live"}, "va": {"label": "VA"}})
    return target_state.read()


@pytest.fixture
def emitted(monkeypatch):
    """Capture the operator-activity emissions instead of posting them."""
    calls: list[dict] = []

    async def record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(session_control, "notify_target_switch_async", record)
    return calls


@pytest.fixture
def records(monkeypatch):
    """Capture the audit records instead of appending them to a ledger."""
    written: list[dict] = []

    def record(**fields):
        written.append(fields)
        return None

    monkeypatch.setattr(audit_writer, "record", record)
    return written


@pytest.fixture
def allow_every_target(monkeypatch):
    """Stub eligibility open, so the gate's third check is not the subject."""
    from osprey.mcp_server.control_system.target_eligibility import TargetAvailability

    def available(config, target, session_target, baseline_target, **kwargs):
        return TargetAvailability(
            target=target,
            eligible=True,
            available_now=True,
            reason=None,
            detail=f"Target {target!r} is available (stubbed).",
            eligible_from_baseline=True,
        )

    monkeypatch.setattr(control_target, "target_availability", available)


# ------------------------------------------------------------------- helpers


def write_request(
    target: str = "va",
    *,
    server_pid: int | None = None,
    age_s: float = 0.0,
    requested_by: str = "operator",
    request_id: str | None = None,
) -> str:
    """Write a switch request the way the web server's route does."""
    rid = request_id or uuid.uuid4().hex
    created = datetime.now(UTC) - timedelta(seconds=age_s)
    target_state.write_request(
        {
            "request_id": rid,
            "target": target,
            "server_pid": os.getpid() if server_pid is None else server_pid,
            "created_at": created.isoformat(),
            "requested_by": requested_by,
        }
    )
    return rid


def write_store(entry: dict[str, str] | None, *, session_key: str = SESSION_KEY) -> None:
    """Write the posture store the way the web server persists it."""
    path = session_store.store_path()
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({session_key: entry or {}}), encoding="utf-8")
    session_store.invalidate_cache()


def write_marker(target: str, *, pid: int | None = None) -> None:
    """Plant an in-flight execution marker the way the executor writes one."""
    directory = target_state.state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory / f"{target_state.INFLIGHT_FILE_PREFIX}{pid or os.getpid()}_"
        f"{uuid.uuid4().hex}{target_state.INFLIGHT_FILE_SUFFIX}"
    )
    path.write_text(
        json.dumps(
            {
                "pid": os.getpid() if pid is None else pid,
                "owner_ppid": os.getppid(),
                "target": target,
                "started_at": "2026-08-30T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )


def clear_markers() -> None:
    for path in target_state.state_dir().glob(target_state.INFLIGHT_FILE_GLOB):
        path.unlink()


def last_switch() -> dict | None:
    record = target_state.read()
    return None if record is None else record.get("last_switch")


def realign() -> dict | None:
    record = target_state.read()
    return None if record is None else record.get("last_posture_realign")


# ------------------------------------------------------------------- requests


class TestSwitchRequests:
    async def test_a_permitted_request_switches_and_publishes_success(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        request_id = write_request("va")

        await reconciler.poll_once()

        assert manager.switch_calls == ["va"]
        assert manager.active_target() == "va"

        outcome = last_switch()
        assert outcome["request_id"] == request_id
        assert outcome["target"] == "va"
        assert outcome["status"] == session_control.STATUS_SUCCESS
        assert outcome["reason"] is None
        assert outcome["at"]
        # The request is consumed: the route reads its absence to allow the
        # next click, and a chip that finds no request finds the outcome above.
        assert target_state.read_request() is None
        assert emitted == [
            {
                "from_target": "live",
                "to_target": "va",
                "outcome": "success",
                "generation": manager.active_generation(),
            }
        ]
        assert [record["subject"] for record in records] == ["control_target_set"]
        assert records[0]["decision"] == "allowed"
        assert records[0]["session"] == SESSION_KEY
        assert "operator" in records[0]["detail"]

    async def test_a_gate_refusal_ends_the_request_without_switching(
        self, monkeypatch, emitted, records
    ):
        """The gate's word, not a second vocabulary invented for the button."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        reconciler = session_control.SessionControlReconciler()
        request_id = write_request("va")

        await reconciler.poll_once()

        assert manager.switch_calls == []
        outcome = last_switch()
        assert outcome["request_id"] == request_id
        assert outcome["target"] == "va"
        assert outcome["status"] == session_control.STATUS_REFUSED
        assert outcome["reason"] == control_target.REASON_READONLY_RUN
        assert "read-only sessions stay on the deployment baseline" in outcome["detail"]
        assert target_state.read_request() is None
        # A refusal is reported to the operator exactly as the tool's is.
        assert [call["reason"] for call in emitted] == [control_target.REASON_READONLY_RUN]
        assert emitted[0]["outcome"] == "failure"
        assert [record["decision"] for record in records] == ["refused"]
        assert records[0]["reason"] == control_target.REASON_READONLY_RUN

    async def test_the_gate_is_asked_immediately_before_the_switch(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """Not at request-write time, and not with anything cached in between.

        A marker planted after the request was written still refuses it: the
        reconciler reads the world at the moment it is about to act.
        """
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_request("va")
        write_marker("live")

        await reconciler.poll_once()

        assert manager.switch_calls == []
        assert last_switch()["reason"] == control_target.REASON_EXECUTION_IN_FLIGHT

    async def test_a_stale_request_expires_without_asking_the_gate(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """The operator who clicked Switch is no longer watching."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        request_id = write_request("va", age_s=target_state.REQUEST_TTL_S + 5)

        await reconciler.poll_once()

        assert manager.switch_calls == []
        outcome = last_switch()
        assert outcome["request_id"] == request_id
        assert outcome["target"] == "va"
        assert outcome["status"] == session_control.STATUS_EXPIRED
        assert outcome["reason"] == session_control.REASON_REQUEST_EXPIRED
        assert target_state.read_request() is None
        # Nothing was attempted, so there is no switch for the activity feed to
        # report — but the ledger still carries the gesture that timed out.
        assert emitted == []
        assert [record["reason"] for record in records] == [session_control.REASON_REQUEST_EXPIRED]

    async def test_a_request_addressed_to_another_server_is_dropped(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """A body naming another PID is not this server's to honour.

        It is removed rather than left: the route reads the file's presence to
        refuse a second click, so residue nobody will ever act on would wedge
        the button.
        """
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        # Written under this server's name, addressed in its body to another.
        path = target_state.request_file_path()
        path.write_text(
            json.dumps(
                {
                    "request_id": "not-ours",
                    "target": "va",
                    "server_pid": os.getpid() + 1,
                    "created_at": datetime.now(UTC).isoformat(),
                    "requested_by": "operator",
                }
            ),
            encoding="utf-8",
        )

        await reconciler.poll_once()

        assert manager.switch_calls == []
        assert last_switch() is None
        assert emitted == []
        assert records == []
        assert target_state.read_request() is None

    async def test_a_failed_switch_is_a_terminus_too(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """An attempt that ended is still an attempt that ended."""
        manager = FakeManager(target="live")
        manager.fail_with = SwitchError("va", "probe", "probe_failed", "the probe never connected")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        request_id = write_request("va")

        await reconciler.poll_once()

        outcome = last_switch()
        assert outcome["request_id"] == request_id
        assert outcome["target"] == "va"
        assert outcome["status"] == session_control.STATUS_FAILED
        assert outcome["reason"] == "probe_failed"
        assert target_state.read_request() is None
        assert [call["reason"] for call in emitted] == ["probe_failed"]
        assert [record["decision"] for record in records] == ["refused"]

    async def test_an_unexpected_exception_still_answers_the_request(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """A request this server looked at is one it owes an answer to.

        The file's signature has already been recorded by the time anything can
        raise, so an escaping exception would mean the request is never
        examined again: no outcome, the route refusing every later click as
        ``request_pending``, and a chip spinning until the operator gives up.
        """
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()

        async def explode(context, wanted):
            raise RuntimeError("the gate blew up")

        monkeypatch.setattr(control_target, "switch_gate", explode)
        request_id = write_request("va")

        await reconciler.poll_once()

        outcome = last_switch()
        assert outcome["request_id"] == request_id
        assert outcome["target"] == "va"
        assert outcome["status"] == session_control.STATUS_FAILED
        assert outcome["reason"] == session_control.REASON_INTERNAL_ERROR
        assert "the gate blew up" in outcome["detail"]
        assert target_state.read_request() is None
        assert len(records) == 1, "exactly one terminus, not two"
        assert len(emitted) == 1

    def test_the_internal_error_reason_is_the_tools_own_word(self):
        """Restated rather than imported (an import cycle), so pinned here."""
        assert session_control.REASON_INTERNAL_ERROR == control_target.REASON_INTERNAL_ERROR

    async def test_an_unchanged_request_file_is_not_acted_on_twice(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """Polls are cheap because only a moved signature costs anything."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_request("va")

        await reconciler.poll_once()
        await reconciler.poll_once()
        await reconciler.poll_once()

        assert manager.switch_calls == ["va"]
        assert len(records) == 1

    async def test_a_second_request_after_the_first_is_seen(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """Two clicks in one filesystem clock tick are two requests.

        The signature carries the inode for exactly this: both requests are
        written atomically, and a coarse mtime would otherwise hide the second.
        """
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()

        write_request("va")
        await reconciler.poll_once()
        second = write_request("live")
        await reconciler.poll_once()

        assert manager.switch_calls == ["va", "live"]
        assert last_switch()["request_id"] == second
        # The published target follows the request that was answered last, so
        # the popover puts the outcome on the row that was actually clicked.
        assert last_switch()["target"] == "live"


# -------------------------------------------------------------- realignment


class TestPostureRealignment:
    async def test_a_narrowing_on_the_active_target_rebuilds_the_connector(
        self, monkeypatch, records
    ):
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()

        await reconciler.poll_once()  # baseline: nothing narrowed
        assert manager.respawns == 0

        write_store({"live": "sandbox"})
        await reconciler.poll_once()

        assert manager.respawns == 1
        assert realign()["state"] == session_control.REALIGN_DONE

    async def test_a_narrowing_waits_for_a_running_execution(self, monkeypatch, records):
        """The rebuild retires the child a running execution was promised."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()

        write_marker("live")
        write_store({"live": "sandbox"})
        await reconciler.poll_once()

        assert manager.respawns == 0
        assert realign()["state"] == session_control.REALIGN_PENDING

        # Still waiting on the next poll, and still saying so.
        await reconciler.poll_once()
        assert manager.respawns == 0
        assert realign()["state"] == session_control.REALIGN_PENDING

        clear_markers()
        await reconciler.poll_once()
        assert manager.respawns == 1
        assert realign()["state"] == session_control.REALIGN_DONE

    async def test_a_rebuild_that_did_not_happen_stays_pending(self, monkeypatch, records):
        """The connector host can refuse to respawn, and it refuses quietly.

        ``invalidate_connector`` catches the ``SwitchError`` itself — the old
        child keeps serving rather than being torn down for a replacement that
        will not come up — so the only evidence is its return value. Reporting
        ``done`` on it would tell an operator their read-only toggle had taken
        effect on a child still connected under the old posture.
        """
        manager = FakeManager(target="live")
        manager.respawn_fails_with = SwitchError(
            "live", "spawn", "spawn_failed", "the child would not come up"
        )
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()

        write_store({"live": "sandbox"})
        await reconciler.poll_once()

        assert manager.respawns == 1, "the rebuild was not attempted"
        assert realign()["state"] == session_control.REALIGN_PENDING

        # And it keeps trying, without needing the store to move again.
        await reconciler.poll_once()
        assert manager.respawns == 2
        assert realign()["state"] == session_control.REALIGN_PENDING

        manager.respawn_fails_with = None
        await reconciler.poll_once()
        assert manager.respawns == 3
        assert realign()["state"] == session_control.REALIGN_DONE

    async def test_a_narrowing_on_another_target_is_not_a_realignment(self, monkeypatch, records):
        """Narrowing the machine the session is not on changes nothing here."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()

        write_store({"va": "sandbox"})
        await reconciler.poll_once()

        assert manager.respawns == 0
        assert realign() is None

    async def test_an_unchanged_store_is_not_reconciled_twice(self, monkeypatch, records):
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()

        write_store({"live": "sandbox"})
        await reconciler.poll_once()
        await reconciler.poll_once()
        await reconciler.poll_once()

        assert manager.respawns == 1

    async def test_a_switch_re_baselines_the_posture_the_session_is_judged_by(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """The child a switch built already read the store; it needs no rebuild."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_store({"va": "sandbox"})
        await reconciler.poll_once()

        write_request("va")
        await reconciler.poll_once()
        assert manager.active_target() == "va"

        # The pass after the switch finds a narrowed active target and rebuilds
        # nothing: the child that switch built read the store on the way up.
        await reconciler.poll_once()
        assert manager.respawns == 0
        assert realign() is None


# --------------------------------------------------------------------- races


class TestRaces:
    async def test_the_reconciler_and_an_agent_switch_spawn_one_child(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """Both callers can be right, and only one child may exist.

        A genuine race, not two calls that merely happen in one coroutine: the
        agent's switch is put *inside* the manager's critical section and held
        there, and only then is the reconciler started. It therefore reaches
        ``hosts.switch()`` while the destination is still ``live`` — its gate
        cannot have seen ``already_active`` — and blocks on the same lock. The
        equality guard is what makes the loser a settled no-op rather than a
        second spawn, and the reconciler still owes the operator a success.
        """
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_request("va")

        manager.release_switch = asyncio.Event()
        agent = asyncio.create_task(manager.switch("va"))
        await manager.entered_switch.wait()  # the agent now holds the lock

        poll = asyncio.create_task(reconciler.poll_once())
        for _ in range(10):  # let the reconciler run down to the held lock
            await asyncio.sleep(0)
        manager.release_switch.set()
        agent_result, _ = await asyncio.gather(agent, poll)

        assert manager.switch_calls == ["va", "va"], "both callers asked"
        assert manager.spawns == 1, "the second caller spawned a second child"
        assert agent_result["target_changed"] is True
        assert manager.active_target() == "va"
        assert last_switch()["status"] == session_control.STATUS_SUCCESS

    async def test_the_loser_of_that_race_reports_the_settled_result(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """The reconciler's own outcome, when it is the one that arrives second.

        Read off the emitted activity line rather than the manager: what the
        reconciler was handed is what it reports, and a settled result carries
        the generation the winner landed on with nothing respawned.
        """
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_request("va")

        manager.release_switch = asyncio.Event()
        agent = asyncio.create_task(manager.switch("va"))
        await manager.entered_switch.wait()
        poll = asyncio.create_task(reconciler.poll_once())
        for _ in range(10):
            await asyncio.sleep(0)
        manager.release_switch.set()
        await asyncio.gather(agent, poll)

        assert emitted == [
            {
                "from_target": "va",
                "to_target": "va",
                "outcome": "success",
                "generation": manager.active_generation(),
            }
        ]
        assert manager.spawns == 1


# ------------------------------------------------------------------ the loop


class TestTheLoop:
    async def test_an_exception_in_one_pass_does_not_stop_the_loop(self, monkeypatch, records):
        """A reconciler that died on one bad poll would strand every later one."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler(interval_s=0.01)
        passes: list[int] = []

        async def flaky() -> None:
            passes.append(len(passes))
            if len(passes) == 1:
                raise RuntimeError("the store went away")

        monkeypatch.setattr(reconciler, "poll_once", flaky)

        await reconciler.start()
        for _ in range(200):
            if len(passes) >= 3:
                break
            await asyncio.sleep(0.01)
        await reconciler.stop()

        assert len(passes) >= 3, "the loop stopped at the first exception"
        assert reconciler.running is False

    async def test_start_is_idempotent_and_stop_is_too(self, monkeypatch):
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler(interval_s=0.01)

        await reconciler.start()
        task = reconciler._task
        await reconciler.start()
        assert reconciler._task is task

        await reconciler.stop()
        await reconciler.stop()
        assert reconciler.running is False

    async def test_a_context_that_is_not_initialized_is_survived(self, monkeypatch, records):
        """A poll before the context exists reports nothing and raises nothing."""
        from osprey.mcp_server.control_system import server_context as server_context_mod

        monkeypatch.setattr(server_context_mod, "_registry", None)
        reconciler = session_control.SessionControlReconciler()

        await reconciler.poll_once()

        assert last_switch() is None


# --------------------------------------------------------------- the lifespan


class TestTheLifespan:
    async def test_the_lifespan_starts_and_stops_the_reconciler(self, monkeypatch):
        """It needs a running loop, so the lifespan owns it — beside the prober."""
        from osprey.mcp_server.control_system import endpoint_prober
        from osprey.mcp_server.control_system import server as server_mod

        manager = FakeManager(target="live")
        install_context(manager, monkeypatch, raw={"control_system": {"type": "mock"}})

        class NoProber:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        monkeypatch.setattr(endpoint_prober, "EndpointProber", NoProber)
        monkeypatch.setattr(server_mod, "_prober", None)
        monkeypatch.setattr(server_mod, "_reconciler", None)

        async with server_mod._lifespan(server_mod.mcp):
            reconciler = server_mod.get_session_reconciler()
            assert reconciler is not None
            assert reconciler.running is True

        assert reconciler.running is False
        assert server_mod.get_session_reconciler() is None

    async def test_a_reconciler_that_will_not_start_does_not_stop_the_server(self, monkeypatch):
        """Reconciling desired state is a service; serving tools is the job."""
        from osprey.mcp_server.control_system import server as server_mod

        def explode(*args, **kwargs):
            raise RuntimeError("no reconciler today")

        monkeypatch.setattr(session_control, "SessionControlReconciler", explode)
        monkeypatch.setattr(server_mod, "_reconciler", None)

        assert await server_mod.start_session_control() is None
        assert server_mod.get_session_reconciler() is None
