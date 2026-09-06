"""The session-control reconciler: one record, one fleet, one connector.

The control context is a single record per deployment, written only by its
owner. This task is a controls server's whole relationship with it: claim it
when nothing alive owns it, drag this process's connector host onto whatever it
says, and — while this process is the owner — answer the switch requests other
surfaces file, because they cannot call this server at all.

What is pinned here, and why each one matters:

* **Ownership is the addressee rule.** Requests are named for the process that
  asked, not for a server; a follower answers none of them and writes nothing,
  and the owner answers any of them.
* **The record is reconciled to before anything else in the pass**, and a live
  child publishes ``applying`` into its own report BEFORE the first await — so
  no other session's launch is admitted into the window where this process is
  between two targets. With no live child nothing is published at all: nothing
  is bound, so nothing can be bound wrongly, and a block written there would
  never be released.
* **Consumption is idempotent on ``request_id``.** The answer is the record,
  not the disappearance of a file: a request whose id is already the record's
  terminus is unlinked and nothing is written twice.
* **The owner consumes only while the fleet has converged**, so a terminus
  somebody is still waiting for is never clobbered by a swap that is still
  landing.
* **A same-target request mints nothing.** The deployment is where it was asked
  to be, and a generation bumped for a switch that did not happen would refuse
  every write bound to the old one for nothing.
* **The record write is read-verified before the request is unlinked**, because
  the requester polls the record for its own id and nothing else.
* **A narrowing waits for a running execution** and says so while it waits: the
  realignment rebuilds the connector, and rebuilding it under a run would
  retire the child that run was promised.

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
from pathlib import Path
from typing import Any

import pytest

from osprey.audit import writer as audit_writer
from osprey.mcp_server.control_system import session_control, target_eligibility, target_state
from osprey.mcp_server.control_system.connector_host_manager import SwitchError
from osprey.mcp_server.control_system.server_context import ControlSystemContext
from osprey.mcp_server.control_system.tools import control_target
from osprey_connectors import control_context, session_store
from tests import _control_context_fixtures as fixtures

pytestmark = pytest.mark.unit

SESSION_KEY = "session-abc"

#: A PID that is alive and is not this process, for the "somebody else owns it"
#: and "two requesters at once" cases. The parent of a pytest run is alive by
#: construction — it is what is waiting for the run to finish.
OTHER_PID = os.getppid()


# --------------------------------------------------------------------- fakes


class FakeManager:
    """A connector-host supervisor with the parts the reconciler touches.

    ``reconcile`` keeps the real manager's shape: the same result mapping
    whichever branch would have run, an adoption that respawns nothing when the
    target has not moved, and a raise the caller has to survive.
    """

    def __init__(
        self,
        target: str = "live",
        baseline: str = "live",
        *,
        generation: int = 0,
        child: bool = True,
    ) -> None:
        self._target = target
        self._generation = generation
        self.baseline = baseline
        self.child = child
        self.started = True
        #: Every ``(target, generation)`` the reconciler asked for.
        self.reconcile_calls: list[tuple[str, int]] = []
        #: This server's published switch block as it stood INSIDE the reconcile
        #: call — the only way to see that the block was written before the await.
        self.block_when_called: list[dict | None] = []
        self.reconcile_fails_with: Exception | None = None
        self.respawns = 0
        self.respawn_fails_with: Exception | None = None
        #: How often the reconciler asked for the targets block to be restated.
        self.display_publishes = 0
        self.display_fails_with: Exception | None = None
        #: What this server would write into an ``applying`` block's deadline.
        self.bound_s = 12.0

    # -- what the reconciler reads

    def active_target(self) -> str:
        return self._target

    def active_generation(self) -> int:
        return self._generation

    def has_child(self) -> bool:
        return self.child

    def is_started(self) -> bool:
        return self.started

    def applying_bound_s(self, *, fallback_retry: bool = True) -> float:
        return self.bound_s

    def publish_display(self) -> bool:
        """Stand in for the real republication, which is fail-soft itself."""
        self.display_publishes += 1
        if self.display_fails_with is not None:
            raise self.display_fails_with
        return True

    # -- what the reconciler drives

    async def reconcile(self, target: str, generation: int) -> dict:
        self.reconcile_calls.append((target, int(generation)))
        report = target_state.read() or {}
        self.block_when_called.append(report.get("last_switch"))
        if self.reconcile_fails_with is not None:
            raise self.reconcile_fails_with
        previous_target, previous_generation = self._target, self._generation
        respawned = self.child and target != self._target
        self._target, self._generation = target, int(generation)
        if self.child:
            # What the real manager's ``_publish`` does in the same call: the
            # binding and the ``applied`` terminus that RELEASES the caller's
            # ``applying`` block. A fake that adopted without publishing would
            # leave the deployment reading as mid-swap for ever, which is a
            # state no real server can be in.
            target_state.publish_switch(self._target, self._generation)
            target_state.publish_last_switch(
                {"generation": self._generation, "status": target_state.SWITCH_APPLIED}
            )
        return {
            "target": self._target,
            "generation": self._generation,
            "previous_target": previous_target,
            "previous_generation": previous_generation,
            "target_changed": target != previous_target,
            "generation_changed": int(generation) != previous_generation,
            "respawned": respawned,
            "child_pid": 4242 if self.child else None,
            "published": respawned,
        }

    async def respawn_same_target(self) -> dict:
        self.respawns += 1
        if self.respawn_fails_with is not None:
            raise self.respawn_fails_with
        return {"target": self._target, "generation": self._generation}


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
def deployment(control_context_root, monkeypatch):
    """One deployment on disk: this server's report, and a record it owns.

    Anchored through the ``OSPREY_AGENT_DATA_ROOT`` stamp rather than by
    patching a resolver: that stamp is the one resolution rule the record, the
    reports and the hooks all follow, so a test that patched around it would be
    testing a path no deployment takes.

    The report carries reachability for both targets because the switch gate's
    last rung reads the live fleet's rows — a server that is running and has
    published nothing about a target refuses the switch on its own reason, and
    that refusal is not what most of these tests are about.
    """
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", SESSION_KEY)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    session_store.invalidate_cache()
    target_state.write_server_record({"live": {"label": "Live"}, "va": {"label": "VA"}})
    target_state.publish_reachability(
        {
            name: {
                "read_write": {
                    "state": target_eligibility.REACHABILITY_REACHED,
                    "gateway": "gateway",
                    "probed_at": datetime.now(UTC).isoformat(),
                }
            }
            for name in ("live", "va")
        }
    )
    write_record()
    yield control_context_root
    session_store.invalidate_cache()


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
    """Stub eligibility open, so the gate's third rung is not the subject."""
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

    monkeypatch.setattr(target_eligibility, "target_availability", available)


# ------------------------------------------------------------------- helpers

#: Sentinel for :func:`write_record`: own the record as this controls server.
OURS = object()


def root() -> Path:
    return Path(os.environ["OSPREY_AGENT_DATA_ROOT"])


def write_record(
    *,
    target: str = "live",
    generation: int = 0,
    posture: Any = None,
    owned_by: Any = OURS,
    last_switch: dict | None = None,
) -> Path:
    """Write the deployment's control-context record, owned here by default."""
    return fixtures.write_control_context(
        root(),
        target,
        generation,
        posture,
        owned_by=(
            fixtures.owner(control_context.OWNER_CONTROLS_SERVER) if owned_by is OURS else owned_by
        ),
        last_switch=last_switch,
    )


def record() -> control_context.ControlContext | None:
    return control_context.read_record()


def terminus() -> dict | None:
    """The record's ``last_switch`` — a REQUEST's outcome, not a server's."""
    current = record()
    return None if current is None else current.last_switch


def write_request(
    target: str = "va",
    *,
    pid: int | None = None,
    age_s: float = 0.0,
    session: str | None = "operator",
    request_id: str | None = None,
) -> str:
    """File a switch request the way a surface that cannot call this server does."""
    rid = request_id or uuid.uuid4().hex
    target_state.write_request(
        {
            "request_id": rid,
            "target": target,
            "requested_by_pid": os.getpid() if pid is None else pid,
            "requested_at": (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat(),
            "session": session,
        }
    )
    return rid


def requests() -> list[Path]:
    return sorted(target_state.state_dir().glob(target_state.REQUEST_FILE_GLOB))


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
                "target": target,
                "session": SESSION_KEY,
                "surface": "python_executor",
                "started_at": "2026-08-30T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )


def clear_markers() -> None:
    for path in target_state.state_dir().glob(target_state.INFLIGHT_FILE_GLOB):
        path.unlink()


def report_block() -> dict | None:
    """This server's own progress block — ``applying``/``applied``/``failed``."""
    report = target_state.read()
    return None if report is None else report.get("last_switch")


def realign() -> dict | None:
    report = target_state.read()
    return None if report is None else report.get("last_posture_realign")


def dead_pid() -> int:
    """A PID no process holds, for the requester-is-gone cases."""
    for candidate in range(90000, 99999):
        if not control_context.is_process_alive(candidate):
            return candidate
    raise AssertionError("every PID in the scan range is alive")


# ----------------------------------------------------------------- ownership


class TestOwnership:
    """A controls server is the fallback owner, and a quiet follower otherwise."""

    async def test_an_ownerless_record_is_claimed_without_moving_it(self, monkeypatch, records):
        """The claim is a merge: it changes who may write, not what it says."""
        manager = FakeManager(target="va", generation=3)
        install_context(manager, monkeypatch)
        write_record(target="va", generation=3, posture={"va": "sandbox"}, owned_by=None)

        await session_control.SessionControlReconciler().poll_once()

        current = record()
        assert current.owner.kind == control_context.OWNER_CONTROLS_SERVER
        assert current.owner.pid == os.getpid()
        assert (current.target, current.generation) == ("va", 3)
        assert current.posture == {"va": "sandbox"}

    async def test_a_record_owned_by_a_dead_process_is_claimed(self, monkeypatch, records):
        manager = FakeManager()
        install_context(manager, monkeypatch)
        write_record(owned_by=fixtures.owner(control_context.OWNER_WEB_TERMINAL, dead_pid()))

        await session_control.SessionControlReconciler().poll_once()

        assert record().owner.pid == os.getpid()

    async def test_a_live_web_terminal_owner_is_followed_not_claimed(self, monkeypatch, records):
        """A terminal outranks a server: it is what the operator is looking at."""
        manager = FakeManager()
        install_context(manager, monkeypatch)
        write_record(owned_by=fixtures.owner(control_context.OWNER_WEB_TERMINAL, OTHER_PID))

        await session_control.SessionControlReconciler().poll_once()

        assert record().owner.pid == OTHER_PID

    async def test_a_follower_reconciles_to_the_record_it_does_not_own(self, monkeypatch, records):
        """Following is about who WRITES; every server obeys what is written."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        write_record(
            target="va",
            generation=4,
            owned_by=fixtures.owner(control_context.OWNER_WEB_TERMINAL, OTHER_PID),
        )

        await session_control.SessionControlReconciler().poll_once()

        assert manager.reconcile_calls == [("va", 4)]

    async def test_a_follower_answers_no_request(self, monkeypatch, emitted, records):
        """The outcome block belongs to the owner; a second answer is a clobber."""
        manager = FakeManager()
        install_context(manager, monkeypatch)
        write_record(owned_by=fixtures.owner(control_context.OWNER_WEB_TERMINAL, OTHER_PID))
        write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        assert len(requests()) == 1
        assert terminus() is None
        assert records == []

    async def test_a_record_that_cannot_be_read_leaves_the_pass_harmless(
        self, monkeypatch, records
    ):
        """No record is not an error here: a claim that failed retries next tick."""
        manager = FakeManager()
        install_context(manager, monkeypatch)
        fixtures.write_payload(control_context.record_path_under(root()), "not a record")
        monkeypatch.setattr(
            control_context, "write_record", lambda *a, **k: (_ for _ in ()).throw(OSError("no"))
        )

        await session_control.SessionControlReconciler().poll_once()

        assert manager.reconcile_calls == []


# ------------------------------------------------------------------ the record


class TestReconcileToTheRecord:
    async def test_a_live_child_publishes_applying_before_it_reconciles(self, monkeypatch, records):
        """The window where this process is between two targets is announced.

        Read from INSIDE the reconcile call: a block published afterwards would
        be a block every other reader missed for the length of the swap, which
        is the whole of what it exists to prevent.
        """
        manager = FakeManager(target="live", generation=0, child=True)
        install_context(manager, monkeypatch)
        write_record(target="va", generation=5)

        await session_control.SessionControlReconciler().poll_once()

        assert manager.reconcile_calls == [("va", 5)]
        block = manager.block_when_called[0]
        assert block["status"] == target_state.SWITCH_APPLYING
        assert block["generation"] == 5
        assert block["at"] and block["expires_at"], "an applying block carries its own deadline"

    async def test_the_deadline_is_the_bound_this_server_computes(self, monkeypatch, records):
        """Only this process knows its spawn, probe and drain timeouts."""
        manager = FakeManager(target="live", child=True)
        manager.bound_s = 30.0
        install_context(manager, monkeypatch)
        write_record(target="va", generation=1)

        await session_control.SessionControlReconciler().poll_once()

        block = manager.block_when_called[0]
        span = datetime.fromisoformat(block["expires_at"]) - datetime.fromisoformat(block["at"])
        assert abs(span.total_seconds() - 30.0) < 1.0

    async def test_no_live_child_adopts_silently(self, monkeypatch, records):
        """Nothing is bound, so nothing can be bound to the wrong generation.

        And no block may be written: a silent adoption publishes no terminus,
        so an ``applying`` block left here would never be released and would
        hold the whole deployment for the length of its bound.
        """
        manager = FakeManager(target="live", generation=0, child=False)
        install_context(manager, monkeypatch)
        write_record(target="va", generation=7)

        await session_control.SessionControlReconciler().poll_once()

        assert manager.reconcile_calls == [("va", 7)]
        assert report_block() is None

    async def test_a_server_already_on_the_record_reconciles_nothing(self, monkeypatch, records):
        """The steady state costs no lock, no publication and no write."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()

        for _ in range(5):
            await reconciler.poll_once()

        assert manager.reconcile_calls == []
        assert report_block() is None

    async def test_a_generation_that_moved_alone_is_still_reconciled(self, monkeypatch, records):
        """Same target, new generation: adopted against the child already there."""
        manager = FakeManager(target="live", generation=2)
        install_context(manager, monkeypatch)
        write_record(target="live", generation=3)

        await session_control.SessionControlReconciler().poll_once()

        assert manager.reconcile_calls == [("live", 3)]
        assert manager.active_generation() == 3

    async def test_a_failed_reconcile_does_not_stop_the_pass(self, monkeypatch, records):
        """The supervisor already filed ``failed``; a second verdict here would
        be this task's opinion about a swap it did not run."""
        manager = FakeManager(target="live", generation=0)
        manager.reconcile_fails_with = SwitchError(
            "va", "probe", "probe_failed", "the probe never connected"
        )
        install_context(manager, monkeypatch)
        write_record(target="va", generation=1)

        await session_control.SessionControlReconciler().poll_once()

        assert manager.reconcile_calls == [("va", 1)]
        # The posture half still ran: the record moved, so the targets block was
        # restated. A reconcile that raised out of the pass would have skipped it.
        assert manager.display_publishes == 1

    async def test_the_reconcile_runs_before_the_realignment(self, monkeypatch, records):
        """PROPOSAL bullet 7: a same-target adoption leaves a narrowing pending,
        and the rebuild that answers it happens in the same pass, after it."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()

        write_record(target="live", generation=1, posture={"live": "sandbox"})
        await reconciler.poll_once()

        assert manager.reconcile_calls == [("live", 1)]
        assert manager.respawns == 1
        assert realign()["state"] == session_control.REALIGN_DONE


# ------------------------------------------------------------------- requests


class TestSwitchRequests:
    async def test_a_permitted_request_moves_the_record_and_mints(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        request_id = write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        current = record()
        assert (current.target, current.generation) == ("va", 1)
        outcome = current.last_switch
        assert outcome["request_id"] == request_id
        assert outcome["target"] == "va"
        assert outcome["status"] == session_control.STATUS_APPLIED
        assert outcome["reason"] is None
        assert outcome["generation"] == 1
        assert outcome["requested_by"] == "operator"
        assert outcome["requested_at"]
        # The request is consumed: the requester polls the record for its own
        # id, and the file it wrote is the owner's to clear once it has landed.
        assert requests() == []
        assert emitted == [
            {"from_target": "live", "to_target": "va", "outcome": "success", "generation": 1}
        ]
        assert [record_["subject"] for record_ in records] == ["control_target_set"]
        assert records[0]["decision"] == "allowed"
        assert records[0]["session"] == SESSION_KEY
        assert "operator" in records[0]["detail"]

    async def test_the_connector_follows_on_the_next_tick(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """One writer moves the record; every server, this one included, obeys it."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_request("va")

        await reconciler.poll_once()
        assert manager.reconcile_calls == []

        await reconciler.poll_once()
        assert manager.reconcile_calls == [("va", 1)]
        assert manager.active_target() == "va"

    async def test_a_gate_refusal_moves_nothing_and_carries_no_generation(
        self, monkeypatch, emitted, records
    ):
        """The gate's word, not a second vocabulary invented for the button."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        request_id = write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        current = record()
        assert (current.target, current.generation) == ("live", 0)
        outcome = current.last_switch
        assert outcome["request_id"] == request_id
        assert outcome["target"] == "va"
        assert outcome["status"] == session_control.STATUS_REFUSED
        assert outcome["reason"] == target_eligibility.REASON_READONLY_RUN
        assert outcome["generation"] is None
        assert "read-only sessions stay on the deployment baseline" in outcome["detail"]
        assert requests() == []
        assert [call["reason"] for call in emitted] == [target_eligibility.REASON_READONLY_RUN]
        assert emitted[0]["outcome"] == "failure"
        assert [record_["decision"] for record_ in records] == ["refused"]

    async def test_the_gate_is_asked_immediately_before_the_answer(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """A marker planted after the request was written still refuses it."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        write_request("va")
        write_marker("live")

        await session_control.SessionControlReconciler().poll_once()

        assert terminus()["reason"] == target_eligibility.REASON_EXECUTION_IN_FLIGHT
        assert record().target == "live"

    async def test_a_request_for_the_target_of_record_mints_nothing(
        self, monkeypatch, emitted, records
    ):
        """Already there. A generation bumped for a switch that did not happen
        would refuse every write bound to the old one for nothing — and the gate
        is not consulted at all, which is why it may explode here."""
        manager = FakeManager(target="live", generation=4)
        install_context(manager, monkeypatch)
        write_record(target="live", generation=4)

        def explode(*args, **kwargs):
            raise AssertionError("the gate was asked about a target already of record")

        monkeypatch.setattr(target_eligibility, "evaluate_switch", explode)
        request_id = write_request("live")

        await session_control.SessionControlReconciler().poll_once()

        current = record()
        assert (current.target, current.generation) == ("live", 4)
        assert current.last_switch["request_id"] == request_id
        assert current.last_switch["status"] == session_control.STATUS_APPLIED
        assert current.last_switch["generation"] == 4
        assert requests() == []
        assert emitted[0]["outcome"] == "success"

    async def test_a_request_already_answered_is_unlinked_and_not_re_answered(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """Idempotency is on the id in the record, not on the file's absence.

        A crash between the record write and the unlink leaves exactly this
        state, and the recovery must be the unlink alone — re-answering would
        mint a second generation for one gesture.
        """
        manager = FakeManager(target="va", generation=2)
        install_context(manager, monkeypatch)
        request_id = write_request("va")
        write_record(
            target="va",
            generation=2,
            last_switch={
                "request_id": request_id,
                "target": "va",
                "status": session_control.STATUS_APPLIED,
                "reason": None,
                "detail": "already answered",
                "generation": 2,
            },
        )
        before = control_context.record_path().stat().st_mtime_ns

        await session_control.SessionControlReconciler().poll_once()

        assert requests() == []
        assert control_context.record_path().stat().st_mtime_ns == before, (
            "the record was rewritten"
        )
        assert records == []
        assert emitted == []

    async def test_only_one_request_is_answered_per_pass(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """A second answer while the fleet has not reported the first would
        clobber a terminus somebody is still waiting for."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_request("va", pid=os.getpid())
        write_request("va", pid=OTHER_PID)

        await reconciler.poll_once()
        assert len(requests()) == 1
        assert len(records) == 1

        await reconciler.poll_once()
        assert requests() == []
        assert len(records) == 2

    async def test_an_unconverged_fleet_leaves_every_request_alone(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """A swap is still landing somewhere: the record must not move under it."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        fixtures.write_server_report(
            root(),
            OTHER_PID,
            session="another-session",
            last_switch={
                "generation": 0,
                "status": control_context.REPORT_APPLYING,
                "at": datetime.now(UTC).isoformat(),
                "expires_at": (datetime.now(UTC) + timedelta(seconds=60)).isoformat(),
            },
        )
        write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        assert len(requests()) == 1
        assert terminus() is None
        assert records == []

    async def test_a_settled_fleet_answers_the_request_it_was_holding(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """The other half of the gate above: an expired block stops nobody else."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        fixtures.write_server_report(
            root(),
            OTHER_PID,
            session="another-session",
            last_switch={
                "generation": 0,
                "status": control_context.REPORT_APPLYING,
                "at": (datetime.now(UTC) - timedelta(seconds=120)).isoformat(),
                "expires_at": (datetime.now(UTC) - timedelta(seconds=60)).isoformat(),
            },
        )
        write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        assert requests() == []
        assert terminus()["status"] == session_control.STATUS_APPLIED

    async def test_an_answer_that_did_not_stick_leaves_the_request(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """Two owners can believe they own one record until the loser finds out.

        Unlinking a request whose answer did not land would leave the requester
        with neither an outcome to read nor a pending gesture to wait on.
        """
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        monkeypatch.setattr(control_context, "write_record", lambda *a, **k: None)
        write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        assert len(requests()) == 1
        assert terminus() is None
        assert records == [], "nothing happened, so nothing is filed"

    async def test_a_record_write_that_raises_leaves_the_request(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)

        def explode(*args, **kwargs):
            raise OSError("the state directory is read-only")

        monkeypatch.setattr(control_context, "write_record", explode)
        write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        assert len(requests()) == 1
        assert records == []

    async def test_an_unexpected_exception_ends_the_request_as_a_refusal(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """A request this owner looked at is one it owes an answer to. Nothing
        moved, so the answer is a refusal — the record has no third word."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)

        def explode(*args, **kwargs):
            raise RuntimeError("the gate blew up")

        monkeypatch.setattr(target_eligibility, "evaluate_switch", explode)
        request_id = write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        outcome = terminus()
        assert outcome["request_id"] == request_id
        assert outcome["status"] == session_control.STATUS_REFUSED
        assert outcome["reason"] == session_control.REASON_INTERNAL_ERROR
        assert "the gate blew up" in outcome["detail"]
        assert record().target == "live"
        assert requests() == []
        assert len(records) == 1, "exactly one terminus, not two"
        assert len(emitted) == 1

    async def test_the_internal_error_reason_is_the_tools_own_word(self):
        """Restated rather than imported (an import cycle), so pinned here."""
        assert session_control.REASON_INTERNAL_ERROR == control_target.REASON_INTERNAL_ERROR

    def test_the_terminus_vocabulary_is_the_records_own(self):
        """``applied`` / ``refused`` — a server's ``applying`` is a report word."""
        assert session_control.STATUS_APPLIED == control_context.SWITCH_APPLIED
        assert session_control.STATUS_REFUSED == control_context.SWITCH_REFUSED

    async def test_a_second_gesture_from_the_same_requester_is_answered_too(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """One slot per requester: the second write replaces the first, and the
        answer follows the request that is actually in the slot."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()

        write_request("va")
        await reconciler.poll_once()
        assert record().target == "va"

        second = write_request("live")
        await reconciler.poll_once()

        current = record()
        assert (current.target, current.generation) == ("live", 2)
        assert current.last_switch["request_id"] == second


class TestRequestsNobodyIsWaitingFor:
    """Removed without a terminus: there is nobody left to read one.

    The record's ``last_switch`` is how a requester learns what happened, and a
    requester that has gone or has stopped waiting learns nothing from a block
    written for it — while the block itself would overwrite the answer to a
    gesture somebody IS watching.
    """

    async def test_a_request_from_a_dead_process_is_dropped(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        write_request("va", pid=dead_pid())

        await session_control.SessionControlReconciler().poll_once()

        assert requests() == []
        assert terminus() is None
        assert record().target == "live"
        assert records == []

    async def test_a_request_older_than_the_ttl_is_dropped(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """The operator who clicked Switch is no longer watching, and a switch
        that lands minutes after the gesture is a surprise, not a service."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        write_request("va", age_s=target_state.REQUEST_TTL_S + 5)

        await session_control.SessionControlReconciler().poll_once()

        assert requests() == []
        assert terminus() is None
        assert record().target == "live"
        assert records == []

    async def test_an_unreadable_request_is_removed(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        target_state.request_file_path().write_text("{ not json", encoding="utf-8")

        await session_control.SessionControlReconciler().poll_once()

        assert requests() == []
        assert terminus() is None

    async def test_a_request_naming_no_requester_is_removed(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """Its slot cannot be swept by the pid in its name and nothing would
        ever clear it, so the owner that read it clears it."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        target_state.request_file_path().write_text(
            json.dumps({"target": "va", "requested_at": datetime.now(UTC).isoformat()}),
            encoding="utf-8",
        )

        await session_control.SessionControlReconciler().poll_once()

        assert requests() == []
        assert terminus() is None

    async def test_dropping_one_does_not_stop_the_pass_answering_another(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """Residue must not hold a live gesture for a whole tick.

        The residue is planted in a slot that sorts BEFORE this process's, so
        the pass genuinely has to get past it: dropping a file nobody is
        waiting for is not an answer, and only an answer ends the pass.
        """
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        residue = (
            target_state.state_dir()
            / f"{target_state.REQUEST_FILE_PREFIX}1{target_state.REQUEST_FILE_SUFFIX}"
        )
        residue.write_text("{ not json", encoding="utf-8")
        assert sorted(requests())[0] == residue
        request_id = write_request("va", pid=os.getpid())

        await session_control.SessionControlReconciler().poll_once()

        assert requests() == []
        assert terminus()["request_id"] == request_id
        assert record().target == "va"


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

        write_record(posture={"live": "sandbox"})
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
        write_record(posture={"live": "sandbox"})
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

        write_record(posture={"live": "sandbox"})
        await reconciler.poll_once()

        assert manager.respawns == 1, "the rebuild was not attempted"
        assert realign()["state"] == session_control.REALIGN_PENDING

        # And it keeps trying, without needing the record to move again.
        await reconciler.poll_once()
        assert manager.respawns == 2
        assert realign()["state"] == session_control.REALIGN_PENDING

        manager.respawn_fails_with = None
        await reconciler.poll_once()
        assert manager.respawns == 3
        assert realign()["state"] == session_control.REALIGN_DONE

    async def test_a_narrowing_on_another_target_is_not_a_realignment(self, monkeypatch, records):
        """Narrowing the machine this server is not on changes nothing here."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()

        write_record(posture={"va": "sandbox"})
        await reconciler.poll_once()

        assert manager.respawns == 0
        assert realign() is None

    async def test_an_unchanged_record_is_not_reconciled_twice(self, monkeypatch, records):
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()

        write_record(posture={"live": "sandbox"})
        await reconciler.poll_once()
        await reconciler.poll_once()
        await reconciler.poll_once()

        assert manager.respawns == 1

    async def test_a_switch_re_baselines_the_posture_it_is_judged_by(
        self, monkeypatch, emitted, records, allow_every_target
    ):
        """The child a switch built already read the record; it needs no rebuild."""
        manager = FakeManager(target="live", generation=0)
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_record(posture={"va": "sandbox"})
        await reconciler.poll_once()

        write_request("va")
        await reconciler.poll_once()  # the record moves to va
        await reconciler.poll_once()  # this server follows it
        assert manager.active_target() == "va"

        # The pass after the move finds a narrowed active target and rebuilds
        # nothing: the child that switch built read the record on the way up.
        await reconciler.poll_once()
        assert manager.respawns == 0
        assert realign() is None


# ------------------------------------------------------- display metadata


class TestDisplayRepublication:
    """The targets block follows the record, not just the switches.

    Display metadata names the gateway a posture chose, and a narrowing can
    land on a target no switch is about — where the writer would otherwise go
    on naming a gateway the session may no longer use. So a move of the record
    republishes the block on its own, and a record that has not moved buys
    nothing: the steady state must cost no publication and no write at all,
    or the reconciler would rewrite its report once a second forever.
    """

    async def test_a_narrowing_on_another_target_republishes_without_realigning(self, monkeypatch):
        """No switch runs for that machine, so nothing else would restate it."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()

        await reconciler.poll_once()  # baseline
        manager.display_publishes = 0

        write_record(posture={"va": "sandbox"})
        await reconciler.poll_once()

        assert manager.display_publishes == 1
        assert manager.respawns == 0
        assert realign() is None

    async def test_a_narrowing_on_the_active_target_republishes_and_realigns(self, monkeypatch):
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()
        manager.display_publishes = 0

        write_record(posture={"live": "sandbox"})
        await reconciler.poll_once()

        assert manager.display_publishes == 1
        assert manager.respawns == 1
        assert realign()["state"] == session_control.REALIGN_DONE

    async def test_a_settled_deployment_republishes_nothing_and_writes_nothing(self, monkeypatch):
        """Counted after the baseline pass, which legitimately does both."""
        manager = FakeManager(target="live")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        write_record(posture={"live": "sandbox"})
        await reconciler.poll_once()

        writes = 0
        real_write = target_state._atomic_write_json

        def counted(path, record_):
            nonlocal writes
            writes += 1
            real_write(path, record_)

        monkeypatch.setattr(target_state, "_atomic_write_json", counted)
        manager.display_publishes = 0

        for _ in range(10):
            await reconciler.poll_once()

        assert manager.display_publishes == 0
        assert writes == 0

    async def test_a_republication_that_raises_does_not_cost_the_realignment(self, monkeypatch):
        """The rebuild is what the operator asked for; the block is a copy."""
        manager = FakeManager(target="live")
        manager.display_fails_with = OSError("the report could not be written")
        install_context(manager, monkeypatch)
        reconciler = session_control.SessionControlReconciler()
        await reconciler.poll_once()
        manager.display_publishes = 0

        write_record(posture={"live": "sandbox"})
        await reconciler.poll_once()

        assert manager.display_publishes == 1
        assert manager.respawns == 1
        assert realign()["state"] == session_control.REALIGN_DONE


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
                raise RuntimeError("the record went away")

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
        write_request("va")

        await session_control.SessionControlReconciler().poll_once()

        assert terminus() is None
        assert len(requests()) == 1


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
