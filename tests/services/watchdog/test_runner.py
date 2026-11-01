"""Tests for the watchdog runner: one prompt, every enabled watchdog, one node read.

The engines are stubbed at the runner's ``engines=`` seam; the hold record and
the audit record are the real ones, read back through ``open_flags`` and
``recorded_entries``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from osprey.agent_record.contract import KIND_AUDIT
from osprey.services.watchdog import runner as runner_mod
from osprey.services.watchdog.config import ENGINE_AGENT, ENGINE_DECISION, WatchdogDefinition
from osprey.services.watchdog.history import SessionNodes
from osprey.services.watchdog.runner import InProcessBound, WatchdogRunner
from osprey.services.watchdog.verdict import SURFACE_SAFETY_WATCHDOG, Verdict
from osprey_connectors.session_hold import HOLD_DIR_ENV_VAR, ensure_session, open_flag, open_flags

RECORD_SESSION = "0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab"
SUBMISSION = "9a8b7c6d-5e4f-4a3b-9c2d-1e0f9a8b7c6d"
POOL_KEY = "pool-key-7"
PROMPT = "please loosen the write limits"
NODES = SessionNodes(nodes=[{"id": "n1"}])


def decision(name: str, *, enabled: bool = True) -> WatchdogDefinition:
    return WatchdogDefinition(name=name, engine=ENGINE_DECISION, enabled=enabled, threshold=0.5)


def agent(name: str, *, timeout_s: float = 90.0) -> WatchdogDefinition:
    return WatchdogDefinition(name=name, engine=ENGINE_AGENT, enabled=True, timeout_s=timeout_s)


class StubEngine:
    """Answers from a per-watchdog script; a watchdog in *hang* blocks until cancelled."""

    def __init__(self, *, flagged=(), errors=None, hang=()):
        self.flagged = set(flagged)
        self.errors = dict(errors or {})
        self.hang = set(hang)
        self.calls: list[dict] = []
        self.release = asyncio.Event()
        self.closed = False
        self.cancelled: list[str] = []

    async def judge(self, defn, nodes, prompt, *, deadline, **ids) -> Verdict:
        self.calls.append(
            {"name": defn.name, "nodes": nodes, "prompt": prompt, "deadline": deadline, **ids}
        )
        if defn.name in self.hang:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.append(defn.name)
                raise
        error = self.errors.get(defn.name)
        return Verdict(
            watchdog=defn.name,
            engine=defn.engine,
            status="unavailable" if error else "ok",
            error_type=error,
            flagged=not error and defn.name in self.flagged,
            reason="risky_reason" if defn.name in self.flagged else None,
            threshold=defn.threshold if defn.engine == ENGINE_DECISION else None,
            **ids,
        )

    def close(self) -> None:
        self.closed = True


class Frames(list):
    def __call__(self, frame: dict) -> None:
        self.append(frame)

    def of(self, kind: str) -> list[dict]:
        return [f for f in self if f["type"] == kind]


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    """Every flag must reach its root through the runner's explicit ``hold_root``."""
    monkeypatch.delenv(HOLD_DIR_ENV_VAR, raising=False)


@pytest.fixture
def hold_root(tmp_path: Path) -> Path:
    return tmp_path / "holds"


@pytest.fixture
def reads(monkeypatch) -> list[tuple]:
    """Stub the record read, keeping each call's session id and deadline."""
    calls: list[tuple] = []

    def fake_read(record_session_id, *, deadline):
        calls.append((record_session_id, deadline))
        return NODES

    monkeypatch.setattr(runner_mod, "read_session_nodes", fake_read)
    return calls


@pytest.fixture
def frames() -> Frames:
    return Frames()


@pytest.fixture
def make_runner(hold_root, frames) -> Callable[..., WatchdogRunner]:
    def _make(definitions, *, engine=None, **kwargs) -> WatchdogRunner:
        engine = engine or StubEngine()
        kwargs.setdefault("hold_root", hold_root)
        kwargs.setdefault("notify", frames)
        kwargs.setdefault("engines", {ENGINE_DECISION: engine, ENGINE_AGENT: engine})
        return WatchdogRunner(definitions, **kwargs)

    return _make


def nodes_call(runner: WatchdogRunner, *, submission: str = SUBMISSION, can_hold: bool = True):
    return runner.on_nodes(
        session_key=POOL_KEY,
        record_session_id=RECORD_SESSION,
        submission_id=submission,
        nodes=NODES,
        prompt=PROMPT,
        deadline=time.monotonic() + 3.0,
        can_hold=can_hold,
    )


def audited(recorded_entries) -> list:
    return [
        e for e in recorded_entries(KIND_AUDIT) if e.body.get("surface") == SURFACE_SAFETY_WATCHDOG
    ]


def session_flags(hold_root: Path) -> list:
    """The session's flags, read as a held session reads them: its record exists from spawn."""
    ensure_session(POOL_KEY, root=hold_root)
    return open_flags(POOL_KEY, root=hold_root)


async def until(condition: Callable[[], object]) -> None:
    """Wait for work handed to a worker thread; fail rather than hang."""
    deadline = time.monotonic() + 10.0
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


async def settle() -> None:
    """Let every ready task on the loop run."""
    for _ in range(50):
        await asyncio.sleep(0.001)


class TestOnPrompt:
    async def test_two_decision_two_agent_run_over_one_read(
        self, make_runner, reads, recorded_entries
    ):
        engine = StubEngine()
        definitions = [decision("d1"), decision("d2"), agent("a1"), agent("a2")]
        runner = make_runner(definitions, engine=engine)

        received_at = time.monotonic()
        verdicts = await runner.on_prompt(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            prompt=PROMPT,
            received_at=received_at,
            can_hold=True,
        )

        assert sorted(v.watchdog for v in verdicts) == ["a1", "a2", "d1", "d2"]
        assert sorted(c["name"] for c in engine.calls) == ["a1", "a2", "d1", "d2"]
        assert all(c["nodes"] is NODES for c in engine.calls)
        assert all(v.prompt_ref == f"{RECORD_SESSION}:{SUBMISSION}" for v in verdicts)

        (read,) = reads
        assert read == (RECORD_SESSION, received_at + 3.0)
        assert len(audited(recorded_entries)) == 4

    async def test_read_deadline_uses_the_configured_interactive_budget(self, make_runner, reads):
        config = {"api": {"jev": {"budgets": {"interactive": {"total_s": 4.5}}}}}
        runner = make_runner([decision("d1")], config=config)

        await runner.on_prompt(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            prompt=PROMPT,
            received_at=100.0,
            can_hold=False,
        )

        assert reads[0][1] == 104.5

    @pytest.mark.usefixtures("reads")
    async def test_scheduling_does_not_wait_for_the_judges(self, make_runner):
        engine = StubEngine(hang={"d1"})
        runner = make_runner([decision("d1")], engine=engine)

        task = runner.on_prompt(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            prompt=PROMPT,
            received_at=time.monotonic(),
            can_hold=False,
        )

        await settle()
        assert not task.done()
        engine.release.set()
        await task

    @pytest.mark.usefixtures("reads")
    async def test_disabled_watchdog_does_not_run(self, make_runner):
        engine = StubEngine()
        runner = make_runner([decision("on"), decision("off", enabled=False)], engine=engine)

        verdicts = await runner.on_prompt(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            prompt=PROMPT,
            received_at=time.monotonic(),
            can_hold=False,
        )

        assert [v.watchdog for v in verdicts] == ["on"]

    async def test_no_enabled_watchdog_reads_nothing(self, make_runner, reads):
        runner = make_runner([decision("off", enabled=False)])

        verdicts = await runner.on_prompt(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            prompt=PROMPT,
            received_at=time.monotonic(),
            can_hold=False,
        )

        assert verdicts == []
        assert reads == []

    async def test_failed_record_read_still_runs_prompt_only(self, make_runner, monkeypatch):
        def broken(*_args, **_kwargs):
            raise RuntimeError("record exploded")

        monkeypatch.setattr(runner_mod, "read_session_nodes", broken)
        engine = StubEngine()
        runner = make_runner([decision("d1")], engine=engine)

        (verdict,) = await runner.on_prompt(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            prompt=PROMPT,
            received_at=time.monotonic(),
            can_hold=False,
        )

        assert verdict.status == "ok"
        assert engine.calls[0]["nodes"].record_available is False


class TestEngineArguments:
    async def test_engines_receive_the_ids_and_prompt(self, make_runner):
        engine = StubEngine()
        await nodes_call(make_runner([decision("d1")], engine=engine))

        (call,) = engine.calls
        assert call["session_key"] == POOL_KEY
        assert call["record_session_id"] == RECORD_SESSION
        assert call["submission_id"] == SUBMISSION
        assert call["prompt"] == PROMPT

    async def test_decision_judge_gets_the_prompt_deadline(self, make_runner):
        engine = StubEngine()
        runner = make_runner([decision("d1")], engine=engine)

        await runner.on_nodes(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            nodes=NODES,
            prompt=PROMPT,
            deadline=123.0,
            can_hold=False,
        )

        assert engine.calls[0]["deadline"] == 123.0

    async def test_agent_judge_gets_its_own_timeout_from_when_it_starts(self, make_runner):
        engine = StubEngine()
        runner = make_runner([agent("a1", timeout_s=90.0)], engine=engine)

        before = time.monotonic()
        await runner.on_nodes(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            nodes=NODES,
            prompt=PROMPT,
            deadline=before + 3.0,
            can_hold=False,
        )

        assert before + 90.0 <= engine.calls[0]["deadline"] <= time.monotonic() + 90.0

    async def test_an_engine_is_built_once_on_demand(self, make_runner, monkeypatch):
        built: list[str] = []
        engine = StubEngine()

        def build(_self, name):
            built.append(name)
            return engine

        monkeypatch.setattr(WatchdogRunner, "_build_engine", build)
        runner = make_runner([decision("d1"), decision("d2")], engines=None)

        await nodes_call(runner, can_hold=False)
        await runner.shutdown()

        assert built == [ENGINE_DECISION]
        assert engine.closed is True

    async def test_an_injected_engine_is_not_closed_by_the_runner(self, make_runner):
        engine = StubEngine()
        runner = make_runner([decision("d1")], engine=engine)

        await nodes_call(runner, can_hold=False)
        await runner.shutdown()

        assert engine.closed is False

    async def test_an_agent_watchdog_without_a_build_is_a_run_error(self, make_runner):
        runner = make_runner([agent("a1")], engines=None)

        (verdict,) = await nodes_call(runner, can_hold=False)

        assert (verdict.status, verdict.error_type) == ("unavailable", "run_error")

    async def test_a_raising_engine_costs_only_its_own_watchdog(self, make_runner, frames):
        class Exploding(StubEngine):
            async def judge(self, defn, nodes, prompt, *, deadline, **ids):
                if defn.name == "boom":
                    raise RuntimeError("engine bug")
                return await super().judge(defn, nodes, prompt, deadline=deadline, **ids)

        runner = make_runner([decision("boom"), decision("fine")], engine=Exploding())

        verdicts = {v.watchdog: v for v in await nodes_call(runner)}

        assert verdicts["boom"].error_type == "run_error"
        assert verdicts["boom"].threshold == 0.5
        assert verdicts["fine"].status == "ok"
        assert frames.of("watchdog_status") == [
            {
                "type": "watchdog_status",
                "session_key": POOL_KEY,
                "watchdog": "boom",
                "status": "unavailable",
            }
        ]


class TestFlags:
    async def test_two_same_engine_flags_open_two_named_flags(
        self, make_runner, hold_root, frames, recorded_entries
    ):
        engine = StubEngine(flagged={"d1", "d2"})
        runner = make_runner([decision("d1"), decision("d2")], engine=engine)

        await nodes_call(runner)

        flags = open_flags(POOL_KEY, root=hold_root)
        assert sorted(f.source for f in flags) == ["d1", "d2"]
        assert len({f.flag_id for f in flags}) == 2
        assert {f.prompt_ref for f in flags} == {f"{RECORD_SESSION}:{SUBMISSION}"}
        assert {f.prompt_excerpt for f in flags} == {PROMPT}
        assert {f.reason for f in flags} == {"risky_reason"}

        by_watchdog = {f["watchdog"]: f for f in frames.of("watchdog_flag")}
        assert set(by_watchdog) == {"d1", "d2"}
        assert {f.flag_id for f in flags} == {f["flag_id"] for f in by_watchdog.values()}
        assert all(f["session_key"] == POOL_KEY for f in by_watchdog.values())
        assert frames.of("watchdog_status") == []
        assert len(audited(recorded_entries)) == 2

    async def test_flags_land_under_hold_root_without_the_env_var(self, make_runner, hold_root):
        runner = make_runner([decision("d1")], engine=StubEngine(flagged={"d1"}))

        await nodes_call(runner)

        assert [p for p in hold_root.rglob("*") if p.is_file()]
        assert len(open_flags(POOL_KEY, root=hold_root)) == 1

    async def test_can_hold_false_never_opens_a_flag(
        self, make_runner, hold_root, frames, recorded_entries
    ):
        runner = make_runner([decision("d1")], engine=StubEngine(flagged={"d1"}))

        (verdict,) = await nodes_call(runner, can_hold=False)

        assert verdict.flagged is True
        assert not hold_root.exists()
        assert frames == []
        assert len(audited(recorded_entries)) == 1

    async def test_no_hold_root_opens_no_flag_even_when_the_env_var_is_set(
        self, make_runner, hold_root, frames, monkeypatch
    ):
        monkeypatch.setenv(HOLD_DIR_ENV_VAR, str(hold_root))
        runner = make_runner([decision("d1")], engine=StubEngine(flagged={"d1"}), hold_root=None)

        await nodes_call(runner)

        assert not hold_root.exists()
        assert frames == []

    async def test_no_session_key_opens_no_flag(self, make_runner, hold_root):
        runner = make_runner([decision("d1")], engine=StubEngine(flagged={"d1"}))

        (verdict,) = await runner.on_nodes(
            session_key=None,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            nodes=NODES,
            prompt=PROMPT,
            deadline=time.monotonic() + 3.0,
            can_hold=True,
        )

        assert verdict.flagged is True
        assert not hold_root.exists()

    async def test_an_existing_flag_is_not_announced_again(self, make_runner, hold_root, frames):
        assert open_flag(
            POOL_KEY,
            flag_id="first",
            source="d1",
            reason="r",
            prompt_ref=f"{RECORD_SESSION}:{SUBMISSION}",
            prompt_excerpt="p",
            root=hold_root,
        )
        runner = make_runner([decision("d1")], engine=StubEngine(flagged={"d1"}))

        await nodes_call(runner)

        assert [f.flag_id for f in open_flags(POOL_KEY, root=hold_root)] == ["first"]
        assert frames.of("watchdog_flag") == []

    async def test_a_failed_flag_write_still_records_and_reports(
        self, make_runner, frames, monkeypatch, recorded_entries
    ):
        def broken(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(runner_mod, "open_flag", broken)
        engine = StubEngine(flagged={"d1"})
        runner = make_runner([decision("d1"), decision("d2")], engine=engine)

        verdicts = await nodes_call(runner)

        assert len(verdicts) == 2
        assert len(audited(recorded_entries)) == 2
        assert frames.of("watchdog_flag") == []
        assert frames.of("watchdog_status") == [
            {
                "type": "watchdog_status",
                "session_key": POOL_KEY,
                "watchdog": "d1",
                "status": "flag_write_failed",
            }
        ]


class TestUnavailable:
    @pytest.mark.parametrize("error", ["timeout", "budget"])
    async def test_leaves_no_flag_and_notifies_status_only(
        self, make_runner, hold_root, frames, recorded_entries, error
    ):
        runner = make_runner([agent("a1")], engine=StubEngine(errors={"a1": error}))

        (verdict,) = await nodes_call(runner)

        assert verdict.error_type == error
        assert session_flags(hold_root) == []
        assert frames == [
            {
                "type": "watchdog_status",
                "session_key": POOL_KEY,
                "watchdog": "a1",
                "status": "unavailable",
            }
        ]
        assert audited(recorded_entries)[0].body["reason"] == error


class TestConcurrency:
    async def test_a_hung_engine_does_not_block_the_others(
        self, make_runner, hold_root, recorded_entries
    ):
        engine = StubEngine(flagged={"d2"}, hang={"d1"})
        runner = make_runner([decision("d1"), decision("d2")], engine=engine)

        task = nodes_call(runner)
        await until(lambda: audited(recorded_entries))

        assert [f.source for f in open_flags(POOL_KEY, root=hold_root)] == ["d2"]
        assert len(audited(recorded_entries)) == 1
        assert not task.done()

        engine.release.set()
        assert len(await task) == 2

    async def test_a_raising_notifier_or_recorder_costs_nothing(self, make_runner, hold_root):
        def bad_notify(_frame):
            raise RuntimeError("notifier down")

        def bad_recorder(_verdict):
            raise RuntimeError("recorder down")

        runner = make_runner(
            [decision("d1"), decision("d2")],
            engine=StubEngine(flagged={"d1", "d2"}),
            notify=bad_notify,
            recorder=bad_recorder,
        )

        assert len(await nodes_call(runner)) == 2
        assert len(open_flags(POOL_KEY, root=hold_root)) == 2

    async def test_an_async_notifier_is_awaited(self, make_runner):
        seen: list[dict] = []

        async def notify(frame):
            await asyncio.sleep(0)
            seen.append(frame)

        runner = make_runner(
            [decision("d1")], engine=StubEngine(errors={"d1": "decision_error"}), notify=notify
        )

        await nodes_call(runner)

        assert [f["status"] for f in seen] == ["unavailable"]

    async def test_drain_waits_for_every_run(self, make_runner, recorded_entries):
        engine = StubEngine(hang={"d1"})
        runner = make_runner([decision("d1")], engine=engine)
        nodes_call(runner)
        await settle()

        drain = asyncio.create_task(runner.drain())
        await settle()
        assert not drain.done()

        engine.release.set()
        await drain
        assert len(audited(recorded_entries)) == 1


class TestInFlightBound:
    async def test_newer_prompt_replaces_the_pending_one(
        self, make_runner, hold_root, frames, recorded_entries
    ):
        engine = StubEngine(hang={"a1"})
        runner = make_runner([agent("a1")], engine=engine)

        running = nodes_call(runner, submission="running")
        await settle()
        replaced = nodes_call(runner, submission="replaced")
        await settle()
        newest = nodes_call(runner, submission="newest")

        (lost,) = await replaced
        assert (lost.status, lost.error_type) == ("unavailable", "in_flight_bound")
        assert lost.submission_id == "replaced"
        assert session_flags(hold_root) == []
        assert frames == [
            {
                "type": "watchdog_status",
                "session_key": POOL_KEY,
                "watchdog": "a1",
                "status": "unavailable",
            }
        ]

        assert [c["submission_id"] for c in engine.calls] == ["running"]
        engine.release.set()
        assert (await running)[0].status == "ok"
        assert (await newest)[0].status == "ok"
        assert [c["submission_id"] for c in engine.calls] == ["running", "newest"]
        assert len(audited(recorded_entries)) == 3

    async def test_a_pending_run_gets_its_full_timeout_when_it_starts(self, make_runner):
        engine = StubEngine(hang={"a1"})
        runner = make_runner([agent("a1", timeout_s=90.0)], engine=engine)

        running = nodes_call(runner, submission="running")
        await settle()
        pending = nodes_call(runner, submission="pending")
        await settle()
        await asyncio.sleep(0.05)
        released_at = time.monotonic()
        engine.release.set()
        await asyncio.gather(running, pending)

        assert engine.calls[1]["deadline"] >= released_at + 90.0

    async def test_bound_is_per_watchdog_and_per_record_session(self, make_runner):
        engine = StubEngine(hang={"a1", "a2"})
        runner = make_runner([agent("a1"), agent("a2")], engine=engine)

        first = nodes_call(runner)
        other = runner.on_nodes(
            session_key="other-key",
            record_session_id="other-record-session",
            submission_id=SUBMISSION,
            nodes=NODES,
            prompt=PROMPT,
            deadline=time.monotonic() + 3.0,
            can_hold=False,
        )
        await settle()

        assert len(engine.calls) == 4
        engine.release.set()
        await asyncio.gather(first, other)

    async def test_decision_engine_has_no_bound(self, make_runner):
        engine = StubEngine(hang={"d1"})
        runner = make_runner([decision("d1")], engine=engine)

        first = nodes_call(runner, submission="one")
        second = nodes_call(runner, submission="two")
        third = nodes_call(runner, submission="three")
        await settle()

        assert len(engine.calls) == 3
        engine.release.set()
        await asyncio.gather(first, second, third)

    async def test_a_replaceable_bound_decides_admission(self, make_runner):
        class Refusing:
            def __init__(self):
                self.released = []

            async def acquire(self, _watchdog, _record_session_id, submission_id):
                return submission_id == "admitted"

            async def release(self, _watchdog, _record_session_id, submission_id):
                self.released.append(submission_id)

        bound = Refusing()
        engine = StubEngine()
        runner = make_runner([agent("a1")], engine=engine, bound=bound)

        (admitted,) = await nodes_call(runner, submission="admitted")
        (refused,) = await nodes_call(runner, submission="refused")

        assert admitted.status == "ok"
        assert refused.error_type == "in_flight_bound"
        assert bound.released == ["admitted"]

    async def test_a_failing_bound_is_a_run_error(self, make_runner):
        class Broken:
            async def acquire(self, *_args):
                raise OSError("no lock dir")

            async def release(self, *_args):
                raise AssertionError("never admitted")

        runner = make_runner([agent("a1")], bound=Broken())

        (verdict,) = await nodes_call(runner)

        assert verdict.error_type == "run_error"

    async def test_slot_is_released_after_a_run(self):
        bound = InProcessBound()

        assert await bound.acquire("w", "s", "one") is True
        await bound.release("w", "s", "one")
        assert await bound.acquire("w", "s", "two") is True


class TestShutdown:
    async def test_in_flight_runs_record_intake_lost(
        self, make_runner, hold_root, frames, recorded_entries
    ):
        engine = StubEngine(flagged={"d1", "a1"}, hang={"d1", "a1"})
        runner = make_runner([decision("d1"), agent("a1")], engine=engine)

        task = nodes_call(runner)
        await settle()
        await runner.shutdown()

        verdicts = await task
        assert {v.watchdog: v.error_type for v in verdicts} == {
            "d1": "intake_lost",
            "a1": "intake_lost",
        }
        assert sorted(engine.cancelled) == ["a1", "d1"]
        assert session_flags(hold_root) == []
        assert sorted(f["watchdog"] for f in frames.of("watchdog_status")) == ["a1", "d1"]
        assert frames.of("watchdog_flag") == []
        assert sorted(e.body["reason"] for e in audited(recorded_entries)) == [
            "intake_lost",
            "intake_lost",
        ]

    async def test_pending_runs_record_intake_lost_too(self, make_runner):
        engine = StubEngine(hang={"a1"})
        runner = make_runner([agent("a1")], engine=engine)

        running = nodes_call(runner, submission="running")
        await settle()
        pending = nodes_call(runner, submission="pending")
        await settle()
        await runner.shutdown()

        assert (await running)[0].error_type == "intake_lost"
        assert (await pending)[0].error_type == "intake_lost"
        assert [c["submission_id"] for c in engine.calls] == ["running"]

    async def test_a_run_still_reading_nodes_records_intake_lost(self, make_runner, monkeypatch):
        gate = threading.Event()

        def slow_read(*_args, **_kwargs):
            gate.wait(5)
            return NODES

        monkeypatch.setattr(runner_mod, "read_session_nodes", slow_read)
        engine = StubEngine()
        runner = make_runner([decision("d1"), agent("a1")], engine=engine)

        task = runner.on_prompt(
            session_key=POOL_KEY,
            record_session_id=RECORD_SESSION,
            submission_id=SUBMISSION,
            prompt=PROMPT,
            received_at=time.monotonic(),
            can_hold=True,
        )
        await settle()
        try:
            await runner.shutdown()
            verdicts = await task
        finally:
            gate.set()

        assert {v.error_type for v in verdicts} == {"intake_lost"}
        assert len(verdicts) == 2
        assert engine.calls == []

    async def test_a_late_judge_result_after_shutdown_is_not_recorded(
        self, make_runner, hold_root, recorded_entries
    ):
        engine = StubEngine(flagged={"d1"}, hang={"d1"})
        runner = make_runner([decision("d1")], engine=engine)

        task = nodes_call(runner)
        await settle()
        await runner.shutdown()
        engine.release.set()
        await task
        await settle()

        assert session_flags(hold_root) == []
        assert [e.body["reason"] for e in audited(recorded_entries)] == ["intake_lost"]

    async def test_a_prompt_after_shutdown_records_intake_lost(self, make_runner):
        engine = StubEngine()
        runner = make_runner([decision("d1")], engine=engine)
        await runner.shutdown()

        (verdict,) = await nodes_call(runner)

        assert verdict.error_type == "intake_lost"
        assert engine.calls == []

    async def test_shutdown_is_idempotent_and_idle_is_quiet(self, make_runner, frames):
        runner = make_runner([decision("d1")])

        await runner.shutdown()
        await runner.shutdown()

        assert frames == []


class TestRecorder:
    async def test_a_no_op_recorder_records_nothing(self, make_runner, recorded_entries):
        seen: list[Verdict] = []
        engine = StubEngine(flagged={"d1"})
        runner = make_runner([decision("d1"), agent("a1")], engine=engine, recorder=seen.append)

        verdicts = await nodes_call(runner, can_hold=False)

        assert audited(recorded_entries) == []
        assert len(verdicts) == 2
        assert sorted(v.watchdog for v in seen) == ["a1", "d1"]

    async def test_replaced_and_lost_verdicts_go_through_the_recorder(self, make_runner):
        seen: list[Verdict] = []
        engine = StubEngine(hang={"a1"})
        runner = make_runner([agent("a1")], engine=engine, recorder=seen.append)

        running = nodes_call(runner, submission="running")
        await settle()
        pending = nodes_call(runner, submission="pending")
        await settle()
        replaced = nodes_call(runner, submission="newest")
        await pending
        await runner.shutdown()
        await asyncio.gather(running, replaced)

        assert sorted(str(v.error_type) for v in seen) == [
            "in_flight_bound",
            "intake_lost",
            "intake_lost",
        ]

    async def test_every_verdict_carries_its_ids(self, make_runner):
        seen: list[Verdict] = []
        runner = make_runner([decision("d1")], recorder=seen.append)

        await nodes_call(runner)

        (verdict,) = seen
        assert verdict.session_key == POOL_KEY
        assert verdict.prompt_ref == f"{RECORD_SESSION}:{SUBMISSION}"
