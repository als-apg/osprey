"""The web terminal's ownership of the control-context record.

The primitive next door answers "may this process write the record". This
suite is about the rung above it: *which* process that is, and what the answer
costs the terminal while it is being worked out.

The rules under test are the deployment's, not the terminal's:

* A web terminal claims over an owner that is **absent, dead, or a controls
  server**, and is a follower only behind a **live other web terminal**. A
  server claims only over an absent or dead owner, so the two rules compose to
  "the terminal wins once it is there" — right, because the terminal is what
  an operator is looking at.
* **A claim is a merge.** The target, generation and posture belong to the
  deployment and outlive every process that reads them. Taking the file over
  changes who may write it and nothing else, which is why an operator's
  narrowing survives a terminal restart.
* **The claim is fail-open.** A terminal that cannot own the record still
  serves; it renders the roster read-only and answers the write routes 503,
  and its owner task tries again a second later.
* **Nothing blocks the loop.** ``is_process_alive`` is a syscall against a
  process this deployment does not control. The test that patches it to block
  for ever and then asserts the loop keeps running is the one that pins that:
  a parked probe costs a worker thread, not the terminal.

The server-side halves of the same matrix are pinned by
``tests/mcp_server/test_session_control_reconciler.py``; the ``409`` a follower
terminal's write routes answer belongs to the routes' own suite.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.interfaces.web_terminal import control_context_owner as owner_module
from osprey.interfaces.web_terminal.control_context_owner import (
    CONTROL_CONTEXT_FRAME,
    ControlContextOwnerTask,
    start_control_context_owner,
    terminal_identity,
)
from osprey_connectors import control_context
from osprey_connectors.control_context import ControlContext, Owner
from tests import _control_context_fixtures as fixtures

#: A live web terminal that is not this one, and a controls server likewise.
#: Both are fictional PIDs, made live by the ``liveness`` fixture rather than
#: by existing — a test that spawns a process to be alive is a test that races.
OTHER_TERMINAL_PID = 424242
SERVER_PID = 535353

BASELINE = "live"


class _Recorder:
    """A broadcaster that keeps what it was handed."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    def broadcast(self, data: dict) -> None:
        self.frames.append(data)


def make_app() -> SimpleNamespace:
    """The two attributes of a FastAPI app this task touches."""
    return SimpleNamespace(state=SimpleNamespace(broadcaster=_Recorder()))


def make_task(app: SimpleNamespace, **kwargs) -> ControlContextOwnerTask:
    """A task claiming as this process, which is what production always does.

    The PID is deliberately the real one: the request half guards on
    :func:`~osprey_connectors.control_context.owned_here`, which is PID
    equality against ``os.getpid()`` — the same guard the controls server
    uses, so the two owners cannot disagree about who owns one file.
    """
    task = ControlContextOwnerTask(app, identity=terminal_identity(), **kwargs)
    # The baseline and the rendered config are the deployment's, and reading
    # them is a config load this suite is not about.
    task._baseline = lambda: BASELINE  # type: ignore[method-assign]
    task._rendered_config = lambda: {}  # type: ignore[method-assign]
    return task


@pytest.fixture
def liveness(monkeypatch):
    """Decide which PIDs are alive, instead of asking the operating system.

    Returns the mutable set. This process is in it by construction; everything
    else is alive only because a test said so, which is what lets one suite
    hold both a live foreign terminal and a dead one.
    """
    alive = {os.getpid()}

    def is_alive(pid: object) -> bool:
        try:
            return int(pid) in alive  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    # Every probe in the owner's path reaches this one binding: the request
    # triage asks it through ``target_state.is_process_alive``, which delegates
    # here at call time, so patching the connectors module covers the whole pass.
    monkeypatch.setattr(control_context, "is_process_alive", is_alive)
    return alive


def read(root: Path) -> ControlContext | None:
    """The record as it is on disk right now, past the signature cache."""
    control_context.invalidate_cache()
    return control_context.read_record(path=control_context.record_path_under(root))


# -- claiming ---------------------------------------------------------------


async def test_an_empty_deployment_is_claimed_at_the_baseline(control_context_root, liveness):
    app = make_app()
    await make_task(app).tick_once()

    stored = read(control_context_root)
    assert stored is not None
    assert (stored.target, stored.generation, stored.posture) == (BASELINE, 0, {})
    assert stored.owner == terminal_identity()
    assert app.state.control_context_follows is None


async def test_a_claim_over_a_dead_owner_merges_the_deployment_s_own_facts(
    control_context_root, write_control_context, liveness
):
    """SC-79: the record says what the deployment is pointed at, not who is running."""
    write_control_context(
        control_context_root,
        target="va",
        generation=7,
        posture={"va": "sandbox"},
        owned_by=fixtures.owner(pid=OTHER_TERMINAL_PID, port=8090),
    )

    await make_task(make_app()).tick_once()

    stored = read(control_context_root)
    assert stored is not None
    assert (stored.target, stored.generation) == ("va", 7)
    assert stored.posture == {"va": "sandbox"}
    assert stored.owner == terminal_identity()


async def test_an_operator_s_narrowing_survives_a_terminal_restart(
    control_context_root, write_control_context, liveness
):
    """The restart case of the merge, stated as the operator experiences it."""
    write_control_context(
        control_context_root,
        target="va",
        generation=4,
        posture={"va": "sandbox", "standin": "sandbox"},
        owned_by=fixtures.owner(pid=OTHER_TERMINAL_PID),
    )

    await make_task(make_app()).tick_once()
    # ... and again, as the next terminal would.
    await make_task(make_app()).tick_once()

    stored = read(control_context_root)
    assert stored is not None
    assert stored.posture == {"va": "sandbox", "standin": "sandbox"}


async def test_a_terminal_claims_over_a_live_controls_server(
    control_context_root, write_control_context, liveness
):
    """A server is the fallback owner; it hands the record over without being asked."""
    liveness.add(SERVER_PID)
    write_control_context(
        control_context_root,
        target="va",
        generation=2,
        owned_by=fixtures.owner(kind=control_context.OWNER_CONTROLS_SERVER, pid=SERVER_PID),
    )

    app = make_app()
    await make_task(app).tick_once()

    stored = read(control_context_root)
    assert stored is not None
    assert stored.owner == terminal_identity()
    assert (stored.target, stored.generation) == ("va", 2)
    assert app.state.control_context_follows is None


async def test_an_unparseable_record_starts_a_fresh_one(control_context_root, liveness):
    path = control_context.record_path_under(control_context_root)
    path.write_text("{not json at all", encoding="utf-8")
    control_context.invalidate_cache()

    await make_task(make_app()).tick_once()

    stored = read(control_context_root)
    assert stored is not None
    assert (stored.target, stored.generation) == (BASELINE, 0)


async def test_a_terminal_that_already_owns_the_record_writes_nothing(
    control_context_root, write_control_context, liveness
):
    """The steady state is silent: a 1 Hz rewrite would wake every reader watching it."""
    path = write_control_context(control_context_root, target="va", generation=3)
    before = path.read_bytes()

    task = make_task(make_app())
    await task.tick_once()
    await task.tick_once()

    assert path.read_bytes() == before


# -- following --------------------------------------------------------------


async def test_a_live_other_terminal_is_followed_and_never_written_over(
    control_context_root, write_control_context, liveness
):
    """SC-79: second terminal → this one follows, and the record is untouched."""
    liveness.add(OTHER_TERMINAL_PID)
    holder = fixtures.owner(pid=OTHER_TERMINAL_PID, port=8090)
    path = write_control_context(control_context_root, target="va", generation=5, owned_by=holder)
    before = path.read_bytes()

    app = make_app()
    await make_task(app).tick_once()

    assert path.read_bytes() == before
    assert app.state.control_context_follows == holder
    assert app.state.control_context_follows.port == 8090


async def test_a_follower_takes_the_record_the_moment_the_other_terminal_goes(
    control_context_root, write_control_context, liveness
):
    liveness.add(OTHER_TERMINAL_PID)
    write_control_context(control_context_root, owned_by=fixtures.owner(pid=OTHER_TERMINAL_PID))
    app = make_app()
    task = make_task(app)

    await task.tick_once()
    assert app.state.control_context_follows is not None

    liveness.discard(OTHER_TERMINAL_PID)
    await task.tick_once()

    assert app.state.control_context_follows is None
    stored = read(control_context_root)
    assert stored is not None and stored.owner == terminal_identity()


# -- what the routes read off app.state -------------------------------------


async def test_an_owning_terminal_publishes_a_writable_owner(control_context_root, liveness):
    app = make_app()
    task = make_task(app)

    await task.tick_once()

    assert app.state.control_context_owner is task.owner
    assert app.state.control_context_follows is None


async def test_a_claim_that_cannot_be_made_leaves_app_state_without_an_owner(monkeypatch):
    """SC-79: startup raise → the app comes up, and the roster has nothing to write through.

    The store is unreachable rather than merely empty — no agent-data root
    resolves — which is the shape the write routes answer ``503
    store_unavailable`` for.
    """
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.control_context_owner.record_path", lambda: None
    )
    app = make_app()

    task = await start_control_context_owner(app, interval_s=3600)
    try:
        assert not hasattr(app.state, "control_context_owner")
        assert not hasattr(app.state, "control_context_follows")
    finally:
        await task.stop()


async def test_the_owner_task_keeps_trying_after_a_failed_startup_claim(
    control_context_root, liveness, monkeypatch
):
    """The retry is the reason a failed claim starts the task anyway."""
    resolved = control_context.record_path()
    broken = True

    def record_path():
        if broken:
            raise OSError("the root is not there yet")
        return resolved

    monkeypatch.setattr(owner_module, "record_path", record_path)
    app = make_app()
    task = await start_control_context_owner(app, interval_s=0.01)
    try:
        assert not hasattr(app.state, "control_context_owner")
        broken = False
        await _until(lambda: hasattr(app.state, "control_context_owner"))
    finally:
        await task.stop()

    stored = read(control_context_root)
    assert stored is not None and stored.owner == terminal_identity()


# -- nothing on the loop ----------------------------------------------------


async def test_a_blocked_liveness_probe_does_not_stall_the_loop(
    control_context_root, write_control_context, monkeypatch
):
    """SC-79's last clause, and the reason every tick's disk work is in a thread.

    ``is_process_alive`` is patched to park for ever. If the claim ran on the
    event loop the whole terminal would stop here; instead one worker thread
    parks, the tick does not finish, and the loop goes on scheduling.
    """
    write_control_context(control_context_root, owned_by=fixtures.owner(pid=OTHER_TERMINAL_PID))
    released = threading.Event()

    def parks_for_ever(pid: object) -> bool:
        released.wait(10)
        return False

    monkeypatch.setattr(control_context, "is_process_alive", parks_for_ever)

    app = make_app()
    task = make_task(app)
    tick = asyncio.create_task(task.tick_once())
    try:
        served = 0
        for _ in range(20):
            await asyncio.sleep(0.005)
            served += 1

        assert served == 20, "the event loop stopped scheduling while the probe was parked"
        assert not tick.done()
    finally:
        released.set()
        await asyncio.wait_for(tick, timeout=10)

    assert app.state.control_context_owner is task.owner


# -- answering switch requests ----------------------------------------------


def file_request(pid: int, target: str, *, request_id: str, session: str | None = None) -> Path:
    """One ``switch_request_<pid>.json``, written the way the tool writes it."""
    from osprey.mcp_server.control_system import target_state

    body: dict = {"request_id": request_id, "target": target, "requested_by_pid": pid}
    if session is not None:
        body["session"] = session
    return target_state.write_request(body)


async def test_a_request_for_the_target_the_deployment_is_on_is_applied_without_a_mint(
    control_context_root, write_control_context, liveness
):
    """The no-mint answer, which both owners must give identically."""
    liveness.add(OTHER_TERMINAL_PID)
    write_control_context(control_context_root, target="va", generation=6)
    path = file_request(OTHER_TERMINAL_PID, "va", request_id="r-1", session="agent-7")

    await make_task(make_app()).tick_once()

    stored = read(control_context_root)
    assert stored is not None
    assert stored.generation == 6, "a switch that did not happen must not cost a generation"
    assert stored.last_switch is not None
    assert stored.last_switch["request_id"] == "r-1"
    assert stored.last_switch["status"] == control_context.SWITCH_APPLIED
    assert stored.last_switch["requested_by"] == "agent-7"
    assert not path.exists(), "an answered request is removed once the answer has stuck"


async def test_a_requester_with_no_session_is_named_by_its_pid(
    control_context_root, write_control_context, liveness
):
    liveness.add(OTHER_TERMINAL_PID)
    write_control_context(control_context_root, target="va", generation=1)
    file_request(OTHER_TERMINAL_PID, "va", request_id="r-2")

    await make_task(make_app()).tick_once()

    stored = read(control_context_root)
    assert stored is not None and stored.last_switch is not None
    assert stored.last_switch["requested_by"] == f"pid:{OTHER_TERMINAL_PID}"


async def test_a_request_from_a_process_that_has_gone_is_dropped_with_no_terminus(
    control_context_root, write_control_context, liveness
):
    """Nobody is left to read an answer, and writing one would overwrite a live gesture."""
    write_control_context(control_context_root, target="va", generation=1)
    path = file_request(OTHER_TERMINAL_PID, "standin", request_id="r-3")

    await make_task(make_app()).tick_once()

    assert not path.exists()
    stored = read(control_context_root)
    assert stored is not None
    assert stored.last_switch is None
    assert (stored.target, stored.generation) == ("va", 1)


async def test_a_follower_answers_nothing(control_context_root, write_control_context, liveness):
    """A follower files requests of its own; it never consumes anybody else's."""
    liveness.update({OTHER_TERMINAL_PID, SERVER_PID})
    write_control_context(
        control_context_root,
        target="va",
        generation=1,
        owned_by=fixtures.owner(pid=OTHER_TERMINAL_PID),
    )
    path = file_request(SERVER_PID, "va", request_id="r-4")

    app = make_app()
    await make_task(app).tick_once()

    assert app.state.control_context_follows is not None
    assert path.exists()
    stored = read(control_context_root)
    assert stored is not None and stored.last_switch is None


async def test_an_already_answered_request_is_removed_and_answered_only_once(
    control_context_root, write_control_context, liveness
):
    """Idempotent on ``request_id``: reaching a terminus twice writes nothing twice."""
    liveness.add(OTHER_TERMINAL_PID)
    write_control_context(control_context_root, target="va", generation=6)
    file_request(OTHER_TERMINAL_PID, "va", request_id="r-5")

    task = make_task(make_app())
    await task.tick_once()
    answered = control_context.record_path_under(control_context_root).read_bytes()

    file_request(OTHER_TERMINAL_PID, "va", request_id="r-5")
    await task.tick_once()

    assert control_context.record_path_under(control_context_root).read_bytes() == answered


# -- the push ---------------------------------------------------------------


async def test_the_context_frame_is_pushed_when_the_record_moves(
    control_context_root, write_control_context, liveness
):
    """A record written by a route reaches the browsers without the route pushing."""
    write_control_context(control_context_root, target="va", generation=1)
    app = make_app()
    task = make_task(app)

    await task.tick_once()
    assert app.state.broadcaster.frames == [], "startup has no browser to tell"

    write_control_context(control_context_root, target="va", generation=2)
    await task.tick_once()

    assert app.state.broadcaster.frames == [CONTROL_CONTEXT_FRAME]


async def test_a_settled_deployment_pushes_nothing(
    control_context_root, write_control_context, liveness
):
    write_control_context(control_context_root, target="va", generation=1)
    app = make_app()
    task = make_task(app)

    await task.tick_once()
    await task.tick_once()
    await task.tick_once()

    assert app.state.broadcaster.frames == []


# -- identity ---------------------------------------------------------------


def test_the_identity_names_the_port_the_terminal_actually_listens_on(monkeypatch):
    """``OSPREY_WEB_PORT`` is the settled port, fallback bind included."""
    monkeypatch.setenv("OSPREY_WEB_PORT", "8123")

    identity = terminal_identity()

    assert identity == Owner(kind=control_context.OWNER_WEB_TERMINAL, pid=os.getpid(), port=8123)


@pytest.mark.parametrize("declared", ["", "  ", "not-a-port"])
def test_an_unusable_port_declaration_names_no_port(monkeypatch, declared):
    """Better no port than one nothing is listening on: the port is where a refusal sends you."""
    monkeypatch.setenv("OSPREY_WEB_PORT", declared)

    assert terminal_identity().port is None


async def _until(predicate, *, timeout: float = 5.0) -> None:
    """Spin the loop until *predicate* holds. Never blocks the loop itself."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting on the owner task")
        await asyncio.sleep(0.01)


async def test_a_refused_switch_moves_neither_target_nor_generation(
    control_context_root, write_control_context, liveness
):
    """The gate's verdict, in the record: a refusal names no binding, so it mints none.

    The config handed to the gate is empty, so ``standin`` is a target this
    deployment cannot reach. Which of the gate's rungs refuses it is the gate's
    own suite to pin; what belongs here is that a refusal is written as a
    terminus, that its generation is null, and that the deployment stays where
    it was.
    """
    liveness.add(OTHER_TERMINAL_PID)
    write_control_context(control_context_root, target="va", generation=6)
    path = file_request(OTHER_TERMINAL_PID, "standin", request_id="r-6")

    await make_task(make_app()).tick_once()

    stored = read(control_context_root)
    assert stored is not None
    assert (stored.target, stored.generation) == ("va", 6)
    assert stored.last_switch is not None
    assert stored.last_switch["request_id"] == "r-6"
    assert stored.last_switch["status"] == control_context.SWITCH_REFUSED
    assert stored.last_switch["generation"] is None
    assert not path.exists()
