"""The ``control_target_set`` tool: ask the record's owner, report what it says.

The switch itself is pinned by ``test_switch_lifecycle.py`` — real children,
real processes, real spawn-then-swap — and the gate by ``test_switch_gate.py``,
against both of its callers. What is pinned HERE is the tool: which of the two
paths it takes, what it writes, what it waits for, and the words an operator
gets for each way it can end.

The target belongs to the deployment, so this tool no longer switches anything.
It reads the control-context record and then either

* **owns it** — takes the gate verdict and writes the terminus itself, a
  refusal or the new target with the minted generation; or
* **does not** — files a request named for this process and polls the record
  until ``last_switch.request_id`` is its own.

Both paths then wait for the same thing, which is not the switch: it is THIS
server's connector host reaching the generation the record now names. Every
bound around that wait is asserted, because each of them exists to stop a tool
call hanging on a process that is never going to answer.

The fixtures are the switch-lifecycle suite's, rebound rather than rebuilt: the
two targets it constructs (a mock connector by dotted path for ``live``, a
fixture connector registered as ``virtual_accelerator`` for ``va``) are exactly
what these tests need, and a second copy would be a second thing to keep true.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime

import pytest

from osprey.mcp_server.control_system import target_eligibility as te
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.server_context import ControlSystemContext
from osprey.mcp_server.control_system.target_eligibility import (
    ACK_LEAF,
    REASON_ARCHIVE_BELONGS_TO_STANDIN,
    REASON_LIMITS_POSTURE,
    REASON_OPERATOR_ACK_MISSING,
    REASON_PROBE_CHANNEL_MISSING,
    REASON_TARGET_UNREACHABLE,
)
from osprey.mcp_server.control_system.tools import control_target
from osprey.mcp_server.python_executor import executor as py_executor
from osprey_connectors import control_context
from osprey_connectors.standin import ARCHIVER_RECORDER_SERVICE
from osprey_connectors.types import LIVE_STANDIN
from tests._control_context_fixtures import owner, write_control_context, write_server_report
from tests.mcp_server import test_switch_lifecycle as switch_suite
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

SETTLE_TIMEOUT_S = switch_suite.SETTLE_TIMEOUT_S
raw_config = switch_suite.raw_config

# The switch-lifecycle suite's fixtures, rebound so pytest collects them in this
# module too. ``state_root`` and ``child_environment`` are autouse there and stay
# autouse here, which is what anchors every state file this module writes under
# tmp_path. Rebound rather than imported by name so that a test's fixture
# parameter does not read as a shadowed import.
child_environment = switch_suite.child_environment
fixture_dir = switch_suite.fixture_dir
make_manager = switch_suite.make_manager
state_root = switch_suite.state_root

TOOL = get_tool_fn(control_target.control_target_set)

#: Fast enough that the bounded waits are pinned in milliseconds rather than in
#: the seconds a real deployment gives them.
TICK_S = 0.005

#: How long a helper standing in for the record's owner waits for the request
#: it is supposed to consume. Generous: it bounds a bug, not the test.
OWNER_PATIENCE_S = 5.0


# ------------------------------------------------------------------ helpers


@pytest.fixture(autouse=True)
def record_root(state_root, monkeypatch):
    """Point the record library at the same scratch root the reports use.

    ``state_root`` patches ``target_state``'s own resolver; the control-context
    library resolves independently, through the agent-data stamp. Stamping it
    here is what makes one directory hold the record, the reports and the
    requests — which is the thing under test, not an incidental of it.
    """
    from osprey_connectors import posture_store

    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(state_root))
    (state_root / posture_store.STATE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    control_context.invalidate_cache()
    yield state_root
    control_context.invalidate_cache()


@pytest.fixture(autouse=True)
def quick_ticks(monkeypatch):
    """The reconciler's second is a second; a test's need not be."""
    monkeypatch.setattr(control_target, "POLL_INTERVAL_S", TICK_S)


@pytest.fixture(autouse=True)
def a_lock_for_this_loop(monkeypatch):
    """One switch lock per test, because there is one event loop per test.

    ``asyncio.Lock`` binds itself to the loop that first *contends* it and
    raises on every later loop. The server has one loop for its whole life, so
    the module-level lock is right there and wrong here: without this, the one
    test that contends the lock would poison every test that ran after it in
    the same process.
    """
    monkeypatch.setattr(control_target, "_SWITCH_LOCK", asyncio.Lock())


def install_context(manager, monkeypatch) -> ControlSystemContext:
    """Make *manager*'s deployment the server context the tool will read."""
    from osprey.mcp_server.control_system import server_context as server_context_mod

    context = ControlSystemContext()
    context._config = manager._config
    context._connector_hosts = manager
    monkeypatch.setattr(server_context_mod, "_registry", context)
    return context


def owned_here(root, target="live", generation=1, **kwargs):
    """A record this process owns, so the tool answers it where it stands."""
    return write_control_context(
        root,
        target=target,
        generation=generation,
        owned_by=owner(control_context.OWNER_CONTROLS_SERVER, pid=os.getpid()),
        **kwargs,
    )


def owned_elsewhere(root, target="live", generation=1, *, owner_pid=None, **kwargs):
    """A record owned by another live process, so the tool has to ask for a switch."""
    return write_control_context(
        root,
        target=target,
        generation=generation,
        owned_by=owner(
            control_context.OWNER_CONTROLS_SERVER,
            pid=os.getppid() if owner_pid is None else owner_pid,
        ),
        **kwargs,
    )


def reachable(*targets: str) -> dict:
    """A reachability block saying the live fleet has reached *targets*.

    A live report with nothing to say about the wanted target is a refusal in
    its own right (``reachability_unknown``: something is there to ask and it
    has not answered) — and ``make_manager`` publishes exactly such a report at
    ``reset_state``, so EVERY test in this module has a live server in it. A
    test that wants the gate to reach anything past that rung has to publish a
    measurement, which is what this and :func:`our_report` are for.
    """
    names = targets or ("live", "va", "standin")
    probed_at = datetime.now(UTC).isoformat()
    return {
        "targets": {
            name: {
                "read_only": {
                    "state": te.REACHABILITY_REACHED,
                    "probed_at": probed_at,
                    "gateway": "127.0.0.1:5064",
                }
            }
            for name in names
        }
    }


def our_report(root, *, reachability=None, **kwargs):
    """This server's own report file — what the tool waits on after a switch.

    Published as having reached every target unless a test says otherwise: the
    reachability rung is pinned in :class:`TestTheGateRefusals`, and a test
    about what happens after the gate should not have to restate it.
    """
    return write_server_report(
        root,
        os.getpid(),
        reachability=reachable() if reachability is None else reachability,
        **kwargs,
    )


def applied_block(generation: int) -> dict:
    """The report block a landed swap leaves behind."""
    return {"generation": generation, "status": target_state.SWITCH_APPLIED}


def write_marker(target: str, *, pid: int, session: str | None = None):
    """Plant an in-flight execution marker the way the executor writes one."""
    directory = target_state.state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory / f"{control_target.INFLIGHT_FILE_PREFIX}{pid}_"
        f"{uuid.uuid4().hex}{control_target.INFLIGHT_FILE_SUFFIX}"
    )
    marker = {
        "pid": pid,
        "session": session,
        "surface": py_executor.INFLIGHT_SURFACE,
        "kernel_id": None,
        "target": target,
        "started_at": "2026-08-22T10:00:00+00:00",
    }
    path.write_text(json.dumps(marker), encoding="utf-8")
    return path


def config_with_gateways(**kwargs):
    """The harness config, plus the gateways table eligibility asks for.

    The switch-lifecycle harness configures no gateways at all — its children
    serve through connectors that talk to no EPICS anywhere, which is what lets
    them run in a test — and eligibility correctly calls such a target
    unconfigured. Adding a gateways table makes the *config-only* verdict come
    out the way a real deployment's would, which is what the refusal tests are
    about.
    """
    raw = raw_config(**kwargs)
    for block in raw["control_system"]["connector"].values():
        block["gateways"] = {"read_only": {"address": "127.0.0.1", "port": 5064}}
    return raw


#: The port this deployment's stand-in soft IOC serves. Deliberately not 5064:
#: a stand-in on the Channel Access default would let a block that simply never
#: set a port pass the deployed-container check.
STANDIN_PORT = 5074
STANDIN_PROBE = "STANDIN:BEAM:CURRENT"
ACK_HOST = "gw.example.org"


def config_with_a_standin(
    *,
    baseline_standin=False,
    strict_limits=True,
    acknowledged=False,
    recorder=False,
):
    """The gateway harness config, plus the stand-in this deployment co-deploys.

    Three connector blocks, so all three targets resolve: the harness's own live
    and virtual-accelerator blocks, and a ``live_standin`` one dialling the soft
    IOC on loopback at :data:`STANDIN_PORT` — matched by the
    ``services.live_standin.port`` the build projects, which is the evidence the
    deployment stood one up.

    *baseline_standin* makes the stand-in this deployment's *own* machine
    (``control_system.type: live_standin``), which is what puts a deployment on
    ``standin`` with nothing switched and makes ``live`` a destination to be
    gated. The remaining three arguments set the FR-8 facts the live family is
    judged on: the limits posture, the operator acknowledgment, and whether an
    ``archiver_recorder`` makes the archive the stand-in's history.
    """
    raw = config_with_gateways()
    control_system = raw["control_system"]
    control_system["connector"][LIVE_STANDIN] = {
        "probe_channel": STANDIN_PROBE,
        "gateways": {"read_only": {"address": "127.0.0.1", "port": STANDIN_PORT}},
    }
    if baseline_standin:
        control_system["type"] = LIVE_STANDIN
    if strict_limits:
        control_system["limits_checking"] = {"enabled": True, "allow_unlisted_channels": False}
    if acknowledged:
        control_system["target_switch"] = {ACK_LEAF: ACK_HOST}
    raw["services"] = {"live_standin": {"port": STANDIN_PORT}}
    raw["deployed_services"] = [ARCHIVER_RECORDER_SERVICE] if recorder else []
    return raw


def allow_every_target(monkeypatch):
    """Stub the eligibility rung open, to reach the rungs behind it.

    The fixture deployment cannot be both eligible and servable at once: a
    gateways table is what eligibility requires, and a child that configures no
    EPICS gateway is what verification then refuses. Eligibility is pinned on
    its own in :class:`TestTheGateRefusals`, so the tests about what happens
    *after* the gate open it here. Patched on the eligibility module, which is
    where the gate resolves it from — the tool no longer has a gate of its own.
    """
    from osprey.mcp_server.control_system.target_eligibility import TargetAvailability

    def available(config, target, control_target, baseline_target, **kwargs):
        return TargetAvailability(
            target=target,
            eligible=True,
            available_now=True,
            reason=None,
            detail=f"Target {target!r} is available (stubbed for the delegation test).",
            eligible_from_baseline=True,
        )

    monkeypatch.setattr(te, "target_availability", available)


def dead_pid() -> int:
    """A PID whose process has been reaped, so nothing answers to it."""
    finished = subprocess.Popen([sys.executable, "-c", ""])
    finished.wait(timeout=SETTLE_TIMEOUT_S)
    return finished.pid


async def await_request(request_id: str | None = None) -> dict:
    """Wait for this process to have filed a switch request, and return it."""
    deadline = asyncio.get_running_loop().time() + OWNER_PATIENCE_S
    while asyncio.get_running_loop().time() < deadline:
        body = target_state.read_file(target_state.request_file_path())
        if isinstance(body, dict) and (request_id is None or body.get("request_id") == request_id):
            return body
        await asyncio.sleep(TICK_S / 2)
    raise AssertionError("no switch request was filed")


async def owner_answers(
    root,
    *,
    status: str,
    reason: str | None = None,
    detail: str = "",
    mint: bool = True,
    report: bool = True,
) -> dict:
    """Stand in for the record's owner: consume the request, write the terminus.

    Exactly the moves ``SessionControlReconciler._consume_requests`` makes, in
    the order it makes them — the record write first, the unlink after it — so
    a tool that polled the file instead of the record would fail here rather
    than pass by luck. With *report*, this server's own report is published for
    the new generation too, which is what its reconcile loop would have done.
    """
    body = await await_request()
    record = control_context.read_record()
    assert record is not None
    generation = record.generation + 1 if mint else None
    write_control_context(
        root,
        target=body["target"] if mint else record.target,
        generation=generation if mint else record.generation,
        owned_by=record.owner,
        last_switch={
            "request_id": body["request_id"],
            "target": body["target"],
            "requested_at": body["requested_at"],
            "requested_by": f"pid:{body['requested_by_pid']}",
            "status": status,
            "reason": reason,
            "detail": detail,
            "generation": generation,
        },
    )
    target_state.remove_request()
    if mint and report:
        our_report(root, last_switch=applied_block(generation))
    return body


@pytest.fixture
def emitted(monkeypatch):
    """Capture the operator-activity emissions instead of posting them.

    Patched in this module's namespace, which is where the tool resolves the
    emitter from. Every outcome of the tool this process decided produces
    exactly one entry — and a terminus another process wrote produces none.
    """
    calls: list[dict] = []

    async def record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(control_target, "notify_target_switch_async", record)
    return calls


# ------------------------------------------------------------ the gate's own


class TestTheGateRefusals:
    """The gate's ladder, reported by the tool in the gate's own words.

    Asked here through the owning server's path, which is the one that takes
    the verdict itself. That the verdict is identical on the follower's path
    and on the terminal route is ``test_switch_gate.py``'s parity test; what
    this class pins is that the tool reports the verdict rather than
    paraphrasing it.
    """

    async def test_a_readonly_run_is_refused_before_every_other_check(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """First, and regardless of what else would refuse.

        There is an execution in flight AND the wanted target has no probe
        channel, so both later rungs would fire. The read-only posture is still
        the answer: it is the one an operator cannot fix by editing config or
        waiting for a run to end.
        """
        manager = make_manager(raw=raw_config(va_probe=None))
        install_context(manager, monkeypatch)
        owned_here(record_root)
        write_marker("live", pid=os.getpid())
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == control_target.REASON_READONLY_RUN
        assert "read-only sessions stay on the deployment baseline" in envelope["error_message"]
        # The operator who saw the prompt sees the outcome of it too.
        assert emitted == [
            {
                "from_target": "live",
                "to_target": "va",
                "outcome": "failure",
                "reason": control_target.REASON_READONLY_RUN,
            }
        ]

    async def test_an_execution_in_flight_names_the_target_and_the_busy_client(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The words are the eligibility module's, including the attribution.

        A marker carrying no session is named as unattributable rather than
        guessed at — the tool used to compare process ancestry here, which
        could only ever have been right inside one process tree.
        """
        manager = make_manager()
        install_context(manager, monkeypatch)
        owned_here(record_root)
        write_marker("va", pid=os.getpid())

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == control_target.REASON_EXECUTION_IN_FLIGHT
        assert envelope["details"]["executing_target"] == "va"
        assert envelope["details"]["surface"] == py_executor.INFLIGHT_SURFACE
        assert "execution in flight on target 'va'; wait or stop it" in envelope["error_message"]
        assert any(
            "another client sharing this deployment" in line for line in envelope["suggestions"]
        ), envelope["suggestions"]
        assert [call["reason"] for call in emitted] == [control_target.REASON_EXECUTION_IN_FLIGHT]

    async def test_an_execution_in_flight_outranks_an_ineligible_target(
        self, make_manager, monkeypatch, record_root
    ):
        """Order again: the run is the thing to deal with first."""
        manager = make_manager(raw=raw_config(va_probe=None))
        install_context(manager, monkeypatch)
        owned_here(record_root)
        write_marker("live", pid=os.getpid())

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        assert ctx["envelope"]["details"]["reason"] == control_target.REASON_EXECUTION_IN_FLIGHT

    async def test_a_marker_from_a_dead_executor_is_ignored_and_swept(
        self, make_manager, monkeypatch, record_root
    ):
        """One killed executor must not wedge every later switch.

        The marker is named for the process that would remove it, so a PID that
        names nothing is residue — reported as no execution at all, and deleted
        so the directory does not fill with it.
        """
        manager = make_manager(raw=config_with_gateways(va_probe=None))
        install_context(manager, monkeypatch)
        owned_here(record_root)
        stale = write_marker("va", pid=dead_pid())

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        # Fell through to eligibility, which is the proof the marker was ignored.
        assert ctx["envelope"]["details"]["reason"] == REASON_PROBE_CHANNEL_MISSING
        assert not stale.exists()
        assert control_target.in_flight_executions() == []

    async def test_an_ineligible_target_is_refused_in_the_gates_own_words(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The refusal text IS the gate's, character for character.

        Compared against a live ``evaluate_switch`` call rather than a copied
        string, so the pin cannot drift into agreeing with itself. The gate is
        asked with the RECORD's target as the current one, which is the fact
        the tool has to pass and the manager's own binding is not.
        """
        raw = config_with_gateways(va_probe=None)
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        owned_here(record_root, target="live", generation=4)
        expected = te.evaluate_switch(
            raw,
            "va",
            current_target="live",
            baseline=manager.baseline,
            writes_enabled=False,
        )

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["error_message"] == expected.detail
        assert envelope["details"] == expected.details
        assert envelope["details"]["reason"] == REASON_PROBE_CHANNEL_MISSING
        assert [call["reason"] for call in emitted] == [REASON_PROBE_CHANNEL_MISSING]

    async def test_an_unknown_target_is_refused_with_a_reason_and_not_an_exception(
        self, make_manager, monkeypatch, record_root
    ):
        """A target name nothing resolves reaches the gate, not a traceback."""
        manager = make_manager()
        install_context(manager, monkeypatch)
        owned_here(record_root)

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="banana")

        assert ctx["envelope"]["details"]["reason"] == "target_unresolvable"

    async def test_a_target_the_fleet_measured_down_is_refused_as_unreachable(
        self, make_manager, monkeypatch, record_root
    ):
        """The rung the tool could not reach before: a live server's measurement.

        The reports are the live fleet's, gathered by this tool and handed to
        the gate — a refusal here is the deployment's own probe answering, not
        this process's guess about a gateway it never dialled.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        owned_here(record_root)
        our_report(
            record_root,
            reachability={
                "targets": {
                    "va": {
                        "read_only": {
                            "state": te.REACHABILITY_DOWN,
                            "probed_at": datetime.now(UTC).isoformat(),
                            "gateway": "127.0.0.1:5064",
                        }
                    }
                }
            },
        )

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        assert ctx["envelope"]["details"]["reason"] == REASON_TARGET_UNREACHABLE

    async def test_a_refusal_is_a_record_write_that_moves_nothing_else(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The owner files its refusals in the record, target and generation intact.

        That is what lets every other surface — the chip, a second session's
        poll — see why the gesture it is watching ended, instead of watching a
        request id disappear with no answer.
        """
        manager = make_manager(raw=config_with_gateways(va_probe=None))
        install_context(manager, monkeypatch)
        owned_here(record_root, target="live", generation=7)

        with assert_raises_error(error_type=control_target.ERROR_REFUSED):
            await TOOL(target="va")

        record = control_context.read_record()
        assert (record.target, record.generation) == ("live", 7)
        terminus = record.last_switch
        assert terminus["status"] == control_context.SWITCH_REFUSED
        assert terminus["target"] == "va"
        assert terminus["reason"] == REASON_PROBE_CHANNEL_MISSING
        # A refusal names no binding, because there is none to name.
        assert terminus["generation"] is None
        assert terminus["request_id"]


class TestTheStandinIsGatedAsAThirdTarget:
    """SC-4 at the switch: three targets, and the live family split in two.

    The stand-in is a real-machine posture, so it meets the strict limits gate
    the facility's machine meets. It does *not* meet the operator
    acknowledgment: that one is the operator saying the configured gateways
    really are this facility's, and the stand-in's equivalent was said at build
    time by the profile line that stood it up.

    The other half is direction. A deployment may be baselined on
    ``live_standin``, and on such a deployment ``live`` is a switch *away* —
    the exemption that lets a stranded deployment come home follows the
    baseline, and the baseline here is the stand-in.
    """

    async def test_the_standin_needs_the_strict_limits_posture(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """And the refusal names the target, not "the live machine"."""
        raw = config_with_a_standin(strict_limits=False)
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        owned_here(record_root)

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="standin")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == REASON_LIMITS_POSTURE
        assert (
            "Switching to target 'standin' requires the strict limits posture"
            in (envelope["error_message"])
        )
        assert [call["reason"] for call in emitted] == [REASON_LIMITS_POSTURE]

    async def test_the_standin_is_never_asked_for_the_operator_acknowledgment(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """Strict limits and no acknowledgment: the gate lets the stand-in through.

        Proven by the record moving, which is what "was not refused" means now
        that the tool's own answer is a record write.
        """
        raw = config_with_a_standin(strict_limits=True, acknowledged=False)
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        assert ACK_LEAF not in raw["control_system"].get("target_switch", {})
        owned_here(record_root, target="live", generation=1)
        our_report(record_root, last_switch=applied_block(2))

        payload = extract_response_dict(await TOOL(target="standin"))

        assert payload["summary"]["target"] == "standin"
        assert control_context.read_record().target == "standin"

    async def test_going_live_from_a_standin_baseline_is_a_switch_away(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The deployment's own machine is the stand-in, so ``live`` is away.

        If the direction were read as a return, FR-8 would be exempted and this
        unacknowledged deployment would hand a session the facility's real
        machine.
        """
        raw = config_with_a_standin(baseline_standin=True, strict_limits=True, acknowledged=False)
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        assert manager.baseline == "standin"
        owned_here(record_root, target="standin")

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="live")

        assert ctx["envelope"]["details"]["reason"] == REASON_OPERATOR_ACK_MISSING
        # The operator's line says where the deployment actually is.
        assert [call["from_target"] for call in emitted] == ["standin"]

    async def test_going_live_from_a_standin_baseline_also_wants_the_limits_posture(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """Same direction, the earlier of the two away-gates, and target-worded."""
        raw = config_with_a_standin(baseline_standin=True, strict_limits=False, acknowledged=True)
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        owned_here(record_root, target="standin")

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="live")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == REASON_LIMITS_POSTURE
        assert (
            "Switching to target 'live' requires the strict limits posture"
            in (envelope["error_message"])
        )

    async def test_a_recorded_standin_archive_refuses_the_live_machine(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The last gate, and the stand-in's alone to create.

        Acknowledged, strict, and still refused: this deployment runs the
        recorder beside a stand-in, so the store holds the stand-in's past and
        a real machine's readings must not be spliced onto it.
        """
        raw = config_with_a_standin(
            baseline_standin=True, strict_limits=True, acknowledged=True, recorder=True
        )
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        owned_here(record_root, target="standin")

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="live")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == REASON_ARCHIVE_BELONGS_TO_STANDIN
        assert ARCHIVER_RECORDER_SERVICE in envelope["error_message"]


# --------------------------------------------------------- the no-mint answer


class TestTheNoMintAnswer:
    """Asking for the target of record is not a switch, and mints nothing.

    It used to be refused ``already_active``. Under one record per deployment
    it cannot be: the owner answers such a request ``applied`` at the current
    generation, and a tool that refused what the owner grants would give two
    answers to one question. A minted generation would be worse than untidy —
    every write bound to the old one would start refusing, for a switch that
    did not happen.
    """

    async def test_the_target_of_record_is_answered_where_it_stands(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        owned_here(record_root, target="live", generation=5)

        payload = extract_response_dict(await TOOL(target="live"))

        assert payload["status"] == "success"
        assert payload["summary"]["target"] == "live"
        assert payload["summary"]["generation"] == 5
        assert payload["summary"]["target_changed"] is False
        assert "already 'live'" in payload["description"]

    async def test_it_writes_no_record_and_files_no_request(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """Nothing happened, so nothing is recorded as having happened.

        Including the operator's feed: a line for a switch that did not happen
        is a switch in the feed that did not happen.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        path = owned_here(record_root, target="live", generation=5)
        before = path.stat().st_ino

        await TOOL(target="live")

        record = control_context.read_record()
        assert (record.target, record.generation) == ("live", 5)
        assert record.last_switch is None
        assert path.stat().st_ino == before, "the record was rewritten for a no-op"
        assert target_state.read_file(target_state.request_file_path()) is None
        assert emitted == []

    async def test_a_follower_answers_it_without_asking_the_owner(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The owner would answer it the same way, so there is nothing to ask.

        This is the one place the tool answers for the owner, and it is safe
        precisely because the answer moves nothing: two processes agreeing that
        the deployment is where it is cannot disagree about anything.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        owned_elsewhere(record_root, target="va", generation=2)

        payload = extract_response_dict(await TOOL(target="va"))

        assert payload["summary"]["generation"] == 2
        assert target_state.read_file(target_state.request_file_path()) is None


# ------------------------------------------------------ the owner's own path


class TestTheOwningServerAnswersItself:
    """This server owns the record, so it takes the verdict and writes it."""

    async def test_the_record_carries_the_new_target_generation_and_terminus(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_here(record_root, target="live", generation=3)
        our_report(record_root, last_switch=applied_block(4))

        payload = extract_response_dict(await TOOL(target="va"))

        assert payload["summary"]["target"] == "va"
        assert payload["summary"]["generation"] == 4
        assert payload["summary"]["previous_target"] == "live"
        assert payload["summary"]["target_changed"] is True

        record = control_context.read_record()
        assert (record.target, record.generation) == ("va", 4)
        terminus = record.last_switch
        assert terminus["status"] == control_context.SWITCH_APPLIED
        assert terminus["generation"] == 4
        assert terminus["target"] == "va"
        assert terminus["requested_by"] == f"pid:{os.getpid()}"
        assert emitted == [
            {"from_target": "live", "to_target": "va", "outcome": "success", "generation": 4}
        ]

    async def test_the_tool_switches_no_connector_host_itself(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The record's owner mints; the reconcile loop swaps. Neither is this call.

        A tool that still called ``hosts.switch()`` would mint a second
        generation for one gesture and swap before the record said to.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_here(record_root, target="live", generation=0)
        our_report(record_root, last_switch=applied_block(1))

        async def refuse(*args, **kwargs):
            raise AssertionError("the tool must not switch the connector host itself")

        monkeypatch.setattr(manager, "switch", refuse)
        monkeypatch.setattr(manager, "reconcile", refuse)

        payload = extract_response_dict(await TOOL(target="va"))

        assert payload["summary"]["generation"] == 1
        assert manager.active_generation() == 0, "the manager minted a generation of its own"

    async def test_the_answer_waits_for_this_servers_own_report(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The record moving is not yet this session's connector.

        Until this server's own report says it reached the generation, the
        tools in this process are still talking to the previous target — so the
        answer is not "done" yet, and the wait is what makes the agent's next
        call true.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_here(record_root, target="live", generation=0)
        our_report(record_root)

        async def publish_when_asked():
            # What the reconcile loop would do, one tick later.
            await asyncio.sleep(TICK_S * 2)
            our_report(record_root, last_switch=applied_block(1))

        publisher = asyncio.create_task(publish_when_asked())
        payload = extract_response_dict(await TOOL(target="va"))
        await publisher

        assert payload["summary"]["generation"] == 1

    async def test_a_failed_report_is_reported_as_a_failed_switch(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The launch failure this server filed is the answer the agent gets.

        A failure, not a refusal: the switch was granted and the record moved.
        What did not happen is this server's own swap, and the record is left
        exactly as it is — the deployment's target of record is not one
        server's to revoke.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_here(record_root, target="live", generation=0)
        our_report(
            record_root,
            last_switch={
                "generation": 1,
                "status": target_state.SWITCH_FAILED,
                "reason": "probe_failed",
                "detail": "the candidate never answered its probe",
            },
        )

        with assert_raises_error(error_type=control_target.ERROR_FAILED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == "probe_failed"
        assert envelope["error_message"] == "the candidate never answered its probe"
        assert [call["reason"] for call in emitted] == ["probe_failed"]
        record = control_context.read_record()
        assert (record.target, record.generation) == ("va", 1)

    async def test_a_swap_that_never_lands_is_bounded_and_says_so(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """An ``applying`` block that never resolves must not hang the tool.

        The bound is this process's own spawn, probe and drain timeouts — the
        only process that holds them — so a wait that outlives them is a swap
        that is not coming.
        """
        manager = make_manager(
            raw=config_with_gateways(),
            drain_timeout_s=0.01,
            probe_timeout_s=0.01,
            spawn_timeout_s=0.01,
        )
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_here(record_root, target="live", generation=0)
        our_report(
            record_root,
            last_switch={"generation": 1, "status": target_state.SWITCH_APPLYING},
        )

        with assert_raises_error(error_type=control_target.ERROR_FAILED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == control_target.REASON_SWAP_INCOMPLETE
        assert "the swap did not complete" in envelope["error_message"]

    async def test_a_server_with_no_child_lands_as_soon_as_it_has_adopted(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """A silent adoption publishes nothing, and is still a landing.

        A server that has never launched a connector host adopts the record in
        memory and writes no report at all. Waiting for one would hold the tool
        for the whole bound over a swap that was over before it started.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_here(record_root, target="live", generation=0)
        our_report(record_root)
        # What ``poll_once`` does with no live child: adopt, publish nothing.
        await manager.reconcile("va", 1)
        assert target_state.read()["last_switch"] is None

        payload = extract_response_dict(await TOOL(target="va"))

        assert payload["summary"]["generation"] == 1
        assert payload["access_details"]["connector_host_alive"] is False

    async def test_an_unsettled_deployment_refuses_naming_the_pids(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """A second terminus mid-swap would overwrite the answer somebody wants.

        So the deployment is asked whether it has settled BEFORE anything is
        written, and the refusal names the server still applying rather than
        merely saying "busy".
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_here(record_root, target="live", generation=2)
        applying = os.getppid()
        write_server_report(
            record_root,
            applying,
            reachability=reachable("va"),
            last_switch={"generation": 2, "status": target_state.SWITCH_APPLYING},
        )

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == control_target.REASON_SWITCH_IN_PROGRESS
        assert envelope["details"]["pids"] == [applying]
        assert f"switch_in_progress:{applying}" in envelope["error_message"]
        # Nothing was written: the gesture in flight still owns the terminus.
        assert control_context.read_record().last_switch is None


# ----------------------------------------------------------- the follower's


class TestAFollowerFilesARequest:
    """Another process owns the record, so this server asks and waits."""

    async def test_the_request_body_is_the_five_fields_a_consumer_reads(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """Named for the requester, so it is swept with the session waiting on it."""
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_elsewhere(record_root, target="live", generation=0)
        our_report(record_root)

        answering = asyncio.create_task(
            owner_answers(record_root, status=control_context.SWITCH_APPLIED)
        )
        payload = extract_response_dict(await TOOL(target="va"))
        body = await answering

        assert set(body) == {
            "request_id",
            "target",
            "requested_at",
            "session",
            "requested_by_pid",
        }
        assert body["target"] == "va"
        assert body["requested_by_pid"] == os.getpid()
        assert body["request_id"]
        assert payload["summary"]["target"] == "va"
        # Consumed: the owner unlinks the slot once its write is read back.
        assert target_state.read_file(target_state.request_file_path()) is None

    async def test_an_applied_terminus_is_reported_as_the_switch(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The generation is the owner's, relayed — never one this server minted."""
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_elsewhere(record_root, target="live", generation=6)
        our_report(record_root)

        answering = asyncio.create_task(
            owner_answers(record_root, status=control_context.SWITCH_APPLIED)
        )
        payload = extract_response_dict(await TOOL(target="va"))
        await answering

        assert payload["summary"]["target"] == "va"
        assert payload["summary"]["generation"] == 7
        assert payload["summary"]["previous_target"] == "live"
        # The owner emitted the operator's line when it consumed the request;
        # a second one here would read as a second switch.
        assert emitted == []

    async def test_a_refused_terminus_is_reported_in_the_owners_words(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The verdict was taken at the moment the switch would have happened.

        So it is relayed rather than re-derived here: this process's facts are
        a moment older than the owner's, and re-judging could only produce a
        second, differently-worded answer to one gesture.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_elsewhere(record_root, target="live", generation=1)
        our_report(record_root)

        answering = asyncio.create_task(
            owner_answers(
                record_root,
                status=control_context.SWITCH_REFUSED,
                reason=REASON_LIMITS_POSTURE,
                detail="Switching to target 'va' requires the strict limits posture.",
                mint=False,
            )
        )
        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")
        await answering

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == REASON_LIMITS_POSTURE
        assert envelope["error_message"] == (
            "Switching to target 'va' requires the strict limits posture."
        )
        assert emitted == []
        record = control_context.read_record()
        assert (record.target, record.generation) == ("live", 1)

    async def test_the_local_gate_refuses_before_anything_is_filed(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """A read-only run is a claim about THIS run, and only this run can see it.

        The owner evaluates its own execution mode, which is another process's,
        so a request filed from a read-only run would be granted. The gate is
        therefore asked here too — the same gate, with this process's facts —
        and nothing is filed when it refuses.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        owned_elsewhere(record_root, target="live")
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        assert ctx["envelope"]["details"]["reason"] == control_target.REASON_READONLY_RUN
        assert target_state.read_file(target_state.request_file_path()) is None
        # A follower writes nothing to the record, refusal included.
        assert control_context.read_record().last_switch is None

    async def test_an_unsettled_deployment_withdraws_the_request(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """Filed, then found to be unanswerable: the request is taken back.

        Left behind, it would be applied later into a session that had stopped
        watching for it — a target switch nobody is waiting for.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_elsewhere(record_root, target="live", generation=2)
        applying = os.getppid()
        write_server_report(
            record_root,
            applying,
            reachability=reachable("va"),
            last_switch={"generation": 2, "status": target_state.SWITCH_APPLYING},
        )

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == control_target.REASON_SWITCH_IN_PROGRESS
        assert envelope["details"]["pids"] == [applying]
        assert target_state.read_file(target_state.request_file_path()) is None
        assert control_context.read_record().last_switch is None

    async def test_no_owner_consuming_it_withdraws_the_request_and_names_the_owner(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """The bound on a wait for a process that may never answer.

        Five owner ticks: long enough that a busy owner is not given up on,
        short enough that the agent is not left hanging, and well inside the
        request TTL so a withdrawn request was never a stale one.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        wedged = os.getppid()
        owned_elsewhere(record_root, target="live", generation=1, owner_pid=wedged)
        our_report(record_root)

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == control_target.REASON_REQUEST_NOT_CONSUMED
        assert envelope["details"]["owner_pid"] == wedged
        assert f"pid {wedged}" in envelope["error_message"]
        assert "no owner consumed the request" in envelope["error_message"]
        assert target_state.read_file(target_state.request_file_path()) is None, (
            "the withdrawn request was left behind"
        )
        assert [call["reason"] for call in emitted] == [control_target.REASON_REQUEST_NOT_CONSUMED]


# ------------------------------------------------------- one switch at a time


class TestOneSwitchAtATime:
    async def test_a_second_concurrent_call_waits_and_re_reads(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """Two calls, one switch, and the second answers about the world it finds.

        Without the lock both would file into the same request slot — the
        second replacing the first, which would then be watching for a
        ``request_id`` no file carries — and both would mint against the same
        starting generation. With it, the second call runs after the first has
        finished and finds the deployment already where it wanted to go: the
        no-mint answer, not a second switch.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)
        owned_here(record_root, target="live", generation=0)
        our_report(record_root, last_switch=applied_block(1))

        first, second = await asyncio.gather(TOOL(target="va"), TOOL(target="va"))

        payloads = [extract_response_dict(first), extract_response_dict(second)]
        assert [payload["summary"]["target"] for payload in payloads] == ["va", "va"]
        assert [payload["summary"]["generation"] for payload in payloads] == [1, 1]
        # Exactly one of them switched anything.
        assert sorted(payload["summary"]["target_changed"] for payload in payloads) == [
            False,
            True,
        ]
        assert control_context.read_record().generation == 1


class TestEveryDeclineIsVisible:
    """Whatever declines an approved attempt, the operator is told it declined.

    The operator saw a prompt and said yes; a switch that then does not happen
    is the event they need to see, and which internal path produced it is not
    their problem. So every exit but a success emits exactly one failure line —
    with the one deliberate exception of a refusal another process already
    reported when it wrote the terminus.
    """

    async def test_a_missing_server_context_is_reported_as_a_declined_attempt(
        self, monkeypatch, emitted
    ):
        from osprey.mcp_server.control_system import server_context as server_context_mod

        monkeypatch.setattr(server_context_mod, "_registry", None)

        with assert_raises_error(error_type=control_target.ERROR_UNAVAILABLE) as ctx:
            await TOOL(target="va")

        assert ctx["envelope"]["details"]["reason"] == control_target.REASON_CONTEXT_UNAVAILABLE
        # The target is exactly what could not be read, so the line says so
        # rather than guessing one.
        assert emitted == [
            {
                "from_target": control_target.UNKNOWN_TARGET,
                "to_target": "va",
                "outcome": "failure",
                "reason": control_target.REASON_CONTEXT_UNAVAILABLE,
            }
        ]

    async def test_a_deployment_with_no_record_is_refused_rather_than_guessed_at(
        self, make_manager, monkeypatch, emitted, record_root
    ):
        """No record is no target of record, and nothing to switch from.

        Fail-closed on purpose: this server's own baseline is what it would
        have to guess with, and a deployment whose record has not been written
        yet is one whose owner has not started.
        """
        manager = make_manager(raw=config_with_gateways())
        install_context(manager, monkeypatch)
        assert control_context.read_record() is None

        with assert_raises_error(error_type=control_target.ERROR_UNAVAILABLE) as ctx:
            await TOOL(target="va")

        assert ctx["envelope"]["details"]["reason"] == control_target.REASON_RECORD_UNAVAILABLE
        assert [call["reason"] for call in emitted] == [control_target.REASON_RECORD_UNAVAILABLE]
        assert target_state.read_file(target_state.request_file_path()) is None


# ---------------------------------------------- the in-flight marker contract


class TestInFlightMarkerContract:
    def test_both_sides_spell_the_marker_the_same_way(self):
        """The reader and the writer live in different server processes.

        Neither imports the other — the executor pulling in the controls server
        (or the reverse) for two string constants would be a far worse coupling
        than a replica with a drift guard, which is the pattern the deployed
        hooks already use.
        """
        assert control_target.INFLIGHT_FILE_PREFIX == py_executor.INFLIGHT_FILE_PREFIX
        assert control_target.INFLIGHT_FILE_SUFFIX == py_executor.INFLIGHT_FILE_SUFFIX

    def test_the_executors_marker_is_what_the_reader_reads(self):
        """Behavioural half of the drift guard: one writes, the other sees it."""
        assert control_target.in_flight_executions() == []

        with py_executor._in_flight_marker("va"):
            live = control_target.in_flight_executions()
            assert len(live) == 1
            assert live[0]["target"] == "va"
            assert live[0]["pid"] == os.getpid()
            # Attribution travels as the writer's session and surface, not as
            # its ancestry: the reader is not in every writer's process tree.
            assert live[0]["surface"] == py_executor.INFLIGHT_SURFACE
            assert "session" in live[0]
            assert "owner_ppid" not in live[0]

        assert control_target.in_flight_executions() == []

    def test_a_marker_that_cannot_be_written_does_not_fail_the_execution(self, monkeypatch):
        """The run is what the operator asked for; the marker is bookkeeping."""
        monkeypatch.setattr(
            target_state, "state_dir", lambda: (_ for _ in ()).throw(OSError("no state dir"))
        )

        with py_executor._in_flight_marker("va"):
            pass  # no exception is the assertion

    def test_a_failed_write_leaves_no_temp_file_behind(self, monkeypatch, state_root):
        """The rename never happened, so the temp file is this writer's to clean up.

        Without this the state directory would collect one orphan per failed
        write — in the very directory the reader globs.
        """
        directory = target_state.state_dir()
        directory.mkdir(parents=True, exist_ok=True)

        def fail_to_rename(src, dst):
            raise OSError("rename refused")

        monkeypatch.setattr(py_executor.os, "replace", fail_to_rename)

        with py_executor._in_flight_marker("va"):
            pass

        assert sorted(p.name for p in directory.iterdir()) == []

    def test_an_unreadable_marker_is_neither_reported_nor_deleted(self, state_root):
        """It says nothing, and it is not this reader's file to remove."""
        directory = target_state.state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        junk = (
            directory / f"{control_target.INFLIGHT_FILE_PREFIX}nonsense"
            f"{control_target.INFLIGHT_FILE_SUFFIX}"
        )
        junk.write_text("{not json", encoding="utf-8")

        assert control_target.in_flight_executions() == []
        assert junk.exists()


# ------------------------------------------------------------ server startup


@pytest.fixture
def _no_prober(monkeypatch):
    """Keep the module-global prober out of the next test's way."""
    from osprey.mcp_server.control_system import server as server_mod

    monkeypatch.setattr(server_mod, "_prober", None)
    yield
    monkeypatch.setattr(server_mod, "_prober", None)


class RecordingProber:
    """Stands in for the endpoint prober; records its own lifecycle."""

    instances: list[RecordingProber] = []

    def __init__(self, config, **kwargs):
        self.config = config
        self.started = False
        self.stopped = False
        RecordingProber.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


class TestServerStartup:
    async def test_create_server_adopts_the_record_and_sweeps_orphans(
        self, tmp_path, monkeypatch, record_root
    ):
        """Start adopts; it does not reset.

        Three things happen, and each is the wiring a refactor of
        ``create_server`` can silently drop: this server writes its own report,
        the children a dead predecessor left behind are killed, and the record
        is claimed — at the baseline, because there was no record to adopt.

        The report publishes no target at all. Identity lives in the record
        now, and a start-time guess published as ``applied_target`` would let a
        reader count a server that has launched nothing as arrived. The
        ``targets`` mapping it does publish is the fail-closed slot set — one
        slot per name in :data:`target_state.TARGET_NAMES` — because its
        readers are hooks that render an identity line and must never have to
        branch on a missing key.
        """
        from osprey.mcp_server.control_system import connector_host_manager
        from osprey.mcp_server.control_system import server as server_mod

        gone = dead_pid()
        write_server_report(record_root, gone, children=[4242])
        swept: list[list[int]] = []
        monkeypatch.setattr(
            connector_host_manager, "kill_orphans", lambda pids, **kw: swept.append(list(pids))
        )

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            "control_system:\n  type: mock\n  writes_enabled: false\n"
            "archiver:\n  type: mongodb_archiver\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(config_file))
        monkeypatch.chdir(tmp_path)

        server_mod.create_server()

        report = target_state.read()
        assert report is not None, "create_server must write this server's report"
        assert report["server_pid"] == os.getpid()
        assert report["applied_target"] is None
        assert report["applied_generation"] is None
        assert set(report["targets"]) == set(target_state.TARGET_NAMES)
        assert set(report["targets"]) == {"live", "va", "standin"}
        # Present, and describing nothing: this deployment stood no stand-in up,
        # so the slot carries neither an endpoint to dial nor a channel to prove
        # one with.
        standin_slot = report["targets"]["standin"]
        assert standin_slot["endpoint"] == ""
        assert "probe_channel" not in standin_slot
        assert swept == [[4242]], "the orphan recorded by the dead server was not swept"

        record = control_context.read_record()
        assert record is not None, "create_server must claim the control context"
        assert (record.target, record.generation) == ("live", 0)
        assert record.owner.pid == os.getpid()
        assert record.owner.kind == control_context.OWNER_CONTROLS_SERVER

    async def test_the_lifespan_runs_the_endpoint_prober(self, monkeypatch, _no_prober):
        """The prober needs a running loop, so the lifespan owns it, not create_server."""
        from osprey.mcp_server.control_system import endpoint_prober
        from osprey.mcp_server.control_system import server as server_mod

        RecordingProber.instances.clear()
        monkeypatch.setattr(endpoint_prober, "EndpointProber", RecordingProber)
        context = ControlSystemContext()
        context._config = type("Config", (), {"raw": {"control_system": {"type": "mock"}}})()
        monkeypatch.setattr("osprey.mcp_server.control_system.server_context._registry", context)

        async with server_mod._lifespan(server_mod.mcp):
            assert len(RecordingProber.instances) == 1
            prober = RecordingProber.instances[0]
            assert prober.started is True
            assert server_mod.get_endpoint_prober() is prober

        assert prober.stopped is True
        assert server_mod.get_endpoint_prober() is None

    async def test_a_prober_that_will_not_start_does_not_stop_the_server(
        self, monkeypatch, _no_prober
    ):
        """Reachability rows are a convenience; serving tools is not."""
        from osprey.mcp_server.control_system import endpoint_prober
        from osprey.mcp_server.control_system import server as server_mod

        def explode(*args, **kwargs):
            raise RuntimeError("no prober today")

        monkeypatch.setattr(endpoint_prober, "EndpointProber", explode)

        assert await server_mod.start_background() is None
        assert server_mod.get_endpoint_prober() is None

    def test_the_server_is_constructed_with_that_lifespan(self):
        """Otherwise nothing would ever enter it.

        Read off the FastMCP instance's own attribute: "was wired at
        construction" has no public spelling, and asserting it here is what
        keeps the two halves of the wiring from drifting apart.
        """
        from osprey.mcp_server.control_system import server as server_mod

        assert server_mod.mcp._lifespan is server_mod._lifespan
