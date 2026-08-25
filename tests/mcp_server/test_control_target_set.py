"""The ``control_target_set`` tool: three refusals in order, then a delegation.

The switch itself is pinned by ``test_switch_lifecycle.py`` — real children,
real processes, real spawn-then-swap. What is pinned HERE is the gate in front
of it, and the gate is almost entirely about order and about wording:

* a read-only run is refused FIRST, even when every later check would also
  refuse, because "you cannot switch at all" is the true answer and sending the
  operator off to fix a config key would not be;
* an execution in flight is seen ACROSS a process boundary, through the marker
  file the python executor writes — including the case that makes such a
  mechanism dangerous, a marker left behind by an executor that was killed;
* an ineligible target is refused in the eligibility module's OWN words, so the
  refusal and the roster row can never disagree about why.

The manager fixtures are imported from ``test_switch_lifecycle`` rather than
rebuilt: the two targets it constructs (a mock connector by dotted path for
``live``, a fixture connector registered as ``virtual_accelerator`` for ``va``)
are exactly what a delegation test needs, and a second copy of them would be a
second thing to keep true.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

import pytest

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.server_context import ControlSystemContext
from osprey.mcp_server.control_system.target_eligibility import (
    REASON_ALREADY_ACTIVE,
    REASON_PROBE_CHANNEL_MISSING,
    target_availability,
)
from osprey.mcp_server.control_system.tools import control_target
from osprey.mcp_server.python_executor import executor as py_executor
from tests.mcp_server import test_switch_lifecycle as switch_suite
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

DEAD_WRITE_PORT = switch_suite.DEAD_WRITE_PORT
GATEWAY_HOST = switch_suite.GATEWAY_HOST
REFUSE_CHANNEL = switch_suite.REFUSE_CHANNEL
SETTLE_TIMEOUT_S = switch_suite.SETTLE_TIMEOUT_S
VA_PROBE = switch_suite.VA_PROBE
gateway_config = switch_suite.gateway_config
raw_config = switch_suite.raw_config
started_on = switch_suite.started_on

# The switch-lifecycle suite's fixtures, rebound so pytest collects them in this
# module too. ``state_root`` and ``child_environment`` are autouse there and stay
# autouse here, which is what anchors every state file this module writes under
# tmp_path. Rebound rather than imported by name so that a test's fixture
# parameter does not read as a shadowed import.
child_environment = switch_suite.child_environment
fixture_dir = switch_suite.fixture_dir
make_manager = switch_suite.make_manager
state_root = switch_suite.state_root
write_armed_project = switch_suite.write_armed_project

TOOL = get_tool_fn(control_target.control_target_set)


# ------------------------------------------------------------------ helpers


def install_context(manager, monkeypatch) -> ControlSystemContext:
    """Make *manager*'s deployment the server context the tool will read."""
    from osprey.mcp_server.control_system import server_context as server_context_mod

    context = ControlSystemContext()
    context._config = manager._config
    context._connector_hosts = manager
    monkeypatch.setattr(server_context_mod, "_registry", context)
    return context


def write_marker(target: str, *, pid: int, owner_ppid: int | None = None):
    """Plant an in-flight execution marker the way the executor writes one."""
    directory = target_state.state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory / f"{control_target.INFLIGHT_FILE_PREFIX}{pid}_"
        f"{uuid.uuid4().hex}{control_target.INFLIGHT_FILE_SUFFIX}"
    )
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "owner_ppid": os.getppid() if owner_ppid is None else owner_ppid,
                "target": target,
                "started_at": "2026-08-22T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def config_with_gateways(**kwargs):
    """The harness config, plus the gateways table eligibility asks for.

    The switch-lifecycle harness configures no gateways at all — its children
    serve through connectors that talk to no EPICS anywhere, which is what lets
    them run in a test — and eligibility correctly calls such a target
    unconfigured. Adding a gateways table makes the *config-only* verdict come
    out the way a real deployment's would, which is what the refusal tests are
    about; it does not make the fixture children verifiable (see
    :meth:`TestDelegation.test_a_successful_switch_returns_the_new_target_and_generation`).
    """
    raw = raw_config(**kwargs)
    for block in raw["control_system"]["connector"].values():
        block["gateways"] = {"read_only": {"address": "127.0.0.1", "port": 5064}}
    return raw


def allow_every_target(monkeypatch):
    """Stub the eligibility gate open, to reach the switch behind it.

    The fixture deployment cannot be both eligible and servable at once: a
    gateways table is what eligibility requires, and a child that configures no
    EPICS gateway is what verification then refuses. Eligibility is pinned on
    its own in :class:`TestRefusalOrder`, so the delegation tests open it here
    and exercise the half the refusal tests cannot reach.
    """
    from osprey.mcp_server.control_system.target_eligibility import TargetAvailability

    def available(config, target, session_target, baseline_target, **kwargs):
        return TargetAvailability(
            target=target,
            eligible=True,
            available_now=True,
            reason=None,
            detail=f"Target {target!r} is available (stubbed for the delegation test).",
            eligible_from_baseline=True,
        )

    monkeypatch.setattr(control_target, "target_availability", available)


def dead_pid() -> int:
    """A PID whose process has been reaped, so nothing answers to it."""
    finished = subprocess.Popen([sys.executable, "-c", ""])
    finished.wait(timeout=SETTLE_TIMEOUT_S)
    return finished.pid


@pytest.fixture
def emitted(monkeypatch):
    """Capture the operator-activity emissions instead of posting them.

    Patched in this module's namespace, which is where the tool resolves the
    emitter from. Every outcome of the tool — including each refusal — must
    produce exactly one entry.
    """
    calls: list[dict] = []

    async def record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(control_target, "notify_target_switch_async", record)
    return calls


# ------------------------------------------------------------ refusal order


class TestRefusalOrder:
    async def test_a_readonly_run_is_refused_before_every_other_check(
        self, make_manager, monkeypatch, emitted
    ):
        """First, and regardless of what else would refuse.

        The session here has an execution in flight AND is asking for a target
        with no probe channel, so both later checks would fire. The read-only
        posture still has to be the answer: it is the one an operator cannot
        fix by editing config or waiting for a run to end.
        """
        manager = make_manager(raw=raw_config(va_probe=None))
        install_context(manager, monkeypatch)
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

    async def test_an_execution_in_flight_names_the_target_it_is_running_on(
        self, make_manager, monkeypatch, emitted
    ):
        manager = make_manager()
        install_context(manager, monkeypatch)
        write_marker("va", pid=os.getpid())

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["details"]["reason"] == control_target.REASON_EXECUTION_IN_FLIGHT
        assert envelope["details"]["executing_target"] == "va"
        assert "execution in flight on target 'va'; wait or stop it" in envelope["error_message"]
        assert [call["reason"] for call in emitted] == [control_target.REASON_EXECUTION_IN_FLIGHT]

    async def test_an_execution_in_flight_outranks_an_ineligible_target(
        self, make_manager, monkeypatch
    ):
        """Order again: the run is the thing to deal with first."""
        manager = make_manager(raw=raw_config(va_probe=None))
        install_context(manager, monkeypatch)
        write_marker("live", pid=os.getpid())

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        assert ctx["envelope"]["details"]["reason"] == control_target.REASON_EXECUTION_IN_FLIGHT

    async def test_a_marker_from_a_dead_executor_is_ignored_and_swept(
        self, make_manager, monkeypatch
    ):
        """One killed executor must not wedge every later switch.

        The marker is named for the process that would remove it, so a PID that
        names nothing is residue — reported as no execution at all, and deleted
        so the directory does not fill with it.
        """
        manager = make_manager(raw=config_with_gateways(va_probe=None))
        install_context(manager, monkeypatch)
        stale = write_marker("va", pid=dead_pid())

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        # Fell through to eligibility, which is the proof the marker was ignored.
        assert ctx["envelope"]["details"]["reason"] == REASON_PROBE_CHANNEL_MISSING
        assert not stale.exists()
        assert control_target.in_flight_executions() == []

    async def test_an_ineligible_target_is_refused_in_the_rosters_own_words(
        self, make_manager, monkeypatch, emitted
    ):
        """The refusal text IS the eligibility module's, character for character.

        Two surfaces answer "why can this session not go there" — this tool and
        the roster — and they answer it from one function. Compared against a
        live call rather than a copied string so the pin cannot drift into
        agreeing with itself.
        """
        raw = config_with_gateways(va_probe=None)
        manager = make_manager(raw=raw)
        install_context(manager, monkeypatch)
        expected = target_availability(raw, "va", manager.active_target(), manager.baseline)

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="va")

        envelope = ctx["envelope"]
        assert envelope["error_message"] == expected.detail
        assert envelope["details"] == expected.as_dict()
        assert envelope["details"]["reason"] == REASON_PROBE_CHANNEL_MISSING
        assert [call["reason"] for call in emitted] == [REASON_PROBE_CHANNEL_MISSING]

    async def test_the_active_target_is_refused_as_already_active(self, make_manager, monkeypatch):
        """Switching to where the session already is is a no-op, and says so."""
        manager = make_manager()
        install_context(manager, monkeypatch)

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target=manager.active_target())

        assert ctx["envelope"]["details"]["reason"] == REASON_ALREADY_ACTIVE

    async def test_an_unknown_target_is_refused_with_a_reason_and_not_an_exception(
        self, make_manager, monkeypatch
    ):
        """A target name nothing resolves reaches eligibility, not a traceback."""
        manager = make_manager()
        install_context(manager, monkeypatch)

        with assert_raises_error(error_type=control_target.ERROR_REFUSED) as ctx:
            await TOOL(target="banana")

        assert ctx["envelope"]["details"]["reason"] == "target_unresolvable"


class TestEveryDeclineIsVisible:
    """Whatever declines an approved attempt, the operator is told it declined.

    The operator saw a prompt and said yes; a switch that then does not happen
    is the event they need to see, and which internal path produced it is not
    their problem. So every exit but success emits a failure line — including
    the two nobody plans for.
    """

    async def test_a_missing_server_context_is_reported_as_a_declined_attempt(
        self, monkeypatch, emitted
    ):
        from osprey.mcp_server.control_system import server_context as server_context_mod

        monkeypatch.setattr(server_context_mod, "_registry", None)

        with assert_raises_error(error_type=control_target.ERROR_UNAVAILABLE) as ctx:
            await TOOL(target="va")

        assert ctx["envelope"]["details"]["reason"] == control_target.REASON_CONTEXT_UNAVAILABLE
        # The session target is exactly what could not be read, so the line
        # says so rather than guessing one.
        assert emitted == [
            {
                "from_target": control_target.UNKNOWN_TARGET,
                "to_target": "va",
                "outcome": "failure",
                "reason": control_target.REASON_CONTEXT_UNAVAILABLE,
            }
        ]

    async def test_an_unclassified_failure_is_reported_and_still_raised(
        self, make_manager, monkeypatch, emitted
    ):
        """A bug in the switch must not also silently lose the operator's report."""
        manager = make_manager()
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)

        async def explode(target):
            raise RuntimeError("something nobody classified")

        monkeypatch.setattr(manager, "switch", explode)

        with pytest.raises(RuntimeError, match="something nobody classified"):
            await TOOL(target="va")

        assert [call["reason"] for call in emitted] == [control_target.REASON_INTERNAL_ERROR]


# --------------------------------------------------------------- delegation


class TestDelegation:
    async def test_a_successful_switch_returns_the_new_target_and_generation(
        self, make_manager, monkeypatch, emitted
    ):
        manager = await started_on(make_manager, "live")
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)

        payload = extract_response_dict(await TOOL(target="va"))

        assert payload["status"] == "success"
        assert payload["summary"]["target"] == "va"
        assert payload["summary"]["generation"] == 1
        assert payload["summary"]["previous_target"] == "live"
        assert payload["summary"]["target_changed"] is True
        assert payload["summary"]["probe_channel"] == VA_PROBE
        assert payload["access_details"]["child_pid"] == manager.status()["child_pid"]
        # The delegation really moved the session, not just the report.
        assert manager.active_target() == "va"
        assert target_state.read()["target"] == "va"
        assert emitted == [
            {
                "from_target": "live",
                "to_target": "va",
                "outcome": "success",
                "generation": 1,
            }
        ]

    async def test_a_fallback_landing_reports_success_with_the_warning(
        self, make_manager, monkeypatch, write_armed_project
    ):
        """Issue #718 at the tool surface: home again, and told at what cost.

        The deployment is write-armed and its write gateway is configured but
        dead, so the return to baseline lands through the read gateway. That
        is a success — the session is home — but not a silent one: the
        response carries the fallback and the description warns about it. No
        eligibility stub here: a dead gateway is exactly what the config-only
        verdict cannot see, so the real gate waves this switch through.
        """
        manager = make_manager(raw=gateway_config(), config_path=write_armed_project)
        install_context(manager, monkeypatch)
        await manager.ensure_started()
        await manager.switch("va")

        payload = extract_response_dict(await TOOL(target="live"))

        assert payload["status"] == "success"
        assert payload["summary"]["target"] == "live"
        assert payload["access_details"]["selected_role"] == "read_only"
        fallback = payload["access_details"]["write_gateway_fallback"]
        assert fallback["port"] == DEAD_WRITE_PORT
        assert "WARNING" in payload["description"]
        assert "read_only" in payload["description"]
        assert manager.active_target() == "live"

    async def test_a_probe_failure_names_the_gateway_in_details_and_suggestions(
        self, make_manager, monkeypatch, write_armed_project
    ):
        """Issue #718, part two, at the tool surface.

        Both roles usually share a hostname and differ only by port, so a
        refusal naming only the probe channel misreads as "the control system
        is down". The error details carry the probed gateway's role, host and
        port, and a suggestion tells the operator to check that endpoint —
        not the control system beside it.
        """
        manager = make_manager(
            raw=gateway_config(read_gateway=False), config_path=write_armed_project
        )
        install_context(manager, monkeypatch)
        await manager.ensure_started()
        await manager.switch("va")

        with assert_raises_error(error_type=control_target.ERROR_FAILED) as ctx:
            await TOOL(target="live")

        details = ctx["envelope"]["details"]
        assert details["gateway"] == {
            "role": "write_access",
            "host": GATEWAY_HOST,
            "port": DEAD_WRITE_PORT,
        }
        suggestions = ctx["envelope"]["suggestions"]
        assert any(
            "write_access" in line and f"{GATEWAY_HOST}:{DEAD_WRITE_PORT}" in line
            for line in suggestions
        ), suggestions
        assert manager.active_target() == "va"

    async def test_a_switch_error_becomes_the_structured_error(
        self, make_manager, monkeypatch, emitted
    ):
        """A destination that spawns but cannot be proven leaves the session put.

        The VA probe channel here is one the fixture connector refuses, so the
        switch fails at its readiness probe — the stage a target reaches only
        after passing eligibility, which is exactly the failure the tool has to
        map rather than pre-empt.
        """
        manager = await started_on(make_manager, "live", raw=raw_config(va_probe=REFUSE_CHANNEL))
        install_context(manager, monkeypatch)
        allow_every_target(monkeypatch)

        with assert_raises_error(error_type=control_target.ERROR_FAILED) as ctx:
            await TOOL(target="va")

        details = ctx["envelope"]["details"]
        assert details["target"] == "va"
        assert details["stage"] == "probe"
        assert details["reason"] == "probe_failed"
        assert manager.active_target() == "live"
        assert manager.has_child() is True
        # A switch that did not happen is still an outcome the operator sees.
        assert emitted == [
            {
                "from_target": "live",
                "to_target": "va",
                "outcome": "failure",
                "reason": "probe_failed",
            }
        ]


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
            assert live[0]["owner_ppid"] == os.getppid()

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
    async def test_create_server_publishes_the_baseline_and_sweeps_orphans(
        self, tmp_path, monkeypatch, state_root
    ):
        """Startup resets the state file and kills what a dead server left.

        The kill itself is the manager's, and pinned where it lives; what is
        pinned here is that server startup *runs* it — the wiring, which is the
        part a refactor of ``create_server`` can silently drop.
        """
        from osprey.mcp_server.control_system import connector_host_manager
        from osprey.mcp_server.control_system import server as server_mod

        stale_dir = state_root / target_state.STATE_DIR_NAME
        stale_dir.mkdir(parents=True, exist_ok=True)
        gone = dead_pid()
        (stale_dir / f"target_state_{gone}.json").write_text(
            json.dumps(
                {
                    "target": "va",
                    "generation": 3,
                    "server_pid": gone,
                    "owner_ppid": 1,
                    "targets": {},
                    "children": [4242],
                }
            ),
            encoding="utf-8",
        )
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

        record = target_state.read()
        assert record is not None, "create_server must publish this server's target state"
        assert record["target"] == "live"
        assert record["generation"] == 0
        assert set(record["targets"]) == {"live", "va"}
        assert swept == [[4242]], "the orphan recorded by the dead server was not swept"

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
