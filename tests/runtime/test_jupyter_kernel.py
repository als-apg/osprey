"""The notebook kernel launcher: what it takes away, and what it must not pull in.

The executor's half of the routing contract is exercised in
``test_executor_target_stamp.py``; what is pinned here is the kernel's own
jobs: preparing the process before a kernel exists, routing every cell from
the deployment's record, and staying importable in processes that have no
kernel stack at all.

The two hooks are exercised in full: a stub shell stands in for ``IPython``'s,
so the refusal handler can be fired with a real refusal and the cell hooks
called directly, without a kernel behind either.

Nothing here starts a kernel. ``main()`` is the five statements after the
preparation, and running them would hand the test session's stdio to
``ipykernel``; the preparation is called directly instead, and the one test
that does call ``main()`` puts a stub in ``IPKernelApp``'s place.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey import jupyter_kernel, runtime
from osprey.audit import posture
from osprey.audit.envelope import DECISION_REFUSED
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.python_executor import executor
from osprey.runtime import ControlTargetChangedError, SwitchInProgressError
from osprey_connectors import posture_store
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError

#: The id Jupyter put in this kernel's connection-file name.
KERNEL_ID = "4f1c2a7e0000400080000000000002"


@pytest.fixture
def kernel_env(monkeypatch):
    """This process's environment, restored whole afterwards.

    The launcher stamps ``os.environ`` and nothing else: the resolvers it calls
    read the process environment, so a dict handed in would be invisible to
    them.
    """
    saved = dict(os.environ)
    yield os.environ
    os.environ.clear()
    os.environ.update(saved)
    posture_store.invalidate_cache()


def connection_argv(root: Path, name: str = f"kernel-{KERNEL_ID}.json") -> list[str]:
    """The arguments the kernelspec hands a kernel started under *root*."""
    return ["-f", str(root / name)]


class TestPreparingTheProcess:
    """What the kernel's environment holds by the time a cell can read it."""

    def test_the_server_token_is_removed(self, tmp_path, kernel_env, monkeypatch):
        """A cell must find no ``JUPYTER_TOKEN`` key, not an empty one.

        The kernelspec carries the name with an empty value and the provisioner
        merges that env OVER ``os.environ``, so setting it empty is what puts
        the key there. Only a pop takes it away.
        """
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.JUPYTER_TOKEN_ENV_VAR, "")

        jupyter_kernel._prepare_environment()

        assert jupyter_kernel.JUPYTER_TOKEN_ENV_VAR not in kernel_env

    def test_a_real_token_is_removed_too(self, tmp_path, kernel_env, monkeypatch):
        """The inherited value, not only the kernelspec's empty one."""
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.JUPYTER_TOKEN_ENV_VAR, "not-for-cells")

        jupyter_kernel._prepare_environment()

        assert jupyter_kernel.JUPYTER_TOKEN_ENV_VAR not in kernel_env

    def test_the_connection_file_names_the_session_stamp(self, tmp_path, kernel_env, monkeypatch):
        """A kernel is its own audit session, named for its connection file.

        Jupyter writes ``kernel-<kernel_id>.json`` per kernel, and that id is
        the only handle the process has on which kernel it is.
        """
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))

        stamps = jupyter_kernel._prepare_environment(connection_argv(tmp_path))

        assert stamps[posture.POSTURE_SESSION_ENV_VAR] == f"kernel:{KERNEL_ID}"
        assert kernel_env[posture.POSTURE_SESSION_ENV_VAR] == f"kernel:{KERNEL_ID}"

    def test_the_joined_flag_names_the_session_stamp_too(self, tmp_path, kernel_env, monkeypatch):
        """``-f=<path>`` is the other spelling ``traitlets`` accepts."""
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        path = tmp_path / f"kernel-{KERNEL_ID}.json"

        stamps = jupyter_kernel._prepare_environment([f"-f={path}"])

        assert stamps[posture.POSTURE_SESSION_ENV_VAR] == f"kernel:{KERNEL_ID}"

    def test_an_unusual_connection_file_still_yields_a_session_stamp(
        self, tmp_path, kernel_env, monkeypatch
    ):
        """A name outside Jupyter's shape is used whole, not refused.

        A kernel that cannot say which one it is is worse than one whose id
        spells something unexpected.
        """
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))

        stamps = jupyter_kernel._prepare_environment(connection_argv(tmp_path, "handmade.json"))

        assert stamps[posture.POSTURE_SESSION_ENV_VAR] == "kernel:handmade"

    def test_an_unnamed_kernel_carries_no_session_stamp(self, tmp_path, kernel_env, monkeypatch):
        """No connection file, no session — and no inherited one either.

        Whatever started the sidecar may carry a session of its own, and
        wearing it would file this kernel's records under another process.
        """
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(posture.POSTURE_SESSION_ENV_VAR, "someone-elses-session")

        stamps = jupyter_kernel._prepare_environment([])

        assert posture.POSTURE_SESSION_ENV_VAR not in stamps
        assert posture.POSTURE_SESSION_ENV_VAR not in kernel_env

    def test_the_session_stamp_leaves_the_kernel_pinned_sandboxed(
        self, tmp_path, kernel_env, monkeypatch
    ):
        """A named kernel is still a sandboxed one: nothing here picks a target.

        Any target stamp inherited from whatever started the sidecar goes,
        because passing one through would route cells at a target nobody
        selected here.
        """
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        for name in executor._STAMP_ENV_NAMES:
            monkeypatch.setenv(name, "inherited")

        stamps = jupyter_kernel._prepare_environment(connection_argv(tmp_path))

        assert stamps[posture_store.LAUNCH_POSTURE_ENV_VAR] == "*=sandbox"
        assert kernel_env[posture_store.LAUNCH_POSTURE_ENV_VAR] == "*=sandbox"
        assert not [name for name in executor._STAMP_ENV_NAMES if name in kernel_env]

    def test_the_config_path_is_published_under_the_name_the_loader_reads(
        self, tmp_path, kernel_env, monkeypatch
    ):
        """``OSPREY_CONFIG`` is what the sidecar passes; ``CONFIG_FILE`` is what loads.

        The kernel's working directory is the notebooks folder, so without the
        second name every runtime call in a cell resolves the defaults.
        """
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.OSPREY_CONFIG_ENV_VAR, str(tmp_path / "config.yml"))
        monkeypatch.delenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, raising=False)

        jupyter_kernel._prepare_environment()

        assert kernel_env[jupyter_kernel.CONFIG_FILE_ENV_VAR] == str(tmp_path / "config.yml")

    def test_an_existing_config_file_is_not_overridden(self, tmp_path, kernel_env, monkeypatch):
        """Whoever set ``CONFIG_FILE`` chose it; the sidecar's name only fills a gap."""
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.OSPREY_CONFIG_ENV_VAR, str(tmp_path / "config.yml"))
        monkeypatch.setenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, str(tmp_path / "chosen.yml"))

        jupyter_kernel._prepare_environment()

        assert kernel_env[jupyter_kernel.CONFIG_FILE_ENV_VAR] == str(tmp_path / "chosen.yml")

    def test_a_blank_config_file_is_filled_like_an_absent_one(
        self, tmp_path, kernel_env, monkeypatch
    ):
        """An empty value is not a choice — the loader reads it as unset.

        The provisioner merges the kernelspec's env over ``os.environ``, so a
        name carried empty arrives as an empty value rather than as no name at
        all. Keeping it would leave every runtime call in a cell on the
        defaults, which is the case the publication exists to prevent.
        """
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.OSPREY_CONFIG_ENV_VAR, str(tmp_path / "config.yml"))
        monkeypatch.setenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, "")

        jupyter_kernel._prepare_environment()

        assert kernel_env[jupyter_kernel.CONFIG_FILE_ENV_VAR] == str(tmp_path / "config.yml")

    def test_no_config_path_publishes_nothing(self, tmp_path, kernel_env, monkeypatch):
        """Neither name set: the launcher invents no path, empty or otherwise."""
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.delenv(jupyter_kernel.OSPREY_CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, raising=False)

        jupyter_kernel._prepare_environment()

        assert jupyter_kernel.CONFIG_FILE_ENV_VAR not in kernel_env


def test_importing_the_module_pulls_in_no_kernel_stack():
    """The launcher is imported by the web terminal, which has no kernel.

    ``ipykernel`` is imported inside ``main()`` for this reason, and the web
    terminal is never imported here at all — the dependency runs one way only.
    A subprocess is the only honest way to ask: this test session has both
    packages imported already.
    """
    probe = (
        "import sys; import osprey.jupyter_kernel; "
        "print(','.join(n for n in "
        "('ipykernel', 'fastapi', 'osprey.interfaces.web_terminal') if n in sys.modules))"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == ""


class StubEvents:
    """As much of ``IPython``'s event registry as the cell hooks touch."""

    def __init__(self):
        self.registered = []

    def register(self, name, function):
        self.registered.append((name, function))


class StubShell:
    """As much of an ``InteractiveShell`` as the hooks touch."""

    def __init__(self):
        self.registered = None
        self.shown = []
        self.events = StubEvents()

    def set_custom_exc(self, exc_tuple, handler):
        self.registered = (exc_tuple, handler)

    def showtraceback(self, exc_tuple=None, tb_offset=None):
        self.shown.append((exc_tuple, tb_offset))


@pytest.fixture
def shell():
    """A stub shell, fresh per test."""
    return StubShell()


@pytest.fixture
def audit_records(monkeypatch):
    """Every field set the handler hands the audit writer, in order."""
    written = []

    def fake_record(**fields):
        written.append(fields)
        return None

    monkeypatch.setattr("osprey.audit.writer.record", fake_record)
    return written


@pytest.fixture
def unstamped(monkeypatch):
    """No launch pin and no target stamp — what every hint case narrows from."""
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.delenv(posture.CONTROL_TARGET_ENV_VAR, raising=False)
    return monkeypatch


def raised(error):
    """*error* with a real traceback, in the triple the shell's hook receives."""
    try:
        raise error
    except Exception:
        return type(error), error, sys.exc_info()[2]


def fire(shell, error, tb_offset=None):
    """Run the handler over *error* and hand back the triple it was given."""
    triple = raised(error)
    jupyter_kernel._refusal_handler(shell, *triple, tb_offset=tb_offset)
    return triple


#: One instance of each class the hook answers for, by the name a test reads.
REFUSALS = {
    "write_blocked": lambda: ChannelWriteBlockedError("SR:MAG:1", "WRITES_DISABLED"),
    "limits": lambda: ChannelLimitsViolationError("SR:MAG:1", 42.0, "range", "above maximum"),
    "target_changed": lambda: ControlTargetChangedError("the target moved"),
}


class TestInstallingTheHook:
    """What the shell is asked to trap, and where the asking happens."""

    def test_the_three_refusal_classes_are_registered_as_a_tuple(self, shell):
        """``set_custom_exc`` rejects a list, and a subclass check needs a tuple."""
        jupyter_kernel.install_refusal_handler(shell)

        exc_tuple, handler = shell.registered
        assert isinstance(exc_tuple, tuple)
        assert set(exc_tuple) == {
            ChannelWriteBlockedError,
            ChannelLimitsViolationError,
            ControlTargetChangedError,
        }
        assert handler is jupyter_kernel._refusal_handler

    def test_both_cell_hooks_are_registered(self, shell):
        """A cell that is opened and never closed leaves its markers behind."""
        jupyter_kernel.install_cell_hooks(shell)

        assert shell.events.registered == [
            ("pre_run_cell", jupyter_kernel.pre_run_cell),
            ("post_run_cell", jupyter_kernel.post_run_cell),
        ]

    def test_the_hook_goes_on_between_initialize_and_start(self, monkeypatch):
        """``initialize`` is what builds the shell; ``start`` does not return.

        So the hooks have exactly one place to go, and a chained call would
        have left them nowhere. The registry is loaded before any of that, from the
        config path the preparation publishes, and the call is the executor
        sandbox's: no export, the path from ``CONFIG_FILE``. Log routing comes
        before all of it, so that what the preparation and the registry log is
        already going to the terminal.
        """
        order = []
        registry_calls = []
        config_path = "/deployment/config.yml"

        def prepare(argv):
            order.append("prepare")
            monkeypatch.setenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, config_path)
            return {}

        def initialize_registry(**kwargs):
            order.append("initialize_registry")
            registry_calls.append(kwargs)

        class RecordingEvents(StubEvents):
            def register(self, name, function):
                order.append(f"register:{name}")
                super().register(name, function)

        class RecordingShell(StubShell):
            def __init__(self):
                super().__init__()
                self.events = RecordingEvents()

            def set_custom_exc(self, exc_tuple, handler):
                order.append("set_custom_exc")
                super().set_custom_exc(exc_tuple, handler)

        class StubKernelApp:
            @classmethod
            def instance(cls):
                app = cls()
                app.shell = RecordingShell()
                return app

            def initialize(self, argv):
                order.append("initialize")

            def start(self):
                order.append("start")

        monkeypatch.setattr(
            jupyter_kernel, "_route_logs_to_process_stderr", lambda: order.append("route_logs")
        )
        monkeypatch.setattr(jupyter_kernel, "_prepare_environment", prepare)
        monkeypatch.setattr("osprey.registry.initialize_registry", initialize_registry)
        monkeypatch.setattr("ipykernel.kernelapp.IPKernelApp", StubKernelApp)

        jupyter_kernel.main([])

        assert order == [
            "route_logs",
            "prepare",
            "initialize_registry",
            "initialize",
            "set_custom_exc",
            "register:pre_run_cell",
            "register:post_run_cell",
            "start",
        ]
        assert registry_calls == [{"auto_export": False, "config_path": config_path}]

    def test_a_registry_that_fails_to_load_does_not_stop_the_kernel(self, monkeypatch, caplog):
        """The executor sandbox's guard: the failure is logged, the kernel starts."""
        order = []

        def initialize_registry(**kwargs):
            raise RuntimeError("no registry here")

        class StubKernelApp:
            @classmethod
            def instance(cls):
                app = cls()
                app.shell = StubShell()
                return app

            def initialize(self, argv):
                order.append("initialize")

            def start(self):
                order.append("start")

        monkeypatch.setattr(jupyter_kernel, "_route_logs_to_process_stderr", lambda: None)
        monkeypatch.setattr(jupyter_kernel, "_prepare_environment", dict)
        monkeypatch.setattr("osprey.registry.initialize_registry", initialize_registry)
        monkeypatch.setattr("ipykernel.kernelapp.IPKernelApp", StubKernelApp)

        with caplog.at_level("WARNING", logger=jupyter_kernel.__name__):
            jupyter_kernel.main([])

        assert order == ["initialize", "start"]
        assert "Registry initialization failed" in caplog.text


class TestTheAuditRecord:
    """One refusal, one record — the ledger's whole claim about a cell."""

    @pytest.mark.parametrize("name", sorted(REFUSALS))
    def test_each_refusal_files_exactly_one_record(self, name, shell, audit_records, unstamped):
        """Every class the hook traps is audited, not only the write refusals."""
        fire(shell, REFUSALS[name]())

        assert len(audit_records) == 1
        assert audit_records[0]["surface"] == jupyter_kernel.SURFACE_NOTEBOOK_KERNEL
        assert audit_records[0]["decision"] == DECISION_REFUSED
        assert audit_records[0]["subject"] == jupyter_kernel.REFUSAL_SUBJECT

    def test_the_channel_is_the_detail_and_the_class_is_the_reason(
        self, shell, audit_records, unstamped
    ):
        """A cell has no name to give, so the channel is what identifies the write."""
        fire(shell, REFUSALS["write_blocked"]())

        assert audit_records[0]["reason"] == "channel_write_blocked"
        assert audit_records[0]["detail"] == "channel=SR:MAG:1"

    def test_a_refusal_with_no_channel_carries_no_detail(self, shell, audit_records, unstamped):
        """``ControlTargetChangedError`` names no channel, and detail is optional."""
        fire(shell, REFUSALS["target_changed"]())

        assert audit_records[0]["detail"] is None
        assert audit_records[0]["reason"] == "control_target_changed"

    def test_a_writer_that_fails_does_not_swallow_the_refusal(self, shell, monkeypatch, unstamped):
        """The audit trail degrades; the traceback the cell needs still arrives."""

        def explode(**fields):
            raise OSError("read-only audit zone")

        monkeypatch.setattr("osprey.audit.writer.record", explode)

        fire(shell, REFUSALS["write_blocked"]())

        assert len(shell.shown) == 1


class TestTheActionLine:
    """At most one line, and only where it names what the message did not."""

    def test_a_moved_target_asks_for_the_cell_to_be_re_run(
        self, shell, audit_records, unstamped, capsys
    ):
        """The kernel re-routes itself from the record before every cell.

        The refusal's own message ends by asking for ``execute()``, which is
        the executor sandbox's remedy and not this surface's.
        """
        fire(shell, REFUSALS["target_changed"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_TARGET_CHANGED + "\n"

    def test_a_switch_in_flight_gets_no_line(self, shell, audit_records, unstamped, capsys):
        """Its message opens with the shared token and ends with the same re-run."""
        error = SwitchInProgressError("switch_in_progress:4242")

        fire(shell, error)

        assert capsys.readouterr().out == ""
        assert str(error).startswith("switch_in_progress:4242")
        assert "re-run the cell" in str(error)

    def test_a_cell_pinned_everywhere_asks_for_the_chip(
        self, shell, audit_records, unstamped, capsys
    ):
        """``*=sandbox`` is what a cell the record could not route is pinned to."""
        unstamped.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")

        fire(shell, REFUSALS["write_blocked"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_WRITES_OFF + "\n"

    def test_a_cell_pinned_on_a_named_target_asks_for_the_chip_too(
        self, shell, audit_records, unstamped, capsys
    ):
        """Writes were off for that target when the cell opened."""
        unstamped.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "accelerator=sandbox")
        unstamped.setenv(posture.CONTROL_TARGET_ENV_VAR, "accelerator")

        fire(shell, REFUSALS["write_blocked"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_WRITES_OFF + "\n"

    def test_a_live_store_refusal_gets_no_line(self, shell, audit_records, unstamped, capsys):
        """The pin permits, so the store refused — and its message already says so."""
        fire(shell, REFUSALS["write_blocked"]())

        assert capsys.readouterr().out == ""

    def test_a_limits_violation_gets_no_line(self, shell, audit_records, unstamped, capsys):
        """Re-running changes nothing about a value outside the configured range."""
        unstamped.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")

        fire(shell, REFUSALS["limits"]())

        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "name",
        [
            "HINT_NO_SESSION",
            "HINT_SESSION_ENDED",
            "LAUNCH_PIN_REMEDIES",
            "_stamped_session_ended",
            "_rewrite_launch_pin_remedy",
        ],
    )
    def test_the_session_binding_advice_is_gone(self, name):
        """A kernel follows the deployment's record, so it joins no session.

        Every line above was about one — which session was bound at launch,
        whether it had ended, and the restart each of them asked for.
        """
        assert not hasattr(jupyter_kernel, name)


@pytest.fixture
def root_handlers():
    """The root logger's handlers, restored whole afterwards."""
    root = logging.getLogger()
    saved = list(root.handlers)
    yield root
    for handler in list(root.handlers):
        if handler not in saved:
            root.removeHandler(handler)
            stream = getattr(handler, "stream", None)
            handler.close()
            if stream is not None:
                stream.close()
    root.handlers[:] = saved


class TestTheProcessLog:
    """A cell shows the refusal and the action line; the log goes to the terminal."""

    def test_records_reach_the_inherited_stderr_and_not_the_replaced_one(
        self, root_handlers, monkeypatch, capfd
    ):
        """``ipykernel`` publishes ``sys.stderr`` into the cell; the log must miss it.

        The replaced stream stands in for that: a handler that resolves stderr
        when it emits writes there, and one holding a duplicate of the
        descriptor the process started on does not.
        """
        cell = io.StringIO()
        monkeypatch.setattr(sys, "stderr", cell)

        jupyter_kernel._route_logs_to_process_stderr()
        logging.getLogger("osprey_connectors.probe").warning("Blocked write to a channel")

        assert "Blocked write to a channel" not in cell.getvalue()
        assert "Blocked write to a channel" in capfd.readouterr().err

    def test_a_handler_bound_to_the_replaced_stderr_is_taken_off(
        self, root_handlers, monkeypatch, capfd
    ):
        """Otherwise the same record arrives twice, once of them in the cell."""
        cell = io.StringIO()
        monkeypatch.setattr(sys, "stderr", cell)
        stale = logging.StreamHandler(sys.stderr)
        root_handlers.addHandler(stale)

        jupyter_kernel._route_logs_to_process_stderr()

        assert stale not in root_handlers.handlers

    def test_the_level_the_root_logger_had_is_the_level_it_keeps(
        self, root_handlers, monkeypatch, capfd
    ):
        """Routing is a destination change; what is logged at all is not touched."""
        before = root_handlers.level

        jupyter_kernel._route_logs_to_process_stderr()

        assert root_handlers.level == before
        assert root_handlers.handlers[-1].level == logging.NOTSET


class TestTheTraceback:
    """The hook adds to the cell's output; it never takes the traceback away."""

    def test_the_original_triple_is_delegated(self, shell, audit_records, unstamped):
        """Not a rebuilt one: the frames a cell needs are the ones that raised."""
        triple = fire(shell, REFUSALS["write_blocked"](), tb_offset=2)

        assert shell.shown == [(triple, 2)]


# ===================================================================
# The per-cell stamp
# ===================================================================


@pytest.fixture
def named_kernel(monkeypatch):
    """A kernel that stamped itself at start, as ``compute_stamps`` leaves it."""
    monkeypatch.setenv(
        posture.POSTURE_SESSION_ENV_VAR, f"{jupyter_kernel.KERNEL_SESSION_PREFIX}{KERNEL_ID}"
    )
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")
    return monkeypatch


@pytest.fixture
def resolvable(monkeypatch):
    """Every target builds on this deployment.

    The predicate is the executor's and is exercised there; what this file pins
    is which branch the cell hook takes on each of its two answers.
    """
    monkeypatch.setattr(executor, "_target_is_resolvable", lambda target: True)
    return monkeypatch


def applying_report(root, write_server_report, pid, generation):
    """A live controls server that is mid-switch AT *generation*, unexpired."""
    bound = datetime.now(UTC) + timedelta(minutes=5)
    return write_server_report(
        root,
        pid,
        last_switch={
            "generation": generation,
            "status": "applying",
            "expires_at": bound.isoformat(),
        },
    )


def stamped(name):
    """The current value of one stamp name, or ``None`` when it is absent."""
    return os.environ.get(name)


class TestTheCellStamp:
    """``pre_run_cell`` is a total function of the record: three outcomes.

    A sandbox is stamped once because it dies with the run; a kernel outlives
    every switch an operator makes under it, so every name it routes on is
    rewritten from the record before each cell rather than carried over from
    the last one.
    """

    def test_a_settled_record_routes_the_cell(
        self, control_context_root, write_control_context, kernel_env, named_kernel, resolvable
    ):
        """Target, generation, and the posture the record answers for it."""
        write_control_context(control_context_root, target="va", generation=3)

        jupyter_kernel.pre_run_cell()

        assert stamped(executor.ENV_CONTROL_TARGET) == "va"
        assert stamped(executor.ENV_CONTROL_TARGET_GENERATION) == "3"
        assert stamped(posture_store.LAUNCH_POSTURE_ENV_VAR) == "va=writes"
        assert stamped(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL) is None

    def test_a_narrowed_target_pins_the_cell_sandboxed_on_it(
        self, control_context_root, write_control_context, kernel_env, named_kernel, resolvable
    ):
        """The cell still reaches the target; its writes are refused there.

        The pin is recomputed from the record every cell, so an operator who
        turns writes back on is followed by the next cell rather than by the
        next kernel.
        """
        write_control_context(
            control_context_root, target="va", generation=3, posture={"va": "sandbox"}
        )

        jupyter_kernel.pre_run_cell()

        assert stamped(executor.ENV_CONTROL_TARGET) == "va"
        assert stamped(posture_store.LAUNCH_POSTURE_ENV_VAR) == "va=sandbox"

    def test_a_widened_record_widens_the_next_cell(
        self, control_context_root, write_control_context, kernel_env, named_kernel, resolvable
    ):
        """The pin the LAST cell wrote must not be what decides this one.

        Inside an executor sandbox the launch pin is deliberately one-way:
        ``store_permits`` ANDs ``launch_permits`` in, and that term reads the
        very environment variable the process is carrying, so a run that
        started narrow stays narrow even after the operator widens the record
        under it. A run ends; a kernel does not. Deriving the cell's pin
        through ``store_permits`` would therefore let a kernel narrow but never
        widen — the operator turns writes back on from the chip and every
        remaining cell of that kernel goes on refusing until it is restarted.

        The derivation reads the record's own posture entry for the record's
        own target instead. Two cells around one widening is the only shape
        that catches the difference: the first cell leaves ``va=sandbox`` in
        the environment, which is exactly what the wrong derivation would read
        back.
        """
        write_control_context(
            control_context_root, target="va", generation=3, posture={"va": "sandbox"}
        )
        jupyter_kernel.pre_run_cell()
        assert stamped(posture_store.LAUNCH_POSTURE_ENV_VAR) == "va=sandbox"

        write_control_context(control_context_root, target="va", generation=3)

        jupyter_kernel.pre_run_cell()

        assert stamped(posture_store.LAUNCH_POSTURE_ENV_VAR) == "va=writes"
        # The consequence, not just the string: the cell's writes are permitted
        # again through the same rule every write surface asks.
        assert posture_store.store_permits("va") is True

    def test_a_switch_in_flight_refuses_the_cell(
        self,
        control_context_root,
        write_control_context,
        write_server_report,
        kernel_env,
        named_kernel,
        resolvable,
    ):
        """No target, sandboxed everywhere, and the refusal names the server.

        The hook cannot refuse by raising — ``IPython`` swallows a callback's
        exception — so the refusal is left in the environment for the cell's
        first control-system call to meet.
        """
        write_control_context(control_context_root, target="va", generation=3)
        applying_report(control_context_root, write_server_report, os.getpid(), 3)

        jupyter_kernel.pre_run_cell()

        assert stamped(executor.ENV_CONTROL_TARGET) is None
        assert stamped(executor.ENV_CONTROL_TARGET_GENERATION) is None
        assert stamped(posture_store.LAUNCH_POSTURE_ENV_VAR) == "*=sandbox"
        assert (
            stamped(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL)
            == f"switch_in_progress:{os.getpid()}"
        )

    def test_the_refusal_token_is_the_one_every_surface_refuses_with(
        self,
        control_context_root,
        write_control_context,
        write_server_report,
        kernel_env,
        named_kernel,
        resolvable,
    ):
        """The executor's message opens with exactly what the kernel stamps."""
        write_control_context(control_context_root, target="va", generation=3)
        applying_report(control_context_root, write_server_report, os.getpid(), 3)

        jupyter_kernel.pre_run_cell()

        refusal = stamped(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL)
        assert executor.switch_in_progress_message((os.getpid(),)).startswith(f"{refusal}.")

    def test_no_record_leaves_the_cell_on_the_baseline(
        self, control_context_root, kernel_env, named_kernel, resolvable
    ):
        """The fail-closed outcome the kernel starts in, and no refusal.

        Nothing is in flight, so there is nobody to wait for: the cell reads
        the deployment baseline and its writes are refused everywhere.
        """
        jupyter_kernel.pre_run_cell()

        assert stamped(executor.ENV_CONTROL_TARGET) is None
        assert stamped(executor.ENV_CONTROL_TARGET_GENERATION) is None
        assert stamped(posture_store.LAUNCH_POSTURE_ENV_VAR) == "*=sandbox"
        assert stamped(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL) is None

    def test_a_target_this_deployment_cannot_build_is_not_stamped(
        self, control_context_root, write_control_context, kernel_env, named_kernel, monkeypatch
    ):
        """The executor's answer, so both surfaces decline the same record.

        Stamping it would move the refusal into the cell, where it would
        surface as a build error the operator did not cause.
        """
        monkeypatch.setattr(executor, "_target_is_resolvable", lambda target: False)
        write_control_context(control_context_root, target="live", generation=3)

        jupyter_kernel.pre_run_cell()

        assert stamped(executor.ENV_CONTROL_TARGET) is None
        assert stamped(posture_store.LAUNCH_POSTURE_ENV_VAR) == "*=sandbox"
        assert stamped(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL) is None

    def test_every_name_is_rewritten_and_none_is_merged(
        self, control_context_root, write_control_context, kernel_env, named_kernel, resolvable
    ):
        """What the last cell left is replaced."""
        for name in executor._STAMP_ENV_NAMES:
            named_kernel.setenv(name, "from-the-last-cell")
        named_kernel.setenv(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL, "switch_in_progress:99")
        write_control_context(control_context_root, target="va", generation=7)

        jupyter_kernel.pre_run_cell()

        assert stamped(executor.ENV_CONTROL_TARGET) == "va"
        assert stamped(executor.ENV_CONTROL_TARGET_GENERATION) == "7"
        assert stamped(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL) is None

    def test_a_stale_target_does_not_survive_a_refusal(
        self,
        control_context_root,
        write_control_context,
        write_server_report,
        kernel_env,
        named_kernel,
        resolvable,
    ):
        """The other direction: a cell that is refused carries no target."""
        named_kernel.setenv(executor.ENV_CONTROL_TARGET, "va")
        named_kernel.setenv(executor.ENV_CONTROL_TARGET_GENERATION, "6")
        write_control_context(control_context_root, target="va", generation=7)
        applying_report(control_context_root, write_server_report, os.getpid(), 7)

        jupyter_kernel.pre_run_cell()

        assert stamped(executor.ENV_CONTROL_TARGET) is None
        assert stamped(executor.ENV_CONTROL_TARGET_GENERATION) is None

    def test_the_cell_is_marked_open(
        self, control_context_root, write_control_context, kernel_env, named_kernel, resolvable
    ):
        """The flag is what makes an in-flight marker a CELL's claim."""
        write_control_context(control_context_root, target="va", generation=3)

        jupyter_kernel.pre_run_cell()

        assert stamped(jupyter_kernel.ENV_IN_CELL) == jupyter_kernel.IN_CELL


# ===================================================================
# Closing the cell
# ===================================================================


def write_marker(root, name, kernel_id):
    """One in-flight marker under *root*, carrying *kernel_id*."""
    directory = root / posture_store.STATE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{target_state.INFLIGHT_FILE_PREFIX}{name}.json"
    path.write_text(
        json.dumps({"pid": os.getpid(), "surface": "notebook_kernel", "kernel_id": kernel_id}),
        encoding="utf-8",
    )
    return path


class TestClosingTheCell:
    """``post_run_cell`` runs in ``run_cell``'s ``finally``, so it always runs."""

    def test_the_cell_is_marked_closed(self, kernel_env, named_kernel):
        """A thread reaching the control system between cells claims nothing."""
        named_kernel.setenv(jupyter_kernel.ENV_IN_CELL, jupyter_kernel.IN_CELL)

        jupyter_kernel.post_run_cell()

        assert stamped(jupyter_kernel.ENV_IN_CELL) is None

    def test_only_this_kernels_markers_are_removed(
        self, control_context_root, kernel_env, named_kernel
    ):
        """Another kernel's cell and another surface's run are not this one's.

        A marker left behind refuses every later target switch; one removed on
        somebody else's behalf lets a switch move the machine under a run that
        is still talking to it.
        """
        mine = write_marker(control_context_root, "1_aaa", KERNEL_ID)
        another_kernel = write_marker(control_context_root, "2_bbb", "another-kernel")
        an_execution = write_marker(control_context_root, "3_ccc", None)

        jupyter_kernel.post_run_cell()

        assert not mine.exists()
        assert another_kernel.exists()
        assert an_execution.exists()

    def test_an_unnamed_kernel_removes_nothing(self, control_context_root, kernel_env, monkeypatch):
        """No session stamp is no id, and no id attributes no marker."""
        monkeypatch.delenv(posture.POSTURE_SESSION_ENV_VAR, raising=False)
        marker = write_marker(control_context_root, "1_aaa", KERNEL_ID)

        jupyter_kernel.post_run_cell()

        assert marker.exists()

    def test_a_sweep_that_fails_does_not_reach_the_cell(
        self, kernel_env, named_kernel, monkeypatch, caplog
    ):
        """It runs in a ``finally``: raising here would replace the cell's own."""

        def unreadable():
            raise RuntimeError("no root")

        monkeypatch.setattr(target_state, "state_dir", unreadable)

        with caplog.at_level(logging.WARNING):
            jupyter_kernel.post_run_cell()

        assert "in-flight markers" in caplog.text


# ===================================================================
# The cell's first control-system call
# ===================================================================


def test_the_runtime_spells_the_cell_contract_the_same_way():
    """The kernel writes these names and the runtime reads them, across a layer.

    They are spelled twice on purpose — a runtime that imported the kernel
    launcher would drag the notebook stack into every sandbox — so this is the
    only thing keeping them equal.
    """
    assert runtime.ENV_CONTROL_TARGET_REFUSAL == jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL
    assert runtime.ENV_IN_CELL == jupyter_kernel.ENV_IN_CELL
    assert runtime.KERNEL_SESSION_PREFIX == jupyter_kernel.KERNEL_SESSION_PREFIX
    assert runtime.INFLIGHT_SURFACE == jupyter_kernel.SURFACE_NOTEBOOK_KERNEL


def cell_markers(root):
    """Every in-flight marker under *root*, read straight off disk."""
    directory = root / posture_store.STATE_DIR_NAME
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob(target_state.INFLIGHT_FILE_GLOB))
    ]


class _StubConnector:
    """A connector with no control system behind it."""

    async def disconnect(self) -> None:  # pragma: no cover - not exercised here
        pass


@pytest.fixture
def stub_factory(monkeypatch):
    """Let ``_get_connector`` finish with no control system and no config.yml.

    What this file pins is the cell contract around the build — the refusal and
    the claim — so the build itself is stood in for; which block a stamp selects
    is pinned in ``test_executor_target_stamp.py``.
    """
    built: list[_StubConnector] = []

    async def create(config=None, control_target=None):
        built.append(_StubConnector())
        return built[-1]

    monkeypatch.setattr(runtime, "_target_connector_config", lambda: None)
    monkeypatch.setattr(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector", create
    )
    return built


@pytest.fixture
def open_cell(named_kernel):
    """A named kernel with a cell open and nothing carried in from the last one."""
    named_kernel.setenv(jupyter_kernel.ENV_IN_CELL, jupyter_kernel.IN_CELL)
    named_kernel.setattr(runtime, "_runtime_connector", None)
    named_kernel.setattr(runtime, "_connector_stamp", None)
    named_kernel.setattr(runtime, "_cell_marker", None)
    return named_kernel


class TestRefusingTheCell:
    """``pre_run_cell`` leaves the refusal; the first call is where it lands."""

    def test_a_refused_cell_raises_on_its_first_call(
        self, control_context_root, kernel_env, open_cell
    ):
        """The token an operator meets here is the one every surface refuses with."""
        open_cell.setenv(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL, "switch_in_progress:11,12")

        with pytest.raises(ControlTargetChangedError) as caught:
            asyncio.run(runtime._get_connector())

        assert isinstance(caught.value, SwitchInProgressError)
        assert str(caught.value).startswith("switch_in_progress:11,12")
        assert "re-run the cell" in str(caught.value)

    def test_a_refused_cell_claims_nothing(self, control_context_root, kernel_env, open_cell):
        """It reached no target, so there is nothing for a switch to wait on."""
        open_cell.setenv(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL, "switch_in_progress:11")

        with pytest.raises(SwitchInProgressError):
            asyncio.run(runtime._get_connector())

        assert cell_markers(control_context_root) == []

    def test_the_refusal_is_read_before_the_stamp(self, kernel_env, open_cell):
        """A refused cell carries no stamp, so nothing may consult one first."""

        def unreachable():
            raise AssertionError("the stamp was read before the refusal")

        open_cell.setenv(jupyter_kernel.ENV_CONTROL_TARGET_REFUSAL, "switch_in_progress:11")
        open_cell.setattr(runtime, "_stamped_target", unreachable)

        with pytest.raises(SwitchInProgressError):
            asyncio.run(runtime._get_connector())

    def test_a_cell_that_was_not_refused_reaches_the_connector(
        self, control_context_root, kernel_env, open_cell, stub_factory
    ):
        """Absence is the normal state, and it gates nothing."""
        asyncio.run(runtime._get_connector())

        assert len(stub_factory) == 1


class TestClaimingTheCell:
    """The marker is a CELL's claim, written by the call that first needs one."""

    def test_the_first_call_claims_the_cell(
        self, control_context_root, kernel_env, open_cell, stub_factory
    ):
        """A switch waits on this, and 5.4 names the busy client from it."""
        open_cell.setenv(executor.ENV_CONTROL_TARGET, "va")

        asyncio.run(runtime._get_connector())

        [marker] = cell_markers(control_context_root)
        assert marker["surface"] == jupyter_kernel.SURFACE_NOTEBOOK_KERNEL
        assert marker["kernel_id"] == KERNEL_ID
        assert marker["session"] == f"{jupyter_kernel.KERNEL_SESSION_PREFIX}{KERNEL_ID}"
        assert marker["pid"] == os.getpid()
        assert marker["target"] == "va"
        assert marker["launch_posture"] == "*=sandbox"

    def test_a_second_call_in_the_same_cell_claims_nothing_more(
        self, control_context_root, kernel_env, open_cell, stub_factory
    ):
        """One cell is one claim; a switch waiting on two would wait twice."""
        asyncio.run(runtime._get_connector())
        asyncio.run(runtime._get_connector())

        assert len(cell_markers(control_context_root)) == 1

    def test_the_next_cell_claims_again(
        self, control_context_root, kernel_env, open_cell, stub_factory
    ):
        """The claim is re-checked against the file the last cell's end removed."""
        asyncio.run(runtime._get_connector())
        jupyter_kernel.post_run_cell()
        os.environ[jupyter_kernel.ENV_IN_CELL] = jupyter_kernel.IN_CELL

        asyncio.run(runtime._get_connector())

        assert len(cell_markers(control_context_root)) == 1

    def test_a_thread_after_the_cell_holds_no_marker(
        self, control_context_root, kernel_env, open_cell, stub_factory
    ):
        """There is no cell for a switch to wait on; the pin holds its writes."""
        asyncio.run(runtime._get_connector())
        jupyter_kernel.post_run_cell()

        asyncio.run(runtime._get_connector())

        assert cell_markers(control_context_root) == []

    def test_a_cell_with_no_control_system_call_holds_no_marker(
        self, control_context_root, write_control_context, kernel_env, named_kernel, resolvable
    ):
        """The claim is lazy: a cell that touches nothing blocks nothing."""
        write_control_context(control_context_root, target="va", generation=3)

        jupyter_kernel.pre_run_cell()
        jupyter_kernel.post_run_cell()

        assert cell_markers(control_context_root) == []

    def test_a_call_outside_a_cell_claims_nothing(
        self, control_context_root, kernel_env, open_cell, stub_factory
    ):
        """The executor's own runs claim their own markers; this is the notebook's."""
        open_cell.delenv(jupyter_kernel.ENV_IN_CELL)

        asyncio.run(runtime._get_connector())

        assert cell_markers(control_context_root) == []

    def test_an_unnamed_kernel_claims_nothing(
        self, control_context_root, kernel_env, open_cell, stub_factory
    ):
        """``post_run_cell`` removes markers by kernel id, and there is none.

        A marker nothing can remove would refuse every later switch on this
        deployment until the process died.
        """
        open_cell.delenv(posture.POSTURE_SESSION_ENV_VAR)

        asyncio.run(runtime._get_connector())

        assert cell_markers(control_context_root) == []

    def test_a_claim_that_cannot_be_written_does_not_reach_the_cell(
        self, control_context_root, kernel_env, open_cell, stub_factory, caplog
    ):
        """The marker is advisory; the generation pin is what actually holds."""

        def unwritable():
            raise OSError("no state directory here")

        open_cell.setattr(target_state, "state_dir", unwritable)

        with caplog.at_level(logging.WARNING):
            asyncio.run(runtime._get_connector())

        assert len(stub_factory) == 1
        assert "claim this cell" in caplog.text
