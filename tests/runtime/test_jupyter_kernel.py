"""The notebook kernel launcher: what it takes away, and what it must not pull in.

The stamping itself is exercised beside the executor's, in
``test_executor_target_stamp.py``, because the two are one routing contract and
belong in one file. What is left here is the launcher's own two jobs: preparing
the process before a kernel exists, and staying importable in processes that
have no kernel stack at all.

The refusal hook is the launcher's third job and is exercised here in full: a
stub shell stands in for ``IPython``'s, so the handler can be fired with a real
refusal without a kernel behind it.

Nothing here starts a kernel. ``main()`` is the four statements after the
preparation, and running them would hand the test session's stdio to
``ipykernel``; the preparation is called directly instead, and the one test
that does call ``main()`` puts a stub in ``IPKernelApp``'s place.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from osprey import jupyter_kernel
from osprey.audit import posture
from osprey.audit.envelope import DECISION_REFUSED
from osprey.mcp_server.python_executor import executor
from osprey.runtime import ControlTargetChangedError
from osprey_connectors import session_store
from osprey_connectors.control_system import base
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError

#: The posture-store key the bound session is recorded under.
SESSION_KEY = "4f1c2a7e-0000-4000-8000-000000000002"


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
    session_store.invalidate_cache()


def write_binding_document(root: Path, **fields) -> None:
    """Put a binding under *root*, in the shape the web terminal writes."""
    document = {"session_id": SESSION_KEY, "pty_pid": None, "agent_data_root": str(root)}
    document.update(fields)
    path = jupyter_kernel.binding_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


class TestPreparingTheProcess:
    """What the kernel's environment holds by the time a cell can read it."""

    def test_the_server_token_is_removed(self, tmp_path, kernel_env, monkeypatch):
        """A cell must find no ``JUPYTER_TOKEN`` key, not an empty one.

        The kernelspec carries the name with an empty value and the provisioner
        merges that env OVER ``os.environ``, so setting it empty is what puts
        the key there. Only a pop takes it away.
        """
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.JUPYTER_TOKEN_ENV_VAR, "")

        jupyter_kernel._prepare_environment()

        assert jupyter_kernel.JUPYTER_TOKEN_ENV_VAR not in kernel_env

    def test_a_real_token_is_removed_too(self, tmp_path, kernel_env, monkeypatch):
        """The inherited value, not only the kernelspec's empty one."""
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.JUPYTER_TOKEN_ENV_VAR, "not-for-cells")

        jupyter_kernel._prepare_environment()

        assert jupyter_kernel.JUPYTER_TOKEN_ENV_VAR not in kernel_env

    def test_the_binding_is_read_from_the_stamped_root(self, tmp_path, kernel_env, monkeypatch):
        """Where the launcher looks, and which root then wins.

        It looks under the root ``session_store.agent_data_root()`` answers —
        the ``OSPREY_AGENT_DATA_ROOT`` stamp the kernelspec carries. The root
        stamped afterwards is the one the BINDING names, which is what makes a
        kernel read the same stores the terminal writes even where the sidecar
        was started somewhere else.
        """
        located = tmp_path / "located"
        bound = tmp_path / "bound"
        bound.mkdir()
        write_binding_document(located, agent_data_root=str(bound))
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(located))

        stamps = jupyter_kernel._prepare_environment()

        assert stamps[session_store.AGENT_DATA_ROOT_ENV_VAR] == str(bound)
        assert kernel_env[posture.POSTURE_SESSION_ENV_VAR] == SESSION_KEY

    def test_no_binding_leaves_the_kernel_pinned_sandboxed(self, tmp_path, kernel_env, monkeypatch):
        """The ordinary state of a kernel started before any terminal attached."""
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))

        stamps = jupyter_kernel._prepare_environment()

        assert stamps[session_store.LAUNCH_POSTURE_ENV_VAR] == "*=sandbox"
        assert kernel_env[session_store.LAUNCH_POSTURE_ENV_VAR] == "*=sandbox"

    def test_a_damaged_binding_is_the_same_as_none(self, tmp_path, kernel_env, monkeypatch):
        """A document caught mid-rewrite must not fail a kernel start."""
        path = jupyter_kernel.binding_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))

        stamps = jupyter_kernel._prepare_environment()

        assert stamps[session_store.LAUNCH_POSTURE_ENV_VAR] == "*=sandbox"

    def test_a_binding_naming_no_session_is_the_same_as_none(
        self, tmp_path, kernel_env, monkeypatch
    ):
        """Identity is both halves or neither: no session key, nothing joined."""
        write_binding_document(tmp_path, session_id="")
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.delenv(posture.POSTURE_SESSION_ENV_VAR, raising=False)

        stamps = jupyter_kernel._prepare_environment()

        assert posture.POSTURE_SESSION_ENV_VAR not in stamps
        assert stamps[session_store.LAUNCH_POSTURE_ENV_VAR] == "*=sandbox"

    def test_the_config_path_is_published_under_the_name_the_loader_reads(
        self, tmp_path, kernel_env, monkeypatch
    ):
        """``OSPREY_CONFIG`` is what the sidecar passes; ``CONFIG_FILE`` is what loads.

        The kernel's working directory is the notebooks folder, so without the
        second name every runtime call in a cell resolves the defaults.
        """
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.OSPREY_CONFIG_ENV_VAR, str(tmp_path / "config.yml"))
        monkeypatch.delenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, raising=False)

        jupyter_kernel._prepare_environment()

        assert kernel_env[jupyter_kernel.CONFIG_FILE_ENV_VAR] == str(tmp_path / "config.yml")

    def test_an_existing_config_file_is_not_overridden(self, tmp_path, kernel_env, monkeypatch):
        """Whoever set ``CONFIG_FILE`` chose it; the sidecar's name only fills a gap."""
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
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
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.setenv(jupyter_kernel.OSPREY_CONFIG_ENV_VAR, str(tmp_path / "config.yml"))
        monkeypatch.setenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, "")

        jupyter_kernel._prepare_environment()

        assert kernel_env[jupyter_kernel.CONFIG_FILE_ENV_VAR] == str(tmp_path / "config.yml")

    def test_no_config_path_publishes_nothing(self, tmp_path, kernel_env, monkeypatch):
        """Neither name set: the launcher invents no path, empty or otherwise."""
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        monkeypatch.delenv(jupyter_kernel.OSPREY_CONFIG_ENV_VAR, raising=False)
        monkeypatch.delenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, raising=False)

        jupyter_kernel._prepare_environment()

        assert jupyter_kernel.CONFIG_FILE_ENV_VAR not in kernel_env


def test_importing_the_module_pulls_in_no_kernel_stack():
    """The binding reader is imported by the web terminal, which has no kernel.

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


class StubShell:
    """As much of an ``InteractiveShell`` as the hook touches."""

    def __init__(self):
        self.registered = None
        self.shown = []

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
    monkeypatch.delenv(session_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.delenv(posture.CONTROL_TARGET_ENV_VAR, raising=False)
    monkeypatch.delenv(executor.ENV_CONTROL_TARGET_STATE_PID, raising=False)
    return monkeypatch


def joined_a_session(monkeypatch, record):
    """Stamp a state pid, and make the record behind it answer *record*.

    *record* is what the controls server that stamped this kernel publishes
    now: a mapping while its session is live, ``None`` once that session has
    ended and its record is gone.
    """
    monkeypatch.setenv(executor.ENV_CONTROL_TARGET_STATE_PID, "4242")
    monkeypatch.setattr("osprey.runtime._current_target_record", lambda: record)


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

    def test_the_hook_goes_on_between_initialize_and_start(self, monkeypatch):
        """``initialize`` is what builds the shell; ``start`` does not return.

        So the hook has exactly one place to go, and a chained call would have
        left it nowhere. The registry is loaded before any of that, from the
        config path the preparation publishes, and the call is the executor
        sandbox's: no export, the path from ``CONFIG_FILE``. Log routing comes
        before all of it, so that what the preparation and the registry log is
        already going to the terminal.
        """
        order = []
        registry_calls = []
        config_path = "/deployment/config.yml"

        def prepare():
            order.append("prepare")
            monkeypatch.setenv(jupyter_kernel.CONFIG_FILE_ENV_VAR, config_path)
            return {}

        def initialize_registry(**kwargs):
            order.append("initialize_registry")
            registry_calls.append(kwargs)

        class RecordingShell(StubShell):
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
    """At most one line, and only where a restart is what changes the answer."""

    def test_a_moved_target_asks_for_a_restart(self, shell, audit_records, unstamped, capsys):
        """The kernel holds one stamp for its whole life."""
        joined_a_session(unstamped, {"target": "accelerator", "generation": 3})

        fire(shell, REFUSALS["target_changed"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_TARGET_CHANGED + "\n"

    def test_a_session_that_ended_is_not_reported_as_a_switch(
        self, shell, audit_records, unstamped, capsys
    ):
        """ "+ New" ends one session and attaches another; the target never moved.

        The pin raises the same refusal either way — it compares the stamp
        against what is published now, and an ended session publishes nothing —
        so the record is what tells the two apart.
        """
        joined_a_session(unstamped, None)

        fire(shell, REFUSALS["target_changed"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_SESSION_ENDED + "\n"

    def test_a_kernel_that_joined_no_session_still_gets_the_target_line(
        self, shell, audit_records, unstamped, capsys
    ):
        """No stamp, no record — which is not the same as a record that went away."""
        unstamped.setattr("osprey.runtime._current_target_record", lambda: None)

        fire(shell, REFUSALS["target_changed"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_TARGET_CHANGED + "\n"

    def test_a_record_check_that_raises_falls_back_to_the_target_line(
        self, shell, audit_records, unstamped, capsys
    ):
        """Both lines ask for the same restart, so the fail-safe claims less."""

        def explode():
            raise OSError("no state directory here")

        joined_a_session(unstamped, None)
        unstamped.setattr("osprey.runtime._current_target_record", explode)

        fire(shell, REFUSALS["target_changed"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_TARGET_CHANGED + "\n"

    def test_a_run_pinned_everywhere_asks_for_a_session(
        self, shell, audit_records, unstamped, capsys
    ):
        """``*=sandbox`` is the launcher's "no session was bound" pin."""
        unstamped.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")

        fire(shell, REFUSALS["write_blocked"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_NO_SESSION + "\n"

    def test_a_run_pinned_on_a_named_target_asks_for_the_chip(
        self, shell, audit_records, unstamped, capsys
    ):
        """Writes were off for that target at launch, and the pin never re-reads."""
        unstamped.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "accelerator=sandbox")
        unstamped.setenv(posture.CONTROL_TARGET_ENV_VAR, "accelerator")

        fire(shell, REFUSALS["write_blocked"]())

        assert capsys.readouterr().out == jupyter_kernel.HINT_WRITES_OFF + "\n"

    def test_a_live_store_refusal_gets_no_line(self, shell, audit_records, unstamped, capsys):
        """The pin permits, so the store refused — and its message already says so."""
        fire(shell, REFUSALS["write_blocked"]())

        assert capsys.readouterr().out == ""

    def test_a_limits_violation_gets_no_line(self, shell, audit_records, unstamped, capsys):
        """A restart changes nothing about a value outside the configured range."""
        unstamped.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")

        fire(shell, REFUSALS["limits"]())

        assert capsys.readouterr().out == ""


def launch_pin_message(monkeypatch, control_target):
    """The refusal message the connector writes when the launch pin refuses.

    Taken from the connector rather than restated, so that a rewrite test
    stops passing the moment the sentence it rewrites stops being written.
    """
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.setattr(base, "_deployment_writes_enabled", lambda *args, **kwargs: True)
    result = base._writes_disabled_result(
        "SR:MAG:1", 1.0, control_target=control_target, store_permits=False
    )
    return result.error_message


class TestTheRefusalsOwnRemedy:
    """The connector's message names a remedy; on this surface it is a restart."""

    def test_the_connector_still_writes_the_sentences_this_module_pins(self, unstamped):
        """The rewrite is a string match, so the strings are asserted at the source.

        Both launch-pin wordings are checked, and a change to either fails here
        rather than reaching a cell as advice that does nothing.
        """
        unstamped.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "accelerator=sandbox")
        named = launch_pin_message(unstamped, "accelerator")
        unstamped.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")
        everywhere = launch_pin_message(unstamped, None)

        assert named.endswith("Re-run the script to pick it up.")
        assert everywhere.endswith("Re-run the script to pick up the current write state.")
        assert set(jupyter_kernel.LAUNCH_PIN_REMEDIES) == {
            "Re-run the script to pick it up.",
            "Re-run the script to pick up the current write state.",
        }

    def test_a_named_targets_pin_asks_for_a_restart_not_a_re_run(
        self, shell, audit_records, unstamped
    ):
        """Re-running the cell reuses the kernel, which still carries the stamp."""
        unstamped.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "accelerator=sandbox")
        unstamped.setenv(posture.CONTROL_TARGET_ENV_VAR, "accelerator")
        message = launch_pin_message(unstamped, "accelerator")

        fire(shell, ChannelWriteBlockedError("SR:MAG:1", "WRITES_DISABLED", message))

        shown = str(shell.shown[0][0][1])
        assert shown.endswith("Restart the kernel to pick it up.")
        assert "Re-run the script" not in shown
        assert (
            shown[: -len("Restart the kernel to pick it up.")]
            == message[: -len("Re-run the script to pick it up.")]
        )

    def test_the_pinned_everywhere_message_is_rewritten_too(self, shell, audit_records, unstamped):
        """Its remedy is the same re-run, in the wording that names the state."""
        unstamped.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")
        message = launch_pin_message(unstamped, None)

        fire(shell, ChannelWriteBlockedError("SR:MAG:1", "WRITES_DISABLED", message))

        assert str(shell.shown[0][0][1]).endswith(
            "Restart the kernel to pick up the current write state."
        )

    def test_a_live_store_refusal_keeps_its_own_message(self, shell, audit_records, unstamped):
        """The pin permits, so the chip is the remedy and nothing is rewritten."""
        message = "Write to 'SR:MAG:1' blocked: Re-run the script to pick it up."

        fire(shell, ChannelWriteBlockedError("SR:MAG:1", "WRITES_DISABLED", message))

        assert str(shell.shown[0][0][1]) == message

    def test_a_message_without_the_pinned_sentence_is_left_alone(
        self, shell, audit_records, unstamped
    ):
        """A wording this module does not know is shown as the connector wrote it."""
        unstamped.setenv(session_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")
        message = "Write to 'SR:MAG:1' blocked: some wording nobody pinned."

        fire(shell, ChannelWriteBlockedError("SR:MAG:1", "WRITES_DISABLED", message))

        assert str(shell.shown[0][0][1]) == message


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
