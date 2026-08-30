"""The connector-host child, exercised as a real process.

Every wire-level test here spawns ``python -m osprey_connectors.ipc.host`` for
real and talks to it over pipes, because the things worth pinning are the ones
that only exist in a separate process: what the child inherits, what it emits
first, whether it dies when it should, and whether a failed request takes it
down with it. Nothing is stubbed out on the far side.

The child is pointed at the mock connector through its dotted class path, which
:func:`osprey_connectors.types.resolve_target` returns verbatim for ``live``
(the mock is not one of the *simulated* type names, so it is treated as a
deployment's own control system). That runs the entire real path — resolver,
factory, ``connect()`` — with no EPICS, no gateway and no network.

The child runs with ``cwd`` set to a scratch directory and no ``CONFIG_FILE``,
so no project config is reachable: writes are disabled, which is the posture
the write tests assert against. The write-posture tests are the exception —
posture is read from the project config, so they put a real config file in
reach and point the child at it.

The report-derivation helpers are unit-tested in-process at the bottom, since
the interesting cases (name-server vs address-list mode, which gateway role was
actually used) belong to a connector that configures an EPICS environment,
which the mock deliberately does not.
"""

import asyncio
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pytest
import yaml

from osprey_connectors import session_store
from osprey_connectors.control_system.base import (
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)
from osprey_connectors.control_system.mock_connector import MockConnector
from osprey_connectors.ipc import frames, host

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHONPATH = os.pathsep.join(
    [str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src")]
)

#: The mock connector by dotted path, so ``live`` resolves to it.
MOCK_TYPE = "osprey_connectors.control_system.mock_connector.MockConnector"

CONTROL_SYSTEM = {
    "type": MOCK_TYPE,
    "writes_enabled": False,
    "connector": {MOCK_TYPE: {"response_delay_ms": 10, "noise_level": 0.0}},
}

#: A deployment whose real machine is EPICS and whose simulator is armed: the
#: deployment-wide posture is off, the virtual accelerator's own block turns
#: writes on, and the live machine's block says nothing about them at all.
MIXED_CONTROL_SYSTEM = {
    "type": "epics",
    "writes_enabled": False,
    "connector": {
        "epics": {
            "gateways": {
                "read_only": {"address": "ro.example.org", "port": 5064},
                "write_access": {"address": "rw.example.org", "port": 5065},
            }
        },
        "virtual_accelerator": {
            "writes_enabled": True,
            "gateways": {
                "read_only": {"address": "va-ro.example.org", "port": 5074},
                "write_access": {"address": "va-rw.example.org", "port": 5075},
            },
        },
    },
}

#: Generous enough that a slow machine is not a failure, tight enough that a
#: hang fails the test instead of the run.
REPLY_TIMEOUT_S = 10.0


class Child:
    """A spawned connector host, with its frame channel pumped by a thread."""

    def __init__(self, cwd, env_extra=None):
        env = {k: v for k, v in os.environ.items() if k != "CONFIG_FILE"}
        env["PYTHONPATH"] = PYTHONPATH
        env.update(env_extra or {})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "osprey_connectors.ipc.host"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env=env,
        )
        self._frames: queue.Queue = queue.Queue()
        self._stderr: deque = deque(maxlen=200)
        self._pump(self.proc.stdout, self._read_frames)
        self._pump(self.proc.stderr, self._read_stderr)

    def _pump(self, stream, target):
        threading.Thread(target=target, args=(stream,), daemon=True).start()

    def _read_frames(self, stream):
        parser = frames.FrameReader()
        while True:
            chunk = stream.read1(65536)
            if not chunk:
                self._frames.put(None)
                return
            for frame in parser.feed(chunk):
                self._frames.put(frame)

    def _read_stderr(self, stream):
        for line in stream:
            self._stderr.append(line.decode("utf-8", "replace").rstrip())

    # -- talking to it ----------------------------------------------------

    def send(self, method, **kwargs):
        """Write one request frame and return its id."""
        request_id = frames.new_request_id()
        self.proc.stdin.write(frames.encode_request(request_id, method, kwargs))
        self.proc.stdin.flush()
        return request_id

    def next_frame(self, timeout=REPLY_TIMEOUT_S):
        """The next frame the child emitted, failing if it emitted none."""
        try:
            frame = self._frames.get(timeout=timeout)
        except queue.Empty:
            pytest.fail(f"child sent no frame within {timeout}s. stderr:\n{self.stderr()}")
        if frame is None:
            pytest.fail(f"child closed its frame channel. stderr:\n{self.stderr()}")
        return frame

    def call(self, method, **kwargs):
        """One request, one reply, matched by request id."""
        request_id = self.send(method, **kwargs)
        frame = self.next_frame()
        assert frame.request_id == request_id
        return frame

    def init(self, target="live", control_system=None, **payload):
        """Send the init frame and return the post-connect report frame."""
        return self.call(
            "init",
            control_system=control_system or CONTROL_SYSTEM,
            target=target,
            **payload,
        )

    def quiet(self, seconds=0.5):
        """Assert the child sends nothing more for a while."""
        try:
            extra = self._frames.get(timeout=seconds)
        except queue.Empty:
            return
        pytest.fail(f"child sent an unexpected extra frame: {extra!r}")

    def stderr(self):
        return "\n".join(self._stderr)

    def close(self):
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=REPLY_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=REPLY_TIMEOUT_S)


@pytest.fixture
def child(tmp_path):
    """A child with no project config in reach, torn down with the test."""
    spawned = Child(cwd=tmp_path)
    try:
        yield spawned
    finally:
        spawned.close()


@pytest.fixture
def ready_child(child):
    """A child that has already answered its init frame."""
    child.init()
    return child


# ------------------------------------------------------------ init / report


def test_first_frame_out_is_the_post_connect_report(child):
    frame = child.init()

    assert isinstance(frame, frames.ResultFrame)
    report = frame.value
    # The five verification fields the parent asserts its derivation against.
    assert set(report) >= {"selected_role", "mode", "host", "port", "_epics_configured"}
    # Mock semantics: no gateway is configured, so there is no endpoint to
    # verify — the report is well-formed and empty rather than absent.
    assert report["selected_role"] is None
    assert report["mode"] is None
    assert report["host"] is None
    assert report["port"] is None
    assert report["_epics_configured"] is False
    # Diagnostics that let the parent tell this child apart from the one it
    # meant to spawn.
    assert report["connector_type"] == MOCK_TYPE
    assert report["target"] == "live"
    assert report["writes_enabled"] is False
    assert report["readonly_run"] is False
    assert report["pid"] == child.proc.pid


def test_a_first_frame_that_is_not_init_fails_the_launch(child):
    frame = child.call("read_channel", channel_address="SR:BEAM:CURRENT")

    assert isinstance(frame, frames.ErrorFrame)
    assert isinstance(frame.exception, ConnectionError)
    assert "init" in frame.message
    assert child.proc.wait(timeout=REPLY_TIMEOUT_S) == host.EXIT_INIT_FAILED


def test_an_unresolvable_target_fails_the_launch_with_a_typed_error(child):
    frame = child.init(target="somewhere")

    assert isinstance(frame, frames.ErrorFrame)
    # ValueError is outside the typed registry, so it fails closed to
    # ConnectionError carrying what the child actually reported.
    assert isinstance(frame.exception, ConnectionError)
    assert "somewhere" in frame.message
    assert child.proc.wait(timeout=REPLY_TIMEOUT_S) == host.EXIT_INIT_FAILED


def test_inherited_epics_variables_do_not_survive_into_the_child(tmp_path):
    junk = {
        "EPICS_CA_ADDR_LIST": "junk.example.org",
        "EPICS_CA_SERVER_PORT": "9999",
        "EPICS_CA_NAME_SERVERS": "junk.example.org:9999",
        "EPICS_PVA_ADDR_LIST": "junk.example.org:9998",
    }
    spawned = Child(cwd=tmp_path, env_extra=junk)
    try:
        report = spawned.init().value

        # Nothing inherited reached the connector, and nothing was left behind
        # for it to pick up: what connect() did not set is simply not there.
        assert report["epics_env"] == {}
        assert report["host"] is None
        assert report["mode"] is None
    finally:
        spawned.close()


# ------------------------------------------------------------------- reads


def test_read_channel_returns_a_channel_value(ready_child):
    frame = ready_child.call("read_channel", channel_address="SR:BEAM:CURRENT")

    assert isinstance(frame, frames.ResultFrame)
    value = frame.value
    assert isinstance(value, ChannelValue)
    assert isinstance(value.value, float)
    assert value.metadata.units == "mA"


def test_a_batched_read_of_n_channels_is_one_round_trip(ready_child):
    channels = [f"SR:BPM:{index}:X" for index in range(6)]

    request_id = ready_child.send("read_multiple_channels", channel_addresses=channels)
    frame = ready_child.next_frame()

    assert frame.request_id == request_id
    assert sorted(frame.value) == sorted(channels)
    assert all(isinstance(value, ChannelValue) for value in frame.value.values())
    # One request in, one result out: the fan-out happened inside the child.
    ready_child.quiet()


# ------------------------------------------------------------------ writes


def test_write_on_a_writes_disabled_deployment_returns_the_refused_result(ready_child):
    frame = ready_child.call("write_channel", channel_address="SR:CORR:1:SP", value=0.5)

    result = frame.value
    assert isinstance(result, ChannelWriteResult)
    # A policy refusal crosses as a result frame, not an error frame, and the
    # outcome is a WriteOutcome member on this side of the boundary.
    assert result.outcome is WriteOutcome.REFUSED
    assert result.refusal_reason == "WRITES_DISABLED"
    assert "writes are disabled" in result.error_message


def test_write_multiple_channels_refuses_every_operation(ready_child):
    frame = ready_child.call(
        "write_multiple_channels",
        operations=[["SR:CORR:1:SP", 0.5], ["SR:CORR:2:SP", 0.25]],
    )

    results = frame.value
    assert [result.channel_address for result in results] == ["SR:CORR:1:SP", "SR:CORR:2:SP"]
    assert all(result.outcome is WriteOutcome.REFUSED for result in results)
    assert all(result.refusal_reason == "WRITES_DISABLED" for result in results)


def test_the_child_reports_the_posture_of_the_block_for_its_own_type(tmp_path):
    """Write posture is per connector type, end to end through a real child.

    The deployment-wide posture is off and the block for the type this child
    resolved arms writes, so only a report that looked its own type up can say
    the child is armed.
    """
    project = tmp_path / "project"
    project.mkdir()
    control_system = {
        "type": MOCK_TYPE,
        "writes_enabled": False,
        "connector": {MOCK_TYPE: {"response_delay_ms": 10, "writes_enabled": True}},
    }
    config_file = project / "config.yml"
    config_file.write_text(yaml.safe_dump({"control_system": control_system}))

    spawned = Child(cwd=tmp_path)
    try:
        report = spawned.init(control_system=control_system, config_file=str(config_file)).value

        assert report["connector_type"] == MOCK_TYPE
        assert report["writes_enabled"] is True
    finally:
        spawned.close()


# ------------------------------------------------------------- spawn_probe


def test_spawn_probe_reads_the_named_channel(ready_child):
    frame = ready_child.call("spawn_probe", channel="SR:BEAM:CURRENT", timeout=5.0)

    assert isinstance(frame, frames.ResultFrame)
    assert isinstance(frame.value, ChannelValue)
    assert isinstance(frame.value.value, float)


def test_a_probe_that_exceeds_its_bound_fails_typed_and_the_child_keeps_serving(ready_child):
    # The mock's own response delay (10 ms) outlasts this bound, so the probe
    # is cut off by the bound rather than by the connector.
    frame = ready_child.call("spawn_probe", channel="SR:BEAM:CURRENT", timeout=0.001)

    assert isinstance(frame, frames.ErrorFrame)
    assert frame.class_tag == "TimeoutError"
    assert isinstance(frame.exception, TimeoutError)
    # asyncio.wait_for raises a bare TimeoutError, and the switch refusal built
    # from it is only as informative as this message: it has to name the
    # channel that would not answer.
    assert "SR:BEAM:CURRENT" in frame.message
    assert "0.001" in frame.message

    # A failed probe means the switch does not happen; it does not mean this
    # child is finished.
    follow_up = ready_child.call("read_channel", channel_address="SR:BEAM:CURRENT")
    assert isinstance(follow_up.value, ChannelValue)


def test_an_unknown_method_is_refused_without_killing_the_child(ready_child):
    frame = ready_child.call("subscribe", channel_address="SR:BEAM:CURRENT")

    assert isinstance(frame, frames.ErrorFrame)
    assert isinstance(frame.exception, ConnectionError)
    assert "subscribe" in frame.message

    follow_up = ready_child.call("read_channel", channel_address="SR:BEAM:CURRENT")
    assert isinstance(follow_up.value, ChannelValue)


# --------------------------------------------------------------- lifecycle


def test_closing_stdin_exits_the_child_cleanly(ready_child):
    ready_child.proc.stdin.close()

    assert ready_child.proc.wait(timeout=REPLY_TIMEOUT_S) == host.EXIT_OK


def test_disconnect_is_acknowledged_before_the_child_exits(ready_child):
    frame = ready_child.call("disconnect")

    assert isinstance(frame, frames.ResultFrame)
    assert frame.value is None
    assert ready_child.proc.wait(timeout=REPLY_TIMEOUT_S) == host.EXIT_OK


def test_the_watchdog_exits_a_child_whose_parent_died(tmp_path):
    """A child orphaned without its pipe closing still goes away.

    EOF cannot cover this: the pipe here is held open by *this* process while
    the child's actual parent exits, which is what a crashed controls server
    looks like from the child's side. Only the ``getppid()`` watchdog can
    notice, so this is the test that would fail if the thread were dropped.
    """
    read_fd, write_fd = os.pipe()
    pid_file = tmp_path / "orphan.pid"
    launcher = (
        "import os, subprocess, sys\n"
        "proc = subprocess.Popen(\n"
        "    [sys.executable, '-m', 'osprey_connectors.ipc.host'],\n"
        f"    stdin={read_fd}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"open({str(pid_file)!r}, 'w').write(str(proc.pid))\n"
        "os._exit(0)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "CONFIG_FILE"}
    env["PYTHONPATH"] = PYTHONPATH

    try:
        subprocess.run(
            [sys.executable, "-c", launcher],
            cwd=str(tmp_path),
            env=env,
            pass_fds=(read_fd,),
            check=True,
            timeout=REPLY_TIMEOUT_S,
        )
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        orphan_pid = int(pid_file.read_text())

        # The write end is still open here, so the child sees no EOF; it is the
        # reparenting to init that has to end it.
        while time.monotonic() < deadline:
            try:
                os.kill(orphan_pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.kill(orphan_pid, signal.SIGKILL)
        pytest.fail(f"orphaned child {orphan_pid} was still alive after {REPLY_TIMEOUT_S}s")
    finally:
        os.close(read_fd)
        os.close(write_fd)


# ------------------------------------------------- report derivation (unit)


def test_scrub_removes_every_epics_variable_and_nothing_else():
    environ = {
        "EPICS_CA_ADDR_LIST": "gw.example.org",
        "EPICS_PVA_NAME_SERVERS": "gw.example.org:5075",
        "PYEPICS_LIBCA": "/opt/libca.dylib",
        "PATH": "/usr/bin",
    }

    removed = host.scrub_epics_env(environ)

    assert removed == {
        "EPICS_CA_ADDR_LIST": "gw.example.org",
        "EPICS_PVA_NAME_SERVERS": "gw.example.org:5075",
    }
    assert environ == {"PYEPICS_LIBCA": "/opt/libca.dylib", "PATH": "/usr/bin"}


def test_installed_endpoint_reads_back_name_server_mode(monkeypatch):
    monkeypatch.setenv("EPICS_CA_NAME_SERVERS", "cagw.example.org:5074")
    monkeypatch.delenv("EPICS_CA_ADDR_LIST", raising=False)

    assert host._installed_endpoint() == ("name_server", "cagw.example.org", 5074)


def test_installed_endpoint_reads_back_address_list_mode(monkeypatch):
    monkeypatch.delenv("EPICS_CA_NAME_SERVERS", raising=False)
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "cagw.example.org")
    monkeypatch.setenv("EPICS_CA_SERVER_PORT", "5064")

    assert host._installed_endpoint() == ("addr_list", "cagw.example.org", 5064)


def test_installed_endpoint_is_empty_when_nothing_was_configured(monkeypatch):
    monkeypatch.delenv("EPICS_CA_NAME_SERVERS", raising=False)
    monkeypatch.delenv("EPICS_CA_ADDR_LIST", raising=False)

    assert host._installed_endpoint() == (None, None, None)


GATEWAYS = {
    "read_only": {"address": "ro.example.org", "port": 5064},
    "write_access": {"address": "rw.example.org", "port": 5065},
}


def test_selected_role_is_the_write_gateway_when_writes_are_on():
    role = host._selected_role(
        GATEWAYS, "addr_list", "rw.example.org", 5065, writes_enabled=True, readonly_run=False
    )

    assert role == "write_access"


def test_selected_role_stays_read_only_in_a_readonly_run():
    role = host._selected_role(
        GATEWAYS, "addr_list", "ro.example.org", 5064, writes_enabled=True, readonly_run=True
    )

    assert role == "read_only"


def test_selected_role_follows_the_endpoint_that_was_actually_installed():
    # The rule would say read_only, but the environment names the write
    # gateway: the report describes what was used, not what was expected.
    role = host._selected_role(
        GATEWAYS, "addr_list", "rw.example.org", 5065, writes_enabled=False, readonly_run=False
    )

    assert role == "write_access"


def test_selected_role_is_none_when_no_gateway_was_configured():
    role = host._selected_role(GATEWAYS, None, None, None, writes_enabled=True, readonly_run=False)

    assert role is None


def test_a_gateway_without_a_port_matches_the_one_that_was_filled_in():
    # The virtual accelerator omits its port and follows the deployed service,
    # so the block never carries the number the environment ends up showing.
    gateways = {"read_only": {"address": "localhost", "use_name_server": True}}

    role = host._selected_role(
        gateways, "name_server", "localhost", 5064, writes_enabled=False, readonly_run=False
    )

    assert role == "read_only"


def test_a_gateway_in_the_other_mode_is_not_a_match():
    gateways = {"read_only": {"address": "localhost", "port": 5064, "use_name_server": True}}

    role = host._selected_role(
        gateways, "addr_list", "localhost", 5064, writes_enabled=False, readonly_run=False
    )

    assert role is None


# ------------------------------------------------ write posture (unit)


def _posture_carrier(connector_type, control_target, *, epics_configured=False):
    """A real connector carrying the two stamps the write posture is read from.

    The report now reads the connector instance's own ``_writes_enabled``, so a
    bare attribute bag would answer the question the test is asking. The mock is
    used as the carrier because that property lives on the base class every
    connector shares — nothing about the mock's own behaviour is exercised, only
    the type and target the factory stamps on whatever it builds.
    """
    connector = MockConnector()
    connector._connector_type = connector_type
    connector._control_target = control_target
    connector._epics_configured = epics_configured
    return connector


@pytest.fixture
def mixed_deployment(tmp_path, monkeypatch):
    """Put :data:`MIXED_CONTROL_SYSTEM` on disk and in reach of this process.

    Posture is read from the project config, where the connector reads it, so a
    test about posture has to point at a real file rather than hand a section
    in.
    """
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump({"control_system": MIXED_CONTROL_SYSTEM}))
    monkeypatch.setenv("CONFIG_FILE", str(config_file))
    return MIXED_CONTROL_SYSTEM


def test_the_report_arms_the_target_whose_own_block_says_so(mixed_deployment):
    report = host._post_connect_report(
        _posture_carrier("virtual_accelerator", "va"), "virtual_accelerator", "va", mixed_deployment
    )

    assert report["target"] == "va"
    assert report["connector_type"] == "virtual_accelerator"
    assert report["writes_enabled"] is True


def test_a_target_whose_block_says_nothing_inherits_the_disabled_deployment(mixed_deployment):
    # The epics block configures gateways and no posture, so this target keeps
    # the deployment-wide answer — which is off, whatever the simulator's block
    # says about the simulator.
    report = host._post_connect_report(
        _posture_carrier("epics", "live"), "epics", "live", mixed_deployment
    )

    assert report["target"] == "live"
    assert report["connector_type"] == "epics"
    assert report["writes_enabled"] is False


def test_a_readonly_run_keeps_an_armed_target_off_the_write_gateway(mixed_deployment, monkeypatch):
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    monkeypatch.delenv("EPICS_CA_NAME_SERVERS", raising=False)
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", "va-ro.example.org")
    monkeypatch.setenv("EPICS_CA_SERVER_PORT", "5074")

    report = host._post_connect_report(
        _posture_carrier("virtual_accelerator", "va", epics_configured=True),
        "virtual_accelerator",
        "va",
        mixed_deployment,
    )

    # The block arms this target and the report says so: that is the deployment
    # posture, and the same input connect() made its selection with. What the
    # readonly run collapses is the selection itself — an armed target still
    # went through the read gateway.
    assert report["writes_enabled"] is True
    assert report["readonly_run"] is True
    assert report["selected_role"] == "read_only"


# ------------------------------- the posture the selection was made with


#: The shipped virtual-accelerator gateway shape: both roles name the same
#: address, the same mode and the same (unset, service-default-filled) port, so
#: nothing installed in the environment tells them apart and the reported role
#: falls through to the posture the report carries.
IDENTICAL_VA_GATEWAYS = {
    "read_only": {"address": "localhost", "use_name_server": True},
    "write_access": {"address": "localhost", "use_name_server": True},
}

#: The same block with the two roles on separate endpoints, which is the shape
#: the reported role can be read off the environment alone.
DISTINCT_VA_GATEWAYS = {
    "read_only": {"address": "va-ro.example.org", "port": 5074},
    "write_access": {"address": "va-rw.example.org", "port": 5075},
}

#: The port an unset virtual-accelerator gateway follows.
VA_SERVICE_PORT = 5064

#: The posture-store key this session is stamped with.
POSTURE_SESSION = "chip-session"


def _va_config(gateways):
    """A switch-capable deployment whose virtual accelerator is armed."""
    return {
        "services": {"virtual_accelerator": {"port": VA_SERVICE_PORT}},
        "control_system": {
            "type": "epics",
            "writes_enabled": False,
            "connector": {
                "epics": {"gateways": {"read_only": {"address": "ro.example.org", "port": 5064}}},
                "virtual_accelerator": {"writes_enabled": True, "gateways": gateways},
            },
        },
    }


# --------------------------------------------- limits posture (child build)
#
# Limits posture is per connector type, and the child never learns which type
# it is by asking: the factory stamps ``_connector_type`` between construction
# and ``connect()``, and the connector builds its validator from that stamp. So
# there is nothing in this package for a per-type block to reach — the child
# inherits the posture by pointing at the project config and resolving its
# target, exactly as the parent would have.
#
# These two run the child's own ``_build_connector`` in this process rather
# than over a pipe, because what is being pinned is an attribute of the
# connector living inside the child and the wire serves connector *methods*
# (:data:`host.PROXY_METHODS`) — the policy a validator was built with never
# crosses it. Everything up to that attribute is the real path: the on-disk
# config the child is pointed at, ``resolve_target``, the factory, ``connect()``.


DEPLOYMENT_WIDE_ALLOW_KEY = "control_system.limits_checking.allow_unlisted_channels"
VA_ALLOW_KEY = (
    "control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels"
)


def _limits_control_system(database_path: Path) -> dict:
    """A deployment that refuses unlisted channels everywhere but its simulator.

    The deployment-wide block is strict and the virtual accelerator's own block
    relaxes it, so the two targets disagree and only a reader that resolved its
    own type can say which answer it got. The deployment's own type is the mock
    by dotted path, so ``live`` resolves to it and needs no EPICS; ``va``
    resolves to ``virtual_accelerator`` whatever the deployment was built for.
    """
    return {
        "type": MOCK_TYPE,
        "limits_checking": {
            "enabled": True,
            "allow_unlisted_channels": False,
            "database_path": str(database_path),
        },
        "connector": {
            MOCK_TYPE: {"response_delay_ms": 10, "noise_level": 0.0},
            "virtual_accelerator": {
                "limits_checking": {"enabled": True, "allow_unlisted_channels": True}
            },
        },
    }


@pytest.fixture
def va_deployment(tmp_path, monkeypatch):
    """Put an armed virtual accelerator on disk and in reach of this process."""

    def _build(gateways):
        config = _va_config(gateways)
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.safe_dump(config))
        monkeypatch.setenv("CONFIG_FILE", str(config_file))
        monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
        return config

    return _build


@pytest.fixture
def narrowed_va(tmp_path, monkeypatch):
    """An operator narrowing of ``va`` to read-only, in reach of this process.

    The narrowing lives in the per-(session, target) posture store, keyed by
    ``OSPREY_POSTURE_SESSION`` under ``OSPREY_AGENT_DATA_ROOT`` — the fixture
    idiom of ``tests/connectors/test_session_store.py``.
    """
    root = tmp_path / "agent-data"
    store = root / session_store.STATE_DIR_NAME / session_store.STORE_FILENAME
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({POSTURE_SESSION: {"va": session_store.POSTURE_SANDBOX}}))
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", POSTURE_SESSION)
    session_store.invalidate_cache()
    yield root
    session_store.invalidate_cache()


def _install_name_server(monkeypatch, address, port):
    """Stand in for what ``connect()`` installs for a name-server gateway."""
    monkeypatch.delenv("EPICS_CA_ADDR_LIST", raising=False)
    monkeypatch.delenv("EPICS_CA_SERVER_PORT", raising=False)
    monkeypatch.setenv("EPICS_CA_NAME_SERVERS", f"{address}:{port}")


def _install_addr_list(monkeypatch, address, port):
    """Stand in for what ``connect()`` installs for an address-list gateway."""
    monkeypatch.delenv("EPICS_CA_NAME_SERVERS", raising=False)
    monkeypatch.setenv("EPICS_CA_ADDR_LIST", address)
    monkeypatch.setenv("EPICS_CA_SERVER_PORT", str(port))


def _verify(config, report, writes_enabled):
    """Run the parent's own verification against a child's report."""
    from osprey.mcp_server.control_system.target_eligibility import (
        derive_endpoints,
        verify_child_report,
    )

    derivation = derive_endpoints(config, "va", writes_enabled=writes_enabled)
    return derivation, verify_child_report(derivation, report)


def test_identical_gateways_report_the_role_the_narrowed_posture_selected(
    va_deployment, narrowed_va, monkeypatch
):
    """The shipped VA shape, with the operator holding ``va`` read-only.

    Both roles name the same endpoint, so there is no evidence in the
    environment to tell them apart and the reported role is decided by the
    posture the report carries. Re-derived from config that posture is armed —
    the block arms the simulator — and the report would name ``write_access``
    where the parent, which derives with the store-aware value, expects
    ``read_only``: the one field ``verify_child_report`` compares, and an
    aborted switch on the deployment's default configuration.
    """
    config = va_deployment(IDENTICAL_VA_GATEWAYS)
    _install_name_server(monkeypatch, "localhost", VA_SERVICE_PORT)
    connector = _posture_carrier("virtual_accelerator", "va", epics_configured=True)

    report = host._post_connect_report(
        connector, "virtual_accelerator", "va", config["control_system"]
    )

    # The deployment arms this target; only the operator's narrowing is holding
    # it down, and the report is the connector's own answer rather than a
    # second reading of the config.
    assert connector._writes_enabled is False
    assert report["selected_role"] == "read_only"

    derivation, verification = _verify(config, report, connector._writes_enabled)
    assert derivation.selected_role == "read_only"
    assert verification.ok, verification.detail

    assert report["writes_enabled"] is False
    assert report["readonly_run"] is False


def test_identical_gateways_still_report_write_access_when_nothing_narrows(
    va_deployment, monkeypatch
):
    """The same deployment with no narrowing — the armed answer still stands.

    The read-only report above is the store's doing, not a report that has been
    pinned to the read gateway.
    """
    config = va_deployment(IDENTICAL_VA_GATEWAYS)
    _install_name_server(monkeypatch, "localhost", VA_SERVICE_PORT)
    connector = _posture_carrier("virtual_accelerator", "va", epics_configured=True)

    report = host._post_connect_report(
        connector, "virtual_accelerator", "va", config["control_system"]
    )

    assert connector._writes_enabled is True
    assert report["writes_enabled"] is True
    assert report["selected_role"] == "write_access"

    derivation, verification = _verify(config, report, connector._writes_enabled)
    assert derivation.selected_role == "write_access"
    assert verification.ok, verification.detail


async def test_a_narrowed_target_reports_the_posture_its_writes_are_refused_on(
    va_deployment, narrowed_va, monkeypatch
):
    """Distinct gateways: the report's posture is the connector's, end to end.

    Here the installed endpoint names the role on its own, so the role would
    read ``read_only`` either way; what the config-derived report got wrong is
    ``writes_enabled``, which claimed the child was armed on a target whose very
    next write the same instance refuses.
    """
    config = va_deployment(DISTINCT_VA_GATEWAYS)
    _install_addr_list(monkeypatch, "va-ro.example.org", 5074)
    connector = _posture_carrier("virtual_accelerator", "va", epics_configured=True)

    report = host._post_connect_report(
        connector, "virtual_accelerator", "va", config["control_system"]
    )

    assert report["writes_enabled"] is False
    assert report["readonly_run"] is False
    assert report["selected_role"] == "read_only"

    # Same instance, same rule: the deployment arms this target, so the refusal
    # can only be the narrowing the report is now reporting.
    result = await connector.write_channel("SR:CORR:1:SP", 0.5)
    assert result.outcome is WriteOutcome.REFUSED

    derivation, verification = _verify(config, report, connector._writes_enabled)
    assert derivation.selected_role == "read_only"
    assert verification.ok, verification.detail


@pytest.fixture
def limits_deployment(tmp_path):
    """The mixed-posture deployment on disk, beside a real limits database.

    The database path is deployment-wide — one file per deployment — and has to
    resolve to something loadable, or every posture collapses to the same
    fail-safe validator and the test would pass without reading a block.
    """
    database = tmp_path / "limits.json"
    database.write_text(json.dumps({"SR:CORR:1:SP": {"min_value": -1.0, "max_value": 1.0}}))
    section = _limits_control_system(database)
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump({"control_system": section}))
    return section, str(config_file)


def _child_limits_policy(section: dict, config_file: str, target: str) -> dict:
    """The policy the child's connector ends up validating writes against."""

    async def build():
        connector, _report = await host._build_connector(
            {"control_system": section, "target": target, "config_file": config_file}
        )
        try:
            return dict(connector._limits_validator.policy)
        finally:
            await connector.disconnect()

    return asyncio.run(build())


def test_child_limits_posture_comes_from_the_block_for_the_target_it_serves(limits_deployment):
    section, config_file = limits_deployment

    policy = _child_limits_policy(section, config_file, "va")

    # The simulator's own block answered, and the refusal an operator would
    # eventually read names that line rather than the deployment-wide one it
    # overrides.
    assert policy["allow_unlisted_channels"] is True
    assert policy["allow_unlisted_key"] == VA_ALLOW_KEY


def test_child_limits_posture_falls_back_to_the_deployment_wide_block(limits_deployment):
    section, config_file = limits_deployment

    policy = _child_limits_policy(section, config_file, "live")

    # This deployment wrote no block for the type ``live`` resolves to, so the
    # deployment-wide refusal is the whole posture — and the simulator's
    # relaxation, two keys away in the same file, does not reach it.
    assert policy["allow_unlisted_channels"] is False
    assert policy["allow_unlisted_key"] == DEPLOYMENT_WIDE_ALLOW_KEY
