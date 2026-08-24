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
the write tests assert against.

The report-derivation helpers are unit-tested in-process at the bottom, since
the interesting cases (name-server vs address-list mode, which gateway role was
actually used) belong to a connector that configures an EPICS environment,
which the mock deliberately does not.
"""

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

from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult
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


def test_write_on_a_writes_disabled_deployment_returns_the_blocked_result(ready_child):
    frame = ready_child.call("write_channel", channel_address="SR:CORR:1:SP", value=0.5)

    result = frame.value
    assert isinstance(result, ChannelWriteResult)
    assert result.blocked is True
    assert result.success is False
    assert result.refusal_reason == "WRITES_DISABLED"
    assert "writes are disabled" in result.error_message


def test_write_multiple_channels_refuses_every_operation(ready_child):
    frame = ready_child.call(
        "write_multiple_channels",
        operations=[["SR:CORR:1:SP", 0.5], ["SR:CORR:2:SP", 0.25]],
    )

    results = frame.value
    assert [result.channel_address for result in results] == ["SR:CORR:1:SP", "SR:CORR:2:SP"]
    assert all(result.refusal_reason == "WRITES_DISABLED" for result in results)


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
