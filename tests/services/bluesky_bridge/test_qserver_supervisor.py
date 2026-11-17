"""The queue-server container's PID 1: it exits when the RE Manager is gone.

Upstream's `start-re-manager` stays alive with no manager inside when the
manager dies before it enabled the watchdog, so nothing restarts the
container. The supervisor runs it as a child and turns "the manager process
is gone" into a process exit the restart policy acts on. Gone means both: no
answer to the health check's ping, and nothing listening on the manager's
control port. A manager that is alive but not answering still holds the port,
and is left alone.

Children here are real processes; the probe and the clock are injected, so no
test waits on a real interval.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from osprey.services.bluesky_bridge import qserver_supervisor as supervisor

#: A stand-in `start-re-manager` that, like the real one, has a child of its
#: own (the manager or worker) and then stays alive whatever happens to it.
#: It writes the child's pid to the path in argv[1].
_PARENT_WITH_CHILD = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
with open(sys.argv[1], "w") as handle:
    handle.write(str(child.pid))
time.sleep(600)
"""

#: Exits 5 on SIGTERM, after saying it is ready to receive it.
_EXITS_5_ON_SIGTERM = """
import signal, sys, time
signal.signal(signal.SIGTERM, lambda *_: sys.exit(5))
print("ready", flush=True)
time.sleep(600)
"""


class _Clock:
    """Time that moves only when the supervisor waits."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def wait_for_exit(self, child: subprocess.Popen[bytes], seconds: float) -> int | None:
        self.now += seconds
        return child.poll()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def spawn() -> Iterator[Callable[..., subprocess.Popen[bytes]]]:
    """Start a child the way the supervisor does, and reap it afterwards."""
    children: list[subprocess.Popen[bytes]] = []

    def _spawn(*argv: str) -> subprocess.Popen[bytes]:
        child = supervisor.start_child([sys.executable, "-c", *argv])
        children.append(child)
        return child

    yield _spawn
    for child in children:
        supervisor.stop_process_group(child, timeout=0)


def test_the_thresholds_are_the_documented_constants() -> None:
    assert (
        supervisor.START_GRACE_S,
        supervisor.PROBE_INTERVAL_S,
        supervisor.MAX_FAILED_PROBES,
    ) == (60.0, 10.0, 6)


@pytest.mark.parametrize("status", [0, 7])
def test_a_child_that_exits_gives_the_supervisor_its_status(
    spawn: Callable[..., subprocess.Popen[bytes]], status: int
) -> None:
    child = spawn(f"import sys; sys.exit({status})")

    assert supervisor.supervise(child, probe=lambda: True, port_refused=lambda: True) == status


def test_a_child_killed_by_a_signal_reads_as_128_plus_the_signal(
    spawn: Callable[..., subprocess.Popen[bytes]],
) -> None:
    child = spawn("import os, signal; os.kill(os.getpid(), signal.SIGKILL)")

    assert (
        supervisor.supervise(child, probe=lambda: True, port_refused=lambda: True)
        == 128 + signal.SIGKILL
    )


def test_a_manager_that_never_answers_ends_the_child_and_its_descendants(
    spawn: Callable[..., subprocess.Popen[bytes]],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pid_file = tmp_path / "grandchild.pid"
    child = spawn(_PARENT_WITH_CHILD, str(pid_file))
    assert _wait_until(lambda: pid_file.exists() and pid_file.read_text() != "")
    grandchild = int(pid_file.read_text())
    clock = _Clock()

    with caplog.at_level("ERROR", logger=supervisor.logger.name):
        status = supervisor.supervise(
            child,
            probe=lambda: False,
            port_refused=lambda: True,
            clock=clock,
            wait_for_exit=clock.wait_for_exit,
        )

    assert status == supervisor.UNRESPONSIVE_EXIT_STATUS != 0
    assert child.poll() is not None
    assert _wait_until(lambda: not _alive(grandchild))
    # One line, and it says why.
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "did not answer" in message
    assert "refused" in message


def test_no_probe_counts_before_the_start_grace_is_over(
    spawn: Callable[..., subprocess.Popen[bytes]],
) -> None:
    child = spawn("import time; time.sleep(600)")
    clock = _Clock()
    probed_at: list[float] = []

    def probe() -> bool:
        probed_at.append(clock.now)
        return False

    supervisor.supervise(
        child,
        probe=probe,
        port_refused=lambda: True,
        clock=clock,
        wait_for_exit=clock.wait_for_exit,
    )

    assert probed_at[0] >= supervisor.START_GRACE_S
    assert len(probed_at) == supervisor.MAX_FAILED_PROBES
    # The earliest exit: the grace, then one interval between each failure.
    assert clock.now == supervisor.START_GRACE_S + supervisor.PROBE_INTERVAL_S * (
        supervisor.MAX_FAILED_PROBES - 1
    )


def test_one_answer_starts_the_count_again(
    spawn: Callable[..., subprocess.Popen[bytes]],
) -> None:
    """Upstream's own watchdog replaces a hung manager in seconds; a manager
    that answers again is healthy, however often it missed before."""
    child = spawn("import time; time.sleep(600)")
    clock = _Clock()
    one_short = [False] * (supervisor.MAX_FAILED_PROBES - 1)
    answers = iter([*one_short, True, *one_short, True])
    probes = 0

    def probe() -> bool:
        nonlocal probes
        probes += 1
        return next(answers, False)

    status = supervisor.supervise(
        child,
        probe=probe,
        port_refused=lambda: True,
        clock=clock,
        wait_for_exit=clock.wait_for_exit,
    )

    assert status == supervisor.UNRESPONSIVE_EXIT_STATUS
    assert probes == 2 * supervisor.MAX_FAILED_PROBES + supervisor.MAX_FAILED_PROBES


def test_a_child_that_exits_while_unanswered_still_gives_its_own_status(
    spawn: Callable[..., subprocess.Popen[bytes]],
) -> None:
    """The manager stopping on its own is not the supervisor's finding."""
    child = spawn("import time; time.sleep(600)")
    clock = _Clock()
    probes = 0

    def probe() -> bool:
        nonlocal probes
        probes += 1
        if probes == supervisor.MAX_FAILED_PROBES:
            child.kill()
            child.wait()
        return False

    status = supervisor.supervise(
        child,
        probe=probe,
        port_refused=lambda: True,
        clock=clock,
        wait_for_exit=clock.wait_for_exit,
    )

    assert status == 128 + signal.SIGKILL


def test_an_unanswering_manager_that_still_holds_its_port_is_left_running(
    spawn: Callable[..., subprocess.Popen[bytes]],
) -> None:
    """Alive but stalled: stopping it would end a plan that may be moving
    hardware, for a manager that can still come back."""
    child = spawn("import time; time.sleep(600)")
    clock = _Clock()
    probes = 0

    def probe() -> bool:
        nonlocal probes
        probes += 1
        if probes == 20 * supervisor.MAX_FAILED_PROBES:
            child.kill()
            child.wait()
        return False

    status = supervisor.supervise(
        child,
        probe=probe,
        port_refused=lambda: False,
        clock=clock,
        wait_for_exit=clock.wait_for_exit,
    )

    # Its own end, twenty thresholds later; never the supervisor's.
    assert status == 128 + signal.SIGKILL


def test_a_port_that_is_held_once_starts_the_count_again(
    spawn: Callable[..., subprocess.Popen[bytes]],
) -> None:
    child = spawn("import time; time.sleep(600)")
    clock = _Clock()
    one_short = [True] * (supervisor.MAX_FAILED_PROBES - 1)
    refusals = iter([*one_short, False, *one_short, False])
    asked = 0

    def port_refused() -> bool:
        nonlocal asked
        asked += 1
        return next(refusals, True)

    status = supervisor.supervise(
        child,
        probe=lambda: False,
        port_refused=port_refused,
        clock=clock,
        wait_for_exit=clock.wait_for_exit,
    )

    assert status == supervisor.UNRESPONSIVE_EXIT_STATUS
    assert asked == 3 * supervisor.MAX_FAILED_PROBES


def test_the_port_is_not_asked_about_a_manager_that_answers(
    spawn: Callable[..., subprocess.Popen[bytes]],
) -> None:
    child = spawn("import time; time.sleep(600)")
    clock = _Clock()
    probes = 0

    def probe() -> bool:
        nonlocal probes
        probes += 1
        if probes == 3:
            child.kill()
            child.wait()
        return True

    def port_refused() -> bool:
        raise AssertionError("an answering manager needs no second question")

    supervisor.supervise(
        child,
        probe=probe,
        port_refused=port_refused,
        clock=clock,
        wait_for_exit=clock.wait_for_exit,
    )


def test_a_port_nothing_holds_reads_as_refused(reserve_port: Callable[[], int]) -> None:
    assert supervisor.control_port_refused("127.0.0.1", reserve_port()) is True


def test_a_listening_port_is_not_refused_even_when_nobody_accepts() -> None:
    """A stalled manager's port: the kernel completes the connection for a
    process that never gets to accept it."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()

        assert supervisor.control_port_refused("127.0.0.1", listener.getsockname()[1]) is False


@pytest.mark.parametrize(
    "error", [TimeoutError("timed out"), OSError("network is unreachable"), socket.gaierror(-2)]
)
def test_a_connect_that_fails_any_other_way_is_not_refused(
    monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    """Only a refusal says "no process holds this port". A timeout, or any
    error that is not a refusal, says nothing of the kind."""

    def create_connection(*_args: object, **_kwargs: object) -> socket.socket:
        raise error

    monkeypatch.setattr(supervisor.socket, "create_connection", create_connection)

    assert supervisor.control_port_refused("127.0.0.1", 60615) is False


@pytest.mark.parametrize(
    ("command", "environ", "endpoint"),
    [
        # The rendered command line.
        (["start-re-manager", "--zmq-control-addr", "tcp://*:60615"], {}, ("127.0.0.1", 60615)),
        (["start-re-manager", "--zmq-control-addr=tcp://*:60700"], {}, ("127.0.0.1", 60700)),
        (
            ["start-re-manager", "--zmq-control-addr", "tcp://0.0.0.0:60701"],
            {},
            ("127.0.0.1", 60701),
        ),
        (
            ["start-re-manager", "--zmq-control-addr", "tcp://10.0.0.5:60702"],
            {},
            ("10.0.0.5", 60702),
        ),
        # Upstream's deprecated spelling of the same argument.
        (["start-re-manager", "--zmq-addr", "tcp://*:60703"], {}, ("127.0.0.1", 60703)),
        # The argument wins over the variable; the variable over the default.
        (
            ["start-re-manager", "--zmq-control-addr", "tcp://*:60704"],
            {"QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER": "tcp://*:60705"},
            ("127.0.0.1", 60704),
        ),
        (
            ["start-re-manager"],
            {"QSERVER_ZMQ_CONTROL_ADDRESS_FOR_SERVER": "tcp://*:60705"},
            ("127.0.0.1", 60705),
        ),
        (["start-re-manager"], {}, ("127.0.0.1", 60615)),
    ],
)
def test_the_control_endpoint_is_the_one_the_manager_is_given(
    command: list[str], environ: dict[str, str], endpoint: tuple[str, int]
) -> None:
    assert supervisor.control_endpoint(command, environ) == endpoint


@pytest.mark.parametrize("address", ["ipc:///tmp/manager", "tcp://*:port", "nonsense"])
def test_an_address_with_no_tcp_port_gives_no_endpoint(address: str) -> None:
    """No port to ask about means no finding that the manager is gone."""
    assert (
        supervisor.control_endpoint(["start-re-manager", "--zmq-control-addr", address], {}) is None
    )


def test_the_probe_is_the_health_checks_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    assert supervisor.ping_manager() is True
    assert calls == [["qserver", "ping"]]


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CompletedProcess(["qserver", "ping"], 3),
        subprocess.TimeoutExpired(["qserver", "ping"], 10),
        FileNotFoundError("qserver"),
    ],
)
def test_a_ping_that_fails_times_out_or_cannot_run_is_no_answer(
    monkeypatch: pytest.MonkeyPatch, failure: object
) -> None:
    def run(*_args: object, **_kwargs: object) -> object:
        if isinstance(failure, BaseException):
            raise failure
        return failure

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    assert supervisor.ping_manager() is False


def _run_supervisor(*command: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", supervisor.__name__, "--", *command],
        stdout=subprocess.PIPE,
        text=True,
    )


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_stop_signal_reaches_the_child_and_its_status_comes_back(signum: int) -> None:
    """`docker stop` signals PID 1 only; the manager must hear it at once."""
    script = _EXITS_5_ON_SIGTERM.replace("SIGTERM", signal.Signals(signum).name)
    process = _run_supervisor(sys.executable, "-c", script)
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signum)
        assert process.wait(timeout=30) == 5
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_the_command_runs_with_its_own_arguments() -> None:
    process = _run_supervisor(sys.executable, "-c", "import sys; sys.exit(int(sys.argv[1]))", "9")

    assert process.wait(timeout=30) == 9


def test_no_command_is_a_usage_error() -> None:
    assert _run_supervisor().wait(timeout=30) == 2


def test_a_command_that_cannot_start_is_127() -> None:
    assert _run_supervisor("/nonexistent/start-re-manager").wait(timeout=30) == 127
