"""Survivable per-worker diagnostics for CI test runs.

A test lane that *fails* explains itself: pytest prints the assertion and the
traceback. A lane that is *killed* explains nothing — the job hits its
``timeout-minutes``, the runner sends a signal, and whatever pytest was about to
say dies in a buffer. That is the case this module covers.

Two records are written per process, both continuously flushed so they are
complete at every instant rather than at exit:

``events-<worker>.jsonl``
    One line per test start and finish. A ``start`` with no matching ``finish``
    is the test that was in flight when the process died — see
    ``scripts/ci/diag_summary.py``.

``stacks-<worker>.txt``
    Every thread's stack, sampled at a fixed interval for the whole session,
    plus a traceback if the interpreter crashes outright. Sampling runs for the
    whole session rather than around each test, so this is a periodic snapshot,
    not a per-test stuck detector: a healthy long run leaves a handful of dumps,
    and a wedged one leaves the same frames over and over, which is what "stuck"
    looks like. Covering the whole session is deliberate — it keeps dumping
    after the LAST test has finished, which is where a shutdown hang lives, and
    a per-test timer would be cancelled by then and show nothing.

A third record is written only when a test fails:

``containers/at-failure/<worker>--<module>/``
    The container engine's ``ps -a``, one ``state.txt`` line per container, a
    ``<name>.log`` per container and ``<name>.health.json`` where the container
    has a health check. Written at the first failing report of each test module
    in each process, from ``pytest_runtest_makereport`` — before that module's
    fixtures tear its stack down. The workflow's capture step cannot take it:
    that teardown happens inside the pytest step, so by the time any later step
    runs, the stack's containers are gone.

All three are gated on ``OSPREY_CI_DIAG_DIR``: unset (every local run) means
this module installs nothing at all.

Why a file rather than stderr
-----------------------------
pytest's built-in ``faulthandler_timeout`` does reach the job log even under
``-n`` — xdist relays worker stderr to the controller, verified against
pytest-xdist 3.8.0. What it cannot do is survive the case this module exists
for. Its dump lands only in the job log, which is exactly what a cancelled job
truncates; it fires once per test, so a stack that is genuinely stuck looks the
same as one that was merely slow; and it says nothing at all about which test
each worker was inside. Writing to a file fixes the first (the artifact is
uploaded whatever happens to the log), repeated sampling fixes the second
(identical frames dumped again and again is what "stuck" looks like), and the
event log fixes the third.

Why the sampler is a Python thread, and not ``dump_traceback_later``
--------------------------------------------------------------------
``faulthandler.dump_traceback_later`` is the obvious way to write this, and it
is not safe here. Its watchdog lives in C and walks every thread's frames
*without holding the GIL* — that is the whole point of it, since it has to work
when the GIL is deadlocked. The cost is that it reads frames the interpreter is
still mutating. Under this suite that is not a theoretical race: the config and
template tests spend much of their time inside PyYAML, ruamel and Jinja2, whose
recursive-descent parsers build and tear down frames continuously, and a sample
landing there walks a half-freed frame and takes the process down with it. The
worker dies with no traceback and no protocol shutdown, xdist reports ``node
down: Not properly terminated``, and the lane then fails in one of two ways
that look nothing like each other: xdist's loadscope scheduler raises
``KeyError`` on the replacement worker, or the controller parks forever in
``dsession.loop_once`` waiting for an event the dead worker will never send.
Both were observed, on consecutive runs, caused by this module.

Calling ``faulthandler.dump_traceback`` from an ordinary Python thread holds
the GIL for the walk, so no frame can move underneath it. What that gives up is
the case the C watchdog exists for: a hard GIL deadlock, or a C extension
spinning without releasing it, starves this thread and no dump appears. Every
hang this lane has actually produced — blocked on a queue, a socket, or a
subprocess — releases the GIL and samples fine. A dump that never comes is
itself the signature of the case we cannot sample.

``faulthandler.enable`` is kept. It installs a fatal-signal handler that only
runs once the process is already lost, so it carries none of this risk, and it
is what turns a segfault in a C extension into something with a stack on it.

The unit lane still passes no ``faulthandler_timeout`` — pytest's built-in is
implemented with ``dump_traceback_later``, so arming it re-introduces exactly
the crash described above, on a per-test timer.

The per-test cap that does exist, and why it is ``signal``
----------------------------------------------------------
The lane passes ``--timeout=600 --timeout-method=signal`` (pytest-timeout), so
a hung test ends in ten minutes with its own name on the failure instead of
running out the 40-minute step cap and taking the log with it (#743). That is a
kill, and it is not in tension with the paragraphs above: the objection there is
to a *watchdog walking frames without the GIL*, which is a crash risk, not to
bounding a test.

``--timeout-method=thread`` would be. It ends the timed-out test with
``os._exit`` of the xdist worker, which is precisely the ``node down: Not
properly terminated`` case described above — the lane would go from one hung
test to a scheduler ``KeyError`` or a controller parked forever. The ``signal``
method raises inside the test and leaves the worker alive to report it, which is
what keeps the failure readable.

What ``signal`` gives up is the same thing this module's sampler gives up: a
hang inside a C extension that holds the GIL never services the SIGALRM, so no
timeout fires. In that case the step cap is again the only backstop, and the
uploaded stacks are the only evidence — which is why they keep being written
whether or not a timeout is armed. The value is 600 s rather than something
tighter so that a cold container image pull in the ``xdist_group("docker")``
files, which legitimately takes minutes, is never mistaken for a hang.
"""

from __future__ import annotations

import faulthandler
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

#: Enables the whole module, and names the directory both records are written
#: to. The CI job uploads that directory as an artifact.
ENV_DIR = "OSPREY_CI_DIAG_DIR"

#: Seconds between whole-process stack snapshots. ``0`` disables stack dumping
#: while leaving the event log on.
ENV_STACK_TIMEOUT = "OSPREY_CI_DIAG_STACK_TIMEOUT"

#: The tests' own container-engine switch, so a snapshot reads the engine they
#: drive. ``docker`` when unset.
ENV_E2E_RUNTIME = "OSPREY_E2E_RUNTIME"

#: Where failure-time container snapshots go, inside the directory the capture
#: action writes its own container records to, so a reader finds both in one
#: place. The action's files there are flat, so a subdirectory cannot collide.
CONTAINER_SNAPSHOT_SUBDIR = Path("containers") / "at-failure"

#: Upper bound, in seconds, on each engine call a snapshot makes.
SNAPSHOT_CALL_TIMEOUT = 30.0

#: The name the snapshot plugin is registered under.
CONTAINER_SNAPSHOT_PLUGIN = "osprey-ci-diag-containers"

#: The capture action's ``state.txt`` format, verbatim, so both files read alike.
_STATE_FORMAT = (
    "{{.Name}} image={{.Config.Image}} status={{.State.Status}} "
    "exit={{.State.ExitCode}} oomkilled={{.State.OOMKilled}} error={{.State.Error}} "
    "started={{.State.StartedAt}} finished={{.State.FinishedAt}}"
)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

#: The sampling interval, not a deadline: nothing is ever killed on this timer.
#: Five minutes keeps a healthy half-hour lane down to a handful of dumps while
#: still landing several inside any hang worth investigating.
DEFAULT_STACK_TIMEOUT = 300.0

#: How long ``stop()`` waits for an in-flight sample to finish before closing
#: the file under it. A dump of a few dozen threads takes milliseconds; this is
#: only here so a wedged sampler can never hold up the end of a run.
SAMPLER_JOIN_TIMEOUT = 5.0


def worker_id() -> str:
    """This process's xdist worker name, or ``main`` when running unparallelised.

    Every artifact is keyed on it. With ``-n 4`` a freeze produces four separate
    files, which is what makes "all workers stopped at the same instant"
    (a runner-level stall) distinguishable from "one worker wedged" (a real
    deadlock in a test).
    """
    return os.environ.get("PYTEST_XDIST_WORKER", "main")


def worker_index() -> int:
    """This process's xdist worker number — ``gw3`` → 3 — or 0 when unparallelised.

    For anything a worker has to own exclusively while its siblings run: a
    port block, a container name, a scratch directory. ``main`` (no xdist)
    takes 0, the same slot as ``gw0``, which is fine because the two never
    coexist in one run.
    """
    worker = worker_id()
    if worker.startswith("gw") and worker[2:].isdigit():
        return int(worker[2:])
    return 0


class DiagnosticsRecorder:
    """Writes the event log and arms the stack dumper for one process."""

    def __init__(
        self,
        directory: Path | str,
        worker: str,
        stack_timeout: float | None = DEFAULT_STACK_TIMEOUT,
    ) -> None:
        self.directory = Path(directory)
        self.worker = worker
        self.stack_timeout = stack_timeout
        self._events: Any = None
        self._stacks: Any = None
        self._stopped = False
        self._sampler: threading.Thread | None = None
        self._stop_sampling = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

        # buffering=1 is line buffering: every record reaches the OS the moment
        # its newline is written, with no dependence on a clean shutdown.
        self._events = (self.directory / f"events-{self.worker}.jsonl").open(
            "a", buffering=1, encoding="utf-8"
        )

        if self.stack_timeout:
            self._stacks = (self.directory / f"stacks-{self.worker}.txt").open(
                "a", buffering=1, encoding="utf-8"
            )
            # enable() covers a hard interpreter crash: a segfault inside a C
            # extension leaves no Python-level trace otherwise. It only fires
            # on a fatal signal, so it cannot itself destabilise a healthy
            # process — unlike the C watchdog this sampler replaces. See the
            # module docstring.
            faulthandler.enable(file=self._stacks, all_threads=True)
            self._sampler = threading.Thread(
                target=self._sample_stacks,
                name="osprey-ci-diag-stacks",
                daemon=True,
            )
            self._sampler.start()

        self.record("session_start", stack_timeout=self.stack_timeout)

    def _sample_stacks(self) -> None:
        """Dump every thread's stack on an interval until ``stop()`` is called.

        Runs as an ordinary Python thread so the walk happens under the GIL and
        cannot read a frame mid-mutation. ``Event.wait`` rather than ``sleep``
        so a finished run is not held up for a whole interval.
        """
        assert self.stack_timeout is not None
        while not self._stop_sampling.wait(self.stack_timeout):
            try:
                faulthandler.dump_traceback(file=self._stacks, all_threads=True)
            except (OSError, ValueError, AttributeError):
                # Closed handle or full disk. Diagnostics never break a run.
                return

    def stop(self) -> None:
        if self._stopped:
            return
        self.record("session_end")
        self._stopped = True

        if self._stacks is not None:
            # Wake the sampler and let any in-flight dump finish before the
            # file goes out from under it.
            self._stop_sampling.set()
            if self._sampler is not None:
                self._sampler.join(SAMPLER_JOIN_TIMEOUT)
                self._sampler = None
            faulthandler.disable()
            self._stacks.close()
            self._stacks = None

        if self._events is not None:
            self._events.close()
            self._events = None

    # -- writing -----------------------------------------------------------

    def record(self, event: str, **fields: Any) -> None:
        """Append one event. Never raises — diagnostics must not break a run."""
        if self._events is None:
            return
        payload = {
            "event": event,
            "worker": self.worker,
            "pid": os.getpid(),
            "t": time.time(),
            **fields,
        }
        try:
            self._events.write(json.dumps(payload, default=str) + "\n")
            self._events.flush()
        except (OSError, ValueError):  # closed handle, full disk
            pass


def _safe(name: str) -> str:
    """``name`` with every character outside ``[A-Za-z0-9._-]`` replaced by ``_``."""
    return _UNSAFE.sub("_", name)


class ContainerSnapshots:
    """Snapshots the container engine at the first failing report of each module.

    A module fixture that brings a stack up removes it in its own teardown,
    which runs inside the pytest process right after the module's last test.
    A failing report is made before that teardown, so a snapshot taken here
    still sees the stack the failure happened against. Every engine call is
    read-only — list, inspect, logs — because self-hosted runners are shared.

    Under ``pytest-rerunfailures`` the first failed attempt's report still
    reads ``failed`` when this hook sees it (the rerun relabel comes later), so
    the snapshot is taken on that first attempt and the module key stops a
    second one.
    """

    def __init__(self, directory: Path | str, worker: str, engine: str) -> None:
        # Absolute at arming time: a test that chdirs is still in its call
        # phase when its report is made.
        self.root = Path(directory).absolute() / CONTAINER_SNAPSHOT_SUBDIR
        self.worker = worker
        self.engine = engine
        self._modules: set[str] = set()

    def _run(self, *args: str) -> tuple[subprocess.CompletedProcess[str] | None, str]:
        """Run one engine call; a failure is returned as text, never raised."""
        try:
            done = subprocess.run(
                [self.engine, *args],
                capture_output=True,
                text=True,
                timeout=SNAPSHOT_CALL_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"({self.engine} {args[0]} failed: {exc})\n"
        return done, done.stdout + done.stderr

    def capture(self, nodeid: str, when: str) -> Path | None:
        """Write one module's snapshot; ``None`` when there is nothing to record."""
        module = nodeid.split("::", 1)[0]
        if module in self._modules:
            return None
        self._modules.add(module)

        if shutil.which(self.engine) is None:
            return None
        listed, _ = self._run("ps", "-aq")
        if listed is None or listed.returncode != 0:
            return None
        ids = listed.stdout.split()
        if not ids:
            return None

        target = self.root / f"{self.worker}--{_safe(module)}"
        target.mkdir(parents=True, exist_ok=True)
        utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        (target / "failure.txt").write_text(f"nodeid={nodeid}\nphase={when}\nutc={utc}\n")
        (target / "ps.txt").write_text(self._run("ps", "-a")[1])

        state = []
        for cid in ids:
            named, _ = self._run("inspect", "--format", "{{.Name}}", cid)
            raw = named.stdout.strip() if named is not None and named.returncode == 0 else ""
            name = _safe(raw.lstrip("/")) or cid

            (target / f"{name}.log").write_text(self._run("logs", "--timestamps", cid)[1])
            state.append(self._run("inspect", "--format", _STATE_FORMAT, cid)[1])

            health, _ = self._run("inspect", "--format", "{{json .State.Health}}", cid)
            if health is not None and health.returncode == 0:
                text = health.stdout.strip()
                if text and text != "null":
                    (target / f"{name}.health.json").write_text(text + "\n")

        (target / "state.txt").write_text("".join(state))
        return target

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item):
        outcome = yield
        report = outcome.get_result()
        if not report.failed:
            return
        try:
            self.capture(item.nodeid, report.when)
        except Exception:  # a diagnostic must never replace the failure it observes
            pass


def register_container_snapshots(
    pluginmanager: pytest.PytestPluginManager,
) -> ContainerSnapshots | None:
    """Register the failure-time container snapshot plugin when CI diag is on.

    ``None`` when ``OSPREY_CI_DIAG_DIR`` is unset or empty, and when the plugin
    is already registered, so calling it twice is harmless.
    """
    if not os.environ.get(ENV_DIR):
        return None
    if pluginmanager.has_plugin(CONTAINER_SNAPSHOT_PLUGIN):
        return None
    snapshots = ContainerSnapshots(
        os.environ[ENV_DIR],
        worker_id(),
        os.environ.get(ENV_E2E_RUNTIME) or "docker",
    )
    pluginmanager.register(snapshots, CONTAINER_SNAPSHOT_PLUGIN)
    return snapshots


def recorder_from_env() -> DiagnosticsRecorder | None:
    """Build a recorder from the environment, or ``None`` when CI diag is off."""
    directory = os.environ.get(ENV_DIR)
    if not directory:
        return None

    raw = os.environ.get(ENV_STACK_TIMEOUT)
    if raw is None:
        stack_timeout: float | None = DEFAULT_STACK_TIMEOUT
    else:
        try:
            stack_timeout = float(raw) or None
        except ValueError:
            stack_timeout = DEFAULT_STACK_TIMEOUT

    return DiagnosticsRecorder(directory, worker=worker_id(), stack_timeout=stack_timeout)
