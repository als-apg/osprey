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
    through the session's teardown after the LAST test, where a per-test timer
    would be cancelled and show nothing. It stops at ``pytest_unconfigure``.
    What follows — the non-daemon threads still alive, the exit callbacks, the
    final garbage collection, and a process that outlives its exit bound — is
    named on every run, CI or not, by the exit report in
    ``_live_threads.py``.

A third record is written only when a test fails:

``containers/at-failure/<worker>--<module>/``
    The container engine's ``ps -a``, one ``state.txt`` line per container, a
    ``<name>.log`` per container and ``<name>.health.json`` where the container
    has a health check, for the containers this run's tests started, by label.
    Written at the first failing report of each test module in each process,
    from ``pytest_runtest_makereport`` — before that module's fixtures tear its
    stack down — with the test's pytest-timeout timer paused and within one
    total deadline, so the snapshot is charged to no test's budget. The
    workflow's capture step cannot take it: that teardown happens inside the
    pytest step, so by the time any later step runs, the stack's containers are
    gone. A snapshot that could not finish says ``capture incomplete`` in its
    ``state.txt``.

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

import contextlib
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

#: Upper bound, in seconds, on one whole snapshot. It equals the shortest
#: per-file ``pytest.mark.timeout`` in the suite, so a snapshot never delays a
#: failure's report by more than one short test's whole budget; a healthy
#: engine answers a few containers' read-only calls in seconds.
SNAPSHOT_DEADLINE = 60.0

#: Label carried by every container of an osprey-deployed stack, matched by key
#: alone: its value is per project, and no run-wide value exists.
_DEPLOY_LABEL = "com.osprey.repo-id"

#: The name the snapshot plugin is registered under.
CONTAINER_SNAPSHOT_PLUGIN = "osprey-ci-diag-containers"

#: The name the recorder plugin is registered under.
RECORDER_PLUGIN = "osprey-ci-diag-recorder"

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

    # -- pytest hooks: registered as a plugin, see register_recorder ---------

    def pytest_runtest_logstart(self, nodeid: str) -> None:
        """Fires before setup — so a test that never returns is still on record."""
        self.record("start", nodeid=nodeid)

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        """Fires after teardown. A `start` left unmatched is the wedged test."""
        self.record("finish", nodeid=nodeid)

    def pytest_unconfigure(self) -> None:
        self.stop()

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


def snapshot_label_filters() -> list[str]:
    """The ``label=`` filters naming the containers this run's tests started.

    One filter per listing, merged by id, because the engine ANDs the filters
    of one call. This process's testcontainers session, when testcontainers is
    installed, and the osprey deploy label by key, which is host-wide: no label
    is carried by every container the suite starts.
    """
    filters = []
    with contextlib.suppress(ImportError):
        from testcontainers.core.labels import LABEL_SESSION_ID, SESSION_ID

        filters.append(f"label={LABEL_SESSION_ID}={SESSION_ID}")
    filters.append(f"label={_DEPLOY_LABEL}")
    return filters


def _safe(name: str) -> str:
    """``name`` with every character outside ``[A-Za-z0-9._-]`` replaced by ``_``."""
    return _UNSAFE.sub("_", name)


class ContainerSnapshots:
    """Snapshots this run's containers at the first failing report of each module.

    The containers are the ones this run's tests started, by label
    (:func:`snapshot_label_filters`). The whole snapshot stays within
    :data:`SNAPSHOT_DEADLINE` and runs with the test's pytest-timeout timer paused.

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
        #: Why the capture in progress stopped short; ``None`` while it has not.
        self._incomplete: str | None = None

    def _run(
        self, deadline: float, *args: str
    ) -> tuple[subprocess.CompletedProcess[str] | None, str]:
        """Run one engine call within ``deadline``; a failure is returned as text, never raised.

        A call the deadline cuts short marks the capture incomplete, and every
        later call of that capture is skipped: an engine that stopped answering
        costs the one deadline, not one bound per call.
        """
        if self._incomplete is not None:
            return None, ""
        remaining = deadline - time.monotonic()
        call = " ".join([self.engine, args[0], *args[1:][-1:]])
        if remaining <= 0:
            self._incomplete = f"snapshot deadline reached before {call}"
            return None, ""
        try:
            done = subprocess.run(
                [self.engine, *args],
                capture_output=True,
                text=True,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            self._incomplete = f"{call} exceeded the {SNAPSHOT_DEADLINE:g} s snapshot deadline"
            return None, f"({self._incomplete})\n"
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"({self.engine} {args[0]} failed: {exc})\n"
        return done, done.stdout + done.stderr

    def capture(self, nodeid: str, when: str) -> Path | None:
        """Write one module's snapshot; ``None`` when there is nothing to record.

        ``state.txt`` is written whatever happens once a container is listed or
        the capture stopped short, and ends with a ``capture incomplete:
        <reason>`` line when a call, the listing included, timed out or the
        capture raised.
        """
        module = nodeid.split("::", 1)[0]
        if module in self._modules:
            return None
        self._modules.add(module)
        self._incomplete = None

        if shutil.which(self.engine) is None:
            return None
        deadline = time.monotonic() + SNAPSHOT_DEADLINE
        target = self.root / f"{self.worker}--{_safe(module)}"
        state: list[str] = []
        ids: list[str] = []
        try:
            filters = snapshot_label_filters()
            for label in filters:
                listed, _ = self._run(deadline, "ps", "-aq", "--filter", label)
                if listed is not None and listed.returncode == 0:
                    ids.extend(cid for cid in listed.stdout.split() if cid not in ids)
            if not ids:
                return None if self._incomplete is None else target
            target.mkdir(parents=True, exist_ok=True)
            utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            (target / "failure.txt").write_text(f"nodeid={nodeid}\nphase={when}\nutc={utc}\n")
            (target / "ps.txt").write_text(
                "".join(self._run(deadline, "ps", "-a", "--filter", label)[1] for label in filters)
            )

            for cid in ids:
                if self._incomplete is not None:
                    break
                named, _ = self._run(deadline, "inspect", "--format", "{{.Name}}", cid)
                raw = named.stdout.strip() if named is not None and named.returncode == 0 else ""
                name = _safe(raw.lstrip("/")) or cid

                logs = self._run(deadline, "logs", "--timestamps", cid)[1]
                (target / f"{name}.log").write_text(logs)
                state.append(self._run(deadline, "inspect", "--format", _STATE_FORMAT, cid)[1])

                health, _ = self._run(
                    deadline, "inspect", "--format", "{{json .State.Health}}", cid
                )
                if health is not None and health.returncode == 0:
                    text = health.stdout.strip()
                    if text and text != "null":
                        (target / f"{name}.health.json").write_text(text + "\n")
        except BaseException as exc:
            if self._incomplete is None:
                self._incomplete = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self._incomplete is not None:
                state.append(f"capture incomplete: {self._incomplete}\n")
            if ids or self._incomplete is not None:
                target.mkdir(parents=True, exist_ok=True)
                (target / "state.txt").write_text("".join(state))
        return target

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item):
        outcome = yield
        report = outcome.get_result()
        if not report.failed:
            return
        # The capture is not test time: an armed pytest-timeout timer is paused
        # around it, so it can neither fire mid-capture nor spend the test's
        # budget. A func_only timer is already cancelled once the call phase
        # has returned, and re-arming it would leave a timer nothing cancels.
        timer = item.config.pluginmanager.get_plugin("timeout")
        settings = timer._get_item_settings(item) if timer is not None else None
        armed = bool(settings is not None and settings.timeout and not settings.func_only)
        if armed:
            item.config.hook.pytest_timeout_cancel_timer(item=item)
        try:
            self.capture(item.nodeid, report.when)
        except (Exception, pytest.fail.Exception, pytest.skip.Exception):
            pass  # a diagnostic must never replace the failure it observes
        finally:
            if armed:
                item.config.hook.pytest_timeout_set_timer(item=item, settings=settings)


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


def register_recorder(pluginmanager: pytest.PytestPluginManager) -> DiagnosticsRecorder | None:
    """Start the recorder and register it as a plugin when CI diag is on.

    A plugin sees each event once however many conftests call this, including
    the worker events the xdist controller relays. ``None`` when
    ``OSPREY_CI_DIAG_DIR`` is unset or empty, and when the recorder is already
    registered, so calling it twice is harmless.
    """
    if pluginmanager.has_plugin(RECORDER_PLUGIN):
        return None
    recorder = recorder_from_env()
    if recorder is None:
        return None
    recorder.start()
    pluginmanager.register(recorder, RECORDER_PLUGIN)
    return recorder


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
