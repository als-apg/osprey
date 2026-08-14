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

Both are gated on ``OSPREY_CI_DIAG_DIR``: unset (every local run) means this
module installs nothing at all.

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
"""

from __future__ import annotations

import faulthandler
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

#: Enables the whole module, and names the directory both records are written
#: to. The CI job uploads that directory as an artifact.
ENV_DIR = "OSPREY_CI_DIAG_DIR"

#: Seconds between whole-process stack snapshots. ``0`` disables stack dumping
#: while leaving the event log on.
ENV_STACK_TIMEOUT = "OSPREY_CI_DIAG_STACK_TIMEOUT"

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
