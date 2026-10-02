"""Name what keeps a finished pytest process alive, registered by ``tests/conftest.py``.

When a pytest process -- serial, the xdist controller, or an xdist worker --
has run its last test and printed its summary, it is not done: interpreter
teardown joins every non-daemon thread (``ThreadPoolExecutor`` workers
included), and ``multiprocessing``'s atexit hook joins every non-daemon child
process. One that never finishes holds the process open after a green summary,
with nothing on screen to say why.

This plugin reports them from ``pytest_unconfigure``, the last hook before that
teardown begins: each non-daemon thread other than the main thread, with its
name and Python stack, and each non-daemon ``multiprocessing.active_children()``
entry, with its name and pid. On a clean run it enumerates the threads once and
writes nothing.

The stacks come from ``sys._current_frames()`` read in the main thread, which
holds the GIL for the walk, so no frame can change underneath it.
``faulthandler.dump_traceback_later`` walks frames without the GIL and is not
safe in this suite -- ``tests/ci_diagnostics.py`` records why.

The report observes and nothing more. It waits at most ``GRACE_SECONDS`` in
total for threads the interpreter would join anyway, and it never kills a
thread or process and never exits the interpreter.
"""

from __future__ import annotations

import multiprocessing
import sys
import threading
import time
import traceback
from typing import TextIO

import pytest

from tests.ci_diagnostics import worker_id

#: The name this plugin is registered under.
PLUGIN_NAME = "osprey-live-threads"

#: Threads a fixture or plugin is already stopping get this long, in total, to
#: finish before they count, so a thread that is merely finishing is not
#: reported.
GRACE_SECONDS = 2.0


def _lingering_threads(grace: float) -> list[threading.Thread]:
    main = threading.main_thread()
    threads = [t for t in threading.enumerate() if t is not main and not t.daemon and t.is_alive()]
    deadline = time.monotonic() + grace
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(remaining)
    return [t for t in threads if t.is_alive()]


def report_live_threads(stream: TextIO, grace: float = GRACE_SECONDS) -> int:
    """Write every non-daemon thread and child process still running to *stream*.

    Args:
        stream: Where the report goes; nothing is written when nothing is left.
        grace: Seconds, shared by all threads, to wait for threads that are
            already finishing before they count.

    Returns:
        The number of threads and processes reported. A report that fails
        writes one line saying so and returns 0; it never raises, because a
        diagnostic must never replace the outcome it observes.
    """
    worker = worker_id()
    try:
        threads = _lingering_threads(grace)
        children = [p for p in multiprocessing.active_children() if not p.daemon]
        count = len(threads) + len(children)
        if count == 0:
            return 0
        frames = sys._current_frames()
        lines = [
            f"[{worker}] pytest has finished, but {count} non-daemon thread(s)/process(es) "
            "are still running; the interpreter waits for them before it exits:\n"
        ]
        for thread in threads:
            lines.append(f"--- thread '{thread.name}' (ident {thread.ident}) ---\n")
            frame = frames.get(thread.ident) if thread.ident is not None else None
            if frame is None:
                lines.append("(finished while this report was written)\n")
            else:
                lines.append("".join(traceback.format_stack(frame)))
        for child in children:
            lines.append(f"--- child process '{child.name}' (pid {child.pid}) ---\n")
        stream.write("".join(lines))
        stream.flush()
        return count
    except Exception as exc:
        try:
            stream.write(f"[{worker}] live-thread report failed: {exc!r}\n")
            stream.flush()
        except Exception:
            pass
        return 0


@pytest.hookimpl(trylast=True)
def pytest_unconfigure() -> None:
    """Report after every other plugin has stopped what it owns."""
    report_live_threads(sys.stderr)
