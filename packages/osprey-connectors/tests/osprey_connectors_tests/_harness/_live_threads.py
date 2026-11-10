"""Name what keeps a finished pytest process alive, and bound how long it takes to end.

Registered by ``_guards.pytest_configure`` for every pytest process a suite starts:
serial, the xdist controller and each xdist worker. Once a process has run its
last test it is not done: interpreter teardown joins every non-daemon thread
(``ThreadPoolExecutor`` workers included), ``multiprocessing``'s atexit hook
joins every non-daemon child process, every other exit callback runs, and the
interpreter collects what is left. Anything slow there holds the process open
after a green run, with nothing on screen to say why. This plugin owns the exit
contract of every such process:

* From ``pytest_unconfigure``, the last hook before that teardown begins, it
  names each non-daemon thread other than the main thread, with its name and
  Python stack, and each non-daemon ``multiprocessing.active_children()``
  entry, with its name and pid. An idle ``ThreadPoolExecutor`` worker, parked
  on its empty work queue, is not one the interpreter should wait for, so its
  executor is shut down here. Such workers are left out of the count and named
  on one trailing line, with where and during which test the executor was made.
* It runs the final garbage collection itself, timed, and names it when it
  takes longer than ``SLOW_CLEANUP_SECONDS``. It collects only what is already
  garbage and leaves the heap as interpreter teardown expects it, so every
  finalizer teardown would run still runs.
* It arms one exit watch. A serial run or the xdist controller ends within
  ``EXIT_BOUND_SECONDS`` after its summary. An xdist worker prints no summary:
  once its tests are done it waits, idle, for its controller's exit order, and
  is killed silently ``NodeManager.EXIT_TIMEOUT`` seconds after that order. Its
  clock therefore starts where its controller's does: the order ends the
  worker's main loop and the interpreter shuts down, and the watch is armed
  from ``threading._register_atexit``, which CPython runs first at shutdown,
  before it joins threads and before any ``atexit`` callback. It fires at half
  of the kill clock, while the worker can still write. A process still alive
  at its bound writes once to its own stderr -- for a worker, the controller's
  terminal -- the main thread's stack, so the module and function holding it,
  the non-daemon threads still alive and the exit callbacks registered with
  ``atexit``, by qualified name and module. A worker held before its exit
  order arrives is outside this watch.

``atexit`` offers no way to list its callbacks, so ``atexit.register`` and
``atexit.unregister`` are wrapped when this module is imported: callbacks
registered earlier are counted, not named. ``ThreadPoolExecutor.__init__`` is
wrapped the same way, so an executor made earlier has no recorded maker.

The stacks come from ``sys._current_frames()`` read under the GIL, so no frame
can change underneath the walk. ``faulthandler.dump_traceback_later`` walks
frames without the GIL and is not safe in this suite --
``ci_diagnostics.py`` records why.

The report observes, with one exception: at the end of a process it shuts down
the idle executors it names. A report made while tests still run leaves them
alone, because later tests may submit to them. It waits at most
``GRACE_SECONDS`` in total for threads the interpreter would join anyway, writes
each line once, never raises, and never kills a thread or process or exits the
interpreter.
"""

from __future__ import annotations

import atexit
import concurrent.futures.thread
import functools
import gc
import multiprocessing
import os
import sys
import threading
import time
import traceback
import weakref
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from types import FrameType
from typing import Any, NamedTuple, TextIO

import pytest

from osprey_connectors_tests._harness.ci_diagnostics import worker_id

#: The name this plugin is registered under.
PLUGIN_NAME = "osprey-live-threads"

#: Threads a fixture or plugin is already stopping get this long, in total, to
#: finish before they count, so a thread that is merely finishing is not
#: reported.
GRACE_SECONDS = 2.0

#: A final garbage collection slower than this is named on stderr.
SLOW_CLEANUP_SECONDS = 2.0

#: How long a serial run or the xdist controller may take, after the end of its
#: last hook, to exit. A worker's bound, counted from the start of its
#: interpreter shutdown, is the smaller of this and half of the clock its
#: controller kills it on.
EXIT_BOUND_SECONDS = 10.0

#: The exit watch's thread name.
WATCH_THREAD_NAME = "osprey-exit-watch"

#: What the idle line says of an executor the record does not hold.
_UNKNOWN_ORIGIN = "created before the record started"

#: Every callback registered through ``atexit.register`` since this module was
#: imported, with the names it is reported by.
_exit_callbacks: list[tuple[Callable[..., Any], str]] = []

#: Where, and during which test, each live executor was made.
_executor_origins: weakref.WeakKeyDictionary[ThreadPoolExecutor, str] = weakref.WeakKeyDictionary()


def _callback_name(func: Callable[..., Any]) -> str:
    name = getattr(func, "__qualname__", None) or type(func).__qualname__
    module = getattr(func, "__module__", None) or type(func).__module__
    return f"{name} ({module})"


def _recording_register(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]:
    _original_register(func, *args, **kwargs)
    _exit_callbacks.append((func, _callback_name(func)))
    return func


def _recording_unregister(func: Callable[..., Any]) -> None:
    _original_unregister(func)
    _exit_callbacks[:] = [entry for entry in _exit_callbacks if entry[0] != func]


_original_register = atexit.register
_original_unregister = atexit.unregister
atexit.register = _recording_register
atexit.unregister = _recording_unregister

_original_executor_init = ThreadPoolExecutor.__init__


@functools.wraps(_original_executor_init)
def _recording_executor_init(self: ThreadPoolExecutor, *args: Any, **kwargs: Any) -> None:
    _original_executor_init(self, *args, **kwargs)
    caller = sys._getframe(1)
    origin = f"created at {caller.f_code.co_filename}:{caller.f_lineno}"
    current = os.environ.get("PYTEST_CURRENT_TEST")
    if current is not None:
        origin += f" during {current.rpartition(' (')[0]}"
    _executor_origins[self] = origin


ThreadPoolExecutor.__init__ = _recording_executor_init  # type: ignore[method-assign]


def registered_exit_callbacks() -> list[str]:
    """Every exit callback registered since this module was imported, oldest first.

    Returns:
        One ``"qualname (module)"`` entry per registration, followed by one
        entry counting the callbacks registered before the record started,
        when there are any.
    """
    names = [name for _, name in _exit_callbacks]
    count = getattr(atexit, "_ncallbacks", None)
    if count is not None:
        earlier = count() - len(names)
        if earlier > 0:
            names.append(f"and {earlier} registered before the report started")
    return names


#: The loop every ``ThreadPoolExecutor`` worker thread runs.
_EXECUTOR_WORKER_CODE = concurrent.futures.thread._worker.__code__


class _IdleWorker(NamedTuple):
    """An executor worker parked on its empty queue; ``executor`` is ``None`` once collected."""

    thread: threading.Thread
    executor: ThreadPoolExecutor | None
    work_queue: Any


def _idle_worker(thread: threading.Thread, frames: dict[int, FrameType]) -> _IdleWorker | None:
    """*thread* as an idle worker, or ``None`` when it is not one."""
    frame = frames.get(thread.ident) if thread.ident is not None else None
    if frame is None or frame.f_code is not _EXECUTOR_WORKER_CODE:
        return None
    try:
        work_queue = frame.f_locals["work_queue"]
        if not work_queue.empty():
            return None
        return _IdleWorker(thread, frame.f_locals["executor_reference"](), work_queue)
    except (KeyError, AttributeError):
        return None


def _idle_executor_worker(thread: threading.Thread, frames: dict[int, FrameType]) -> bool:
    """Whether *thread* is an executor worker parked on an empty work queue."""
    return _idle_worker(thread, frames) is not None


def _shut_down(worker: _IdleWorker) -> None:
    """Let *worker* leave its loop, through its executor while that still exists."""
    if worker.executor is None:
        worker.work_queue.put(None)
    else:
        worker.executor.shutdown(wait=False, cancel_futures=True)


def _named(worker: _IdleWorker) -> str:
    """*worker*'s name and where its executor came from, as the idle line says them."""
    origin = None if worker.executor is None else _executor_origins.get(worker.executor)
    return f"{worker.thread.name} ({origin or _UNKNOWN_ORIGIN})"


def _lingering_threads(grace: float) -> tuple[list[threading.Thread], list[_IdleWorker]]:
    """The threads the interpreter will wait for, and the idle executor workers.

    Idle workers are set aside before the grace join: they never finish on
    their own, so joining them only spends the grace.
    """
    main = threading.main_thread()
    candidates = [
        t for t in threading.enumerate() if t is not main and not t.daemon and t.is_alive()
    ]
    frames = sys._current_frames()
    idle = [w for w in (_idle_worker(t, frames) for t in candidates) if w is not None]
    parked = {w.thread for w in idle}
    threads = [t for t in candidates if t not in parked]
    deadline = time.monotonic() + grace
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(remaining)
    return [t for t in threads if t.is_alive()], idle


def _write(stream: TextIO, text: str) -> None:
    """Write *text* to *stream*, or one line saying the exit report failed; never raise."""
    try:
        stream.write(text)
        stream.flush()
    except Exception as exc:
        try:
            stream.write(f"[{worker_id()}] exit report failed: {exc!r}\n")
            stream.flush()
        except Exception:
            pass


def report_live_threads(
    stream: TextIO, grace: float = GRACE_SECONDS, *, shut_down_idle: bool = False
) -> int:
    """Write every non-daemon thread and child process still running to *stream*.

    Args:
        stream: Where the report goes; nothing is written when nothing is left.
        grace: Seconds, shared by all threads, to wait for threads that are
            already finishing before they count.
        shut_down_idle: Shut down each idle worker's executor. Only the end of
            a process asks for it: a report made while tests still run would
            take a shared executor away from the tests after it.

    Returns:
        The number of threads and processes reported; idle executor workers
        are not counted. A report that fails writes one line saying so and
        returns 0; it never raises, because a diagnostic must never replace
        the outcome it observes.
    """
    worker = worker_id()
    try:
        threads, idle = _lingering_threads(grace)
        children = [p for p in multiprocessing.active_children() if not p.daemon]
        count = len(threads) + len(children)
        lines: list[str] = []
        if count:
            frames = sys._current_frames()
            lines.append(
                f"[{worker}] pytest has finished, but {count} non-daemon thread(s)/process(es) "
                "are still running; the interpreter waits for them before it exits:\n"
            )
            for thread in threads:
                lines.append(f"--- thread '{thread.name}' (ident {thread.ident}) ---\n")
                frame = frames.get(thread.ident) if thread.ident is not None else None
                if frame is None:
                    lines.append("(finished while this report was written)\n")
                else:
                    lines.append("".join(traceback.format_stack(frame)))
            for child in children:
                lines.append(f"--- child process '{child.name}' (pid {child.pid}) ---\n")
        if idle:
            if shut_down_idle:
                for parked in idle:
                    _shut_down(parked)
            state = "shut down, not counted" if shut_down_idle else "not counted"
            names = ", ".join(_named(parked) for parked in idle)
            lines.append(f"[{worker}] {len(idle)} idle executor worker(s) {state}: {names}\n")
        if not lines:
            return 0
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


def final_collection(stream: TextIO, slow: float | None = None) -> float:
    """Run the final garbage collection, time it, and name it on *stream* when slow.

    Args:
        stream: Where the line goes when the collection took longer than *slow*.
        slow: Seconds; ``None`` reads ``SLOW_CLEANUP_SECONDS`` at call time.

    Returns:
        The collection's duration in seconds, or 0.0 when it failed.
    """
    limit = SLOW_CLEANUP_SECONDS if slow is None else slow
    try:
        start = time.monotonic()
        gc.collect()
        took = time.monotonic() - start
        if took > limit:
            _write(
                stream,
                f"[{worker_id()}] final garbage collection took {took:.2f} s "
                f"over {len(gc.get_objects())} objects\n",
            )
        return took
    except Exception as exc:
        _write(stream, f"[{worker_id()}] final garbage collection failed: {exc!r}\n")
        return 0.0


def exit_report(bound: float, after: str) -> str:
    """What this process is doing now, headed as still exiting *bound* seconds *after*."""
    main = threading.main_thread()
    lines = [f"[{worker_id()}] still exiting {bound:g} s after {after}; the main thread is at:\n"]
    frame = sys._current_frames().get(main.ident) if main.ident is not None else None
    lines.append("".join(traceback.format_stack(frame)) if frame is not None else "(no frame)\n")
    threads = [
        t.name for t in threading.enumerate() if t is not main and not t.daemon and t.is_alive()
    ]
    lines.append(f"non-daemon threads still running: {', '.join(threads) or 'none'}\n")
    callbacks = registered_exit_callbacks()
    lines.append(f"exit callbacks: {', '.join(callbacks) or 'none'}\n")
    return "".join(lines)


def _watch(bound: float, after: str) -> None:
    time.sleep(bound)
    try:
        text = exit_report(bound, after)
    except Exception as exc:
        text = f"[{worker_id()}] exit report failed: {exc!r}\n"
    _write(sys.stderr, text)


def arm_exit_watch(bound: float, after: str) -> threading.Thread:
    """Start the one-shot daemon thread that reports this process if it outlives *bound*.

    Args:
        bound: Seconds from now.
        after: What the bound counts from, as the header says it.

    Returns:
        The started thread. It writes at most once and never stops the process.
    """
    watch = threading.Thread(
        target=_watch, args=(bound, after), name=WATCH_THREAD_NAME, daemon=True
    )
    watch.start()
    return watch


def _in_worker() -> bool:
    return "PYTEST_XDIST_WORKER" in os.environ


def worker_exit_bound() -> float:
    """A worker's bound: within half of the clock its controller kills it on."""
    from xdist.workermanage import NodeManager

    return min(EXIT_BOUND_SECONDS, NodeManager.EXIT_TIMEOUT / 2)


def _arm(bound: float, after: str) -> None:
    try:
        arm_exit_watch(bound, after)
    except Exception as exc:
        _write(sys.stderr, f"[{worker_id()}] exit watch failed: {exc!r}\n")


def _arm_at_shutdown(bound: float, after: str) -> None:
    """Arm the watch when the interpreter begins to shut down, before it joins threads."""
    try:
        register = threading._register_atexit  # type: ignore[attr-defined]
        register(_arm, bound, after)
    except Exception as exc:
        _write(sys.stderr, f"[{worker_id()}] exit watch failed: {exc!r}\n")


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_unconfigure() -> Generator[None, None, None]:
    """Wrap every other plugin's teardown and report after all of it.

    The report runs after every other plugin has stopped what it owns. A serial
    run's or the controller's watch is then armed; a worker's is armed when its
    interpreter begins to shut down, the moment its controller's exit order
    takes effect.
    """
    try:
        return (yield)
    finally:
        report_live_threads(sys.stderr, shut_down_idle=True)
        final_collection(sys.stderr)
        try:
            if _in_worker():
                _arm_at_shutdown(worker_exit_bound(), "its exit began")
            else:
                _arm(EXIT_BOUND_SECONDS, "its summary")
        except Exception as exc:
            _write(sys.stderr, f"[{worker_id()}] exit watch failed: {exc!r}\n")
