"""The exit report that names what keeps a finished pytest process alive.

Every thread or process a test here starts is released and joined in a
``finally``, so none of them outlives its test and shows up in the report this
very run prints at its own end. Other modules in the same worker may leave
non-daemon threads behind, so every in-process assertion is relative to what the
host process already carries, and the silence of a clean run is pinned in a
fresh interpreter.
"""

from __future__ import annotations

import atexit
import gc
import io
import multiprocessing
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import FrameType

import pytest

from osprey_connectors_tests._harness import _live_threads
from osprey_connectors_tests._harness._live_threads import (
    arm_exit_watch,
    final_collection,
    registered_exit_callbacks,
    report_live_threads,
)
from tests._live import wait_until
from tests._nested_pytest import outer_run_free_env, run_nested_pytest, with_harness_path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The inner run loads only the plugins it names, so its start-up cost does not
#: grow with every plugin installed in the environment.
_INNER_PLUGINS = (
    "xdist.plugin",
    "osprey_connectors_tests._harness._live_threads",
    "no:cacheprovider",
)

#: Under ``--cov`` coverage's subprocess patch hands every child process a key
#: with this prefix, and each inner process then saves its coverage data at
#: exit, inside the 0.5 s exit bound the inner modules set and, on a busy
#: runner, past it. The inner run executes only pytest and test code, which
#: coverage does not measure.
_COVERAGE_HANDOFF_PREFIX = "COVERAGE_PROCESS_"


def _hold(release: threading.Event) -> None:
    release.wait(30)


def _already_running() -> int:
    """How many non-daemon threads and children this process already carries.

    Other modules in the same worker may leave threads behind; the report
    counts them too, so every in-process assertion here is relative to it.
    """
    return report_live_threads(io.StringIO(), grace=0.0)


def _run_inner_pytest(
    tmp_path: Path, module_file: Path, extra: list[str]
) -> subprocess.CompletedProcess[str]:
    env = with_harness_path(outer_run_free_env(also_drop=(_COVERAGE_HANDOFF_PREFIX,)))
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    plugin_args = [arg for name in _INNER_PLUGINS for arg in ("-p", name)]
    return run_nested_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            *plugin_args,
            "-c",
            os.devnull,
            "--rootdir",
            str(tmp_path),
            str(module_file),
            "-q",
            *extra,
        ],
        cwd=_REPO_ROOT,
        env=env,
    )


def test_a_non_daemon_thread_is_named_with_its_stack():
    release = threading.Event()
    thread = threading.Thread(target=_hold, args=(release,), name="held-by-test")
    before = _already_running()
    thread.start()
    try:
        buf = io.StringIO()
        assert report_live_threads(buf, grace=0.0) == before + 1
        out = buf.getvalue()
        assert "'held-by-test'" in out
        assert "non-daemon" in out
        assert "in _hold" in out
    finally:
        release.set()
        thread.join(10)


def test_a_daemon_thread_is_not_reported():
    release = threading.Event()
    thread = threading.Thread(target=_hold, args=(release,), name="daemon-by-test", daemon=True)
    before = _already_running()
    thread.start()
    try:
        buf = io.StringIO()
        assert report_live_threads(buf, grace=0.0) == before
        assert "'daemon-by-test'" not in buf.getvalue()
    finally:
        release.set()
        thread.join(10)


def test_nothing_left_writes_nothing(tmp_path):
    module_file = tmp_path / "test_clean.py"
    module_file.write_text(_CLEAN_MODULE)
    result = _run_inner_pytest(tmp_path, module_file, [])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "pytest has finished" not in result.stderr
    assert "live-thread report failed" not in result.stderr
    assert "final garbage collection" not in result.stderr
    assert "still exiting" not in result.stderr
    assert "exit report failed" not in result.stderr


def test_a_thread_that_finishes_within_the_grace_is_not_reported():
    thread = threading.Thread(target=time.sleep, args=(0.2,), name="finishing-by-test")
    before = _already_running()
    thread.start()
    try:
        buf = io.StringIO()
        assert report_live_threads(buf, grace=2.0) == before
        assert "'finishing-by-test'" not in buf.getvalue()
    finally:
        thread.join(10)


def _wait_until_idle(name: str) -> None:
    """Return once the executor worker *name* is parked on its empty queue."""

    def idle() -> bool:
        frames = sys._current_frames()
        return any(
            thread.name == name and _live_threads._idle_executor_worker(thread, frames)
            for thread in threading.enumerate()
        )

    wait_until(idle, timeout=30.0, interval=0.01, what=f"executor worker {name!r} going idle")


def _only_this_worker_is_idle(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Keep the host process's other idle executors, which later tests still use, out of it."""
    real = _live_threads._idle_worker

    def only_named(thread: threading.Thread, frames: dict[int, FrameType]):
        return real(thread, frames) if thread.name == name else None

    monkeypatch.setattr(_live_threads, "_idle_worker", only_named)


def test_an_idle_executor_worker_is_shut_down_and_not_counted(request, monkeypatch):
    _only_this_worker_is_idle(monkeypatch, "idle-by-test_0")
    executor = ThreadPoolExecutor(1, thread_name_prefix="idle-by-test")
    created_at = sys._getframe().f_lineno - 1
    before = _already_running()
    try:
        assert executor.submit(int, "1").result(10) == 1
        _wait_until_idle("idle-by-test_0")
        worker = next(t for t in threading.enumerate() if t.name == "idle-by-test_0")
        buf = io.StringIO()
        assert report_live_threads(buf, grace=0.0, shut_down_idle=True) == before
        trailing = buf.getvalue().splitlines()[-1]
        assert "1 idle executor worker(s) shut down, not counted: idle-by-test_0" in trailing
        assert f"(created at {__file__}:{created_at} during {request.node.nodeid})" in trailing
        worker.join(10)
        assert not worker.is_alive()
    finally:
        executor.shutdown(wait=True)


def test_a_report_during_a_run_leaves_an_idle_executor_working():
    executor = ThreadPoolExecutor(1, thread_name_prefix="kept-by-test")
    try:
        assert executor.submit(int, "1").result(10) == 1
        _wait_until_idle("kept-by-test_0")
        buf = io.StringIO()
        report_live_threads(buf, grace=0.0)
        assert "idle executor worker(s) not counted" in buf.getvalue()
        assert "kept-by-test_0 (created at" in buf.getvalue()
        assert executor.submit(int, "2").result(10) == 2
    finally:
        executor.shutdown(wait=True)


def test_an_idle_worker_of_an_unrecorded_executor_says_so(monkeypatch):
    _only_this_worker_is_idle(monkeypatch, "unrecorded-by-test_0")
    executor = ThreadPoolExecutor(1, thread_name_prefix="unrecorded-by-test")
    try:
        assert executor.submit(int, "1").result(10) == 1
        _wait_until_idle("unrecorded-by-test_0")
        del _live_threads._executor_origins[executor]
        buf = io.StringIO()
        report_live_threads(buf, grace=0.0)
        trailing = buf.getvalue().splitlines()[-1]
        assert "unrecorded-by-test_0 (created before the record started)" in trailing
    finally:
        executor.shutdown(wait=True)


def test_an_idle_worker_whose_executor_is_gone_is_released_through_its_queue():
    work_queue: queue.SimpleQueue[None] = queue.SimpleQueue()
    orphan = _live_threads._IdleWorker(threading.current_thread(), None, work_queue)
    _live_threads._shut_down(orphan)
    assert work_queue.qsize() == 1
    assert work_queue.get_nowait() is None


def test_a_busy_executor_worker_is_a_holder():
    release = threading.Event()
    executor = ThreadPoolExecutor(1, thread_name_prefix="busy-by-test")
    before = _already_running()
    try:
        executor.submit(_hold, release)
        buf = io.StringIO()
        assert report_live_threads(buf, grace=0.0) == before + 1
        out = buf.getvalue()
        assert "'busy-by-test_0'" in out
        assert "in _hold" in out
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_a_non_daemon_child_process_is_named():
    child = multiprocessing.get_context("spawn").Process(
        target=time.sleep, args=(30,), name="child-held-by-test"
    )
    before = _already_running()
    child.start()
    try:
        buf = io.StringIO()
        assert report_live_threads(buf, grace=0.0) == before + 1
        out = buf.getvalue()
        assert "'child-held-by-test'" in out
        assert f"pid {child.pid}" in out
    finally:
        child.terminate()
        child.join(10)


class _FailsOnce(io.StringIO):
    """A stream whose first write raises, as a closed or broken pipe would."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def write(self, s: str) -> int:
        if not self.failed:
            self.failed = True
            raise OSError("stream is gone")
        return super().write(s)


def test_the_report_never_raises():
    release = threading.Event()
    thread = threading.Thread(target=_hold, args=(release,), name="held-by-test")
    thread.start()
    try:
        stream = _FailsOnce()
        report_live_threads(stream, grace=0.0)
        assert stream.failed
        assert "live-thread report failed" in stream.getvalue()
    finally:
        release.set()
        thread.join(10)


def test_the_plugin_is_registered_for_every_run(request):
    assert request.config.pluginmanager.has_plugin(_live_threads.PLUGIN_NAME)


_CLEAN_MODULE = textwrap.dedent(
    """
    def test_clean():
        assert True
    """
)


_LEAKING_MODULE = textwrap.dedent(
    """
    import threading
    import time


    def _linger():
        time.sleep(3)


    def test_leak():
        threading.Thread(target=_linger, name="leaky-worker").start()


    def test_sibling():
        assert True
    """
)


@pytest.mark.parametrize("extra", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_a_leaked_thread_is_reported_after_the_summary(tmp_path, extra):
    module_file = tmp_path / "test_leak.py"
    module_file.write_text(_LEAKING_MODULE)
    result = _run_inner_pytest(tmp_path, module_file, extra)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "'leaky-worker'" in result.stderr
    assert "in _linger" in result.stderr
    assert "2 passed" in result.stdout
    if extra:
        assert "[gw" in result.stderr
        header = next(line for line in result.stderr.splitlines() if "pytest has finished" in line)
        assert header.lstrip().startswith("[gw")


def _named_exit_callback() -> None:
    pass


def test_an_exit_callback_is_recorded_by_name_and_module_until_unregistered():
    atexit.register(_named_exit_callback)
    try:
        assert f"_named_exit_callback ({__name__})" in registered_exit_callbacks()
    finally:
        atexit.unregister(_named_exit_callback)
    assert f"_named_exit_callback ({__name__})" not in registered_exit_callbacks()


def test_register_keeps_its_decorator_return():
    assert atexit.register(_named_exit_callback) is _named_exit_callback
    atexit.unregister(_named_exit_callback)


def test_a_fast_final_collection_writes_nothing_and_a_slow_one_is_named():
    buf = io.StringIO()
    frozen = gc.get_freeze_count()
    final_collection(buf, slow=60.0)
    assert buf.getvalue() == ""
    final_collection(buf, slow=-1.0)
    assert gc.get_freeze_count() == frozen
    line = buf.getvalue()
    assert "final garbage collection took" in line
    assert "objects" in line
    assert line.count("\n") == 1


_FINALIZED_PROBE = textwrap.dedent(
    """
    import sys

    from osprey_connectors_tests._harness._live_threads import final_collection


    class Cycle:
        def __init__(self):
            self.me = self

        def __del__(self):
            sys.stderr.write("finalized-at-teardown\\n")
            sys.stderr.flush()


    alive = Cycle()
    final_collection(sys.stderr)
    """
)


def test_teardown_still_finalizes_what_was_alive_at_the_final_collection():
    result = subprocess.run(
        [sys.executable, "-c", _FINALIZED_PROBE],
        cwd=_REPO_ROOT,
        env=with_harness_path(outer_run_free_env()),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "finalized-at-teardown" in result.stderr


def _held_here(seconds: float) -> None:
    time.sleep(seconds)


def test_the_exit_watch_names_where_the_main_thread_is_held(capsys):
    watch = arm_exit_watch(0.2, "its summary")
    assert watch.daemon
    _held_here(1.0)
    watch.join(10)
    err = capsys.readouterr().err
    assert "still exiting 0.2 s after its summary" in err
    assert "in _held_here" in err
    assert "exit callbacks:" in err
    assert err.count("still exiting") == 1


_SLOW_COLLECTION_MODULE = textwrap.dedent(
    """
    from osprey_connectors_tests._harness import _live_threads

    _live_threads.SLOW_CLEANUP_SECONDS = 0


    def test_clean():
        assert True
    """
)


def test_a_slow_final_collection_is_named_after_the_summary(tmp_path):
    module_file = tmp_path / "test_slow_collection.py"
    module_file.write_text(_SLOW_COLLECTION_MODULE)
    result = _run_inner_pytest(tmp_path, module_file, [])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "final garbage collection took" in result.stderr


_SLOW_EXIT_MODULE = textwrap.dedent(
    """
    import atexit
    import time

    from osprey_connectors_tests._harness import _live_threads

    _live_threads.EXIT_BOUND_SECONDS = 0.5


    def _slow_exit():
        time.sleep(3)


    atexit.register(_slow_exit)


    def test_one():
        assert True


    def test_two():
        assert True
    """
)


@pytest.mark.parametrize("extra", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_a_process_that_outlives_its_bound_names_its_exit_callback(tmp_path, extra):
    module_file = tmp_path / "test_slow_exit.py"
    module_file.write_text(_SLOW_EXIT_MODULE)
    result = _run_inner_pytest(tmp_path, module_file, extra)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "in _slow_exit" in result.stderr
    assert "_slow_exit (test_slow_exit)" in result.stderr
    header = next(line for line in result.stderr.splitlines() if "still exiting" in line)
    if extra:
        assert header.lstrip().startswith("[gw"), header
        assert "after its exit began" in header
    else:
        assert "after its summary" in header


_UNEVEN_TAIL_MODULE = textwrap.dedent(
    """
    import time

    from osprey_connectors_tests._harness import _live_threads

    _live_threads.EXIT_BOUND_SECONDS = 0.5


    def test_quick():
        assert True


    def test_slow():
        time.sleep(3)
    """
)


def test_a_worker_idle_while_its_sibling_finishes_is_not_reported(tmp_path):
    module_file = tmp_path / "test_uneven_tail.py"
    module_file.write_text(_UNEVEN_TAIL_MODULE)
    result = _run_inner_pytest(tmp_path, module_file, ["-n", "2"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "still exiting" not in result.stderr, result.stderr


_ABANDONED_EXECUTOR_MODULE = textwrap.dedent(
    """
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from osprey_connectors_tests._harness._live_threads import _idle_executor_worker

    _abandoned = []


    def test_abandon():
        _abandoned.append(ThreadPoolExecutor(1, thread_name_prefix="abandoned-by-test"))
        assert _abandoned[0].submit(int, "1").result(10) == 1
        while not any(
            _idle_executor_worker(thread, sys._current_frames())
            for thread in threading.enumerate()
        ):
            time.sleep(0.01)
    """
)


def test_an_executor_never_shut_down_is_named_and_lets_the_process_exit(tmp_path):
    module_file = tmp_path / "test_abandoned.py"
    module_file.write_text(_ABANDONED_EXECUTOR_MODULE)
    result = _run_inner_pytest(tmp_path, module_file, [])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    line = next(line for line in result.stderr.splitlines() if "idle executor worker" in line)
    assert "1 idle executor worker(s) shut down, not counted: abandoned-by-test_0" in line
    assert f"(created at {module_file}:" in line
    assert "during test_abandoned.py::test_abandon)" in line
    assert "pytest has finished" not in result.stderr
    assert "still exiting" not in result.stderr, result.stderr
