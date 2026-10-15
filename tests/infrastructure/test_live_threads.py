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
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from tests import _live_threads
from tests._live_threads import (
    arm_exit_watch,
    final_collection,
    registered_exit_callbacks,
    report_live_threads,
)
from tests._nested_pytest import outer_run_free_env, run_nested_pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The inner run loads only the plugins it names, so its start-up cost does not
#: grow with every plugin installed in the environment.
_INNER_PLUGINS = ("xdist.plugin", "tests._live_threads", "no:cacheprovider")


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
    env = outer_run_free_env()
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
    try:
        final_collection(buf, slow=60.0)
        assert buf.getvalue() == ""
        final_collection(buf, slow=-1.0)
    finally:
        gc.unfreeze()
    line = buf.getvalue()
    assert "final garbage collection took" in line
    assert "objects" in line
    assert line.count("\n") == 1


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
    from tests import _live_threads

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

    from tests import _live_threads

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

    from tests import _live_threads

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
