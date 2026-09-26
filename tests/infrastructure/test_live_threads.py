"""The end-of-run report that names what keeps a finished pytest process alive.

Every thread or process a test here starts is released and joined in a
``finally``, so none of them outlives its test and shows up in the report this
very run prints at its own end. Other modules in the same worker may leave
non-daemon threads behind, so every in-process assertion is relative to what the
host process already carries, and the silence of a clean run is pinned in a
fresh interpreter.
"""

from __future__ import annotations

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
from tests._live_threads import report_live_threads

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Keys of the outer run that must not reach an inner pytest: its worker id,
#: its diagnostics directory, and any options it was started with.
_OUTER_RUN_KEYS = ("PYTEST_XDIST_", "PYTEST_ADDOPTS", "OSPREY_CI_DIAG_DIR")


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
    env = {k: v for k, v in os.environ.items() if not k.startswith(_OUTER_RUN_KEYS)}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests._live_threads",
            "-p",
            "no:cacheprovider",
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
        capture_output=True,
        text=True,
        timeout=120,
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
