"""The survivable-diagnostics writer and its reader.

The whole point of this machinery is to leave a usable record behind when the
process is killed rather than exiting — a wedged xdist worker, a step timeout,
a runner OOM. So the properties under test are all about what survives:
every event is on disk *before* the next test starts, and the reader tolerates
the half-written final line a SIGKILL leaves.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from tests.ci_diagnostics import (
    ENV_DIR,
    ENV_STACK_TIMEOUT,
    DiagnosticsRecorder,
    recorder_from_env,
    worker_id,
)

_SUMMARY = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "diag_summary.py"


def _load_summary():
    """Import the reader by path — it is a standalone script, not a package."""
    spec = importlib.util.spec_from_file_location("diag_summary", _SUMMARY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diag_summary"] = module
    spec.loader.exec_module(module)
    return module


# ===================================================================
# Env gating
# ===================================================================


def test_recorder_from_env_is_none_when_unset(monkeypatch, tmp_path):
    """Local runs must be untouched — no directory, no faulthandler timer."""
    monkeypatch.delenv(ENV_DIR, raising=False)
    assert recorder_from_env() is None


def test_recorder_from_env_builds_recorder_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "diag"))
    recorder = recorder_from_env()
    assert isinstance(recorder, DiagnosticsRecorder)
    assert recorder.directory == tmp_path / "diag"


def test_worker_id_defaults_to_main(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    assert worker_id() == "main"


def test_worker_id_follows_xdist(monkeypatch):
    """Per-worker filenames are what make a four-worker freeze readable."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw2")
    assert worker_id() == "gw2"


# ===================================================================
# Writing
# ===================================================================


def test_events_are_on_disk_before_stop(tmp_path):
    """The load-bearing property: a SIGKILL mid-run still leaves the record.

    Nothing may sit in a buffer waiting for a clean shutdown that never comes,
    so the assertions read the file while the recorder is still open.
    """
    recorder = DiagnosticsRecorder(tmp_path, worker="gw0", stack_timeout=None)
    recorder.start()
    recorder.record("start", nodeid="tests/test_a.py::test_one")

    events = _read_events(tmp_path / "events-gw0.jsonl")
    assert [e["event"] for e in events] == ["session_start", "start"]
    assert events[-1]["nodeid"] == "tests/test_a.py::test_one"
    recorder.stop()


def test_records_carry_worker_pid_and_monotonic_time(tmp_path):
    recorder = DiagnosticsRecorder(tmp_path, worker="gw1", stack_timeout=None)
    recorder.start()
    recorder.record("start", nodeid="tests/test_a.py::test_one")
    recorder.stop()

    event = _read_events(tmp_path / "events-gw1.jsonl")[-1]
    assert event["worker"] == "gw1"
    assert isinstance(event["pid"], int)
    assert isinstance(event["t"], float)


def test_stop_is_idempotent_and_records_session_end(tmp_path):
    recorder = DiagnosticsRecorder(tmp_path, worker="gw0", stack_timeout=None)
    recorder.start()
    recorder.stop()
    recorder.stop()

    events = _read_events(tmp_path / "events-gw0.jsonl")
    assert [e["event"] for e in events] == ["session_start", "session_end"]


def test_recording_after_stop_is_a_no_op(tmp_path):
    """pytest_unconfigure can land before a late hook; it must not raise."""
    recorder = DiagnosticsRecorder(tmp_path, worker="gw0", stack_timeout=None)
    recorder.start()
    recorder.stop()
    recorder.record("start", nodeid="tests/test_a.py::test_late")  # must not raise


def test_stack_dump_file_is_opened_when_timeout_set(tmp_path):
    """faulthandler must target a FILE, not stderr.

    A dump that exists only in the job log is lost the moment the job is
    cancelled and its log truncated — which is the case this whole module is
    for. The file is uploaded as an artifact regardless.
    """
    recorder = DiagnosticsRecorder(tmp_path, worker="gw3", stack_timeout=600)
    recorder.start()
    try:
        assert (tmp_path / "stacks-gw3.txt").exists()
    finally:
        recorder.stop()


def test_sampling_never_arms_the_c_level_watchdog(tmp_path, monkeypatch):
    """``dump_traceback_later`` must not be used, however tempting it is.

    Its watchdog walks every thread's frames without the GIL, so a sample that
    lands inside a parser building and discarding frames — which this suite does
    for minutes at a time, in PyYAML, ruamel and Jinja2 — reads a frame that is
    being freed and kills the process. That crashed real workers on three
    consecutive CI runs, which xdist reported as ``node down: Not properly
    terminated``. The module docstring has the full chain.
    """
    import faulthandler

    called: list[object] = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: called.append(a))

    recorder = DiagnosticsRecorder(tmp_path, worker="gw0", stack_timeout=0.01)
    recorder.start()
    try:
        time.sleep(0.1)
    finally:
        recorder.stop()

    assert called == [], "the GIL-free faulthandler watchdog must never be armed"


def test_sampler_writes_dumps_on_its_interval(tmp_path):
    """The thread must actually produce dumps, not merely exist."""
    recorder = DiagnosticsRecorder(tmp_path, worker="gw0", stack_timeout=0.01)
    recorder.start()
    try:
        deadline = time.monotonic() + 5.0
        stacks = tmp_path / "stacks-gw0.txt"
        while time.monotonic() < deadline and "Thread" not in stacks.read_text():
            time.sleep(0.02)
    finally:
        recorder.stop()

    assert "Thread" in (tmp_path / "stacks-gw0.txt").read_text()


def test_stop_shuts_the_sampler_down(tmp_path):
    """No thread may outlive the recorder and write into a closed file."""
    recorder = DiagnosticsRecorder(tmp_path, worker="gw0", stack_timeout=0.01)
    recorder.start()
    sampler = recorder._sampler
    assert sampler is not None and sampler.is_alive()

    recorder.stop()

    sampler.join(timeout=5.0)
    assert not sampler.is_alive()


def test_no_stack_file_when_timeout_disabled(tmp_path):
    recorder = DiagnosticsRecorder(tmp_path, worker="gw3", stack_timeout=None)
    recorder.start()
    recorder.stop()
    assert not (tmp_path / "stacks-gw3.txt").exists()


def test_stack_timeout_read_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_STACK_TIMEOUT, "45")
    assert recorder_from_env().stack_timeout == 45.0


def test_stack_timeout_of_zero_disables_dumping(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_STACK_TIMEOUT, "0")
    assert recorder_from_env().stack_timeout is None


def test_directory_is_created(tmp_path):
    target = tmp_path / "nested" / "diag"
    recorder = DiagnosticsRecorder(target, worker="gw0", stack_timeout=None)
    recorder.start()
    recorder.stop()
    assert target.is_dir()


# ===================================================================
# Reading
# ===================================================================


def test_summary_names_the_in_flight_test_per_worker(tmp_path):
    """The question a frozen lane needs answered: what was each worker inside?"""
    _write_jsonl(
        tmp_path / "events-gw0.jsonl",
        [
            {"event": "start", "nodeid": "tests/test_a.py::done", "worker": "gw0"},
            {"event": "finish", "nodeid": "tests/test_a.py::done", "worker": "gw0"},
            {"event": "start", "nodeid": "tests/test_a.py::wedged", "worker": "gw0"},
        ],
    )
    _write_jsonl(
        tmp_path / "events-gw1.jsonl",
        [{"event": "start", "nodeid": "tests/test_b.py::also_wedged", "worker": "gw1"}],
    )

    stalled = _load_summary().stalled_tests(tmp_path)
    assert stalled == {
        "gw0": "tests/test_a.py::wedged",
        "gw1": "tests/test_b.py::also_wedged",
    }


def test_summary_reports_no_stall_when_every_test_finished(tmp_path):
    _write_jsonl(
        tmp_path / "events-gw0.jsonl",
        [
            {"event": "start", "nodeid": "tests/test_a.py::one", "worker": "gw0"},
            {"event": "finish", "nodeid": "tests/test_a.py::one", "worker": "gw0"},
            {"event": "session_end", "worker": "gw0"},
        ],
    )
    assert _load_summary().stalled_tests(tmp_path) == {}


def test_summary_tolerates_a_truncated_final_line(tmp_path):
    """A SIGKILL lands mid-`write`, so the last line is routinely half a record.

    Losing the whole report to one malformed line would defeat the purpose.
    """
    path = tmp_path / "events-gw0.jsonl"
    _write_jsonl(path, [{"event": "start", "nodeid": "tests/test_a.py::wedged", "worker": "gw0"}])
    with path.open("a") as handle:
        handle.write('{"event": "fin')  # killed mid-write

    assert _load_summary().stalled_tests(tmp_path) == {"gw0": "tests/test_a.py::wedged"}


def test_summary_handles_a_missing_directory(tmp_path):
    assert _load_summary().stalled_tests(tmp_path / "absent") == {}


def test_summary_renders_a_report_naming_the_stalled_test(tmp_path):
    _write_jsonl(
        tmp_path / "events-gw0.jsonl",
        [{"event": "start", "nodeid": "tests/test_a.py::wedged", "worker": "gw0"}],
    )
    report = _load_summary().render_report(tmp_path)
    assert "gw0" in report
    assert "tests/test_a.py::wedged" in report


def test_summary_report_says_so_when_nothing_stalled(tmp_path):
    _write_jsonl(
        tmp_path / "events-gw0.jsonl",
        [
            {"event": "start", "nodeid": "tests/test_a.py::one", "worker": "gw0"},
            {"event": "finish", "nodeid": "tests/test_a.py::one", "worker": "gw0"},
        ],
    )
    report = _load_summary().render_report(tmp_path)
    assert "no stalled" in report.lower()


def test_summary_report_distinguishes_an_empty_run(tmp_path):
    """An empty directory means the suite never started — not "all clear"."""
    report = _load_summary().render_report(tmp_path)
    assert "never started" in report.lower()


def test_summary_report_flags_an_all_worker_stall(tmp_path):
    """Every worker stalling at once is the runner-stall signature.

    Worth calling out by name: it is the difference between "one of our tests
    deadlocks" and "the VM went away", and it decides whether a rerun is
    legitimate or a papering-over.
    """
    for worker in ("gw0", "gw1"):
        _write_jsonl(
            tmp_path / f"events-{worker}.jsonl",
            [{"event": "start", "nodeid": f"tests/test_{worker}.py::t", "worker": worker}],
        )
    report = _load_summary().render_report(tmp_path)
    assert "all workers" in report.lower()
    assert "runner stalling" in report.lower()


# ===================================================================
# Round trip
# ===================================================================


def test_writer_output_is_readable_by_the_summary(tmp_path):
    """Writer and reader are separate modules; pin that they still agree."""
    recorder = DiagnosticsRecorder(tmp_path, worker="gw0", stack_timeout=None)
    recorder.start()
    recorder.record("start", nodeid="tests/test_a.py::finished")
    recorder.record("finish", nodeid="tests/test_a.py::finished")
    recorder.record("start", nodeid="tests/test_a.py::wedged")

    assert _load_summary().stalled_tests(tmp_path) == {"gw0": "tests/test_a.py::wedged"}
    recorder.stop()


# ===================================================================
# Helpers
# ===================================================================


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep the ambient CI diag settings out of these tests."""
    monkeypatch.delenv(ENV_DIR, raising=False)
    monkeypatch.delenv(ENV_STACK_TIMEOUT, raising=False)
