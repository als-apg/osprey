"""What a lifecycle step reports, and which class each line belongs to.

A lifecycle phase prints two kinds of line. The header and the line naming a
streamed step are phase record: they describe the build's progress and go quiet
with the rest of the record under global ``--verbose``. The pass line, the
failure, the child's transcript and the test-results table are the step's own
report -- the answer the operator ran the build to get -- so they print through
the renderer whichever reporter is installed.

These tests pin that split, the shape of each line, and the fact that the
messages the module raises did not change with the way it prints them.
"""

from __future__ import annotations

import ast
import re
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from osprey.cli import build_lifecycle
from osprey.cli.build_profile_schema import LifecycleStep
from osprey.cli.phase_reporter import NullReporter, PhaseReporter, install_reporter
from osprey.errors import BuildProfileError

_JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="checks" tests="3">
    <testcase name="reads_config" time="0.5"/>
    <testcase name="writes_output" time="1.25"><failure message="boom">boom</failure></testcase>
    <testcase name="needs_hardware" time="0"><skipped/></testcase>
  </testsuite>
</testsuites>
"""


class _RecordingReporter(PhaseReporter):
    """A reporter that keeps its phase-record lines instead of printing them.

    Subclasses :class:`PhaseReporter` so the module meets the seam it actually
    calls. Only ``emit`` is intercepted: the renderer's own output does not pass
    through here, and a test reads that off the captured streams instead.
    """

    def __init__(self) -> None:
        super().__init__(color=False)
        self._lock = threading.Lock()
        self._emitted: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:
        with self._lock:
            self._emitted.append(text)

    def emitted(self) -> list[str]:
        """The phase-record lines so far, in order."""
        with self._lock:
            return list(self._emitted)


@pytest.fixture
def recorder() -> Iterator[_RecordingReporter]:
    """Install a reporter that records the phase record for one test."""
    recording = _RecordingReporter()
    previous = install_reporter(recording)
    try:
        yield recording
    finally:
        install_reporter(previous)


@pytest.fixture
def verbose_reporter() -> Iterator[None]:
    """Install the reporter global ``--verbose`` installs, for one test."""
    previous = install_reporter(NullReporter(verbose=True))
    try:
        yield
    finally:
        install_reporter(previous)


def _echo_step(name: str, text: str = "all good", *, timeout: int = 30) -> LifecycleStep:
    """A quiet step that prints ``text`` and succeeds."""
    return LifecycleStep(name=name, run=f"echo {text}", timeout=timeout)


def test_the_phase_header_is_phase_record(tmp_path: Path, recorder: _RecordingReporter) -> None:
    """The header naming the phase goes through the reporter, not the renderer."""
    build_lifecycle._run_lifecycle_phase("post_build", [], tmp_path, tmp_path)

    assert recorder.emitted() == ["  Running post_build commands..."]


def test_a_streamed_step_is_named_on_the_record(
    tmp_path: Path, recorder: _RecordingReporter
) -> None:
    """The line opening a streamed step is record too, and comes after the header."""
    steps = [LifecycleStep(name="stream me", run="echo hello", timeout=30, stream=True)]

    build_lifecycle._run_lifecycle_phase("validate", steps, tmp_path, tmp_path)

    assert recorder.emitted() == ["  Running validate commands...", "  > stream me"]


def test_a_passing_step_reports_through_the_renderer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pass line carries the step, its cost and the tail of its output."""
    build_lifecycle._run_lifecycle_phase(
        "post_build", [_echo_step("quiet step")], tmp_path, tmp_path
    )

    printed = capsys.readouterr().out
    assert re.search(r"  ✓ quiet step \(\d+\.\ds\): all good", printed), printed
    assert "—" not in printed


def test_a_pass_line_survives_the_verbose_reporter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], verbose_reporter: None
) -> None:
    """``osprey -v build`` still gets the step's own result: it is not record."""
    build_lifecycle._run_lifecycle_phase(
        "post_build", [_echo_step("quiet step")], tmp_path, tmp_path
    )

    printed = capsys.readouterr().out
    assert "✓ quiet step" in printed
    assert "Running post_build commands" not in printed


def test_a_failing_step_fails_on_stderr_and_still_raises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure takes the renderer's shape; the raised message is unchanged."""
    steps = [LifecycleStep(name="bad step", run="sh -c 'echo the reason; exit 3'", timeout=30)]

    with pytest.raises(BuildProfileError) as raised:
        build_lifecycle._run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)

    captured = capsys.readouterr()
    assert re.search(
        r"✗ Lifecycle post_build step 'bad step' failed \(exit 3, \d+\.\ds\)", captured.err
    ), captured.err
    assert "  the reason" in captured.err
    # The step's own output belongs on the trouble stream with the failure, not
    # in whatever a caller is piping stdout into.
    assert "the reason" not in captured.out
    assert str(raised.value).startswith("Lifecycle post_build step 'bad step' failed (exit 3")
    assert str(raised.value).endswith(":\nthe reason")


def test_a_tolerated_failure_warns_instead(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``validate`` reports a failed step as a warning and carries on."""
    steps = [
        LifecycleStep(name="soft step", run="sh -c 'echo the reason; exit 1'", timeout=30),
        _echo_step("after it"),
    ]

    build_lifecycle._run_lifecycle_phase(
        "validate", steps, tmp_path, tmp_path, abort_on_failure=False
    )

    captured = capsys.readouterr()
    assert "⚠ Lifecycle validate step 'soft step' failed" in captured.err
    assert "  the reason" in captured.err
    assert "✓ after it" in captured.out


def test_a_timed_out_step_reports_its_last_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A timeout names itself, then the tail of what the step managed to say."""
    steps = [
        LifecycleStep(name="slow step", run="sh -c 'echo nearly there; sleep 30'", timeout=1),
    ]

    with pytest.raises(BuildProfileError) as raised:
        build_lifecycle._run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)

    captured = capsys.readouterr()
    assert "✗ Lifecycle post_build step 'slow step' timed out" in captured.err
    assert "  Last output:" in captured.err
    assert "  nearly there" in captured.err
    assert "\n  Last output:\nnearly there" in str(raised.value)


def test_a_step_that_cannot_start_reports_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A command that is not there fails with the OS error as its cause."""
    steps = [LifecycleStep(name="missing tool", run="osprey-no-such-command", timeout=30)]

    with pytest.raises(BuildProfileError) as raised:
        build_lifecycle._run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)

    captured = capsys.readouterr()
    assert "✗ Lifecycle post_build step 'missing tool' failed to start" in captured.err
    assert "No such file or directory" in captured.err
    assert str(raised.value).startswith(
        "Lifecycle post_build step 'missing tool' failed to start: "
    )


def test_the_test_results_table_prints_through_the_renderer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A step that left JUnit results gets them summarised under its pass line."""
    (tmp_path / "check_results.xml").write_text(_JUNIT_XML)

    build_lifecycle._run_lifecycle_phase("post_build", [_echo_step("checks")], tmp_path, tmp_path)

    printed = capsys.readouterr().out
    assert "Integration Test Results" in printed
    for cell in ("reads_config", "writes_output", "needs_hardware", "0.50s", "1.25s", "skip"):
        assert cell in printed, printed
    # Statuses are pre-styled text, so no markup tag reaches the operator.
    assert "[red]" not in printed
    assert "[green]" not in printed


def test_the_test_results_table_survives_the_verbose_reporter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], verbose_reporter: None
) -> None:
    """The table is the step's report, so ``--verbose`` does not swallow it."""
    (tmp_path / "check_results.xml").write_text(_JUNIT_XML)

    build_lifecycle._run_lifecycle_phase("post_build", [_echo_step("checks")], tmp_path, tmp_path)

    assert "Integration Test Results" in capsys.readouterr().out


def test_results_that_are_absent_or_unreadable_print_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Most steps leave no test results, and a half-written file is not an error."""
    build_lifecycle._format_junit_summary(tmp_path / "check_results.xml")
    (tmp_path / "check_results.xml").write_text("<testsuites>")
    build_lifecycle._format_junit_summary(tmp_path / "check_results.xml")

    assert capsys.readouterr().out == ""


def test_the_module_owns_no_console_of_its_own() -> None:
    """Every line here goes through the reporter or the renderer, nothing else."""
    source = Path(build_lifecycle.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    direct = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Console", "print"}
    ]

    assert direct == []
    assert "click.echo" not in source
