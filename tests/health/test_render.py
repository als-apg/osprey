"""Tests for health-report rendering (Rich grouped output and machine-clean JSON)."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

from osprey.health.models import CheckReport, CheckResult, Status
from osprey.health.render import render_json, render_report, run_progress


def _capture_console() -> tuple[Console, StringIO]:
    """A plain (no-color) console writing to an in-memory buffer for assertions."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=200)
    return console, buf


def _report(*results: CheckResult, **kwargs) -> CheckReport:
    return CheckReport(results=list(results), **kwargs)


def _ok(name: str, category: str = "file_system", **kw) -> CheckResult:
    return CheckResult(name, category, Status.OK, kw.pop("message", f"{name} ok"), **kw)


def _warn(name: str, category: str = "file_system", **kw) -> CheckResult:
    return CheckResult(name, category, Status.WARNING, kw.pop("message", f"{name} warn"), **kw)


def _err(name: str, category: str = "file_system", **kw) -> CheckResult:
    return CheckResult(name, category, Status.ERROR, kw.pop("message", f"{name} err"), **kw)


def _skip(name: str, category: str = "file_system", **kw) -> CheckResult:
    return CheckResult(name, category, Status.SKIP, kw.pop("message", f"{name} skip"), **kw)


# --------------------------------------------------------------------------- #
# render_report
# --------------------------------------------------------------------------- #


class TestRenderReport:
    def test_per_status_glyphs(self) -> None:
        report = _report(
            _ok("a", message="alpha ok"),
            _warn("b", message="beta warn"),
            _err("c", message="gamma err"),
            _skip("d", message="delta skip"),
        )
        console, buf = _capture_console()
        render_report(report, console=console)
        out = buf.getvalue()
        assert "✓ alpha ok" in out
        assert "! beta warn" in out
        assert "✗ gamma err" in out
        assert "- delta skip" in out

    def test_humanized_category_headers(self) -> None:
        report = _report(
            _ok("a", category="file_system"),
            _ok("b", category="python_environment"),
        )
        console, buf = _capture_console()
        render_report(report, console=console)
        out = buf.getvalue()
        assert "File System" in out
        assert "Python Environment" in out

    def test_grouping_preserves_first_seen_order(self) -> None:
        report = _report(
            _ok("a", category="containers"),
            _ok("b", category="providers"),
            _ok("c", category="containers"),
        )
        console, buf = _capture_console()
        render_report(report, console=console)
        out = buf.getvalue()
        assert out.index("Containers") < out.index("Providers")

    def test_value_is_appended_in_parentheses(self) -> None:
        report = _report(_ok("beam", message="beam current", value="401.2 mA"))
        console, buf = _capture_console()
        render_report(report, console=console)
        assert "(401.2 mA)" in buf.getvalue()

    def test_panel_title_and_summary_line(self) -> None:
        report = _report(_ok("a"), _warn("b"))
        console, buf = _capture_console()
        render_report(report, console=console)
        out = buf.getvalue()
        assert "Osprey Health Check Results" in out
        assert f"Summary: {report.summary_line()}" in out

    def test_verbose_details_section_lists_warnings_and_errors(self) -> None:
        report = _report(
            _ok("a", message="fine"),
            _warn("b", message="be careful", details="try X"),
            _err("c", message="broken", details="fix Y"),
        )
        console, buf = _capture_console()
        render_report(report, verbose=True, console=console)
        out = buf.getvalue()
        assert "Details:" in out
        assert "b: be careful" in out
        assert "try X" in out
        assert "c: broken" in out
        assert "fix Y" in out
        # An ok row is not enumerated in the details section.
        assert "a: fine" not in out

    def test_non_verbose_omits_details_section(self) -> None:
        report = _report(_ok("a"), _warn("b", details="hint"))
        console, buf = _capture_console()
        render_report(report, verbose=False, console=console)
        assert "Details:" not in buf.getvalue()

    def test_details_section_absent_when_all_ok_even_if_verbose(self) -> None:
        report = _report(_ok("a"), _ok("b"))
        console, buf = _capture_console()
        render_report(report, verbose=True, console=console)
        assert "Details:" not in buf.getvalue()


# --------------------------------------------------------------------------- #
# render_json
# --------------------------------------------------------------------------- #


class TestRenderJson:
    def test_stream_receives_exact_report_dict(self) -> None:
        report = _report(_ok("a"), _warn("b"), elapsed_ms=12.3)
        out = StringIO()
        render_json(report, out=out)
        assert json.loads(out.getvalue()) == report.to_dict()

    def test_defaults_to_stdout_and_writes_nothing_to_stderr(self, capsys) -> None:
        report = _report(_ok("a"))
        render_json(report)
        captured = capsys.readouterr()
        assert json.loads(captured.out) == report.to_dict()
        assert captured.err == ""

    def test_stdout_stays_pure_json_when_human_output_goes_to_stderr(self) -> None:
        # Simulate the CLI's --json wiring: human output to a stderr console,
        # the JSON document to stdout. stdout must remain a single JSON object.
        report = _report(
            _ok("a", message="alpha ok"),
            _warn("b", message="beta warn"),
            _err("c", message="gamma err"),
        )
        stderr_console, stderr_buf = _capture_console()
        stdout_buf = StringIO()

        render_report(report, console=stderr_console)  # human output -> "stderr"
        render_json(report, out=stdout_buf)  # machine output -> "stdout"

        # stdout parses cleanly and carries no human glyphs or headers.
        assert json.loads(stdout_buf.getvalue()) == report.to_dict()
        for marker in ("✓", "!", "✗", "Summary:", "File System"):
            assert marker not in stdout_buf.getvalue()
        # The human surface did carry them.
        assert "beta warn" in stderr_buf.getvalue()

    def test_output_is_single_line_document(self) -> None:
        report = _report(_ok("a"), _err("b"))
        out = StringIO()
        render_json(report, out=out)
        # One JSON document terminated by exactly one trailing newline.
        assert out.getvalue().endswith("\n")
        assert out.getvalue().count("\n") == 1


# --------------------------------------------------------------------------- #
# run_progress
# --------------------------------------------------------------------------- #


class TestRunProgress:
    def test_context_manager_yields_and_exits_cleanly(self) -> None:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)
        entered = False
        with run_progress("checking things", console=console):
            entered = True
        assert entered  # body ran and the context exited without error

    def test_transient_spinner_leaves_no_residual_report_text(self) -> None:
        # The spinner is transient; after exit the buffer should not retain the
        # description as visible content (Live erases it on close).
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)
        with run_progress("secret-marker", console=console):
            pass
        # Transient Live erases its render; the plain description text should not
        # survive as a stable visible line.
        assert "secret-marker\n" not in buf.getvalue()
