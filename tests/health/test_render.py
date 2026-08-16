"""Tests for health-report rendering (renderer output and machine-clean JSON).

The human report prints through :mod:`osprey.cli.output`, so these tests read
the two real streams (``capsys``) rather than handing the renderer a console:
the grouped rows are report-class lines on stdout, and the verbose recap of
degraded checks wears the renderer's warning/failure shape on stderr.
"""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

from osprey.cli import output
from osprey.health.models import CheckReport, CheckResult, Status
from osprey.health.render import render_json, render_report, run_progress


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
    def test_per_status_glyphs(self, capsys) -> None:
        report = _report(
            _ok("a", message="alpha ok"),
            _warn("b", message="beta warn"),
            _err("c", message="gamma err"),
            _skip("d", message="delta skip"),
        )
        render_report(report)
        out = capsys.readouterr().out
        assert "✓ alpha ok" in out
        # The warning mark is the renderer's, and STATUS_ICONS', mark: not "!".
        assert "⚠ beta warn" in out
        assert "✗ gamma err" in out
        assert "- delta skip" in out

    def test_rows_are_indented_under_their_category_header(self, capsys) -> None:
        report = _report(_ok("a", message="alpha ok"), _skip("d", message="delta skip"))
        render_report(report)
        lines = capsys.readouterr().out.splitlines()
        assert "File System" in lines
        assert "  ✓ alpha ok" in lines
        assert "  - delta skip" in lines

    def test_humanized_category_headers(self, capsys) -> None:
        report = _report(
            _ok("a", category="file_system"),
            _ok("b", category="python_environment"),
        )
        render_report(report)
        out = capsys.readouterr().out
        assert "File System" in out
        assert "Python Environment" in out

    def test_grouping_preserves_first_seen_order(self, capsys) -> None:
        report = _report(
            _ok("a", category="containers"),
            _ok("b", category="providers"),
            _ok("c", category="containers"),
        )
        render_report(report)
        out = capsys.readouterr().out
        assert out.index("Containers") < out.index("Providers")

    def test_value_is_appended_in_parentheses(self, capsys) -> None:
        report = _report(_ok("beam", message="beam current", value="401.2 mA"))
        render_report(report)
        assert "(401.2 mA)" in capsys.readouterr().out

    def test_summary_line(self, capsys) -> None:
        report = _report(_ok("a"), _warn("b"))
        render_report(report)
        assert f"Summary: {report.summary_line()}" in capsys.readouterr().out

    def test_rows_and_summary_are_stdout_not_stderr(self, capsys) -> None:
        # The grouped report is the document the verb was run to produce, so all
        # of it lands on stdout, including the error rows.
        report = _report(_ok("a", message="alpha ok"), _err("c", message="gamma err"))
        render_report(report)
        captured = capsys.readouterr()
        assert "gamma err" in captured.out
        assert captured.err == ""

    def test_verbose_recap_uses_the_warn_and_fail_shapes_on_stderr(self, capsys) -> None:
        report = _report(
            _ok("a", message="fine"),
            _warn("b", message="be careful", details="try X"),
            _err("c", message="broken", details="fix Y"),
        )
        render_report(report, verbose=True)
        err = capsys.readouterr().err
        assert "⚠ b: be careful" in err
        assert "  try X" in err
        assert "✗ c: broken" in err
        assert "  fix Y" in err
        # An ok row is not enumerated in the recap.
        assert "a: fine" not in err

    def test_non_verbose_omits_the_recap(self, capsys) -> None:
        report = _report(_ok("a"), _warn("b", details="hint"))
        render_report(report, verbose=False)
        assert capsys.readouterr().err == ""

    def test_recap_absent_when_all_ok_even_if_verbose(self, capsys) -> None:
        report = _report(_ok("a"), _ok("b"))
        render_report(report, verbose=True)
        assert capsys.readouterr().err == ""

    def test_off_terminal_output_is_plain(self, capsys) -> None:
        # Under pytest the consoles are not terminals: no ANSI escape reaches
        # either stream, so a piped report is readable as written.
        report = _report(_ok("a", message="alpha ok"), _err("c", message="gamma err"))
        render_report(report, verbose=True)
        captured = capsys.readouterr()
        assert "\x1b[" not in captured.out
        assert "\x1b[" not in captured.err


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

    def test_stdout_stays_pure_json_when_the_report_renders_in_machine_mode(self, capsys) -> None:
        # The CLI's --json wiring: the whole human report renders inside machine
        # mode (every renderer line to stderr) while the document goes to stdout.
        report = _report(
            _ok("a", message="alpha ok"),
            _warn("b", message="beta warn"),
            _err("c", message="gamma err"),
        )
        stdout_buf = StringIO()

        with output.machine_mode():
            render_report(report)
            render_json(report, out=stdout_buf)

        captured = capsys.readouterr()
        # stdout parses cleanly and carries no human glyphs or headers.
        assert json.loads(stdout_buf.getvalue()) == report.to_dict()
        for marker in ("✓", "⚠", "✗", "Summary:", "File System"):
            assert marker not in stdout_buf.getvalue()
        # Machine mode put every human line on stderr, and none on stdout.
        assert "beta warn" in captured.err
        assert captured.out == ""

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
        entered = False
        with run_progress("checking things"):
            entered = True
        assert entered  # body ran and the context exited without error

    def test_transient_spinner_leaves_no_residual_report_text(self, capsys) -> None:
        # The spinner is transient; after exit nothing should retain the
        # description as visible content (Live erases it on close).
        with run_progress("secret-marker"):
            pass
        assert "secret-marker\n" not in capsys.readouterr().out

    def test_spinner_mounts_on_the_reporters_console(self, monkeypatch) -> None:
        # The region belongs to the console that owns the cursor: the installed
        # reporter's, not one this module builds.
        from osprey.cli import phase_reporter

        buf = StringIO()
        stub = Console(file=buf, force_terminal=True, width=80)
        monkeypatch.setattr(phase_reporter.PhaseReporter, "out", lambda self: stub)
        with run_progress("mounted here"):
            pass
        assert buf.getvalue() != ""

    def test_machine_mode_moves_the_region_to_the_stderr_console(self, monkeypatch) -> None:
        # A live region paints at its console's stream: under --json that must
        # not be stdout, so machine mode moves it to the stderr console.
        from osprey.cli import phase_reporter

        stdout_buf = StringIO()
        stub = Console(file=stdout_buf, force_terminal=True, width=80)
        monkeypatch.setattr(phase_reporter.PhaseReporter, "out", lambda self: stub)

        with output.machine_mode(), run_progress("mounted elsewhere"):
            pass

        assert stdout_buf.getvalue() == ""
