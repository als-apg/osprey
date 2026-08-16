"""``osprey health``'s output surface, after the move onto the CLI renderer.

``test_health_cmd.py`` owns what the command *decides* (anchors, category
selection, exit codes, the wire shape of ``--json``). This module owns what it
*prints*, and on which stream:

* the default view is renderer output and nothing else — category headers,
  glyphed rows and the summary line, with no log-shaped line from health itself;
* ``--json`` runs in machine mode, so stdout is exactly one JSON document and
  every human line (deprecation notice, progress, the verbose recap) is on
  stderr;
* a degraded check read back under ``-v`` wears the renderer's warning and
  failure shapes, and health's own failures wear the failure shape;
* off a terminal every line is plain: no escape sequence reaches either stream.

The suite is patched at ``_run_suite`` so a report of a known shape is rendered
without running a single check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.health_cmd import health
from osprey.health.models import CheckReport, CheckResult, Status

# A line a logging handler would have written: a level name at the head of the
# line, or a bracketed logger name. The renderer emits neither.
_LOG_SHAPED = re.compile(r"^\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b|^\s*\[(CONFIG|registry)\]")

_ESCAPE = "\x1b["


@pytest.fixture
def cli_runner() -> CliRunner:
    """A Click CLI test runner (stdout and stderr are captured separately)."""
    return CliRunner()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project directory whose ``config.yml`` parses and touches no network."""
    path = tmp_path / "proj"
    path.mkdir()
    (path / "config.yml").write_text(
        "project_name: health_output\n"
        "models:\n"
        "  python_code_generator:\n"
        "    provider: mock\n"
        "    model_id: mock-model\n"
    )
    return path


def _result(status: Status, name: str, message: str, **kw: object) -> CheckResult:
    return CheckResult(name=name, category="demo", status=status, message=message, **kw)


def _mixed_report() -> CheckReport:
    """One check of every status, so each row shape is exercised at once."""
    return CheckReport(
        results=[
            _result(Status.OK, "fine", "everything is fine"),
            _result(Status.WARNING, "shaky", "the store is stale", details="last write 3d ago"),
            _result(Status.ERROR, "broken", "the port is bound", details="8080 belongs to pid 42"),
            _result(Status.SKIP, "absent", "no container runtime"),
        ],
        elapsed_ms=1.5,
        deadline_hit=False,
    )


def _run(cli_runner: CliRunner, project: Path, *args: str, report: CheckReport | None = None):
    """Invoke ``health`` over *project* with the suite patched to return *report*."""
    payload = _mixed_report() if report is None else report
    with patch("osprey.cli.health_cmd._run_suite", AsyncMock(return_value=payload)):
        return cli_runner.invoke(health, ["--project", str(project), *args])


# --------------------------------------------------------------------------- #
# The default view
# --------------------------------------------------------------------------- #


class TestDefaultView:
    def test_renders_header_rows_and_summary(self, cli_runner, project) -> None:
        result = _run(cli_runner, project)
        lines = result.stdout.splitlines()
        assert "Demo" in lines
        assert "  ✓ everything is fine" in lines
        assert "  ⚠ the store is stale" in lines
        assert "  ✗ the port is bound" in lines
        assert "  - no container runtime" in lines
        assert any(line.startswith("Summary: ") for line in lines)

    def test_no_log_shaped_line_reaches_either_stream(self, cli_runner, project) -> None:
        # Health's own output is renderer output; a level-prefixed or
        # logger-bracketed line would mean loader chatter leaked into the report.
        result = _run(cli_runner, project)
        for stream in (result.stdout, result.stderr):
            offenders = [line for line in stream.splitlines() if _LOG_SHAPED.match(line)]
            assert offenders == []

    def test_report_stays_on_stdout(self, cli_runner, project) -> None:
        # Without -v the report is the whole human output: nothing on stderr.
        result = _run(cli_runner, project)
        assert "the port is bound" in result.stdout
        assert result.stderr == ""

    def test_off_terminal_output_is_plain(self, cli_runner, project) -> None:
        result = _run(cli_runner, project, "-v")
        assert _ESCAPE not in result.stdout
        assert _ESCAPE not in result.stderr


# --------------------------------------------------------------------------- #
# Degraded checks and health's own failures
# --------------------------------------------------------------------------- #


class TestTroubleShapes:
    def test_verbose_recap_wears_the_warn_and_fail_shapes(self, cli_runner, project) -> None:
        result = _run(cli_runner, project, "-v")
        err = result.stderr.splitlines()
        assert "⚠ shaky: the store is stale" in err
        assert "  last write 3d ago" in err
        assert "✗ broken: the port is bound" in err
        assert "  8080 belongs to pid 42" in err

    def test_verbose_keeps_its_display_detail_semantics(self, cli_runner, project) -> None:
        # -v is health's own detail switch: it adds the per-check recap and
        # changes nothing else about the report or the exit code.
        plain = _run(cli_runner, project)
        verbose = _run(cli_runner, project, "-v")
        assert plain.stdout == verbose.stdout
        assert plain.exit_code == verbose.exit_code
        assert "shaky: the store is stale" not in plain.stderr
        assert "shaky: the store is stale" in verbose.stderr

    def test_deprecated_basic_flag_warns_on_stderr_only(self, cli_runner, project) -> None:
        result = _run(cli_runner, project, "--basic")
        assert "⚠ --basic is deprecated and has no effect" in result.stderr
        assert "deprecated" not in result.stdout

    def test_a_failed_run_wears_the_fail_shape_on_stderr(self, cli_runner, project) -> None:
        with patch("osprey.cli.health_cmd._run_suite", AsyncMock(side_effect=RuntimeError("boom"))):
            result = cli_runner.invoke(health, ["--project", str(project)])
        assert result.exit_code == 3
        err = result.stderr.splitlines()
        assert "✗ Health check failed" in err
        assert "  boom" in err
        assert "Health check failed" not in result.stdout


# --------------------------------------------------------------------------- #
# Machine mode
# --------------------------------------------------------------------------- #


class TestJsonIsMachineClean:
    def test_stdout_is_exactly_one_json_document(self, cli_runner, project) -> None:
        result = _run(cli_runner, project, "--json")
        payload = json.loads(result.stdout)  # must not raise
        assert payload["summary"]
        # One document, one trailing newline, nothing before or after it.
        assert result.stdout.endswith("\n")
        assert result.stdout.count("\n") == 1

    def test_no_human_line_reaches_stdout(self, cli_runner, project) -> None:
        result = _run(cli_runner, project, "--json")
        for marker in ("✓", "⚠", "✗", "Summary:", "Demo"):
            assert marker not in result.stdout

    def test_human_output_lands_on_stderr(self, cli_runner, project) -> None:
        result = _run(cli_runner, project, "--json", "--basic")
        assert "⚠ --basic is deprecated and has no effect" in result.stderr
        json.loads(result.stdout)

    def test_the_progress_region_does_not_paint_on_stdout(self, cli_runner, project) -> None:
        # The spinner still runs under --json; machine mode moves its live
        # region to the stderr console so the document is never painted over.
        result = _run(cli_runner, project, "--json")
        assert json.loads(result.stdout)
        assert _ESCAPE not in result.stdout

    def test_verbose_adds_no_prose_to_a_json_run(self, cli_runner, project) -> None:
        # ``--json`` renders the document instead of the human report, so -v has
        # no recap to add: the details are already in every row of the payload.
        result = _run(cli_runner, project, "--json", "-v")
        payload = json.loads(result.stdout)
        assert payload["results"][1]["details"] == "last write 3d ago"
        assert "shaky: the store is stale" not in result.stderr

    def test_a_failed_run_leaves_stdout_empty(self, cli_runner, project) -> None:
        # No document was produced, so stdout carries nothing at all: a caller
        # piping it to a parser sees an empty stream rather than half a report.
        with patch("osprey.cli.health_cmd._run_suite", AsyncMock(side_effect=RuntimeError("boom"))):
            result = cli_runner.invoke(health, ["--project", str(project), "--json"])
        assert result.exit_code == 3
        assert result.stdout == ""
        assert "✗ Health check failed" in result.stderr
