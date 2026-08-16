"""What ``osprey audit`` shows a person, and what it hands a program.

The verb has two audiences and one body. ``--json`` writes the report document
to stdout and nothing else, which is the seam
``tests/cli/test_json_keyset_capture.py`` pins the key set of. Everything else
goes through the CLI renderer: report / note / section lines for prose, and the
shared factories for the summary panel and the findings table.

The look assertions read the renderable rather than the painted output, because
a border style is a theme token the console resolves at render time: the object
is where the factory's choice is observable at all.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from osprey.cli.audit_cmd import _display_report
from osprey.cli.audit_prompts import AuditFinding, AuditReport
from osprey.cli.styles import Styles


@pytest.fixture
def printed():
    """Every renderable ``audit`` handed to ``output.table``, in order."""
    recorded: list[object] = []
    with patch("osprey.cli.output.table", side_effect=recorded.append):
        yield recorded


def _finding(severity: str = "error", **overrides: str) -> AuditFinding:
    """One finding, with every field filled and any of them replaceable."""
    fields = {
        "category": "safety",
        "severity": severity,
        "title": "Critical issue",
        "explanation": "The hook does not check the channel name.",
        "file_path": "hooks.sh",
        "recommendation": "Check it before the write.",
    }
    fields.update(overrides)
    return AuditFinding(**fields)  # type: ignore[arg-type]


def _report(*findings: AuditFinding, risk: str = "high") -> AuditReport:
    """A report carrying ``findings``."""
    return AuditReport(summary="Issues found", overall_risk=risk, findings=list(findings))


class TestMachineOutput:
    """``--json`` owns stdout: one document, nothing beside it."""

    def test_json_output_is_the_document_alone(self, capsys, printed):
        report = _report(_finding())

        _display_report(report, json_output=True, verbose=False)

        captured = capsys.readouterr()
        assert json.loads(captured.out)["overall_risk"] == "high"
        assert printed == []

    def test_json_output_prints_no_panel_and_no_table(self, capsys, printed):
        """The renderer is not reached at all on the machine path."""
        _display_report(_report(_finding()), json_output=True, verbose=True, cost=0.05, turns=10)

        assert not [r for r in printed if isinstance(r, Panel | Table)]


class TestSummaryPanel:
    """The risk verdict, framed by the shared panel factory."""

    def test_panel_comes_from_the_shared_factory(self, printed):
        _display_report(_report(), json_output=False, verbose=False)

        panels = [r for r in printed if isinstance(r, Panel)]
        assert len(panels) == 1
        assert panels[0].border_style == Styles.BORDER_DIM
        assert panels[0].box is box.ROUNDED
        assert panels[0].title == "Audit Summary"

    def test_risk_verdict_carries_the_severity_token(self, printed):
        """The verdict is a styled ``Text``, not a markup string.

        The renderer prints with markup off, and Rich carries that option into
        every nested renderable, so a panel body written as ``[error]…[/error]``
        would show its tags to the operator.
        """
        _display_report(_report(risk="high"), json_output=False, verbose=False)

        body = [r for r in printed if isinstance(r, Panel)][0].renderable
        assert isinstance(body, Text)
        assert body.plain.startswith("HIGH RISK")
        assert body.spans[0].style == Styles.ERROR

    @pytest.mark.parametrize(
        ("risk", "expected"),
        [("low", Styles.SUCCESS), ("medium", Styles.WARNING), ("high", Styles.ERROR)],
    )
    def test_each_risk_level_has_its_own_token(self, risk, expected, printed):
        _display_report(_report(risk=risk), json_output=False, verbose=False)

        body = [r for r in printed if isinstance(r, Panel)][0].renderable
        assert body.spans[0].style == expected


class TestFindings:
    """The findings table and the detail block under it."""

    def test_table_comes_from_the_shared_factory(self, printed):
        _display_report(_report(_finding()), json_output=False, verbose=False)

        tables = [r for r in printed if isinstance(r, Table)]
        assert len(tables) == 1
        assert tables[0].border_style == Styles.BORDER_DIM
        assert tables[0].header_style == Styles.HEADER
        assert tables[0].box is box.HEAVY_HEAD
        assert tables[0].show_lines is True

    def test_severity_cells_are_styled_text_not_markup(self, printed):
        """Same trap as the panel body: a cell of markup would show its tags."""
        _display_report(
            _report(_finding("error"), _finding("warning"), _finding("info")),
            json_output=False,
            verbose=False,
        )

        table = [r for r in printed if isinstance(r, Table)][0]
        cells = list(table.columns[0].cells)
        assert all(isinstance(cell, Text) for cell in cells)
        assert [cell.style for cell in cells] == [Styles.ERROR, Styles.WARNING, Styles.INFO]
        assert [cell.plain for cell in cells] == ["error", "warning", "info"]

    def test_an_unknown_severity_still_lands_in_the_table(self, printed):
        """A model the reviewer invented a level for still shows its finding."""
        _display_report(_report(_finding("critical")), json_output=False, verbose=False)

        table = [r for r in printed if isinstance(r, Table)][0]
        cell = list(table.columns[0].cells)[0]
        assert cell.plain == "critical"

    def test_detail_block_reads_at_three_altitudes(self, capsys, printed):
        """Title, then the explanation as a note, then the recommendation."""
        _display_report(_report(_finding()), json_output=False, verbose=False)

        lines = [line.rstrip() for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert "Critical issue" in lines
        assert "  The hook does not check the channel name." in lines
        assert "  Recommendation   Check it before the write." in lines

    def test_stats_line_counts_every_severity(self, capsys, printed):
        _display_report(
            _report(_finding("error"), _finding("warning"), _finding("info")),
            json_output=False,
            verbose=False,
        )

        assert "Findings: 1 errors, 1 warnings, 1 info" in capsys.readouterr().out


class TestCleanAudit:
    """A report with nothing in it still says so."""

    def test_clean_audit_says_so_and_prints_no_table(self, capsys, printed):
        _display_report(_report(risk="low"), json_output=False, verbose=False)

        assert "No findings" in capsys.readouterr().out
        assert not [r for r in printed if isinstance(r, Table)]


class TestVerboseFooter:
    """``-v`` adds what the run cost, as a note under the counts."""

    def test_cost_and_turns_print_as_a_note(self, capsys, printed):
        _display_report(_report(_finding()), json_output=False, verbose=True, cost=0.05, turns=10)

        assert "  Cost: $0.0500 | Turns: 10" in capsys.readouterr().out

    def test_no_footer_without_the_flag(self, capsys, printed):
        _display_report(_report(_finding()), json_output=False, verbose=False, cost=0.05, turns=10)

        assert "Cost:" not in capsys.readouterr().out


class TestVerbTrouble:
    """What the verb prints when it will not deliver a report."""

    def test_missing_sdk_fails_on_stderr(self, tmp_path):
        from osprey.cli.audit_cmd import audit

        with patch("osprey.cli.audit_cmd._SDK_AVAILABLE", False):
            result = CliRunner().invoke(audit, [str(tmp_path)])

        assert result.exit_code == 1
        # Re-pinned: trouble is stderr's, so the name is no longer in stdout.
        assert "claude-agent-sdk is not installed" in result.stderr
        assert "uv add claude-agent-sdk" in result.stderr
        assert "claude-agent-sdk" not in result.stdout

    def test_build_flag_on_a_directory_fails_on_stderr(self, tmp_path):
        from osprey.cli.audit_cmd import audit

        with patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True):
            result = CliRunner().invoke(audit, [str(tmp_path), "--build"])

        assert result.exit_code == 1
        assert "--build needs a .yml or .yaml profile" in result.stderr
