"""Tests for the osprey audit command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from osprey.cli.audit_cmd import _detect_target_type, _extract_json, _list_files
from osprey.cli.audit_prompts import AuditFinding, AuditReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_report() -> AuditReport:
    return AuditReport(
        summary="Test audit complete",
        overall_risk="low",
        findings=[
            AuditFinding(
                category="permissions",
                severity="warning",
                title="Open permission",
                explanation="A permission is too broad",
                file_path="settings.json",
                recommendation="Restrict the permission scope",
            ),
        ],
    )


@pytest.fixture
def sample_report_json(sample_report) -> str:
    return sample_report.model_dump_json()


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory."""
    (tmp_path / "config.yml").write_text("name: test\n")
    (tmp_path / ".claude" / "settings.json").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".claude" / "settings.json").rmdir()
    (tmp_path / ".claude").rmdir()
    subdir = tmp_path / "hooks"
    subdir.mkdir()
    (subdir / "pre_write.sh").write_text("#!/bin/bash\n")
    return tmp_path


@pytest.fixture
def tmp_profile(tmp_path):
    """Create a minimal profile YAML."""
    profile = tmp_path / "test-profile.yml"
    profile.write_text("name: test\nprovider: mock\nmodel: test\n")
    return profile


# ---------------------------------------------------------------------------
# Input detection tests
# ---------------------------------------------------------------------------


class TestDetectTargetType:
    def test_detect_yaml_profile(self, tmp_profile):
        assert _detect_target_type(str(tmp_profile)) == "profile"

    def test_detect_yaml_extension(self, tmp_path):
        f = tmp_path / "profile.yaml"
        f.write_text("name: test\n")
        assert _detect_target_type(str(f)) == "profile"

    def test_detect_directory(self, tmp_project):
        assert _detect_target_type(str(tmp_project)) == "project"

    def test_invalid_target(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        with pytest.raises(click.BadParameter, match="must be a .yml/.yaml"):
            _detect_target_type(str(f))


# ---------------------------------------------------------------------------
# File listing tests
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_lists_files(self, tmp_project):
        listing = _list_files(tmp_project)
        assert "config.yml" in listing
        assert "hooks/pre_write.sh" in listing

    def test_max_files_limit(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("")
        listing = _list_files(tmp_path, max_files=3)
        assert "... and 7 more files" in listing


# ---------------------------------------------------------------------------
# JSON extraction tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_extract_raw_json(self, sample_report_json):
        text = f"Here is the report: {sample_report_json}"
        result = _extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["overall_risk"] == "low"

    def test_extract_markdown_fenced(self, sample_report_json):
        text = f"```json\n{sample_report_json}\n```"
        result = _extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["overall_risk"] == "low"

    def test_extract_no_fence_label(self, sample_report_json):
        text = f"```\n{sample_report_json}\n```"
        result = _extract_json(text)
        assert result is not None

    def test_invalid_no_json(self):
        assert _extract_json("No JSON here at all") is None

    def test_partial_json(self):
        # Truncated output — still extracts what's there
        result = _extract_json('{"summary": "test", "overall_risk": "low"}')
        assert result is not None


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestAuditModels:
    def test_finding_roundtrip(self):
        f = AuditFinding(
            category="safety",
            severity="error",
            title="Missing limits hook",
            explanation="channel_write lacks bounds checking",
            file_path="hooks/pre_write.sh",
            recommendation="Add limits hook",
        )
        data = json.loads(f.model_dump_json())
        f2 = AuditFinding.model_validate(data)
        assert f2.category == "safety"

    def test_report_roundtrip(self, sample_report):
        data = json.loads(sample_report.model_dump_json())
        r2 = AuditReport.model_validate(data)
        assert r2.overall_risk == "low"
        assert len(r2.findings) == 1

    def test_report_empty_findings(self):
        r = AuditReport(summary="Clean", overall_risk="low", findings=[])
        assert r.findings == []


# ---------------------------------------------------------------------------
# CLI invocation tests (mock SDK)
# ---------------------------------------------------------------------------


def _make_mock_query(report_json: str):
    """Create a mock async generator that yields an AssistantMessage with report JSON."""

    async def mock_query(prompt, options):
        msg = MagicMock()
        msg.__class__.__name__ = "AssistantMessage"
        # Make isinstance check work
        text_block = MagicMock()
        text_block.text = report_json
        msg.content = [text_block]
        yield msg

    return mock_query


class TestAuditCLI:
    """Test CLI invocation with mocked SDK."""

    def _get_audit_cmd(self):
        from osprey.cli.audit_cmd import audit

        return audit

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    def test_audit_project_success(self, mock_asyncio, runner, tmp_project, sample_report):
        report_json = sample_report.model_dump_json()
        mock_asyncio.run.return_value = (report_json, 0.01, 5)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])
        assert result.exit_code == 0

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    def test_audit_json_output(self, mock_asyncio, runner, tmp_project, sample_report):
        report_json = sample_report.model_dump_json()
        mock_asyncio.run.return_value = (report_json, 0.01, 5)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project), "--json"])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["overall_risk"] == "low"

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    def test_audit_verbose(self, mock_asyncio, runner, tmp_project, sample_report):
        report_json = sample_report.model_dump_json()
        mock_asyncio.run.return_value = (report_json, 0.05, 10)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project), "-v"])
        assert result.exit_code == 0

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", False)
    def test_audit_missing_sdk(self, runner, tmp_project):
        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])
        assert result.exit_code == 1
        assert "claude-agent-sdk" in result.output

    def test_audit_nonexistent_target(self, runner):
        result = runner.invoke(self._get_audit_cmd(), ["/nonexistent/path"])
        assert result.exit_code == 2

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    def test_audit_invalid_json_output(self, mock_asyncio, runner, tmp_project):
        mock_asyncio.run.return_value = ("Not valid JSON at all", None, None)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])
        assert result.exit_code == 1

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    def test_audit_markdown_fenced_json(self, mock_asyncio, runner, tmp_project, sample_report):
        report_json = sample_report.model_dump_json()
        fenced = f"```json\n{report_json}\n```"
        mock_asyncio.run.return_value = (fenced, 0.01, 5)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Build flag tests
# ---------------------------------------------------------------------------


class TestBuildFlag:
    def _get_audit_cmd(self):
        from osprey.cli.audit_cmd import audit

        return audit

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    def test_build_flag_requires_profile(self, runner, tmp_project):
        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project), "--build"])
        assert result.exit_code == 1
        assert "requires a .yml/.yaml profile" in result.output

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @patch("osprey.cli.audit_cmd.click.get_current_context")
    def test_build_flag_invokes_build_cmd(
        self, mock_ctx, mock_asyncio, runner, tmp_profile, sample_report
    ):
        report_json = sample_report.model_dump_json()
        mock_asyncio.run.return_value = (report_json, 0.01, 5)

        mock_context = MagicMock()
        mock_ctx.return_value = mock_context

        runner.invoke(self._get_audit_cmd(), [str(tmp_profile), "--build"])
        # The build invoke is called on the context
        assert mock_context.invoke.called
        call_kwargs = mock_context.invoke.call_args
        # project_name should start with "audit-"
        args = call_kwargs[1] if call_kwargs[1] else {}
        if "project_name" in args:
            assert args["project_name"].startswith("audit-")


# ---------------------------------------------------------------------------
# Display tests
# ---------------------------------------------------------------------------


class TestDisplay:
    def test_display_error_finding(self, sample_report, capsys):
        from osprey.cli.audit_cmd import _display_report

        report = AuditReport(
            summary="Issues found",
            overall_risk="high",
            findings=[
                AuditFinding(
                    category="safety",
                    severity="error",
                    title="Critical issue",
                    explanation="Details",
                    file_path="hooks.sh",
                    recommendation="Fix it",
                ),
            ],
        )
        # Just verify it doesn't crash — Rich output goes to console, not capsys
        _display_report(report, json_output=False, verbose=False)

    def test_display_json_output(self, sample_report, capsys):
        from osprey.cli.audit_cmd import _display_report

        _display_report(sample_report, json_output=True, verbose=False)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["overall_risk"] == "low"

    def test_display_empty_findings(self):
        from osprey.cli.audit_cmd import _display_report

        report = AuditReport(summary="Clean", overall_risk="low", findings=[])
        _display_report(report, json_output=False, verbose=False)

    def test_display_verbose_with_cost(self, sample_report):
        from osprey.cli.audit_cmd import _display_report

        _display_report(
            sample_report,
            json_output=False,
            verbose=True,
            cost=0.05,
            turns=10,
        )
