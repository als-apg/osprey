"""Unit tests for the ``osprey query`` CLI command.

All tests use ``click.testing.CliRunner`` and mock out ``run_query`` (and its
helpers) so no live model is invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.agent_runner import (
    EXIT_PASS,
    EXIT_USAGE,
    EXIT_VERDICT_FAIL,
    SDKWorkflowResult,
    ToolTrace,
)
from osprey.cli.query_cmd import query

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    text_blocks: list[str] | None = None,
    tool_traces: list[ToolTrace] | None = None,
    mcp_servers: list[dict] | None = None,
    is_error: bool = False,
    subtype: str = "end_turn",
) -> SDKWorkflowResult:
    """Build a minimal SDKWorkflowResult for mocking."""
    result_msg = MagicMock()
    result_msg.is_error = is_error
    result_msg.subtype = subtype

    r = SDKWorkflowResult(
        text_blocks=text_blocks or ["Agent response here."],
        tool_traces=tool_traces or [],
        mcp_servers=mcp_servers or [],
    )
    r.result = result_msg
    return r


def _connected_servers(*names: str) -> list[dict]:
    """Return a mock MCP server snapshot with all *names* connected."""
    return [{"name": n, "status": "connected", "tools": []} for n in names]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project_with_mcp(tmp_path: Path) -> Path:
    """Minimal valid project: .mcp.json declaring 'controls' + a config.yml.

    ``config.yml`` is what ``_is_valid_project`` keys on; ``run_query`` is
    mocked in these tests, so the file's contents are never parsed.
    """
    mcp_json = {"mcpServers": {"controls": {"command": "osprey-controls"}}}
    (tmp_path / ".mcp.json").write_text(json.dumps(mcp_json))
    (tmp_path / "config.yml").write_text("api:\n  providers: {}\n")
    return tmp_path


@pytest.fixture
def project_no_mcp(tmp_path: Path) -> Path:
    """Empty directory — not a valid OSPREY project (no config.yml)."""
    return tmp_path


# ---------------------------------------------------------------------------
# Missing-project detection
# ---------------------------------------------------------------------------


class TestMissingProject:
    def test_empty_dir_exits_usage(self, runner: CliRunner, project_no_mcp: Path) -> None:
        result = runner.invoke(query, ["--project", str(project_no_mcp), "hello"])
        assert result.exit_code == EXIT_USAGE
        assert "osprey build" in result.output.lower() or "--project" in result.output

    def test_error_message_is_actionable(self, runner: CliRunner, project_no_mcp: Path) -> None:
        result = runner.invoke(query, ["--project", str(project_no_mcp), "hello"])
        # User must be told how to fix it
        assert "osprey build" in result.output or "--project" in result.output

    def test_mcp_json_only_without_config_exits_usage(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A .mcp.json-only directory is NOT valid: the runner needs config.yml,
        so it must fail fast with exit 2 rather than crash mid-resolution."""
        (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}')
        result = runner.invoke(query, ["--project", str(tmp_path), "hello"])
        assert result.exit_code == EXIT_USAGE


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_exit_zero_on_success(self, runner: CliRunner, project_with_mcp: Path) -> None:
        mock_result = _make_result(
            text_blocks=["The answer is 42."],
            mcp_servers=_connected_servers("controls"),
        )
        with (
            patch("osprey.cli.query_cmd.run_query", new=AsyncMock(return_value=mock_result)),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value={"controls"}),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "What is 6*7?"])
        assert result.exit_code == EXIT_PASS

    def test_prints_final_text(self, runner: CliRunner, project_with_mcp: Path) -> None:
        mock_result = _make_result(
            text_blocks=["The answer is 42."],
            mcp_servers=_connected_servers("controls"),
        )
        with (
            patch("osprey.cli.query_cmd.run_query", new=AsyncMock(return_value=mock_result)),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value={"controls"}),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "What is 6*7?"])
        assert "The answer is 42." in result.output


# ---------------------------------------------------------------------------
# Verdict failure (exit 1)
# ---------------------------------------------------------------------------


class TestVerdictFailure:
    def test_missing_server_exits_one(self, runner: CliRunner, project_with_mcp: Path) -> None:
        # controls server is expected but not in the snapshot
        mock_result = _make_result(
            text_blocks=["No controls server found."],
            mcp_servers=[],  # empty — controls not connected
        )
        with (
            patch("osprey.cli.query_cmd.run_query", new=AsyncMock(return_value=mock_result)),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value={"controls"}),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "ping"])
        assert result.exit_code == EXIT_VERDICT_FAIL


# ---------------------------------------------------------------------------
# Safety: disallowed_tools must always include channel_write
# ---------------------------------------------------------------------------


class TestWriteToolSafety:
    def test_disallowed_tools_cover_mcp_and_builtin_writes(
        self, runner: CliRunner, project_with_mcp: Path
    ) -> None:
        """The disallowed_tools passed to run_query must block BOTH the MCP write
        tools and the built-in write/exec tools (Bash/Write/Edit).

        Under bypassPermissions disallowed_tools is the sole write guard, so a
        gap here means the agent could write hardware via Bash. read_only_
        disallowed_tools is NOT mocked — this verifies the real union reaches
        run_query.
        """
        mock_result = _make_result(mcp_servers=_connected_servers("controls"))
        captured: list[list[str]] = []

        async def capturing_run_query(project_dir, prompt, *, disallowed_tools, **kwargs):
            captured.append(list(disallowed_tools))
            return mock_result

        with (
            patch("osprey.cli.query_cmd.run_query", side_effect=capturing_run_query),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value={"controls"}),
        ):
            runner.invoke(query, ["--project", str(project_with_mcp), "safe query"])

        assert captured, "run_query was never called"
        disallowed = captured[0]
        # MCP write tools (fail-closed fallback, no hook_config.json in fixture)
        assert "mcp__controls__channel_write" in disallowed
        assert "mcp__python__execute" in disallowed
        # Built-in write/exec/network tools — the escape hatch this guards
        for builtin in ("Bash", "Write", "Edit", "WebFetch", "WebSearch"):
            assert builtin in disallowed, f"{builtin} must be in disallowed_tools"


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_flag_parses(self, runner: CliRunner, project_with_mcp: Path) -> None:
        trace = ToolTrace(
            name="mcp__controls__read_pv",
            input={"pv": "BEAM:CURRENT"},
            result="42 mA",
            is_error=False,
        )
        mock_result = _make_result(
            text_blocks=["Beam current is 42 mA."],
            tool_traces=[trace],
            mcp_servers=_connected_servers("controls"),
        )
        with (
            patch("osprey.cli.query_cmd.run_query", new=AsyncMock(return_value=mock_result)),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value={"controls"}),
        ):
            result = runner.invoke(
                query, ["--project", str(project_with_mcp), "--json", "beam current"]
            )

        assert result.exit_code == EXIT_PASS
        data = json.loads(result.output)
        assert "final_text" in data
        assert "tool_traces" in data
        assert "mcp_servers" in data
        assert "exit_code" in data

    def test_json_final_text_content(self, runner: CliRunner, project_with_mcp: Path) -> None:
        mock_result = _make_result(
            text_blocks=["Beam current is 42 mA."],
            mcp_servers=_connected_servers("controls"),
        )
        with (
            patch("osprey.cli.query_cmd.run_query", new=AsyncMock(return_value=mock_result)),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value={"controls"}),
        ):
            result = runner.invoke(
                query, ["--project", str(project_with_mcp), "--json", "beam current"]
            )

        data = json.loads(result.output)
        assert "Beam current is 42 mA." in data["final_text"]

    def test_json_tool_traces_schema(self, runner: CliRunner, project_with_mcp: Path) -> None:
        trace = ToolTrace(
            name="mcp__controls__read_pv", input={"pv": "X"}, result="1", is_error=False
        )
        mock_result = _make_result(
            tool_traces=[trace],
            mcp_servers=_connected_servers("controls"),
        )
        with (
            patch("osprey.cli.query_cmd.run_query", new=AsyncMock(return_value=mock_result)),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value={"controls"}),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "--json", "q"])

        data = json.loads(result.output)
        assert len(data["tool_traces"]) == 1
        t = data["tool_traces"][0]
        assert set(t.keys()) >= {"name", "input", "result", "is_error"}

    def test_json_emits_valid_payload_on_verdict_failure(
        self, runner: CliRunner, project_with_mcp: Path
    ) -> None:
        """--json must still emit a parseable object on verdict failure, with
        both the process exit code and the embedded exit_code == 1."""
        mock_result = _make_result(
            text_blocks=["No controls server found."],
            mcp_servers=[],  # controls expected but absent → verdict fail
        )
        with (
            patch("osprey.cli.query_cmd.run_query", new=AsyncMock(return_value=mock_result)),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value={"controls"}),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "--json", "ping"])

        assert result.exit_code == EXIT_VERDICT_FAIL
        data = json.loads(result.output)
        assert data["exit_code"] == EXIT_VERDICT_FAIL


# ---------------------------------------------------------------------------
# Infrastructure / usage errors map to exit code 2
# ---------------------------------------------------------------------------


class TestInfraErrors:
    def test_import_error_exits_usage(self, runner: CliRunner, project_with_mcp: Path) -> None:
        with (
            patch(
                "osprey.cli.query_cmd.run_query",
                side_effect=ImportError("claude_agent_sdk not installed"),
            ),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value=set()),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "hello"])
        assert result.exit_code == EXIT_USAGE

    def test_runtime_error_exits_usage(self, runner: CliRunner, project_with_mcp: Path) -> None:
        with (
            patch(
                "osprey.cli.query_cmd.run_query",
                side_effect=RuntimeError("provider not configured"),
            ),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value=set()),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "hello"])
        assert result.exit_code == EXIT_USAGE

    def test_os_error_exits_usage(self, runner: CliRunner, project_with_mcp: Path) -> None:
        """A config-read failure (e.g. missing config.yml deep in resolution)
        surfaces as a usage error (exit 2), not an uncaught traceback."""
        with (
            patch(
                "osprey.cli.query_cmd.run_query",
                side_effect=FileNotFoundError("config.yml"),
            ),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value=set()),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "hello"])
        assert result.exit_code == EXIT_USAGE
        assert "configuration" in result.output.lower()

    def test_malformed_yaml_exits_usage(self, runner: CliRunner, project_with_mcp: Path) -> None:
        """A malformed config.yml (yaml.YAMLError) maps to exit 2."""
        import yaml

        with (
            patch(
                "osprey.cli.query_cmd.run_query",
                side_effect=yaml.YAMLError("could not parse config.yml"),
            ),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value=set()),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "hello"])
        assert result.exit_code == EXIT_USAGE

    def test_unknown_provider_value_error_exits_usage(
        self, runner: CliRunner, project_with_mcp: Path
    ) -> None:
        """An unknown/misspelled provider (resolver raises ValueError) maps to
        exit 2, and the resolver's actionable message reaches the user."""
        with (
            patch(
                "osprey.cli.query_cmd.run_query",
                side_effect=ValueError(
                    "Unknown Claude Code provider 'typo'. Built-in providers: als-apg, anthropic"
                ),
            ),
            patch(
                "osprey.cli.query_cmd.read_only_disallowed_tools",
                return_value=["mcp__controls__channel_write"],
            ),
            patch("osprey.cli.query_cmd._expected_mcp_servers", return_value=set()),
        ):
            result = runner.invoke(query, ["--project", str(project_with_mcp), "hello"])
        assert result.exit_code == EXIT_USAGE
        assert "Built-in providers" in result.output
