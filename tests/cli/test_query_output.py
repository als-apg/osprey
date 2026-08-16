"""What ``osprey query`` writes, and which stream it writes it on.

The verb has two registers and they must never mix. Its framing — the drift
warning, the refusal when there is no render, the failure when the SDK will not
start — is OSPREY's own copy and goes through the renderer onto stderr. Its
payload is somebody else's: the agent's own words, or the ``--json`` document a
script parses, both onto stdout exactly as produced.

``--json`` is the stricter half. The document has to be alone on stdout, and it
is not enough that this command prints nothing else there: a helper three frames
down that reports through the renderer would land in the middle of it. The verb
opens :func:`osprey.cli.output.machine_mode` for its whole body so that anything
in the process reaching for the renderer is redirected to stderr, and that is
what these tests pin.

``result.stdout`` and ``result.stderr`` are asserted separately throughout, never
``result.output``: click folds both into ``output`` in the order they were
written, so a line on the wrong stream would be invisible there — the exact bug
this module exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.agent_runner import EXIT_PASS, EXIT_USAGE, SDKWorkflowResult
from osprey.cli import output
from osprey.cli.query_cmd import query
from tests.cli._lifecycle_build import stub_build


def _make_result(text_blocks: list[str]) -> SDKWorkflowResult:
    """A minimal completed run whose expected servers are all connected."""
    result = SDKWorkflowResult(
        text_blocks=text_blocks,
        tool_traces=[],
        mcp_servers=[{"name": "controls", "status": "connected", "tools": []}],
    )
    result.result = MagicMock(is_error=False, subtype="end_turn")
    return result


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def deployment(lifecycle_repo: Path) -> Path:
    """A clean deployment whose render declares one MCP server."""
    build = stub_build(lifecycle_repo, config="api:\n  providers: {}\n")
    (build / ".mcp.json").write_text(json.dumps({"mcpServers": {"controls": {"command": "x"}}}))
    return lifecycle_repo


@pytest.fixture
def drifted(lifecycle_repo: Path) -> Path:
    """A deployment whose profile has moved on since the build was rendered."""
    build = stub_build(
        lifecycle_repo, stamped_hash="stale-fingerprint", config="api:\n  providers: {}\n"
    )
    (build / ".mcp.json").write_text(json.dumps({"mcpServers": {"controls": {"command": "x"}}}))
    return lifecycle_repo


def _run(runner: CliRunner, repo: Path, *args: str, answer: str = "42 mA", printer=None):
    """Invoke ``query`` over a stubbed agent run.

    Args:
        printer: Called instead of the agent run, for a test that needs
            something else in the process to print while the verb is open.
    """

    async def stubbed(*_args, **_kwargs):
        if printer is not None:
            printer()
        return _make_result([answer])

    with (
        patch("osprey.cli.query_cmd.run_query", new=stubbed),
        patch("osprey.cli.query_cmd.read_only_disallowed_tools", return_value=[]),
        patch("osprey.cli.query_cmd.expected_mcp_servers", return_value={"controls"}),
    ):
        return runner.invoke(query, ["--repo", str(repo), *args, "beam current"])


class TestPayloadPassesThrough:
    """Neither register of the payload is OSPREY's to shape."""

    def test_agent_text_is_alone_on_stdout_and_unchanged(
        self, runner: CliRunner, deployment: Path
    ) -> None:
        """Including the characters a renderer would have interpreted.

        Markup tags, a box-drawing rule and trailing spaces all survive: the
        answer is written the way the agent wrote it, not the way OSPREY would
        have styled it.
        """
        answer = "[bold]literal[/bold] │ 100%   "

        result = _run(runner, deployment, answer=answer)

        assert result.exit_code == EXIT_PASS
        assert result.stdout == answer + "\n"

    def test_json_is_the_whole_of_stdout(self, runner: CliRunner, deployment: Path) -> None:
        result = _run(runner, deployment, "--json")

        assert result.exit_code == EXIT_PASS
        assert json.loads(result.stdout)["final_text"] == "42 mA"

    def test_a_renderer_line_from_inside_the_run_cannot_reach_the_document(
        self, runner: CliRunner, deployment: Path
    ) -> None:
        """The point of the machine-mode block, and the thing it alone buys.

        The verb prints nothing of its own here. What would break the document
        is any of the code it calls reaching for the renderer, so the stub does
        exactly that: a report line and a note, from inside the run.
        """
        marker = "renderer-probe-marker"

        def printer() -> None:
            output.report(marker)
            output.note(marker)

        result = _run(runner, deployment, "--json", printer=printer)

        assert result.exit_code == EXIT_PASS
        assert json.loads(result.stdout)["final_text"] == "42 mA"
        assert marker not in result.stdout
        # Non-vacuous: the lines really were printed, and they went to stderr.
        assert result.stderr.count(marker) == 2

    def test_without_json_the_renderer_still_owns_stdout(
        self, runner: CliRunner, deployment: Path
    ) -> None:
        """Machine mode is entered for ``--json`` only, not for every run."""
        marker = "renderer-probe-marker"

        result = _run(runner, deployment, printer=lambda: output.report(marker))

        assert result.exit_code == EXIT_PASS
        assert marker in result.stdout


class TestFramingIsOnStderr:
    """Warnings and refusals leave the pipe alone."""

    def test_a_drift_warning_never_touches_stdout(self, runner: CliRunner, drifted: Path) -> None:
        result = _run(runner, drifted)

        assert result.exit_code == EXIT_PASS
        assert result.stdout == "42 mA\n"
        assert "⚠" in result.stderr
        assert "Answering from build/ as it was last rendered." in result.stderr

    def test_a_drift_warning_stays_off_a_json_document(
        self, runner: CliRunner, drifted: Path
    ) -> None:
        result = _run(runner, drifted, "--json")

        assert result.exit_code == EXIT_PASS
        assert json.loads(result.stdout)["final_text"] == "42 mA"
        assert "⚠" in result.stderr

    def test_no_render_is_refused_in_the_failure_shape(
        self, runner: CliRunner, lifecycle_repo: Path
    ) -> None:
        """``✗ what failed`` / cause / ``→ remedy``, and the exit code is usage."""
        result = runner.invoke(query, ["--repo", str(lifecycle_repo), "beam current"])

        assert result.exit_code == EXIT_USAGE
        assert result.stdout == ""
        assert "✗ No build found at" in result.stderr
        assert "nothing rendered yet" in result.stderr
        assert "→ run `osprey build` first" in result.stderr

    def test_a_run_that_cannot_start_is_reported_on_stderr(
        self, runner: CliRunner, deployment: Path
    ) -> None:
        with (
            patch(
                "osprey.cli.query_cmd.run_query",
                side_effect=RuntimeError("provider not configured"),
            ),
            patch("osprey.cli.query_cmd.read_only_disallowed_tools", return_value=[]),
            patch("osprey.cli.query_cmd.expected_mcp_servers", return_value=set()),
        ):
            result = runner.invoke(query, ["--repo", str(deployment), "beam current"])

        assert result.exit_code == EXIT_USAGE
        assert result.stdout == ""
        assert "provider not configured" in result.stderr

    def test_a_config_failure_carries_the_resolvers_own_hint(
        self, runner: CliRunner, deployment: Path
    ) -> None:
        """The actionable half of the message is the exception's, so keep it."""
        with (
            patch(
                "osprey.cli.query_cmd.run_query",
                side_effect=ValueError("Unknown provider 'typo'. Built-in providers: anthropic"),
            ),
            patch("osprey.cli.query_cmd.read_only_disallowed_tools", return_value=[]),
            patch("osprey.cli.query_cmd.expected_mcp_servers", return_value=set()),
        ):
            result = runner.invoke(query, ["--repo", str(deployment), "beam current"])

        assert result.exit_code == EXIT_USAGE
        assert "Could not load the project configuration" in result.stderr
        assert "Built-in providers: anthropic" in result.stderr


class TestMachineModeIsClosed:
    """The block is a context manager, so it cannot leak past the verb."""

    def test_it_is_off_again_after_a_json_run(self, runner: CliRunner, deployment: Path) -> None:
        result = _run(runner, deployment, "--json")

        assert result.exit_code == EXIT_PASS
        assert output.machine_mode_active() is False

    def test_it_is_off_again_after_a_refusal(self, runner: CliRunner, lifecycle_repo: Path) -> None:
        """A verb that exits from inside the block still closes it."""
        result = runner.invoke(query, ["--repo", str(lifecycle_repo), "--json", "beam current"])

        assert result.exit_code == EXIT_USAGE
        assert output.machine_mode_active() is False
