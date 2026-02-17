"""End-to-end tests for the full Claude Code + OSPREY MCP integration.

These tests verify the complete workflow:
1. `osprey init` creates a project with Claude Code integration files
2. The Claude Code CLI discovers the OSPREY MCP server via .mcp.json
3. Claude calls MCP tools (archiver_read, python_execute, channel_find)
4. The tools produce real artifacts (archiver data files, PNG plots)

This is the Claude Code equivalent of test_tutorials.py's BPM tutorial test,
proving the MCP integration works soup-to-nuts.

Requires:
- Claude Code CLI installed (`brew install claude` or `npm install -g @anthropic-ai/claude-code`)
- ANTHROPIC_API_KEY environment variable set (for API tests)

Safety Note - Permission Bypass:
API tests use --dangerously-skip-permissions because:
1. Tests run in isolated tmp_path directories with no real codebase
2. Prompts are controlled and only request data retrieval + plotting
3. The project uses mock connectors (no real EPICS hardware)
4. --max-budget-usd caps API spend
This follows Anthropic's guidance for sandboxed testing environments.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.init_cmd import init


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_claude_code_available() -> bool:
    """Check if Claude Code CLI is installed and functional."""
    try:
        # Must unset CLAUDECODE to avoid nested-session guard when
        # this test file is collected from within a Claude Code session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def has_anthropic_api_key() -> bool:
    """Check if ANTHROPIC_API_KEY is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def init_project(
    tmp_path: Path,
    name: str,
    template: str = "control_assistant",
    provider: str = "anthropic",
    model: str = "claude-haiku-4-5-20251001",
) -> Path:
    """Create a project via ``osprey init`` CLI, return project_dir.

    Uses the Click test runner so we don't need a real shell.
    """
    runner = CliRunner()
    args = [
        name,
        "--template", template,
        "--output-dir", str(tmp_path),
        "--provider", provider,
        "--model", model,
    ]
    result = runner.invoke(init, args)
    assert result.exit_code == 0, f"osprey init failed: {result.output}"
    project_dir = tmp_path / name
    assert project_dir.exists(), f"Project directory not created: {project_dir}"
    return project_dir


def run_claude(
    project_dir: Path,
    prompt: str,
    timeout: int = 180,
    max_budget: str = "1.00",
) -> subprocess.CompletedProcess:
    """Run Claude Code CLI non-interactively in *project_dir*.

    Unsets ``CLAUDECODE`` env var to avoid the nested-session guard that
    triggers when ``claude`` is invoked from within an existing Claude
    Code session (e.g. during development).
    """
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    return subprocess.run(
        [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--permission-mode", "bypassPermissions",
            "--max-budget-usd", max_budget,
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(project_dir),
        env=env,
    )


def find_png_files(root: Path) -> list[Path]:
    """Recursively find all .png files under *root*."""
    return sorted(root.rglob("*.png"))


# ---------------------------------------------------------------------------
# Module-level markers & skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not is_claude_code_available(),
        reason="Claude Code CLI not installed (run: brew install claude)",
    ),
]


# ===========================================================================
# Test 1 — Smoke test (no API key required)
# ===========================================================================

class TestInitProjectClaudeCodeFilesSmoke:
    """Quick sanity check that ``osprey init`` produces valid Claude Code files.

    This complements the unit tests in ``tests/cli/test_claude_code_integration.py``
    by running in the e2e suite with the full init flow.
    """

    @pytest.mark.e2e_smoke
    def test_init_creates_valid_claude_code_files(self, tmp_path):
        """osprey init creates all 8 Claude Code files with valid content."""
        project_dir = init_project(tmp_path, "smoke-test")

        # -- All 8 files exist --
        assert (project_dir / ".mcp.json").exists()
        assert (project_dir / "CLAUDE.md").exists()
        assert (project_dir / ".claude" / "settings.json").exists()
        assert (project_dir / ".claude" / "rules" / "safety.md").exists()
        assert (project_dir / ".claude" / "hooks" / "osprey_writes_check.py").exists()
        assert (project_dir / ".claude" / "hooks" / "osprey_limits.py").exists()
        assert (project_dir / ".claude" / "hooks" / "osprey_approval.py").exists()
        assert (project_dir / ".claude" / "hooks" / "osprey_audit.py").exists()

        # -- .mcp.json has correct MCP server entry --
        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "mcpServers" in mcp_data
        assert "osprey" in mcp_data["mcpServers"]
        server = mcp_data["mcpServers"]["osprey"]
        assert server["command"] == "python"
        assert server["args"] == ["-m", "osprey.mcp_server"]
        assert "OSPREY_CONFIG" in server["env"]

        # -- Hook scripts are executable --
        hooks_dir = project_dir / ".claude" / "hooks"
        for hook_name in [
            "osprey_writes_check.py",
            "osprey_limits.py",
            "osprey_approval.py",
            "osprey_audit.py",
        ]:
            hook_path = hooks_dir / hook_name
            mode = os.stat(hook_path).st_mode
            assert mode & 0o111, f"Hook {hook_name} should be executable"

        # -- config.yml uses mock connectors --
        config_text = (project_dir / "config.yml").read_text()
        assert "mock" in config_text.lower(), (
            "control_assistant template config should use mock connectors"
        )


# ===========================================================================
# Test 2 — archiver_read + python_execute (API required)
# ===========================================================================

class TestClaudeExecutesArchiverAndPlots:
    """Verify Claude can call archiver_read then python_execute to plot data.

    This test bypasses channel_find to reduce LLM non-determinism and cost.
    It uses hardcoded channel names that the mock archiver accepts.
    """

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.skipif(
        not has_anthropic_api_key(),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_claude_executes_archiver_and_plots(self, tmp_path):
        project_dir = init_project(tmp_path, "archiver-plot-test")

        prompt = (
            "Use the archiver_read tool to retrieve data for channels "
            "'DIAG:BPM01:POSITION:X', 'DIAG:BPM02:POSITION:X', "
            "'DIAG:BPM03:POSITION:X' over the last 24 hours. "
            "Then use python_execute to create a timeseries plot of "
            "the data and save it as a PNG file in the current directory."
        )

        result = run_claude(project_dir, prompt)

        # -- Debug output --
        print(f"\n--- archiver+plot test ---")
        print(f"  return code: {result.returncode}")
        print(f"  stdout length: {len(result.stdout)} chars")
        print(f"  stderr length: {len(result.stderr)} chars")
        print(f"  stdout (first 500): {result.stdout[:500]}")
        if result.stderr:
            print(f"  stderr (first 500): {result.stderr[:500]}")

        # -- Assertions --
        assert result.returncode == 0, (
            f"Claude Code exited with code {result.returncode}\n"
            f"stderr: {result.stderr[:2000]}"
        )

        # Archiver data was produced
        archiver_dir = project_dir / "osprey-workspace" / "archiver"
        archiver_files = list(archiver_dir.rglob("*")) if archiver_dir.exists() else []
        assert len(archiver_files) > 0, (
            "No archiver data files found in osprey-workspace/archiver/. "
            "archiver_read may not have been called."
        )

        # A plot PNG was created somewhere in the project tree
        png_files = find_png_files(project_dir)
        assert len(png_files) > 0, (
            "No PNG files found anywhere in the project. "
            "python_execute may not have created a plot."
        )

        # Output mentions relevant terms (belt-and-suspenders)
        output_lower = result.stdout.lower()
        assert any(
            term in output_lower for term in ["archiver", "data", "retrieved", "channel"]
        ), f"Output doesn't mention archiver/data terms. Output: {result.stdout[:500]}"

        assert any(
            term in output_lower for term in ["plot", "figure", "png", "image", "chart", "saved"]
        ), f"Output doesn't mention plot terms. Output: {result.stdout[:500]}"

        print(f"  archiver files: {len(archiver_files)}")
        print(f"  PNG files: {[p.name for p in png_files]}")


# ===========================================================================
# Test 3 — Full BPM analysis pipeline (API required)
# ===========================================================================

class TestClaudeFullBpmAnalysisPipeline:
    """Full multi-tool pipeline: channel_find -> archiver_read -> python_execute.

    This is the Claude Code equivalent of
    ``test_tutorials.py::test_bpm_timeseries_and_correlation_tutorial``.
    It exercises channel_find (which makes its own LLM call internally)
    to discover BPM channels, then retrieves archiver data and plots.
    """

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.skipif(
        not has_anthropic_api_key(),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_claude_full_bpm_analysis_pipeline(self, tmp_path):
        project_dir = init_project(tmp_path, "bpm-pipeline-test")

        prompt = (
            "Give me a timeseries and a correlation plot of all horizontal "
            "BPM positions over the last 24 hours. Use the channel_find tool "
            "to discover BPM channels, then archiver_read to get historical "
            "data, then python_execute to create the plots. Save the plots "
            "as PNG files."
        )

        result = run_claude(project_dir, prompt, timeout=360, max_budget="1.50")

        # -- Debug output --
        print(f"\n--- full BPM pipeline test ---")
        print(f"  return code: {result.returncode}")
        print(f"  stdout length: {len(result.stdout)} chars")
        print(f"  stderr length: {len(result.stderr)} chars")
        print(f"  stdout (first 800): {result.stdout[:800]}")
        if result.stderr:
            print(f"  stderr (first 500): {result.stderr[:500]}")

        # -- Assertions --
        assert result.returncode == 0, (
            f"Claude Code exited with code {result.returncode}\n"
            f"stderr: {result.stderr[:2000]}"
        )

        # Archiver data was retrieved
        workspace_dir = project_dir / "osprey-workspace"
        archiver_dir = workspace_dir / "archiver"
        archiver_files = list(archiver_dir.rglob("*")) if archiver_dir.exists() else []
        assert len(archiver_files) > 0, (
            "No archiver data files found in osprey-workspace/archiver/. "
            "The archiver_read tool may not have been called."
        )

        # At least one PNG plot was created
        png_files = find_png_files(project_dir)
        assert len(png_files) > 0, (
            "No PNG files found in the project. "
            "python_execute may not have created plots."
        )

        # Output contains BPM-related terms
        output_lower = result.stdout.lower()
        assert any(
            term in output_lower for term in ["bpm", "channel", "position", "beam"]
        ), f"Output doesn't mention BPM terms. Output: {result.stdout[:500]}"

        # Output mentions plotting
        assert any(
            term in output_lower
            for term in ["plot", "figure", "correlation", "timeseries", "chart", "png"]
        ), f"Output doesn't mention plot terms. Output: {result.stdout[:500]}"

        print(f"  archiver files: {len(archiver_files)}")
        print(f"  PNG files: {[p.name for p in png_files]}")
