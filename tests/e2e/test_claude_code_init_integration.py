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
import yaml
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

    On timeout, kills the process and returns a ``CompletedProcess`` with
    returncode=-1 so callers can inspect partial stdout/stderr and run
    diagnostics instead of crashing with an unhandled ``TimeoutExpired``.
    """
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--permission-mode", "bypassPermissions",
        "--max-budget-usd", max_budget,
        prompt,
    ]
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_dir),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        # Extract whatever output was captured before timeout
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode()
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode()
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-1,
            stdout=f"[TIMEOUT after {timeout}s]\n{stdout}",
            stderr=stderr,
        )


def disable_approval(project_dir: Path) -> None:
    """Set approval.global_mode to 'disabled' in the project's config.yml.

    These E2E tests exercise the MCP tool pipeline, not the approval hooks.
    Disabling approval prevents the hooks from returning ``permissionDecision:
    ask`` which would block non-interactive ``claude --print`` invocations.
    """
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config.setdefault("approval", {})["global_mode"] = "disabled"
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def allow_all_tools(project_dir: Path) -> None:
    """Move all tools from ``permissions.ask`` to ``permissions.allow``.

    Even with ``--dangerously-skip-permissions``, tools in the ``ask`` list
    may be blocked in non-interactive ``--print`` mode. Moving them to
    ``allow`` ensures the full MCP pipeline runs unimpeded.
    """
    settings_path = project_dir / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text())
    permissions = settings.get("permissions", {})
    ask_tools = permissions.pop("ask", [])
    allow_tools = permissions.get("allow", [])
    allow_tools.extend(ask_tools)
    permissions["allow"] = allow_tools
    permissions["ask"] = []
    settings["permissions"] = permissions
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def find_png_files(root: Path) -> list[Path]:
    """Recursively find generated .png files under *root*.

    Excludes template assets (logos, icons) that ship with ``osprey init``
    and therefore don't prove that ``python_execute`` created a plot.
    """
    template_names = {"ALS_assistant_logo.png"}
    return sorted(
        p for p in root.rglob("*.png") if p.name not in template_names
    )


def diagnose_workspace(project_dir: Path, max_depth: int = 3) -> str:
    """Return a depth-limited directory tree of osprey-workspace/.

    Useful for seeing exactly what was created, even if artifacts
    landed somewhere unexpected.
    """
    workspace = project_dir / "osprey-workspace"
    if not workspace.exists():
        return "osprey-workspace/ does not exist"

    lines = []

    def _walk(path: Path, depth: int, prefix: str = "") -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, depth + 1, prefix + extension)
            else:
                size = entry.stat().st_size
                lines.append(f"{prefix}{connector}{entry.name} ({size}B)")

    lines.append("osprey-workspace/")
    _walk(workspace, 1)
    return "\n".join(lines)


def diagnose_python_execute(project_dir: Path) -> str:
    """Build a diagnostic string showing python_execute execution evidence.

    Checks four evidence layers:
    1. Execution folder existence (proves subprocess ran)
    2. Execution metadata (success/failure + errors)
    3. Artifact store index (MCP figure-save pipeline completed)
    4. Artifact PNGs (figures saved to canonical location)
    """
    parts = []

    # Layer 1: Execution folders
    exec_dir = project_dir / "osprey-workspace" / "data" / "python_executions"
    if not exec_dir.exists():
        parts.append("python_executions/ does not exist — tool likely never ran")
    else:
        runs = sorted(exec_dir.iterdir()) if exec_dir.is_dir() else []
        parts.append(f"python_executions/ has {len(runs)} run(s)")
        for run in runs:
            parts.append(f"  run: {run.name}")
            # Layer 1b: figures in execution folder
            figures = (
                list((run / "figures").glob("*.png"))
                if (run / "figures").exists()
                else []
            )
            parts.append(
                f"    figures/: {[f.name for f in figures] if figures else 'empty'}"
            )
            # Layer 2: Execution metadata
            metadata_path = run / "execution_metadata.json"
            if metadata_path.exists():
                try:
                    meta = json.loads(metadata_path.read_text())
                    parts.append(f"    success: {meta.get('success')}")
                    if meta.get("error"):
                        parts.append(f"    error: {meta['error'][:300]}")
                    if meta.get("stderr"):
                        parts.append(f"    stderr: {meta['stderr'][:300]}")
                except Exception:
                    parts.append("    metadata: unreadable")
            else:
                parts.append("    metadata: missing")
            script_path = run / "wrapped_script.py"
            parts.append(
                f"    wrapped_script.py: {'exists' if script_path.exists() else 'missing'}"
            )

    # Layer 3: Artifact store index
    artifacts_json = project_dir / "osprey-workspace" / "artifacts" / "artifacts.json"
    if artifacts_json.exists():
        try:
            index_data = json.loads(artifacts_json.read_text())
            entries = index_data.get("entries", [])
            image_entries = [
                a for a in entries
                if a.get("artifact_type", "").startswith("image")
            ]
            parts.append(
                f"artifacts.json: {len(entries)} entries, "
                f"{len(image_entries)} image(s)"
            )
            for entry in image_entries[:5]:
                parts.append(
                    f"  artifact: {entry.get('id', '?')} "
                    f"type={entry.get('artifact_type', '?')} "
                    f"filename={entry.get('filename', '?')}"
                )
        except Exception:
            parts.append("artifacts.json: exists but unreadable")
    else:
        parts.append("artifacts.json: does not exist")

    # Layer 4: Artifact PNGs
    artifacts_dir = project_dir / "osprey-workspace" / "artifacts"
    if artifacts_dir.exists():
        artifact_pngs = list(artifacts_dir.glob("*.png"))
        parts.append(
            f"artifact PNGs: {[p.name for p in artifact_pngs] if artifact_pngs else 'none'}"
        )
    else:
        parts.append("artifacts/ directory: does not exist")

    return "\n".join(parts)


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

        # -- .mcp.json has correct MCP server entries --
        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "mcpServers" in mcp_data
        # Core servers must be present
        assert "osprey-control-system" in mcp_data["mcpServers"]
        assert "osprey-python-executor" in mcp_data["mcpServers"]
        assert "osprey-workspace" in mcp_data["mcpServers"]
        assert "ariel" in mcp_data["mcpServers"]
        # Control system server has correct config env
        server = mcp_data["mcpServers"]["osprey-control-system"]
        assert "OSPREY_CONFIG" in server["env"]
        # No sentinel entries
        for key in mcp_data["mcpServers"]:
            assert "sentinel" not in key.lower(), f"Sentinel '{key}' in mcpServers"

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
        disable_approval(project_dir)
        allow_all_tools(project_dir)

        prompt = (
            "Use the archiver_read tool to retrieve data for channels "
            "'DIAG:BPM01:POSITION:X', 'DIAG:BPM02:POSITION:X', "
            "'DIAG:BPM03:POSITION:X' over the last 24 hours. "
            "Then use python_execute with execution_mode='readonly' to create "
            "a timeseries plot of the data and save it as a PNG file in the "
            "current directory. All permissions are pre-approved; do not ask "
            "for confirmation."
        )

        result = run_claude(project_dir, prompt, timeout=300)

        # -- Debug output --
        print("\n--- archiver+plot test ---")
        print(f"  return code: {result.returncode}")
        print(f"  stdout length: {len(result.stdout)} chars")
        print(f"  stderr length: {len(result.stderr)} chars")
        print(f"  stdout (first 500): {result.stdout[:500]}")
        if result.stderr:
            print(f"  stderr (first 500): {result.stderr[:500]}")

        # -- Assertions --
        assert result.returncode == 0, (
            f"Claude Code exited with code {result.returncode}\n"
            f"--- stderr (first 2000) ---\n{result.stderr[:2000]}\n"
            f"--- stdout (first 2000) ---\n{result.stdout[:2000]}\n"
            f"--- Workspace tree ---\n{diagnose_workspace(project_dir)}\n"
            f"--- Execution diagnostics ---\n{diagnose_python_execute(project_dir)}"
        )

        # Archiver data was produced (saved to osprey-workspace/data/ by DataContextStore)
        workspace_dir = project_dir / "osprey-workspace"
        data_dir = workspace_dir / "data"
        data_files = list(data_dir.rglob("*")) if data_dir.exists() else []
        assert len(data_files) > 0, (
            "No data files found in osprey-workspace/data/. "
            "archiver_read may not have been called. "
            f"Workspace contents: {list(workspace_dir.rglob('*')) if workspace_dir.exists() else 'N/A'}"
        )

        # A plot PNG was created somewhere in the project tree
        png_files = find_png_files(project_dir)
        exec_diag = diagnose_python_execute(project_dir)
        workspace_tree = diagnose_workspace(project_dir)
        assert len(png_files) > 0, (
            "No PNG files found anywhere in the project. "
            "python_execute may not have created a plot.\n"
            f"--- Execution diagnostics ---\n{exec_diag}\n"
            f"--- Workspace tree ---\n{workspace_tree}\n"
            f"--- stderr (first 1000) ---\n{result.stderr[:1000]}\n"
            f"--- Claude output (first 1000) ---\n{result.stdout[:1000]}"
        )

        # Secondary check: artifact store should have image entries
        artifacts_json = project_dir / "osprey-workspace" / "artifacts" / "artifacts.json"
        if artifacts_json.exists():
            index_data = json.loads(artifacts_json.read_text())
            entries = index_data.get("entries", [])
            image_artifacts = [
                a for a in entries
                if a.get("artifact_type", "").startswith("image")
            ]
            print(f"  artifact store: {len(entries)} entries, {len(image_artifacts)} images")

        # Output mentions relevant terms (belt-and-suspenders)
        output_lower = result.stdout.lower()
        assert any(
            term in output_lower for term in ["archiver", "data", "retrieved", "channel"]
        ), f"Output doesn't mention archiver/data terms. Output: {result.stdout[:500]}"

        assert any(
            term in output_lower for term in ["plot", "figure", "png", "image", "chart", "saved"]
        ), f"Output doesn't mention plot terms. Output: {result.stdout[:500]}"

        print(f"  data files: {len(data_files)}")
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
        disable_approval(project_dir)
        allow_all_tools(project_dir)

        prompt = (
            "Give me a timeseries and a correlation plot of all horizontal "
            "BPM positions over the last 24 hours. Use the channel_find tool "
            "to discover BPM channels, then archiver_read to get historical "
            "data, then python_execute with execution_mode='readonly' to create "
            "the plots. Save the plots as PNG files. All permissions are "
            "pre-approved; do not ask for confirmation."
        )

        result = run_claude(project_dir, prompt, timeout=360, max_budget="1.50")

        # -- Debug output --
        print("\n--- full BPM pipeline test ---")
        print(f"  return code: {result.returncode}")
        print(f"  stdout length: {len(result.stdout)} chars")
        print(f"  stderr length: {len(result.stderr)} chars")
        print(f"  stdout (first 800): {result.stdout[:800]}")
        if result.stderr:
            print(f"  stderr (first 500): {result.stderr[:500]}")

        # -- Assertions --
        assert result.returncode == 0, (
            f"Claude Code exited with code {result.returncode}\n"
            f"--- stderr (first 2000) ---\n{result.stderr[:2000]}\n"
            f"--- stdout (first 2000) ---\n{result.stdout[:2000]}\n"
            f"--- Workspace tree ---\n{diagnose_workspace(project_dir)}\n"
            f"--- Execution diagnostics ---\n{diagnose_python_execute(project_dir)}"
        )

        # Archiver data was retrieved (saved to osprey-workspace/data/ by DataContextStore)
        workspace_dir = project_dir / "osprey-workspace"
        data_dir = workspace_dir / "data"
        data_files = list(data_dir.rglob("*")) if data_dir.exists() else []
        assert len(data_files) > 0, (
            "No data files found in osprey-workspace/data/. "
            "The archiver_read tool may not have been called. "
            f"Workspace contents: {list(workspace_dir.rglob('*')) if workspace_dir.exists() else 'N/A'}"
        )

        # At least one PNG plot was created
        png_files = find_png_files(project_dir)
        exec_diag = diagnose_python_execute(project_dir)
        workspace_tree = diagnose_workspace(project_dir)
        assert len(png_files) > 0, (
            "No PNG files found in the project. "
            "python_execute may not have created plots.\n"
            f"--- Execution diagnostics ---\n{exec_diag}\n"
            f"--- Workspace tree ---\n{workspace_tree}\n"
            f"--- stderr (first 1000) ---\n{result.stderr[:1000]}\n"
            f"--- Claude output (first 1000) ---\n{result.stdout[:1000]}"
        )

        # Secondary check: artifact store should have image entries
        artifacts_json = project_dir / "osprey-workspace" / "artifacts" / "artifacts.json"
        if artifacts_json.exists():
            index_data = json.loads(artifacts_json.read_text())
            entries = index_data.get("entries", [])
            image_artifacts = [
                a for a in entries
                if a.get("artifact_type", "").startswith("image")
            ]
            print(f"  artifact store: {len(entries)} entries, {len(image_artifacts)} images")

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

        print(f"  data files: {len(data_files)}")
        print(f"  PNG files: {[p.name for p in png_files]}")
