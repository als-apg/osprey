"""End-to-end tests for OSPREY's Claude Code skill installation workflow.

These tests verify the complete workflow:
1. `osprey claude install <task>` correctly installs a skill
2. The installed skill can be invoked by Claude Code CLI
3. Claude actually reads the instructions from the installed files

Requires:
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code` or via brew)
- ANTHROPIC_API_KEY environment variable set

Safety Note - Permission Bypass:
API tests use --dangerously-skip-permissions because:
1. Tests run in isolated tmp_path directories with no real codebase
2. Prompts are controlled and only request file reading + text output
3. Skills restrict allowed-tools (e.g., "Read" only)
4. No destructive operations or sensitive data exposure
This follows Anthropic's guidance: "Recommended only for sandboxes with no internet access"
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.claude_cmd import install_skill


def is_claude_code_available() -> bool:
    """Check if Claude Code CLI is available."""
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
    """Check if Anthropic API key is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# Skip all tests in this module if Claude Code CLI is not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not is_claude_code_available(),
        reason="Claude Code CLI not installed (run: brew install claude or npm install -g @anthropic-ai/claude-code)",
    ),
]


# =============================================================================
# Mock skill with magic number for verification
# =============================================================================
# This skill is created in a temp directory and used to verify that:
# 1. OSPREY's install command works correctly
# 2. Claude Code can read the installed instructions
# The magic number 42857 proves Claude read our specific instructions.

MOCK_SKILL_MD = """---
name: osprey-test-verification
description: >
  Test skill for verifying OSPREY skill installation works correctly.
  Outputs a specific format with a magic number to prove instructions were read.
allowed-tools: Read
---

# Test Verification Skill

Follow the instructions in instructions.md EXACTLY.
"""

MOCK_INSTRUCTIONS_MD = """# OSPREY Installation Verification

## Purpose

This skill verifies that OSPREY's skill installation workflow works correctly
and that Claude Code can read the installed instruction files.

## Required Output Format

When this skill is invoked, respond with EXACTLY this format:

```
OSPREY_VERIFICATION_OUTPUT
==========================
STATUS: INSTALLATION_VERIFIED
MAGIC_NUMBER: 42857
==========================
```

The magic number 42857 is required to prove you read these instructions.
Do not output anything else before or after this block.
"""


class TestOspreySkillInstallationWorkflow:
    """E2E tests for OSPREY's skill installation and usage workflow.

    These tests verify:
    1. `osprey claude install` correctly installs skills
    2. Installed skills have correct structure
    3. Claude Code can use the installed skills
    """

    @pytest.fixture
    def cli_runner(self):
        """Provide a Click CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_skill_source(self, tmp_path):
        """Create a mock skill in a temp directory for testing.

        This creates the source files that `osprey claude install` will read from.
        We patch get_integrations_root and get_tasks_root to include this.
        """
        # Create mock integrations directory structure
        integrations_dir = tmp_path / "mock_source" / "integrations" / "claude_code"
        skill_integration_dir = integrations_dir / "test-verification"
        skill_integration_dir.mkdir(parents=True)
        (skill_integration_dir / "SKILL.md").write_text(MOCK_SKILL_MD)

        # Create mock tasks directory structure
        tasks_dir = tmp_path / "mock_source" / "tasks"
        skill_task_dir = tasks_dir / "test-verification"
        skill_task_dir.mkdir(parents=True)
        (skill_task_dir / "instructions.md").write_text(MOCK_INSTRUCTIONS_MD)

        return {
            "integrations_root": tmp_path / "mock_source" / "integrations",
            "tasks_root": tasks_dir,
        }

    @pytest.mark.e2e_smoke
    def test_osprey_claude_install_creates_skill_files(self, cli_runner, tmp_path):
        """Test that `osprey claude install` creates the correct skill structure.

        This tests OSPREY's workflow - NOT Claude Code itself.
        We verify that the installation copies SKILL.md and instructions.md correctly.
        """
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            # Run OSPREY's installation command with real skills
            result = cli_runner.invoke(install_skill, ["pre-commit"])

            if result.exit_code != 0:
                pytest.skip(f"pre-commit skill not available: {result.output}")

            # Verify the skill was installed with correct structure
            skill_dir = Path(".claude") / "skills" / "pre-commit"
            assert skill_dir.exists(), "Skill directory not created"

            skill_file = skill_dir / "SKILL.md"
            assert skill_file.exists(), "SKILL.md not installed"

            instructions_file = skill_dir / "instructions.md"
            assert instructions_file.exists(), "instructions.md not installed"

            # Verify SKILL.md has valid frontmatter
            skill_content = skill_file.read_text()
            assert skill_content.startswith("---"), "SKILL.md missing YAML frontmatter"
            assert "name:" in skill_content, "SKILL.md missing name field"
            assert "osprey-" in skill_content, "SKILL.md name should start with osprey-"

            # Verify instructions.md is not empty
            instructions_content = instructions_file.read_text()
            assert len(instructions_content) > 100, "instructions.md appears empty"

    @pytest.mark.e2e_smoke
    def test_osprey_claude_install_with_mock_skill(self, cli_runner, tmp_path, mock_skill_source):
        """Test that `osprey claude install` works with a mocked skill source.

        This verifies our mocking strategy works before using it in API tests.
        """
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            # Patch the source directories to include our mock skill
            with (
                patch("osprey.cli.claude_cmd.get_tasks_root") as mock_tasks,
                patch("osprey.cli.claude_cmd.get_integrations_root") as mock_int,
            ):
                mock_tasks.return_value = mock_skill_source["tasks_root"]
                mock_int.return_value = mock_skill_source["integrations_root"]

                result = cli_runner.invoke(install_skill, ["test-verification"])

            assert result.exit_code == 0, f"Install failed: {result.output}"

            # Verify the mock skill was installed
            skill_dir = Path(".claude") / "skills" / "test-verification"
            assert skill_dir.exists(), "Skill directory not created"
            assert (skill_dir / "SKILL.md").exists(), "SKILL.md not installed"
            assert (skill_dir / "instructions.md").exists(), "instructions.md not installed"

            # Verify content includes our magic number
            instructions = (skill_dir / "instructions.md").read_text()
            assert "42857" in instructions, "Magic number not in installed instructions"

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.skipif(
        not has_anthropic_api_key(),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_full_workflow_install_and_invoke_skill(self, cli_runner, tmp_path, mock_skill_source):
        """E2E test: Install skill via OSPREY, then verify Claude Code reads it.

        This is the KEY test that proves the full workflow works:
        1. OSPREY installs the skill files correctly (using mocked source)
        2. Claude Code finds and reads the installed skill
        3. Claude outputs the magic number, proving it read our instructions

        The magic number 42857 is ONLY in our mock instructions - if Claude
        outputs it, we know the full installation workflow succeeded.
        """
        # Use tmp_path directly (not isolated_filesystem) so the CWD for
        # both the install step and the claude invocation is the same.
        project_dir = tmp_path / "skill-test"
        project_dir.mkdir()

        # Step 1: Install the mock skill using OSPREY's CLI
        original_cwd = Path.cwd()
        try:
            os.chdir(project_dir)
            with (
                patch("osprey.cli.claude_cmd.get_tasks_root") as mock_tasks,
                patch("osprey.cli.claude_cmd.get_integrations_root") as mock_int,
            ):
                mock_tasks.return_value = mock_skill_source["tasks_root"]
                mock_int.return_value = mock_skill_source["integrations_root"]

                result = cli_runner.invoke(install_skill, ["test-verification"])
                assert result.exit_code == 0, f"Install failed: {result.output}"
        finally:
            os.chdir(original_cwd)

        # Verify skill files exist before invoking Claude
        skill_dir = project_dir / ".claude" / "skills" / "test-verification"
        assert (skill_dir / "instructions.md").exists(), (
            f"instructions.md not found at {skill_dir}. Contents: {list(project_dir.rglob('*'))}"
        )

        # Step 2: Invoke Claude Code to read the installed skill
        prompt = (
            "Read the file .claude/skills/test-verification/instructions.md "
            "and follow its Required Output Format EXACTLY. "
            "Do not read any other files. Do not summarize other skills. "
            "Output only what the instructions.md file tells you to output."
        )

        # SAFETY: Permission bypass is safe here because:
        # - Runs in isolated tmp_path (not real codebase)
        # - Controlled prompt (only reads files, outputs text)
        # - Skill's allowed-tools: Read (no writes)
        # See module docstring for full justification.
        #
        # Unset CLAUDECODE to avoid the nested-session guard when
        # this test is run from within a Claude Code session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        claude_result = subprocess.run(
            [
                "claude",
                "--print",
                "--dangerously-skip-permissions",
                "--permission-mode",
                "bypassPermissions",
                "--max-budget-usd",
                "0.50",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_dir),
            env=env,
        )

        assert claude_result.returncode == 0, f"Claude Code failed: {claude_result.stderr}"

        output = claude_result.stdout

        # Debug output
        print("\n  OSPREY Skill Installation E2E Test:")
        print(f"    Output length: {len(output)} chars")
        print(f"    Output: {output[:500]}")

        # Step 3: THE KEY ASSERTION - verify magic number
        # This number (42857) is ONLY in our mock instructions.md
        # If Claude outputs it, we KNOW the full workflow succeeded:
        # - OSPREY installed the files correctly
        # - Claude Code found the installed skill
        # - Claude read our specific instructions
        assert "42857" in output, (
            f"Claude did NOT read the installed skill instructions!\n"
            f"The magic number 42857 was not found in the output.\n"
            f"This means the OSPREY skill installation workflow failed.\n"
            f"Output: {output[:1000]}"
        )

        print("  Full OSPREY skill installation workflow verified!")


class TestClaudeCodeCliAvailability:
    """Smoke tests for Claude Code CLI availability."""

    @pytest.mark.e2e_smoke
    def test_claude_code_cli_version(self, tmp_path):
        """Verify Claude Code CLI is functional."""
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"Claude Code CLI failed: {result.stderr}"
        print(f"\n📦 Claude Code version: {result.stdout.strip()}")
