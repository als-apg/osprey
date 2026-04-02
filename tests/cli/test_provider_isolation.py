"""Tests for per-project provider isolation.

Covers MANAGED_ENV_VARS, auth field passthrough, conflict detection,
and chat command env scrubbing.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.claude_cmd import chat_claude
from osprey.cli.claude_code_resolver import (
    MANAGED_ENV_VARS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
)

# ── MANAGED_ENV_VARS ─────────────────────────────────────────────


class TestManagedEnvVars:
    """MANAGED_ENV_VARS contains the right vars."""

    EXPECTED = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
    }

    def test_contains_expected(self):
        assert MANAGED_ENV_VARS == self.EXPECTED

    def test_excludes_secret_keys(self):
        """Provider-specific secret keys should NOT be in the managed set."""
        assert "CBORG_API_KEY" not in MANAGED_ENV_VARS
        assert "ALS_APG_API_KEY" not in MANAGED_ENV_VARS


# ── Auth field passthrough ───────────────────────────────────────


class TestAuthFieldPassthrough:
    """resolve() passes auth_env_var and auth_secret_env through to spec."""

    def test_cborg(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.auth_env_var == "ANTHROPIC_AUTH_TOKEN"
        assert spec.auth_secret_env == "CBORG_API_KEY"

    def test_anthropic(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.auth_env_var == "ANTHROPIC_API_KEY"
        assert spec.auth_secret_env == "ANTHROPIC_API_KEY"

    def test_als_apg(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"})
        assert spec.auth_env_var == "ANTHROPIC_AUTH_TOKEN"
        assert spec.auth_secret_env == "ALS_APG_API_KEY"

    def test_custom_provider(self):
        """Custom providers derive auth_secret_env from provider name."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "my-lab"},
            api_providers={"my-lab": {"base_url": "https://proxy.example.com"}},
        )
        assert spec.auth_env_var == "ANTHROPIC_AUTH_TOKEN"
        assert spec.auth_secret_env == "MY_LAB_API_KEY"


# ── detect_env_conflicts ─────────────────────────────────────────


class TestDetectEnvConflicts:
    """ClaudeCodeModelSpec.detect_env_conflicts."""

    def test_finds_mismatch(self):
        spec = ClaudeCodeModelSpec(
            provider="test",
            env_block={"ANTHROPIC_BASE_URL": "https://project.example.com"},
        )
        conflicts = spec.detect_env_conflicts({"ANTHROPIC_BASE_URL": "https://shell.example.com"})
        assert "ANTHROPIC_BASE_URL" in conflicts
        assert conflicts["ANTHROPIC_BASE_URL"] == (
            "https://shell.example.com",
            "https://project.example.com",
        )

    def test_ignores_match(self):
        spec = ClaudeCodeModelSpec(
            provider="test",
            env_block={"ANTHROPIC_MODEL": "claude-opus-4-6"},
        )
        conflicts = spec.detect_env_conflicts({"ANTHROPIC_MODEL": "claude-opus-4-6"})
        assert conflicts == {}

    def test_ignores_absent(self):
        spec = ClaudeCodeModelSpec(
            provider="test",
            env_block={"ANTHROPIC_BASE_URL": "https://project.example.com"},
        )
        conflicts = spec.detect_env_conflicts({})
        assert conflicts == {}


# ── Chat command provider isolation ──────────────────────────────


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def cborg_project(tmp_path):
    """Create a minimal project configured for cborg provider."""
    from osprey.cli.templates.manager import TemplateManager

    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="isolation-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"default_provider": "cborg"},
    )
    return project_dir


class TestChatProviderIsolation:
    """chat_claude() wires auth and scrubs managed vars."""

    @pytest.fixture(autouse=True)
    def _no_dotenv(self):
        """Prevent inject_provider_env from loading the scaffolded .env file."""
        with patch("dotenv.dotenv_values", return_value={}):
            yield

    @patch("subprocess.run")
    def test_chat_sets_auth_token(self, mock_run, cli_runner, cborg_project):
        """Auth token is set from secret env var before exec."""
        captured_env = {}

        def capture_env(*_args, **_kwargs):
            captured_env.update(os.environ)
            return type("Result", (), {"returncode": 0})()

        mock_run.side_effect = capture_env

        env = {
            "CBORG_API_KEY": "test-secret-123",
            "PATH": os.environ.get("PATH", ""),
        }
        with patch.dict(os.environ, env, clear=True):
            cli_runner.invoke(chat_claude, ["--project", str(cborg_project)])

        assert captured_env.get("ANTHROPIC_AUTH_TOKEN") == "test-secret-123"

    @patch("subprocess.run")
    def test_chat_scrubs_managed_vars(self, mock_run, cli_runner, cborg_project):
        """Managed vars are scrubbed then re-injected from provider env block."""
        captured_env = {}

        def capture_env(*_args, **_kwargs):
            captured_env.update(os.environ)
            return type("Result", (), {"returncode": 0})()

        mock_run.side_effect = capture_env

        env = {
            "CBORG_API_KEY": "test-secret-123",
            "ANTHROPIC_BASE_URL": "https://stale-shell-value.example.com",
            "ANTHROPIC_MODEL": "stale-model",
            "ANTHROPIC_API_KEY": "should-be-scrubbed",
            "PATH": os.environ.get("PATH", ""),
        }
        with patch.dict(os.environ, env, clear=True):
            cli_runner.invoke(chat_claude, ["--project", str(cborg_project)])

        # ANTHROPIC_AUTH_TOKEN should be set (auth injected from CBORG_API_KEY)
        assert "ANTHROPIC_AUTH_TOKEN" in captured_env
        # ANTHROPIC_BASE_URL should be present with CBORG value (scrubbed then re-injected)
        assert captured_env.get("ANTHROPIC_BASE_URL") == "https://api.cborg.lbl.gov"
        # ANTHROPIC_MODEL should be present with CBORG model ID
        assert "ANTHROPIC_MODEL" in captured_env
        # ANTHROPIC_API_KEY should still be scrubbed (not in cborg env block)
        assert "ANTHROPIC_API_KEY" not in captured_env

    @patch("subprocess.run")
    def test_chat_injects_full_env_block(self, mock_run, cli_runner, cborg_project):
        """All env_block keys are present with correct values after chat setup."""
        captured_env = {}

        def capture_env(*_args, **_kwargs):
            captured_env.update(os.environ)
            return type("Result", (), {"returncode": 0})()

        mock_run.side_effect = capture_env

        env = {
            "CBORG_API_KEY": "test-secret-123",
            "PATH": os.environ.get("PATH", ""),
        }
        with patch.dict(os.environ, env, clear=True):
            cli_runner.invoke(chat_claude, ["--project", str(cborg_project)])

        # All tier model vars should be injected
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" in captured_env
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in captured_env
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" in captured_env
        # Base URL should be the CBORG proxy
        assert captured_env.get("ANTHROPIC_BASE_URL") == "https://api.cborg.lbl.gov"

    @patch("subprocess.run")
    def test_chat_warns_missing_secret(self, mock_run, cli_runner, cborg_project):
        """Warning is shown when secret env var is missing."""
        env = {"PATH": os.environ.get("PATH", "")}
        with patch.dict(os.environ, env, clear=True):
            result = cli_runner.invoke(chat_claude, ["--project", str(cborg_project)])
        assert "CBORG_API_KEY" in result.output
        assert "not found" in result.output.lower()
