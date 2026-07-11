"""Tests for per-project provider isolation.

Covers MANAGED_ENV_VARS, auth field passthrough, conflict detection,
and chat command env scrubbing.
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.claude_cmd import chat_claude
from osprey.cli.claude_code_resolver import (
    MANAGED_ENV_VARS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
    detect_managed_policy_conflicts,
    inject_provider_env,
)

# ── MANAGED_ENV_VARS ─────────────────────────────────────────────


class TestManagedEnvVars:
    """MANAGED_ENV_VARS contains the right vars."""

    EXPECTED = {
        # Auth + endpoint
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        # Model selectors
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        # Backend selectors
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_MANTLE",
        # Backend endpoint / auth overrides
        "ANTHROPIC_BEDROCK_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH",
        "CLAUDE_CODE_SKIP_FOUNDRY_AUTH",
    }

    def test_contains_expected(self):
        assert MANAGED_ENV_VARS == self.EXPECTED

    def test_excludes_secret_keys(self):
        """Provider-specific secret keys should NOT be in the managed set."""
        assert "CBORG_API_KEY" not in MANAGED_ENV_VARS
        assert "ALS_APG_API_KEY" not in MANAGED_ENV_VARS

    def test_excludes_shared_cloud_sdk_vars(self):
        """Cloud-SDK vars are shared with non-Claude tooling in the same shell.

        They only affect Claude Code when a CLAUDE_CODE_USE_* backend flag is
        set, and that flag is scrubbed. Clearing them would break boto3, gcloud,
        and anything else the operator runs alongside the agent.
        """
        for var in ("AWS_REGION", "GCLOUD_PROJECT", "CLOUD_ML_REGION"):
            assert var not in MANAGED_ENV_VARS

    def test_excludes_custom_headers(self):
        """ANTHROPIC_CUSTOM_HEADERS is additive, not a selector.

        Headers cannot redirect the endpoint, so a stale value fails loudly
        (401 against the configured provider) rather than silently rerouting.
        It is also the only way to add corporate-proxy headers, since env_block
        has no key for them — so OSPREY passes it through.
        """
        assert "ANTHROPIC_CUSTOM_HEADERS" not in MANAGED_ENV_VARS


# ── Backend / model selector scrubbing (#356) ────────────────────


class TestBackendSelectorScrubbing:
    """A stale backend selector must not survive into the agent's environment.

    Regression guard for #356: a leftover ``export CLAUDE_CODE_USE_BEDROCK=1``
    in an operator's shell profile silently routed the whole agent — main model
    and subagents — to a different backend than the project configured.
    """

    BACKEND_SELECTORS = [
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_MANTLE",
    ]

    MODEL_SELECTORS = [
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
    ]

    BACKEND_OVERRIDES = [
        "ANTHROPIC_BEDROCK_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH",
        "CLAUDE_CODE_SKIP_FOUNDRY_AUTH",
    ]

    @pytest.mark.parametrize("var", BACKEND_SELECTORS + MODEL_SELECTORS + BACKEND_OVERRIDES)
    def test_stale_selector_is_scrubbed(self, var):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        environ = {"ANTHROPIC_API_KEY": "secret-123", var: "1"}

        inject_provider_env(environ, spec)

        assert var not in environ

    def test_all_selectors_scrubbed_together(self):
        """The realistic case: a shell profile that configured Bedrock wholesale."""
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        environ = {
            "ANTHROPIC_API_KEY": "secret-123",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_SKIP_BEDROCK_AUTH": "1",
            "ANTHROPIC_BEDROCK_BASE_URL": "https://gateway.internal/bedrock",
            "CLAUDE_CODE_SUBAGENT_MODEL": "some-other-model",
            "AWS_REGION": "us-west-2",
        }

        inject_provider_env(environ, spec)

        assert "CLAUDE_CODE_USE_BEDROCK" not in environ
        assert "CLAUDE_CODE_SKIP_BEDROCK_AUTH" not in environ
        assert "ANTHROPIC_BEDROCK_BASE_URL" not in environ
        assert "CLAUDE_CODE_SUBAGENT_MODEL" not in environ
        # Project auth survives, re-injected after the scrub:
        assert environ["ANTHROPIC_API_KEY"] == "secret-123"
        # Shared cloud-SDK var is left alone for other tooling:
        assert environ["AWS_REGION"] == "us-west-2"

    def test_custom_headers_survive(self):
        """Corporate-proxy headers are passed through, not scrubbed."""
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        environ = {
            "ANTHROPIC_API_KEY": "secret-123",
            "ANTHROPIC_CUSTOM_HEADERS": "X-Corp-Trace: abc123",
        }

        inject_provider_env(environ, spec)

        assert environ["ANTHROPIC_CUSTOM_HEADERS"] == "X-Corp-Trace: abc123"


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


# ── Managed-policy conflict detection (#355) ─────────────────────


class TestManagedPolicyConflicts:
    """detect_managed_policy_conflicts scans the enterprise policy scope.

    Managed policy outranks the process environment and the
    ``--setting-sources project`` restriction, so a policy ``env`` block setting
    a provider variable silently redirects the agent. The CLI refuses to launch
    on a non-empty result.
    """

    def _write(self, path, obj):
        path.write_text(json.dumps(obj))

    def test_flags_managed_var_in_policy_env(self, tmp_path):
        policy = tmp_path / "managed-settings.json"
        self._write(policy, {"env": {"ANTHROPIC_BASE_URL": "https://evil.example"}})

        conflicts = detect_managed_policy_conflicts([policy])

        assert conflicts["ANTHROPIC_BASE_URL"] == (
            "https://evil.example",
            str(policy),
        )

    def test_ignores_unmanaged_var(self, tmp_path):
        policy = tmp_path / "managed-settings.json"
        self._write(policy, {"env": {"CLAUDE_CODE_ENABLE_TELEMETRY": "1"}})

        assert detect_managed_policy_conflicts([policy]) == {}

    def test_empty_when_no_file(self, tmp_path):
        assert detect_managed_policy_conflicts([tmp_path / "absent.json"]) == {}

    def test_empty_when_no_env_block(self, tmp_path):
        policy = tmp_path / "managed-settings.json"
        self._write(policy, {"permissions": {"allow": []}})

        assert detect_managed_policy_conflicts([policy]) == {}

    def test_malformed_json_is_skipped(self, tmp_path):
        policy = tmp_path / "managed-settings.json"
        policy.write_text("{ not valid json")

        assert detect_managed_policy_conflicts([policy]) == {}

    def test_dropin_fragment_overrides_main_source(self, tmp_path):
        """A later fragment wins, and its path is the one reported."""
        main = tmp_path / "managed-settings.json"
        fragment = tmp_path / "20-provider.json"
        self._write(main, {"env": {"ANTHROPIC_MODEL": "from-main"}})
        self._write(fragment, {"env": {"ANTHROPIC_MODEL": "from-fragment"}})

        conflicts = detect_managed_policy_conflicts([main, fragment])

        assert conflicts["ANTHROPIC_MODEL"] == ("from-fragment", str(fragment))


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
        context={"default_provider": "cborg", "channel_finder_mode": "hierarchical"},
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

    @patch("subprocess.run")
    def test_chat_launches_with_setting_sources_project(self, mock_run, cli_runner, cborg_project):
        """The launched argv restricts settings to project scope, so a user's
        global settings.json cannot override the injected provider env (#355)."""
        captured_argv = []

        def capture_argv(argv, *_args, **_kwargs):
            captured_argv.extend(argv)
            return type("Result", (), {"returncode": 0})()

        mock_run.side_effect = capture_argv

        env = {"CBORG_API_KEY": "test-secret-123", "PATH": os.environ.get("PATH", "")}
        with patch.dict(os.environ, env, clear=True):
            cli_runner.invoke(chat_claude, ["--project", str(cborg_project)])

        assert "--setting-sources" in captured_argv
        i = captured_argv.index("--setting-sources")
        assert captured_argv[i + 1] == "project"

    @patch("osprey.cli.claude_code_resolver.detect_managed_policy_conflicts")
    @patch("subprocess.run")
    def test_chat_refuses_on_managed_policy_conflict(
        self, mock_run, mock_detect, cli_runner, cborg_project
    ):
        """A managed-policy env block overriding a provider var aborts launch —
        managed policy outranks even --setting-sources, so the framework refuses
        rather than start against the wrong backend."""
        mock_detect.return_value = {
            "ANTHROPIC_BASE_URL": ("https://evil.example", "/etc/.../managed-settings.json")
        }

        env = {"CBORG_API_KEY": "test-secret-123", "PATH": os.environ.get("PATH", "")}
        with patch.dict(os.environ, env, clear=True):
            result = cli_runner.invoke(chat_claude, ["--project", str(cborg_project)])

        assert result.exit_code == 1
        assert "Refusing to launch" in result.output
        assert "ANTHROPIC_BASE_URL" in result.output
        mock_run.assert_not_called()


# ── Proxy env var warning ────────────────────────────────────────


class TestProxyEnvWarning:
    """inject_provider_env warns when a .env-sourced proxy var looks like a placeholder.

    Background: ``inject_provider_env`` loads every key from the project's ``.env``
    into the process environment before launching Claude Code.  If a user copies
    ``env.example`` to ``.env`` without editing it, placeholder values like
    ``HTTP_PROXY=http-proxy`` land in the environment verbatim.  Claude Code then
    refuses to start with "Invalid proxy URL in HTTP_PROXY".

    The warning fires at the inject site — before any launch path is taken —
    so the cli, web terminal, and dispatch worker all surface the same message.
    The value is intentionally left in the environment: Claude Code's own error
    is precise and actionable, and blanking the variable would hide the problem
    from other tools (httpx, requests, DuckDB) that also read proxy env vars.
    """

    @pytest.mark.parametrize("var", ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"])
    def test_warns_on_placeholder_proxy_value(self, tmp_path, var, caplog):
        # Simulate a .env file that has a placeholder proxy value (e.g. from cp env.example .env)
        env_file = tmp_path / ".env"
        env_file.write_text(f"{var}=http-proxy\n")

        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        environ = {"ANTHROPIC_API_KEY": "secret-123"}

        # Patch dotenv_values so we don't need a real .env file on disk
        with patch("dotenv.dotenv_values", return_value={var: "http-proxy"}):
            with caplog.at_level(logging.WARNING, logger="osprey.inject_provider_env"):
                inject_provider_env(environ, spec, project_dir=tmp_path)

        # A warning naming the offending variable should have been emitted
        assert any(var in r.message for r in caplog.records), (
            f"Expected a warning mentioning {var!r} but got: {[r.message for r in caplog.records]}"
        )

    def test_valid_proxy_value_no_warning(self, tmp_path, caplog):
        # A properly formatted proxy URL should not trigger any warning
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        environ = {"ANTHROPIC_API_KEY": "secret-123"}

        with patch(
            "dotenv.dotenv_values",
            return_value={"HTTP_PROXY": "http://proxy.example.com:8080"},
        ):
            with caplog.at_level(logging.WARNING, logger="osprey.inject_provider_env"):
                inject_provider_env(environ, spec, project_dir=tmp_path)

        assert caplog.records == [], f"Unexpected warnings: {[r.message for r in caplog.records]}"

    def test_value_passes_through_unchanged(self, tmp_path, caplog):
        # The bad value must survive into the environment unchanged —
        # blanking it would hide the problem from httpx/requests/DuckDB
        # and turn a good Claude Code diagnostic into a mystery failure.
        (tmp_path / ".env").write_text("HTTP_PROXY=http-proxy\n")
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        environ = {"ANTHROPIC_API_KEY": "secret-123"}

        with patch(
            "dotenv.dotenv_values",
            return_value={"HTTP_PROXY": "http-proxy"},
        ):
            with caplog.at_level(logging.WARNING, logger="osprey.inject_provider_env"):
                inject_provider_env(environ, spec, project_dir=tmp_path)

        assert environ.get("HTTP_PROXY") == "http-proxy"
