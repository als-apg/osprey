"""A completed personal ``/login`` outranks the deployment provider when the deployment allows it (#925).

Claude Code's credential precedence puts a subscription login last, so the
only way a terminal reaches it is for every provider variable to be absent.
``withhold_provider_env`` is the scrub half of ``inject_provider_env`` with
nothing injected after it; ``personal_login_present`` is the detection, by
the credential file's existence in the terminal's own ``CLAUDE_CONFIG_DIR``,
never by its contents.
"""

from __future__ import annotations

from pathlib import Path

from osprey.agent_runner.provider_env import (
    LOGIN_CREDENTIAL_FILE,
    MANAGED_ENV_VARS,
    PERSONAL_LOGIN_KEY,
    personal_login_allowed,
    personal_login_present,
    withhold_provider_env,
)


class TestAllowed:
    def test_the_key_is_off_by_default(self):
        assert personal_login_allowed({}) is False
        assert personal_login_allowed(None) is False
        assert personal_login_allowed({"claude_code": {"provider": "anthropic"}}) is False

    def test_only_a_true_boolean_turns_it_on(self):
        assert personal_login_allowed({"claude_code": {"allow_personal_login": True}}) is True
        assert personal_login_allowed({"claude_code": {"allow_personal_login": "true"}}) is False
        assert personal_login_allowed({"claude_code": {"allow_personal_login": 1}}) is False

    def test_the_key_is_the_documented_one(self):
        assert PERSONAL_LOGIN_KEY == "claude_code.allow_personal_login"


class TestPresent:
    def test_a_credential_file_in_the_config_dir_is_a_login(self, tmp_path: Path):
        (tmp_path / LOGIN_CREDENTIAL_FILE).write_text("{}", encoding="utf-8")
        assert personal_login_present({"CLAUDE_CONFIG_DIR": str(tmp_path)}) is True

    def test_no_file_is_no_login(self, tmp_path: Path):
        assert personal_login_present({"CLAUDE_CONFIG_DIR": str(tmp_path)}) is False

    def test_no_config_dir_is_no_login(self, tmp_path: Path, monkeypatch):
        """Without a per-user config dir there is no login the deployment could honour."""
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / LOGIN_CREDENTIAL_FILE).write_text("{}", encoding="utf-8")
        assert personal_login_present({}) is False
        assert personal_login_present({"CLAUDE_CONFIG_DIR": "  "}) is False

    def test_the_credential_is_never_read(self, tmp_path: Path):
        """A file this process cannot read still counts: existence is the whole signal."""
        secret = tmp_path / LOGIN_CREDENTIAL_FILE
        secret.write_text("not json at all", encoding="utf-8")
        secret.chmod(0o000)
        try:
            assert personal_login_present({"CLAUDE_CONFIG_DIR": str(tmp_path)}) is True
        finally:
            secret.chmod(0o600)


class TestWithhold:
    def test_every_managed_variable_is_gone_and_nothing_is_injected(self):
        env = {
            "ANTHROPIC_AUTH_TOKEN": "deployment",
            "ANTHROPIC_API_KEY": "deployment",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_MODEL": "m",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "HOME": "/data/claude-config",
            "CLAUDE_CONFIG_DIR": "/data/claude-config",
        }
        withheld = withhold_provider_env(env)
        assert not (set(env) & MANAGED_ENV_VARS)
        assert env["HOME"] == "/data/claude-config"
        assert env["CLAUDE_CONFIG_DIR"] == "/data/claude-config"
        assert withheld == [
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_MODEL",
            "CLAUDE_CODE_USE_BEDROCK",
        ]

    def test_the_project_env_chain_still_reaches_the_agent(self, tmp_path: Path):
        """MCP addressing from .env is unrelated to who pays for the model."""
        (tmp_path / ".env").write_text(
            "EPICS_CA_ADDR_LIST=10.0.0.1\nANTHROPIC_API_KEY=from-dotenv\n", encoding="utf-8"
        )
        env: dict[str, str] = {}
        withhold_provider_env(env, tmp_path)
        assert env["EPICS_CA_ADDR_LIST"] == "10.0.0.1"
        assert "ANTHROPIC_API_KEY" not in env, "a key in .env is scrubbed like a shell export"
