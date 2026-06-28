"""Tests for osprey.agent_runner.primitives provider-env resolution.

Focus: the auth-var propagation contract that the in-context channel-finder
benchmark depends on. The in_context backend spawns an MCP stdio subprocess
that inherits only a tiny safe-list and then expands config.yml's
``api_key: ${SECRET}`` against its own env — so ``provider_env_for_project``
MUST carry the *raw* secret var (e.g. ``CBORG_API_KEY``), not just the
auth-token var (``ANTHROPIC_AUTH_TOKEN``).

Regression guard for the single-source refactor: the deleted benchmark
``sdk_env`` propagated the raw secret; the unified one initially did not, which
silently broke proxy-provider auth for the in_context backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.agent_runner import sdk_env
from osprey.agent_runner.primitives import provider_env_for_project


def _write_config(project_dir: Path, provider: str) -> None:
    """Write a minimal config.yml selecting *provider* for the claude_code path."""
    (project_dir / "config.yml").write_text(
        f"claude_code:\n  provider: {provider}\n  model: haiku\n"
    )


# ---------------------------------------------------------------------------
# Raw-secret propagation (proxy providers: auth_env_var != auth_secret_env)
# ---------------------------------------------------------------------------


def test_proxy_provider_propagates_both_auth_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cborg: both ANTHROPIC_AUTH_TOKEN (CLI auth) and the raw CBORG_API_KEY
    (MCP-subprocess ${SECRET} expansion) must carry the secret."""
    _write_config(tmp_path, "cborg")
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg-secret")

    env = provider_env_for_project(tmp_path)

    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-cborg-secret"
    assert env["CBORG_API_KEY"] == "sk-cborg-secret", (
        "raw secret var must be propagated so config.yml ${CBORG_API_KEY} "
        "expands inside the MCP stdio subprocess"
    )


def test_provider_override_propagates_raw_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider= override (used by the benchmark sweep) also propagates the
    raw secret for the overridden provider, not the config's."""
    _write_config(tmp_path, "anthropic")
    monkeypatch.setenv("ALS_APG_API_KEY", "sk-als-secret")

    env = provider_env_for_project(tmp_path, provider="als-apg")

    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-als-secret"
    assert env["ALS_APG_API_KEY"] == "sk-als-secret"


# ---------------------------------------------------------------------------
# Direct provider: auth_env_var == auth_secret_env (anthropic)
# ---------------------------------------------------------------------------


def test_direct_provider_sets_single_auth_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """anthropic: the two var names coincide (ANTHROPIC_API_KEY) — still set."""
    _write_config(tmp_path, "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    env = provider_env_for_project(tmp_path)

    assert env["ANTHROPIC_API_KEY"] == "sk-ant-secret"


# ---------------------------------------------------------------------------
# .env override: a project-level .env wins over a stale shell export
# ---------------------------------------------------------------------------


def test_project_dotenv_overrides_shell_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A freshly-configured key in the project .env overrides a stale shell
    export (mirrors inject_provider_env's dotenv precedence)."""
    pytest.importorskip("dotenv")
    _write_config(tmp_path, "cborg")
    monkeypatch.setenv("CBORG_API_KEY", "sk-stale-shell")
    (tmp_path / ".env").write_text("CBORG_API_KEY=sk-fresh-from-dotenv\n")

    env = provider_env_for_project(tmp_path)

    assert env["CBORG_API_KEY"] == "sk-fresh-from-dotenv"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-fresh-from-dotenv"


# ---------------------------------------------------------------------------
# Failure mode: unresolvable provider raises (fail loud, not silent {})
# ---------------------------------------------------------------------------


def test_unresolvable_provider_raises(tmp_path: Path) -> None:
    """No claude_code block → no resolvable provider → RuntimeError, not {}."""
    (tmp_path / "config.yml").write_text("api:\n  providers: {}\n")

    with pytest.raises(RuntimeError, match="no resolvable provider"):
        provider_env_for_project(tmp_path)


# ---------------------------------------------------------------------------
# sdk_env composition
# ---------------------------------------------------------------------------


def test_sdk_env_bypasses_nested_guard_without_project() -> None:
    """sdk_env() with no project returns only the CLAUDECODE bypass."""
    assert sdk_env() == {"CLAUDECODE": ""}


def test_sdk_env_merges_provider_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sdk_env(project) includes CLAUDECODE plus the resolved provider env,
    including the raw secret var."""
    _write_config(tmp_path, "cborg")
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg-secret")

    env = sdk_env(tmp_path)

    assert env["CLAUDECODE"] == ""
    assert env["CBORG_API_KEY"] == "sk-cborg-secret"
    assert env["ANTHROPIC_BASE_URL"]  # provider env block merged in


# ---------------------------------------------------------------------------
# ${VAR} expansion for custom / non-native providers (#307)
# ---------------------------------------------------------------------------

_ARGO_CONFIG = """\
api:
  providers:
    argo:
      base_url: ${ARGO_PROD_URL}
claude_code:
  provider: argo
"""


def test_custom_provider_base_url_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom provider's ${VAR} base_url is expanded (not passed literally)."""
    monkeypatch.delenv("ARGO_PROD_URL", raising=False)
    (tmp_path / "config.yml").write_text(_ARGO_CONFIG)
    (tmp_path / ".env").write_text("ARGO_PROD_URL=https://argo.example/v1\nARGO_API_KEY=sk-argo\n")

    env = provider_env_for_project(tmp_path)

    assert env["ANTHROPIC_BASE_URL"] == "https://argo.example/v1"
    assert "${ARGO_PROD_URL}" not in env["ANTHROPIC_BASE_URL"]


def test_native_provider_env_block_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a literal-URL native config still resolves to the exact
    same env block as the raw resolver — locks the e2e no-op guarantee."""
    from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver

    _write_config(tmp_path, "cborg")
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg-secret")

    env = provider_env_for_project(tmp_path)
    direct = ClaudeCodeModelResolver.resolve({"provider": "cborg", "model": "haiku"}, {})

    for key, value in direct.env_block.items():
        assert env[key] == value
