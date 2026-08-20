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
    """sdk_env() with no project returns the CLAUDECODE bypass and the
    background-tasks disable flag (so subagent delegation runs synchronously
    and its results are drained in-turn rather than backgrounded)."""
    assert sdk_env() == {
        "CLAUDECODE": "",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    }


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
      models:
        haiku: claudehaiku45
        sonnet: claudesonnet45
        opus: claudeopus41
claude_code:
  provider: argo
"""


def test_custom_provider_base_url_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom provider's ${VAR} base_url is expanded (not passed literally)."""
    monkeypatch.delenv("ARGO_PROD_URL", raising=False)
    (tmp_path / "config.yml").write_text(_ARGO_CONFIG)
    (tmp_path / ".env").write_text("ARGO_PROD_URL=https://argo.example/v1\nARGO_API_KEY=sk-argo\n")

    env = provider_env_for_project(tmp_path)

    # ${VAR} expanded, and /v1 stripped for the Claude-Code-facing var (issue #312).
    assert env["ANTHROPIC_BASE_URL"] == "https://argo.example"
    assert "${ARGO_PROD_URL}" not in env["ANTHROPIC_BASE_URL"]


def test_native_provider_env_block_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a literal-URL native config still resolves to the exact
    same env block as the raw resolver — locks the e2e no-op guarantee."""
    from osprey.build.claude_code_resolver import ClaudeCodeModelResolver

    _write_config(tmp_path, "cborg")
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg-secret")

    env = provider_env_for_project(tmp_path)
    direct = ClaudeCodeModelResolver.resolve({"provider": "cborg", "model": "haiku"}, {})

    for key, value in direct.env_block.items():
        assert env[key] == value


# ---------------------------------------------------------------------------
# OSPREY_E2E_FORCE_MODEL — derived force-key set (#350 / #357)
# ---------------------------------------------------------------------------


def test_e2e_force_derives_forced_keys_from_single_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_apply_e2e_overrides`` forces exactly ``{ANTHROPIC_MODEL} ∪
    TIER_MODEL_ENV_VARS.values()`` (derived, not a literal tuple) and stays
    consistent with the resolved env_block.

    Guards the #350 drift: a model var present in env_block but absent from the
    force list left the matrix sending the wrong model on background calls.
    """
    from osprey.agent_runner.primitives import _apply_e2e_overrides
    from osprey.build.claude_code_resolver import (
        TIER_MODEL_ENV_VARS,
        ClaudeCodeModelResolver,
    )

    monkeypatch.setenv("OSPREY_E2E_FORCE_MODEL", "forced-model-x")
    spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})

    forced = _apply_e2e_overrides(spec)

    derived = {"ANTHROPIC_MODEL"} | set(TIER_MODEL_ENV_VARS.values())
    # Every derived key present in env_block is rewritten to the forced model.
    for key in derived:
        if key in forced.env_block:
            assert forced.env_block[key] == "forced-model-x"
    # The forced-key set is a subset of what resolve() can produce — no key is
    # forced that env_block could never carry (the drift guard).
    assert derived <= set(forced.env_block) | {"ANTHROPIC_MODEL"}
    # Every tier collapses onto the single forced model.
    assert set(forced.tier_to_model.values()) == {"forced-model-x"}


def test_e2e_force_inert_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither override env var set, the spec is returned unchanged."""
    from osprey.agent_runner.primitives import _apply_e2e_overrides
    from osprey.build.claude_code_resolver import ClaudeCodeModelResolver

    monkeypatch.delenv("OSPREY_E2E_FORCE_MODEL", raising=False)
    monkeypatch.delenv("OSPREY_E2E_PROXY_BASE_URL", raising=False)
    spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})

    assert _apply_e2e_overrides(spec) is spec


# ---------------------------------------------------------------------------
# Telemetry credentials a deploy has not issued yet
# ---------------------------------------------------------------------------


def _telemetry_config(project_dir: Path, password: str) -> None:
    """Write a config whose telemetry block names ``password`` as its secret.

    Shaped like the block every bundled preset ships: an OpenObserve backend
    with no explicit endpoint (it derives one) and a user that already has a
    value, so the password is the only credential a test is varying.
    """
    (project_dir / "config.yml").write_text(
        "claude_code:\n"
        "  provider: anthropic\n"
        "  telemetry:\n"
        "    enabled: true\n"
        "    backend: openobserve\n"
        "    openobserve:\n"
        "      user: ingest@example.com\n"
        f"      password: {password}\n"
        "      org: default\n",
        encoding="utf-8",
    )


class TestTelemetryCredentialNotIssuedYet:
    """Spawning an agent in a project that has never run ``osprey up``.

    The shipped telemetry block names ``${ZO_INGEST_SA_TOKEN}`` with no
    fallback, and the store mints that token into the repo's ``.env`` only when
    a deploy starts it. Resolving the provider resolves telemetry too, so
    without a carve-out every agent spawned against a built-but-undeployed
    project dies on a value the operator has no way to supply.

    Same rule as ``osprey chat``: as wide as the store-issued registry and no
    wider. Every other unresolved credential is an ordinary missing secret and
    keeps refusing.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_store_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A developer machine that happens to export the token would hide the bug."""
        monkeypatch.delenv("ZO_INGEST_SA_TOKEN", raising=False)
        monkeypatch.delenv("OPERATOR_OTLP_SECRET", raising=False)

    def test_the_token_under_test_is_really_store_issued(self) -> None:
        """Guards every assertion below from passing for the wrong reason."""
        from osprey.deployment.container_lifecycle import _STORE_ISSUED_VARS

        assert "ZO_INGEST_SA_TOKEN" in _STORE_ISSUED_VARS
        assert "OPERATOR_OTLP_SECRET" not in _STORE_ISSUED_VARS

    def test_an_unissued_store_credential_resolves_without_telemetry(self, tmp_path: Path) -> None:
        _telemetry_config(tmp_path, "${ZO_INGEST_SA_TOKEN}")

        env = provider_env_for_project(tmp_path)

        # Degraded, not deferred: an exporter without its auth header would post
        # to an auth-gated store and drop every span, so the whole block goes.
        assert "CLAUDE_CODE_ENABLE_TELEMETRY" not in env
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in env
        # The provider half of the same read is untouched — the run still routes.
        assert env["ANTHROPIC_MODEL"]

    def test_the_caller_is_told_which_verb_issues_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silence would read as "this project has no telemetry configured"."""
        _telemetry_config(tmp_path, "${ZO_INGEST_SA_TOKEN}")

        with caplog.at_level("WARNING", logger="osprey.agent_runner.primitives"):
            provider_env_for_project(tmp_path)

        assert "ZO_INGEST_SA_TOKEN" in caplog.text
        assert "osprey up" in caplog.text

    def test_an_operator_supplied_credential_still_raises(self, tmp_path: Path) -> None:
        """The carve-out reads the registry, not "unresolved" — this one is a
        real missing secret, and a run started without it would hide a broken
        pipeline behind a warning."""
        from osprey.build.claude_code_telemetry import ObservabilityCredentialError

        _telemetry_config(tmp_path, "${OPERATOR_OTLP_SECRET}")

        with pytest.raises(ObservabilityCredentialError):
            provider_env_for_project(tmp_path)

    def test_a_mixed_refusal_is_not_deferred(self, tmp_path: Path) -> None:
        """One name in the set the operator does have to supply and the whole set
        stands refused."""
        from osprey.build.claude_code_telemetry import ObservabilityCredentialError

        _telemetry_config(tmp_path, "${OPERATOR_OTLP_SECRET}${ZO_INGEST_SA_TOKEN}")

        with pytest.raises(ObservabilityCredentialError) as caught:
            provider_env_for_project(tmp_path)

        assert caught.value.unresolved_vars == ("OPERATOR_OTLP_SECRET", "ZO_INGEST_SA_TOKEN")

    def test_a_blank_credential_still_raises(self, tmp_path: Path) -> None:
        """No variable is named at all, so there is nothing a deploy could issue."""
        from osprey.build.claude_code_telemetry import ObservabilityCredentialError

        _telemetry_config(tmp_path, '""')

        with pytest.raises(ObservabilityCredentialError):
            provider_env_for_project(tmp_path)

    def test_a_resolvable_token_keeps_telemetry_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deferral is about absence only — once a deploy has written the
        token, the same config resolves and the run exports normally."""
        _telemetry_config(tmp_path, "${ZO_INGEST_SA_TOKEN}")
        monkeypatch.setenv("ZO_INGEST_SA_TOKEN", "issued-by-the-store")

        env = provider_env_for_project(tmp_path)

        assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
        assert "Authorization=Basic " in env["OTEL_EXPORTER_OTLP_HEADERS"]
