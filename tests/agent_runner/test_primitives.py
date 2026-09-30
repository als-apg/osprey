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

import os
import time
from pathlib import Path
from typing import Any

import pytest

from osprey.agent_runner import primitives, sdk_env
from osprey.agent_runner.primitives import provider_env_for_project


def _write_config(project_dir: Path, provider: str) -> None:
    """Write a minimal config.yml selecting *provider* for the claude_code path."""
    (project_dir / "config.yml").write_text(
        f"claude_code:\n  provider: {provider}\n  model: claude-haiku-4-5\n"
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
    # als-apg ships no endpoint of its own, so a deployment names one; here the
    # break-glass variable stands in for the deployment's providers.yml entry.
    monkeypatch.setenv("ALS_APG_BASE_URL", "https://gw.test/v1")

    env = provider_env_for_project(tmp_path, provider="als-apg")

    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-als-secret"
    assert env["ALS_APG_API_KEY"] == "sk-als-secret"


# ---------------------------------------------------------------------------
# Raw-endpoint propagation (gateway providers that ship no default base_url)
# ---------------------------------------------------------------------------


def test_gateway_provider_propagates_raw_base_url_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw endpoint var travels beside the raw secret.

    ``ANTHROPIC_BASE_URL`` is the CLI's variable and carries the endpoint with
    its ``/v1`` stripped, so it cannot stand in for the gateway's own. The MCP
    subprocess expands ``base_url: ${ALS_APG_BASE_URL}`` from config.yml against
    its own environment; without the variable the provider refuses the call for
    a missing endpoint.
    """
    _write_config(tmp_path, "als-apg")
    monkeypatch.setenv("ALS_APG_API_KEY", "sk-als-secret")
    monkeypatch.setenv("ALS_APG_BASE_URL", "https://gw.test/v1")

    env = provider_env_for_project(tmp_path)

    assert env["ALS_APG_BASE_URL"] == "https://gw.test/v1"


def test_in_context_backend_env_carries_the_gateway_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The benchmark's direct backend hands that variable to the subprocess.

    ``mcp.client.stdio`` replaces the parent environment with
    ``{**get_default_environment(), **env}`` — a six-name safe-list plus what
    the caller passes — so a variable absent from this dict is absent from the
    server, whatever the shell exported.
    """
    from osprey.services.channel_finder.benchmarks.backends.in_context_backend import (
        InContextBackend,
    )

    _write_config(tmp_path, "als-apg")
    monkeypatch.setenv("ALS_APG_API_KEY", "sk-als-secret")
    monkeypatch.setenv("ALS_APG_BASE_URL", "https://gw.test/v1")

    backend = InContextBackend(tmp_path, "als-apg/claude-haiku-4-5-20251001")

    assert backend._env["ALS_APG_BASE_URL"] == "https://gw.test/v1"
    assert backend._env["ALS_APG_API_KEY"] == "sk-als-secret"


# ---------------------------------------------------------------------------
# Custom request headers
# ---------------------------------------------------------------------------


def test_litellm_gateway_merges_attribution_into_the_operators_custom_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A LiteLLM gateway adds its attribution beside the operator's own headers."""
    _write_config(tmp_path, "cborg")
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg-secret")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Corp-Trace: abc123")

    env = provider_env_for_project(tmp_path)

    lines = env["ANTHROPIC_CUSTOM_HEADERS"].splitlines()
    assert lines[0] == "X-Corp-Trace: abc123"
    assert any(line.startswith("x-litellm-end-user-id: ") for line in lines)
    assert any(line.startswith("x-litellm-tags: ") for line in lines)


def test_a_direct_provider_carries_the_operators_custom_headers_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct provider gets the operator's headers and no attribution."""
    _write_config(tmp_path, "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Corp-Trace: abc123")

    env = provider_env_for_project(tmp_path)

    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Corp-Trace: abc123"


def test_no_custom_headers_without_an_operator_value_or_a_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither an operator value nor a gateway means the variable is not set."""
    _write_config(tmp_path, "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)

    env = provider_env_for_project(tmp_path)

    assert "ANTHROPIC_CUSTOM_HEADERS" not in env


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


def test_the_no_provider_error_names_the_key_the_operator_sets(tmp_path: Path) -> None:
    """The remedy has to be one an operator can carry out.

    This branch fires when the key is unset, so it names that key and the file
    it is set in — not a five-name provider list (an *unknown* name gets the
    registry-derived union elsewhere) and not a test-only helper.
    """
    (tmp_path / "config.yml").write_text("api:\n  providers: {}\n")

    with pytest.raises(RuntimeError) as excinfo:
        provider_env_for_project(tmp_path)

    message = str(excinfo.value)
    assert "claude_code.provider" in message
    assert "profile.yml" in message
    assert "osprey build" in message
    assert "init_project" not in message


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
      default_model: claudesonnet45
      models: [claudehaiku45, claudesonnet45, claudeopus41]
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
    from osprey.agent_runner.provider_env import ClaudeCodeModelResolver

    _write_config(tmp_path, "cborg")
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg-secret")

    env = provider_env_for_project(tmp_path)
    direct = ClaudeCodeModelResolver.resolve({"provider": "cborg", "model": "claude-haiku-4-5"}, {})

    for key, value in direct.env_block.items():
        assert env[key] == value


# ---------------------------------------------------------------------------
# OSPREY_E2E_FORCE_MODEL — derived force-key set (#350 / #357)
# ---------------------------------------------------------------------------


def test_e2e_force_derives_forced_keys_from_single_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_apply_e2e_overrides`` forces ``{ANTHROPIC_MODEL} ∪
    TIER_MODEL_ENV_VARS.values()`` (derived, not a literal tuple) plus
    ``CLAUDE_CODE_SUBAGENT_MODEL``, and stays consistent with the resolved
    env_block.

    A model var present in env_block but absent from the force list would send
    the wrong model on background calls; an agent's frontmatter names its id
    directly, so the subagent var is what redirects it.
    """
    from osprey.agent_runner.primitives import _apply_e2e_overrides
    from osprey.agent_runner.provider_env import (
        TIER_MODEL_ENV_VARS,
        ClaudeCodeModelResolver,
    )

    monkeypatch.setenv("OSPREY_E2E_FORCE_MODEL", "forced-model-x")
    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "cborg", "agent_models": {"channel-finder": "claude-sonnet-5"}}
    )

    forced = _apply_e2e_overrides(spec)

    derived = {"ANTHROPIC_MODEL"} | set(TIER_MODEL_ENV_VARS.values())
    # Every derived key present in env_block is rewritten to the forced model.
    for key in derived:
        if key in forced.env_block:
            assert forced.env_block[key] == "forced-model-x"
    # The forced-key set is a subset of what resolve() can produce — no key is
    # forced that env_block could never carry (the drift guard).
    assert derived <= set(forced.env_block) | {"ANTHROPIC_MODEL"}
    # The main model, every alias and every agent collapse onto the forced model.
    assert forced.default_model_id == "forced-model-x"
    assert set(forced.alias_models.values()) == {"forced-model-x"}
    assert forced.agent_model("channel-finder") == "forced-model-x"
    assert forced.env_block["CLAUDE_CODE_SUBAGENT_MODEL"] == "forced-model-x"


def test_e2e_force_inert_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither override env var set, the spec is returned unchanged."""
    from osprey.agent_runner.primitives import _apply_e2e_overrides
    from osprey.agent_runner.provider_env import ClaudeCodeModelResolver

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


class TestTheMcpReadinessBudgetIsNamedForWhatItGates:
    """``OSPREY_MCP_READY_TIMEOUT``: the readiness barrier's ceiling.

    It gates production ``osprey query`` (exit 1 when a declared server never
    registers), the interactive session, and the dispatch worker — not only
    E2E, which is what the old ``OSPREY_E2E_`` spelling implied.

    The reader takes the environment as an argument, so these cases hand it a
    dict rather than reloading the module: a reload would rebind the module's
    classes while every importer keeps the originals.
    """

    def test_the_default_applies_with_neither_name_set(self) -> None:
        assert primitives._mcp_ready_timeout_from_env({}) == 90.0

    def test_the_new_name_is_read(self) -> None:
        assert primitives._mcp_ready_timeout_from_env({"OSPREY_MCP_READY_TIMEOUT": "12"}) == 12.0

    def test_the_old_name_still_works_for_one_release(self) -> None:
        """A host that already sets the E2E spelling keeps its value."""
        assert primitives._mcp_ready_timeout_from_env({"OSPREY_E2E_MCP_READY_TIMEOUT": "7"}) == 7.0

    def test_the_new_name_wins_when_both_are_set(self) -> None:
        assert (
            primitives._mcp_ready_timeout_from_env(
                {
                    "OSPREY_MCP_READY_TIMEOUT": "12",
                    "OSPREY_E2E_MCP_READY_TIMEOUT": "7",
                }
            )
            == 12.0
        )

    def test_the_module_constant_comes_from_this_process_environment(self) -> None:
        """The barrier's own default is what the reader answers for this host."""
        assert primitives.MCP_READY_TIMEOUT_S == primitives._mcp_ready_timeout_from_env(os.environ)
        assert primitives._MCP_READY_TIMEOUT_S == primitives.MCP_READY_TIMEOUT_S


class _ScriptedMcpStatusClient:
    """A stand-in for ``ClaudeSDKClient`` that answers ``get_mcp_status()`` with
    a fixed snapshot and counts how often the barrier asked."""

    def __init__(self, servers: list[dict[str, Any]]) -> None:
        self._servers = servers
        self.calls = 0

    async def get_mcp_status(self) -> dict[str, Any]:
        self.calls += 1
        return {"mcpServers": self._servers}


def _server(name: str, status: str) -> dict[str, Any]:
    return {"name": name, "status": status, "tools": [], "error": None}


class TestTheReadinessBarrierStopsOnEveryTerminalMcpStatus:
    """A status the CLI will not revise without a reconnect ends the wait.

    ``event_dispatcher`` is reachable only through a web terminal's panel proxy,
    the one in-container holder of the dispatcher bearer. A render that carries
    the entry but runs where the proxy is not — ``osprey chat``, a dispatch
    worker — gets ``needs-auth`` (nothing supplies the bearer) or ``failed``
    (nothing listening). Both are final, so such a run must reach its first turn
    immediately instead of spending the whole readiness budget on a server that
    was never going to register.
    """

    async def test_a_needs_auth_snapshot_returns_before_the_deadline(self) -> None:
        """``needs-auth`` is the SDK's own literal for a server that answered but
        would not admit us; no amount of waiting supplies the credential."""
        client = _ScriptedMcpStatusClient([_server("event_dispatcher", "needs-auth")])

        started = time.monotonic()
        servers = await primitives.await_mcp_ready(
            client, {"event_dispatcher"}, timeout_s=5.0, poll_s=0.01
        )

        assert time.monotonic() - started < 1.0
        assert client.calls == 1
        assert [s["status"] for s in servers] == ["needs-auth"]

    async def test_a_failed_snapshot_returns_before_the_deadline(self) -> None:
        """The other terminal status, pinned beside it so the pair cannot drift."""
        client = _ScriptedMcpStatusClient([_server("event_dispatcher", "failed")])

        started = time.monotonic()
        servers = await primitives.await_mcp_ready(
            client, {"event_dispatcher"}, timeout_s=5.0, poll_s=0.01
        )

        assert time.monotonic() - started < 1.0
        assert client.calls == 1
        assert [s["status"] for s in servers] == ["failed"]

    async def test_a_pending_snapshot_still_waits(self) -> None:
        """``pending`` is a server still starting: the barrier exists for it, so
        it keeps polling to the deadline and then hands back what it last saw."""
        client = _ScriptedMcpStatusClient([_server("event_dispatcher", "pending")])

        started = time.monotonic()
        servers = await primitives.await_mcp_ready(
            client, {"event_dispatcher"}, timeout_s=0.05, poll_s=0.01
        )

        assert time.monotonic() - started >= 0.05
        assert client.calls > 1
        assert [s["status"] for s in servers] == ["pending"]

    def test_the_terminal_statuses_are_the_three_final_ones(self) -> None:
        assert primitives._MCP_TERMINAL_STATUSES == frozenset({"connected", "failed", "needs-auth"})


# ---------------------------------------------------------------------------
# build_agent_options: every option a caller sets reaches the agent options
# ---------------------------------------------------------------------------


class _RoutedSpec:
    """The slice of a resolved provider spec the options builder reads."""

    def __init__(self, *, needs_proxy: bool = False, provider: str = "anthropic") -> None:
        self.needs_proxy = needs_proxy
        self.auth_env_var = "ANTHROPIC_AUTH_TOKEN"
        self.upstream_base_url = "https://gateway.example/v1" if needs_proxy else None
        self.provider = provider
        self.supports_images = None
        self.default_model_id = f"{provider}-main"


async def _hook(
    _input: Any, _tool_use_id: str | None, _context: Any
) -> dict[str, Any]:  # pragma: no cover - never invoked
    return {}


async def _can_use_tool(
    _name: str, _input: dict[str, Any], _context: Any
) -> Any:  # pragma: no cover - never invoked
    return None


def _stderr_sink(_line: str) -> None:  # pragma: no cover - never invoked
    return None


class TestBuildAgentOptions:
    _ROUTED_ENV = {"CLAUDECODE": "", "ANTHROPIC_BASE_URL": "https://api.example"}

    @pytest.fixture()
    def routing(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Stub the project-routing lookups; each stub records its calls."""
        from unittest.mock import MagicMock

        stubs = {
            "sdk_env": MagicMock(side_effect=lambda *_a, **_k: dict(self._ROUTED_ENV)),
            "resolve_default_model": MagicMock(return_value="main-model"),
            "_resolve_project_spec": MagicMock(return_value=_RoutedSpec()),
            "start_proxy": MagicMock(return_value=8123),
        }
        for name, stub in stubs.items():
            monkeypatch.setattr(primitives, name, stub)
        return stubs

    @pytest.mark.usefixtures("routing")
    def test_defaults_build_exactly_the_options_osprey_query_gets(self, tmp_path: Path) -> None:
        from claude_agent_sdk import ClaudeAgentOptions

        options = primitives.build_agent_options(tmp_path, disallowed_tools=["Write"])

        assert options == ClaudeAgentOptions(
            model="main-model",
            cwd=str(tmp_path),
            permission_mode="bypassPermissions",
            max_turns=25,
            max_budget_usd=2.0,
            env=dict(self._ROUTED_ENV),
            setting_sources=["project"],
            disallowed_tools=["Write"],
            mcp_servers=str(tmp_path / ".mcp.json"),
            strict_mcp_config=True,
        )

    @pytest.mark.parametrize("setting_sources", [None, ["project"], []], ids=repr)
    @pytest.mark.parametrize(
        "mcp_servers",
        [None, {"cf": {"command": "cf-mcp"}}, Path("/p/.mcp.json")],
        ids=["default", "mapping", "path"],
    )
    @pytest.mark.usefixtures("routing")
    def test_every_run_is_strict_about_mcp_servers(
        self, tmp_path: Path, setting_sources: Any, mcp_servers: Any
    ) -> None:
        options = primitives.build_agent_options(
            tmp_path,
            disallowed_tools=[],
            setting_sources=setting_sources,
            mcp_servers=mcp_servers,
        )

        assert options.strict_mcp_config is True

    @pytest.mark.usefixtures("routing")
    def test_the_project_layer_loads_the_rendered_config(self, tmp_path: Path) -> None:
        options = primitives.build_agent_options(tmp_path, disallowed_tools=[])

        assert options.mcp_servers == str(tmp_path / ".mcp.json")

    @pytest.mark.usefixtures("routing")
    def test_a_run_without_the_project_layer_loads_no_server(self, tmp_path: Path) -> None:
        options = primitives.build_agent_options(tmp_path, disallowed_tools=[], setting_sources=[])

        assert options.mcp_servers == {}

    @pytest.mark.parametrize("setting_sources", [None, []], ids=repr)
    @pytest.mark.usefixtures("routing")
    def test_caller_named_servers_replace_the_rendered_config(
        self, tmp_path: Path, setting_sources: Any
    ) -> None:
        servers = {"cf": {"command": "cf-mcp"}}
        config = tmp_path / "elsewhere" / "servers.json"

        from_mapping = primitives.build_agent_options(
            tmp_path,
            disallowed_tools=[],
            setting_sources=setting_sources,
            mcp_servers=servers,  # type: ignore[arg-type]
        )
        from_path = primitives.build_agent_options(
            tmp_path, disallowed_tools=[], setting_sources=setting_sources, mcp_servers=config
        )

        assert from_mapping.mcp_servers == servers
        assert from_path.mcp_servers == str(config)

    @pytest.mark.parametrize(
        ("kwargs", "field_name", "expected"),
        [
            ({"allowed_tools": ("Read", "mcp__x__*")}, "allowed_tools", ["Read", "mcp__x__*"]),
            ({"system_prompt": "be terse"}, "system_prompt", "be terse"),
            ({"session_id": "sid-1"}, "session_id", "sid-1"),
            ({"resume": "sid-0"}, "resume", "sid-0"),
            ({"stderr": _stderr_sink}, "stderr", _stderr_sink),
            ({"can_use_tool": _can_use_tool}, "can_use_tool", _can_use_tool),
            (
                {"mcp_servers": {"cf": {"command": "cf-mcp"}}},
                "mcp_servers",
                {"cf": {"command": "cf-mcp"}},
            ),
            ({"mcp_servers": Path("/p/.mcp.json")}, "mcp_servers", "/p/.mcp.json"),
            ({"max_turns": None}, "max_turns", None),
            ({"max_budget_usd": None}, "max_budget_usd", None),
            ({"permission_mode": None}, "permission_mode", None),
        ],
        ids=[
            "allowed_tools",
            "system_prompt",
            "session_id",
            "resume",
            "stderr",
            "can_use_tool",
            "mcp_servers-mapping",
            "mcp_servers-path",
            "max_turns-omitted",
            "max_budget_usd-omitted",
            "permission_mode-omitted",
        ],
    )
    @pytest.mark.usefixtures("routing")
    def test_each_option_reaches_the_agent_options(
        self,
        tmp_path: Path,
        kwargs: dict[str, Any],
        field_name: str,
        expected: Any,
    ) -> None:
        options = primitives.build_agent_options(tmp_path, disallowed_tools=[], **kwargs)

        assert getattr(options, field_name) == expected

    @pytest.mark.usefixtures("routing")
    def test_a_preset_system_prompt_reaches_the_agent_options(self, tmp_path: Path) -> None:
        preset = {"type": "preset", "preset": "claude_code", "append": "Answer briefly."}

        options = primitives.build_agent_options(
            tmp_path,
            disallowed_tools=[],
            system_prompt=preset,  # type: ignore[arg-type]
        )

        assert options.system_prompt == preset

    @pytest.mark.usefixtures("routing")
    def test_pre_tool_use_hooks_become_one_unscoped_matcher(self, tmp_path: Path) -> None:
        options = primitives.build_agent_options(
            tmp_path, disallowed_tools=[], pre_tool_use_hooks=[_hook]
        )

        assert options.hooks is not None
        assert list(options.hooks) == ["PreToolUse"]
        [matcher] = options.hooks["PreToolUse"]
        assert matcher.matcher is None
        assert matcher.hooks == [_hook]

    @pytest.mark.usefixtures("routing")
    def test_no_hooks_leave_the_hooks_field_unset(self, tmp_path: Path) -> None:
        options = primitives.build_agent_options(tmp_path, disallowed_tools=[])

        assert options.hooks is None

    def test_a_caller_env_is_used_verbatim_and_nothing_is_resolved(
        self, tmp_path: Path, routing: dict[str, Any]
    ) -> None:
        env = {"ANTHROPIC_BASE_URL": "https://caller.example", "ANTHROPIC_MODEL": "m-x"}

        options = primitives.build_agent_options(tmp_path, disallowed_tools=[], env=env)

        assert options.env == env
        assert options.env is not env
        assert options.model is None
        for name in ("sdk_env", "resolve_default_model", "_resolve_project_spec", "start_proxy"):
            routing[name].assert_not_called()

    def test_a_provider_override_routes_env_model_and_proxy_through_that_provider(
        self, tmp_path: Path, routing: dict[str, Any]
    ) -> None:
        routing["sdk_env"].side_effect = lambda *_a, **_k: {
            "CLAUDECODE": "",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_AUTH_TOKEN": "sk-gw",
        }
        routing["resolve_default_model"].return_value = "argo-main"
        routing["_resolve_project_spec"].return_value = _RoutedSpec(
            needs_proxy=True, provider="argo"
        )

        options = primitives.build_agent_options(tmp_path, disallowed_tools=[], provider="argo")

        routing["sdk_env"].assert_called_once_with(tmp_path, provider="argo")
        routing["resolve_default_model"].assert_called_once_with(tmp_path, provider="argo")
        routing["_resolve_project_spec"].assert_called_once_with(tmp_path, provider="argo")
        routing["start_proxy"].assert_called_once_with(
            "https://gateway.example/v1",
            "sk-gw",
            provider="argo",
            forward_headers=frozenset(),
            supports_images=None,
        )
        assert options.model == "argo-main"
        assert options.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8123"

    @pytest.mark.usefixtures("routing")
    def test_env_and_provider_together_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="env and provider are exclusive"):
            primitives.build_agent_options(
                tmp_path, disallowed_tools=[], env={"A": "1"}, provider="argo"
            )

    @pytest.mark.usefixtures("routing")
    def test_session_id_and_resume_together_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="session_id and resume are exclusive"):
            primitives.build_agent_options(
                tmp_path, disallowed_tools=[], session_id="new", resume="old"
            )

    def test_resolve_default_model_honours_a_provider_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str | None] = []

        def _spec(_project_dir: Path, *, provider: str | None = None) -> _RoutedSpec:
            seen.append(provider)
            return _RoutedSpec(provider=provider or "anthropic")

        monkeypatch.setattr(primitives, "_resolve_project_spec", _spec)

        assert primitives.resolve_default_model(tmp_path) == "anthropic-main"
        assert primitives.resolve_default_model(tmp_path, provider="argo") == "argo-main"
        assert seen == [None, "argo"]
