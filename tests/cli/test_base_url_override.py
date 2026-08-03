"""`api.providers.<name>.base_url` overrides the built-in provider URL.

Before this, `ANTHROPIC_BASE_URL` for a built-in provider (anthropic / cborg /
als-apg) came solely from the hardcoded `CLAUDE_CODE_PROVIDERS` table, so a
facility that pointed a built-in provider at its own gateway got a green
`osprey health` probe against an endpoint the agent never used. The config
value now wins, on the same precedence rule as the `models` tier map, with the
trailing-`/v1` strip unchanged.
"""

from __future__ import annotations

import pytest

from osprey.build.claude_code_resolver import CLAUDE_CODE_PROVIDERS, ClaudeCodeModelResolver

FACILITY_GATEWAY = "https://llm.facility.example.org"


def _tier_map() -> dict[str, str]:
    return {"haiku": "fast", "sonnet": "balanced", "opus": "capable"}


class TestBuiltInProviderOverride:
    """A base_url under api.providers reaches ANTHROPIC_BASE_URL."""

    @pytest.mark.parametrize("provider", sorted(CLAUDE_CODE_PROVIDERS))
    def test_custom_base_url_reaches_env_block(self, provider):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": provider},
            api_providers={provider: {"base_url": FACILITY_GATEWAY}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    @pytest.mark.parametrize("provider", sorted(CLAUDE_CODE_PROVIDERS))
    def test_override_does_not_mutate_the_builtin_table(self, provider):
        """The built-in dict is module-level shared state — resolve() must not write to it."""
        before = CLAUDE_CODE_PROVIDERS[provider]["base_url"]
        ClaudeCodeModelResolver.resolve(
            {"provider": provider},
            api_providers={provider: {"base_url": FACILITY_GATEWAY}},
        )
        assert CLAUDE_CODE_PROVIDERS[provider]["base_url"] == before

    def test_builtin_url_survives_when_config_names_no_base_url(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"},
            api_providers={"als-apg": {"api_key": "k"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://llm.gianlucamartino.com"

    def test_anthropic_gains_a_base_url_only_when_config_gives_one(self):
        """The built-in anthropic entry has no URL; config is the only source."""
        bare = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert "ANTHROPIC_BASE_URL" not in bare.env_block

        fronted = ClaudeCodeModelResolver.resolve(
            {"provider": "anthropic"},
            api_providers={"anthropic": {"base_url": FACILITY_GATEWAY}},
        )
        assert fronted.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY


class TestV1StripSurvivesTheOverride:
    """Claude Code appends /v1/messages, so the env var must never end in /v1."""

    def test_overridden_url_is_stripped_of_v1(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"base_url": f"{FACILITY_GATEWAY}/v1"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_overridden_url_is_stripped_of_trailing_slash_and_v1(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"base_url": f"{FACILITY_GATEWAY}/v1/"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_shipped_template_urls_are_unchanged_by_the_override(self):
        """cborg/als-apg templates ship the built-in URL + /v1 — same result as before."""
        for provider, shipped, expected in (
            ("cborg", "https://api.cborg.lbl.gov/v1", "https://api.cborg.lbl.gov"),
            (
                "als-apg",
                "https://llm.gianlucamartino.com/v1",
                "https://llm.gianlucamartino.com",
            ),
        ):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": provider}, api_providers={provider: {"base_url": shipped}}
            )
            assert spec.env_block["ANTHROPIC_BASE_URL"] == expected


class TestProxyUpstreamFollowsTheOverride:
    """upstream_base_url stays the single source the launch paths start the proxy from."""

    def test_builtin_providers_are_native_so_no_upstream_is_set(self):
        """Built-ins skip the translation proxy — overriding the URL must not change that."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"base_url": f"{FACILITY_GATEWAY}/v1"}},
        )
        assert spec.needs_proxy is False
        assert spec.upstream_base_url is None

    def test_custom_proxy_upstream_keeps_v1_from_the_same_resolved_url(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "my-gateway"},
            api_providers={
                "my-gateway": {"base_url": f"{FACILITY_GATEWAY}/v1", "models": _tier_map()}
            },
        )
        assert spec.needs_proxy is True
        assert spec.upstream_base_url == f"{FACILITY_GATEWAY}/v1"
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY


class TestEndToEndThroughLoadProviderSpec:
    """The on-disk path: config.yml override reaches the spec, ${VAR} included."""

    def test_config_yml_override_reaches_env_block(self, tmp_path):
        from osprey.build.claude_code_resolver import load_provider_spec

        (tmp_path / "config.yml").write_text(
            "api:\n"
            "  providers:\n"
            "    cborg:\n"
            f"      base_url: {FACILITY_GATEWAY}/v1\n"
            "claude_code:\n"
            "  provider: cborg\n"
        )
        spec = load_provider_spec(tmp_path, include_telemetry=False)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY

    def test_env_placeholder_in_builtin_override_is_expanded(self, tmp_path, monkeypatch):
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.delenv("FACILITY_GATEWAY_URL", raising=False)
        (tmp_path / "config.yml").write_text(
            "api:\n"
            "  providers:\n"
            "    cborg:\n"
            "      base_url: ${FACILITY_GATEWAY_URL}\n"
            "claude_code:\n"
            "  provider: cborg\n"
        )
        (tmp_path / ".env").write_text(f"FACILITY_GATEWAY_URL={FACILITY_GATEWAY}/v1\n")

        spec = load_provider_spec(tmp_path, include_telemetry=False)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == FACILITY_GATEWAY
