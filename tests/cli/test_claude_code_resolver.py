"""Tests for Claude Code model provider resolver."""

import pytest

from osprey.cli.claude_code_resolver import (
    AGENT_DEFAULT_TIERS,
    CLAUDE_CODE_PROVIDERS,
    VALID_TIERS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
)


class TestResolveReturnsNone:
    """resolve() returns None when no provider is configured."""

    def test_empty_config(self):
        assert ClaudeCodeModelResolver.resolve({}) is None

    def test_no_provider_key(self):
        assert ClaudeCodeModelResolver.resolve({"models": {"haiku": "x"}}) is None

    def test_provider_is_none(self):
        assert ClaudeCodeModelResolver.resolve({"provider": None}) is None

    def test_provider_is_empty_string(self):
        assert ClaudeCodeModelResolver.resolve({"provider": ""}) is None


class TestAnthropicProvider:
    """Anthropic direct provider configuration."""

    def test_env_block_no_auth_key(self):
        """Auth is handled via shell exports, not env block."""
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec is not None
        assert "ANTHROPIC_API_KEY" not in spec.env_block
        assert "ANTHROPIC_AUTH_TOKEN" not in spec.env_block

    def test_env_block_no_base_url(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert "ANTHROPIC_BASE_URL" not in spec.env_block

    def test_shell_exports_for_api_key(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert len(spec.shell_exports) == 1
        assert "ANTHROPIC_API_KEY" in spec.shell_exports[0]

    def test_model_tiers(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.tier_to_model["haiku"] == "claude-haiku-4-5-20251001"
        assert spec.tier_to_model["sonnet"] == "claude-sonnet-4-5-20250929"
        assert spec.tier_to_model["opus"] == "claude-opus-4-6"


class TestCBORGProvider:
    """CBORG (LBNL proxy) provider configuration."""

    def test_env_block_has_base_url_but_no_auth(self):
        """Auth is handled via shell exports, not env block."""
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert "ANTHROPIC_AUTH_TOKEN" not in spec.env_block
        assert "ANTHROPIC_API_KEY" not in spec.env_block
        assert "ANTHROPIC_BASE_URL" in spec.env_block

    def test_base_url_is_literal_no_v1(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://api.cborg.lbl.gov"

    def test_base_url_ignores_providers_section(self):
        """Base URL always uses provider literal, never user config (which has /v1)."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"base_url": "https://cborg.lbl.gov/v1"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://api.cborg.lbl.gov"

    def test_shell_exports_for_auth_token(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert len(spec.shell_exports) == 1
        assert 'ANTHROPIC_AUTH_TOKEN="$CBORG_API_KEY"' in spec.shell_exports[0]

    def test_model_tiers(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.tier_to_model["haiku"] == "anthropic/claude-haiku"
        assert spec.tier_to_model["sonnet"] == "anthropic/claude-sonnet"
        assert spec.tier_to_model["opus"] == "anthropic/claude-opus"


class TestUnsupportedProvider:
    """Unsupported provider raises ValueError."""

    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported.*'openai'"):
            ClaudeCodeModelResolver.resolve({"provider": "openai"})

    def test_error_lists_supported(self):
        with pytest.raises(ValueError, match="anthropic.*cborg"):
            ClaudeCodeModelResolver.resolve({"provider": "bad"})


class TestAgentModel:
    """ClaudeCodeModelSpec.agent_model() resolution."""

    def test_uses_default_tier(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        # channel-finder default tier is haiku
        assert spec.agent_model("channel-finder") == "anthropic/claude-haiku"

    def test_respects_per_agent_override(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "agent_models": {"channel-finder": "sonnet"}}
        )
        assert spec.agent_model("channel-finder") == "anthropic/claude-sonnet"

    def test_unknown_agent_returns_sonnet(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.agent_model("unknown-agent") == "anthropic/claude-sonnet"

    def test_logbook_deep_research_default_opus(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.agent_model("logbook-deep-research") == "claude-opus-4-6"


class TestPerTierOverrides:
    """Per-tier model override in config overrides provider default."""

    def test_override_single_tier(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "models": {"sonnet": "anthropic/claude-sonnet-v2"}}
        )
        assert spec.tier_to_model["sonnet"] == "anthropic/claude-sonnet-v2"
        # Others unchanged
        assert spec.tier_to_model["haiku"] == "anthropic/claude-haiku"

    def test_invalid_tier_ignored(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "models": {"gpt-4": "openai/gpt-4"}}
        )
        assert "gpt-4" not in spec.tier_to_model

    def test_override_affects_agent_resolution(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "models": {"haiku": "anthropic/claude-haiku-v2"}}
        )
        assert spec.agent_model("channel-finder") == "anthropic/claude-haiku-v2"


class TestValidateProvider:
    """validate_provider() static method."""

    def test_supported_providers(self):
        assert ClaudeCodeModelResolver.validate_provider("anthropic") is True
        assert ClaudeCodeModelResolver.validate_provider("cborg") is True

    def test_unsupported_provider(self):
        assert ClaudeCodeModelResolver.validate_provider("openai") is False


class TestAgentDefaultTiersConsistency:
    """AGENT_DEFAULT_TIERS entries are all valid tiers."""

    def test_all_tiers_valid(self):
        for agent, tier in AGENT_DEFAULT_TIERS.items():
            assert tier in VALID_TIERS, f"Agent '{agent}' has invalid tier '{tier}'"

    def test_all_agents_in_all_providers(self):
        """Every default tier has a model in every provider."""
        for provider_name, provider_def in CLAUDE_CODE_PROVIDERS.items():
            for agent, tier in AGENT_DEFAULT_TIERS.items():
                assert tier in provider_def["models"], (
                    f"Provider '{provider_name}' missing model for tier '{tier}' "
                    f"(needed by agent '{agent}')"
                )


class TestEnvBlockTierModels:
    """Env block contains ANTHROPIC_DEFAULT_*_MODEL vars for all providers."""

    def test_anthropic_has_all_tier_model_vars(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.env_block["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-haiku-4-5-20251001"
        assert spec.env_block["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-4-5-20250929"
        assert spec.env_block["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-opus-4-6"

    def test_cborg_has_all_tier_model_vars(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.env_block["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "anthropic/claude-haiku"
        assert spec.env_block["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "anthropic/claude-sonnet"
        assert spec.env_block["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "anthropic/claude-opus"

    def test_custom_tier_override_propagates_to_env_block(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "models": {"sonnet": "anthropic/claude-sonnet-v2"}}
        )
        assert spec.env_block["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "anthropic/claude-sonnet-v2"
        # Others unchanged
        assert spec.env_block["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "anthropic/claude-haiku"
        assert spec.env_block["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "anthropic/claude-opus"

    def test_all_three_vars_always_present(self):
        for provider_name in CLAUDE_CODE_PROVIDERS:
            spec = ClaudeCodeModelResolver.resolve({"provider": provider_name})
            for var in (
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
            ):
                assert var in spec.env_block, f"{var} missing for {provider_name}"


class TestDefaultModelTier:
    """ClaudeCodeModelSpec.default_model_tier field."""

    def test_cborg_defaults_to_opus(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.default_model_tier == "opus"

    def test_anthropic_defaults_to_sonnet(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.default_model_tier == "sonnet"

    def test_config_override_via_default_model(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "default_model": "haiku"}
        )
        assert spec.default_model_tier == "haiku"

    def test_invalid_tier_falls_back_to_provider_default(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "default_model": "gpt-4"}
        )
        assert spec.default_model_tier == "opus"

    def test_field_present_on_spec(self):
        spec = ClaudeCodeModelSpec(provider="test")
        assert spec.default_model_tier == "sonnet"  # dataclass default


class TestAgentTier:
    """ClaudeCodeModelSpec.agent_tier() returns tier aliases, not model IDs."""

    def test_returns_tier_not_model_id(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.agent_tier("channel-finder") == "haiku"

    def test_per_agent_override(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "agent_models": {"channel-finder": "sonnet"}}
        )
        assert spec.agent_tier("channel-finder") == "sonnet"

    def test_unknown_agent_falls_back_to_sonnet(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.agent_tier("unknown-agent") == "sonnet"

    def test_consistency_agent_model_uses_agent_tier(self):
        """agent_model(x) == tier_to_model[agent_tier(x)] for all known agents."""
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        for agent_name in AGENT_DEFAULT_TIERS:
            tier = spec.agent_tier(agent_name)
            assert spec.agent_model(agent_name) == spec.tier_to_model[tier]

    def test_logbook_deep_research_default_opus(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.agent_tier("logbook-deep-research") == "opus"


class TestAuthVarSeparation:
    """Providers use the correct auth env var in shell_exports (not env block)."""

    def test_anthropic_shell_export_uses_api_key(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert any("ANTHROPIC_API_KEY" in e for e in spec.shell_exports)
        assert not any("ANTHROPIC_AUTH_TOKEN" in e for e in spec.shell_exports)

    def test_cborg_shell_export_uses_auth_token(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert any("ANTHROPIC_AUTH_TOKEN" in e for e in spec.shell_exports)
        assert not any("ANTHROPIC_API_KEY" in e for e in spec.shell_exports)

    def test_cborg_shell_export_references_cborg_api_key(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert any("CBORG_API_KEY" in e for e in spec.shell_exports)

    def test_env_block_never_contains_auth_keys(self):
        """Auth keys must not be in env block (Claude Code doesn't expand ${VAR})."""
        for provider_name in CLAUDE_CODE_PROVIDERS:
            spec = ClaudeCodeModelResolver.resolve({"provider": provider_name})
            assert "ANTHROPIC_API_KEY" not in spec.env_block
            assert "ANTHROPIC_AUTH_TOKEN" not in spec.env_block


class TestModelSpecFrozen:
    """ClaudeCodeModelSpec is immutable."""

    def test_cannot_set_attributes(self):
        spec = ClaudeCodeModelSpec(provider="test")
        with pytest.raises(AttributeError):
            spec.provider = "other"
