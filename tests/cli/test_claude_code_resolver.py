"""Tests for Claude Code model provider resolver."""

import pytest

from osprey.cli.claude_code_resolver import (
    AGENT_DEFAULT_TIERS,
    CLAUDE_CODE_PROVIDERS,
    VALID_TIERS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
    inject_provider_env,
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


class TestAlsApgProvider:
    """ALS-APG (LBL AWS proxy) provider configuration."""

    def test_env_block_has_base_url_but_no_auth(self):
        """Auth is handled via shell exports, not env block."""
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"})
        assert "ANTHROPIC_AUTH_TOKEN" not in spec.env_block
        assert "ANTHROPIC_API_KEY" not in spec.env_block
        assert "ANTHROPIC_BASE_URL" in spec.env_block

    def test_base_url_is_correct(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"})
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://llm.gianlucamartino.com"

    def test_shell_exports_use_als_apg_api_key(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"})
        assert len(spec.shell_exports) == 1
        assert 'ANTHROPIC_AUTH_TOKEN="$ALS_APG_API_KEY"' in spec.shell_exports[0]

    def test_model_tiers(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"})
        assert spec.tier_to_model["haiku"] == "claude-haiku-4-5-20251001"
        assert spec.tier_to_model["sonnet"] == "claude-sonnet-4-6"
        assert spec.tier_to_model["opus"] == "claude-opus-4-6"

    def test_default_model_tier_is_haiku(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"})
        assert spec.default_model_tier == "haiku"


class TestUnsupportedProvider:
    """Unknown provider without api_providers entry raises ValueError."""

    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown.*'openai'"):
            ClaudeCodeModelResolver.resolve({"provider": "openai"})

    def test_error_lists_built_ins(self):
        with pytest.raises(ValueError, match="anthropic.*cborg"):
            ClaudeCodeModelResolver.resolve({"provider": "bad"})

    def test_error_mentions_api_providers(self):
        with pytest.raises(ValueError, match="api.providers"):
            ClaudeCodeModelResolver.resolve({"provider": "my-proxy"})

    def test_known_in_api_providers_does_not_raise(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "my-proxy"},
            api_providers={"my-proxy": {"base_url": "https://my-proxy.example.com"}},
        )
        assert spec is not None


class TestCustomProxyProvider:
    """Custom Anthropic-compatible proxy via api.providers.

    With the redesign, custom proxies own their model IDs via
    api.providers[name].models.  If not specified, the last-resort
    Anthropic direct IDs are used so env block generation never crashes.
    """

    _API_PROVIDERS = {
        "lbl-aws": {
            "api_key": "${LBL_AWS_API_KEY}",
            "base_url": "https://llm.example.com",
            "models": {
                "haiku": "claude-haiku-4-5-20251001",
                "sonnet": "claude-sonnet-4-6",
                "opus": "claude-opus-4-6",
            },
        }
    }

    def test_resolves_to_spec(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"}, api_providers=self._API_PROVIDERS
        )
        assert spec is not None
        assert spec.provider == "lbl-aws"

    def test_injects_base_url_from_api_providers(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"}, api_providers=self._API_PROVIDERS
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://llm.example.com"

    def test_uses_model_ids_from_api_providers(self):
        """Model IDs come from api.providers[name].models, not from hardcoded defaults."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"}, api_providers=self._API_PROVIDERS
        )
        assert spec.tier_to_model["haiku"] == "claude-haiku-4-5-20251001"
        assert spec.tier_to_model["sonnet"] == "claude-sonnet-4-6"
        assert spec.tier_to_model["opus"] == "claude-opus-4-6"

    def test_default_model_tier_is_opus(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"}, api_providers=self._API_PROVIDERS
        )
        assert spec.default_model_tier == "opus"

    def test_shell_exports_use_auth_token(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"}, api_providers=self._API_PROVIDERS
        )
        assert any("ANTHROPIC_AUTH_TOKEN" in e for e in spec.shell_exports)

    def test_env_block_has_tier_model_vars(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"}, api_providers=self._API_PROVIDERS
        )
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" in spec.env_block
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in spec.env_block
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" in spec.env_block

    def test_per_tier_overrides_still_apply(self):
        spec = ClaudeCodeModelResolver.resolve(
            {
                "provider": "lbl-aws",
                "models": {"sonnet": "claude-sonnet-special"},
            },
            api_providers=self._API_PROVIDERS,
        )
        assert spec.tier_to_model["sonnet"] == "claude-sonnet-special"
        assert spec.tier_to_model["haiku"] == "claude-haiku-4-5-20251001"  # from api.providers

    def test_no_models_in_api_providers_falls_back_to_last_resort(self):
        """Without api.providers.models, Anthropic direct IDs are used as last resort."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"},
            api_providers={"lbl-aws": {"base_url": "https://llm.example.com"}},
        )
        # Falls back to last-resort Anthropic direct IDs
        assert spec.tier_to_model["haiku"] == "claude-haiku-4-5-20251001"
        assert spec.tier_to_model["opus"] == "claude-opus-4-6"

    def test_hyphenated_name_generates_valid_secret_env(self):
        """Provider name 'lbl-aws' → secret env var 'LBL_AWS_API_KEY'."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"}, api_providers=self._API_PROVIDERS
        )
        assert any("LBL_AWS_API_KEY" in e for e in spec.shell_exports)


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


class TestApiProvidersModelAuthority:
    """api.providers[name].models is the authoritative source for model IDs.

    These tests verify that model IDs defined in api.providers override the
    built-in fallback values in CLAUDE_CODE_PROVIDERS, and that claude_code.models
    overrides api.providers.models.
    """

    def test_api_providers_models_override_builtin_for_cborg(self):
        """api.providers.cborg.models overrides the built-in cborg fallback."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={
                "cborg": {
                    "base_url": "https://api.cborg.lbl.gov/v1",
                    "models": {
                        "haiku": "anthropic/claude-haiku-4",
                        "sonnet": "anthropic/claude-sonnet-4",
                        "opus": "anthropic/claude-opus-4",
                    },
                }
            },
        )
        assert spec.tier_to_model["haiku"] == "anthropic/claude-haiku-4"
        assert spec.tier_to_model["sonnet"] == "anthropic/claude-sonnet-4"
        assert spec.tier_to_model["opus"] == "anthropic/claude-opus-4"

    def test_api_providers_models_override_builtin_for_als_apg(self):
        """api.providers.als-apg.models overrides the built-in als-apg fallback."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"},
            api_providers={
                "als-apg": {
                    "base_url": "https://llm.gianlucamartino.com/v1",
                    "models": {
                        "haiku": "claude-haiku-4-5-20251001",
                        "sonnet": "claude-sonnet-4-6",
                        "opus": "claude-opus-4-6",
                    },
                }
            },
        )
        assert spec.tier_to_model["haiku"] == "claude-haiku-4-5-20251001"
        assert spec.tier_to_model["sonnet"] == "claude-sonnet-4-6"
        assert spec.tier_to_model["opus"] == "claude-opus-4-6"

    def test_claude_code_models_override_api_providers_models(self):
        """claude_code.models takes highest priority over api.providers.models."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "models": {"sonnet": "anthropic/claude-sonnet-special"}},
            api_providers={"cborg": {"models": {"sonnet": "anthropic/claude-sonnet-4"}}},
        )
        assert spec.tier_to_model["sonnet"] == "anthropic/claude-sonnet-special"
        # haiku comes from api.providers (not builtin, not claude_code.models)
        assert spec.tier_to_model["haiku"] == "anthropic/claude-haiku"  # builtin fallback

    def test_resolution_priority_chain(self):
        """Full priority chain: claude_code.models > api.providers.models > builtin."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "models": {"opus": "override-opus"}},
            api_providers={"cborg": {"models": {"sonnet": "api-sonnet"}}},
        )
        assert spec.tier_to_model["opus"] == "override-opus"  # claude_code.models
        assert spec.tier_to_model["sonnet"] == "api-sonnet"  # api.providers.models
        assert spec.tier_to_model["haiku"] == "anthropic/claude-haiku"  # builtin fallback

    def test_custom_proxy_uses_api_providers_models(self):
        """Custom proxy reads model IDs from api.providers, not from hardcoded fallback."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "my-proxy"},
            api_providers={
                "my-proxy": {
                    "base_url": "https://my-proxy.example.com",
                    "models": {
                        "haiku": "my-haiku-model",
                        "sonnet": "my-sonnet-model",
                        "opus": "my-opus-model",
                    },
                }
            },
        )
        assert spec.tier_to_model["haiku"] == "my-haiku-model"
        assert spec.tier_to_model["sonnet"] == "my-sonnet-model"
        assert spec.tier_to_model["opus"] == "my-opus-model"


class TestValidateProvider:
    """validate_provider() static method."""

    def test_built_in_providers(self):
        assert ClaudeCodeModelResolver.validate_provider("anthropic") is True
        assert ClaudeCodeModelResolver.validate_provider("cborg") is True

    def test_unknown_without_api_providers(self):
        assert ClaudeCodeModelResolver.validate_provider("openai") is False

    def test_custom_provider_in_api_providers(self):
        assert (
            ClaudeCodeModelResolver.validate_provider(
                "my-proxy", api_providers={"my-proxy": {"base_url": "https://x.example.com"}}
            )
            is True
        )

    def test_custom_provider_not_in_api_providers(self):
        assert (
            ClaudeCodeModelResolver.validate_provider("my-proxy", api_providers={"other": {}})
            is False
        )


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

    def test_cborg_defaults_to_haiku(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.default_model_tier == "haiku"

    def test_anthropic_defaults_to_sonnet(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.default_model_tier == "sonnet"

    def test_config_override_via_default_model(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": "haiku"})
        assert spec.default_model_tier == "haiku"

    def test_invalid_tier_falls_back_to_provider_default(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": "gpt-4"})
        assert spec.default_model_tier == "haiku"

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


class TestInjectProviderEnv:
    """inject_provider_env() scrubs, injects env block, and wires auth."""

    def test_scrubs_managed_vars(self):
        env = {"ANTHROPIC_BASE_URL": "stale", "ANTHROPIC_MODEL": "stale", "HOME": "/home"}
        spec = ClaudeCodeModelSpec(provider="test", env_block={})
        inject_provider_env(env, spec)
        assert "ANTHROPIC_BASE_URL" not in env
        assert "ANTHROPIC_MODEL" not in env
        assert env["HOME"] == "/home"

    def test_injects_env_block(self):
        env = {}
        spec = ClaudeCodeModelSpec(
            provider="test",
            env_block={"ANTHROPIC_BASE_URL": "https://proxy.example.com", "ANTHROPIC_MODEL": "m"},
        )
        inject_provider_env(env, spec)
        assert env["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"
        assert env["ANTHROPIC_MODEL"] == "m"

    def test_injects_auth(self):
        env = {"CBORG_API_KEY": "secret-123"}
        spec = ClaudeCodeModelSpec(
            provider="cborg",
            env_block={},
            auth_env_var="ANTHROPIC_AUTH_TOKEN",
            auth_secret_env="CBORG_API_KEY",
        )
        inject_provider_env(env, spec)
        assert env["ANTHROPIC_AUTH_TOKEN"] == "secret-123"

    def test_reads_auth_before_scrub(self):
        """Anthropic provider: auth_secret_env == ANTHROPIC_API_KEY (a managed var)."""
        env = {"ANTHROPIC_API_KEY": "my-key"}
        spec = ClaudeCodeModelSpec(
            provider="anthropic",
            env_block={},
            auth_env_var="ANTHROPIC_API_KEY",
            auth_secret_env="ANTHROPIC_API_KEY",
        )
        inject_provider_env(env, spec)
        # Key should survive: read before scrub, then re-injected as auth
        assert env["ANTHROPIC_API_KEY"] == "my-key"

    def test_returns_injected_keys(self):
        env = {}
        spec = ClaudeCodeModelSpec(
            provider="test",
            env_block={"ANTHROPIC_MODEL": "m", "ANTHROPIC_BASE_URL": "u"},
        )
        result = inject_provider_env(env, spec)
        assert result == ["ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"]
