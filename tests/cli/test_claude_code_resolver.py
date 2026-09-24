"""Tests for Claude Code model provider resolver."""

import logging
import os
import re
from pathlib import Path

import pytest

from osprey.build.claude_code_resolver import (
    CLAUDE_CODE_PROVIDERS,
    TIER_MODEL_ENV_VARS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
    alias_substitution,
    inject_provider_env,
    unserved_model_ids,
)
from osprey.profiles.providers import load_provider_catalog
from tests.conftest import GATEWAY_BASE_URL, GATEWAY_ORIGIN


def _resolve_builtin(provider_name: str):
    """Resolve a built-in provider, naming a gateway for the ones that need one.

    An entry that requires an endpoint and ships none (``requires_base_url``)
    is refused unless config or the environment names one, so a sweep over the
    whole table has to supply it — otherwise the sweep would only ever pass for
    the providers that ship a URL.
    """
    api_providers = (
        {provider_name: {"base_url": GATEWAY_BASE_URL}}
        if CLAUDE_CODE_PROVIDERS[provider_name].get("requires_base_url")
        else {}
    )
    return ClaudeCodeModelResolver.resolve({"provider": provider_name}, api_providers)


def _gateway(default: str = "s", models: tuple[str, ...] = ("h", "s", "o"), **extra) -> dict:
    """An api.providers entry for a custom gateway."""
    return {
        "base_url": "https://gateway.example.org/v1",
        "default_model": default,
        "models": list(models),
        **extra,
    }


#: What the als-apg entry lists as served, in the catalog's spelling.
ALS_APG_SERVED = [
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
]


class TestResolveReturnsNone:
    """resolve() returns None when no provider is configured."""

    def test_empty_config(self):
        assert ClaudeCodeModelResolver.resolve({}) is None

    def test_no_provider_key(self):
        assert ClaudeCodeModelResolver.resolve({"aliases": {"haiku": "x"}}) is None

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

    def test_main_model_and_aliases_come_from_the_packaged_entry(self):
        """Named by provider alone, direct Anthropic reads its packaged catalog entry."""
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        entry = load_provider_catalog(None).entries["anthropic"]
        assert spec.default_model_id == entry["default_model"] == "claude-sonnet-5"
        assert spec.served_models == entry["models"]
        assert spec.alias_models == {
            "haiku": "claude-haiku-4-5",
            "sonnet": "claude-sonnet-5",
            "opus": "claude-opus-5-5",
        }


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

    def test_base_url_from_providers_section_overrides_builtin(self):
        """A config base_url wins over the built-in literal; the /v1 is still stripped."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"base_url": "https://cborg.lbl.gov/v1"}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://cborg.lbl.gov"

    def test_base_url_falls_back_to_builtin_when_config_names_none(self):
        """An api.providers entry without base_url leaves the built-in URL in place."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"},
            api_providers={"cborg": {"default_model": "s", "models": ["h", "s", "o"]}},
        )
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://api.cborg.lbl.gov"

    def test_shell_exports_for_auth_token(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert len(spec.shell_exports) == 1
        assert 'ANTHROPIC_AUTH_TOKEN="$CBORG_API_KEY"' in spec.shell_exports[0]

    def test_aliases_are_the_versioned_ids(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.alias_models == {
            "haiku": "claude-haiku-4-5",
            "sonnet": "claude-sonnet-5",
            "opus": "claude-opus-5",
        }
        assert spec.default_model_id == "claude-haiku-4-5"


class TestAlsApgProvider:
    """The ``als-apg`` gateway provider configuration.

    The table ships the gateway's own address, so a deployment that names none
    still routes — :meth:`test_the_shipped_gateway_stands_when_nothing_names_one`
    pins that. The other cases name an endpoint the way a site with its own
    gateway does, which is what :attr:`ENVIRON` supplies.
    """

    ENVIRON = {"ALS_APG_BASE_URL": GATEWAY_BASE_URL}

    def _spec(self):
        return ClaudeCodeModelResolver.resolve({"provider": "als-apg"}, environ=self.ENVIRON)

    def test_env_block_has_base_url_but_no_auth(self):
        """Auth is handled via shell exports, not env block."""
        spec = self._spec()
        assert "ANTHROPIC_AUTH_TOKEN" not in spec.env_block
        assert "ANTHROPIC_API_KEY" not in spec.env_block
        assert "ANTHROPIC_BASE_URL" in spec.env_block

    def test_base_url_is_the_configured_gateway(self):
        spec = self._spec()
        assert spec.env_block["ANTHROPIC_BASE_URL"] == GATEWAY_ORIGIN

    def test_the_shipped_gateway_stands_when_nothing_names_one(self):
        """An empty environment must still route, and route to the gateway.

        An omitted ANTHROPIC_BASE_URL would mean "Anthropic direct", i.e. the
        gateway's bearer token presented to api.anthropic.com.
        """
        spec = ClaudeCodeModelResolver.resolve({"provider": "als-apg"}, environ={})
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://llm.als.lbl.gov"

    def test_shell_exports_use_als_apg_api_key(self):
        spec = self._spec()
        assert len(spec.shell_exports) == 1
        assert 'ANTHROPIC_AUTH_TOKEN="$ALS_APG_API_KEY"' in spec.shell_exports[0]

    def test_aliases_are_derived_from_the_served_list(self):
        spec = self._spec()
        assert spec.alias_models == {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-5",
            "opus": "claude-opus-5-5",
        }
        assert spec.alias_origin == dict.fromkeys(TIER_MODEL_ENV_VARS, "derived")

    def test_main_model_is_the_catalog_default(self):
        assert self._spec().default_model_id == "claude-sonnet-5"


class TestUnrecognisedApiProtocol:
    """A misspelled api_protocol is refused where an unknown provider is."""

    def test_resolve_surfaces_the_refusal(self):
        api_providers = {"my-gateway": _gateway(api_protocol="Anthropic")}
        with pytest.raises(ValueError, match="api.providers.my-gateway.api_protocol"):
            ClaudeCodeModelResolver.resolve({"provider": "my-gateway"}, api_providers)

    def test_a_valid_protocol_resolves(self):
        """The check refuses spellings, not the two values themselves."""
        api_providers = {"my-gateway": _gateway(api_protocol="anthropic")}
        spec = ClaudeCodeModelResolver.resolve({"provider": "my-gateway"}, api_providers)
        assert spec is not None
        assert spec.needs_proxy is False
        assert spec.upstream_base_url is None

    def test_an_absent_protocol_still_routes_through_the_proxy(self):
        api_providers = {"my-gateway": _gateway()}
        spec = ClaudeCodeModelResolver.resolve({"provider": "my-gateway"}, api_providers)
        assert spec is not None
        assert spec.needs_proxy is True
        assert spec.upstream_base_url == "https://gateway.example.org/v1"


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

    def test_error_names_the_configured_providers_too(self):
        """The resolver accepts the UNION of built-ins and api.providers, so the
        error must name that union — not only the three built-ins (#725). The
        breakdown says which half each name came from."""
        api_providers = {
            "stanford": {"base_url": "https://x", "default_model": "gpt-4o", "models": ["gpt-4o"]},
            "argo": {
                "base_url": "https://y",
                "default_model": "claudesonnet45",
                "models": ["claudesonnet45"],
            },
        }
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "slac"}, api_providers)

        message = str(excinfo.value)
        assert "Available providers: als-apg, anthropic, argo, cborg, stanford" in message
        assert "built-in: als-apg, anthropic, cborg" in message
        assert "from api.providers in config.yml: argo, stanford" in message

    def test_error_without_api_providers_says_none_configured(self):
        with pytest.raises(ValueError, match=r"from api.providers in config.yml: none"):
            ClaudeCodeModelResolver.resolve({"provider": "bad"})

    def test_error_suggests_a_close_match(self):
        api_providers = {"stanford": _gateway()}
        with pytest.raises(ValueError, match=r"Did you mean 'stanford'\?"):
            ClaudeCodeModelResolver.resolve({"provider": "stanfrod"}, api_providers)
        with pytest.raises(ValueError, match=r"Did you mean 'anthropic'\?"):
            ClaudeCodeModelResolver.resolve({"provider": "anthropc"})

    def test_error_makes_no_suggestion_for_a_distant_name(self):
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "zzzz-nothing-like-it"})
        assert "Did you mean" not in str(excinfo.value)

    def test_known_in_api_providers_does_not_raise(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "my-proxy"},
            api_providers={"my-proxy": _gateway("proxy-sonnet", ("proxy-haiku", "proxy-sonnet"))},
        )
        assert spec is not None


class TestCustomProxyProvider:
    """Custom Anthropic-compatible proxy via api.providers.

    Custom proxies own their model ids via api.providers[name].models and
    default_model. A proxy that names neither is refused — the framework never
    substitutes another provider's model ids.
    """

    _API_PROVIDERS = {
        "lbl-aws": {
            "api_key": "${LBL_AWS_API_KEY}",
            "base_url": "https://llm.example.com",
            "default_model": "claude-sonnet-4-6",
            "models": [
                "claude-haiku-4-5-20251001",
                "claude-sonnet-4-6",
                "claude-opus-4-6",
            ],
        }
    }

    def _spec(self, **cc):
        return ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws", **cc}, api_providers=self._API_PROVIDERS
        )

    def test_resolves_to_spec(self):
        spec = self._spec()
        assert spec is not None
        assert spec.provider == "lbl-aws"

    def test_injects_base_url_from_api_providers(self):
        assert self._spec().env_block["ANTHROPIC_BASE_URL"] == "https://llm.example.com"

    def test_uses_model_ids_from_api_providers(self):
        """Model ids come from api.providers[name].models, not from hardcoded defaults."""
        spec = self._spec()
        assert spec.served_models == self._API_PROVIDERS["lbl-aws"]["models"]
        assert spec.alias_models == {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-6",
        }

    def test_main_model_is_the_entry_default(self):
        spec = self._spec()
        assert spec.default_model_id == "claude-sonnet-4-6"
        assert spec.env_block["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"

    def test_shell_exports_use_auth_token(self):
        assert any("ANTHROPIC_AUTH_TOKEN" in e for e in self._spec().shell_exports)

    def test_env_block_has_alias_model_vars(self):
        spec = self._spec()
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" in spec.env_block
        assert "ANTHROPIC_DEFAULT_SONNET_MODEL" in spec.env_block
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" in spec.env_block

    def test_per_alias_overrides_still_apply(self):
        spec = self._spec(aliases={"sonnet": "claude-sonnet-special"})
        assert spec.alias_models["sonnet"] == "claude-sonnet-special"
        assert spec.alias_models["haiku"] == "claude-haiku-4-5-20251001"  # derived

    def test_no_models_in_api_providers_is_refused(self):
        """A proxy that lists no models is an error, not an Anthropic-id fill.

        Full coverage of the refusal lives in test_provider_models_required.py.
        """
        with pytest.raises(ValueError, match="lists no models and names no default_model"):
            ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws"},
                api_providers={"lbl-aws": {"base_url": "https://llm.example.com"}},
            )

    def test_hyphenated_name_generates_valid_secret_env(self):
        """Provider name 'lbl-aws' → secret env var 'LBL_AWS_API_KEY'."""
        assert any("LBL_AWS_API_KEY" in e for e in self._spec().shell_exports)


class TestAgentModel:
    """ClaudeCodeModelSpec.agent_model(): the agent's own id, else the main model."""

    def test_an_agent_without_its_own_model_runs_the_main_model(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.agent_model("channel-finder") == spec.default_model_id == "claude-haiku-4-5"

    def test_respects_per_agent_model(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "agent_models": {"channel-finder": "claude-sonnet-5"}}
        )
        assert spec.agent_model("channel-finder") == "claude-sonnet-5"
        assert spec.agent_model("logbook-search") == "claude-haiku-4-5"

    def test_unknown_agent_runs_the_main_model(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.agent_model("unknown-agent") == "claude-sonnet-5"

    def test_a_bare_alias_word_is_refused_naming_the_key(self):
        with pytest.raises(ValueError, match=r"claude_code\.agent_models\.channel-finder: haiku"):
            ClaudeCodeModelResolver.resolve(
                {"provider": "cborg", "agent_models": {"channel-finder": "haiku"}}
            )

    def test_an_unserved_id_is_trusted_and_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": "cborg", "agent_models": {"channel-finder": "claude-next"}}
            )
        assert spec.agent_model("channel-finder") == "claude-next"
        assert "claude_code.agent_models.channel-finder" in caplog.text
        assert "trusting the gateway" in caplog.text


class TestPerAliasOverrides:
    """claude_code.aliases overrides a single Claude Code alias."""

    def test_override_single_alias(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "aliases": {"sonnet": "anthropic/claude-sonnet-v2"}}
        )
        assert spec.alias_models["sonnet"] == "anthropic/claude-sonnet-v2"
        assert spec.alias_origin["sonnet"] == "claude_code.aliases"
        # Others unchanged
        assert spec.alias_models["haiku"] == "claude-haiku-4-5"
        assert spec.alias_origin["haiku"] == "derived"

    def test_a_key_that_is_not_an_alias_name_is_dropped_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": "cborg", "aliases": {"sonet": "claude-sonnet-9"}}
            )
        assert "sonet" not in spec.alias_models
        assert "sonet" in caplog.text

    def test_an_alias_never_reaches_an_agent(self):
        """Agents name their model by id; overriding an alias does not move them."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "aliases": {"haiku": "anthropic/claude-haiku-v2"}}
        )
        assert spec.agent_model("channel-finder") == "claude-haiku-4-5"

    def test_an_alias_value_spelled_as_an_alias_word_is_refused(self):
        with pytest.raises(ValueError, match=r"claude_code\.aliases\.opus: sonnet"):
            ClaudeCodeModelResolver.resolve({"provider": "cborg", "aliases": {"opus": "sonnet"}})


class TestCatalogAuthority:
    """The api.providers entry is the one source of served ids and the default."""

    def test_the_entry_replaces_the_packaged_list_for_a_builtin(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"},
            api_providers={
                "als-apg": {
                    "base_url": GATEWAY_BASE_URL,
                    "default_model": "claude-sonnet-4-6",
                    "models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
                }
            },
        )
        assert spec.default_model_id == "claude-sonnet-4-6"
        assert spec.alias_models == {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-6",
        }

    def test_an_entry_naming_only_an_endpoint_reads_the_packaged_list(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg"}, api_providers={"cborg": {"base_url": "https://x/v1"}}
        )
        assert spec.served_models == load_provider_catalog(None).entries["cborg"]["models"]

    def test_a_tier_map_is_refused_with_the_refresh_command(self):
        with pytest.raises(ValueError, match="osprey profile expand --providers"):
            ClaudeCodeModelResolver.resolve(
                {"provider": "my-proxy"},
                api_providers={
                    "my-proxy": {
                        "base_url": "https://x",
                        "default_model": "a",
                        "models": {"haiku": "a", "sonnet": "b", "opus": "c"},
                    }
                },
            )


class TestAliasDerivation:
    """Claude Code's aliases: derived, then catalog, then claude_code.aliases."""

    def test_als_apg_derives_all_three(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "gw"}, api_providers={"gw": _gateway("claude-sonnet-5", ALS_APG_SERVED)}
        )
        assert spec.alias_models == {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-5",
            "opus": "claude-opus-5-5",
        }
        assert set(spec.alias_origin.values()) == {"derived"}

    def test_a_gateway_serving_no_claude_model_runs_the_main_model_and_records_it(self, caplog):
        """One INFO record for the sinks; the verbs promote the sentence themselves."""
        entry = _gateway("gpt-6-sol", ("gpt-6-astra", "gpt-6-sol", "gpt-6-luna"))
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve({"provider": "openai"}, {"openai": entry})
        assert spec.alias_models == dict.fromkeys(TIER_MODEL_ENV_VARS, "gpt-6-sol")
        assert spec.alias_origin == dict.fromkeys(TIER_MODEL_ENV_VARS, "main model")
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        records = [r.getMessage() for r in caplog.records if "main model" in r.getMessage()]
        assert len(records) == 1
        text = records[0]
        assert "haiku, sonnet, opus aliases run the main model gpt-6-sol" in text
        assert "serves no Claude models" in text
        assert "claude_code.aliases" in text

    def test_a_partial_family_names_only_the_missing_alias(self, caplog):
        entry = _gateway("claude-sonnet-5", ("claude-sonnet-5", "claude-opus-5"))
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve({"provider": "gw"}, {"gw": entry})
        assert spec.alias_models["haiku"] == "claude-sonnet-5"
        assert spec.alias_origin["haiku"] == "main model"
        assert spec.alias_origin["opus"] == "derived"
        assert "haiku alias runs the main model" in caplog.text

    def test_catalog_aliases_beat_derivation(self):
        entry = _gateway(
            "claude-sonnet-5",
            ALS_APG_SERVED,
            claude_code_aliases={"opus": "claude-fable-5-1"},
        )
        spec = ClaudeCodeModelResolver.resolve({"provider": "gw"}, {"gw": entry})
        assert spec.alias_models["opus"] == "claude-fable-5-1"
        assert spec.alias_origin["opus"] == "catalog"
        assert spec.alias_origin["sonnet"] == "derived"

    def test_deployment_aliases_beat_the_catalog(self):
        entry = _gateway(
            "claude-sonnet-5",
            ALS_APG_SERVED,
            claude_code_aliases={"opus": "claude-fable-5-1"},
        )
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "gw", "aliases": {"opus": "claude-opus-5"}}, {"gw": entry}
        )
        assert spec.alias_models["opus"] == "claude-opus-5"
        assert spec.alias_origin["opus"] == "claude_code.aliases"

    def test_the_alias_env_vars_carry_the_alias_models(self):
        entry = _gateway("gpt-6-sol", ("gpt-6-sol",))
        spec = ClaudeCodeModelResolver.resolve({"provider": "gw"}, {"gw": entry})
        for alias, var in TIER_MODEL_ENV_VARS.items():
            assert spec.env_block[var] == spec.alias_models[alias]


class TestServedListReaders:
    """What a resolved spec says about the served list, for the verbs to print."""

    def test_the_substitution_is_read_off_the_spec(self):
        entry = _gateway("gpt-6-sol", ("gpt-6-astra", "gpt-6-sol"))
        spec = ClaudeCodeModelResolver.resolve({"provider": "openai"}, {"openai": entry})
        assert alias_substitution(spec) == (
            "Claude Code's haiku, sonnet, opus aliases run the main model gpt-6-sol: "
            "'openai' serves no Claude models."
        )

    def test_a_partial_family_names_the_one_alias(self):
        entry = _gateway("claude-sonnet-5", ("claude-sonnet-5", "claude-opus-5"))
        spec = ClaudeCodeModelResolver.resolve({"provider": "gw"}, {"gw": entry})
        assert alias_substitution(spec) == (
            "Claude Code's haiku alias runs the main model claude-sonnet-5: "
            "'gw' serves no model of that family."
        )

    def test_a_full_family_has_nothing_to_say(self):
        entry = _gateway("claude-sonnet-5", ALS_APG_SERVED)
        spec = ClaudeCodeModelResolver.resolve({"provider": "gw"}, {"gw": entry})
        assert alias_substitution(spec) is None
        assert unserved_model_ids(spec) == []

    def test_configured_ids_outside_the_list_are_named_once_each(self):
        entry = _gateway("gpt-6-sol", ("gpt-6-astra", "gpt-6-sol"))
        spec = ClaudeCodeModelResolver.resolve(
            {
                "provider": "openai",
                "aliases": {"opus": "gpt-7"},
                "agent_models": {
                    "a": "claude-sonnet-5",
                    "b": "claude-sonnet-5",
                    "c": "gpt-6-astra",
                },
            },
            {"openai": entry},
        )
        assert unserved_model_ids(spec) == ["claude-sonnet-5", "gpt-7"]

    def test_a_provider_that_lists_nothing_has_no_list_to_be_outside_of(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "gw", "default_model": "x", "agent_models": {"a": "y"}},
            {"gw": {"base_url": "https://gateway.example.org/v1"}},
        )
        assert unserved_model_ids(spec) == []


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


class TestAgentTemplatesNameAModelId:
    """Each framework agent's frontmatter names a model id, never an alias word.

    The template asks the spec for the agent's model, and its literal fallback —
    rendered only when no spec is passed — is a full id the packaged direct
    Anthropic entry serves.
    """

    AGENTS_DIR = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "osprey"
        / "templates"
        / "claude_code"
        / "claude"
        / "agents"
    )
    PATTERN = re.compile(r'model: \{\{.*agent_model\("(?P<name>[^"]+)"\).*?else "(?P<id>[^"]+)"')

    def test_every_framework_agent_template_asks_for_its_own_model(self):
        from osprey.registry.mcp import FRAMEWORK_AGENTS

        seen = {}
        for path in sorted(self.AGENTS_DIR.glob("*.md.j2")):
            match = self.PATTERN.search(path.read_text())
            assert match, f"{path.name} declares no model line with a fallback id"
            assert match.group("name") == path.name.removesuffix(".md.j2")
            seen[match.group("name")] = match.group("id")
        assert set(seen) == set(FRAMEWORK_AGENTS)

    def test_each_template_fallback_is_a_served_anthropic_id(self):
        served = load_provider_catalog(None).entries["anthropic"]["models"]
        for path in sorted(self.AGENTS_DIR.glob("*.md.j2")):
            match = self.PATTERN.search(path.read_text())
            assert match.group("id") in served, path.name
            assert match.group("id") not in TIER_MODEL_ENV_VARS


class TestEnvBlockAliasModels:
    """Env block contains ANTHROPIC_DEFAULT_*_MODEL vars for all providers."""

    def test_anthropic_has_all_alias_model_vars(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.env_block["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-haiku-4-5"
        assert spec.env_block["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-5"
        assert spec.env_block["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-opus-5-5"

    def test_cborg_has_all_alias_model_vars(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.env_block["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-haiku-4-5"
        assert spec.env_block["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-5"
        assert spec.env_block["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-opus-5"

    def test_custom_alias_override_propagates_to_env_block(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "aliases": {"sonnet": "anthropic/claude-sonnet-v2"}}
        )
        assert spec.env_block["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "anthropic/claude-sonnet-v2"
        # Others unchanged
        assert spec.env_block["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-haiku-4-5"
        assert spec.env_block["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-opus-5"

    def test_all_three_vars_always_present(self):
        for provider_name in CLAUDE_CODE_PROVIDERS:
            spec = _resolve_builtin(provider_name)
            for var in (
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
            ):
                assert var in spec.env_block, f"{var} missing for {provider_name}"


class TestDefaultModel:
    """claude_code.default_model: a model id, or the entry's default_model."""

    def test_cborg_defaults_to_its_catalog_default(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.default_model_id == "claude-haiku-4-5"
        assert spec.env_block["ANTHROPIC_MODEL"] == "claude-haiku-4-5"

    def test_anthropic_defaults_to_sonnet_5(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
        assert spec.default_model_id == "claude-sonnet-5"

    def test_a_served_id_is_the_main_model(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "default_model": "claude-sonnet-5"}
        )
        assert spec.default_model_id == "claude-sonnet-5"
        assert spec.env_block["ANTHROPIC_MODEL"] == "claude-sonnet-5"

    def test_a_bare_alias_word_is_refused_naming_the_served_ids(self):
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "als-apg", "default_model": "sonnet"})
        message = str(excinfo.value)
        assert "`claude_code.default_model: sonnet` is not a model id" in message
        assert (
            "Provider 'als-apg' serves: claude-fable-5-1, claude-opus-5-5, claude-sonnet-5, "
            "claude-haiku-4-5-20251001." in message
        )

    def test_an_unserved_id_is_trusted(self, caplog):
        """Full coverage lives in test_default_model_three_branch.py."""
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": "gpt-4"})
        assert spec.env_block["ANTHROPIC_MODEL"] == "gpt-4"
        assert spec.default_model_id == "gpt-4"
        assert "trusting the gateway" in caplog.text


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
            spec = _resolve_builtin(provider_name)
            assert "ANTHROPIC_API_KEY" not in spec.env_block
            assert "ANTHROPIC_AUTH_TOKEN" not in spec.env_block


class TestModelSpecFrozen:
    """ClaudeCodeModelSpec is immutable."""

    def test_cannot_set_attributes(self):
        spec = ClaudeCodeModelSpec(provider="test", default_model_id="m")
        with pytest.raises(AttributeError):
            spec.provider = "other"


class TestInjectProviderEnv:
    """inject_provider_env() scrubs, injects env block, and wires auth."""

    def test_scrubs_managed_vars(self):
        env = {"ANTHROPIC_BASE_URL": "stale", "ANTHROPIC_MODEL": "stale", "HOME": "/home"}
        spec = ClaudeCodeModelSpec(provider="test", default_model_id="m", env_block={})
        inject_provider_env(env, spec)
        assert "ANTHROPIC_BASE_URL" not in env
        assert "ANTHROPIC_MODEL" not in env
        assert env["HOME"] == "/home"

    def test_injects_env_block(self):
        env = {}
        spec = ClaudeCodeModelSpec(
            provider="test",
            default_model_id="m",
            env_block={"ANTHROPIC_BASE_URL": "https://proxy.example.com", "ANTHROPIC_MODEL": "m"},
        )
        inject_provider_env(env, spec)
        assert env["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"
        assert env["ANTHROPIC_MODEL"] == "m"

    def test_injects_auth(self):
        env = {"CBORG_API_KEY": "secret-123"}
        spec = ClaudeCodeModelSpec(
            provider="cborg",
            default_model_id="m",
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
            default_model_id="m",
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
            default_model_id="m",
            env_block={"ANTHROPIC_MODEL": "m", "ANTHROPIC_BASE_URL": "u"},
        )
        result = inject_provider_env(env, spec)
        assert result == ["ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"]


ARGO_CONFIG = """\
api:
  providers:
    argo:
      base_url: ${ARGO_PROD_URL}
      default_model: claudesonnet45
      models: [claudehaiku45, claudesonnet45, claudeopus41]
claude_code:
  provider: argo
"""

CBORG_CONFIG = """\
api:
  providers:
    cborg: {}
claude_code:
  provider: cborg
"""


def _write_project(tmp_path, config_text, env_text=None):
    (tmp_path / "config.yml").write_text(config_text)
    if env_text is not None:
        (tmp_path / ".env").write_text(env_text)
    return tmp_path


class TestLoadProviderSpec:
    """load_provider_spec() reads config.yml and expands ${VAR} before resolving."""

    def test_expands_custom_base_url_from_dotenv(self, tmp_path, monkeypatch):
        """${VAR} in a custom provider base_url is expanded from the project .env."""
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.delenv("ARGO_PROD_URL", raising=False)
        proj = _write_project(tmp_path, ARGO_CONFIG, "ARGO_PROD_URL=https://argo.example/v1\n")

        spec = load_provider_spec(proj)

        assert spec is not None
        assert spec.needs_proxy is True
        # Claude-Code-facing var is stripped of the OpenAI /v1 (issue #312)…
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://argo.example"
        # …while the proxy upstream keeps it (proxy appends /chat/completions).
        assert spec.upstream_base_url == "https://argo.example/v1"

    def test_expands_from_os_environ_when_no_dotenv(self, tmp_path, monkeypatch):
        """${VAR} also resolves from os.environ when there is no .env."""
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.setenv("ARGO_PROD_URL", "https://argo.from-env/v1")
        proj = _write_project(tmp_path, ARGO_CONFIG)

        spec = load_provider_spec(proj)

        # /v1 stripped for the Claude-Code-facing var; upstream retains it.
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://argo.from-env"
        assert spec.upstream_base_url == "https://argo.from-env/v1"

    def test_dotenv_overrides_os_environ(self, tmp_path, monkeypatch):
        """A project .env value wins over a stale shell export."""
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.setenv("ARGO_PROD_URL", "https://stale-shell/v1")
        proj = _write_project(tmp_path, ARGO_CONFIG, "ARGO_PROD_URL=https://fresh-dotenv/v1\n")

        spec = load_provider_spec(proj)

        # /v1 stripped for the Claude-Code-facing var; upstream retains it.
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://fresh-dotenv"
        assert spec.upstream_base_url == "https://fresh-dotenv/v1"

    def test_native_config_byte_identical(self, tmp_path):
        """A literal-URL native config resolves identically to the raw resolver."""
        from osprey.build.claude_code_resolver import load_provider_spec

        proj = _write_project(tmp_path, CBORG_CONFIG)
        loaded = load_provider_spec(proj)
        direct = ClaudeCodeModelResolver.resolve({"provider": "cborg"}, {"cborg": {}})
        assert loaded.env_block == direct.env_block

    def test_provider_override(self, tmp_path):
        """provider= overrides claude_code.provider before resolving."""
        from osprey.build.claude_code_resolver import load_provider_spec

        proj = _write_project(tmp_path, CBORG_CONFIG)
        spec = load_provider_spec(proj, provider="anthropic")
        assert spec.provider == "anthropic"

    def test_returns_none_when_no_provider(self, tmp_path):
        from osprey.build.claude_code_resolver import load_provider_spec

        proj = _write_project(tmp_path, "api:\n  providers: {}\n")
        assert load_provider_spec(proj) is None

    def test_does_not_mutate_os_environ(self, tmp_path, monkeypatch):
        """Resolving against the .env overlay must not leak into os.environ."""
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.delenv("ARGO_PROD_URL", raising=False)
        proj = _write_project(tmp_path, ARGO_CONFIG, "ARGO_PROD_URL=https://argo.example/v1\n")

        load_provider_spec(proj)

        assert "ARGO_PROD_URL" not in os.environ


class TestBaseUrlV1Normalization:
    """ANTHROPIC_BASE_URL vs upstream_base_url /v1 handling (issue #312).

    Claude Code appends ``/v1/messages`` to ``ANTHROPIC_BASE_URL``, so that var
    must never end in ``/v1``. The proxy appends ``/chat/completions`` to
    ``upstream_base_url``, so that one must KEEP its ``/v1``. A single
    configured ``base_url`` feeds both; these tests pin the split.
    """

    def _resolve(self, base_url, *, native):
        entry = {
            "base_url": base_url,
            "default_model": "claudesonnet45",
            "models": ["claudehaiku45", "claudesonnet45", "claudeopus41"],
        }
        if native:
            entry["api_protocol"] = "anthropic"
        return ClaudeCodeModelResolver.resolve({"provider": "argo"}, {"argo": entry})

    def test_anthropic_native_strips_v1_and_skips_proxy(self):
        """The #312 case: native provider + /v1 URL → single /v1, no proxy."""
        spec = self._resolve("https://apps.inside.anl.gov/argoapi/v1", native=True)
        assert spec.needs_proxy is False
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://apps.inside.anl.gov/argoapi"
        # Claude Code appends /v1/messages → exactly one /v1.
        assert (
            spec.env_block["ANTHROPIC_BASE_URL"] + "/v1/messages"
            == "https://apps.inside.anl.gov/argoapi/v1/messages"
        )
        assert spec.upstream_base_url is None

    def test_openai_proxy_keeps_v1_on_upstream(self):
        """Proxy provider: env var stripped, upstream keeps /v1 for the proxy."""
        spec = self._resolve("https://apps.inside.anl.gov/argoapi/v1", native=False)
        assert spec.needs_proxy is True
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://apps.inside.anl.gov/argoapi"
        assert spec.upstream_base_url == "https://apps.inside.anl.gov/argoapi/v1"
        # Proxy appends /chat/completions → the /v1 must survive.
        assert (
            spec.upstream_base_url.rstrip("/") + "/chat/completions"
            == "https://apps.inside.anl.gov/argoapi/v1/chat/completions"
        )

    def test_trailing_slash_before_v1_is_stripped(self):
        spec = self._resolve("https://host/argoapi/v1/", native=True)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://host/argoapi"

    def test_url_without_v1_is_left_alone(self):
        spec = self._resolve("https://api.example.com", native=True)
        assert spec.env_block["ANTHROPIC_BASE_URL"] == "https://api.example.com"
