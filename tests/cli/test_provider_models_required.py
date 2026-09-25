"""A provider names what it serves, and never borrows another provider's ids.

A provider entry lists the model ids its gateway serves and names its
``default_model``. An entry that does neither — and a deployment that names no
``claude_code.default_model`` either — has no model to run, and the build
refuses with the keys to write. Claude Code's own alias names are filled from
the provider's served list; an alias nothing fills runs the main model, with a
warning naming the substitution, and is never filled with Anthropic's direct
ids: a proxy asked for ``claude-opus-5`` answers 404 when strict and a silently
different model when not.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from osprey.build.claude_code_resolver import (
    TIER_MODEL_ENV_VARS,
    ClaudeCodeModelResolver,
)
from osprey.profiles.providers import packaged_catalog_path

#: Catalog entries whose gateway fronts more than one vendor. Each lists a
#: Claude id beside another vendor's ids, every one in the gateway's spelling.
_MULTI_VENDOR_GATEWAYS = frozenset({"als-apg"})


def _shipped_providers() -> dict:
    """The provider stanzas a build renders into ``api.providers``.

    One packaged catalog answers for every deployment: ``providers.yml`` ships
    beside the presets, ``osprey init`` writes it into the deployment, and the
    build renders it into ``api.providers``.
    """
    catalog = yaml.safe_load(packaged_catalog_path().read_text(encoding="utf-8")) or {}
    return catalog.get("providers") or {}


class TestModelLessProviderIsRefused:
    """No models, no default model and no configured one raises, with a usable message."""

    _MODEL_LESS = {"lbl-aws": {"base_url": "https://proxy.example.org/v1"}}

    def test_no_models_raises(self):
        with pytest.raises(ValueError, match="lists no models and names no default_model"):
            ClaudeCodeModelResolver.resolve({"provider": "lbl-aws"}, api_providers=self._MODEL_LESS)

    def test_error_names_the_keys_to_write(self):
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "lbl-aws"}, api_providers=self._MODEL_LESS)
        message = str(excinfo.value)
        assert "api.providers.lbl-aws" in message
        assert "`models:`" in message
        assert "`default_model:`" in message

    def test_a_configured_default_model_is_enough(self, caplog):
        """The minimal custom-gateway config: ``provider:`` and ``model: <id>``.

        With no served list, every Claude Code alias runs that model, and the
        resolver's record names the substitution.
        """
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws", "default_model": "gateway-model-id"},
                api_providers=self._MODEL_LESS,
            )
        assert spec.env_block["ANTHROPIC_MODEL"] == "gateway-model-id"
        assert spec.alias_models == dict.fromkeys(TIER_MODEL_ENV_VARS, "gateway-model-id")
        message = "\n".join(record.getMessage() for record in caplog.records)
        assert "haiku, sonnet, opus aliases run the main model gateway-model-id" in message

    def test_no_anthropic_ids_leak_into_the_message(self):
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "lbl-aws"}, api_providers=self._MODEL_LESS)
        assert "claude-opus" not in str(excinfo.value)


class TestAliasSubstitutionIsLoud:
    """An alias no source fills runs the main model, and the build says so."""

    def test_a_partial_family_is_recorded_and_falls_back(self, caplog):
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws"},
                api_providers={
                    "lbl-aws": {
                        "base_url": "https://proxy.example.org/v1",
                        "default_model": "claude-haiku-4-5",
                        "models": ["claude-haiku-4-5"],
                    }
                },
            )
        assert spec.alias_models == dict.fromkeys(TIER_MODEL_ENV_VARS, "claude-haiku-4-5")
        message = "\n".join(record.getMessage() for record in caplog.records)
        assert "sonnet, opus aliases run the main model claude-haiku-4-5" in message
        assert "claude-opus" not in message  # no Anthropic ids borrowed or named

    def test_claude_code_aliases_can_complete_the_set(self, caplog):
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {
                    "provider": "lbl-aws",
                    "default_model": "x-sonnet",
                    "aliases": {"haiku": "x-haiku", "sonnet": "x-sonnet", "opus": "x-opus"},
                },
                api_providers={"lbl-aws": {"base_url": "https://proxy.example.org/v1"}},
            )
        assert spec.alias_models == {"haiku": "x-haiku", "sonnet": "x-sonnet", "opus": "x-opus"}
        assert not caplog.records

    def test_a_key_that_is_not_an_alias_name_is_named(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            ClaudeCodeModelResolver.resolve({"provider": "cborg", "aliases": {"opusx": "some-id"}})
        message = "\n".join(record.getMessage() for record in caplog.records)
        assert "claude_code.aliases" in message
        assert "opusx" in message

    def test_a_catalog_alias_key_that_is_not_an_alias_name_is_named(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            ClaudeCodeModelResolver.resolve(
                {"provider": "gw"},
                api_providers={
                    "gw": {
                        "base_url": "https://gw/v1",
                        "default_model": "claude-sonnet-5",
                        "models": ["claude-sonnet-5"],
                        "claude_code_aliases": {"sonet": "claude-sonnet-5"},
                    }
                },
            )
        message = "\n".join(record.getMessage() for record in caplog.records)
        assert "api.providers.gw.claude_code_aliases" in message
        assert "sonet" in message

    def test_a_claude_gateway_warns_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert not caplog.records


class TestTheShippedCatalogResolves:
    """No stanza in the packaged provider catalog can trip the refusal."""

    def test_every_provider_lists_models_and_names_a_default_it_serves(self):
        providers = _shipped_providers()
        assert providers, "the packaged catalog declares no providers"
        for name, entry in providers.items():
            models = entry.get("models")
            assert isinstance(models, list) and models, name
            assert entry.get("default_model") in models, name

    def test_every_provider_resolves_to_its_own_default(self):
        providers = _shipped_providers()
        for name, entry in providers.items():
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": name}, providers, include_telemetry=False
            )
            assert spec is not None, f"providers.yml: {name!r} resolved to None"
            assert spec.default_model_id == entry["default_model"]
            assert spec.env_block["ANTHROPIC_MODEL"] == entry["default_model"]
            assert set(spec.alias_models) == set(TIER_MODEL_ENV_VARS)
            for model_id in spec.alias_models.values():
                assert model_id in entry["models"], (name, model_id)

    def test_no_provider_borrows_another_providers_ids(self):
        """The list must be the provider's own naming, not Anthropic's.

        A gateway that fronts several vendors is named in
        ``_MULTI_VENDOR_GATEWAYS`` and must list more than one vendor.
        """
        providers = _shipped_providers()
        assert _MULTI_VENDOR_GATEWAYS <= set(providers)
        other_families = {"gpt", "gemini", "mistral", "deepseek"}
        for name, entry in providers.items():
            models = entry.get("models") or []
            families = {
                family for family in other_families for model_id in models if family in model_id
            }
            if name in _MULTI_VENDOR_GATEWAYS:
                assert families and any("claude" in model_id for model_id in models), (
                    f"providers.yml: {name} is named a multi-vendor gateway but lists one vendor."
                )
                continue
            if families:
                assert not any("claude" in model_id for model_id in models), (
                    f"providers.yml: {name} mixes Claude ids into a {sorted(families)} provider."
                )
