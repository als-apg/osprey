"""An unmapped tier falls back to the default model — loudly, never borrowed.

``setdefault``-ing every unmapped tier to the built-in Anthropic direct model
IDs, so the env block can always be built, costs a silent lie: selecting a
provider that ships no ``models`` map would launch the agent asking *that*
provider for ``claude-opus-4-6`` — a 404 from a strict proxy, and a silently
different model from a permissive one.

So a missing tier is never filled with another provider's IDs. It falls back
to the resolved default model instead, with a warning that names each
substitution — the build proceeds, and nothing is silent. Refusal remains only
for the case with nothing to fall back to: no map and no default model at all,
and that refusal is actionable (it names ``api.providers.<name>.models``, the
tiers, and the shape to write). Every provider stanza the shipped templates
offer carries a real map, so neither the warning nor the refusal can fire on a
config OSPREY itself generated.
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


def _shipped_providers() -> dict:
    """The provider stanzas a build renders into ``api.providers``.

    One packaged catalog answers for every deployment now: ``providers.yml``
    ships beside the presets, ``osprey init`` writes it into the deployment,
    and the build renders it into ``api.providers``. No config template carries
    a provider stanza of its own, so there is one source to check rather than
    one per app.
    """
    catalog = yaml.safe_load(packaged_catalog_path().read_text(encoding="utf-8")) or {}
    return catalog.get("providers") or {}


class TestMapLessProviderIsRefused:
    """No map and no default model raises, with a usable message.

    Anything short of that — a partial map, or no map but a free-form default
    model — builds with a loud fallback warning instead (tests below).
    """

    _MAP_LESS = {"lbl-aws": {"base_url": "https://proxy.example.org/v1"}}

    def test_no_models_raises(self):
        with pytest.raises(ValueError, match="defines no models mapping"):
            ClaudeCodeModelResolver.resolve({"provider": "lbl-aws"}, api_providers=self._MAP_LESS)

    def test_error_names_the_config_key_and_the_tiers(self):
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "lbl-aws"}, api_providers=self._MAP_LESS)
        message = str(excinfo.value)
        assert "api.providers.lbl-aws.models" in message
        for tier in TIER_MODEL_ENV_VARS:
            assert tier in message
        # The message shows the shape to write, not just the key name.
        assert "models:" in message
        assert "haiku:" in message

    def test_partial_map_warns_and_falls_back(self, caplog):
        """A half-filled map builds, but each missing tier is named — loudly.

        No tier gets borrowed IDs: the fallback is the provider's own resolved
        default model, and the warning names every substitution (#350/#357).
        """
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws"},
                api_providers={
                    "lbl-aws": {
                        "base_url": "https://proxy.example.org/v1",
                        "models": {"haiku": "custom-haiku-id"},
                    }
                },
            )
        assert spec.tier_to_model == {
            "haiku": "custom-haiku-id",
            "sonnet": "custom-haiku-id",
            "opus": "custom-haiku-id",
        }
        message = "\n".join(record.getMessage() for record in caplog.records)
        assert "sonnet -> custom-haiku-id" in message
        assert "opus -> custom-haiku-id" in message
        assert "claude-opus" not in message  # no Anthropic IDs borrowed or named

    def test_claude_code_models_can_complete_the_map(self):
        """The per-tier override is a valid way to supply the missing IDs."""
        spec = ClaudeCodeModelResolver.resolve(
            {
                "provider": "lbl-aws",
                "models": {
                    "haiku": "x-haiku",
                    "sonnet": "x-sonnet",
                    "opus": "x-opus",
                },
            },
            api_providers=self._MAP_LESS,
        )
        assert spec.tier_to_model == {
            "haiku": "x-haiku",
            "sonnet": "x-sonnet",
            "opus": "x-opus",
        }

    def test_free_form_default_model_fills_a_missing_map(self, caplog):
        """A map-less provider plus a free-form default model builds.

        This is the minimal custom-gateway config: ``provider: my-gateway``
        and ``model: <id>`` with no tier map at all. Every tier falls back to
        the configured model, and the warning names all three substitutions.
        """
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws", "default_model": "gateway-model-id"},
                api_providers=self._MAP_LESS,
            )
        assert spec.env_block["ANTHROPIC_MODEL"] == "gateway-model-id"
        assert spec.tier_to_model == {
            "haiku": "gateway-model-id",
            "sonnet": "gateway-model-id",
            "opus": "gateway-model-id",
        }
        message = "\n".join(record.getMessage() for record in caplog.records)
        for tier in TIER_MODEL_ENV_VARS:
            assert f"{tier} -> gateway-model-id" in message

    def test_no_anthropic_ids_leak_into_the_message(self):
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "lbl-aws"}, api_providers=self._MAP_LESS)
        assert "claude-opus" not in str(excinfo.value)


class TestNonTierKeysWarn:
    """Non-tier keys in a ``models:`` map are dropped — but named, not silent.

    A typo like ``sonet:`` used to vanish without a trace, leaving that tier
    on its fallback with no hint why.
    """

    def test_api_providers_non_tier_keys_are_named(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws"},
                api_providers={
                    "lbl-aws": {
                        "base_url": "https://proxy.example.org/v1",
                        "models": {
                            "haiku": "custom-haiku-id",
                            "sonnet": "custom-sonnet-id",
                            "opus": "custom-opus-id",
                            "sonet": "typo-id",
                        },
                    }
                },
            )
        assert "typo-id" not in spec.tier_to_model.values()
        message = "\n".join(record.getMessage() for record in caplog.records)
        assert "api.providers.lbl-aws.models" in message
        assert "sonet" in message

    def test_claude_code_models_non_tier_keys_are_named(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            ClaudeCodeModelResolver.resolve({"provider": "cborg", "models": {"opusx": "some-id"}})
        message = "\n".join(record.getMessage() for record in caplog.records)
        assert "claude_code.models" in message
        assert "opusx" in message

    def test_valid_maps_warn_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger="osprey.build.claude_code_resolver"):
            ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert not caplog.records


class TestTheShippedCatalogCarriesRealMaps:
    """No stanza in the packaged provider catalog can trip the refusal."""

    def test_every_provider_maps_all_tiers(self):
        providers = _shipped_providers()
        assert providers, "the packaged catalog declares no providers"
        for name, entry in providers.items():
            models = (entry or {}).get("models") or {}
            missing = [tier for tier in TIER_MODEL_ENV_VARS if tier not in models]
            assert not missing, (
                f"providers.yml: {name} maps no model for "
                f"{missing} — selecting it would fail to resolve."
            )

    def test_every_provider_resolves(self):
        providers = _shipped_providers()
        for name in providers:
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": name}, providers, include_telemetry=False
            )
            assert spec is not None, f"providers.yml: {name!r} resolved to None"
            assert set(spec.tier_to_model) == set(TIER_MODEL_ENV_VARS)
            assert spec.env_block["ANTHROPIC_MODEL"] == spec.tier_to_model[spec.default_model_tier]

    def test_no_provider_borrows_another_providers_ids(self):
        """The map must be the provider's own naming, not Anthropic's.

        Anthropic-direct IDs under a non-Anthropic, non-proxy-to-Anthropic
        stanza are the exact residue the removed fallback used to manufacture.
        """
        anthropic_only = {"gpt", "gemini", "mistral", "deepseek"}
        for name, entry in _shipped_providers().items():
            models = (entry or {}).get("models") or {}
            families = {
                family
                for family in anthropic_only
                for model_id in models.values()
                if family in model_id
            }
            if families:
                assert not any("claude" in model_id for model_id in models.values()), (
                    f"providers.yml: {name} mixes Claude IDs into a "
                    f"{sorted(families)} provider map."
                )
