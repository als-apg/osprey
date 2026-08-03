"""A provider that maps no models is refused, not filled with Anthropic's IDs.

``resolve()`` used to ``setdefault`` every unmapped tier to the built-in
Anthropic direct model IDs so the env block could always be built. The cost was
a silent lie: selecting a provider that ships no ``models`` map launched the
agent asking *that* provider for ``claude-opus-4-6`` — a 404 from a strict
proxy, and a silently different model from a permissive one.

The map is now required. This module pins both halves of that contract: the
refusal is actionable (it names ``api.providers.<name>.models``, the tiers, and
the shape to write), and every provider stanza the shipped templates offer
carries a real map, so the new error can never fire on a config OSPREY itself
generated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import osprey.templates
from osprey.build.claude_code_resolver import (
    TIER_MODEL_ENV_VARS,
    ClaudeCodeModelResolver,
)

TEMPLATE_ROOT = Path(osprey.templates.__file__).parent

# Every shipped config template carrying an `api.providers` block.
SHIPPED_TEMPLATES = (
    "project/config.yml.j2",
    "apps/control_assistant/config.yml.j2",
    "apps/hello_world/config.yml.j2",
    "apps/ariel_standalone/config.yml.j2",
    "apps/channel_finder_standalone/config.yml.j2",
)


def _render(relative_path: str) -> dict:
    """Render a shipped template the way ``osprey build`` does and parse it.

    ``ChainableUndefined`` lets attribute chains on unsupplied context vars
    render empty instead of raising, so one context serves every template.
    """
    from jinja2 import ChainableUndefined, Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_ROOT)),
        undefined=ChainableUndefined,
        keep_trailing_newline=True,
    )
    rendered = env.get_template(relative_path).render(
        project_name="demo",
        facility_name="Demo Facility",
        default_provider="anthropic",
        default_model="claude-haiku-4-5-20251001",
        channel_finder_mode="in_context",
        default_pipeline="in_context",
        enable_in_context=True,
        enable_hierarchical=False,
        enable_middle_layer=False,
        channel_finder_tools=[],
        project_root="/tmp/demo",
    )
    return yaml.safe_load(rendered) or {}


def _shipped_providers(relative_path: str) -> dict:
    return _render(relative_path).get("api", {}).get("providers", {}) or {}


class TestMapLessProviderIsRefused:
    """Selecting a provider with no model map raises, with a usable message."""

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

    def test_partial_map_names_the_missing_tiers(self):
        """A half-filled map is the same failure: no tier gets borrowed IDs."""
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws"},
                api_providers={
                    "lbl-aws": {
                        "base_url": "https://proxy.example.org/v1",
                        "models": {"haiku": "custom-haiku-id"},
                    }
                },
            )
        message = str(excinfo.value)
        assert "sonnet" in message and "opus" in message
        assert "defines no models mapping" not in message

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

    def test_missing_map_beats_a_bad_default_model(self):
        """The map error fires before default_model resolution.

        Order matters: ``_resolve_default_tier`` reports the provider's model
        IDs, so running it against an empty map would report the Anthropic
        fallbacks as if the provider served them.
        """
        with pytest.raises(ValueError, match="defines no models mapping"):
            ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws", "default_model": "no-such-model"},
                api_providers=self._MAP_LESS,
            )

    def test_no_anthropic_ids_leak_into_the_message(self):
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "lbl-aws"}, api_providers=self._MAP_LESS)
        assert "claude-opus" not in str(excinfo.value)


class TestShippedTemplatesCarryRealMaps:
    """No shipped provider stanza can trip the new error."""

    @pytest.mark.parametrize("relative_path", SHIPPED_TEMPLATES)
    def test_every_provider_maps_all_tiers(self, relative_path):
        providers = _shipped_providers(relative_path)
        assert providers, f"{relative_path} declares no api.providers"
        for name, entry in providers.items():
            models = (entry or {}).get("models") or {}
            missing = [tier for tier in TIER_MODEL_ENV_VARS if tier not in models]
            assert not missing, (
                f"{relative_path}: api.providers.{name} maps no model for "
                f"{missing} — selecting it would fail to resolve."
            )

    @pytest.mark.parametrize("relative_path", SHIPPED_TEMPLATES)
    def test_every_provider_resolves(self, relative_path):
        providers = _shipped_providers(relative_path)
        for name in providers:
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": name}, providers, include_telemetry=False
            )
            assert spec is not None, f"{relative_path}: {name!r} resolved to None"
            assert set(spec.tier_to_model) == set(TIER_MODEL_ENV_VARS)
            assert spec.env_block["ANTHROPIC_MODEL"] == spec.tier_to_model[spec.default_model_tier]

    @pytest.mark.parametrize("relative_path", SHIPPED_TEMPLATES)
    def test_no_provider_borrows_another_providers_ids(self, relative_path):
        """The map must be the provider's own naming, not Anthropic's.

        Anthropic-direct IDs under a non-Anthropic, non-proxy-to-Anthropic
        stanza are the exact residue the removed fallback used to manufacture.
        """
        anthropic_only = {"gpt", "gemini", "mistral", "deepseek"}
        for name, entry in _shipped_providers(relative_path).items():
            models = (entry or {}).get("models") or {}
            families = {
                family
                for family in anthropic_only
                for model_id in models.values()
                if family in model_id
            }
            if families:
                assert not any("claude" in model_id for model_id in models.values()), (
                    f"{relative_path}: api.providers.{name} mixes Claude IDs into a "
                    f"{sorted(families)} provider map."
                )
