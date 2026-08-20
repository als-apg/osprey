"""``claude_code.default_model`` resolves in four branches — never refusing.

The key used to accept only tier aliases and silently substitute the provider's
default tier for anything else, so a config naming Opus could run Haiku with no
log line. Resolution now has exactly four outcomes: a canonical tier, a model
ID the effective provider map serves, unset (the provider's default tier), or
any other model ID — which passes through verbatim as ``ANTHROPIC_MODEL``, so a
newly released ID or gateway-only alias is usable before the tier map names it.
"""

from pathlib import Path

import pytest
import yaml

from osprey.build.claude_code_resolver import (
    CLAUDE_CODE_PROVIDERS,
    ClaudeCodeModelResolver,
)

PRESET_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"

# A provider whose model IDs exist only in config — proves branch 2 consults the
# *effective* map (built-in table + api.providers + claude_code.models), not the
# built-in table alone.
CUSTOM_PROVIDERS = {
    "lbl-aws": {
        "base_url": "https://proxy.example.org/v1",
        "models": {
            "haiku": "custom-haiku-id",
            "sonnet": "custom-sonnet-id",
            "opus": "custom-opus-id",
        },
    }
}


class TestBranchOneTierName:
    """A canonical tier name selects that tier."""

    @pytest.mark.parametrize("tier", ["haiku", "sonnet", "opus"])
    def test_tier_selects_that_tier(self, tier):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": tier})
        assert spec.default_model_tier == tier
        assert spec.env_block["ANTHROPIC_MODEL"] == CLAUDE_CODE_PROVIDERS["cborg"]["models"][tier]

    def test_unset_falls_back_to_provider_default_tier(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.default_model_tier == "haiku"


class TestBranchTwoExplicitModelId:
    """A model ID the provider serves reaches ANTHROPIC_MODEL verbatim."""

    def test_builtin_provider_model_id_passes_through(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "default_model": "claude-opus-4-7"}
        )
        assert spec.env_block["ANTHROPIC_MODEL"] == "claude-opus-4-7"

    def test_model_id_from_api_providers_map_is_accepted(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws", "default_model": "custom-sonnet-id"},
            api_providers=CUSTOM_PROVIDERS,
        )
        assert spec.env_block["ANTHROPIC_MODEL"] == "custom-sonnet-id"

    def test_model_id_from_claude_code_models_override_is_accepted(self):
        spec = ClaudeCodeModelResolver.resolve(
            {
                "provider": "cborg",
                "models": {"opus": "claude-opus-4-8-preview"},
                "default_model": "claude-opus-4-8-preview",
            }
        )
        assert spec.env_block["ANTHROPIC_MODEL"] == "claude-opus-4-8-preview"

    def test_default_model_tier_stays_a_tier(self):
        """Consumers index tier_to_model[default_model_tier] — it must not hold an ID.

        ``channel_finder_in_context/server_context.py`` and
        ``channel_finder/benchmarks/sdk.py`` both do exactly this lookup.
        """
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "default_model": "claude-opus-4-7"}
        )
        assert spec.default_model_tier == "opus"
        assert spec.tier_to_model[spec.default_model_tier] == "claude-opus-4-7"


class TestBranchFourFreeFormPassThrough:
    """Any other model ID passes through verbatim — the provider is trusted."""

    def test_unknown_value_passes_through(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": "gpt-4"})
        assert spec.env_block["ANTHROPIC_MODEL"] == "gpt-4"
        assert spec.default_model_id == "gpt-4"
        # The tier map is untouched: only the default is free-form.
        assert spec.tier_to_model == CLAUDE_CODE_PROVIDERS["cborg"]["models"]

    def test_default_model_tier_stays_a_valid_key(self):
        """Consumers index tier_to_model[default_model_tier] — it must not KeyError."""
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": "gpt-4"})
        assert spec.default_model_tier == "haiku"  # cborg's own default tier
        assert spec.default_model_tier in spec.tier_to_model

    def test_model_id_of_a_different_provider_passes_through(self):
        """Formerly the common real-world mis-resolution; now trusted verbatim,
        so a just-released Anthropic ID is usable before the map names it."""
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "anthropic", "default_model": "claude-opus-4-7"}
        )
        assert spec.env_block["ANTHROPIC_MODEL"] == "claude-opus-4-7"

    def test_custom_provider_free_form_id(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws", "default_model": "brand-new-id"},
            api_providers=CUSTOM_PROVIDERS,
        )
        assert spec.env_block["ANTHROPIC_MODEL"] == "brand-new-id"
        assert spec.default_model_id == "brand-new-id"

    def test_mapped_ids_do_not_set_default_model_id(self):
        """Branches 1-3 leave default_model_id None — the ID travels via the tier."""
        for cc_config in (
            {"provider": "cborg"},
            {"provider": "cborg", "default_model": "sonnet"},
            {"provider": "cborg", "default_model": "claude-opus-4-7"},
        ):
            spec = ClaudeCodeModelResolver.resolve(cc_config)
            assert spec.default_model_id is None, cc_config


def _effective_model_and_provider(stem: str) -> tuple[str | None, str | None]:
    """Walk a preset's ``extends`` chain for the model/provider it renders with."""
    model = provider = None
    seen: set[str] = set()
    while stem and stem not in seen:
        seen.add(stem)
        preset = yaml.safe_load((PRESET_DIR / f"{stem}.yml").read_text()) or {}
        model = model or preset.get("model")
        provider = provider or preset.get("provider")
        stem = preset.get("extends")
    return model, provider


class TestShippedPresetsResolve:
    """Every bundled preset ships a value the resolver accepts."""

    @pytest.mark.parametrize("preset_path", sorted(PRESET_DIR.glob("*.yml")), ids=lambda p: p.stem)
    def test_preset_default_model_resolves(self, preset_path):
        model, provider = _effective_model_and_provider(preset_path.stem)
        assert model, f"{preset_path.stem} resolves to no model"
        assert provider, f"{preset_path.stem} resolves to no provider"
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": provider, "default_model": model}, include_telemetry=False
        )
        assert spec.default_model_tier in spec.tier_to_model

    def test_every_shipped_preset_is_covered(self):
        """No preset may drop out of the parametrization by shipping no yml."""
        assert len(list(PRESET_DIR.glob("*.yml"))) >= 6

    def test_the_defaults_view_renders_a_resolvable_default_model(self, tmp_path):
        """``osprey config --defaults`` is the config people read to start from.

        It answers "what keys exist and what do they default to", so a default
        model tier it names that the resolver cannot map is a wrong answer at
        the one place someone is most likely to copy from.
        """
        from click.testing import CliRunner

        from osprey.cli.config_cmd import config

        result = CliRunner().invoke(config, ["--defaults"])
        assert result.exit_code == 0, result.output
        exported = yaml.safe_load(result.output)
        cc_config = exported["claude_code"]
        spec = ClaudeCodeModelResolver.resolve(
            cc_config, exported.get("api", {}).get("providers", {}), include_telemetry=False
        )
        assert spec.default_model_tier in spec.tier_to_model
