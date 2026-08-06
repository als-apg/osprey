"""``claude_code.default_model`` resolves in three branches — or refuses.

The key used to accept only tier aliases and silently substitute the provider's
default tier for anything else, so a config naming Opus could run Haiku with no
log line. Resolution now has exactly three outcomes: a canonical tier, a model
ID the effective provider map serves, or a ValueError that names the key, the
tiers, and the provider's model IDs.
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


class TestBranchThreeHardError:
    """Anything else refuses, actionably."""

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError) as exc:
            ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": "gpt-4"})
        message = str(exc.value)
        assert "claude_code.default_model" in message, "error must name the key"
        assert "gpt-4" in message, "error must quote the rejected value"
        for tier in ("haiku", "sonnet", "opus"):
            assert tier in message, f"error must list the valid tier {tier}"
        for model_id in CLAUDE_CODE_PROVIDERS["cborg"]["models"].values():
            assert model_id in message, f"error must list the provider's model ID {model_id}"

    def test_model_id_of_a_different_provider_raises(self):
        """The old silent fallback made this the common real-world mis-resolution."""
        with pytest.raises(ValueError, match="claude_code.default_model"):
            ClaudeCodeModelResolver.resolve(
                {"provider": "anthropic", "default_model": "claude-opus-4-7"}
            )

    def test_error_names_the_configured_provider(self):
        with pytest.raises(ValueError) as exc:
            ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws", "default_model": "claude-opus-4-7"},
                api_providers=CUSTOM_PROVIDERS,
            )
        message = str(exc.value)
        assert "lbl-aws" in message
        assert "custom-opus-id" in message


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

    def test_config_export_renders_a_resolvable_default_model(self, tmp_path):
        """``osprey config export`` is the canonical starting config people copy."""
        from click.testing import CliRunner

        from osprey.cli.config_cmd import export

        out = tmp_path / "exported.yml"
        result = CliRunner().invoke(export, ["--output", str(out)])
        assert result.exit_code == 0, result.output
        exported = yaml.safe_load(out.read_text())
        cc_config = exported["claude_code"]
        spec = ClaudeCodeModelResolver.resolve(
            cc_config, exported.get("api", {}).get("providers", {}), include_telemetry=False
        )
        assert spec.default_model_tier in spec.tier_to_model
