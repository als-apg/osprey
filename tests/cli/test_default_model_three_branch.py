"""``claude_code.default_model`` resolves in three branches.

Unset, the provider entry's ``default_model`` answers. A model id — served or
not — is the main model verbatim, so a newly released id or a gateway-only
alias is usable before the catalog lists it. A bare ``haiku``/``sonnet``/``opus``
is Claude Code's alias word, not a model id, and is refused with the ids the
provider serves.
"""

import logging
from pathlib import Path

import pytest
import yaml

from osprey.build.claude_code_resolver import ClaudeCodeModelResolver
from osprey.profiles.providers import load_provider_catalog

PRESET_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"

# A provider whose model ids exist only in config — proves the served list is
# read from api.providers, not from a built-in table.
CUSTOM_PROVIDERS = {
    "lbl-aws": {
        "base_url": "https://proxy.example.org/v1",
        "default_model": "custom-sonnet-id",
        "models": ["custom-haiku-id", "custom-sonnet-id", "custom-opus-id"],
    }
}


class TestBranchOneUnset:
    """Unset, the provider entry's default_model is the main model."""

    def test_builtin_provider_uses_its_catalog_default(self):
        spec = ClaudeCodeModelResolver.resolve({"provider": "cborg"})
        assert spec.default_model_id == "claude-haiku-4-5"
        assert spec.env_block["ANTHROPIC_MODEL"] == "claude-haiku-4-5"

    def test_custom_provider_uses_its_entry_default(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "lbl-aws"}, api_providers=CUSTOM_PROVIDERS
        )
        assert spec.default_model_id == "custom-sonnet-id"


class TestBranchTwoModelId:
    """A model id reaches ANTHROPIC_MODEL verbatim, served or not."""

    def test_served_id_passes_through(self):
        spec = ClaudeCodeModelResolver.resolve(
            {"provider": "cborg", "default_model": "claude-opus-5"}
        )
        assert spec.env_block["ANTHROPIC_MODEL"] == "claude-opus-5"
        assert spec.default_model_id == "claude-opus-5"

    def test_id_from_the_api_providers_list_is_accepted(self, caplog):
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws", "default_model": "custom-opus-id"},
                api_providers=CUSTOM_PROVIDERS,
            )
        assert spec.env_block["ANTHROPIC_MODEL"] == "custom-opus-id"
        assert "trusting the gateway" not in caplog.text

    @pytest.mark.parametrize("model_id", ["claude-opus-4-8-preview", "gpt-4", "vendor/some-model"])
    def test_an_unserved_id_is_trusted_and_logged(self, model_id, caplog):
        with caplog.at_level(logging.INFO, logger="osprey.build.claude_code_resolver"):
            spec = ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": model_id})
        assert spec.env_block["ANTHROPIC_MODEL"] == model_id
        assert spec.default_model_id == model_id
        assert f"{model_id!r} is not in the served list of provider 'cborg'" in caplog.text
        assert "trusting the gateway" in caplog.text


class TestBranchThreeBareAliasWord:
    """A bare alias word is refused with the ids the provider serves."""

    @pytest.mark.parametrize("word", ["haiku", "sonnet", "opus"])
    def test_refused_naming_the_served_ids(self, word):
        with pytest.raises(ValueError) as excinfo:
            ClaudeCodeModelResolver.resolve({"provider": "cborg", "default_model": word})
        message = str(excinfo.value)
        assert f"`claude_code.default_model: {word}` is not a model id" in message
        assert (
            "Provider 'cborg' serves: claude-opus-5, claude-sonnet-5, claude-haiku-4-5." in message
        )

    def test_refused_for_a_custom_provider_too(self):
        with pytest.raises(ValueError, match="custom-haiku-id, custom-sonnet-id, custom-opus-id"):
            ClaudeCodeModelResolver.resolve(
                {"provider": "lbl-aws", "default_model": "haiku"},
                api_providers=CUSTOM_PROVIDERS,
            )


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
    """Every bundled preset ships a provider and model the resolver accepts."""

    @pytest.mark.parametrize("preset_path", sorted(PRESET_DIR.glob("*.yml")), ids=lambda p: p.stem)
    def test_preset_default_model_resolves(self, preset_path):
        model, provider = _effective_model_and_provider(preset_path.stem)
        assert provider, f"{preset_path.stem} resolves to no provider"
        cc_config = {"provider": provider}
        if model:
            cc_config["default_model"] = model
        spec = ClaudeCodeModelResolver.resolve(
            cc_config, load_provider_catalog(None).entries, include_telemetry=False
        )
        assert spec.default_model_id in spec.served_models

    def test_every_shipped_preset_is_covered(self):
        """No preset may drop out of the parametrization by shipping no yml."""
        assert len(list(PRESET_DIR.glob("*.yml"))) >= 6
