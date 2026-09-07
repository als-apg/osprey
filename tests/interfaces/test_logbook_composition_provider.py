"""Logbook composition resolves its provider from config, never from a hardcoded default.

The compose panel used to fall back to a built-in ``anthropic`` provider and a
built-in ``model_id`` of ``"haiku"`` — a tier name, not a model ID. A project
configured against a proxy would then quietly bill a different account, or send
a literal ``"haiku"`` upstream. Both built-ins are gone: the provider comes from
``logbook.composition.provider`` or the project's ``claude_code.provider``, the
model ID comes from the provider's tier mapping, and every gap raises naming the
key to fill in.

The shipped presets are pinned here too — they carry no ``model_id``, because
``default_tier`` plus the provider catalog's tier mapping already determines it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.interfaces.artifacts.logbook import _resolve_composition_model
from osprey.profiles.providers import load_provider_catalog

_PROVIDER_WITH_TIERS = {
    "api_key": "test-key",
    "models": {
        "haiku": "proxy/claude-haiku",
        "sonnet": "proxy/claude-sonnet",
        "opus": "proxy/claude-opus",
    },
}


def _resolve(config: dict[str, Any], providers: dict[str, dict], model: str | None = None):
    """Resolve with *config* as the dotted-path config store and *providers* as api.providers."""

    def fake_get_config_value(path: str, default: Any = None, config_path: str | None = None):
        return config.get(path, default)

    def fake_get_provider_config(name: str, config_path: str | None = None):
        return providers.get(name, {})

    with (
        patch("osprey.utils.config.get_config_value", fake_get_config_value),
        patch("osprey.models.config.get_provider_config", fake_get_provider_config),
    ):
        return _resolve_composition_model(model)


class TestProviderResolution:
    @pytest.mark.unit
    def test_explicit_composition_provider_wins(self):
        provider, model_id = _resolve(
            {
                "logbook.composition": {"provider": "cborg", "default_tier": "haiku"},
                "claude_code.provider": "als-apg",
            },
            {"cborg": _PROVIDER_WITH_TIERS},
        )

        assert provider == "cborg"
        assert model_id == "proxy/claude-haiku"

    @pytest.mark.unit
    def test_falls_back_to_configured_claude_code_provider(self):
        # The whole point of the fix: no composition provider means "use what the
        # project is already configured against", not "use anthropic".
        provider, model_id = _resolve(
            {
                "logbook.composition": {"default_tier": "sonnet"},
                "claude_code.provider": "cborg",
            },
            {"cborg": _PROVIDER_WITH_TIERS},
        )

        assert provider == "cborg"
        assert model_id == "proxy/claude-sonnet"

    @pytest.mark.unit
    def test_missing_composition_section_still_uses_configured_provider(self):
        provider, _ = _resolve({"claude_code.provider": "cborg"}, {"cborg": _PROVIDER_WITH_TIERS})

        assert provider == "cborg"

    @pytest.mark.unit
    def test_no_provider_anywhere_raises_naming_the_key(self):
        # An anthropic entry is available to be picked up; nothing may pick it up.
        with pytest.raises(HTTPException) as exc:
            _resolve({}, {"anthropic": _PROVIDER_WITH_TIERS})

        assert exc.value.status_code == 503
        assert "logbook.composition.provider" in exc.value.detail
        assert "claude_code.provider" in exc.value.detail

    @pytest.mark.unit
    def test_blank_provider_is_treated_as_unset(self):
        provider, _ = _resolve(
            {"logbook.composition": {"provider": ""}, "claude_code.provider": "cborg"},
            {"cborg": _PROVIDER_WITH_TIERS},
        )

        assert provider == "cborg"

    @pytest.mark.unit
    def test_unknown_provider_raises_naming_api_providers(self):
        with pytest.raises(HTTPException) as exc:
            _resolve(
                {"logbook.composition": {"provider": "not-declared"}},
                {"cborg": _PROVIDER_WITH_TIERS},
            )

        assert exc.value.status_code == 503
        assert "api.providers" in exc.value.detail


class TestModelIdResolution:
    @pytest.mark.unit
    def test_ui_tier_overrides_default_tier(self):
        _, model_id = _resolve(
            {"logbook.composition": {"provider": "cborg", "default_tier": "haiku"}},
            {"cborg": _PROVIDER_WITH_TIERS},
            model="opus",
        )

        assert model_id == "proxy/claude-opus"

    @pytest.mark.unit
    def test_tier_mapping_wins_over_pinned_model_id(self):
        _, model_id = _resolve(
            {
                "logbook.composition": {
                    "provider": "cborg",
                    "default_tier": "haiku",
                    "model_id": "pinned/model",
                }
            },
            {"cborg": _PROVIDER_WITH_TIERS},
        )

        assert model_id == "proxy/claude-haiku"

    @pytest.mark.unit
    def test_pinned_model_id_covers_an_unmapped_tier(self):
        _, model_id = _resolve(
            {
                "logbook.composition": {
                    "provider": "cborg",
                    "default_tier": "opus",
                    "model_id": "pinned/model",
                }
            },
            {"cborg": {"models": {"haiku": "proxy/claude-haiku"}}},
        )

        assert model_id == "pinned/model"

    @pytest.mark.unit
    def test_unmapped_tier_without_pin_raises_instead_of_sending_a_tier_name(self):
        with pytest.raises(HTTPException) as exc:
            _resolve(
                {"logbook.composition": {"provider": "cborg", "default_tier": "opus"}},
                {"cborg": {"models": {"haiku": "proxy/claude-haiku"}}},
            )

        assert exc.value.status_code == 503
        assert "opus" in exc.value.detail
        assert "api.providers.cborg.models" in exc.value.detail


#: The bundled presets that ship a logbook, and so a composition block.
_PRESETS_WITH_LOGBOOK = ["control-assistant", "ariel-standalone"]


def _profile(preset: str):
    profile, _profile_dir = resolve_build_profile(None, preset)
    return profile


def _composition_block(preset: str) -> dict[str, Any]:
    return _expand_dotted(_profile(preset).config)["logbook"]["composition"]


class TestShippedPresets:
    @pytest.mark.unit
    @pytest.mark.parametrize("preset", _PRESETS_WITH_LOGBOOK)
    def test_preset_carries_no_model_id(self, preset: str):
        assert "model_id" not in _composition_block(preset)

    @pytest.mark.unit
    @pytest.mark.parametrize("preset", _PRESETS_WITH_LOGBOOK)
    def test_preset_states_a_tier_and_the_profile_states_the_provider(self, preset: str):
        """The composition block names a tier; the provider is the profile's.

        A preset that pinned ``logbook.composition.provider`` would state the
        provider twice, and ``osprey set provider=...`` would move only one of
        them.
        """
        block = _composition_block(preset)

        assert block["default_tier"] == "haiku"
        assert "provider" not in block
        assert _profile(preset).provider == "anthropic"

    @pytest.mark.unit
    @pytest.mark.parametrize("preset", _PRESETS_WITH_LOGBOOK)
    def test_shipped_tier_is_mapped_by_the_shipped_provider(self, preset: str):
        # A preset that ships a tier its own provider cannot map would fail at
        # compose time; the pairing is only honest if it resolves.
        profile = _profile(preset)
        tier = _composition_block(preset)["default_tier"]
        entry = load_provider_catalog(None).entries[profile.provider]

        assert tier in entry.get("models", {})
