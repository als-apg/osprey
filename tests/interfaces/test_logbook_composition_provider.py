"""Logbook composition resolves its provider from config, never from a hardcoded default.

The provider comes from ``logbook.composition.provider`` or the project's
``claude_code.provider``; there is no built-in one, because a wrong provider
silently bills the wrong account. The model is the id the operator picked in the
compose panel — refused unless the provider serves it, because that value comes
from a browser — else ``logbook.composition.model``, else the deployment's main
model. Every gap raises naming the key to fill in.

The shipped presets are pinned here too: they name no composition model and no
provider of their own, so the deployment's main model answers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.interfaces.artifacts.logbook import _resolve_composition_model

_PROVIDER = {
    "api_key": "test-key",
    "default_model": "proxy-haiku-4-5",
    "models": ["proxy-opus-5", "proxy-sonnet-5", "proxy-haiku-4-5"],
}


def _resolve(config: dict[str, Any], providers: dict[str, dict], model: str | None = None):
    """Resolve with *config* as the dotted-path config store and *providers* as api.providers."""

    def fake_get_config_value(path: str, default: Any = None, _config_path: str | None = None):
        return config.get(path, default)

    def fake_get_provider_config(name: str, _config_path: str | None = None):
        return providers.get(name, {})

    with (
        patch("osprey.utils.config.get_config_value", fake_get_config_value),
        patch("osprey.models.config.get_provider_config", fake_get_provider_config),
    ):
        return _resolve_composition_model(model)


class TestProviderResolution:
    def test_explicit_composition_provider_wins(self):
        provider, model_id = _resolve(
            {
                "logbook.composition": {"provider": "cborg"},
                "claude_code.provider": "als-apg",
            },
            {"cborg": _PROVIDER},
        )

        assert provider == "cborg"
        assert model_id == "proxy-haiku-4-5"

    def test_falls_back_to_configured_claude_code_provider(self):
        # The whole point of the fix: no composition provider means "use what the
        # project is already configured against", not "use anthropic".
        provider, model_id = _resolve(
            {
                "logbook.composition": {"model": "proxy-sonnet-5"},
                "claude_code.provider": "cborg",
            },
            {"cborg": _PROVIDER},
        )

        assert provider == "cborg"
        assert model_id == "proxy-sonnet-5"

    def test_missing_composition_section_still_uses_configured_provider(self):
        provider, _ = _resolve({"claude_code.provider": "cborg"}, {"cborg": _PROVIDER})

        assert provider == "cborg"

    def test_no_provider_anywhere_raises_naming_the_key(self):
        # An anthropic entry is available to be picked up; nothing may pick it up.
        with pytest.raises(HTTPException) as exc:
            _resolve({}, {"anthropic": _PROVIDER})

        assert exc.value.status_code == 503
        assert "logbook.composition.provider" in exc.value.detail
        assert "claude_code.provider" in exc.value.detail

    def test_blank_provider_is_treated_as_unset(self):
        provider, _ = _resolve(
            {"logbook.composition": {"provider": ""}, "claude_code.provider": "cborg"},
            {"cborg": _PROVIDER},
        )

        assert provider == "cborg"

    def test_unknown_provider_raises_naming_api_providers(self):
        with pytest.raises(HTTPException) as exc:
            _resolve(
                {"logbook.composition": {"provider": "not-declared"}},
                {"cborg": _PROVIDER},
            )

        assert exc.value.status_code == 503
        assert "api.providers" in exc.value.detail


class TestModelIdResolution:
    def test_the_panel_choice_wins(self):
        _, model_id = _resolve(
            {"logbook.composition": {"provider": "cborg", "model": "proxy-haiku-4-5"}},
            {"cborg": _PROVIDER},
            model="proxy-opus-5",
        )

        assert model_id == "proxy-opus-5"

    @pytest.mark.parametrize("picked", ["gpt-6-sol", "opus"])
    def test_a_panel_choice_the_provider_does_not_serve_is_refused(self, picked):
        with pytest.raises(HTTPException) as exc:
            _resolve({"logbook.composition": {"provider": "cborg"}}, {"cborg": _PROVIDER}, picked)

        assert exc.value.status_code == 400
        assert picked in exc.value.detail
        assert "proxy-opus-5, proxy-sonnet-5, proxy-haiku-4-5" in exc.value.detail

    def test_the_composition_model_answers_when_the_panel_names_none(self):
        _, model_id = _resolve(
            {"logbook.composition": {"provider": "cborg", "model": "proxy-sonnet-5"}},
            {"cborg": _PROVIDER},
        )

        assert model_id == "proxy-sonnet-5"

    def test_the_deployment_main_model_answers_last(self):
        _, model_id = _resolve(
            {
                "claude_code.provider": "cborg",
                "claude_code.default_model": "proxy-opus-5",
            },
            {"cborg": _PROVIDER},
        )

        assert model_id == "proxy-opus-5"

    def test_the_provider_default_answers_when_the_deployment_names_no_model(self):
        _, model_id = _resolve({"claude_code.provider": "cborg"}, {"cborg": _PROVIDER})

        assert model_id == "proxy-haiku-4-5"

    def test_no_model_anywhere_raises_naming_the_keys(self):
        with pytest.raises(HTTPException) as exc:
            _resolve(
                {"logbook.composition": {"provider": "cborg"}},
                {"cborg": {"api_key": "k", "models": []}},
            )

        assert exc.value.status_code == 503
        assert "logbook.composition.model" in exc.value.detail
        assert "api.providers.cborg.default_model" in exc.value.detail


#: The bundled presets that ship a logbook, and so a composition block.
_PRESETS_WITH_LOGBOOK = ["control-assistant", "ariel-standalone"]


def _profile(preset: str):
    profile, _profile_dir = resolve_build_profile(None, preset)
    return profile


def _composition_block(preset: str) -> dict[str, Any]:
    return _expand_dotted(_profile(preset).config).get("logbook", {}).get("composition", {})


class TestShippedPresets:
    @pytest.mark.parametrize("preset", _PRESETS_WITH_LOGBOOK)
    def test_preset_carries_no_model_id(self, preset: str):
        assert "model_id" not in _composition_block(preset)

    @pytest.mark.parametrize("preset", _PRESETS_WITH_LOGBOOK)
    def test_preset_names_no_composition_model_or_provider(self, preset: str):
        """The deployment's main model answers; the provider is the profile's.

        A preset that pinned ``logbook.composition.provider`` would state the
        provider twice, and ``osprey set provider=...`` would move only one of
        them; a pinned composition model would go stale on the same switch.
        """
        block = _composition_block(preset)

        assert "model" not in block
        assert "default_tier" not in block
        assert "provider" not in block
