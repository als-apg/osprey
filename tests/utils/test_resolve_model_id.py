"""Tests for resolve_model_id() tier-to-provider-ID resolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osprey.utils.config import resolve_model_id


class TestResolveModelId:
    """resolve_model_id maps tier names to provider-specific model IDs."""

    @pytest.fixture(autouse=True)
    def _mock_provider_config(self):
        """Provide a fake provider config for all tests."""
        configs = {
            "cborg": {
                "models": {
                    "haiku": "anthropic/claude-haiku",
                    "sonnet": "anthropic/claude-sonnet",
                    "opus": "anthropic/claude-opus",
                },
            },
            "anthropic": {
                "models": {
                    "haiku": "claude-haiku-4-5-20251001",
                    "sonnet": "claude-sonnet-4-5-20250929",
                    "opus": "claude-opus-4-6",
                },
            },
        }
        with patch(
            "osprey.utils.config.get_provider_config",
            side_effect=lambda name, config_path=None: configs.get(name, {}),
        ):
            yield

    def test_cborg_haiku_resolves(self):
        assert resolve_model_id("cborg", "haiku") == "anthropic/claude-haiku"

    def test_cborg_sonnet_resolves(self):
        assert resolve_model_id("cborg", "sonnet") == "anthropic/claude-sonnet"

    def test_cborg_opus_resolves(self):
        assert resolve_model_id("cborg", "opus") == "anthropic/claude-opus"

    def test_anthropic_haiku_resolves(self):
        assert resolve_model_id("anthropic", "haiku") == "claude-haiku-4-5-20251001"

    def test_non_tier_passes_through(self):
        assert resolve_model_id("cborg", "anthropic/claude-haiku") == "anthropic/claude-haiku"

    def test_unknown_provider_returns_tier_unchanged(self):
        assert resolve_model_id("unknown-provider", "haiku") == "haiku"

    def test_tier_not_in_provider_models_returns_unchanged(self):
        """Provider exists but has no models mapping for the tier."""
        with patch(
            "osprey.utils.config.get_provider_config",
            return_value={"models": {}},
        ):
            assert resolve_model_id("cborg", "haiku") == "haiku"
