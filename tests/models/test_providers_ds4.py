"""Unit tests for the ds4 (DwarfStar local DeepSeek V4) provider."""
import pytest

from osprey.models.providers.ds4 import DS4ProviderAdapter


class TestDS4Provider:
    @pytest.mark.unit
    def test_metadata(self):
        assert DS4ProviderAdapter.name == "ds4"
        assert DS4ProviderAdapter.is_openai_compatible is True
        assert DS4ProviderAdapter.requires_api_key is False
        assert DS4ProviderAdapter.default_base_url == "http://127.0.0.1:8000/v1"

    @pytest.mark.unit
    def test_uses_prompt_fallback_for_structured_output(self):
        # ds4 ignores response_format json_schema -> must NOT use native path
        assert DS4ProviderAdapter.supports_native_structured_output is False

    @pytest.mark.unit
    def test_serves_deepseek_models(self):
        assert "deepseek-v4-flash" in DS4ProviderAdapter.available_models
        assert "deepseek-v4-pro" in DS4ProviderAdapter.available_models
