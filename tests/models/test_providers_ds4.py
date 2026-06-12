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

    @pytest.mark.unit
    def test_check_health_builds_models_url_without_mangling(self, monkeypatch):
        """removesuffix('/v1') must strip only the literal suffix, not characters.

        Uses base_url 'http://host:8001/v1' as a discriminating input: the old
        rstrip('/v1') would also strip the trailing '1' of '8001' and the '/',
        mangling it to 'http://host:800', whereas removesuffix('/v1') correctly
        yields 'http://host:8001'.
        """
        captured = {}

        class _Resp:
            status_code = 200

            def json(self):
                return {"data": [{"id": "deepseek-v4-flash"}]}

        def fake_get(url, timeout=None):
            captured["url"] = url
            return _Resp()

        import httpx

        import osprey.models.providers.ds4 as ds4mod

        monkeypatch.setattr(httpx, "get", fake_get)
        # Short-circuit the downstream litellm health call.
        monkeypatch.setattr(ds4mod, "check_litellm_health", lambda **kw: (True, "ok"))

        DS4ProviderAdapter().check_health(api_key="EMPTY", base_url="http://host:8001/v1")
        assert captured["url"] == "http://host:8001/v1/models"
