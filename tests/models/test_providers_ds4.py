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
        # ds4 accepts but ignores response_format json_schema -> must NOT use the
        # native path. The behavioral consequence (this flag actually routing ds4
        # to the prompt fallback) is verified in
        # test_litellm_adapter.py::TestStructuredOutputCapabilityFlag::test_ds4_declares_false_end_to_end.
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

        def fake_health(**kw):
            captured["health_kwargs"] = kw
            return True, "ok"

        import httpx

        import osprey.models.providers.ds4 as ds4mod

        monkeypatch.setattr(httpx, "get", fake_get)
        # Capture the downstream litellm health call's kwargs and short-circuit it.
        monkeypatch.setattr(ds4mod, "check_litellm_health", fake_health)

        result = DS4ProviderAdapter().check_health(api_key="EMPTY", base_url="http://host:8001/v1")
        assert captured["url"] == "http://host:8001/v1/models"
        # The discovered model_id must be forwarded to the litellm health check.
        assert captured["health_kwargs"]["model_id"] == "deepseek-v4-flash"
        # check_health must return check_litellm_health's result verbatim.
        assert result == (True, "ok")

    @pytest.mark.unit
    def test_check_health_no_models_loaded(self, monkeypatch):
        """200 with empty data short-circuits before the litellm health call."""

        class _Resp:
            status_code = 200

            def json(self):
                return {"data": []}

        def fake_get(url, timeout=None):
            return _Resp()

        import httpx

        monkeypatch.setattr(httpx, "get", fake_get)

        result = DS4ProviderAdapter().check_health(api_key="EMPTY", base_url="http://host:8001/v1")
        assert result == (False, "ds4 server running but no models loaded")

    @pytest.mark.unit
    def test_check_health_non_200(self, monkeypatch):
        """A non-200 response reports the status code and stops."""

        class _Resp:
            status_code = 503

            def json(self):
                return {}

        def fake_get(url, timeout=None):
            return _Resp()

        import httpx

        monkeypatch.setattr(httpx, "get", fake_get)

        result = DS4ProviderAdapter().check_health(api_key="EMPTY", base_url="http://host:8001/v1")
        assert result == (False, "ds4 server returned 503")

    @pytest.mark.unit
    def test_check_health_connect_error(self, monkeypatch):
        """A refused connection yields the connect-specific diagnostic (the most
        likely real failure for a local inference server) and never reaches the
        litellm health call."""
        import httpx

        def fake_get(url, timeout=None):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "get", fake_get)

        result = DS4ProviderAdapter().check_health(api_key="EMPTY", base_url="http://host:8001/v1")
        assert result == (False, "Cannot connect to ds4 server at http://host:8001/v1")

    @pytest.mark.unit
    def test_check_health_query_error(self, monkeypatch):
        """A non-connect error querying the server is wrapped in the generic
        'Error querying ds4' message rather than propagating."""
        import httpx

        def fake_get(url, timeout=None):
            raise ValueError("boom")

        monkeypatch.setattr(httpx, "get", fake_get)

        result = DS4ProviderAdapter().check_health(api_key="EMPTY", base_url="http://host:8001/v1")
        assert result == (False, "Error querying ds4: boom")

    @pytest.mark.unit
    def test_check_health_model_without_id(self, monkeypatch):
        """A model entry lacking an 'id' leaves model_id unset, so the post-query
        guard reports no usable model instead of forwarding None to litellm."""

        class _Resp:
            status_code = 200

            def json(self):
                return {"data": [{"object": "model"}]}  # no "id" key

        def fake_get(url, timeout=None):
            return _Resp()

        import httpx

        monkeypatch.setattr(httpx, "get", fake_get)

        result = DS4ProviderAdapter().check_health(api_key="EMPTY", base_url="http://host:8001/v1")
        assert result == (False, "No model available for health check")
