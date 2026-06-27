"""Unit tests for the ds4 (DwarfStar local DeepSeek V4) provider."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

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

    @pytest.mark.unit
    def test_check_health_explicit_model_id_skips_discovery(self, monkeypatch):
        """When model_id is supplied, the server-discovery httpx.get must be
        skipped entirely (the `if not model_id` block) and the given model
        forwarded straight to the litellm health check."""
        captured = {}

        def fail_get(url, timeout=None):
            raise AssertionError("httpx.get must not be called when model_id is given")

        def fake_health(**kw):
            captured["health_kwargs"] = kw
            return True, "ok"

        import httpx

        import osprey.models.providers.ds4 as ds4mod

        monkeypatch.setattr(httpx, "get", fail_get)
        monkeypatch.setattr(ds4mod, "check_litellm_health", fake_health)

        result = DS4ProviderAdapter().check_health(
            api_key="EMPTY", base_url="http://host:8001/v1", model_id="deepseek-v4-pro"
        )
        assert result == (True, "ok")
        assert captured["health_kwargs"]["model_id"] == "deepseek-v4-pro"


class _Sample(BaseModel):
    name: str


class TestDS4ExecuteCompletion:
    """ds4.execute_completion delegates to the shared litellm path with ds4's
    keyless ('EMPTY') and base-url defaults applied."""

    @patch("osprey.models.providers.ds4.execute_litellm_completion")
    def test_defaults_empty_key_and_base_url(self, mock_exec):
        """No api_key + no base_url -> 'EMPTY' placeholder and the default URL."""
        mock_exec.return_value = "ok"

        result = DS4ProviderAdapter().execute_completion(
            message="hello",
            model_id="deepseek-v4-flash",
            api_key=None,
            base_url=None,
        )

        assert result == "ok"
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["provider"] == "ds4"
        assert kwargs["message"] == "hello"
        assert kwargs["model_id"] == "deepseek-v4-flash"
        assert kwargs["api_key"] == "EMPTY"
        assert kwargs["base_url"] == "http://127.0.0.1:8000/v1"
        assert kwargs["output_format"] is None

    @patch("osprey.models.providers.ds4.execute_litellm_completion")
    def test_empty_string_key_becomes_placeholder(self, mock_exec):
        """A falsy api_key (empty string) is also coerced to 'EMPTY'."""
        mock_exec.return_value = "ok"

        DS4ProviderAdapter().execute_completion(
            message="hi",
            model_id="deepseek-v4-flash",
            api_key="",
            base_url=None,
        )

        assert mock_exec.call_args.kwargs["api_key"] == "EMPTY"

    @patch("osprey.models.providers.ds4.execute_litellm_completion")
    def test_passes_through_explicit_key_and_base_url(self, mock_exec):
        """An explicit key and base_url are forwarded unchanged (not overridden
        by the defaults)."""
        mock_exec.return_value = "ok"

        DS4ProviderAdapter().execute_completion(
            message="hi",
            model_id="deepseek-v4-pro",
            api_key="sk-real",
            base_url="http://gpu-host:9000/v1",
        )

        kwargs = mock_exec.call_args.kwargs
        assert kwargs["api_key"] == "sk-real"
        assert kwargs["base_url"] == "http://gpu-host:9000/v1"

    @patch("osprey.models.providers.ds4.execute_litellm_completion")
    def test_output_format_is_forwarded(self, mock_exec):
        """output_format is passed through so the shared layer can route ds4 to
        its prompt-based JSON fallback."""
        mock_exec.return_value = _Sample(name="x")

        DS4ProviderAdapter().execute_completion(
            message="extract",
            model_id="deepseek-v4-flash",
            api_key="EMPTY",
            base_url=None,
            output_format=_Sample,
        )

        assert mock_exec.call_args.kwargs["output_format"] is _Sample
