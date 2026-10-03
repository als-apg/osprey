"""Tests for Ollama provider adapter."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from osprey.models.providers.ollama import OllamaProviderAdapter


class SampleOutput(BaseModel):
    """Sample output model for testing."""

    result: str
    value: int


class TestOllamaMetadata:
    """Test Ollama provider metadata."""

    def test_provider_name(self):
        """Test provider name is set correctly."""
        assert OllamaProviderAdapter.name == "ollama"

    def test_provider_description(self):
        """Test provider has description."""
        assert "ollama" in OllamaProviderAdapter.description.lower()

    def test_requires_api_key(self):
        """Test provider does not require API key."""
        assert OllamaProviderAdapter.requires_api_key is False

    def test_requires_base_url(self):
        """Test provider requires base URL."""
        assert OllamaProviderAdapter.requires_base_url is True

    def test_requires_model_id(self):
        """Test provider requires model ID."""
        assert OllamaProviderAdapter.requires_model_id is True

    def test_supports_proxy(self):
        """Test provider does not support HTTP proxy."""
        assert OllamaProviderAdapter.supports_proxy is False

    def test_has_default_base_url(self):
        """Test provider has localhost default."""
        assert OllamaProviderAdapter.default_base_url is not None
        assert "localhost" in OllamaProviderAdapter.default_base_url

    def test_has_default_model_id(self):
        """Test provider has default model."""
        assert OllamaProviderAdapter.default_model_id is not None

    def test_has_health_check_model(self):
        """Test provider has health check model."""
        assert OllamaProviderAdapter.health_check_model_id is not None

    def test_api_key_note(self):
        """Test provider notes no API key needed."""
        assert OllamaProviderAdapter.api_key_note is not None
        assert "local" in OllamaProviderAdapter.api_key_note.lower()


class TestOllamaFallbackUrls:
    """Test Ollama URL fallback logic."""

    def test_get_fallback_urls_from_container(self):
        """Test fallback URLs when running in container."""
        urls = OllamaProviderAdapter._get_fallback_urls("http://host.containers.internal:11434")
        assert "localhost" in str(urls)
        assert "http://localhost:11434" in urls

    def test_get_fallback_urls_from_localhost(self):
        """Test fallback URLs when running on localhost."""
        urls = OllamaProviderAdapter._get_fallback_urls("http://localhost:11434")
        assert "host.containers.internal" in str(urls)

    def test_get_fallback_urls_generic(self):
        """Test fallback URLs for generic base URL."""
        urls = OllamaProviderAdapter._get_fallback_urls("http://custom.server:11434")
        assert len(urls) > 0
        assert "localhost" in str(urls) or "host.containers.internal" in str(urls)


class TestOllamaTestConnection:
    """Test Ollama connection testing."""

    def test_test_connection_success(self):
        """Test successful connection test."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = OllamaProviderAdapter._test_connection("http://localhost:11434")
            assert result is True

    def test_test_connection_failure(self):
        """Test failed connection test."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            result = OllamaProviderAdapter._test_connection("http://localhost:11434")
            assert result is False

    def test_test_connection_exception(self):
        """Test connection test handles exceptions."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            result = OllamaProviderAdapter._test_connection("http://localhost:11434")
            assert result is False


class TestOllamaExecuteCompletion:
    """Test Ollama completion execution via direct API."""

    @patch("httpx.post")
    @patch.object(OllamaProviderAdapter, "_test_connection", return_value=True)
    def test_execute_text_completion(self, _mock_test, mock_post):
        """Test basic text completion via direct Ollama API."""
        provider = OllamaProviderAdapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Test response"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = provider.execute_completion(
            message="Hello",
            model_id="mistral:7b",
            api_key=None,
            base_url="http://localhost:11434",
        )

        assert result == "Test response"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"
        assert call_args[1]["json"]["model"] == "mistral:7b"

    @patch("httpx.post")
    @patch.object(OllamaProviderAdapter, "_test_connection", return_value=True)
    def test_execute_structured_output(self, _mock_test, mock_post):
        """output_format routes through the direct structured-output path,
        requesting format=json and validating the response into the model."""
        provider = OllamaProviderAdapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": '{"result": "ok", "value": 42}'}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = provider.execute_completion(
            message="Extract",
            model_id="mistral:7b",
            api_key=None,
            base_url="http://localhost:11434",
            output_format=SampleOutput,
        )

        assert isinstance(result, SampleOutput)
        assert result == SampleOutput(result="ok", value=42)
        # The structured path must ask Ollama for JSON-formatted output.
        assert mock_post.call_args[1]["json"]["format"] == "json"

    @patch("httpx.post")
    @patch.object(OllamaProviderAdapter, "_test_connection", return_value=True)
    def test_execute_structured_output_invalid_json_raises(self, _mock_test, mock_post):
        """Unparseable content surfaces as a ValueError rather than propagating a
        raw pydantic error or returning garbage."""
        provider = OllamaProviderAdapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "not valid json"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to parse structured output"):
            provider.execute_completion(
                message="Extract",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                output_format=SampleOutput,
            )

    @patch("httpx.post")
    @patch.object(OllamaProviderAdapter, "_test_connection", return_value=True)
    def test_execute_structured_output_asks_once_more_through_the_adapter(
        self, _mock_test, mock_post
    ):
        """A reply that does not parse is asked for once more through the adapter."""
        provider = OllamaProviderAdapter()

        def reply(content):
            response = MagicMock()
            response.json.return_value = {"message": {"content": content}}
            response.raise_for_status = MagicMock()
            return response

        mock_post.side_effect = [reply("not valid json"), reply('{"result": "ok", "value": 1}')]

        result = provider.execute_completion(
            message="Extract",
            model_id="mistral:7b",
            api_key=None,
            base_url="http://localhost:11434",
            output_format=SampleOutput,
        )

        assert result == SampleOutput(result="ok", value=1)
        assert mock_post.call_count == 2

    @patch("httpx.post")
    @patch.object(
        OllamaProviderAdapter,
        "_test_connection",
        side_effect=[False, True],  # First fails, second succeeds
    )
    def test_execute_completion_with_fallback(self, _mock_test, mock_post):
        """Test completion execution with fallback."""
        provider = OllamaProviderAdapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Response"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = provider.execute_completion(
            message="Hello",
            model_id="mistral:7b",
            api_key=None,
            base_url="http://localhost:11434",
        )

        assert result == "Response"
        # The primary localhost URL failed (_test_connection -> False), so the
        # request must target the resolved fallback host, not the original URL.
        # Asserting the URL is what distinguishes this from the no-fallback path.
        assert mock_post.call_args[0][0] == "http://host.containers.internal:11434/api/chat"

    @patch.object(OllamaProviderAdapter, "_test_connection", return_value=False)
    def test_execute_completion_all_connections_fail(self, _mock_test):
        """Test completion fails when all connections fail."""
        provider = OllamaProviderAdapter()

        with pytest.raises(ValueError, match="Failed to connect"):
            provider.execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
            )


class TestOllamaHealthCheck:
    """Test Ollama health check functionality."""

    def test_health_check_without_base_url_probes_the_declared_default(self):
        """No configured URL means "the declared default", not "unconfigured".

        The health check has to probe whatever a completion would actually reach.
        It used to report "Base URL not configured" while ``execute_completion``
        for the same config went to ``default_base_url`` — two answers to one
        question.
        """
        provider = OllamaProviderAdapter()

        with patch.object(
            OllamaProviderAdapter, "_test_connection", return_value=True
        ) as mock_test:
            success, message = provider.check_health(api_key=None, base_url=None)

        assert success is True
        assert mock_test.call_args[0][0] == OllamaProviderAdapter.default_base_url
        assert OllamaProviderAdapter.default_base_url in message

    def test_health_check_success(self):
        """Test successful health check."""
        provider = OllamaProviderAdapter()

        with patch.object(OllamaProviderAdapter, "_test_connection", return_value=True):
            success, message = provider.check_health(
                api_key=None, base_url="http://localhost:11434"
            )
            assert success is True
            assert "accessible" in message.lower()

    def test_health_check_failure(self):
        """Test failed health check."""
        provider = OllamaProviderAdapter()

        with patch.object(OllamaProviderAdapter, "_test_connection", return_value=False):
            success, message = provider.check_health(
                api_key=None, base_url="http://localhost:11434"
            )
            assert success is False
            assert "not accessible" in message.lower()

    def test_health_check_exception(self):
        """Test health check handles exceptions."""
        provider = OllamaProviderAdapter()

        with patch.object(
            OllamaProviderAdapter, "_test_connection", side_effect=Exception("Test error")
        ):
            success, message = provider.check_health(
                api_key=None, base_url="http://localhost:11434"
            )
            assert success is False
            assert "failed" in message.lower()


LS_PROBE = "osprey.models.providers._local_server.probe"


class TestOllamaEmbedding:
    """The embedding endpoint of the unified Ollama adapter."""

    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

    def test_embedding_defaults(self):
        """Both embedding model defaults are nomic-embed-text."""
        assert OllamaProviderAdapter.default_embedding_model_id == "nomic-embed-text"
        assert OllamaProviderAdapter.health_check_embedding_model_id == "nomic-embed-text"
        assert OllamaProviderAdapter.fallback_probe_path == "/api/tags"

    def test_serves_chat_and_embeddings(self):
        """The adapter serves chat and text embeddings."""
        assert OllamaProviderAdapter.supports_chat()
        assert OllamaProviderAdapter.supports_embeddings()

    def test_empty_texts_short_circuits(self):
        """Empty input returns [] before any probe."""
        with patch(LS_PROBE) as mock_probe:
            assert OllamaProviderAdapter().execute_embedding(texts=[], model_id="x") == []
        mock_probe.assert_not_called()

    def test_execute_embedding_builds_litellm_call(self):
        """Model is 'ollama/<id>', the resolved URL is api_base, vectors come from data."""
        with (
            patch(LS_PROBE, return_value=True),
            patch("litellm.embedding") as mock_embed,
        ):
            mock_embed.return_value = MagicMock(data=[{"embedding": [0.5, 0.6]}])
            result = OllamaProviderAdapter().execute_embedding(
                texts=["hi"], model_id="nomic-embed-text", base_url="http://localhost:11434"
            )
        assert result == [[0.5, 0.6]]
        kwargs = mock_embed.call_args[1]
        assert kwargs["model"] == "ollama/nomic-embed-text"
        assert kwargs["api_base"] == "http://localhost:11434"
        assert "dimensions" not in kwargs

    def test_execute_embedding_forwards_dimensions(self):
        """dimensions is forwarded when set."""
        with patch(LS_PROBE, return_value=True), patch("litellm.embedding") as mock_embed:
            mock_embed.return_value = MagicMock(data=[{"embedding": [0.1]}])
            OllamaProviderAdapter().execute_embedding(texts=["a"], model_id="m", dimensions=1)
        assert mock_embed.call_args[1]["dimensions"] == 1

    def test_embedding_falls_back_with_probe_path(self):
        """Primary down -> the docker fallback is used, probed on /api/tags."""
        with (
            patch(LS_PROBE, side_effect=[False, True]) as mock_probe,
            patch("litellm.embedding") as mock_embed,
        ):
            mock_embed.return_value = MagicMock(data=[{"embedding": [1.0]}])
            OllamaProviderAdapter().execute_embedding(
                texts=["a"], model_id="m", base_url="http://localhost:11434"
            )
        assert mock_embed.call_args[1]["api_base"] == "http://host.docker.internal:11434"
        assert all(call.args[1] == "/api/tags" for call in mock_probe.call_args_list)

    def test_embedding_uses_ollama_host(self, monkeypatch):
        """OLLAMA_HOST is tried first on the embedding path."""
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
        with patch(LS_PROBE, return_value=True), patch("litellm.embedding") as mock_embed:
            mock_embed.return_value = MagicMock(data=[{"embedding": [1.0]}])
            OllamaProviderAdapter().execute_embedding(texts=["a"], model_id="m")
        assert mock_embed.call_args[1]["api_base"] == "http://ollama:11434"

    def test_embedding_unreachable_raises_runtime_error(self):
        """No answering server raises a RuntimeError naming Ollama."""
        with patch(LS_PROBE, return_value=False):
            with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
                OllamaProviderAdapter().execute_embedding(texts=["a"], model_id="m")

    def test_embedding_request_failure_wrapped(self):
        """A failing litellm call surfaces as RuntimeError with the cause chained."""
        with (
            patch(LS_PROBE, return_value=True),
            patch("litellm.embedding", side_effect=ValueError("bad")),
        ):
            with pytest.raises(RuntimeError, match="Failed to generate embeddings") as info:
                OllamaProviderAdapter().execute_embedding(texts=["a"], model_id="m")
        assert isinstance(info.value.__cause__, ValueError)

    def test_chat_path_ignores_ollama_host(self, monkeypatch):
        """OLLAMA_HOST does not reach the chat resolution."""
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
        with patch.object(OllamaProviderAdapter, "_test_connection", return_value=True) as conn:
            resolved = OllamaProviderAdapter()._resolve_base_url("http://localhost:11434")
        assert resolved == "http://localhost:11434"
        conn.assert_called_once_with("http://localhost:11434", timeout=2.0)


class TestOllamaEmbeddingHealth:
    """check_embedding_health verdicts."""

    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

    @staticmethod
    def _tags(*names):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"models": [{"name": n} for n in names]}
        return resp

    def test_healthy_with_default_model(self):
        """model_id=None checks nomic-embed-text and answers healthy."""
        with (
            patch(LS_PROBE, return_value=True),
            patch("requests.get", return_value=self._tags("nomic-embed-text:latest")),
        ):
            result = OllamaProviderAdapter().check_embedding_health(
                api_key=None, base_url="http://localhost:11434", model_id=None
            )
        assert result.reachable is True
        assert result.reason is None
        assert "connected" in result.message

    def test_model_not_pulled(self):
        """A model not pulled answers reason 'model'."""
        with (
            patch(LS_PROBE, return_value=True),
            patch("requests.get", return_value=self._tags("other-model:latest")),
        ):
            result = OllamaProviderAdapter().check_embedding_health(
                api_key=None, base_url="http://localhost:11434", model_id=None
            )
        assert result.reachable is False
        assert result.reason == "model"
        assert "ollama pull nomic-embed-text" in result.message

    def test_unreachable(self):
        """No answering candidate answers reason 'unreachable'."""
        with patch(LS_PROBE, return_value=False):
            result = OllamaProviderAdapter().check_embedding_health(
                api_key=None, base_url="http://localhost:11434"
            )
        assert result.reachable is False
        assert result.reason == "unreachable"
        assert result.message == "Cannot connect to Ollama at http://localhost:11434"

    def test_request_failure_classified(self):
        """A 404 on the tags request is classified by failure_reason."""
        import requests

        resp = MagicMock(status_code=404)
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
        with patch(LS_PROBE, return_value=True), patch("requests.get", return_value=resp):
            result = OllamaProviderAdapter().check_embedding_health(
                api_key=None, base_url="http://localhost:11434"
            )
        assert result.reachable is False
        assert result.reason == "model"

    def test_unclassified_failure_is_unreachable(self):
        """A failure failure_reason does not know answers 'unreachable'."""
        with (
            patch(LS_PROBE, return_value=True),
            patch("requests.get", side_effect=ValueError("garbled")),
        ):
            result = OllamaProviderAdapter().check_embedding_health(
                api_key=None, base_url="http://localhost:11434"
            )
        assert result.reachable is False
        assert result.reason == "unreachable"


class TestOllamaCompletionTimeout:
    """A caller's timeout bounds both the base-URL probes and the request."""

    @staticmethod
    def _reply():
        response = MagicMock()
        response.json.return_value = {"message": {"content": "ok"}}
        response.raise_for_status = MagicMock()
        return response

    @patch("httpx.post")
    def test_refusing_primary_probes_each_candidate_within_the_timeout(self, mock_post):
        """With timeout=0.5 each probe gets 0.5 s and the request gets 0.5 s."""
        mock_post.return_value = self._reply()
        with patch.object(
            OllamaProviderAdapter, "_test_connection", side_effect=[False, True]
        ) as probe:
            result = OllamaProviderAdapter().execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                timeout=0.5,
            )

        assert result == "ok"
        assert probe.call_count == 2
        assert [c.kwargs["timeout"] for c in probe.call_args_list] == [0.5, 0.5]
        assert mock_post.call_args.kwargs["timeout"] == 0.5

    @patch("httpx.post")
    def test_a_long_timeout_still_caps_each_probe_at_two_seconds(self, mock_post):
        """A timeout above two seconds leaves each probe at two."""
        mock_post.return_value = self._reply()
        with patch.object(OllamaProviderAdapter, "_test_connection", return_value=True) as probe:
            OllamaProviderAdapter().execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
                timeout=30.0,
            )

        assert probe.call_args.kwargs["timeout"] == 2.0
        assert mock_post.call_args.kwargs["timeout"] == 30.0

    @patch("httpx.post")
    def test_no_timeout_keeps_two_second_probes_and_120_second_request(self, mock_post):
        """Without a timeout the probes stay at two seconds and the request at 120."""
        mock_post.return_value = self._reply()
        with patch.object(OllamaProviderAdapter, "_test_connection", return_value=True) as probe:
            OllamaProviderAdapter().execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
            )

        assert probe.call_args.kwargs["timeout"] == 2.0
        assert mock_post.call_args.kwargs["timeout"] == 120.0

    def test_probe_passes_its_timeout_to_requests(self):
        """_test_connection bounds its GET by the timeout it is given."""
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200)
            assert OllamaProviderAdapter._test_connection("http://localhost:11434", timeout=0.5)

        assert mock_get.call_args.kwargs["timeout"] == 0.5


class TestOllamaUnreachableError:
    """A dead Ollama is both a connection error and the ValueError callers catch."""

    @patch.object(OllamaProviderAdapter, "_test_connection", return_value=False)
    def test_every_candidate_refusing_raises_connection_and_value_error(self, _mock_test):
        """With every candidate refusing, the error is both kinds, with the known message."""
        from osprey.models.providers.ollama import OllamaUnreachableError

        with pytest.raises(OllamaUnreachableError) as info:
            OllamaProviderAdapter().execute_completion(
                message="Hello",
                model_id="mistral:7b",
                api_key=None,
                base_url="http://localhost:11434",
            )

        assert isinstance(info.value, ConnectionError)
        assert isinstance(info.value, ValueError)
        assert str(info.value).startswith(
            "Failed to connect to Ollama at configured URL 'http://localhost:11434' "
            "and all fallback URLs"
        )
