"""Ollama Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access,
while preserving Ollama-specific fallback URL logic for development workflows.
"""

from typing import Any

from osprey.utils.logger import get_logger

from . import _local_server
from .base import BaseProvider
from .health import HealthResult, failure_reason
from .litellm_adapter import execute_litellm_completion

logger = get_logger("ollama")

#: Seconds each base-URL probe may take; a caller's shorter timeout lowers it.
_PROBE_TIMEOUT = 2.0


class OllamaUnreachableError(ConnectionError, ValueError):
    """No candidate Ollama server answered.

    A ``ConnectionError``, so an availability check reads a dead server as
    unreachable by type, and a ``ValueError``, so every caller that already
    catches the base-URL failure as a ``ValueError`` still does.
    """


class OllamaProviderAdapter(BaseProvider):
    """Ollama local model provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "ollama"
    description = "Ollama (local models)"
    requires_api_key = False
    requires_base_url = True
    requires_model_id = True
    supports_proxy = False
    default_base_url = "http://localhost:11434"
    default_model_id = "mistral:7b"  # Mistral 7B as recommended default
    health_check_model_id = "mistral:7b"  # Same for health check (local, no cost)

    # API key acquisition information
    api_key_url = None
    api_key_instructions = []
    api_key_note = "Ollama runs locally and does not require an API key"

    # Provider facts (see BaseProvider)
    api_key_env_var = None
    api_protocol = "openai"
    supports_interactive_login = False
    # Image input depends on the model each site serves, so none is assumed.
    supports_images = False
    supports_thinking = False

    # Embedding defaults
    default_embedding_model_id = "nomic-embed-text"
    health_check_embedding_model_id = "nomic-embed-text"

    # LiteLLM integration
    litellm_prefix = "ollama"

    # Embedding-path server resolution: the path a live server answers 200 on,
    # the environment override tried first, and the well-known port.
    fallback_probe_path = "/api/tags"
    host_env_var = "OLLAMA_HOST"
    default_port = 11434

    @staticmethod
    def _get_fallback_urls(base_url: str) -> list[str]:
        """Generate fallback URLs for Ollama based on the current base URL."""
        fallback_urls = []

        if "host.containers.internal" in base_url:
            # Running in container but Ollama might be on localhost
            fallback_urls = [
                base_url.replace("host.containers.internal", "localhost"),
                "http://localhost:11434",
            ]
        elif "localhost" in base_url:
            # Running locally but Ollama might be in container context
            fallback_urls = [
                base_url.replace("localhost", "host.containers.internal"),
                "http://host.containers.internal:11434",
            ]
        else:
            # Generic fallbacks for other scenarios
            fallback_urls = ["http://localhost:11434", "http://host.containers.internal:11434"]

        return fallback_urls

    @staticmethod
    def _test_connection(base_url: str, timeout: float = _PROBE_TIMEOUT) -> bool:
        """Test if Ollama is accessible at the given URL within *timeout* seconds."""
        return _local_server.probe(base_url, "/v1/models", timeout)

    def _resolve_base_url(self, base_url: str | None, probe_timeout: float = _PROBE_TIMEOUT) -> str:
        """Resolve a reachable base URL, probing container/localhost variants.

        The configured value is resolved through
        :meth:`~osprey.models.providers.base.BaseProvider.require_effective_base_url`
        first, so this method never has to interpret ``None`` — the connectivity
        fallbacks below all do substring matching, which a ``None`` breaks with an
        unhelpful ``TypeError``. Each candidate is probed for at most
        *probe_timeout* seconds.

        Raises:
            ValueError: When no base URL is configured or declared.
            OllamaUnreachableError: When no candidate answers.
        """
        base_url = self.require_effective_base_url(base_url)

        # Test primary URL first
        if self._test_connection(base_url, timeout=probe_timeout):
            logger.debug(f"Successfully connected to Ollama at {base_url}")
            return base_url

        logger.debug(f"Failed to connect to Ollama at {base_url}")

        # Try fallback URLs
        fallback_urls = self._get_fallback_urls(base_url)
        for fallback_url in fallback_urls:
            logger.debug(f"Attempting fallback connection to Ollama at {fallback_url}")
            if self._test_connection(fallback_url, timeout=probe_timeout):
                logger.warning(
                    f"Ollama connection fallback: configured URL '{base_url}' failed, "
                    f"using fallback '{fallback_url}'. Consider updating your configuration."
                )
                return fallback_url

        # All connection attempts failed
        raise OllamaUnreachableError(
            f"Failed to connect to Ollama at configured URL '{base_url}' "
            f"and all fallback URLs {fallback_urls}. Please ensure Ollama is running "
            f"and accessible, or update your configuration."
        )

    def execute_completion(
        self,
        message: str,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking: dict | None = None,  # noqa: ARG002 - provider adapter contract; adapters that support extended thinking read it
        system_prompt: str | None = None,  # noqa: ARG002 - provider adapter contract; adapters that send a system turn read it
        output_format: Any | None = None,
        **kwargs,
    ) -> str | Any:
        """Execute Ollama chat completion via LiteLLM with fallback support.

        A ``timeout`` keyword bounds the request and lowers each base-URL probe
        to ``min(2.0, timeout)``, so the whole call stays within ``timeout``
        plus at most one probe bound per candidate.
        """
        timeout = kwargs.get("timeout")
        probe_timeout = _PROBE_TIMEOUT if timeout is None else min(_PROBE_TIMEOUT, timeout)

        # Resolve working base URL with fallbacks
        effective_base_url = self._resolve_base_url(base_url, probe_timeout=probe_timeout)

        return execute_litellm_completion(
            provider=self.name,
            message=message,
            model_id=model_id,
            api_key=api_key or "ollama",  # Ollama doesn't need real key
            base_url=effective_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            output_format=output_format,
            **kwargs,
        )

    def check_health(
        self,
        api_key: str | None,  # noqa: ARG002 - provider adapter contract; adapters that authenticate read the key
        base_url: str | None,
        timeout: float = 5.0,  # noqa: ARG002 - provider adapter contract; adapters that bound the probe read the timeout
        model_id: str | None = None,  # noqa: ARG002 - provider adapter contract; adapters that probe a specific model read the model id
    ) -> tuple[bool, str]:
        """Check Ollama connectivity (no API key needed).

        Probes the same endpoint ``execute_completion`` would use, so a missing
        base_url is reported against the declared default rather than as
        "unconfigured" — the health check must not disagree with what a request
        will actually reach.
        """
        try:
            probe_url = self.require_effective_base_url(base_url)
        except ValueError as e:
            return False, str(e)

        try:
            if self._test_connection(probe_url):
                return True, f"Accessible at {probe_url}"
            else:
                return False, f"Not accessible at {probe_url}"
        except Exception as e:
            return False, f"Connection test failed: {str(e)[:50]}"

    def _resolve_embedding_base_url(self, base_url: str | None) -> str:
        """Resolve a reachable server for the embedding path.

        Tries ``OLLAMA_HOST`` first, then the configured URL, then its container
        fallbacks, probing :attr:`fallback_probe_path` on every call.

        Raises:
            ValueError: When no base URL is configured or declared.
            LocalServerUnreachable: When no candidate answers.
        """
        url = self.require_effective_base_url(base_url)
        return _local_server.resolve_local_server(
            url,
            probe_path=self.fallback_probe_path,
            env_var=self.host_env_var,
            default_port=self.default_port,
            label="Ollama",
        )

    def execute_embedding(
        self,
        texts: list[str],
        model_id: str,
        api_key: str | None = None,  # noqa: ARG002 - provider adapter contract; Ollama needs no key
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 600.0,
    ) -> list[list[float]]:
        """Embed *texts* with an Ollama model via LiteLLM.

        Raises:
            ValueError: When no base URL is configured or declared.
            LocalServerUnreachable: When no candidate server answers.
            RuntimeError: When the embedding request fails.
        """
        if not texts:
            return []

        resolved_url = self._resolve_embedding_base_url(base_url)

        try:
            import litellm

            embed_kwargs: dict[str, Any] = {
                "model": f"{self.litellm_prefix}/{model_id}",
                "input": texts,
                "timeout": timeout,
                "api_base": resolved_url,
            }
            if dimensions is not None:
                embed_kwargs["dimensions"] = dimensions

            response = litellm.embedding(**embed_kwargs)
            return [item["embedding"] for item in response.data]
        except ImportError as e:
            raise RuntimeError(
                "litellm is required for Ollama embedding support. "
                "Install with: pip install litellm"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings with Ollama: {e}") from e

    def check_embedding_health(
        self,
        api_key: str | None,  # noqa: ARG002 - provider adapter contract; Ollama needs no key
        base_url: str | None,
        model_id: str | None = None,
        timeout: float = 10.0,
    ) -> HealthResult:
        """Check that an Ollama server answers and has the embedding model pulled.

        A model not pulled answers reason ``model``; no answering server answers
        ``unreachable``; any other failure is classified by
        :func:`~osprey.models.providers.health.failure_reason`, else ``unreachable``.
        """
        try:
            configured = self.require_effective_base_url(base_url)
        except ValueError as e:
            return HealthResult(False, str(e), "config")

        try:
            url = self._resolve_embedding_base_url(configured)
        except _local_server.LocalServerUnreachable:
            return HealthResult(False, f"Cannot connect to Ollama at {configured}", "unreachable")

        model = model_id or self.health_check_embedding_model_id
        try:
            import requests

            response = requests.get(url.rstrip("/") + self.fallback_probe_path, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return HealthResult(
                False,
                f"Failed to check model availability: {e}",
                failure_reason(e) or "unreachable",
            )

        available = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        if model and model.split(":")[0] not in available:
            return HealthResult(
                False,
                f"Model '{model}' not found. Available: {available}. "
                f"Run 'ollama pull {model}' to download.",
                "model",
            )
        return HealthResult(True, f"Ollama connected at {url}", None)
