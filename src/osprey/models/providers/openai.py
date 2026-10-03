"""OpenAI Provider Adapter Implementation.

This provider uses LiteLLM as the backend for unified API access.
"""

from .health import HealthResult, failure_reason
from .litellm_adapter import check_litellm_health, execute_litellm_completion
from .litellm_delegating import LiteLLMDelegatingProvider

__all__ = ["OpenAIProviderAdapter", "check_litellm_health", "execute_litellm_completion"]

#: The chat model families that take a caller-chosen temperature. A family
#: matches its own id and every id that extends it with a hyphen (``gpt-4o``
#: matches ``gpt-4o-mini`` and ``gpt-4o-2024-08-06``).
_TEMPERATURE_FAMILIES = ("gpt-3.5", "gpt-4", "gpt-4o", "gpt-4.1", "gpt-4.5", "chatgpt-4o")


class OpenAIProviderAdapter(LiteLLMDelegatingProvider):
    """OpenAI provider implementation using LiteLLM."""

    # Metadata (single source of truth)
    name = "openai"
    description = "OpenAI (GPT models)"
    requires_api_key = True
    requires_base_url = False
    requires_model_id = True
    supports_proxy = True
    default_base_url = "https://api.openai.com/v1"
    models_probe = "bearer"
    models_probe_base_url = "https://api.openai.com/v1"
    default_model_id = "gpt-5.6-sol"  # Flagship for general use
    health_check_model_id = "gpt-5.6-luna"  # Cheapest listed model for health checks
    default_embedding_model_id = "text-embedding-3-small"
    health_check_embedding_model_id = "text-embedding-3-small"

    # API key acquisition information
    api_key_url = "https://platform.openai.com/api-keys"
    api_key_instructions = [
        "Sign up or log in to your OpenAI account",
        "Add billing information if not already set up",
        "Click '+ Create new secret key'",
        "Name your key and copy it (shown only once!)",
    ]
    api_key_note = None

    # Provider facts (see BaseProvider)
    api_key_env_var = "OPENAI_API_KEY"
    api_protocol = "openai"
    supports_interactive_login = False
    supports_images = True
    supports_thinking = False
    self_hosted = False

    # LiteLLM integration - OpenAI models don't need a prefix in LiteLLM
    litellm_prefix = ""
    # OpenAI's API takes max_completion_tokens on every chat model and refuses
    # max_tokens on its reasoning models, whichever generation LiteLLM knows.
    max_tokens_param = "max_completion_tokens"

    @classmethod
    def accepts_temperature(cls, model_id: str) -> bool:
        """Whether a request for *model_id* carries the caller's temperature.

        OpenAI's chat families take one; its reasoning models refuse every
        temperature but their default. An id outside the chat families is
        treated as a reasoning model, so a new reasoning id is never sent one.

        Args:
            model_id: The bare OpenAI model identifier.

        Returns:
            True for an id in one of the chat families.
        """
        return any(
            model_id == family or model_id.startswith(family + "-")
            for family in _TEMPERATURE_FAMILIES
        )

    # execute_completion / check_health inherited from LiteLLMDelegatingProvider.

    def execute_embedding(
        self,
        texts: list[str],
        model_id: str | None,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 600.0,
    ) -> list[list[float]]:
        """Embed *texts* through LiteLLM, one vector per text, in input order.

        A key, base URL or dimension count that is not given is left out of the
        call, so LiteLLM applies its own default for it.

        Args:
            texts: Texts to embed.
            model_id: Embedding model id (``default_embedding_model_id`` when None).
            api_key: OpenAI API key.
            base_url: Custom endpoint, e.g. a proxy or Azure OpenAI.
            dimensions: Output vector length (v3 models only).
            timeout: Request timeout in seconds.

        Returns:
            One embedding vector per input text.

        Raises:
            RuntimeError: When LiteLLM is missing or the call fails; the
                original exception is the ``__cause__``.
        """
        if not texts:
            return []

        try:
            import litellm
        except ImportError as e:
            raise RuntimeError(
                "litellm is required for OpenAI embedding support. "
                "Install with: pip install litellm"
            ) from e

        embed_kwargs: dict = {
            "model": model_id or self.default_embedding_model_id,
            "input": texts,
            "timeout": timeout,
        }
        if api_key:
            embed_kwargs["api_key"] = api_key
        if base_url:
            embed_kwargs["api_base"] = base_url
        if dimensions is not None:
            embed_kwargs["dimensions"] = dimensions

        try:
            response = litellm.embedding(**embed_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings with OpenAI: {e}") from e
        return [item["embedding"] for item in response.data]

    def check_embedding_health(
        self,
        api_key: str | None,
        base_url: str | None,
        model_id: str | None = None,
        timeout: float = 10.0,
    ) -> HealthResult:
        """Embed one one-token text against the configured model.

        Never raises for a provider failure: the exception is classified by
        :func:`~osprey.models.providers.health.failure_reason`, and anything it
        does not know is reported ``unreachable``.

        Args:
            api_key: OpenAI API key; without one no request is made.
            base_url: Custom endpoint, e.g. a proxy or Azure OpenAI.
            model_id: Model to check (``health_check_embedding_model_id`` when None).
            timeout: Request timeout in seconds.

        Returns:
            The verdict.
        """
        if not api_key:
            return HealthResult(False, "OpenAI API key is required", "config")

        model = model_id or self.health_check_embedding_model_id
        try:
            self.execute_embedding(
                ["ping"], model_id=model, api_key=api_key, base_url=base_url, timeout=timeout
            )
        except Exception as e:
            cause = e.__cause__ if isinstance(e.__cause__, BaseException) else e
            reason = failure_reason(cause) or "unreachable"
            return HealthResult(False, f"OpenAI embedding health check failed: {cause}", reason)
        return HealthResult(True, f"OpenAI embedding API healthy (model: {model})", None)
