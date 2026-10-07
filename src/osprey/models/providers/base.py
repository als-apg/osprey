"""Base Provider Interface for AI Model Access."""

import os
from abc import ABC
from typing import Any, Literal

from osprey_connectors.config import is_unresolved_placeholder

from .health import HealthResult

# A keyless endpoint (an on-prem vLLM, a local Ollama) still needs a non-empty
# key on the wire: the OpenAI-compatible clients underneath refuse to send
# without one.
KEYLESS_API_KEY_PLACEHOLDER = "EMPTY"

# One image for an image-embedding call: its raw bytes and their MIME type.
ImageInput = tuple[bytes, str]
# One text for an image-embedding call, embedded into the same space as images.
TextInput = str


class EmbeddingDimensionError(ValueError):
    """The requested ``dimensions`` exceed the length of the vector the model returned.

    A configuration fault: every vector from the same model fails the same way,
    so no retry or per-input skip can help. A ``ValueError`` because the value a
    caller passed is what is wrong.
    """


class DegenerateVectorError(ValueError):
    """A returned vector has zero norm or a non-finite component.

    A fault of one input, not of the configuration: the vector cannot be
    L2-normalised, so it cannot be stored or ranked. A ``ValueError`` because
    the value the model returned is what is wrong.
    """


class BaseProvider(ABC):  # noqa: B024 - every endpoint has a "not served" default
    """Abstract base class for AI model providers.

    All provider implementations inherit from this class and override the
    endpoints they serve: ``execute_completion``/``check_health`` for chat,
    ``execute_embedding`` for text embeddings, ``execute_image_embedding`` for
    image embeddings, and ``check_embedding_health`` for either embedding route.
    Every endpoint method defaults to "not served" (``NotImplementedError`` or an
    unhealthy verdict), and what a provider serves is derived from which of them
    it overrides — see :meth:`supports_chat`, :meth:`supports_embeddings` and
    :meth:`supports_image_embeddings`.

    **Metadata as Class Attributes** (SINGLE SOURCE OF TRUTH):
    Subclasses define provider metadata as class attributes. The registry
    introspects these attributes after loading the class, avoiding duplication
    between ProviderRegistration and the class itself. This follows the same
    pattern as capabilities and context classes in the framework.

    Metadata Attributes (define on subclass):
        name: Provider identifier (e.g., "anthropic", "openai")
        description: User-friendly description (e.g., "Anthropic (Claude models)")
        requires_api_key: Whether provider requires API key for authentication
        requires_base_url: Whether provider requires custom base URL
        requires_model_id: Whether provider requires model ID specification
        supports_proxy: Whether provider supports HTTP proxy configuration
        default_base_url: Default API endpoint URL if applicable
        base_url_env_var: Name of an env var that, when set, overrides every
            other base_url source (explicit argument, config, default) for this
            provider — the runtime lever for redirecting an already-deployed
            system at a different gateway without a rebuild. None disables the
            override (the default; providers opt in explicitly so the name
            never collides with an env var another layer owns, e.g.
            ANTHROPIC_BASE_URL).
        default_model_id: Default model recommended for general use (used in templates)
        health_check_model_id: Cheapest/fastest model for health checks
        available_models: List of available model IDs for this provider
        api_key_url: URL where users can obtain an API key (e.g., "https://console.anthropic.com/")
        api_key_instructions: Step-by-step instructions for obtaining an API key
        api_key_note: Additional notes or requirements (e.g., "Requires affiliation")

    Provider Facts (stated by every built-in adapter):
        api_key_env_var: The environment variable a deployment keeps this
            provider's key in; None exactly when requires_api_key is False.
        api_protocol: The protocol a Claude Code launch speaks to this
            provider's endpoint, one of the catalog's api_protocol values —
            "anthropic" (Messages API, no translation proxy) or "openai" (Chat
            Completions, through the translation proxy). Not the same fact as
            is_openai_compatible, which names the route LiteLLM calls: a
            gateway that speaks both answers "anthropic" here and True there.
        supports_interactive_login: Whether a launch that finds no key can sign
            in interactively instead of failing.
        supports_images: Whether this provider's OpenAI-protocol route accepts
            image input as image_url content parts. That is the route the
            translation proxy calls; a launch that speaks Anthropic to the
            provider carries images regardless. False where the vendor
            documents no such support, and for a local server, whose image
            input depends on the model each site serves.
        supports_thinking: Whether this provider's OpenAI-protocol route takes
            the request's thinking setting and returns the model's thinking.

    Declared Behaviour:
        accepts_chat_request: Whether a chat call honours the full request a
            caller builds (its ``chat_request`` and its timeout). False on a
            provider whose completion path drops them.
        host_override_env_var: An environment variable naming a server URL that
            the reachability walk tries before the configured one. Read only by
            health checks and the out-of-call resolver, never by a model call.
            None (the default) means there is no such override.
        resolves_fallback_outside_calls: Whether a reachable endpoint for this
            provider is found (env override, configured URL, container
            fallbacks) and cached by the caller before a model call, instead of
            by the adapter during every call. False by default.
        truncates_to_dimensions: Whether an embedding call honours
            ``dimensions`` by truncating each returned vector and
            L2-renormalising it. Callers send ``dimensions`` only to a provider
            that declares this. False by default.

    Embedding Attributes:
        default_embedding_model_id: Default embedding model for templates.
        health_check_embedding_model_id: Model an embedding health check probes
            when the caller names none.

    LiteLLM Integration Attributes:
        litellm_prefix: LiteLLM provider prefix (e.g., "anthropic", "gemini"). If None,
            uses the provider name. Set to empty string "" if no prefix needed.
        is_openai_compatible: True if this provider uses an OpenAI-compatible API
            endpoint with custom base_url (e.g., CBORG, Stanford, ARGO, vLLM).
            When True, LiteLLM routes via "openai/{model}" with api_base parameter.
        supports_native_structured_output: True=native json_schema, False=prompt fallback, None=auto-detect

    This interface ensures consistent provider behavior across the framework
    while allowing provider-specific implementations.
    """

    # Metadata - subclasses MUST override these class attributes
    name: str = NotImplemented  # Provider identifier (e.g., "anthropic")
    description: str = (
        NotImplemented  # User-friendly description (e.g., "Anthropic (Claude models)")
    )
    requires_api_key: bool = NotImplemented
    requires_base_url: bool = NotImplemented
    requires_model_id: bool = NotImplemented
    supports_proxy: bool = NotImplemented
    default_base_url: str | None = None
    base_url_env_var: str | None = None  # Env var overriding all base_url sources (opt-in)
    # How a health check lists this route's models: "bearer" for an
    # OpenAI-compatible GET /v1/models with a bearer key, "anthropic" for the
    # Anthropic listing (x-api-key + anthropic-version). None means the route
    # serves no listing a probe can trust, and it is never probed.
    models_probe: Literal["bearer", "anthropic"] | None = None
    # Endpoint the listing probe uses when neither the caller nor a required
    # default supplies one. Read only by the probe, so it never pins the route
    # a model call takes.
    models_probe_base_url: str | None = None
    default_model_id: str | None = None  # Default model for templates/general use
    health_check_model_id: str | None = None  # Cheapest model for health checks
    available_models: list[str] = []  # List of available models for this provider

    # API key acquisition information (for CLI help and documentation)
    api_key_url: str | None = None  # URL where users can obtain an API key
    api_key_instructions: list[str] = []  # Step-by-step instructions for obtaining the key
    api_key_note: str | None = None  # Additional notes or requirements

    # LiteLLM integration configuration
    # These attributes allow providers to declare their LiteLLM routing behavior,
    # eliminating hardcoded provider checks in the adapter layer.
    litellm_prefix: str | None = None  # LiteLLM prefix (e.g., "anthropic", "gemini")
    is_openai_compatible: bool = False  # True for OpenAI-compatible endpoints (CBORG, etc.)
    # The gateway kind fronting this provider, when it is a proxy rather than a
    # model vendor. "litellm" makes every request carry the acting identity for
    # spend attribution (see models/spend_attribution.py); None sends nothing.
    gateway: str | None = None
    # Structured output routing:
    #   True  -> send response_format json_schema (native constrained decoding)
    #   False -> use OSPREY's prompt-based JSON fallback
    #   None  -> defer to litellm.supports_response_schema() (auto-detect)
    supports_native_structured_output: bool | None = None
    # The request parameter that carries the output-token cap. LiteLLM maps
    # max_tokens to each route's own parameter for the models it recognises; an
    # endpoint that refuses max_tokens outright declares the parameter it takes.
    max_tokens_param: str = "max_tokens"

    # Provider facts. Every built-in adapter states each one in its own class body.
    api_key_env_var: str | None = None
    api_protocol: Literal["anthropic", "openai"] = "openai"
    supports_interactive_login: bool = False
    supports_images: bool = False
    supports_thinking: bool = False
    # True when the site runs the model server itself; False when a vendor or
    # an institution runs the service the provider calls.
    self_hosted: bool = False

    # Whether a chat call carries the caller's full request (chat_request, timeout).
    accepts_chat_request: bool = True
    # Server-URL environment override tried first when looking for a reachable endpoint.
    host_override_env_var: str | None = None
    # Whether the caller resolves and caches a reachable endpoint before calling.
    resolves_fallback_outside_calls: bool = False
    # Whether ``dimensions`` is honoured by truncating and L2-renormalising each vector.
    truncates_to_dimensions: bool = False

    # Embedding health surface
    default_embedding_model_id: str | None = None
    health_check_embedding_model_id: str | None = None

    @classmethod
    def _implements(cls, name: str) -> bool:
        """Whether this class overrides *name* somewhere below :class:`BaseProvider`.

        Resolved through the MRO, so an intermediate base that implements the
        method (``LiteLLMDelegatingProvider`` for chat) counts for every
        subclass of it.
        """
        return getattr(cls, name) is not getattr(BaseProvider, name)

    @classmethod
    def supports_chat(cls) -> bool:
        """Whether this provider serves chat completions."""
        return cls._implements("execute_completion")

    @classmethod
    def supports_embeddings(cls) -> bool:
        """Whether this provider serves text embeddings."""
        return cls._implements("execute_embedding")

    @classmethod
    def supports_image_embeddings(cls) -> bool:
        """Whether this provider serves image embeddings."""
        return cls._implements("execute_image_embedding")

    @classmethod
    def accepts_temperature(cls, model_id: str) -> bool:
        """Whether a request for *model_id* carries the caller's sampling temperature.

        True on every model unless the adapter says otherwise. A provider whose
        models differ overrides this and decides from the model id. A request for
        a model that answers False carries no temperature and samples at the
        model's default.

        Args:
            model_id: The provider's bare model identifier.

        Returns:
            True when the request carries the caller's temperature.
        """
        return True

    @classmethod
    def effective_base_url(cls, base_url: str | None) -> str | None:
        """Resolve the base_url actually used: env override > caller value > default.

        Lives on the base class because two callers must agree on the answer: the
        adapter that finally calls the endpoint, and whatever validates
        :attr:`requires_base_url` before it. When only the adapter knew this rule,
        a provider carrying a perfectly good :attr:`default_base_url` still failed
        validation for "missing" base_url, and the default it declared was
        unreachable — visible only once the env override was removed.

        A ``base_url`` that is still an unresolved ``${VAR}`` reference counts as
        no value at all. :func:`~osprey_connectors.config.resolve_env_vars` keeps
        such a reference verbatim when the variable is unset, so a config
        declaring ``base_url: ${MY_GATEWAY}`` with nothing exported would
        otherwise hand the literal string to the HTTP client and fail somewhere
        far from the cause.

        **When a missing value falls back to** :attr:`default_base_url`: when
        this provider also requires one. Requiring an endpoint is what makes a
        config that omits ``base_url`` mean "the default I declare" — an
        openai-compatible route that would otherwise fall through to
        api.openai.com, or a local server on a well-known port — and it is what
        would otherwise make the declared default unreachable, since
        :mod:`osprey.models.completion` rejects the call for a missing base_url
        before any adapter body runs. A provider that requires no endpoint keeps
        resolving to ``None`` even with a default declared: litellm derives the
        endpoint from the model prefix there, and forwarding a default would pin
        a route the client is meant to choose.

        Args:
            base_url: The caller's value, usually from deployment config. May be
                ``None``.

        Returns:
            The URL this provider will use, or ``None`` when it has no source for
            one — which is the only case a ``requires_base_url`` provider should
            be rejected for.
        """
        if cls.base_url_env_var:
            override = os.environ.get(cls.base_url_env_var)
            if override:
                return override
        if is_unresolved_placeholder(base_url):
            base_url = None
        if cls.requires_base_url and cls.default_base_url:
            return base_url or cls.default_base_url
        return base_url

    @classmethod
    def validate_base_url(cls, url: str | None) -> None:
        """Refuse a base URL this provider cannot use, before any request is built.

        Accepts every URL by default. A provider whose request paths are built
        on the base URL overrides this to refuse a shape that would silently
        reach the wrong path.

        Args:
            url: A resolved base URL, or None.

        Raises:
            ValueError: When this provider cannot use *url*.
        """
        return None

    @classmethod
    def require_effective_base_url(cls, base_url: str | None) -> str:
        """Same resolution as :meth:`effective_base_url`, but never ``None``.

        For adapters that build a request URL in their own body: they need a
        ``str``, and the rule they must apply is the one
        :meth:`effective_base_url` implements. A local
        ``base_url or self.default_base_url`` looks equivalent and is not — it
        skips the env override, and it disagrees with the ``requires_base_url``
        check in :mod:`osprey.models.completion`, which then rejects the call
        before the adapter body ever runs.

        Args:
            base_url: The caller's value, usually from deployment config. May be
                ``None``.

        Returns:
            The URL this provider will call.

        Raises:
            ValueError: When no source supplies one — the same condition
                :func:`osprey.models.completion.get_chat_completion` rejects,
                reported with the same wording.
        """
        resolved = cls.effective_base_url(base_url)
        if not resolved:
            raise ValueError(f"Base URL required for {cls.name}")
        return resolved

    @classmethod
    def resolve_base_url(cls, base_url: str | None) -> str | None:
        """The endpoint to call, refusing ``None`` when this provider needs one.

        Composes the two resolvers above by the provider's own
        :attr:`requires_base_url` declaration, so a caller that just wants "the
        endpoint, resolved correctly for whichever provider this is" does not
        have to branch on that attribute itself.

        Args:
            base_url: The caller's value, usually from deployment config. May be
                ``None``.

        Returns:
            The URL this provider will use, or ``None`` for a provider that
            needs none.

        Raises:
            ValueError: When this provider requires an endpoint and no source
                supplies one.
        """
        if cls.requires_base_url:
            return cls.require_effective_base_url(base_url)
        return cls.effective_base_url(base_url)

    @classmethod
    def effective_api_key(cls, api_key: str | None) -> str | None:
        """The key to send: the placeholder when this provider declares none is needed.

        An absent key is passed through untouched for a provider that requires
        one — refusing it is the requirement gate's job, and substituting here
        would turn "no key configured" into an authentication failure from the
        vendor, reported nowhere near its cause.

        Args:
            api_key: The caller's value, usually from deployment config. May be
                ``None``.

        Returns:
            The key to put on the wire, or ``None`` when there is none and this
            provider requires one.
        """
        if api_key:
            return api_key
        return None if cls.requires_api_key else KEYLESS_API_KEY_PLACEHOLDER

    def execute_completion(
        self,
        message: str,
        model_id: str,
        api_key: str | None,
        base_url: str | None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking: dict | None = None,
        system_prompt: str | None = None,
        output_format: Any | None = None,
        **kwargs,
    ) -> str | Any:
        """Execute a direct chat completion.

        Not served unless a subclass overrides it; :meth:`supports_chat` answers
        from that override.

        :param message: User message to send
        :param model_id: Model identifier
        :param api_key: API authentication key
        :param base_url: Custom API endpoint URL
        :param max_tokens: Maximum tokens to generate
        :param temperature: Sampling temperature
        :param thinking: Extended thinking configuration (if supported)
        :param system_prompt: System prompt (if supported)
        :param output_format: Structured output format (Pydantic model or TypedDict)
        :param kwargs: Additional provider-specific arguments
        :return: Model response text or structured output
        :raises NotImplementedError: When this provider has no chat endpoint
        """
        raise NotImplementedError(f"{self.name} has no chat endpoint")

    def check_health(
        self,
        api_key: str | None,  # noqa: ARG002 - provider adapter contract
        base_url: str | None,  # noqa: ARG002 - provider adapter contract
        timeout: float = 5.0,  # noqa: ARG002 - provider adapter contract
        model_id: str | None = None,  # noqa: ARG002 - provider adapter contract
    ) -> tuple[bool, str]:
        """Test provider connectivity and authentication.

        Makes a minimal API call to verify the API key works. For paid providers,
        uses the cheapest available model with minimal tokens (~$0.0001 per check).
        A provider without a chat endpoint answers unhealthy without a call.

        :param api_key: API authentication key
        :param base_url: Custom API endpoint URL
        :param timeout: Request timeout in seconds
        :param model_id: Optional model ID to test with (uses cheapest if not provided)
        :return: (success, message) tuple
        """
        return False, f"{self.name} has no chat endpoint"

    def execute_embedding(
        self,
        texts: list[str],
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 600.0,
    ) -> list[list[float]]:
        """Embed *texts*, one vector per text, in input order.

        :param texts: Texts to embed
        :param model_id: Embedding model identifier
        :param api_key: API authentication key
        :param base_url: Custom API endpoint URL
        :param dimensions: When given, every returned vector has exactly this length
        :param timeout: Request timeout in seconds
        :return: One embedding vector per input text
        :raises NotImplementedError: When this provider has no text-embedding endpoint
        """
        raise NotImplementedError(f"{self.name} has no embedding endpoint")

    def execute_image_embedding(
        self,
        inputs: list[ImageInput | TextInput],
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 600.0,
    ) -> list[list[float]]:
        """Embed images and texts into one shared vector space, in input order.

        :param inputs: Each an :data:`ImageInput` ``(bytes, mime)`` or a :data:`TextInput`
        :param model_id: Embedding model identifier
        :param api_key: API authentication key
        :param base_url: Custom API endpoint URL
        :param dimensions: When given, every returned vector has exactly this length
            (truncated and L2-renormalised), so stored and query vectors agree
        :param timeout: Request timeout in seconds
        :return: One embedding vector per input
        :raises NotImplementedError: When this provider has no image-embedding endpoint
        """
        raise NotImplementedError(f"{self.name} has no image embedding endpoint")

    def check_embedding_health(
        self,
        api_key: str | None,  # noqa: ARG002 - provider adapter contract
        base_url: str | None,  # noqa: ARG002 - provider adapter contract
        model_id: str | None = None,  # noqa: ARG002 - provider adapter contract
        timeout: float = 10.0,  # noqa: ARG002 - provider adapter contract
    ) -> HealthResult:
        """Test the embedding endpoint, answering with one typed verdict.

        Never raises for a server or HTTP failure. An unhealthy verdict always
        carries a reason — :func:`~osprey.models.providers.health.failure_reason`
        of the exception, else ``unreachable`` for a timeout or an endpoint that
        never answered — so no caller classifies a verdict from its message.

        :param api_key: API authentication key
        :param base_url: Custom API endpoint URL
        :param model_id: Model to check (``health_check_embedding_model_id`` if omitted)
        :param timeout: Request timeout in seconds
        :return: The verdict
        """
        return HealthResult(False, f"{self.name} has no embedding endpoint", "config")
