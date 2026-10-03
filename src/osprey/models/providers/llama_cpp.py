"""llama.cpp server (``llama-server``) Provider Adapter.

Serves text and image embeddings from one multimodal embedding model, and no
chat. Every input is one ``POST <base>/v1/embeddings`` whose single input item
carries one content part — an ``image_url`` data URL or a ``text`` part — the
request shape ``llama-server --embedding --mmproj`` accepts for a vision
embedding model. The vectors come back in the standard OpenAI embeddings
envelope.

The execute methods call exactly the base URL they are given: they never read
:attr:`LlamaCppProviderAdapter.host_override_env_var` and never probe. Finding a
reachable server (the env override, the configured URL, its container
fallbacks) is the health check's job and the out-of-call resolver's, which
caches the answer before any call is made.
"""

from __future__ import annotations

import base64
import math
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from osprey.utils.logger import get_logger

from . import _local_server
from .base import (
    BaseProvider,
    DegenerateVectorError,
    EmbeddingDimensionError,
    ImageInput,
    TextInput,
)
from .health import HealthResult, failure_reason

logger = get_logger("llama_cpp")

#: The embedding model a site's llama-server is expected to serve, by the name it
#: advertises (``--alias``). The upstream model name lower-cased, as the catalog
#: names models by their upstream ids.
LLAMA_CPP_DEFAULT_MODEL = "qwen3-vl-embedding-2b"

#: Seconds each reachability probe may take; a caller's shorter timeout lowers it.
_PROBE_TIMEOUT = 2.0

_EMBEDDINGS_PATH = "/v1/embeddings"


def _redact(url: str) -> str:
    """*url* with any userinfo replaced, so it can appear in a message."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


def _content_part(item: ImageInput | TextInput) -> dict:
    """The one content part that carries *item* in an embeddings request."""
    if isinstance(item, str):
        return {"type": "text", "text": item}
    data, mime = item
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _fit(vector: list[float], dimensions: int | None) -> list[float]:
    """*vector* checked, and truncated to *dimensions* and L2-renormalised when given.

    Raises:
        EmbeddingDimensionError: When *dimensions* is below 1 or exceeds the
            vector's length.
        DegenerateVectorError: When the (truncated) vector has a non-finite
            component or zero norm.
    """
    if dimensions is not None:
        if dimensions < 1 or dimensions > len(vector):
            raise EmbeddingDimensionError(
                f"dimensions={dimensions} cannot be taken from a {len(vector)}-dimensional vector"
            )
        vector = vector[:dimensions]
    if not all(math.isfinite(value) for value in vector):
        raise DegenerateVectorError("embedding vector has a non-finite component")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise DegenerateVectorError("embedding vector has zero norm")
    if dimensions is None:
        return list(vector)
    return [value / norm for value in vector]


class LlamaCppProviderAdapter(BaseProvider):
    """llama.cpp server provider: text and image embeddings, no chat."""

    # Metadata (single source of truth)
    name = "llama-cpp"
    description = "llama.cpp server (text and image embeddings, no chat, site-run)"
    requires_api_key = False
    requires_base_url = True
    requires_model_id = True
    supports_proxy = False
    default_base_url = "http://localhost:8080"

    # API key acquisition information
    api_key_url = None
    api_key_instructions = []
    api_key_note = "llama-server runs on site and does not require an API key"

    # Provider facts (see BaseProvider)
    api_key_env_var = None
    api_protocol = "openai"
    supports_interactive_login = False
    supports_images = False
    supports_thinking = False

    # Declared behaviour (see BaseProvider)
    host_override_env_var = "LLAMA_CPP_HOST"
    resolves_fallback_outside_calls = True
    truncates_to_dimensions = True

    # Embedding defaults
    default_embedding_model_id = LLAMA_CPP_DEFAULT_MODEL
    health_check_embedding_model_id = LLAMA_CPP_DEFAULT_MODEL

    # The path a live server answers 200 on, listing the model it serves.
    fallback_probe_path = "/v1/models"

    @classmethod
    def validate_base_url(cls, url: str | None) -> None:
        """Refuse a base URL ending in ``/v1``.

        Every request path (``/v1/embeddings``, ``/v1/models``) is appended to
        the server root, so a ``…/v1`` base would reach ``/v1/v1/…`` and read
        as a missing model.

        Raises:
            ValueError: When the URL's path ends in ``/v1``.
        """
        if url and urlsplit(url).path.rstrip("/").endswith("/v1"):
            raise ValueError("base_url is the server root (as OLLAMA_HOST), not …/v1")

    @classmethod
    def _default_port(cls) -> int:
        """The well-known port, read from :attr:`default_base_url`."""
        return urlsplit(cls.default_base_url).port or 8080

    def _embed(
        self,
        items: list[ImageInput | TextInput],
        model_id: str,
        base_url: str | None,
        dimensions: int | None,
        timeout: float,
    ) -> list[list[float]]:
        """One POST per item to ``<base>/v1/embeddings``, results in input order.

        Each POST is bounded by its share of the time left of *timeout*.
        """
        if not items:
            return []
        import requests

        base = self.require_effective_base_url(base_url)
        self.validate_base_url(base)
        url = base.rstrip("/") + _EMBEDDINGS_PATH
        deadline = time.monotonic() + timeout

        vectors: list[list[float]] = []
        for index, item in enumerate(items):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise requests.Timeout(
                    f"llama-cpp embedding timed out after {timeout}s "
                    f"({index} of {len(items)} inputs embedded)"
                )
            body: dict[str, Any] = {
                "model": model_id,
                "input": [{"content": [_content_part(item)]}],
            }
            response = requests.post(url, json=body, timeout=remaining / (len(items) - index))
            response.raise_for_status()
            payload = response.json()
            try:
                vector = payload["data"][0]["embedding"]
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(f"llama-cpp returned no embedding in its response: {e}") from e
            vectors.append(_fit(vector, dimensions))
        return vectors

    def execute_embedding(
        self,
        texts: list[str],
        model_id: str,
        api_key: str | None = None,  # noqa: ARG002 - provider adapter contract; keyless
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 600.0,
    ) -> list[list[float]]:
        """Embed *texts*, each as a text content part, one POST per text.

        Args:
            texts: Texts to embed.
            model_id: The model id the server advertises.
            api_key: Unused; the server is keyless.
            base_url: The server root; the declared default when None.
            dimensions: Truncate each vector to this length and L2-renormalise.
            timeout: Seconds for the whole call, shared across the POSTs.

        Returns:
            One vector per text, in input order.

        Raises:
            ValueError: When the base URL is missing or ends in ``/v1``.
            EmbeddingDimensionError: When *dimensions* exceeds a returned vector.
            DegenerateVectorError: When a vector has zero norm or a non-finite value.
            requests.RequestException: When a request fails or times out.
        """
        return self._embed(list(texts), model_id, base_url, dimensions, timeout)

    def execute_image_embedding(
        self,
        inputs: list[ImageInput | TextInput],
        model_id: str,
        api_key: str | None = None,  # noqa: ARG002 - provider adapter contract; keyless
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 600.0,
    ) -> list[list[float]]:
        """Embed images and texts into one vector space, one POST per input.

        Args:
            inputs: Each an ``(bytes, mime)`` image or a text.
            model_id: The model id the server advertises.
            api_key: Unused; the server is keyless.
            base_url: The server root; the declared default when None.
            dimensions: Truncate each vector to this length and L2-renormalise.
            timeout: Seconds for the whole call, shared across the POSTs.

        Returns:
            One vector per input, in input order.

        Raises:
            ValueError: When the base URL is missing or ends in ``/v1``.
            EmbeddingDimensionError: When *dimensions* exceeds a returned vector.
            DegenerateVectorError: When a vector has zero norm or a non-finite value.
            requests.RequestException: When a request fails or times out.
        """
        return self._embed(list(inputs), model_id, base_url, dimensions, timeout)

    def check_embedding_health(
        self,
        api_key: str | None,  # noqa: ARG002 - provider adapter contract; keyless
        base_url: str | None,
        model_id: str | None = None,
        timeout: float = 10.0,
    ) -> HealthResult:
        """Check that a llama-server answers and serves the configured model.

        Tries ``LLAMA_CPP_HOST`` first, then the configured URL, then its
        container fallbacks, and reads ``data[0].id`` of ``GET /v1/models`` from
        the first that answers. A different id answers reason ``model``.
        """
        try:
            configured = self.require_effective_base_url(base_url)
            self.validate_base_url(configured)
        except ValueError as e:
            return HealthResult(False, str(e), "config")

        try:
            url = _local_server.resolve_local_server(
                configured,
                probe_path=self.fallback_probe_path,
                env_var=self.host_override_env_var,
                default_port=self._default_port(),
                timeout=min(_PROBE_TIMEOUT, timeout),
                label="llama-cpp",
            )
        except _local_server.LocalServerUnreachable:
            return HealthResult(
                False, f"Cannot connect to llama-cpp at {_redact(configured)}", "unreachable"
            )

        model = model_id or self.health_check_embedding_model_id
        try:
            import requests

            response = requests.get(url.rstrip("/") + self.fallback_probe_path, timeout=timeout)
            response.raise_for_status()
            served = response.json()["data"][0]["id"]
        except Exception as e:
            return HealthResult(
                False,
                f"llama-cpp at {_redact(url)} did not list its model: {type(e).__name__}",
                failure_reason(e) or "unreachable",
            )

        if model and served != model:
            return HealthResult(
                False,
                f"llama-cpp serves {served}, config names {model}: "
                f"start llama-server with --alias {model}",
                "model",
            )
        return HealthResult(True, f"llama-cpp serves {served} at {_redact(url)}", None)
