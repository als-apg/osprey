"""Fake embedding provider classes for ARIEL tests.

Production code resolves a provider *class* from the provider registry and
reads provider facts (``effective_base_url``, ``truncates_to_dimensions``)
through it, so a fake must be a :class:`BaseProvider` subclass whose facts are
class attributes: an attribute set on an instance is invisible to a
classmethod. :func:`make_fake_embedding_provider` builds a fresh class per test,
so the ``calls`` list it records into is never shared between tests.
"""

from __future__ import annotations

from typing import Any

from osprey.models.provider_registry import get_provider_registry
from osprey.models.providers.base import BaseProvider
from osprey.models.providers.health import HealthResult


class _FakeEmbeddingProviderBase(BaseProvider):
    """Embedding-only provider that records every call into its class's ``calls``.

    ``execute_embedding`` is synchronous, matching the real adapters: ARIEL
    calls it from async code without awaiting. It returns ``vectors`` when set,
    else ``vector`` once per text, else a ``dimension``-long vector per text;
    ``error`` makes it raise, which is how embedding-failure branches are reached.
    """

    name = "fake"
    description = "Fake embedding provider for tests"
    requires_api_key = False
    requires_base_url = True
    requires_model_id = False
    supports_proxy = False
    default_base_url = "http://fake-embeddings.invalid:11434"
    default_model_id = "nomic-embed-text"
    truncates_to_dimensions = False

    dimension: int = 4
    vector: list[float] | None = None
    vectors: list[list[float]] | None = None
    error: Exception | None = None
    healthy: HealthResult = HealthResult(True, "ok", None)
    calls: list[dict[str, Any]] = []

    def execute_embedding(  # type: ignore[override]
        self,
        texts: list[str],
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": list(texts),
                "model_id": model_id,
                "api_key": api_key,
                "base_url": base_url,
                **kwargs,
            }
        )
        if self.error is not None:
            raise self.error
        if self.vectors is not None:
            return [list(vector) for vector in self.vectors]
        if self.vector is not None:
            return [list(self.vector) for _ in texts]
        return [[0.1] * self.dimension for _ in texts]

    def check_embedding_health(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        timeout: float = 10.0,  # noqa: ARG002 - provider adapter contract
    ) -> HealthResult:
        self.calls.append(
            {
                "check_embedding_health": True,
                "api_key": api_key,
                "base_url": base_url,
                "model_id": model_id,
            }
        )
        return self.healthy


def make_fake_embedding_provider(**kw: Any) -> type[BaseProvider]:
    """Build a fresh fake embedding provider class.

    Args:
        **kw: Class attributes to set. Provider facts (``requires_base_url``,
            ``requires_api_key``, ``default_base_url``,
            ``truncates_to_dimensions``, ``name``) and behaviour (``vector``,
            ``vectors``, ``dimension``, ``error``, ``healthy``). ``healthy`` may
            be a ``(reachable, message)`` pair; an unhealthy pair gets reason
            ``unreachable``.

    Returns:
        A new :class:`BaseProvider` subclass with its own empty ``calls`` list.

    Raises:
        TypeError: If a keyword names no attribute of the fake.
    """
    unknown = [key for key in kw if not hasattr(_FakeEmbeddingProviderBase, key)]
    if unknown:
        raise TypeError(f"unknown fake provider attribute(s): {unknown}")
    healthy = kw.get("healthy")
    if healthy is not None and not isinstance(healthy, HealthResult):
        reachable, message = healthy
        kw["healthy"] = HealthResult(reachable, message, None if reachable else "unreachable")
    return type("FakeEmbeddingProvider", (_FakeEmbeddingProviderBase,), {**kw, "calls": []})


def ollama_text_embedder() -> BaseProvider:
    """Return an instance of the registered ``ollama`` provider adapter.

    The Ollama-gated ARIEL tests embed through this one helper, so they reach
    the adapter exactly the way production code does: by registry name.

    Raises:
        LookupError: If no provider named ``ollama`` is registered.
    """
    provider_cls = get_provider_registry().get_provider("ollama")
    if provider_cls is None:
        raise LookupError("no provider named 'ollama' is registered")
    return provider_cls()
