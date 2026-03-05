"""Lightweight Embedding Provider Registry — lazy-loaded provider class resolution.

Standalone singleton that resolves embedding provider names to
BaseEmbeddingProvider subclasses without depending on the full RegistryManager.

Mirrors the pattern in ``provider_registry.py`` for LLM providers.

Adding a new built-in embedding provider = one entry in
``_BUILTIN_EMBEDDING_PROVIDERS`` below.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.models.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger("osprey.models.embedding_registry")


@dataclass(frozen=True, slots=True)
class _EmbeddingProviderEntry:
    """Lazy-load descriptor for an embedding provider class."""

    module_path: str
    class_name: str


# ── Built-in embedding provider table (single source of truth) ─────────
_BUILTIN_EMBEDDING_PROVIDERS: dict[str, _EmbeddingProviderEntry] = {
    "ollama": _EmbeddingProviderEntry("osprey.models.embeddings.ollama", "OllamaEmbeddingProvider"),
    "openai": _EmbeddingProviderEntry("osprey.models.embeddings.openai", "OpenAIEmbeddingProvider"),
}


class EmbeddingProviderRegistry:
    """Lightweight registry — lazily loads embedding provider classes by name.

    Provider classes are imported only on first ``get_provider()`` call, keeping
    module-level imports minimal.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _EmbeddingProviderEntry] = dict(_BUILTIN_EMBEDDING_PROVIDERS)
        self._providers: dict[str, type[BaseEmbeddingProvider]] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def get_provider(self, name: str) -> type[BaseEmbeddingProvider] | None:
        """Return the embedding provider class for *name*, importing lazily.

        Returns ``None`` if the name is unknown.
        """
        if name in self._providers:
            return self._providers[name]

        entry = self._entries.get(name)
        if entry is None:
            return None

        return self._load(name, entry)

    def register_provider(
        self,
        name: str,
        module_path: str,
        class_name: str,
    ) -> None:
        """Register a custom embedding provider (visible globally via the singleton).

        If *name* already exists the entry is overwritten, allowing runtime
        overrides for testing or site-local customizations.
        """
        self._entries[name] = _EmbeddingProviderEntry(module_path, class_name)
        self._providers.pop(name, None)  # evict cache so next get_provider re-imports

    def list_providers(self) -> list[str]:
        """Return sorted list of all known embedding provider names."""
        return sorted(self._entries)

    # ── Internal ───────────────────────────────────────────────────────

    def _load(
        self, name: str, entry: _EmbeddingProviderEntry
    ) -> type[BaseEmbeddingProvider] | None:
        """Import the embedding provider module and cache the class."""
        try:
            module = importlib.import_module(entry.module_path)
            cls = getattr(module, entry.class_name)
            self._providers[name] = cls
            logger.debug(
                "Loaded embedding provider: %s (%s.%s)",
                name,
                entry.module_path,
                entry.class_name,
            )
            return cls
        except (ImportError, AttributeError) as exc:
            logger.warning("Failed to load embedding provider %s: %s", name, exc)
            return None


# ── Singleton ──────────────────────────────────────────────────────────

_registry: EmbeddingProviderRegistry | None = None


def get_embedding_registry() -> EmbeddingProviderRegistry:
    """Return the global ``EmbeddingProviderRegistry`` singleton."""
    global _registry
    if _registry is None:
        _registry = EmbeddingProviderRegistry()
    return _registry


def reset_embedding_registry() -> None:
    """Reset the singleton (for test teardown)."""
    global _registry
    _registry = None
