"""Lightweight Provider Registry — lazy-loaded provider class resolution.

Standalone singleton that resolves provider names to BaseProvider subclasses
without depending on the full RegistryManager.

Any component needing LLM access (MCP tools, FastAPI routes, CLI commands) can
call ``get_provider_registry().get_provider("cborg")`` to obtain the provider
class, then use it via ``get_chat_completion()`` or ``aget_chat_completion()``.

Adding a new built-in provider = one entry in ``_BUILTIN_PROVIDERS`` below.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.models.providers.base import BaseProvider

logger = logging.getLogger("osprey.models.provider_registry")


@dataclass(frozen=True, slots=True)
class _ProviderEntry:
    """Lazy-load descriptor for a provider class."""

    module_path: str
    class_name: str


# ── Provider → API key env var (single source of truth) ───────────────
# Maps each provider name to the environment variable holding its API key.
# ``None`` means the provider does not require an API key.
PROVIDER_API_KEYS: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cborg": "CBORG_API_KEY",
    "amsc": "AMSC_I2_API_KEY",
    "argo": "ARGO_API_KEY",
    "stanford": "STANFORD_API_KEY",
    "als-apg": "ALS_APG_API_KEY",
    "ollama": None,
    "asksage": None,  # uses different auth
    "vllm": None,  # local, no key
}


# ── Built-in provider table (single source of truth) ──────────────────
_BUILTIN_PROVIDERS: dict[str, _ProviderEntry] = {
    "anthropic": _ProviderEntry("osprey.models.providers.anthropic", "AnthropicProviderAdapter"),
    "openai": _ProviderEntry("osprey.models.providers.openai", "OpenAIProviderAdapter"),
    "google": _ProviderEntry("osprey.models.providers.google", "GoogleProviderAdapter"),
    "ollama": _ProviderEntry("osprey.models.providers.ollama", "OllamaProviderAdapter"),
    "cborg": _ProviderEntry("osprey.models.providers.cborg", "CBorgProviderAdapter"),
    "stanford": _ProviderEntry("osprey.models.providers.stanford", "StanfordProviderAdapter"),
    "argo": _ProviderEntry("osprey.models.providers.argo", "ArgoProviderAdapter"),
    "asksage": _ProviderEntry("osprey.models.providers.asksage", "AskSageProviderAdapter"),
    "vllm": _ProviderEntry("osprey.models.providers.vllm", "VLLMProviderAdapter"),
}


class ProviderRegistry:
    """Lightweight registry — lazily loads provider classes by name.

    Provider classes are imported only on first ``get_provider()`` call, keeping
    module-level imports minimal and avoiding network-triggering side effects on
    air-gapped machines.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _ProviderEntry] = dict(_BUILTIN_PROVIDERS)
        self._providers: dict[str, type[BaseProvider]] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def get_provider(self, name: str) -> type[BaseProvider] | None:
        """Return the provider class for *name*, importing lazily on first access.

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
        """Register a custom provider (visible globally via the singleton).

        If *name* already exists the entry is overwritten, allowing runtime
        overrides for testing or site-local customizations.
        """
        self._entries[name] = _ProviderEntry(module_path, class_name)
        self._providers.pop(name, None)  # evict cache so next get_provider re-imports

    def list_providers(self) -> list[str]:
        """Return sorted list of all known provider names (built-in + custom)."""
        return sorted(self._entries)

    def load_providers(
        self,
        configured_names: set[str] | None = None,
        excluded_names: set[str] | None = None,
    ) -> dict[str, type[BaseProvider]]:
        """Bulk-load providers, returning ``{name: class}`` dict.

        Used by ``RegistryManager._initialize_providers()`` to delegate the
        import loop while respecting config-driven and exclusion filtering.

        :param configured_names: If given, only load these providers.
        :param excluded_names: If given, skip these providers.
        """
        excluded = excluded_names or set()
        result: dict[str, type[BaseProvider]] = {}

        for name in list(self._entries):
            if name in excluded:
                logger.debug("  Skipping excluded provider: %s", name)
                continue
            if configured_names is not None and name not in configured_names:
                logger.debug("  Skipping unconfigured provider: %s", name)
                continue

            cls = self.get_provider(name)
            if cls is not None:
                result[name] = cls

        return result

    # ── Internal ───────────────────────────────────────────────────────

    def _load(self, name: str, entry: _ProviderEntry) -> type[BaseProvider] | None:
        """Import the provider module and cache the class."""
        try:
            module = importlib.import_module(entry.module_path)
            cls = getattr(module, entry.class_name)
            self._providers[name] = cls
            logger.debug("Loaded provider: %s (%s.%s)", name, entry.module_path, entry.class_name)
            return cls
        except (ImportError, AttributeError) as exc:
            logger.warning("Failed to load provider %s: %s", name, exc)
            return None


# ── Singleton ──────────────────────────────────────────────────────────

_registry: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    """Return the global ``ProviderRegistry`` singleton (created on first call)."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def reset_provider_registry() -> None:
    """Reset the singleton (for test teardown)."""
    global _registry
    _registry = None
