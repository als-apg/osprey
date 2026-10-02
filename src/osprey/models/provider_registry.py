"""Lightweight Provider Registry — lazy-loaded provider class resolution.

Standalone singleton that resolves provider names to BaseProvider subclasses
without depending on the full RegistryManager.

Any component needing LLM access (MCP tools, FastAPI routes, CLI commands) can
call ``get_provider_registry().get_provider("cborg")`` to obtain the provider
class, then use it via ``get_chat_completion()`` or ``aget_chat_completion()``.

Adding a new built-in provider = one entry in ``_BUILTIN_PROVIDERS`` below.
The entry names the provider's key variable and launch protocol, which the
adapter class declares too; a parity test keeps the two equal.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from osprey.models.providers.base import BaseProvider

logger = logging.getLogger("osprey.models.provider_registry")


@dataclass(frozen=True, slots=True)
class _ProviderEntry:
    """Lazy-load descriptor for a provider class.

    A built-in entry restates two class attributes of its adapter,
    ``api_key_env_var`` and ``api_protocol``, so the facts every build and
    launch reads come from this table without importing the adapter; a parity
    test holds each to its class. ``key_env_var = None`` on a built-in means the
    provider is keyless. An entry made by ``ProviderRegistry.register_provider``
    carries neither fact (``api_protocol is None``), and its class is the only
    declaration.
    """

    module_path: str
    class_name: str
    key_env_var: str | None = None
    api_protocol: Literal["anthropic", "openai"] | None = None


# ── Built-in provider table (single source of truth) ──────────────────
_BUILTIN_PROVIDERS: dict[str, _ProviderEntry] = {
    "anthropic": _ProviderEntry(
        "osprey.models.providers.anthropic",
        "AnthropicProviderAdapter",
        key_env_var="ANTHROPIC_API_KEY",
        api_protocol="anthropic",
    ),
    "openai": _ProviderEntry(
        "osprey.models.providers.openai",
        "OpenAIProviderAdapter",
        key_env_var="OPENAI_API_KEY",
        api_protocol="openai",
    ),
    "google": _ProviderEntry(
        "osprey.models.providers.google",
        "GoogleProviderAdapter",
        key_env_var="GOOGLE_API_KEY",
        api_protocol="openai",
    ),
    "cborg": _ProviderEntry(
        "osprey.models.providers.cborg",
        "CBorgProviderAdapter",
        key_env_var="CBORG_API_KEY",
        api_protocol="anthropic",
    ),
    "amsc-i2": _ProviderEntry(
        "osprey.models.providers.amsc_i2",
        "AMSCI2ProviderAdapter",
        key_env_var="AMSC_I2_API_KEY",
        api_protocol="openai",
    ),
    "argo": _ProviderEntry(
        "osprey.models.providers.argo",
        "ArgoProviderAdapter",
        key_env_var="ARGO_API_KEY",
        api_protocol="openai",
    ),
    "stanford": _ProviderEntry(
        "osprey.models.providers.stanford",
        "StanfordProviderAdapter",
        key_env_var="STANFORD_API_KEY",
        api_protocol="openai",
    ),
    "als-apg": _ProviderEntry(
        "osprey.models.providers.als_apg",
        "ALSAPGProviderAdapter",
        key_env_var="ALS_APG_API_KEY",
        api_protocol="anthropic",
    ),
    "ollama": _ProviderEntry(
        "osprey.models.providers.ollama",
        "OllamaProviderAdapter",
        key_env_var=None,
        api_protocol="openai",
    ),
    "asksage": _ProviderEntry(
        "osprey.models.providers.asksage",
        "AskSageProviderAdapter",
        key_env_var="ASKSAGE_API_KEY",
        api_protocol="openai",
    ),
    "vllm": _ProviderEntry(
        "osprey.models.providers.vllm",
        "VLLMProviderAdapter",
        key_env_var=None,  # local, no key
        api_protocol="openai",
    ),
    "ds4": _ProviderEntry(
        "osprey.models.providers.ds4",
        "DS4ProviderAdapter",
        key_env_var=None,  # local DwarfStar server, no key
        api_protocol="openai",
    ),
}


# ── Provider → API key env var (read from the table above) ────────────
# Maps each built-in provider to the environment variable its API key arrives
# in; ``None`` means the provider is keyless. To add a provider, add an entry
# to ``_BUILTIN_PROVIDERS``. A name registered at run time is answered by
# ``ProviderRegistry.api_key_env_var``, not by this view. Read-only, so no
# caller can add a provider here instead of to the table.
PROVIDER_API_KEYS: Mapping[str, str | None] = MappingProxyType(
    {name: entry.key_env_var for name, entry in _BUILTIN_PROVIDERS.items()}
)


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

    def exclude(self, name: str) -> None:
        """Remove *name* from the registry.

        ``get_provider(name)`` returns ``None`` afterwards and ``list_providers()``
        no longer lists it. A later ``register_provider`` under the same name adds
        it back.
        """
        self._entries.pop(name, None)
        self._providers.pop(name, None)
        logger.debug("Excluded provider: %s", name)

    def list_providers(self) -> list[str]:
        """Return a sorted list of every provider name the registry resolves.

        Built-in and registered, less excluded.
        """
        return sorted(self._entries)

    def api_key_env_var(self, name: str) -> str | None:
        """Return the variable *name*'s API key arrives in.

        A built-in provider that has not been replaced answers from its entry,
        without importing its adapter; a registered provider answers from its
        class. Returns ``None`` when the provider is keyless, unknown, or
        registered with a class that does not load.
        """
        entry = self._entries.get(name)
        if entry is None:
            return None
        if entry.api_protocol is not None:
            return entry.key_env_var
        cls = self.get_provider(name)
        if cls is None:
            return None
        return cls.api_key_env_var

    def api_protocol(self, name: str) -> Literal["anthropic", "openai"] | None:
        """Return the protocol a launch speaks to *name*, as its provider declares it.

        A built-in provider that has not been replaced answers from its entry,
        without importing its adapter; a registered provider answers from its
        class, including one registered under a built-in name. Returns ``None``
        for a name no entry describes, or one registered with a class that does
        not load.
        """
        entry = self._entries.get(name)
        if entry is None:
            return None
        if entry.api_protocol is not None:
            return entry.api_protocol
        cls = self.get_provider(name)
        if cls is None:
            return None
        return cls.api_protocol

    def api_key_env_vars(self) -> dict[str, str | None]:
        """Map every provider the registry holds to its API-key variable.

        Built-ins come first in table order, then registrations in registration
        order; a keyless provider maps to ``None``. A registered name whose
        class does not load is left out, since it declares nothing.
        """
        result: dict[str, str | None] = {}
        for name, entry in list(self._entries.items()):
            if entry.api_protocol is None and self.get_provider(name) is None:
                continue
            result[name] = self.api_key_env_var(name)
        return result

    def load_providers(
        self,
        configured_names: set[str] | None = None,
    ) -> dict[str, type[BaseProvider]]:
        """Bulk-load providers, returning ``{name: class}`` dict.

        Used by ``initialize_providers()`` to delegate the import loop while
        respecting config-driven filtering; an excluded provider has already left
        the table.

        :param configured_names: If given, only load these providers.
        """
        result: dict[str, type[BaseProvider]] = {}

        for name in list(self._entries):
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
            cls: type[BaseProvider] = getattr(module, entry.class_name)
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
