"""Resolve an ARIEL module's embedding provider from its configuration.

Two steps, kept apart so configuration never waits on the network:

- :func:`resolve_provider` turns a module's ``provider`` value into the adapter
  class, one adapter instance, the configured base URL and the API key. It does
  no I/O and never raises for reachability; a provider the registry does not
  know, one that does not serve what the module needs, or a base URL the
  adapter refuses raises :class:`ModuleConfigError` naming the module's
  provider key.
- :func:`resolve_reachable_base_url` finds the URL a local model server
  actually answers on (the class's environment override, the configured URL,
  its container fallbacks) for adapters that declare
  ``resolves_fallback_outside_calls``. It blocks, so callers run it off the
  event loop or on a path that already blocks.

Base URL precedence, for a provider name and an inline provider mapping alike:
the inline ``base_url``, else the provider's ``api.providers`` entry, else the
adapter's own default (by the class's ``effective_base_url`` rule).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from osprey.services.ariel_search.exceptions import ModuleConfigError
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.models.providers.base import BaseProvider

logger = get_logger("ariel")

#: Seconds each reachability probe waits for an answer.
PROBE_TIMEOUT_S = 2.0

#: What a module needs its provider to serve.
Serves = Literal["embeddings", "image_embeddings"]


@dataclass(frozen=True)
class ResolvedProvider:
    """A module's provider, resolved without touching the network.

    Attributes:
        cls: The adapter class; it answers the provider facts
            (``supports_*``, ``validate_base_url``, ``truncates_to_dimensions``).
        instance: The one adapter instance, ``cls()``, that takes every call
            (``execute_embedding``, ``execute_image_embedding``,
            ``check_embedding_health``).
        base_url: The configured base URL (inline > ``api.providers`` entry >
            adapter default), or ``None`` for a provider that needs none.
        api_key: The configured API key, or ``None``.
    """

    cls: type[BaseProvider]
    instance: BaseProvider
    base_url: str | None
    api_key: str | None


def provider_name(
    provider: str | Mapping[str, Any] | None, *, provider_key: str, default: str | None
) -> str:
    """Name a module's provider: the name, an inline mapping's ``name``, else ``default``.

    Args:
        provider: The module's ``provider`` value: a provider name, an inline
            mapping ``{name, base_url, api_key}``, or ``None``.
        provider_key: The config key ``provider`` came from, for error messages.
        default: The provider name used when none is given; ``None`` makes a
            provider required.

    Returns:
        The provider name.

    Raises:
        ModuleConfigError: If no provider is named and there is no default, or
            ``provider`` is neither a name nor a mapping.
    """
    if isinstance(provider, Mapping):
        name = provider.get("name") or default
    elif provider is None or provider == "":
        name = default
    elif isinstance(provider, str):
        name = provider
    else:
        raise ModuleConfigError(
            f"{provider_key}: expected a provider name or a mapping "
            f"{{name, base_url, api_key}}, got {type(provider).__name__}",
            key=provider_key,
        )
    if not name or not isinstance(name, str):
        raise ModuleConfigError(f"{provider_key}: no provider is configured", key=provider_key)
    return name


def provider_given(provider: Any) -> bool:
    """Whether a module's ``provider`` value names a provider at all.

    A mapping counts only with a non-blank ``name``, a string only when not
    blank; any other non-``None`` value counts, so :func:`provider_name` can
    refuse its type.
    """
    if isinstance(provider, Mapping):
        name = provider.get("name")
        return isinstance(name, str) and bool(name.strip())
    if isinstance(provider, str):
        return bool(provider.strip())
    return provider is not None


def provider_settings(
    provider: str | Mapping[str, Any] | None, *, provider_key: str, default: str | None
) -> tuple[str, str | None, str | None]:
    """Name a module's provider and read its raw base URL and API key.

    Args:
        provider: The module's ``provider`` value: a provider name, an inline
            mapping ``{name, base_url, api_key}``, or ``None``.
        provider_key: The config key ``provider`` came from, for error messages.
        default: The provider name used when none is given; ``None`` makes a
            provider required.

    Returns:
        ``(name, base_url, api_key)``; ``base_url`` is the inline value, else the
        ``api.providers`` entry's, else ``None`` (the adapter default applies
        later). ``api_key`` follows the same order.

    Raises:
        ModuleConfigError: As :func:`provider_name`.
    """
    name = provider_name(provider, provider_key=provider_key, default=default)
    inline: Mapping[str, Any] = provider if isinstance(provider, Mapping) else {}
    entry = _provider_entry(name)
    base_url = inline.get("base_url")
    if base_url is None:
        base_url = entry.get("base_url")
    api_key = inline.get("api_key")
    if api_key is None:
        api_key = entry.get("api_key")
    return name, base_url, api_key


def resolve_provider_class(
    provider: str | Mapping[str, Any] | None,
    *,
    provider_key: str,
    default: str | None,
    serves: Serves = "embeddings",
) -> type[BaseProvider]:
    """Look a module's provider up in the provider registry; reads no configuration file.

    Args:
        provider: The module's ``provider`` value (name, inline mapping or ``None``).
        provider_key: The config key ``provider`` came from; every refusal names it.
        default: The provider name used when none is given; ``None`` makes a
            provider required.
        serves: What the module needs the provider to serve.

    Returns:
        The adapter class.

    Raises:
        ModuleConfigError: If no provider is configured and there is no
            default, or the provider is unknown or does not serve ``serves``.
    """
    from osprey.models.provider_registry import get_provider_registry

    name = provider_name(provider, provider_key=provider_key, default=default)
    provider_cls = get_provider_registry().get_provider(name)
    supported = provider_cls is not None and (
        provider_cls.supports_image_embeddings()
        if serves == "image_embeddings"
        else provider_cls.supports_embeddings()
    )
    if provider_cls is None or not supported:
        what = "image embeddings" if serves == "image_embeddings" else "embeddings"
        raise ModuleConfigError(
            f"{provider_key}: {name!r} is not a provider that serves {what}", key=provider_key
        )
    return provider_cls


def resolve_provider(
    provider: str | Mapping[str, Any] | None,
    *,
    provider_key: str,
    default: str | None,
    serves: Serves = "embeddings",
) -> ResolvedProvider:
    """Resolve a module's provider to its adapter class, instance, base URL and key.

    Does no network I/O. The base URL is checked with the class's
    ``validate_base_url``, so a URL the adapter can never use (a llama-cpp
    ``…/v1`` base) is refused here, at configuration time.

    Args:
        provider: The module's ``provider`` value: a provider name, an inline
            mapping ``{name, base_url, api_key}``, or ``None``.
        provider_key: The config key ``provider`` came from; every refusal
            names it.
        default: The provider name used when none is configured; ``None``
            makes a provider required.
        serves: What the module needs the provider to serve.

    Returns:
        The resolved provider.

    Raises:
        ModuleConfigError: If no provider is configured and there is no
            default, the provider is unknown or does not serve ``serves``, or
            the adapter refuses the base URL.
    """
    provider_cls = resolve_provider_class(
        provider, provider_key=provider_key, default=default, serves=serves
    )
    _, raw_base_url, api_key = provider_settings(
        provider, provider_key=provider_key, default=default
    )
    base_url = provider_cls.effective_base_url(raw_base_url)
    try:
        provider_cls.validate_base_url(base_url)
    except ValueError as e:
        raise ModuleConfigError(f"{provider_key}: {e}", key=provider_key) from e

    return ResolvedProvider(
        cls=provider_cls, instance=provider_cls(), base_url=base_url, api_key=api_key
    )


def resolve_reachable_base_url(
    provider_cls: type[BaseProvider],
    configured: str,
    *,
    deadline_s: float | None = None,
    refresh: bool = False,
) -> str:
    """The URL a local model server answers on, for adapters resolved outside calls.

    For a class declaring ``resolves_fallback_outside_calls`` the walk tries the
    class's ``host_override_env_var``, then ``configured``, then its container
    fallbacks (port from the class's ``default_base_url``), probing
    ``fallback_probe_path`` with a 2 s timeout. The answer is cached in the one
    local-server cache shared with every other caller. Any other class gets
    ``configured`` back unchanged: it resolves inside its own calls.

    Blocking; never raises for reachability.

    Args:
        provider_cls: The adapter class.
        configured: The configured base URL.
        deadline_s: Stop the candidate walk after this many seconds and return
            ``configured``; a walk that finishes later still writes the cache.
        refresh: Give a cached URL one probe and walk again when it fails, so a
            server that moved is found without a restart.

    Returns:
        The first answering URL (cached), else ``configured``.
    """
    if not getattr(provider_cls, "resolves_fallback_outside_calls", False):
        return configured
    probe_path = getattr(provider_cls, "fallback_probe_path", None)
    if not probe_path:
        return configured

    from osprey.models.providers import _local_server

    default = urlsplit(provider_cls.default_base_url or "")
    default_port = default.port or (443 if default.scheme == "https" else 80)
    return _local_server.resolve_cached(
        configured,
        probe_path=probe_path,
        env_var=provider_cls.host_override_env_var,
        default_port=default_port,
        refresh=refresh,
        deadline_s=deadline_s,
        timeout=PROBE_TIMEOUT_S,
    )


def _provider_entry(name: str) -> Mapping[str, Any]:
    """The provider's ``api.providers`` entry, or empty when there is no config file."""
    from osprey.models.config import get_provider_config

    try:
        entry = get_provider_config(name)
    except FileNotFoundError:
        logger.debug(f"No config.yml found, using empty provider config for '{name}'")
        return {}
    return entry or {}
