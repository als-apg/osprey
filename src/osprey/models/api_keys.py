"""What counts as "no API key" for a provider entry, and which variable holds it.

The completion entry point refuses a call on this rule, and
:mod:`osprey.models.inner_calls` reports a feature on the same rule, so the two
agree. It lives below both: the entry point does not depend on the feature list.
"""

from __future__ import annotations

from osprey_connectors.config import is_unresolved_placeholder, unresolved_placeholders

__all__ = ["api_key_env_var", "api_key_missing"]


def api_key_missing(api_key: object) -> bool:
    """Whether a provider entry's ``api_key`` is no key at all.

    Empty or absent, or a ``${VAR}`` reference the config loader left verbatim
    because ``VAR`` is unset
    (:func:`osprey_connectors.config.is_unresolved_placeholder`). The second
    case is a non-empty string, so a bare truth test would send it to the
    provider as the key.
    """
    return not api_key or is_unresolved_placeholder(api_key)


def api_key_env_var(provider: str, api_key: object = None) -> str | None:
    """The environment variable *provider*'s API key arrives in.

    The variable the config entry itself references when *api_key* is an
    unresolved ``${VAR}``, since that is the one the deployment reads; else the
    one the provider declares. ``None`` when neither names one.

    Args:
        provider: A provider name.
        api_key: The provider entry's ``api_key`` value, raw or resolved.
    """
    if is_unresolved_placeholder(api_key):
        return _placeholder_variable(api_key)
    from osprey.models.provider_registry import get_provider_registry

    return get_provider_registry().api_key_env_var(provider)


def _placeholder_variable(placeholder: object) -> str:
    """The variable a lone unresolved ``${VAR}`` or ``$VAR`` names."""
    return unresolved_placeholders(placeholder)[0].strip("${}")
