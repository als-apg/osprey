"""Tests for the resolution rules :class:`BaseProvider` holds for every adapter.

Each rule here used to be a per-adapter opt-in or a hand-copied literal: a flag
each provider set to turn the default-endpoint fallback on, and an ``"EMPTY"``
string substituted in four adapter bodies. Both are now derived from what a
provider already declares, so these tests sweep the whole registry rather than
naming adapters — a provider added later inherits the rules instead of having to
remember them.
"""

from __future__ import annotations

import pytest

from osprey.models.provider_registry import get_provider_registry
from osprey.models.providers.base import KEYLESS_API_KEY_PLACEHOLDER, BaseProvider


@pytest.fixture(autouse=True)
def _no_ambient_base_url_override(monkeypatch):
    """Drop every provider's base_url env override.

    An export supplies an endpoint and beats every other source, which would
    hide exactly the fallback these tests check.
    """
    registry = get_provider_registry()
    for name in registry.list_providers():
        provider_class = registry.get_provider(name)
        if provider_class is not None and provider_class.base_url_env_var:
            monkeypatch.delenv(provider_class.base_url_env_var, raising=False)


def _registered_providers() -> list[tuple[str, type[BaseProvider]]]:
    registry = get_provider_registry()
    loaded = []
    for name in registry.list_providers():
        provider_class = registry.get_provider(name)
        if provider_class is not None:
            loaded.append((name, provider_class))
    return loaded


class TestDefaultBaseURLFallback:
    """A declared default is used exactly when the provider requires an endpoint."""

    def test_a_required_endpoint_falls_back_to_the_declared_default(self):
        checked = []
        for name, provider_class in _registered_providers():
            if not (provider_class.requires_base_url and provider_class.default_base_url):
                continue
            checked.append(name)
            assert provider_class.effective_base_url(None) == provider_class.default_base_url, (
                f"{name} requires a base_url and declares a default the resolver never returns"
            )
        assert checked, "no registered provider both requires a base_url and declares a default"

    def test_an_unrequired_endpoint_keeps_resolving_to_nothing(self):
        """A default on a provider that needs no endpoint stays documentation.

        ``openai`` declares ``https://api.openai.com/v1`` and requires no
        base_url: litellm derives the endpoint from the model prefix, so
        forwarding the default would pin a route the client is meant to choose.
        """
        checked = []
        for name, provider_class in _registered_providers():
            if provider_class.requires_base_url or not provider_class.default_base_url:
                continue
            checked.append(name)
            assert provider_class.effective_base_url(None) is None, (
                f"{name} declares a default it does not require; it must not be forwarded"
            )
        assert checked, "no registered provider declares a default without requiring one"

    def test_a_configured_value_still_beats_the_default(self):
        for name, provider_class in _registered_providers():
            if not provider_class.default_base_url:
                continue
            assert provider_class.effective_base_url("https://configured.example") == (
                "https://configured.example"
            ), f"{name} overrode a configured base_url with its default"


class TestKeylessAPIKeyPlaceholder:
    """An absent key is substituted exactly for providers that declare none is needed."""

    def test_a_keyless_provider_substitutes_the_placeholder(self):
        checked = []
        for name, provider_class in _registered_providers():
            if provider_class.requires_api_key:
                continue
            checked.append(name)
            assert provider_class.effective_api_key(None) == KEYLESS_API_KEY_PLACEHOLDER, (
                f"{name} declares no API key is needed but sends nothing on the wire"
            )
        assert checked, "no registered provider declares itself keyless"

    def test_a_key_requiring_provider_passes_an_absent_key_through(self):
        """The requirement gate refuses a missing key, not the adapter.

        Substituting here would turn "you forgot the key" into an
        authentication failure from the vendor.
        """
        checked = []
        for name, provider_class in _registered_providers():
            if not provider_class.requires_api_key:
                continue
            checked.append(name)
            assert provider_class.effective_api_key(None) is None, (
                f"{name} requires an API key but invented one for an absent value"
            )
        assert checked, "no registered provider requires an API key"

    def test_a_supplied_key_is_never_replaced(self):
        for _name, provider_class in _registered_providers():
            assert provider_class.effective_api_key("real-key") == "real-key"
