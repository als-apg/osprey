"""Tests for the resolution rules :class:`BaseProvider` holds for every adapter.

The rule here used to be a per-adapter opt-in: a flag each provider set to turn
the default-endpoint fallback on. It is now derived from what a provider already
declares, so this test sweeps the whole registry rather than naming adapters — a
provider added later inherits the rule instead of having to remember it.
"""

from __future__ import annotations

import pytest

from osprey.models.provider_registry import get_provider_registry
from osprey.models.providers.base import BaseProvider


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
