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
from osprey.models.providers.asksage import AskSageProviderAdapter
from osprey.models.providers.base import KEYLESS_API_KEY_PLACEHOLDER, BaseProvider
from osprey.models.providers.health import HealthResult, failure_reason
from osprey.models.providers.litellm_delegating import LiteLLMDelegatingProvider


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


class _ChatViaDelegation(LiteLLMDelegatingProvider):
    """Inherits its chat endpoint from the LiteLLM base, overriding nothing itself."""

    name = "delegating-stub"


class _ImageEmbeddingOnly(BaseProvider):
    name = "image-only-stub"

    def execute_image_embedding(self, inputs, *_args, **_kwargs):
        return [[0.0] for _ in inputs]


class _TextEmbeddingOnly(BaseProvider):
    name = "text-only-stub"

    def execute_embedding(self, texts, *_args, **_kwargs):
        return [[0.0] for _ in texts]


class TestDerivedCapabilities:
    """What a provider serves is read from which endpoint methods it overrides."""

    def test_a_litellm_delegating_subclass_counts_as_chat(self):
        assert _ChatViaDelegation.supports_chat()
        assert not _ChatViaDelegation.supports_embeddings()
        assert not _ChatViaDelegation.supports_image_embeddings()

    def test_overriding_only_image_embedding_is_not_chat_or_text_embedding(self):
        assert _ImageEmbeddingOnly.supports_image_embeddings()
        assert not _ImageEmbeddingOnly.supports_chat()
        assert not _ImageEmbeddingOnly.supports_embeddings()

    def test_overriding_only_text_embedding_is_text_embedding_only(self):
        assert _TextEmbeddingOnly.supports_embeddings()
        assert not _TextEmbeddingOnly.supports_chat()
        assert not _TextEmbeddingOnly.supports_image_embeddings()

    def test_the_base_itself_serves_nothing(self):
        assert not BaseProvider.supports_chat()
        assert not BaseProvider.supports_embeddings()
        assert not BaseProvider.supports_image_embeddings()

    def test_every_registered_chat_provider_supports_chat(self):
        for name, provider_class in _registered_providers():
            if name == "llama-cpp":
                # llama-cpp is registered for embeddings only and serves no chat.
                assert not provider_class.supports_chat()
                continue
            assert provider_class.supports_chat(), f"{name} is registered as chat but serves none"

    def test_capabilities_are_not_declared_facts(self):
        from tests.models.test_provider_declarations import PROVIDER_FACTS

        for derived in ("supports_chat", "supports_embeddings", "supports_image_embeddings"):
            assert derived not in PROVIDER_FACTS


class TestUnservedEndpointDefaults:
    """A provider that does not serve an endpoint says so instead of failing obscurely."""

    def test_unserved_methods_raise_not_implemented(self):
        provider = _ImageEmbeddingOnly()
        with pytest.raises(NotImplementedError):
            provider.execute_completion("hi", "m", None, None)
        with pytest.raises(NotImplementedError):
            provider.execute_embedding(["hi"], "m")
        with pytest.raises(NotImplementedError):
            _TextEmbeddingOnly().execute_image_embedding([(b"x", "image/png"), "hi"], "m")

    def test_chat_health_default_names_the_provider(self):
        assert _ImageEmbeddingOnly().check_health(None, None) == (
            False,
            "image-only-stub has no chat endpoint",
        )

    def test_embedding_health_default_is_a_config_verdict(self):
        verdict = _ChatViaDelegation().check_embedding_health(None, None)
        assert isinstance(verdict, HealthResult)
        assert verdict == HealthResult(False, "delegating-stub has no embedding endpoint", "config")

    def test_the_embedding_health_surface_defaults(self):
        assert BaseProvider.default_embedding_model_id is None
        assert BaseProvider.health_check_embedding_model_id is None


class TestAcceptsChatRequest:
    def test_defaults_to_true(self):
        assert BaseProvider.accepts_chat_request is True

    def test_asksage_declares_false_in_its_own_body(self):
        assert vars(AskSageProviderAdapter)["accepts_chat_request"] is False

    def test_asksage_instantiates_without_the_abstract_method_override(self):
        assert "__abstractmethods__" not in vars(AskSageProviderAdapter) or not (
            AskSageProviderAdapter.__abstractmethods__
        )
        AskSageProviderAdapter()


class _StatusResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def _requests_status_error(status: int):
    import requests

    return requests.HTTPError("boom", response=_StatusResponse(status))


def _httpx_status_error(status: int):
    import httpx

    request = httpx.Request("GET", "http://example.invalid/")
    return httpx.HTTPStatusError(
        "boom", request=request, response=httpx.Response(status, request=request)
    )


def _litellm_error(cls_name: str):
    litellm = pytest.importorskip("litellm")
    import httpx

    cls = getattr(litellm, cls_name)
    if cls_name == "PermissionDeniedError":
        request = httpx.Request("GET", "http://example.invalid/")
        response = httpx.Response(403, request=request)
        return cls(message="boom", llm_provider="openai", model="m", response=response)
    return cls(message="boom", llm_provider="openai", model="m")


class TestFailureReason:
    """One row per entry of the provider failure table; message text is never read."""

    def test_builtin_connection_error_is_unreachable(self):
        assert failure_reason(ConnectionRefusedError("x")) == "unreachable"
        assert failure_reason(ConnectionError("x")) == "unreachable"

    def test_requests_connection_error_is_unreachable(self):
        import requests

        assert failure_reason(requests.ConnectionError("x")) == "unreachable"

    def test_httpx_connect_error_is_unreachable(self):
        import httpx

        assert failure_reason(httpx.ConnectError("x")) == "unreachable"

    def test_litellm_api_connection_error_is_unreachable(self):
        assert failure_reason(_litellm_error("APIConnectionError")) == "unreachable"

    @pytest.mark.parametrize("status", [401, 403])
    def test_requests_auth_status_is_auth(self, status):
        assert failure_reason(_requests_status_error(status)) == "auth"

    @pytest.mark.parametrize("status", [401, 403])
    def test_httpx_auth_status_is_auth(self, status):
        assert failure_reason(_httpx_status_error(status)) == "auth"

    @pytest.mark.parametrize("cls_name", ["AuthenticationError", "PermissionDeniedError"])
    def test_litellm_auth_errors_are_auth(self, cls_name):
        assert failure_reason(_litellm_error(cls_name)) == "auth"

    def test_requests_404_is_model(self):
        assert failure_reason(_requests_status_error(404)) == "model"

    def test_httpx_404_is_model(self):
        assert failure_reason(_httpx_status_error(404)) == "model"

    def test_litellm_not_found_is_model(self):
        assert failure_reason(_litellm_error("NotFoundError")) == "model"

    def test_other_statuses_are_unknown(self):
        assert failure_reason(_requests_status_error(500)) is None
        assert failure_reason(_httpx_status_error(500)) is None

    def test_a_connection_error_that_is_also_a_value_error_is_unreachable(self):
        class Both(ConnectionError, ValueError):
            pass

        assert failure_reason(Both("x")) == "unreachable"

    def test_message_text_is_never_read(self):
        assert failure_reason(ValueError("connection refused")) is None
        assert failure_reason(RuntimeError("401 Unauthorized")) is None
