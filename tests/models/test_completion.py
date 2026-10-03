"""Tests for chat completion module."""

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from osprey.models.completion import (
    _convert_typed_dict_to_pydantic,
    _is_typed_dict,
)


class TestIsTypedDict:
    """Test TypedDict detection utility."""

    def test_detects_typed_dict(self):
        """Test that actual TypedDict is correctly identified."""

        class MyTypedDict(TypedDict):
            field1: str
            field2: int

        assert _is_typed_dict(MyTypedDict) is True

    def test_rejects_regular_class(self):
        """Test that regular classes are not identified as TypedDict."""

        class RegularClass:
            field1: str
            field2: int

        assert _is_typed_dict(RegularClass) is False

    def test_rejects_pydantic_model(self):
        """Test that Pydantic models are not identified as TypedDict."""

        class PydanticModel(BaseModel):
            field1: str
            field2: int

        assert _is_typed_dict(PydanticModel) is False

    def test_rejects_none(self):
        """Test that None is not identified as TypedDict."""
        assert _is_typed_dict(None) is False

    def test_rejects_plain_dict(self):
        """Test that plain dict instances are not identified as TypedDict."""
        assert _is_typed_dict(dict) is False


class TestConvertTypedDictToPydantic:
    """Test TypedDict to Pydantic model conversion."""

    def test_converts_simple_typed_dict(self):
        """Test conversion of simple TypedDict to Pydantic model."""

        class SimpleTypedDict(TypedDict):
            name: str
            age: int

        pydantic_model = _convert_typed_dict_to_pydantic(SimpleTypedDict)

        # Check that result is a Pydantic model
        assert issubclass(pydantic_model, BaseModel)

        # Check that fields are preserved
        assert "name" in pydantic_model.model_fields
        assert "age" in pydantic_model.model_fields

        # Check that we can instantiate it
        instance = pydantic_model(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30

    def test_converts_nested_typed_dict(self):
        """Test conversion handles nested type annotations."""

        class AddressTypedDict(TypedDict):
            street: str
            city: str

        class PersonTypedDict(TypedDict):
            name: str
            addresses: list

        pydantic_model = _convert_typed_dict_to_pydantic(PersonTypedDict)

        # Check fields exist
        assert "name" in pydantic_model.model_fields
        assert "addresses" in pydantic_model.model_fields

        # Check instantiation
        instance = pydantic_model(name="Bob", addresses=[])
        assert instance.name == "Bob"

    def test_raises_on_non_typed_dict(self):
        """Test that conversion raises ValueError for non-TypedDict classes."""

        class NotATypedDict:
            field: str

        with pytest.raises(ValueError, match="Expected TypedDict"):
            _convert_typed_dict_to_pydantic(NotATypedDict)

    def test_model_name_has_pydantic_suffix(self):
        """Test that generated Pydantic model has 'Pydantic' suffix."""

        class MyData(TypedDict):
            value: str

        pydantic_model = _convert_typed_dict_to_pydantic(MyData)
        assert pydantic_model.__name__ == "MyDataPydantic"

    def test_preserves_field_types(self):
        """Test that field types are preserved during conversion."""

        class TypedData(TypedDict):
            text: str
            count: int
            active: bool
            items: list

        pydantic_model = _convert_typed_dict_to_pydantic(TypedData)

        # Check field types in model fields
        fields = pydantic_model.model_fields
        assert fields["text"].annotation is str
        assert fields["count"].annotation is int
        assert fields["active"].annotation is bool
        assert fields["items"].annotation is list


# Every provider that both requires a base_url and declares its own default
# endpoint, with the endpoint it must resolve to when nothing else supplies one.
# Kept explicit so a new provider forces a deliberate entry here rather than
# silently inheriting whichever behavior its class attributes happen to give it.
PROVIDERS_DECLARING_A_DEFAULT_ENDPOINT = [
    ("als-apg", "https://llm.als.lbl.gov/v1"),
    ("argo", "https://apps.inside.anl.gov/argoapi/v1"),
    ("ds4", "http://127.0.0.1:8000/v1"),
    ("ollama", "http://localhost:11434"),
    ("stanford", "https://aiapi-prod.stanford.edu/v1"),
    ("vllm", "http://localhost:8000/v1"),
]

# Providers with no chat route that require a base_url and declare a default.
# Only the resolver applies to them: get_chat_completion refuses them from the
# registry table before any adapter runs, so no completion call reaches the default.
EMBEDDING_ONLY_PROVIDERS_DECLARING_A_DEFAULT_ENDPOINT = [
    ("llama-cpp", "http://localhost:8080"),
]

# Providers that require a base_url and declare no default: nothing but config
# (or an env override) can supply their endpoint, so the gate must keep
# rejecting them.
PROVIDERS_WITH_NO_ENDPOINT_SOURCE = ["amsc-i2", "asksage", "cborg"]


def _clear_base_url_overrides(monkeypatch):
    """Remove every provider's base_url env override for the duration of a test.

    An override supplies a base_url and would hide exactly what these tests
    check. The overrides' own behavior is covered in the provider-adapter tests.
    """
    from osprey.models.provider_registry import get_provider_registry

    registry = get_provider_registry()
    for name in registry.list_providers():
        provider_class = registry.get_provider(name)
        if provider_class is not None and provider_class.base_url_env_var:
            monkeypatch.delenv(provider_class.base_url_env_var, raising=False)


class TestBaseUrlRequirementHonorsProviderDefaults:
    """The ``requires_base_url`` gate must agree with what the provider resolves.

    Both layers already existed and both were individually correct: the adapter
    resolved a missing base_url to its ``default_base_url``, and the gate rejected
    a missing base_url. What nobody checked is that the gate ran FIRST, so a
    provider carrying a perfectly good default was refused for "missing" base_url
    and its default was unreachable.

    These tests pin the agreement rather than either half, since either half
    alone passes while the pair is broken.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_override(self, monkeypatch):
        _clear_base_url_overrides(monkeypatch)

    @pytest.mark.parametrize(
        ("provider", "expected"),
        PROVIDERS_DECLARING_A_DEFAULT_ENDPOINT
        + EMBEDDING_ONLY_PROVIDERS_DECLARING_A_DEFAULT_ENDPOINT,
    )
    def test_a_declared_default_satisfies_the_requirement(self, provider, expected):
        from osprey.models.provider_registry import get_provider_registry

        provider_class = get_provider_registry().get_provider(provider)
        assert provider_class.requires_base_url, f"{provider} must still require a base_url"
        # None in, the declared default out — the value the gate must accept.
        assert provider_class.effective_base_url(None) == expected

    def test_an_env_override_supplies_and_beats_a_configured_value(self, monkeypatch):
        from osprey.models.provider_registry import get_provider_registry

        monkeypatch.setenv("ALS_APG_BASE_URL", "https://fallback.example")
        provider_class = get_provider_registry().get_provider("als-apg")
        assert provider_class.effective_base_url(None) == "https://fallback.example"
        # ...and beats an explicitly configured value, which is the whole point of
        # a break-glass redirect for an already-deployed system.
        assert provider_class.effective_base_url("https://baked-in") == "https://fallback.example"

    @pytest.mark.parametrize("provider", ["als-apg", "cborg", "vllm"])
    def test_an_unresolved_placeholder_counts_as_no_value(self, provider):
        """``base_url: ${VAR}`` with nothing exported is "unset", not a hostname.

        The config resolver keeps the reference verbatim when the variable is
        unset, so every provider has to read that shape as an absent value or
        the literal reaches the HTTP client.
        """
        from osprey.models.provider_registry import get_provider_registry

        provider_class = get_provider_registry().get_provider(provider)
        resolved = provider_class.effective_base_url("${SOME_GATEWAY_URL}")
        assert resolved == provider_class.default_base_url

    def test_a_configured_value_still_wins_over_the_default(self):
        from osprey.models.provider_registry import get_provider_registry

        provider_class = get_provider_registry().get_provider("stanford")
        assert provider_class.effective_base_url("https://configured") == "https://configured"

    @pytest.mark.parametrize("provider", PROVIDERS_WITH_NO_ENDPOINT_SOURCE)
    def test_a_provider_with_no_default_still_resolves_to_none(self, provider):
        # The gate must keep rejecting a provider that genuinely has no endpoint
        # source; the fix widens what counts as "supplied", not what counts as
        # required. These providers declare no default_base_url at all, so config
        # is their only source — unlike vllm, which does declare one.
        from osprey.models.provider_registry import get_provider_registry

        provider_class = get_provider_registry().get_provider(provider)
        assert provider_class is not None
        assert provider_class.requires_base_url
        assert provider_class.default_base_url is None
        assert provider_class.effective_base_url(None) is None

    def test_no_registered_provider_declares_an_unreachable_default(self):
        # The sweep, so a provider added later cannot reintroduce the shape:
        # requires_base_url + a declared default that the resolver never returns,
        # which the gate then rejects as "missing" while the adapter body would
        # have used it.
        from osprey.models.provider_registry import get_provider_registry

        registry = get_provider_registry()
        unreachable = []
        for name in registry.list_providers():
            provider_class = registry.get_provider(name)
            if provider_class is None or not provider_class.default_base_url:
                continue
            if not provider_class.requires_base_url:
                continue
            if provider_class.effective_base_url(None) != provider_class.default_base_url:
                unreachable.append(name)

        assert unreachable == [], (
            f"providers declare a default_base_url the requirement gate rejects: {unreachable}"
        )


class TestGetChatCompletionAcceptsAProviderDefault:
    """The end-to-end shape of the regression, through the public entry point.

    The class above pins the resolver; this pins the GATE, and only this one
    would have caught the bug. The resolver was always right — ``check_health``
    with ``base_url=None`` already resolved to the default and its adapter test
    passed throughout. What failed was ``get_chat_completion`` rejecting the call
    before the adapter was ever reached, which no resolver test can observe.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_override(self, monkeypatch):
        _clear_base_url_overrides(monkeypatch)

    @pytest.mark.parametrize(("provider", "expected"), PROVIDERS_DECLARING_A_DEFAULT_ENDPOINT)
    def test_no_base_url_anywhere_reaches_the_provider_on_its_default(
        self, provider, expected, monkeypatch
    ):
        from osprey.models import completion as completion_module
        from osprey.models.provider_registry import get_provider_registry

        seen: dict = {}

        def fake_execute(self, **kwargs):  # noqa: ARG001 - stands in for the provider adapter's execute, which collects keyword arguments
            seen.update(kwargs)
            return "ok"

        # A config with credentials but NO base_url — a deployment relying on the
        # provider's own default, which is the state the removed env override left.
        monkeypatch.setattr(
            completion_module,
            "get_provider_config",
            lambda provider: {"api_key": "k", "default_model_id": "a-model"},
        )
        provider_class = get_provider_registry().get_provider(provider)
        monkeypatch.setattr(provider_class, "execute_completion", fake_execute)

        result = completion_module.get_chat_completion(
            message="ping", provider=provider, max_tokens=4
        )

        assert result == "ok"
        assert seen["base_url"] == expected

    def test_a_provider_with_no_endpoint_source_is_still_rejected(self, monkeypatch):
        # The gate must not become a rubber stamp: a requires_base_url provider
        # with nothing to fall back on still fails, and still names itself.
        from osprey.models import completion as completion_module

        monkeypatch.setattr(
            completion_module,
            "get_provider_config",
            lambda provider: {"api_key": "k", "default_model_id": "anthropic/claude-haiku"},
        )

        with pytest.raises(ValueError, match="Base URL required for cborg"):
            completion_module.get_chat_completion(message="ping", provider="cborg", max_tokens=4)

    def test_an_unresolved_placeholder_is_rejected_like_a_missing_url(self, monkeypatch):
        """A deployment that never exported its gateway variable is refused.

        A config may spell an endpoint as a reference; unset, the literal
        survives config resolution, and the gate has to read it as "no URL"
        rather than let it through to the HTTP client.
        """
        from osprey.models import completion as completion_module

        monkeypatch.setattr(
            completion_module,
            "get_provider_config",
            lambda provider: {
                "api_key": "k",
                "base_url": "${CBORG_BASE_URL}",
                "default_model_id": "anthropic/claude-haiku",
            },
        )

        with pytest.raises(ValueError, match="Base URL required for cborg"):
            completion_module.get_chat_completion(message="ping", provider="cborg", max_tokens=4)

    def test_provider_config_extra_body_reaches_provider(self, monkeypatch):
        """Provider catalog entries may carry LiteLLM request-body extensions.

        This is the shape needed for a Delphi-fronted BYOK provider: the normal
        provider api_key authenticates to Delphi, while extra_body.api_key is
        the user's upstream key forwarded by LiteLLM clientside auth.
        """
        from osprey.models import completion as completion_module
        from osprey.models.provider_registry import get_provider_registry

        seen: dict = {}

        def fake_execute(self, **kwargs):  # noqa: ARG001 - stands in for the provider adapter's execute, which collects keyword arguments
            seen.update(kwargs)
            return "ok"

        monkeypatch.setattr(
            completion_module,
            "get_provider_config",
            lambda provider: {
                "api_key": "delphi-key",
                "base_url": "http://127.0.0.1:4000/v1",
                "default_model_id": "amsc/gpt-oss-120b-safeguard",
                "extra_body": {"api_key": "upstream-amsc-key"},
            },
        )
        provider_class = get_provider_registry().get_provider("amsc-i2")
        monkeypatch.setattr(provider_class, "execute_completion", fake_execute)

        result = completion_module.get_chat_completion(
            message="ping",
            provider="amsc-i2",
            max_tokens=4,
        )

        assert result == "ok"
        assert seen["api_key"] == "delphi-key"
        assert seen["base_url"] == "http://127.0.0.1:4000/v1"
        assert seen["model_id"] == "amsc/gpt-oss-120b-safeguard"
        assert seen["extra_body"] == {"api_key": "upstream-amsc-key"}


class TestCompletionNamesItsProviderAndModel:
    """A completion runs only once it knows which provider and which model it calls."""

    def test_a_provider_that_waives_the_model_id_is_still_refused_without_one(self, monkeypatch):
        from osprey.models import completion as completion_module
        from osprey.models.provider_registry import get_provider_registry

        calls: list[dict] = []

        def fake_execute(self, **kwargs):  # noqa: ARG001 - stands in for the provider adapter's execute, which collects keyword arguments
            calls.append(kwargs)
            return "ok"

        monkeypatch.setattr(
            completion_module, "get_provider_config", lambda provider: {"api_key": "k"}
        )
        cls = get_provider_registry().get_provider("anthropic")
        monkeypatch.setattr(cls, "requires_model_id", False)
        monkeypatch.setattr(cls, "execute_completion", fake_execute)

        with pytest.raises(ValueError, match="Model ID required for anthropic"):
            completion_module.get_chat_completion(
                message="ping", provider="anthropic", max_tokens=4
            )
        assert calls == []

    def test_a_model_config_without_a_provider_is_refused_by_name(self):
        from osprey.models import completion as completion_module

        with pytest.raises(
            ValueError, match="Provider must be specified either directly or via model_config"
        ):
            completion_module.get_chat_completion(message="ping", model_config={"model_id": "m"})


def _text_response(content: str = "ok"):
    """A LiteLLM-shaped completion response carrying plain text."""
    from unittest.mock import MagicMock

    message = MagicMock()
    message.tool_calls = None
    message.content = content
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


class TestChatCompletionTimeoutAndRetries:
    """A caller bounds a chat call by its own timeout and retry count."""

    @pytest.fixture
    def openai_config(self, monkeypatch):
        from osprey.models import completion as completion_module

        monkeypatch.setattr(
            completion_module,
            "get_provider_config",
            lambda provider: {"api_key": "k", "default_model_id": "gpt-4o"},
        )

    def test_timeout_and_zero_retries_reach_litellm(self, openai_config):  # noqa: ARG002 - fixture patches the config
        """A given timeout and num_retries=0 are what litellm.completion receives."""
        from unittest.mock import patch

        from osprey.models import completion as completion_module

        with patch("litellm.completion", return_value=_text_response()) as mock_completion:
            result = completion_module.get_chat_completion(
                message="ping", provider="openai", max_tokens=4, timeout=7.5, num_retries=0
            )

        assert result == "ok"
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["timeout"] == 7.5
        assert kwargs["num_retries"] == 0

    def test_defaults_send_two_retries_and_no_timeout(self, openai_config):  # noqa: ARG002 - fixture patches the config
        """With timeout=None nothing is sent for it, and retries default to two."""
        from unittest.mock import patch

        from osprey.models import completion as completion_module

        with patch("litellm.completion", return_value=_text_response()) as mock_completion:
            completion_module.get_chat_completion(
                message="ping", provider="openai", max_tokens=4, timeout=None
            )

        kwargs = mock_completion.call_args.kwargs
        assert "timeout" not in kwargs
        assert kwargs["num_retries"] == 2

    def test_timeout_reaches_the_provider_adapter(self, monkeypatch):
        """The timeout travels through completion_kwargs to the adapter."""
        from osprey.models import completion as completion_module
        from osprey.models.provider_registry import get_provider_registry

        seen: dict = {}

        def fake_execute(self, **kwargs):  # noqa: ARG001 - stands in for the provider adapter's execute
            seen.update(kwargs)
            return "ok"

        monkeypatch.setattr(
            completion_module,
            "get_provider_config",
            lambda provider: {"base_url": "http://localhost:11434"},
        )
        cls = get_provider_registry().get_provider("ollama")
        monkeypatch.setattr(cls, "execute_completion", fake_execute)

        completion_module.get_chat_completion(
            message="ping", provider="ollama", model_id="m", timeout=3.0
        )

        assert seen["timeout"] == 3.0
        assert "num_retries" not in seen


class TestEmbeddingOnlyProviderIsRefused:
    """A provider that serves embeddings only is refused before anything else runs."""

    @pytest.fixture
    def embeddings_only(self, monkeypatch):
        from osprey.models import completion as completion_module
        from osprey.models.provider_registry import get_provider_registry

        registry = get_provider_registry()
        real_is_chat = registry.is_chat
        monkeypatch.setattr(
            registry, "is_chat", lambda name: False if name == "openai" else real_is_chat(name)
        )

        def no_config(provider):
            raise AssertionError(f"config read for {provider}")

        monkeypatch.setattr(completion_module, "get_provider_config", no_config)

        constructed: list = []
        cls = registry.get_provider("openai")
        real_init = cls.__init__

        def tracking_init(self, *args, **kwargs):
            constructed.append(self)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", tracking_init)
        return constructed

    def test_direct_provider_is_refused(self, embeddings_only):
        """Naming the provider directly raises before config or adapter."""
        from osprey.models import completion as completion_module

        with pytest.raises(ValueError, match="openai serves embeddings only"):
            completion_module.get_chat_completion(message="ping", provider="openai")
        assert embeddings_only == []

    def test_model_config_provider_is_refused(self, embeddings_only):
        """Naming the provider through model_config raises the same way."""
        from osprey.models import completion as completion_module

        with pytest.raises(ValueError, match="openai serves embeddings only"):
            completion_module.get_chat_completion(
                message="ping", model_config={"provider": "openai", "model_id": "m"}
            )
        assert embeddings_only == []

    def test_llama_cpp_is_refused_from_its_table_row(self):
        """llama-cpp serves embeddings only, and its table row says so with no patching."""
        from osprey.models import completion as completion_module

        with pytest.raises(ValueError, match="llama-cpp serves embeddings only"):
            completion_module.get_chat_completion(message="ping", provider="llama-cpp")

    def test_unknown_provider_is_named_unknown(self, monkeypatch):
        """An unknown name reports itself as unknown, never as embeddings-only."""
        from osprey.models import completion as completion_module

        monkeypatch.setattr(completion_module, "get_provider_config", lambda provider: {})

        with pytest.raises(ValueError, match="Unknown provider: nope"):
            completion_module.get_chat_completion(message="ping", provider="nope", model_id="m")
