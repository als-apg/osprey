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
    ("als-apg", "https://llm.gianlucamartino.com"),
    ("argo", "https://apps.inside.anl.gov/argoapi/v1"),
    ("ds4", "http://127.0.0.1:8000/v1"),
    ("ollama", "http://localhost:11434"),
    ("stanford", "https://aiapi-prod.stanford.edu/v1"),
    ("vllm", "http://localhost:8000/v1"),
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

    That disagreement was invisible for as long as an env override happened to
    supply the value — it surfaced the moment ``ALS_APG_BASE_URL`` was removed
    once the default gateway came back. These tests pin the agreement rather than
    either half, since either half alone passes while the pair is broken.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_override(self, monkeypatch):
        _clear_base_url_overrides(monkeypatch)

    @pytest.mark.parametrize(("provider", "expected"), PROVIDERS_DECLARING_A_DEFAULT_ENDPOINT)
    def test_a_declared_default_satisfies_the_requirement(self, provider, expected):
        from osprey.models.provider_registry import get_provider_registry

        provider_class = get_provider_registry().get_provider(provider)
        assert provider_class.requires_base_url, f"{provider} must still require a base_url"
        # None in, the declared default out — the value the gate must accept.
        assert provider_class.effective_base_url(None) == expected

    def test_an_env_override_still_beats_the_default(self, monkeypatch):
        from osprey.models.provider_registry import get_provider_registry

        monkeypatch.setenv("ALS_APG_BASE_URL", "https://fallback.example")
        provider_class = get_provider_registry().get_provider("als-apg")
        assert provider_class.effective_base_url(None) == "https://fallback.example"
        # ...and beats an explicitly configured value, which is the whole point of
        # a break-glass redirect for an already-deployed system.
        assert provider_class.effective_base_url("https://baked-in") == "https://fallback.example"

    def test_a_configured_value_still_wins_over_the_default(self):
        from osprey.models.provider_registry import get_provider_registry

        provider_class = get_provider_registry().get_provider("als-apg")
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

        def fake_execute(self, **kwargs):
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
        from osprey.models.provider_registry import get_provider_registry

        provider_class = get_provider_registry().get_provider("als-apg")
        monkeypatch.setattr(provider_class, "apply_default_base_url_fallback", False)
        monkeypatch.setattr(
            completion_module,
            "get_provider_config",
            lambda provider: {"api_key": "k", "default_model_id": "claude-haiku-4-5-20251001"},
        )

        with pytest.raises(ValueError, match="Base URL required for als-apg"):
            completion_module.get_chat_completion(message="ping", provider="als-apg", max_tokens=4)
