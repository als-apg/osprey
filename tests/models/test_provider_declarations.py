"""An adapter states its provider's facts; every table that restates one agrees.

The Claude Code launch table and the packaged catalog each restate some of those
facts by provider name, and the provider key table is derived from the registry's
entry per provider. These tests hold every restatement, and the derived table, to
the declaration on the adapter class.
"""

from __future__ import annotations

import re
from typing import Any, get_args, get_type_hints

import pytest
import yaml

from osprey.build.claude_code_resolver import provider_auth_secret_env
from osprey.models.provider_registry import (
    _BUILTIN_PROVIDERS,
    ProviderRegistry,
    get_provider_registry,
)
from osprey.models.providers.base import BaseProvider
from osprey.profiles.providers import VALID_API_PROTOCOLS, packaged_catalog_path

#: The facts every built-in adapter states in its own class body.
PROVIDER_FACTS = (
    "api_key_env_var",
    "api_protocol",
    "supports_interactive_login",
    "supports_images",
    "supports_thinking",
)

_ENV_REFERENCE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _adapters() -> dict[str, type[BaseProvider]]:
    """Every registered built-in provider, loaded to its adapter class."""
    registry = get_provider_registry()
    loaded = {name: registry.get_provider(name) for name in registry.list_providers()}
    missing = sorted(name for name, cls in loaded.items() if cls is None)
    assert not missing, f"no provider adapter resolves for: {missing}"
    return {name: cls for name, cls in loaded.items() if cls is not None}


def _catalog() -> dict[str, dict[str, Any]]:
    """The packaged provider catalog's entries, keyed by provider name."""
    document: dict[str, Any] = yaml.safe_load(packaged_catalog_path().read_text())
    providers: dict[str, dict[str, Any]] = document["providers"]
    return providers


class TestBaseDeclaration:
    def test_an_undeclared_provider_gets_the_defaults(self):
        assert BaseProvider.api_key_env_var is None
        assert BaseProvider.api_protocol == "openai"
        assert BaseProvider.supports_interactive_login is False
        assert BaseProvider.supports_images is False
        assert BaseProvider.supports_thinking is False

    def test_the_protocol_type_names_the_catalog_protocols(self):
        protocols = get_args(get_type_hints(BaseProvider)["api_protocol"])
        assert set(protocols) == set(VALID_API_PROTOCOLS)


class TestEveryAdapterDeclares:
    @pytest.mark.parametrize("fact", PROVIDER_FACTS)
    def test_every_built_in_states_the_fact_in_its_own_body(self, fact):
        silent = sorted(name for name, cls in _adapters().items() if fact not in vars(cls))
        assert not silent, f"{fact} is inherited, not stated, by: {silent}"

    def test_a_declared_protocol_is_one_the_catalog_takes(self):
        for name, cls in _adapters().items():
            assert cls.api_protocol in VALID_API_PROTOCOLS, name

    def test_a_key_variable_is_named_exactly_when_a_key_is_required(self):
        for name, cls in _adapters().items():
            assert (cls.api_key_env_var is not None) == cls.requires_api_key, name

    def test_only_a_native_provider_that_takes_a_key_offers_an_interactive_login(self):
        for name, cls in _adapters().items():
            if cls.supports_interactive_login:
                assert cls.api_protocol == "anthropic", name
                assert cls.requires_api_key, name

    @pytest.mark.parametrize("name", ["ollama", "vllm", "ds4"])
    def test_a_local_server_assumes_no_image_input(self, name):
        assert _adapters()[name].supports_images is False

    def test_no_route_claims_thinking_the_translation_does_not_carry(self):
        """The translated route maps no thinking in either direction.

        A True here would be a claim the translation proxy cannot honour; this
        test changes together with a translator that carries thinking.
        """
        claiming = sorted(name for name, cls in _adapters().items() if cls.supports_thinking)
        assert not claiming, f"supports_thinking claimed by: {claiming}"


class TestTablesAgreeWithTheDeclarations:
    def test_every_builtin_entry_states_its_adapters_declarations(self):
        for name, entry in _BUILTIN_PROVIDERS.items():
            cls = ProviderRegistry().get_provider(name)
            assert cls is not None, name
            assert entry.key_env_var == cls.api_key_env_var, name
            assert entry.api_protocol == cls.api_protocol, name

    def test_the_derived_secret_variable_is_the_adapters_for_every_keyed_provider(self):
        for name, cls in _adapters().items():
            if cls.api_key_env_var is None:
                continue
            assert provider_auth_secret_env(name, {name: {}}) == cls.api_key_env_var, name

    def test_the_catalog_references_the_adapters_key_variable(self):
        adapters = _adapters()
        catalog = _catalog()
        shared = sorted(set(catalog) & set(adapters))
        assert shared, "the packaged catalog lists no registered provider"
        for name in shared:
            match = _ENV_REFERENCE.match(str(catalog[name].get("api_key", "")))
            referenced = match.group(1) if match else None
            assert referenced == adapters[name].api_key_env_var, name
