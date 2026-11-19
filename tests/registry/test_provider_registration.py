"""Tests for custom AI model provider registration.

This test module validates the provider registration feature that allows
applications to register custom AI model providers through the registry system.

Architecture note: Built-in providers (anthropic, openai, etc.) live in
``osprey.models.provider_registry`` as the single source of truth.
``initialize_providers()`` applies the application's registrations and
exclusions to it and ``RegistryManager`` answers provider lookups from it, so
built-in entries are checked against ``get_provider_registry()``, not against
``manager.config.providers``.

Tests cover:
- Registering custom providers in application registries
- Provider merging during registry initialization
- Provider exclusions (excluding framework providers)
- Helper function support for providers
- Provider override scenarios
- The manager's provider answers are the ProviderRegistry's
- Registry init imports no provider adapter
"""

import json
import subprocess
import sys
import textwrap

import pytest

from osprey.models.provider_registry import (
    _BUILTIN_PROVIDERS,
    get_provider_registry,
    reset_provider_registry,
)
from osprey.registry.base import (
    ProviderRegistration,
    RegistryConfig,
)
from osprey.registry.helpers import extend_framework_registry
from osprey.registry.manager import RegistryManager


@pytest.fixture(autouse=True)
def _clean_provider_registry():
    """Reset the provider registry singleton around each test."""
    reset_provider_registry()
    yield
    reset_provider_registry()


class TestProviderMerging:
    """Test that application providers are properly merged into registry."""

    def test_application_providers_merged_into_config(self, tmp_path):
        """Test that providers from application registry are added to merged config."""
        # Create application registry with custom providers
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry, ProviderRegistration

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            providers=[
                ProviderRegistration(
                    module_path="test_app.providers.custom",
                    class_name="CustomAIProvider"
                ),
                ProviderRegistration(
                    module_path="test_app.providers.institutional",
                    class_name="InstitutionalAIProvider"
                )
            ]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Custom providers should be in config.providers
        provider_module_paths = [p.module_path for p in manager.config.providers]
        assert "test_app.providers.custom" in provider_module_paths
        assert "test_app.providers.institutional" in provider_module_paths

        # Built-in providers are resolved via ProviderRegistry (not config.providers)
        pr = get_provider_registry()
        assert pr.get_provider("anthropic") is not None
        assert pr.get_provider("openai") is not None

    def test_multiple_providers_in_single_application(self, tmp_path):
        """Test that single application can add multiple providers."""
        # Create application registry with multiple providers
        app_file = tmp_path / "app" / "registry.py"
        app_file.parent.mkdir(parents=True)
        app_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry, ProviderRegistration

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            providers=[
                ProviderRegistration(
                    module_path="app.providers.custom1",
                    class_name="Custom1Provider"
                ),
                ProviderRegistration(
                    module_path="app.providers.custom2",
                    class_name="Custom2Provider"
                )
            ]
        )
"""
        )

        # Load registry
        manager = RegistryManager(registry_path=str(app_file))

        # Both application providers should be in merged config
        provider_module_paths = [p.module_path for p in manager.config.providers]
        assert "app.providers.custom1" in provider_module_paths
        assert "app.providers.custom2" in provider_module_paths

    def test_empty_providers_list_handled_correctly(self, tmp_path):
        """Test that empty providers list doesn't cause errors."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            providers=[]  # Empty list
        )
"""
        )

        # Should not raise
        RegistryManager(registry_path=str(registry_file))

        # Built-in providers should still be resolvable via ProviderRegistry
        pr = get_provider_registry()
        assert len(pr.list_providers()) >= 9


class TestProviderExclusions:
    """Test excluding framework providers."""

    def test_exclude_framework_provider(self, tmp_path):
        """Test excluding specific framework providers."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            exclude_providers=["anthropic", "google"]
        )
"""
        )

        # Load registry
        manager = RegistryManager(registry_path=str(registry_file))

        # Verify exclusions were recorded
        assert "anthropic" in manager._excluded_provider_names
        assert "google" in manager._excluded_provider_names

    def test_exclude_all_framework_providers(self, tmp_path):
        """Test excluding all framework providers (custom-only setup)."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry, ProviderRegistration

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            providers=[
                ProviderRegistration(
                    module_path="app.providers.only",
                    class_name="OnlyProvider"
                )
            ],
            exclude_providers=["anthropic", "openai", "google", "ollama", "cborg", "amsc-i2"]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # All listed framework providers should be in exclusion list
        expected_exclusions = ["anthropic", "openai", "google", "ollama", "cborg", "amsc-i2"]
        assert len(manager._excluded_provider_names) == len(expected_exclusions)
        for provider in expected_exclusions:
            assert provider in manager._excluded_provider_names

        # Custom provider should still be in config
        provider_modules = [p.module_path for p in manager.config.providers]
        assert "app.providers.only" in provider_modules

    def _excluding(self, tmp_path, names: list[str]):
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text(
            f"""
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            exclude_providers={names!r}
        )
"""
        )
        return registry_file

    def _initialize(self, registry_file) -> RegistryManager:
        manager = RegistryManager(registry_path=str(registry_file))
        manager.initialize(silent=True)
        return manager

    def test_an_excluded_builtin_is_gone_from_the_provider_registry(self, tmp_path):
        manager = self._initialize(self._excluding(tmp_path, ["openai"]))

        pr = get_provider_registry()
        assert pr.get_provider("openai") is None
        assert "openai" not in pr.list_providers()
        assert "openai" not in manager.list_providers()
        assert "anthropic" in manager.list_providers()
        assert manager.get_provider("openai") is None

    def test_an_excluded_provider_fails_closed_at_completion(self, tmp_path):
        from osprey.models.completion import get_chat_completion

        self._initialize(self._excluding(tmp_path, ["openai"]))

        with pytest.raises(ValueError, match="Unknown provider: openai"):
            get_chat_completion(
                message="hi",
                provider="openai",
                model_id="gpt-4o",
                provider_config={"api_key": "k"},
            )

    def test_an_exclusion_that_names_no_builtin_warns(self, tmp_path, caplog):
        import logging

        from osprey.models.provider_registry import _BUILTIN_PROVIDERS

        registry_file = self._excluding(tmp_path, ["no-such-provider"])
        with caplog.at_level(logging.WARNING):
            self._initialize(registry_file)

        assert "no-such-provider" in caplog.text
        assert "not a built-in provider" in caplog.text
        pr = get_provider_registry()
        for name in _BUILTIN_PROVIDERS:
            assert name in pr.list_providers()

    def test_reset_registry_drops_the_exclusion(self, tmp_path):
        from osprey.registry import reset_registry

        self._initialize(self._excluding(tmp_path, ["openai"]))

        reset_registry()

        assert get_provider_registry().get_provider("openai") is not None


class TestHelperFunctionProviderSupport:
    """Test extend_framework_registry helper with provider parameters."""

    def test_helper_with_providers_parameter(self):
        """Test that helper function accepts providers parameter."""
        config = extend_framework_registry(
            providers=[
                ProviderRegistration(
                    module_path="my_app.providers.custom", class_name="CustomProvider"
                )
            ],
        )

        # Returns application config with providers
        assert len(config.providers) == 1
        assert config.providers[0].module_path == "my_app.providers.custom"
        assert config.providers[0].class_name == "CustomProvider"

    def test_helper_with_exclude_providers(self):
        """Test that helper function accepts exclude_providers parameter."""
        config = extend_framework_registry(exclude_providers=["anthropic", "google"])

        # Exclusions stored in framework_exclusions
        assert config.framework_exclusions is not None
        assert "providers" in config.framework_exclusions
        assert "anthropic" in config.framework_exclusions["providers"]
        assert "google" in config.framework_exclusions["providers"]

    def test_helper_with_override_providers(self):
        """Test that helper function accepts override_providers parameter."""
        override_provider = ProviderRegistration(
            module_path="my_app.providers.custom_anthropic", class_name="CustomAnthropicProvider"
        )

        config = extend_framework_registry(override_providers=[override_provider])

        # Override provider included in config
        assert len(config.providers) == 1
        assert config.providers[0].module_path == "my_app.providers.custom_anthropic"

    def test_helper_with_providers_and_overrides(self):
        """Test combining regular providers and overrides."""
        regular_provider = ProviderRegistration(
            module_path="my_app.providers.new", class_name="NewProvider"
        )

        override_provider = ProviderRegistration(
            module_path="my_app.providers.custom_openai", class_name="CustomOpenAIProvider"
        )

        config = extend_framework_registry(
            providers=[regular_provider], override_providers=[override_provider]
        )

        # Both should be included
        assert len(config.providers) == 2
        provider_modules = [p.module_path for p in config.providers]
        assert "my_app.providers.new" in provider_modules
        assert "my_app.providers.custom_openai" in provider_modules

    def test_helper_used_in_registry_provider(self, tmp_path):
        """Test that helper with providers works in actual registry provider file.

        This validates the most common use case: an application using the
        extend_framework_registry helper to add custom providers.
        """
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    extend_framework_registry,
    ProviderRegistration
)

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            providers=[
                ProviderRegistration(
                    module_path="app.providers.custom",
                    class_name="CustomAIProvider"
                )
            ]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))
        provider_modules = [p.module_path for p in manager.config.providers]

        # Custom provider should be in config.providers
        assert "app.providers.custom" in provider_modules

        # Built-in providers are resolved via ProviderRegistry
        pr = get_provider_registry()
        assert pr.get_provider("anthropic") is not None
        assert pr.get_provider("openai") is not None

    def test_helper_with_exclusions_in_registry_provider(self, tmp_path):
        """Test helper with provider exclusions in actual registry."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    RegistryConfig,
    extend_framework_registry,
    ProviderRegistration
)

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            providers=[
                ProviderRegistration(
                    module_path="app.providers.custom",
                    class_name="CustomProvider"
                )
            ],
            exclude_providers=["anthropic", "google"]
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Exclusions should be recorded
        assert "anthropic" in manager._excluded_provider_names
        assert "google" in manager._excluded_provider_names

        # Custom provider should be in config
        provider_modules = [p.module_path for p in manager.config.providers]
        assert "app.providers.custom" in provider_modules


class TestProviderRegistrationDataclass:
    """Test ProviderRegistration dataclass behavior."""

    def test_name_defaults_to_none(self):
        """ProviderRegistration.name defaults to None for backward compat."""
        reg = ProviderRegistration(module_path="some.module", class_name="SomeProvider")
        assert reg.name is None

    def test_name_stores_explicit_value(self):
        """ProviderRegistration with explicit name stores it correctly."""
        reg = ProviderRegistration(
            module_path="some.module", class_name="SomeProvider", name="custom"
        )
        assert reg.name == "custom"

    def test_provider_registration_in_registry_config(self):
        """Test ProviderRegistration can be used in RegistryConfig."""
        provider_reg = ProviderRegistration(module_path="test.provider", class_name="TestProvider")

        config = RegistryConfig(providers=[provider_reg])

        assert len(config.providers) == 1
        assert config.providers[0] is provider_reg


class TestProviderIntegrationScenarios:
    """Test realistic provider registration scenarios.

    These tests demonstrate complete, production-ready patterns for
    customizing provider configuration in real applications.
    """

    def test_replace_framework_provider_scenario(self, tmp_path):
        """Test scenario where app replaces a framework provider with custom version."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    ProviderRegistration
)

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            override_providers=[
                ProviderRegistration(
                    module_path="my_app.providers.custom_openai",
                    class_name="CustomOpenAIProvider"
                )
            ],
            exclude_providers=["openai"]  # Exclude framework OpenAI
        )
"""
        )

        manager = RegistryManager(registry_path=str(registry_file))

        # Framework OpenAI should be excluded
        assert "openai" in manager._excluded_provider_names

        # Custom provider should be present in config.providers
        provider_modules = [p.module_path for p in manager.config.providers]
        assert "my_app.providers.custom_openai" in provider_modules

        # Other built-in providers still resolvable via ProviderRegistry
        pr = get_provider_registry()
        assert pr.get_provider("anthropic") is not None


class TestApplicationOverridesTakeEffect:
    """A same-name registration REPLACES the built-in it shadows.

    ``register_provider`` documents overwrite-by-name and ``override_providers``
    documents "replace framework versions (by name)", but the registry is
    pre-seeded with every built-in, so a name check against it dropped exactly
    the registrations that were meant to replace one — silently.
    """

    def _registry_file(self, tmp_path, body: str):
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text(body)
        return registry_file

    def test_a_same_name_registration_replaces_the_builtin(self, tmp_path, monkeypatch):
        module = tmp_path / "house_openai.py"
        module.write_text(
            """
from osprey.models.providers.base import BaseProvider


class HouseOpenAIProvider(BaseProvider):
    name = "openai"
    requires_api_key = False
    requires_base_url = False
    supports_proxy = False
"""
        )
        monkeypatch.syspath_prepend(tmp_path)
        registry_file = self._registry_file(
            tmp_path,
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    ProviderRegistration,
)


class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            override_providers=[
                ProviderRegistration(
                    module_path="house_openai",
                    class_name="HouseOpenAIProvider",
                    name="openai",
                )
            ],
            exclude_providers=["openai"],
        )
""",
        )
        RegistryManager(registry_path=str(registry_file)).initialize(silent=True)

        pr = get_provider_registry()
        provider_class = pr.get_provider("openai")
        assert provider_class is not None
        # The app's class, not the framework's.
        assert provider_class.__name__ == "HouseOpenAIProvider"
        assert "openai" in pr.list_providers()

    def test_a_nameless_registration_warns_that_it_is_inert(self, tmp_path, caplog):
        registry_file = self._registry_file(
            tmp_path,
            """
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    ProviderRegistration,
)


class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            providers=[
                ProviderRegistration(
                    module_path="my_app.providers.custom",
                    class_name="CustomProvider",
                )
            ]
        )
""",
        )
        import logging

        with caplog.at_level(logging.WARNING):
            RegistryManager(registry_path=str(registry_file)).initialize(silent=True)

        assert "CustomProvider" in caplog.text
        assert "name=" in caplog.text


_SITE_MODULE = """
from osprey.models.providers.base import BaseProvider


class SiteGatewayProvider(BaseProvider):
    name = "site-gateway"
    requires_api_key = False
    requires_base_url = False
    supports_proxy = False


class HouseOpenAIProvider(BaseProvider):
    name = "openai"
    requires_api_key = False
    requires_base_url = False
    supports_proxy = False
"""


def _app_registry(tmp_path, *, providers=(), exclude=()) -> str:
    """Write a site module and an application registry; return the registry's path."""
    (tmp_path / "site_providers.py").write_text(_SITE_MODULE)
    registrations = ", ".join(
        f"ProviderRegistration(module_path='site_providers', class_name={cls!r}, name={name!r})"
        for name, cls in providers
    )
    registry_file = tmp_path / "app" / "registry.py"
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(
        textwrap.dedent(
            f"""
            from osprey.registry import (
                ProviderRegistration,
                RegistryConfigProvider,
                extend_framework_registry,
            )


            class AppProvider(RegistryConfigProvider):
                def get_registry_config(self):
                    return extend_framework_registry(
                        providers=[{registrations}], exclude_providers={list(exclude)!r}
                    )
            """
        )
    )
    return str(registry_file)


_SCENARIOS = {
    "built-ins": {},
    "an app registration": {"providers": [("site-gateway", "SiteGatewayProvider")]},
    "an override of a built-in": {"providers": [("openai", "HouseOpenAIProvider")]},
    "an exclusion": {"exclude": ["google"]},
    "an override of an excluded built-in": {
        "providers": [("openai", "HouseOpenAIProvider")],
        "exclude": ["openai"],
    },
}


class TestTheManagerAnswersFromTheProviderRegistry:
    """``RegistryManager`` holds no provider table of its own.

    ``get_provider`` and ``list_providers`` read the ``ProviderRegistry`` the
    initializer applied the application's registrations and exclusions to, so
    the two can never disagree about which providers exist.
    """

    @pytest.mark.parametrize("scenario", list(_SCENARIOS))
    def test_lookup_and_listing_agree(self, tmp_path, monkeypatch, scenario):
        monkeypatch.syspath_prepend(str(tmp_path))
        manager = RegistryManager(registry_path=_app_registry(tmp_path, **_SCENARIOS[scenario]))
        manager.initialize(silent=True)

        pr = get_provider_registry()
        names = pr.list_providers()
        assert manager.list_providers() == names
        for name in [*_BUILTIN_PROVIDERS, "site-gateway"]:
            assert manager.get_provider(name) is pr.get_provider(name), name

    def test_the_scenarios_resolve_as_documented(self, tmp_path, monkeypatch):
        """Not vacuous: each scenario changes what the registry resolves."""
        monkeypatch.syspath_prepend(str(tmp_path))
        manager = RegistryManager(
            registry_path=_app_registry(
                tmp_path,
                providers=[
                    ("site-gateway", "SiteGatewayProvider"),
                    ("openai", "HouseOpenAIProvider"),
                ],
                exclude=["openai", "google"],
            )
        )
        manager.initialize(silent=True)

        assert manager.get_provider("site-gateway").__name__ == "SiteGatewayProvider"
        assert manager.get_provider("openai").__name__ == "HouseOpenAIProvider"
        assert manager.get_provider("google") is None
        assert "google" not in manager.list_providers()
        assert {"site-gateway", "openai", "anthropic"} <= set(manager.list_providers())


class TestRegistryInitImportsNoProvider:
    """Registry init applies registrations without importing a provider adapter.

    The executor child and every Jupyter kernel run ``initialize_registry``
    only to register connectors; importing the adapters (and litellm through
    them) there cost seconds per run. A provider is imported when something
    asks for its class.
    """

    def test_init_leaves_every_adapter_and_litellm_unimported(self, tmp_path):
        registry_path = _app_registry(
            tmp_path,
            providers=[("site-gateway", "SiteGatewayProvider"), ("openai", "HouseOpenAIProvider")],
            exclude=["google"],
        )
        config = tmp_path / "config.yml"
        config.write_text(f"registry_path: {registry_path}\n")
        adapters = sorted({entry.module_path for entry in _BUILTIN_PROVIDERS.values()})
        code = textwrap.dedent(
            f"""
            import json, sys
            sys.path.insert(0, {str(tmp_path)!r})
            from osprey.registry import get_registry, initialize_registry

            initialize_registry(auto_export=False, config_path={str(config)!r}, silent=True)
            loaded = sorted(
                m for m in {adapters!r} + ["site_providers", "litellm"] if m in sys.modules
            )
            print(json.dumps({{"loaded": loaded, "listed": get_registry().list_providers()}}))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            cwd=str(tmp_path),
        )
        report = json.loads(result.stdout.strip().splitlines()[-1])

        assert report["loaded"] == []
        assert "site-gateway" in report["listed"]
        assert "google" not in report["listed"]

    def test_a_registered_module_that_raises_fails_its_first_lookup_not_init(
        self, tmp_path, monkeypatch
    ):
        """Init imports no provider, so a broken one cannot fail it: the error
        surfaces when that provider is first asked for, and names it.
        """
        (tmp_path / "site_boom_module.py").write_text("raise ValueError('site setting missing')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        registry_file.write_text(
            textwrap.dedent(
                """
                from osprey.registry import (
                    ProviderRegistration,
                    RegistryConfigProvider,
                    extend_framework_registry,
                )


                class AppProvider(RegistryConfigProvider):
                    def get_registry_config(self):
                        return extend_framework_registry(
                            providers=[
                                ProviderRegistration(
                                    module_path="site_boom_module",
                                    class_name="BoomAdapter",
                                    name="site-boom",
                                )
                            ]
                        )
                """
            )
        )
        manager = RegistryManager(registry_path=str(registry_file))
        manager.initialize(silent=True)

        assert "site-boom" in manager.list_providers()
        assert manager.get_provider("anthropic") is not None
        with pytest.raises(RuntimeError, match=r"Provider 'site-boom'.*'site_boom_module'"):
            manager.get_provider("site-boom")


class TestProviderMergingEdgeCases:
    """Test edge cases in provider merging."""

    def test_no_providers_in_application_registry(self, tmp_path):
        """Test that applications without providers work fine."""
        registry_file = tmp_path / "app" / "registry.py"
        registry_file.parent.mkdir(parents=True)
        registry_file.write_text(
            """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class AppProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry()  # No providers field at all
"""
        )

        # Should not raise
        RegistryManager(registry_path=str(registry_file))

        # Built-in providers resolved via ProviderRegistry (not config.providers)
        pr = get_provider_registry()
        assert pr.get_provider("anthropic") is not None
        assert pr.get_provider("openai") is not None

    def test_framework_only_registry_has_providers(self):
        """Test that framework-only setup still resolves built-in providers."""
        RegistryManager(registry_path=None)

        # config.providers is now empty (built-ins live in ProviderRegistry)
        # but providers are still resolvable
        pr = get_provider_registry()
        assert len(pr.list_providers()) >= 9
        assert pr.get_provider("anthropic") is not None
        assert pr.get_provider("openai") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
