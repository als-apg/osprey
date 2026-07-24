"""Unit tests for registry helper functions.

Covers ``extend_framework_registry`` (the declarative extend-mode builder),
``get_framework_defaults`` (framework baseline), and
``generate_explicit_registry_code`` (source-code generator). All are pure
functions — no registry initialization or I/O — so they run without fixtures.
"""

from __future__ import annotations

from osprey.registry.base import (
    ExtendedRegistryConfig,
    ProviderRegistration,
    RegistryConfig,
    ServiceRegistration,
)
from osprey.registry.helpers import (
    extend_framework_registry,
    generate_explicit_registry_code,
    get_framework_defaults,
)


class TestExtendFrameworkRegistry:
    """The declarative extend-mode config builder."""

    def test_empty_call_returns_extended_config_with_no_exclusions(self):
        cfg = extend_framework_registry()

        assert isinstance(cfg, ExtendedRegistryConfig)
        # Marker subclass is still a RegistryConfig.
        assert isinstance(cfg, RegistryConfig)
        assert cfg.services == []
        assert cfg.providers == []
        assert cfg.connectors == []
        # No exclusions requested → the field is None, not an empty dict.
        assert cfg.framework_exclusions is None

    def test_additive_components_are_carried_through(self):
        svc = ServiceRegistration(
            name="svc", module_path="pkg.mod", class_name="Svc", description="d"
        )
        prov = ProviderRegistration(module_path="pkg.prov", class_name="Prov")

        cfg = extend_framework_registry(services=[svc], providers=[prov])

        assert cfg.services == [svc]
        assert cfg.providers == [prov]

    def test_exclusions_land_under_typed_keys(self):
        cfg = extend_framework_registry(
            exclude_providers=["openai"],
            exclude_connectors=["doocs"],
            exclude_ariel_search_modules=["sm"],
            exclude_ariel_enhancement_modules=["em"],
            exclude_ariel_ingestion_adapters=["ia"],
        )

        assert cfg.framework_exclusions == {
            "providers": ["openai"],
            "connectors": ["doocs"],
            "ariel_search_modules": ["sm"],
            "ariel_enhancement_modules": ["em"],
            "ariel_ingestion_adapters": ["ia"],
        }

    def test_partial_exclusions_only_include_requested_keys(self):
        cfg = extend_framework_registry(exclude_providers=["openai"])

        assert cfg.framework_exclusions == {"providers": ["openai"]}
        assert "connectors" not in cfg.framework_exclusions

    def test_overrides_are_appended_after_additive_entries(self):
        add = ProviderRegistration(module_path="pkg.a", class_name="A")
        override = ProviderRegistration(module_path="pkg.b", class_name="B")

        cfg = extend_framework_registry(providers=[add], override_providers=[override])

        # Additive first, then overrides — order matters for last-wins merge.
        assert cfg.providers == [add, override]

    def test_override_connectors_appended_after_additive_connectors(self):
        from osprey.registry.base import ConnectorRegistration

        add = ConnectorRegistration(
            name="mock",
            connector_type="control_system",
            module_path="pkg.mock",
            class_name="Mock",
            description="mock cs",
        )
        override = ConnectorRegistration(
            name="epics",
            connector_type="control_system",
            module_path="pkg.epics",
            class_name="Epics",
            description="epics cs",
        )

        cfg = extend_framework_registry(connectors=[add], override_connectors=[override])

        assert cfg.connectors == [add, override]

    def test_input_lists_are_copied_not_aliased(self):
        services = [ServiceRegistration(name="s", module_path="m", class_name="C", description="d")]
        cfg = extend_framework_registry(services=services)

        # Mutating the caller's list must not leak into the config.
        services.append(
            ServiceRegistration(name="s2", module_path="m2", class_name="C2", description="d2")
        )
        assert len(cfg.services) == 1


class TestGetFrameworkDefaults:
    """The framework baseline registry."""

    def test_returns_plain_registry_config_not_extended(self):
        cfg = get_framework_defaults()

        assert isinstance(cfg, RegistryConfig)
        # Framework baseline is standalone, not an extend-mode marker.
        assert not isinstance(cfg, ExtendedRegistryConfig)

    def test_baseline_ships_services(self):
        cfg = get_framework_defaults()

        # The framework always ships its core internal services.
        assert len(cfg.services) > 0
        assert all(isinstance(s, ServiceRegistration) for s in cfg.services)


class TestGenerateExplicitRegistryCode:
    """Source-code generation for explicit-style registries."""

    def test_generated_code_is_valid_python(self):
        code = generate_explicit_registry_code(
            app_class_name="MyRegistry",
            app_display_name="My App",
            package_name="my_app",
        )

        # Must compile — the whole point is a runnable registry module.
        compile(code, "<generated>", "exec")

    def test_generated_code_names_the_class_and_app(self):
        code = generate_explicit_registry_code(
            app_class_name="MyRegistry",
            app_display_name="My App",
            package_name="my_app",
        )

        assert "class MyRegistry(RegistryConfigProvider):" in code
        assert "My App" in code
        assert "def get_registry_config(self):" in code

    def test_framework_providers_are_emitted(self):
        code = generate_explicit_registry_code(
            app_class_name="R", app_display_name="A", package_name="a"
        )

        # Every framework provider should appear as a ProviderRegistration line.
        framework = get_framework_defaults()
        assert code.count("ProviderRegistration(") == len(framework.providers)

    def test_application_services_section_included_when_services_given(self):
        svc = ServiceRegistration(
            name="my_service",
            module_path="my_app.services",
            class_name="MyService",
            description="Does a thing",
            provides=["result"],
            requires=["input"],
        )

        code = generate_explicit_registry_code(
            app_class_name="R",
            app_display_name="A",
            package_name="a",
            services=[svc],
        )

        compile(code, "<generated>", "exec")
        assert "Application Services" in code
        assert 'name="my_service"' in code
        assert 'class_name="MyService"' in code

    def test_no_application_services_section_when_omitted(self):
        code = generate_explicit_registry_code(
            app_class_name="R", app_display_name="A", package_name="a"
        )

        assert "Application Services" not in code
