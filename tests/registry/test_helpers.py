"""Unit tests for registry helper functions.

Covers ``extend_framework_registry`` (the declarative extend-mode builder).
It is a pure function — no registry initialization or I/O — so the tests run
without fixtures.
"""

from __future__ import annotations

from osprey.registry.base import (
    ExtendedRegistryConfig,
    ProviderRegistration,
    RegistryConfig,
    ServiceRegistration,
)
from osprey.registry.helpers import extend_framework_registry


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
        # No exclusions requested → an empty table, the field's declared default.
        assert cfg.framework_exclusions == {}

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
