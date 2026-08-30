"""The framework registry advertises exactly the connectors the factory ships.

Two independent tables name the built-in connectors, and both are load-bearing
because two *different* code paths populate the factory from them:

* ``register_builtin_connectors()`` walks the factory's own
  ``_BUILTIN_CONTROL_SYSTEMS`` / ``_BUILTIN_ARCHIVERS`` tuples. The MCP
  control-system server, the health runtime and the queueserver startup all
  call it.
* ``initialize_registry()`` walks
  :class:`~osprey.registry.builtins.FrameworkRegistryProvider`'s
  ``connectors`` list and registers nothing else. That is the setup step every
  python-executor sandbox runs before any agent code, so a connector missing
  from the provider is a connector no sandbox can build — the factory answers
  ``Unknown control system type`` with a list that simply lacks it.

A name in one table and not the other is therefore not a tidiness problem: it
is a connector that works on some entry points and not others. ``live_standin``
was exactly that — registered by the factory's tuple since the stand-in shipped
as a third control target, absent from the provider, and so unreachable from
any sandbox stamped ``OSPREY_CONTROL_TARGET=standin``.

The archiver half of this parity is also pinned from the connector side in
``tests/connectors/test_connector_factory.py``; it is restated here because the
provider is this package's file and the control-system half had no pin at all.
"""

from __future__ import annotations

import pytest

from osprey.connectors import types
from osprey.connectors.factory import (
    _BUILTIN_ARCHIVERS,
    _BUILTIN_CONTROL_SYSTEMS,
    ConnectorFactory,
    isolated_connector_registries,
)
from osprey.registry.builtins import FrameworkRegistryProvider


def _provider_names(connector_type: str) -> set[str]:
    """Names the framework registry provider advertises for one kind."""
    return {
        registration.name
        for registration in FrameworkRegistryProvider().get_registry_config().connectors
        if registration.connector_type == connector_type
    }


def _provider_entry(name: str):
    """The provider's single registration for *name*."""
    matches = [
        registration
        for registration in FrameworkRegistryProvider().get_registry_config().connectors
        if registration.name == name
    ]
    assert len(matches) == 1, f"expected exactly one registry entry for {name!r}"
    return matches[0]


class TestBuiltinConnectorParity:
    """The provider's connector table and the factory's built-in tuples agree."""

    def test_control_system_names_match_the_factory_builtins(self) -> None:
        """Every shipped control system is advertised, and nothing else is.

        The direction that bit: a type the factory registers but the provider
        omits is unreachable from ``initialize_registry()``. The other
        direction is a type the registry advertises and
        ``register_builtin_connectors()`` never heals into a partially
        populated factory.
        """
        assert _provider_names("control_system") == set(_BUILTIN_CONTROL_SYSTEMS)

    def test_archiver_names_match_the_factory_builtins(self) -> None:
        assert _provider_names("archiver") == set(_BUILTIN_ARCHIVERS)

    def test_the_stand_in_is_among_them(self) -> None:
        """Named on its own, so the parity pin cannot go green while empty."""
        assert types.LIVE_STANDIN in _provider_names("control_system")

    def test_the_stand_in_entry_names_the_epics_connector(self) -> None:
        """Served by ``EPICSConnector``, keyed apart from ``epics``.

        The stand-in is a soft IOC, so Channel Access reaches it — but the
        registration *name* is what the factory stamps as ``_connector_type``,
        and that stamp selects the connector block and the write posture read
        out of it. An entry named ``epics`` would hand the stand-in the
        facility's authored block and the facility's arming.
        """
        entry = _provider_entry(types.LIVE_STANDIN)

        assert entry.connector_type == "control_system"
        assert entry.module_path == "osprey.connectors.control_system.epics_connector"
        assert entry.class_name == "EPICSConnector"
        assert entry.name != types.EPICS

    def test_the_provider_entry_loads_the_class_the_factory_registers(self) -> None:
        """The lazy module path/class name pair actually resolves, and to the
        same class ``register_builtin_connectors()`` puts under that key."""
        import importlib

        entry = _provider_entry(types.LIVE_STANDIN)
        loaded = getattr(importlib.import_module(entry.module_path), entry.class_name)

        with isolated_connector_registries(clear=True):
            from osprey.connectors.factory import register_builtin_connectors

            register_builtin_connectors()
            assert ConnectorFactory._control_system_connectors[types.LIVE_STANDIN] is loaded


class TestRegistryInitializationRegistersTheStandIn:
    """The sandbox path: ``initialize_registry()`` and nothing else.

    ``register_builtin_connectors()`` is deliberately not called here. It is
    the other entry point, and calling it would register the stand-in for
    reasons that have nothing to do with the provider — which is the failure
    mode this class exists to catch.
    """

    @pytest.fixture
    def framework_registry(self, tmp_path, monkeypatch):
        """Run a framework-only registry initialization over a clean factory.

        The global singleton is reset either side so no other test inherits
        this registry, and the connector registries are snapshot/restored so
        registrations made elsewhere in the process survive teardown.
        """
        from osprey.registry import manager

        config_file = tmp_path / "config.yml"
        config_file.write_text(f"project_root: {tmp_path}\n", encoding="utf-8")
        monkeypatch.setenv("CONFIG_FILE", str(config_file))

        manager.reset_registry()
        try:
            with isolated_connector_registries(clear=True):
                manager.initialize_registry(
                    auto_export=False, config_path=str(config_file), silent=True
                )
                yield
        finally:
            manager.reset_registry()

    def test_initialize_registry_registers_the_stand_in(self, framework_registry) -> None:
        """The bug: the sandbox's own setup step leaves the type unregistered."""
        from osprey.connectors.control_system.epics_connector import EPICSConnector

        assert types.LIVE_STANDIN in ConnectorFactory.list_control_systems()
        assert ConnectorFactory._control_system_connectors[types.LIVE_STANDIN] is EPICSConnector

    def test_every_builtin_control_system_is_registered(self, framework_registry) -> None:
        """Not just the stand-in: the registry path registers the whole set."""
        registered = set(ConnectorFactory.list_control_systems())
        assert set(_BUILTIN_CONTROL_SYSTEMS) <= registered

    @pytest.mark.asyncio
    async def test_the_stand_in_resolves_after_initialize_registry(
        self, framework_registry
    ) -> None:
        """Resolving ``live_standin`` succeeds, and is stamped with its own type.

        Previously this raised ``Unknown control system type: 'live_standin'``
        before any posture was read, which is what a run stamped
        ``OSPREY_CONTROL_TARGET=standin`` died on.
        """
        from osprey.connectors.control_system.epics_connector import EPICSConnector

        # No gateways: connect() then touches no EPICS_* environment variable
        # and opens no CA context, so nothing here reaches a network.
        connector = await ConnectorFactory.create_control_system_connector(
            {
                "type": types.LIVE_STANDIN,
                "connector": {types.LIVE_STANDIN: {"timeout": 1.0}},
            },
            control_target=types.TARGET_STANDIN,
        )

        assert isinstance(connector, EPICSConnector)
        assert connector._connector_type == types.LIVE_STANDIN
        assert connector._control_target == types.TARGET_STANDIN

        await connector.disconnect()
