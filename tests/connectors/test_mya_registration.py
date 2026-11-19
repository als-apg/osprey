"""Tests for the ``mya_archiver`` connector type surface.

MYA is Jefferson Lab's archiver, read over the ``myquery`` HTTP service. These
tests pin the name across every layer that has to agree on it: the type
constants, the factory's built-in registration, the framework registry, and the
CLI choice list.

``jlab_archiver_client`` is never imported here — the archive defers that
import to its constructor, which is what makes registering it unconditionally
safe on a machine with no JLab environment.
"""

from unittest.mock import patch

from osprey_connectors import connector_types
from osprey_connectors.factory import (
    ConnectorFactory,
)
from osprey_connectors_tests._harness import isolated_connector_registries


class TestTypeConstants:
    """``connector_types.py`` is the single source of truth for the name string."""

    def test_archiver_constant(self):
        assert connector_types.MYA_ARCHIVER == "mya_archiver"


class TestBuiltinRegistration:
    """``ConnectorFactory._ensure_builtins()`` mints the name."""

    def test_mya_archiver_registers_as_builtin_archiver(self):
        """`archiver.type: mya_archiver` must resolve without a dotted path."""
        from osprey_connectors.archiver._mya_archiver_connector import MYAArchiverConnector

        with isolated_connector_registries(clear=True):
            ConnectorFactory._ensure_builtins()

            assert connector_types.MYA_ARCHIVER in ConnectorFactory.list_services("archive")
            registered = ConnectorFactory._service_backends["archive"][connector_types.MYA_ARCHIVER]
            assert registered is MYAArchiverConnector

    def test_registration_needs_no_client_library(self, monkeypatch):
        """Registration must not import ``jlab_archiver_client``.

        The archive imports it when it is built. If that ever moved to
        module scope, registering the built-ins would raise ImportError on
        every machine without the library and take the whole framework down.

        The connector module is dropped from
        ``sys.modules`` first: in a suite run it is already cached, and a
        cached module never re-runs its module-scope imports.
        """
        import sys

        import osprey_connectors.archiver as archiver_pkg

        canonical = "osprey_connectors.archiver._mya_archiver_connector"
        cached = sys.modules.get(canonical)
        # The re-import rebinds the package attribute; put it back afterwards.
        monkeypatch.setattr(archiver_pkg, "_mya_archiver_connector", cached, raising=False)

        with (
            patch.dict(sys.modules, {"jlab_archiver_client": None}),
            isolated_connector_registries(clear=True),
        ):
            sys.modules.pop(canonical, None)

            ConnectorFactory._ensure_builtins()

            assert sys.modules[canonical] is not cached, "the connector was not re-imported"
            assert connector_types.MYA_ARCHIVER in ConnectorFactory.list_services("archive")


class TestFrameworkRegistryEntries:
    """The registry provider carries a matching entry for discovery/export."""

    def test_registry_lists_the_mya_archiver(self):
        from osprey.registry.builtins import FrameworkRegistryProvider

        connectors = FrameworkRegistryProvider().get_registry_config().connectors
        by_name = {c.name: c for c in connectors}

        assert by_name[connector_types.MYA_ARCHIVER].connector_type == "archiver"
        assert by_name[connector_types.MYA_ARCHIVER].class_name == "MYAArchiverConnector"
        assert (
            by_name[connector_types.MYA_ARCHIVER].module_path
            == "osprey_connectors.archiver._mya_archiver_connector"
        )
