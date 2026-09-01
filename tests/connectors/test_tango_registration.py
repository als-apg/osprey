"""Tests for the ``tango`` connector type surface.

The TANGO connector ships in-tree, so its name has to agree across every layer
that speaks it: the type constants, the factory's built-in registration, the
framework registry, and the CLI entry points. These tests pin that agreement,
mirroring ``test_doocs_registration.py``.

``tango`` (PyTango) is never imported here — the connector defers that import
to ``connect()``, which is exactly what makes registering it unconditionally
safe on machines with no TANGO environment.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.connectors import types
from osprey.connectors.factory import (
    ConnectorFactory,
    isolated_connector_registries,
    register_builtin_connectors,
)


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestTypeConstants:
    """``types.py`` is the single source of truth for the name string."""

    def test_control_system_constant(self):
        assert types.TANGO == "tango"

    def test_constant_appears_in_cli_choice_list(self):
        assert types.TANGO in types.CLI_CONTROL_SYSTEM_TYPES

    def test_tango_is_a_live_capable_type(self):
        """A TANGO deployment's own block is its ``live`` machine.

        ``resolve_target`` refuses to derive ``live`` from simulated or
        stand-in types; ``tango`` names real hardware and must resolve as
        written, exactly as ``epics`` and ``doocs`` do.
        """
        assert types.resolve_target({"type": types.TANGO}, types.TARGET_LIVE) == types.TANGO


class TestBuiltinRegistration:
    """``register_builtin_connectors()`` mints the name."""

    def test_tango_registers_as_builtin_control_system(self):
        with isolated_connector_registries(clear=True):
            register_builtin_connectors()

            assert types.TANGO in ConnectorFactory.list_control_systems()
            registered = ConnectorFactory._control_system_connectors[types.TANGO]
            assert registered.__name__ == "TangoConnector"

    def test_registration_needs_no_pytango(self):
        """Registration must not import PyTango.

        The connector imports ``tango`` inside ``connect()``. If that ever
        moved to module scope, registering the built-ins would raise
        ImportError on every machine without a TANGO environment and take the
        whole framework down with it.
        """
        import sys

        with (
            patch.dict(sys.modules, {"tango": None}),
            isolated_connector_registries(clear=True),
        ):
            register_builtin_connectors()

            assert types.TANGO in ConnectorFactory.list_control_systems()


class TestFrameworkRegistryEntries:
    """The registry provider carries a matching entry for discovery/export."""

    def test_registry_lists_the_tango_connector(self):
        from osprey.registry.builtins import FrameworkRegistryProvider

        connectors = FrameworkRegistryProvider().get_registry_config().connectors
        by_name = {c.name: c for c in connectors}

        assert by_name[types.TANGO].connector_type == "control_system"
        assert by_name[types.TANGO].class_name == "TangoConnector"
        assert by_name[types.TANGO].module_path == (
            "osprey.connectors.control_system.tango_connector"
        )


class TestCliSurface:
    """The CLI can select the type, so a registered connector is reachable.

    Registration alone does not make a connector usable: an operator turns one
    on with ``osprey set connector=tango``, which folds the shorthand into
    ``config.control_system.type`` in the deployment's own profile. A type the
    registry knows and the CLI refuses is a connector nobody can select.
    """

    def test_the_shorthand_reads_the_registered_type_list(self):
        """``SET_CONTROL_SYSTEM_TYPES`` is what the shorthand validates
        against, so a connector missing from it cannot be selected however
        well it is registered underneath."""
        assert types.TANGO in types.SET_CONTROL_SYSTEM_TYPES

    def test_set_connector_writes_tango_into_the_profile(self, cli_runner, tmp_path):
        """End to end through the verb: the shorthand lands as the dotted
        config key a build renders from, in the source the facility owns."""
        from osprey.cli.set_cmd import set as set_command

        repo = tmp_path / "tango-deployment"
        repo.mkdir()
        (repo / "profile.yml").write_text(
            "name: TANGO Test\ndata_bundle: hello_world\nprovider: anthropic\n",
            encoding="utf-8",
        )

        result = cli_runner.invoke(
            set_command,
            ["--repo", str(repo), f"connector={types.TANGO}"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        profile = (repo / "profile.yml").read_text(encoding="utf-8")
        assert f"control_system.type: {types.TANGO}" in profile
