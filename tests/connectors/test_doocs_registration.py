"""Tests for the ``doocs`` / ``doocs_archiver`` connector type surface.

The DOOCS connectors shipped in-tree but were reachable only through a dotted
class path, because no name was ever minted for them. These tests pin the name
across every layer that has to agree on it: the type constants, the factory's
built-in registration, the framework registry, and the two CLI entry points.

``doocs4py`` is never imported here — both connectors defer that import to
``connect()``, which is exactly what makes registering them unconditionally
safe on machines with no DOOCS environment.
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
    """``types.py`` is the single source of truth for the name strings."""

    def test_control_system_constant(self):
        assert types.DOOCS == "doocs"

    def test_archiver_constant(self):
        assert types.DOOCS_ARCHIVER == "doocs_archiver"

    def test_constants_appear_in_cli_choice_lists(self):
        assert types.DOOCS in types.CLI_CONTROL_SYSTEM_TYPES
        assert types.DOOCS_ARCHIVER in types.CLI_ARCHIVER_TYPES


class TestBuiltinRegistration:
    """``register_builtin_connectors()`` mints both names."""

    def test_doocs_registers_as_builtin_control_system(self):
        with isolated_connector_registries(clear=True):
            register_builtin_connectors()

            assert types.DOOCS in ConnectorFactory.list_control_systems()
            registered = ConnectorFactory._control_system_connectors[types.DOOCS]
            assert registered.__name__ == "DOOCSConnector"

    def test_doocs_archiver_registers_as_builtin_archiver(self):
        with isolated_connector_registries(clear=True):
            register_builtin_connectors()

            assert types.DOOCS_ARCHIVER in ConnectorFactory.list_archivers()
            registered = ConnectorFactory._archiver_connectors[types.DOOCS_ARCHIVER]
            assert registered.__name__ == "DOOCSArchiverConnector"

    def test_registration_needs_no_doocs4py(self):
        """Registration must not import doocs4py.

        Both connectors import it inside ``connect()``. If that ever moved to
        module scope, registering the built-ins would raise ImportError on
        every non-DESY machine and take the whole framework down with it.
        """
        import sys

        with (
            patch.dict(sys.modules, {"doocs4py": None}),
            isolated_connector_registries(clear=True),
        ):
            register_builtin_connectors()

            assert types.DOOCS in ConnectorFactory.list_control_systems()


class TestFrameworkRegistryEntries:
    """The registry provider carries matching entries for discovery/export."""

    def test_registry_lists_both_doocs_connectors(self):
        from osprey.registry.builtins import FrameworkRegistryProvider

        connectors = FrameworkRegistryProvider().get_registry_config().connectors
        by_name = {c.name: c for c in connectors}

        assert by_name[types.DOOCS].connector_type == "control_system"
        assert by_name[types.DOOCS].class_name == "DOOCSConnector"
        assert by_name[types.DOOCS_ARCHIVER].connector_type == "archiver"
        assert by_name[types.DOOCS_ARCHIVER].class_name == "DOOCSArchiverConnector"


class TestCliSurface:
    """The CLI can select the type, so a registered connector is reachable.

    Registration alone does not make a connector usable: an operator turns one
    on with ``osprey set connector=doocs``, which folds the shorthand into
    ``config.control_system.type`` in the deployment's own profile. A type the
    registry knows and the CLI refuses is a connector nobody can select.
    """

    def test_the_shorthand_reads_the_registered_type_list(self):
        """``CLI_CONTROL_SYSTEM_TYPES`` is what the shorthand validates against,
        so a connector missing from it cannot be selected however well it is
        registered underneath."""
        from osprey.connectors.types import CLI_CONTROL_SYSTEM_TYPES

        assert types.DOOCS in CLI_CONTROL_SYSTEM_TYPES

    def test_set_connector_writes_doocs_into_the_profile(self, cli_runner, tmp_path):
        """End to end through the verb: the shorthand lands as the dotted config
        key a build renders from, in the source the facility owns."""
        from osprey.cli.set_cmd import set as set_command

        repo = tmp_path / "doocs-deployment"
        repo.mkdir()
        (repo / "profile.yml").write_text(
            "name: DOOCS Test\ndata_bundle: hello_world\nprovider: anthropic\n",
            encoding="utf-8",
        )

        result = cli_runner.invoke(
            set_command,
            ["--repo", str(repo), f"connector={types.DOOCS}"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        profile = (repo / "profile.yml").read_text(encoding="utf-8")
        assert f"control_system.type: {types.DOOCS}" in profile
