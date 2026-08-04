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
    """Both CLI entry points offer the type, so they cannot disagree."""

    def test_set_control_system_choice_includes_doocs(self):
        from osprey.cli.config_cmd import set_control_system

        param = next(p for p in set_control_system.params if p.name == "system_type")
        assert types.DOOCS in param.type.choices

    def test_set_control_system_accepts_doocs(self, cli_runner, tmp_path):
        from osprey.cli.config_cmd import set_control_system

        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")

        with (
            patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve,
            patch("osprey.utils.config_writer.set_control_system_type") as mock_update,
        ):
            mock_resolve.return_value = str(config_file)
            mock_update.return_value = ("new content", "preview")

            result = cli_runner.invoke(set_control_system, [types.DOOCS])

            assert result.exit_code == 0
            assert mock_update.call_args.args[1] == types.DOOCS

    def test_interactive_menu_offers_doocs(self, tmp_path):
        """The wizard's control-system menu lists DOOCS alongside EPICS."""
        from osprey.cli import project_actions

        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")

        with (
            patch.object(project_actions.questionary, "select") as mock_select,
            patch.object(project_actions, "console"),
        ):
            # Backing out of the menu ends the flow before any prompt or write.
            mock_select.return_value.ask.return_value = "back"

            project_actions.handle_set_control_system(tmp_path)

        choices = mock_select.call_args.kwargs["choices"]
        assert types.DOOCS in [c.value for c in choices]

    def test_doocs_control_system_pairs_with_doocs_archiver(self, tmp_path):
        """Picking DOOCS must not offer EPICS Archiver Appliance or MongoDB.

        The DOOCS archiver reads DOOCS *local histories*, so it is the only
        archiver that pairs with a DOOCS control system. The wizard therefore
        selects it directly instead of prompting.
        """
        from osprey.cli import project_actions

        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")

        with (
            patch.object(project_actions.questionary, "select") as mock_select,
            patch.object(project_actions.questionary, "confirm") as mock_confirm,
            patch.object(project_actions, "console"),
            patch("builtins.input", return_value=""),
            patch(
                "osprey.utils.config_writer.set_control_system_type",
                return_value=("new content", "preview"),
            ) as mock_set,
        ):
            mock_select.return_value.ask.return_value = types.DOOCS
            mock_confirm.return_value.ask.return_value = False

            project_actions.handle_set_control_system(tmp_path)

            # Exactly one prompt: the control-system menu. No archiver prompt.
            assert mock_select.call_count == 1
            assert mock_set.call_args.args[2] == types.DOOCS_ARCHIVER
