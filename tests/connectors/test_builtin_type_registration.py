"""Tests for the ``doocs`` / ``doocs_archiver`` / ``tango`` connector type surface.

These connectors ship in-tree, so each name has to agree across every layer
that speaks it: the type constants, the factory's built-in registration, the
framework registry, and the CLI entry points. The DOOCS connectors were once
reachable only through a dotted class path because no name had been minted for
them; these tests pin the name in every layer.

No driver (``doocs4py``, PyTango) is imported here -- each connector defers that
import to ``connect()``, which is what makes registering them unconditionally
safe on machines without that control system. That property is pinned in a
fresh interpreter by
``test_import_isolation.py::test_builtin_registration_needs_no_driver``.
"""

from dataclasses import dataclass

import pytest
from click.testing import CliRunner

from osprey.connectors import types
from osprey.connectors.factory import (
    ConnectorFactory,
    isolated_connector_registries,
    register_builtin_connectors,
)


@dataclass(frozen=True)
class _Row:
    type_name: str
    value: str
    class_name: str
    module_path: str
    kind: str  # "control_system" or "archiver"


_ROWS = [
    _Row(
        types.DOOCS,
        "doocs",
        "DOOCSConnector",
        "osprey.connectors.control_system.doocs_connector",
        "control_system",
    ),
    _Row(
        types.DOOCS_ARCHIVER,
        "doocs_archiver",
        "DOOCSArchiverConnector",
        "osprey.connectors.archiver.doocs_archiver_connector",
        "archiver",
    ),
    _Row(
        types.TANGO,
        "tango",
        "TangoConnector",
        "osprey.connectors.control_system.tango_connector",
        "control_system",
    ),
]
_CONTROL_SYSTEM_ROWS = [row for row in _ROWS if row.kind == "control_system"]


def _ids(rows):
    return [row.value for row in rows]


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestTypeConstants:
    """``types.py`` is the single source of truth for the name strings."""

    @pytest.mark.parametrize("row", _ROWS, ids=_ids(_ROWS))
    def test_constant_value(self, row):
        assert row.type_name == row.value

    @pytest.mark.parametrize("row", _ROWS, ids=_ids(_ROWS))
    def test_constant_appears_in_cli_choice_list(self, row):
        cli_list = (
            types.CLI_CONTROL_SYSTEM_TYPES
            if row.kind == "control_system"
            else types.CLI_ARCHIVER_TYPES
        )
        assert row.type_name in cli_list

    @pytest.mark.parametrize("row", _CONTROL_SYSTEM_ROWS, ids=_ids(_CONTROL_SYSTEM_ROWS))
    def test_type_is_live_capable(self, row):
        """A deployment's own block of this type is its ``live`` machine.

        ``resolve_target`` refuses to derive ``live`` from simulated or
        stand-in types; these types name real hardware and must resolve as
        written, exactly as ``epics`` does.
        """
        assert types.resolve_target({"type": row.type_name}, types.TARGET_LIVE) == row.type_name


class TestBuiltinRegistration:
    """``register_builtin_connectors()`` mints each name."""

    @pytest.mark.parametrize("row", _ROWS, ids=_ids(_ROWS))
    def test_registers_as_builtin(self, row):
        with isolated_connector_registries(clear=True):
            register_builtin_connectors()

            if row.kind == "control_system":
                assert row.type_name in ConnectorFactory.list_control_systems()
                registered = ConnectorFactory._control_system_connectors[row.type_name]
            else:
                assert row.type_name in ConnectorFactory.list_archivers()
                registered = ConnectorFactory._archiver_connectors[row.type_name]
            assert registered.__name__ == row.class_name


class TestFrameworkRegistryEntries:
    """The registry provider carries a matching entry for discovery/export."""

    @pytest.mark.parametrize("row", _ROWS, ids=_ids(_ROWS))
    def test_registry_lists_the_connector(self, row):
        from osprey.registry.builtins import FrameworkRegistryProvider

        connectors = FrameworkRegistryProvider().get_registry_config().connectors
        by_name = {c.name: c for c in connectors}

        assert by_name[row.type_name].connector_type == row.kind
        assert by_name[row.type_name].class_name == row.class_name
        assert by_name[row.type_name].module_path == row.module_path


class TestCliSurface:
    """The CLI can select the type, so a registered connector is reachable.

    Registration alone does not make a connector usable: an operator turns one
    on with ``osprey set connector=<type>``, which folds the shorthand into
    ``config.control_system.type`` in the deployment's own profile. A type the
    registry knows and the CLI refuses is a connector nobody can select.
    """

    @pytest.mark.parametrize("row", _CONTROL_SYSTEM_ROWS, ids=_ids(_CONTROL_SYSTEM_ROWS))
    def test_the_shorthand_reads_the_registered_type_list(self, row):
        """``SET_CONTROL_SYSTEM_TYPES`` is what the shorthand validates
        against, so a connector missing from it cannot be selected however
        well it is registered underneath."""
        assert row.type_name in types.SET_CONTROL_SYSTEM_TYPES

    @pytest.mark.parametrize("row", _CONTROL_SYSTEM_ROWS, ids=_ids(_CONTROL_SYSTEM_ROWS))
    def test_set_connector_writes_the_type_into_the_profile(self, row, cli_runner, tmp_path):
        """End to end through the verb: the shorthand lands as the dotted
        config key a build renders from, in the source the facility owns."""
        from osprey.cli.set_cmd import set as set_command

        repo = tmp_path / f"{row.value}-deployment"
        repo.mkdir()
        (repo / "profile.yml").write_text(
            f"name: {row.value.upper()} Test\ndata: data\nprovider: anthropic\n",
            encoding="utf-8",
        )

        result = cli_runner.invoke(
            set_command,
            ["--repo", str(repo), f"connector={row.type_name}"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        profile = (repo / "profile.yml").read_text(encoding="utf-8")
        assert f"control_system.type: {row.type_name}" in profile
