"""Tests for the virtual_accelerator control-system type CLI surface.

Covers the config-selectable third type across the two CLI entry points:
- `osprey config set-control-system virtual_accelerator` (click.Choice picks up
  CLI_CONTROL_SYSTEM_TYPES from connectors/types.py, landed in Task 1.3).
- The interactive control-system menu in cli/project_actions.py, which
  previously hardcoded Mock/EPICS only.

The menu is also where the archiver is chosen, so it is where the honesty rule
has to be reachable as a *choice* rather than only as a later refusal: a wizard
that can author a virtual accelerator with a mock past is a wizard that walks
its user into an error the build, the deploy and the MCP server all raise.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_archiver import (
    CONNECTION_CONFIG_PREFIX,
    VAArchiverConfig,
    va_archiver_config_overrides,
)
from osprey.cli.config_cmd import set_control_system
from osprey.cli.project_actions import (
    build_archiver_choices,
    handle_set_control_system,
    mongodb_archiver_settings,
)
from osprey.connectors import types
from osprey.connectors.archiver.mongodb_archiver_connector import MongoDBArchiverConnector
from osprey.utils.config_writer import get_control_system_type, set_control_system_type

#: A project already reading a real archive, so switching it onto the virtual
#: accelerator is a legal switch rather than the refused pairing.
ARCHIVED_PROJECT = (
    "control_system:\n  type: mock\n\narchiver:\n  type: mongodb_archiver\n"
    "  mongodb_archiver:\n    host: localhost\n"
)

#: The same project with the archiver that synthesizes its history — the one
#: the virtual accelerator may not be paired with.
STORELESS_PROJECT = "control_system:\n  type: mock\n\narchiver:\n  type: mock_archiver\n"


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestSetControlSystemCliAcceptsVirtualAccelerator:
    """`osprey config set-control-system` accepts the virtual_accelerator choice."""

    def test_click_choice_includes_virtual_accelerator(self):
        """The underlying click.Choice for the argument lists virtual_accelerator."""
        system_type_param = next(p for p in set_control_system.params if p.name == "system_type")
        assert "virtual_accelerator" in system_type_param.type.choices

    def test_set_control_system_accepts_virtual_accelerator(self, cli_runner, tmp_path):
        """CLI invocation with 'virtual_accelerator' is not rejected by click.

        The project already reads a real archive — switching one that does not
        is refused, which ``tests/cli/test_config_cmd.py`` covers alongside the
        command's other contracts.
        """
        config_file = tmp_path / "config.yml"
        config_file.write_text(ARCHIVED_PROJECT)

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            with patch("osprey.utils.config_writer.set_control_system_type") as mock_update:
                mock_resolve.return_value = str(config_file)
                mock_update.return_value = ("new content", "preview")

                result = cli_runner.invoke(set_control_system, ["virtual_accelerator"])

                assert result.exit_code == 0
                assert mock_update.called
                # system_type is lower-cased before being passed to the writer
                called_args = mock_update.call_args.args
                assert called_args[1] == "virtual_accelerator"

    def test_set_control_system_rejects_unknown_type(self, cli_runner):
        """Sanity check: click still rejects types that aren't registered."""
        result = cli_runner.invoke(set_control_system, ["not_a_real_type"])

        assert result.exit_code == 2
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()

    def test_set_control_system_virtual_accelerator_writes_expected_config(
        self, cli_runner, tmp_path
    ):
        """End-to-end (no mocking of the writer): config.yml ends up correct."""
        config_file = tmp_path / "config.yml"
        config_file.write_text(ARCHIVED_PROJECT)

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.return_value = str(config_file)

            result = cli_runner.invoke(set_control_system, ["virtual_accelerator"])

            assert result.exit_code == 0

        assert get_control_system_type(config_file) == "virtual_accelerator"


class TestConfigWriterAcceptsVirtualAccelerator:
    """utils/config_writer.set_control_system_type handles the VA type directly."""

    def test_set_control_system_type_to_virtual_accelerator(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text(
            "control_system:\n  type: mock\n\narchiver:\n  type: mock_archiver\n"
        )

        new_content, preview = set_control_system_type(
            config_path, "virtual_accelerator", "epics_archiver"
        )

        assert "virtual_accelerator" in preview
        assert "type: virtual_accelerator" in new_content
        assert "type: epics_archiver" in new_content

        config_path.write_text(new_content)
        assert get_control_system_type(config_path) == "virtual_accelerator"
        assert get_control_system_type(config_path, key="archiver.type") == "epics_archiver"


class TestInteractiveMenuOffersVirtualAccelerator:
    """The interactive control-system menu (project_actions.py) has a VA choice."""

    def test_menu_choices_include_virtual_accelerator(self, tmp_path):
        """Selecting VA in the questionary menu routes through the VA branch."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")

        select_mock = MagicMock()
        # First select() call = control-system type; second = archiver choice.
        select_mock.return_value.ask.side_effect = [
            types.VIRTUAL_ACCELERATOR,
            types.EPICS_ARCHIVER,
        ]

        confirm_mock = MagicMock()
        confirm_mock.return_value.ask.return_value = True

        with patch("osprey.cli.project_actions.questionary.select", select_mock):
            with patch("osprey.cli.project_actions.questionary.confirm", confirm_mock):
                with patch("osprey.cli.project_actions.input", return_value=""):
                    handle_set_control_system(project_path=tmp_path)

        # The first select() call's choices must include a VA option.
        first_call_choices = select_mock.call_args_list[0].kwargs["choices"]
        va_values = [c.value for c in first_call_choices if getattr(c, "value", None) is not None]
        assert types.VIRTUAL_ACCELERATOR in va_values

        # Because VA is routed like EPICS, it should have prompted for an archiver
        # (two select() calls: control-system type, then archiver).
        assert select_mock.call_count == 2

        assert get_control_system_type(config_file) == types.VIRTUAL_ACCELERATOR
        assert get_control_system_type(config_file, key="archiver.type") == types.EPICS_ARCHIVER

    def test_menu_treats_virtual_accelerator_like_epics_for_next_steps(self, tmp_path, capsys):
        """VA selection prints the gateway-configuration next steps, not the mock message."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")

        select_mock = MagicMock()
        select_mock.return_value.ask.side_effect = [
            types.VIRTUAL_ACCELERATOR,
            types.MONGODB_ARCHIVER,
        ]
        confirm_mock = MagicMock()
        confirm_mock.return_value.ask.return_value = True

        with patch("osprey.cli.project_actions.questionary.select", select_mock):
            with patch("osprey.cli.project_actions.questionary.confirm", confirm_mock):
                with patch("osprey.cli.project_actions.input", return_value=""):
                    handle_set_control_system(project_path=tmp_path)

        captured = capsys.readouterr()
        assert "Configure EPICS gateway" in captured.out
        assert "You're now in Mock mode" not in captured.out


def _run_wizard(config_file, answers):
    """Drive the control-system menu with *answers*, one per select() prompt."""
    select_mock = MagicMock()
    select_mock.return_value.ask.side_effect = answers
    confirm_mock = MagicMock()
    confirm_mock.return_value.ask.return_value = True

    with patch("osprey.cli.project_actions.questionary.select", select_mock):
        with patch("osprey.cli.project_actions.questionary.confirm", confirm_mock):
            with patch("osprey.cli.project_actions.input", return_value=""):
                handle_set_control_system(project_path=config_file.parent)

    return select_mock


class TestVirtualAcceleratorNeverOffersAMockPast:
    """The wizard cannot author the pairing the honesty rule refuses."""

    def test_va_choice_list_omits_the_mock_archiver(self):
        """A choice the build would refuse is a trap, not an option."""
        values = [c.value for c in build_archiver_choices(types.VIRTUAL_ACCELERATOR)]

        assert types.MOCK_ARCHIVER not in values
        assert types.MONGODB_ARCHIVER in values

    def test_epics_choice_list_keeps_the_mock_archiver(self):
        """EPICS with no archive attached yet is a real facility's real state."""
        values = [c.value for c in build_archiver_choices(types.EPICS)]

        assert types.MOCK_ARCHIVER in values

    def test_va_menu_asks_from_the_va_choice_list(self, tmp_path):
        """The VA branch passes its own type down, so the list it shows is gated."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")

        select_mock = _run_wizard(config_file, [types.VIRTUAL_ACCELERATOR, types.MONGODB_ARCHIVER])

        archiver_choices = select_mock.call_args_list[1].kwargs["choices"]
        assert types.MOCK_ARCHIVER not in [c.value for c in archiver_choices]

    def test_va_with_a_forced_mock_selection_writes_nothing(self, tmp_path, capsys):
        """Refused past the choice list too, for any path that skips the list."""
        config_file = tmp_path / "config.yml"
        original = "control_system:\n  type: mock\n\narchiver:\n  type: mock_archiver\n"
        config_file.write_text(original)

        _run_wizard(config_file, [types.VIRTUAL_ACCELERATOR, types.MOCK_ARCHIVER])

        assert config_file.read_text() == original
        captured = capsys.readouterr()
        assert "mock past" in captured.out

    def test_va_with_a_cancelled_archiver_prompt_writes_nothing(self, tmp_path):
        """Backing out must not leave the control system switched and the past mock."""
        config_file = tmp_path / "config.yml"
        original = "control_system:\n  type: mock\n\narchiver:\n  type: mock_archiver\n"
        config_file.write_text(original)

        _run_wizard(config_file, [types.VIRTUAL_ACCELERATOR, None])

        assert config_file.read_text() == original


class TestMongoDBSelectionWritesAConstructibleConfig:
    """Selecting MongoDB has to write the store's coordinates, not just its name."""

    @pytest.fixture
    def written_config(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            "control_system:\n  type: mock\n\narchiver:\n  type: mock_archiver\n"
        )

        _run_wizard(config_file, [types.VIRTUAL_ACCELERATOR, types.MONGODB_ARCHIVER])

        return yaml.safe_load(config_file.read_text())

    def test_selection_lands_as_nested_sections(self, written_config):
        """A rendered config.yml is read as nested sections; a flat dotted line
        at the top level configures nothing at all."""
        assert [key for key in written_config if "." in key] == []
        assert written_config["control_system"]["type"] == types.VIRTUAL_ACCELERATOR
        assert written_config["archiver"]["type"] == types.MONGODB_ARCHIVER

    def test_connection_keys_match_the_build_derivation(self, written_config):
        """The wizard and the build describe the same store, from one source."""
        derived = {
            key.removeprefix(f"{CONNECTION_CONFIG_PREFIX}."): value
            for key, value in va_archiver_config_overrides(VAArchiverConfig()).items()
            if key.startswith(f"{CONNECTION_CONFIG_PREFIX}.")
        }

        assert written_config["archiver"]["mongodb_archiver"] == derived
        assert derived["host"] == "localhost"

    def test_connector_accepts_the_written_settings(self, written_config, monkeypatch):
        """The connector validates the config the wizard wrote, unedited.

        Every key MongoDB refuses to default has to be present, and the only
        thing left between this config and a live store is the password `deploy
        up` mints — which is a deployment state, not an authoring error, and is
        the one failure this connector reports as a connection problem.
        """
        settings = written_config["archiver"]["mongodb_archiver"]
        monkeypatch.delenv(settings["password_env"], raising=False)

        connector = MongoDBArchiverConnector()
        with pytest.raises(ConnectionError) as exc:
            asyncio.run(connector.connect(settings))

        assert settings["password_env"] in str(exc.value)

    def test_type_alone_would_not_construct(self, monkeypatch):
        """Contrast: this is what selecting the type without the block writes."""
        monkeypatch.delenv("MONGO_ROOT_PASSWORD", raising=False)

        connector = MongoDBArchiverConnector()
        with pytest.raises(ValueError):
            asyncio.run(connector.connect({}))

    def test_settings_helper_covers_every_connection_key(self):
        """Nothing the connector requires may be left for the user to add."""
        settings = mongodb_archiver_settings()

        leaves = {key.removeprefix(f"{CONNECTION_CONFIG_PREFIX}.") for key in settings}
        assert leaves == {
            "host",
            "port",
            "name",
            "collection",
            "auth",
            "username",
            "password_env",
            "timeout",
        }
        assert all(key.startswith(f"{CONNECTION_CONFIG_PREFIX}.") for key in settings)
