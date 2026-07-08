"""Tests for the virtual_accelerator control-system type CLI surface.

Covers the config-selectable third type across the two CLI entry points:
- `osprey config set-control-system virtual_accelerator` (click.Choice picks up
  CLI_CONTROL_SYSTEM_TYPES from connectors/types.py, landed in Task 1.3).
- The interactive control-system menu in cli/project_actions.py, which
  previously hardcoded Mock/EPICS only.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.config_cmd import set_control_system
from osprey.cli.project_actions import handle_set_control_system
from osprey.connectors import types
from osprey.utils.config_writer import get_control_system_type, set_control_system_type


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
        """CLI invocation with 'virtual_accelerator' is not rejected by click."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")

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
        config_file.write_text("control_system:\n  type: mock\n")

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
            types.MOCK_ARCHIVER,
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
