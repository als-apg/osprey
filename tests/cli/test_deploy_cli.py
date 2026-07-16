"""Tests for the deploy CLI surface: new lifecycle actions and flag validation.

These tests cover the argument-validation layer added for the multi-user
web-terminal lifecycle actions (`decommission`, `prune`, `nuke`, `seed`).
Validation runs BEFORE any project/config resolution or lazy import of the
`osprey.deployment.web_terminals` package, so these tests must pass even
though that package does not exist yet.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.deploy_cmd import deploy


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestDeployCliHelp:
    """Verify --help advertises the new actions and flags."""

    def test_deploy_cli_help_exits_zero(self, cli_runner):
        result = cli_runner.invoke(deploy, ["--help"])
        assert result.exit_code == 0

    def test_deploy_cli_help_lists_new_actions(self, cli_runner):
        result = cli_runner.invoke(deploy, ["--help"])
        assert "decommission" in result.output
        assert "prune" in result.output
        assert "nuke" in result.output
        assert "seed" in result.output

    def test_deploy_cli_help_lists_new_flags(self, cli_runner):
        result = cli_runner.invoke(deploy, ["--help"])
        assert "--archive" in result.output
        assert "--purge" in result.output
        assert "--yes" in result.output
        assert "--dry-run" in result.output

    def test_deploy_cli_help_still_lists_existing_actions(self, cli_runner):
        """Existing actions must remain unaffected by the new surface."""
        result = cli_runner.invoke(deploy, ["--help"])
        for action in ("up", "down", "restart", "status", "build", "clean", "rebuild"):
            assert action in result.output


class TestDeployCliValidation:
    """Argument validation must fire before project/config resolution or import."""

    def test_deploy_cli_decommission_without_user_errors(self, cli_runner):
        result = cli_runner.invoke(deploy, ["decommission"])
        assert result.exit_code != 0
        assert "user" in result.output.lower()

    def test_deploy_cli_seed_without_user_passes_validation(self, cli_runner):
        """'seed' with no USER is valid (reseeds the whole roster): it must not
        be rejected by the require-USER check (unlike 'decommission')."""
        result = cli_runner.invoke(deploy, ["seed"])
        assert "requires a USER argument" not in result.output

    def test_deploy_cli_seed_with_user_passes_validation(self, cli_runner):
        result = cli_runner.invoke(deploy, ["seed", "alice"])
        assert "requires a USER argument" not in result.output
        assert "does not take a USER argument" not in result.output

    def test_deploy_cli_archive_and_purge_mutually_exclusive(self, cli_runner):
        """No lifecycle module or project setup needed: validation fires first."""
        result = cli_runner.invoke(deploy, ["decommission", "alice", "--archive", "--purge"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_deploy_cli_archive_rejected_for_up(self, cli_runner):
        result = cli_runner.invoke(deploy, ["up", "--archive"])
        assert result.exit_code != 0
        assert "archive" in result.output.lower()

    def test_deploy_cli_purge_rejected_for_status(self, cli_runner):
        result = cli_runner.invoke(deploy, ["status", "--purge"])
        assert result.exit_code != 0
        assert "purge" in result.output.lower()

    def test_deploy_cli_archive_allowed_for_prune_validation_passes(self, cli_runner):
        """--archive on 'prune' should get past validation (may fail later on
        project/config resolution, but not on the archive/purge action check)."""
        result = cli_runner.invoke(deploy, ["prune", "--archive"])
        # Should not be rejected for the archive/action-mismatch reason.
        assert "only valid for" not in result.output.lower()

    def test_deploy_cli_invalid_action_still_rejected(self, cli_runner):
        result = cli_runner.invoke(deploy, ["bogus-action"])
        assert result.exit_code == 2

    def test_deploy_cli_dry_run_rejected_for_non_prune(self, cli_runner):
        """--dry-run is prune-only; misapplying it errors rather than silently no-op'ing."""
        result = cli_runner.invoke(deploy, ["up", "--dry-run"])
        assert result.exit_code != 0
        assert "dry-run" in result.output.lower()

    def test_deploy_cli_dry_run_allowed_for_prune(self, cli_runner):
        result = cli_runner.invoke(deploy, ["prune", "--dry-run"])
        assert "only valid for" not in result.output.lower()

    def test_deploy_cli_stray_user_rejected_for_nuke(self, cli_runner):
        """A USER on an action that doesn't consume it is a typo, not a no-op."""
        result = cli_runner.invoke(deploy, ["nuke", "alice"])
        assert result.exit_code != 0
        assert "does not take a user" in result.output.lower()

    def test_deploy_cli_stray_user_rejected_for_up(self, cli_runner):
        result = cli_runner.invoke(deploy, ["up", "alice"])
        assert result.exit_code != 0
        assert "does not take a user" in result.output.lower()


class TestDeployCliExistingActionsUnaffected:
    """Existing actions/options must keep working exactly as before."""

    def test_deploy_cli_up_action_still_dispatches(self, cli_runner, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_up") as mock_deploy_up:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["up", "--config", str(config_file)])

                assert mock_deploy_up.called
                assert result.exit_code == 0

    def test_deploy_cli_no_user_argument_does_not_break_up(self, cli_runner, tmp_path):
        """The new optional USER argument must not interfere with 'up'."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_up") as mock_deploy_up:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["up", "--config", str(config_file)])

                assert result.exit_code == 0
                assert mock_deploy_up.called


class TestDeployCliLifecycleWiring:
    """Verify lazy imports wire new actions to the expected target functions.

    These tests inject fake modules into sys.modules under the exact dotted
    paths used by the lazy imports in deploy_cmd.py, so they pass regardless
    of whether osprey.deployment.web_terminals actually exists yet.
    """

    def test_deploy_cli_decommission_wires_to_lifecycle_module(self, cli_runner, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        mock_decommission = MagicMock()
        fake_pkg = types.ModuleType("osprey.deployment.web_terminals")
        fake_lifecycle = types.ModuleType("osprey.deployment.web_terminals.lifecycle")
        fake_lifecycle.decommission_user = mock_decommission

        with patch.dict(
            sys.modules,
            {
                "osprey.deployment.web_terminals": fake_pkg,
                "osprey.deployment.web_terminals.lifecycle": fake_lifecycle,
            },
        ):
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(
                    deploy,
                    ["decommission", "alice", "--archive", "--config", str(config_file)],
                )

        assert result.exit_code == 0
        assert mock_decommission.called
        call_args, call_kwargs = mock_decommission.call_args
        assert call_args[0] == config_file
        assert call_args[1] == "alice"
        assert call_kwargs["archive"] is True
        assert call_kwargs["purge"] is False
        assert call_kwargs["assume_yes"] is False

    def test_deploy_cli_prune_wires_to_lifecycle_module(self, cli_runner, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        mock_prune = MagicMock()
        fake_pkg = types.ModuleType("osprey.deployment.web_terminals")
        fake_lifecycle = types.ModuleType("osprey.deployment.web_terminals.lifecycle")
        fake_lifecycle.prune_users = mock_prune

        with patch.dict(
            sys.modules,
            {
                "osprey.deployment.web_terminals": fake_pkg,
                "osprey.deployment.web_terminals.lifecycle": fake_lifecycle,
            },
        ):
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(
                    deploy,
                    ["prune", "--dry-run", "--purge", "--yes", "--config", str(config_file)],
                )

        assert result.exit_code == 0
        assert mock_prune.called
        call_args, call_kwargs = mock_prune.call_args
        assert call_args[0] == config_file
        assert call_kwargs["dry_run"] is True
        assert call_kwargs["archive"] is False
        assert call_kwargs["purge"] is True
        assert call_kwargs["assume_yes"] is True

    def test_deploy_cli_nuke_wires_to_lifecycle_module(self, cli_runner, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        mock_nuke = MagicMock()
        fake_pkg = types.ModuleType("osprey.deployment.web_terminals")
        fake_lifecycle = types.ModuleType("osprey.deployment.web_terminals.lifecycle")
        fake_lifecycle.nuke_stack = mock_nuke

        with patch.dict(
            sys.modules,
            {
                "osprey.deployment.web_terminals": fake_pkg,
                "osprey.deployment.web_terminals.lifecycle": fake_lifecycle,
            },
        ):
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["nuke", "--yes", "--config", str(config_file)])

        assert result.exit_code == 0
        assert mock_nuke.called
        call_args, call_kwargs = mock_nuke.call_args
        assert call_args[0] == config_file
        assert call_kwargs["assume_yes"] is True

    def test_deploy_cli_seed_with_user_wires_to_seeding_module(self, cli_runner, tmp_path):
        """'osprey deploy seed alice' forwards user='alice' to seed_web_terminals."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        mock_seed = MagicMock()
        fake_pkg = types.ModuleType("osprey.deployment.web_terminals")
        fake_seeding = types.ModuleType("osprey.deployment.web_terminals.seeding")
        fake_seeding.seed_web_terminals = mock_seed

        with patch.dict(
            sys.modules,
            {
                "osprey.deployment.web_terminals": fake_pkg,
                "osprey.deployment.web_terminals.seeding": fake_seeding,
            },
        ):
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["seed", "alice", "--config", str(config_file)])

        assert result.exit_code == 0
        assert mock_seed.called
        call_args, _call_kwargs = mock_seed.call_args
        assert call_args[0] == config_file
        assert call_args[1] == "alice"

    def test_deploy_cli_seed_without_user_wires_to_seeding_module(self, cli_runner, tmp_path):
        """'osprey deploy seed' (no USER) forwards user=None to seed_web_terminals,
        which reseeds the whole roster."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        mock_seed = MagicMock()
        fake_pkg = types.ModuleType("osprey.deployment.web_terminals")
        fake_seeding = types.ModuleType("osprey.deployment.web_terminals.seeding")
        fake_seeding.seed_web_terminals = mock_seed

        with patch.dict(
            sys.modules,
            {
                "osprey.deployment.web_terminals": fake_pkg,
                "osprey.deployment.web_terminals.seeding": fake_seeding,
            },
        ):
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["seed", "--config", str(config_file)])

        assert result.exit_code == 0
        assert mock_seed.called
        call_args, _call_kwargs = mock_seed.call_args
        assert call_args[0] == config_file
        assert call_args[1] is None
