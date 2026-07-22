"""Tests for the interactive-menu action handlers (project_actions.py).

These handlers shell out to ``osprey deploy`` subprocesses, run health checks,
render config, and rewrite config.yml — all driven by ``questionary`` prompts
and gated on ``input('Press ENTER...')``. Tests mock that terminal boundary
(``questionary``, ``console``, ``input``, ``subprocess``) and assert the
behavioral contracts: which subprocess command would run, when destructive
actions are gated on confirmation, directory save/restore, and dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from osprey.cli import project_actions
from osprey.connectors import types


class _FakeAsk:
    """A questionary prompt object whose ``.ask()`` pops the next queued value."""

    def __init__(self, queue):
        self._queue = queue

    def ask(self):
        return self._queue.pop(0)


def _fake_questionary(select=None, confirm=None, text=None):
    """Build a fake ``questionary`` module with queued select/confirm/text answers."""
    q = MagicMock()
    q.select.side_effect = lambda *a, **k: _FakeAsk(select if select is not None else [])
    q.confirm.side_effect = lambda *a, **k: _FakeAsk(confirm if confirm is not None else [])
    q.text.side_effect = lambda *a, **k: _FakeAsk(text if text is not None else [])
    return q


class TestHandleDeployAction:
    def test_back_returns_without_subprocess(self):
        q = _fake_questionary(select=["back"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("subprocess.run") as run:
                    project_actions.handle_deploy_action()
        run.assert_not_called()

    def test_up_builds_module_invocation_command(self):
        q = _fake_questionary(select=["up"])
        captured = {}

        def _run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return MagicMock(returncode=0)

        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch("subprocess.run", side_effect=_run):
                        project_actions.handle_deploy_action()

        # Re-enters the CLI through the running interpreter, detached, with config.
        assert captured["cmd"] == [
            sys.executable,
            "-m",
            "osprey",
            "deploy",
            "up",
            "-d",
            "--config",
            "config.yml",
        ]
        assert captured["kwargs"]["timeout"] == 300
        assert captured["kwargs"]["env"]["OSPREY_QUIET"] == "1"

    def test_status_action_omits_detached_flag(self):
        q = _fake_questionary(select=["status"])
        captured = {}
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "subprocess.run",
                        side_effect=lambda cmd, **k: (
                            captured.update(cmd=cmd) or MagicMock(returncode=0)
                        ),
                    ):
                        project_actions.handle_deploy_action()
        assert "-d" not in captured["cmd"]
        assert captured["cmd"][4] == "status"

    def test_show_help_loops_back_to_menu(self):
        # First selection asks for help, second backs out.
        q = _fake_questionary(select=["show_help", "back"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("osprey.cli.menu_display.show_deploy_help") as help_fn:
                    with patch("subprocess.run") as run:
                        project_actions.handle_deploy_action()
        help_fn.assert_called_once()
        run.assert_not_called()

    def test_clean_cancelled_when_not_confirmed(self):
        q = _fake_questionary(select=["clean", "back"], confirm=[False])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch("subprocess.run") as run:
                        project_actions.handle_deploy_action()
        # Declined destructive confirm → no subprocess, loops back to menu.
        run.assert_not_called()

    def test_clean_runs_when_confirmed(self):
        q = _fake_questionary(select=["clean"], confirm=[True])
        captured = {}
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "subprocess.run",
                        side_effect=lambda cmd, **k: (
                            captured.update(cmd=cmd) or MagicMock(returncode=0)
                        ),
                    ):
                        project_actions.handle_deploy_action()
        assert captured["cmd"][4] == "clean"

    def test_timeout_is_handled_gracefully(self):
        import subprocess

        # After a timeout the handler loops back to the menu, so queue a
        # follow-up 'back' to exit cleanly.
        q = _fake_questionary(select=["up", "back"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console") as console:
                with patch("builtins.input"):
                    with patch(
                        "subprocess.run",
                        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=300),
                    ):
                        # Must not propagate — reported to the console instead.
                        project_actions.handle_deploy_action()
        assert console.print.called

    def test_down_reports_success_on_zero_exit(self):
        q = _fake_questionary(select=["down"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console") as console:
                with patch("builtins.input"):
                    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                        project_actions.handle_deploy_action()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "stopped" in printed.lower()

    def test_nonzero_exit_reports_warning(self):
        q = _fake_questionary(select=["up"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console") as console:
                with patch("builtins.input"):
                    with patch("subprocess.run", return_value=MagicMock(returncode=3)):
                        project_actions.handle_deploy_action()
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "exited with code 3" in printed

    def test_project_path_changes_and_restores_directory(self, tmp_path):
        _write = tmp_path / "config.yml"
        _write.write_text("project_root: .\n")
        original = Path.cwd()
        q = _fake_questionary(select=["up"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                        project_actions.handle_deploy_action(project_path=tmp_path)
        # cwd restored after the action completes.
        assert Path.cwd() == original


class TestHandleHealthAction:
    def test_runs_checker_and_reports_success(self):
        checker = MagicMock()
        checker.check_all.return_value = True
        with patch("osprey.cli.health_cmd.HealthChecker", return_value=checker):
            with patch.object(project_actions, "console") as console:
                with patch("builtins.input"):
                    project_actions.handle_health_action()
        checker.check_all.assert_called_once()
        assert console.print.called

    def test_checker_exception_is_reported_not_raised(self):
        with patch("osprey.cli.health_cmd.HealthChecker", side_effect=RuntimeError("boom")):
            with patch.object(project_actions, "console") as console:
                with patch("builtins.input"):
                    project_actions.handle_health_action()
        assert console.print.called

    def test_bad_project_path_aborts_before_check(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        with patch.object(project_actions, "console"):
            with patch("builtins.input"):
                with patch("osprey.cli.health_cmd.HealthChecker") as checker:
                    project_actions.handle_health_action(project_path=missing)
        checker.assert_not_called()


class TestHandleExportAction:
    def test_shows_project_config_when_present(self, tmp_path):
        (tmp_path / "config.yml").write_text("project_root: .\nfoo: bar\n")
        with patch.object(project_actions, "console") as console:
            with patch("builtins.input"):
                project_actions.handle_export_action(project_path=tmp_path)
        assert console.print.called

    def test_missing_project_config_reports_error(self, tmp_path):
        with patch.object(project_actions, "console") as console:
            with patch("builtins.input"):
                project_actions.handle_export_action(project_path=tmp_path)
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "No config.yml" in printed

    def test_no_project_path_renders_framework_template(self):
        with patch.object(project_actions, "console") as console:
            with patch("builtins.input"):
                project_actions.handle_export_action()
        assert console.print.called


class TestConfigMenu:
    def test_show_config_menu_returns_none_without_questionary(self):
        with patch.object(project_actions, "questionary", None):
            assert project_actions.show_config_menu() is None

    def test_show_config_menu_returns_selection(self):
        q = _fake_questionary(select=["show"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                assert project_actions.show_config_menu() == "show"

    def test_config_action_back_returns(self):
        with patch.object(project_actions, "show_config_menu", return_value="back"):
            project_actions.handle_config_action()  # returns cleanly

    def test_config_action_show_dispatches_export(self):
        with patch.object(project_actions, "show_config_menu", side_effect=["show", "back"]):
            with patch.object(project_actions, "handle_export_action") as export:
                with patch("builtins.input"):
                    project_actions.handle_config_action()
        export.assert_called_once()

    def test_config_action_dispatches_set_control_system(self):
        with patch.object(
            project_actions, "show_config_menu", side_effect=["set_control_system", "back"]
        ):
            with patch.object(project_actions, "handle_set_control_system") as handler:
                project_actions.handle_config_action()
        handler.assert_called_once()


class TestHandleSetControlSystem:
    def test_missing_config_reports_and_returns(self, tmp_path):
        q = _fake_questionary()
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch("osprey.utils.config_writer.find_config_file", return_value=None):
                        project_actions.handle_set_control_system()
        q.select.assert_not_called()

    def test_back_selection_returns_without_writing(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("control_system:\n  type: mock\n")
        q = _fake_questionary(select=["back"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "osprey.utils.config_writer.get_control_system_type", return_value="mock"
                    ):
                        project_actions.handle_set_control_system(project_path=tmp_path)
        # File untouched.
        assert cfg.read_text() == "control_system:\n  type: mock\n"

    def test_mock_selection_confirmed_writes_config(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("control_system:\n  type: epics\n")
        q = _fake_questionary(select=[types.MOCK], confirm=[True])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "osprey.utils.config_writer.get_control_system_type", return_value="epics"
                    ):
                        with patch(
                            "osprey.utils.config_writer.set_control_system_type",
                            return_value=("control_system:\n  type: mock\n", "PREVIEW"),
                        ):
                            project_actions.handle_set_control_system(project_path=tmp_path)
        assert "mock" in cfg.read_text()

    def test_selection_declined_leaves_config(self, tmp_path):
        cfg = tmp_path / "config.yml"
        original = "control_system:\n  type: epics\n"
        cfg.write_text(original)
        q = _fake_questionary(select=[types.MOCK], confirm=[False])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "osprey.utils.config_writer.get_control_system_type", return_value="epics"
                    ):
                        with patch(
                            "osprey.utils.config_writer.set_control_system_type",
                            return_value=("NEW", "PREVIEW"),
                        ):
                            project_actions.handle_set_control_system(project_path=tmp_path)
        assert cfg.read_text() == original


class TestHandleSetEpicsGateway:
    def test_missing_config_reports_and_returns(self, tmp_path):
        q = _fake_questionary()
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch("osprey.utils.config_writer.find_config_file", return_value=None):
                        project_actions.handle_set_epics_gateway()
        q.select.assert_not_called()

    def test_back_selection_returns_without_writing(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("x: 1\n")
        q = _fake_questionary(select=["back"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "osprey.utils.config_writer.get_facility_from_gateway_config",
                        return_value=None,
                    ):
                        project_actions.handle_set_epics_gateway(project_path=tmp_path)
        assert cfg.read_text() == "x: 1\n"

    def test_custom_facility_gathers_prompts_and_writes(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("control_system:\n  type: epics\n")
        # select 'custom'; 4 text answers; confirms: name-server=False,
        # write=True (mode switch not offered since type is already epics).
        q = _fake_questionary(
            select=["custom"],
            text=["gw.example.edu", "5064", "gw.example.edu", "5084"],
            confirm=[False, True],
        )
        captured = {}

        def _set_gateway(path, facility, custom=None):
            captured["facility"] = facility
            captured["custom"] = custom
            return ("custom-gateway\n", "PREVIEW")

        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "osprey.utils.config_writer.get_facility_from_gateway_config",
                        return_value=None,
                    ):
                        with patch(
                            "osprey.utils.config_writer.set_epics_gateway_config",
                            side_effect=_set_gateway,
                        ):
                            with patch(
                                "osprey.utils.config_writer.get_control_system_type",
                                return_value=types.EPICS,
                            ):
                                project_actions.handle_set_epics_gateway(project_path=tmp_path)

        assert captured["facility"] == "custom"
        # Ports were coerced to ints from the text answers.
        assert captured["custom"]["read_only"]["port"] == 5064
        assert captured["custom"]["write_access"]["port"] == 5084
        assert "custom-gateway" in cfg.read_text()

    def test_custom_facility_aborts_on_empty_read_address(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("control_system:\n  type: epics\n")
        # Empty read address short-circuits the custom flow before any write.
        q = _fake_questionary(select=["custom"], text=[""])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "osprey.utils.config_writer.get_facility_from_gateway_config",
                        return_value=None,
                    ):
                        with patch("osprey.utils.config_writer.set_epics_gateway_config") as set_gw:
                            project_actions.handle_set_epics_gateway(project_path=tmp_path)
        set_gw.assert_not_called()

    def test_preset_facility_confirmed_writes_and_offers_mode_switch(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("control_system:\n  type: mock\n")
        # Select 'aps' preset, confirm gateway write True, decline mode switch False.
        q = _fake_questionary(select=["aps"], confirm=[True, False])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("builtins.input"):
                    with patch(
                        "osprey.utils.config_writer.get_facility_from_gateway_config",
                        return_value=None,
                    ):
                        with patch(
                            "osprey.utils.config_writer.set_epics_gateway_config",
                            return_value=("gateway-config\n", "PREVIEW"),
                        ):
                            with patch(
                                "osprey.utils.config_writer.get_control_system_type",
                                return_value=types.MOCK,
                            ):
                                project_actions.handle_set_epics_gateway(project_path=tmp_path)
        assert "gateway-config" in cfg.read_text()


class TestHandleProjectSelection:
    def test_back_returns_immediately(self, tmp_path):
        (tmp_path / "config.yml").write_text("project_root: .\n")
        q = _fake_questionary(select=["back"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("osprey.cli.menu_display.show_banner"):
                    project_actions.handle_project_selection(tmp_path)

    def test_health_then_back_dispatches_health(self, tmp_path):
        (tmp_path / "config.yml").write_text("project_root: .\n")
        q = _fake_questionary(select=["health", "back"])
        with patch.object(project_actions, "questionary", q):
            with patch.object(project_actions, "console"):
                with patch("osprey.cli.menu_display.show_banner"):
                    with patch.object(project_actions, "handle_health_action") as health:
                        project_actions.handle_project_selection(tmp_path)
        health.assert_called_once_with(project_path=tmp_path)
