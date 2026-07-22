"""Tests for the interactive-menu orchestrator (interactive_menu.py).

Covers project detection, config parsing, nearby-project discovery, menu
construction, the context-aware main menu, mount-safety checks, and the
navigation/entry-point loops. The terminal boundary (``questionary`` prompts,
``console``, ``input``, ``subprocess``) is mocked throughout — no TUI, no
container runtime, no real prompts. cwd-dependent code uses ``monkeypatch.chdir``
so nothing leaks between serial tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osprey.cli import interactive_menu


def _write_config(directory: Path, body: str = "project_root: .\n") -> Path:
    cfg = directory / "config.yml"
    cfg.write_text(body)
    return cfg


class TestIsProjectInitialized:
    def test_true_when_config_present(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert interactive_menu.is_project_initialized() is True

    def test_false_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert interactive_menu.is_project_initialized() is False


class TestGetProjectInfo:
    def test_missing_file_returns_empty(self, tmp_path):
        assert interactive_menu.get_project_info(tmp_path / "nope.yml") == {}

    def test_extracts_provider_and_model(self, tmp_path):
        cfg = _write_config(
            tmp_path,
            "project_root: /root\n"
            "registry_path: reg\n"
            "models:\n"
            "  orchestrator:\n"
            "    provider: anthropic\n"
            "    model_id: claude-x\n",
        )
        info = interactive_menu.get_project_info(cfg)
        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-x"
        assert info["project_root"] == "/root"

    def test_empty_config_returns_empty(self, tmp_path):
        cfg = _write_config(tmp_path, "")
        assert interactive_menu.get_project_info(cfg) == {}

    def test_non_dict_config_returns_empty(self, tmp_path):
        cfg = _write_config(tmp_path, "- just\n- a\n- list\n")
        assert interactive_menu.get_project_info(cfg) == {}

    def test_invalid_yaml_returns_empty(self, tmp_path):
        cfg = _write_config(tmp_path, "invalid: yaml: content:\n")
        with patch.object(interactive_menu, "console"):
            assert interactive_menu.get_project_info(cfg) == {}

    def test_defaults_project_root_to_parent_when_absent(self, tmp_path):
        cfg = _write_config(tmp_path, "registry_path: reg\n")
        info = interactive_menu.get_project_info(cfg)
        assert info["project_root"] == str(tmp_path)

    def test_non_dict_models_section_is_ignored(self, tmp_path):
        cfg = _write_config(tmp_path, "models: not-a-dict\n")
        info = interactive_menu.get_project_info(cfg)
        # No provider/model keys, but no crash and base info present.
        assert "provider" not in info


class TestDiscoverNearbyProjects:
    def test_finds_projects_in_immediate_subdirs(self, tmp_path, monkeypatch):
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        _write_config(tmp_path / "alpha")
        _write_config(tmp_path / "beta")
        (tmp_path / "not_a_project").mkdir()  # no config.yml
        monkeypatch.chdir(tmp_path)

        found = interactive_menu.discover_nearby_projects()
        names = [name for name, _ in found]
        assert names == ["alpha", "beta"]  # sorted, project-only

    def test_ignores_hidden_and_blacklisted_dirs(self, tmp_path, monkeypatch):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "node_modules").mkdir()
        _write_config(tmp_path / ".hidden")
        _write_config(tmp_path / "node_modules")
        monkeypatch.chdir(tmp_path)

        assert interactive_menu.discover_nearby_projects() == []

    def test_respects_max_dirs_limit(self, tmp_path, monkeypatch):
        for i in range(5):
            d = tmp_path / f"p{i}"
            d.mkdir()
            _write_config(d)
        monkeypatch.chdir(tmp_path)

        # With max_dirs=2 only the first two (sorted) subdirs are inspected.
        found = interactive_menu.discover_nearby_projects(max_dirs=2)
        assert len(found) <= 2


class TestGetProjectMenuChoices:
    def test_exit_variant_ends_with_exit(self):
        choices = interactive_menu.get_project_menu_choices(exit_action="exit")
        values = [c.value for c in choices]
        assert values == ["deploy", "health", "config", "registry", "help", "exit"]

    def test_back_variant_ends_with_back(self):
        choices = interactive_menu.get_project_menu_choices(exit_action="back")
        assert choices[-1].value == "back"


class TestShowMainMenu:
    def test_returns_none_without_questionary(self):
        with patch.object(interactive_menu, "questionary", None):
            with patch.object(interactive_menu, "console"):
                assert interactive_menu.show_main_menu() is None

    def test_project_menu_uses_project_choices(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _write_config(tmp_path)

        select = MagicMock()
        select.return_value.ask.return_value = "deploy"
        fake_q = MagicMock(select=select)
        with patch.object(interactive_menu, "questionary", fake_q):
            with patch.object(interactive_menu, "console"):
                with patch.object(interactive_menu, "is_project_initialized", return_value=True):
                    with patch.object(interactive_menu, "get_project_info", return_value={}):
                        result = interactive_menu.show_main_menu()

        assert result == "deploy"
        assert select.called

    def test_no_project_menu_lists_discovered_projects(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        select = MagicMock()
        select.return_value.ask.return_value = "exit"
        fake_q = MagicMock(select=select)
        discovered = [("alpha", tmp_path / "alpha")]
        with patch.object(interactive_menu, "questionary", fake_q):
            with patch.object(interactive_menu, "console"):
                with patch.object(interactive_menu, "is_project_initialized", return_value=False):
                    with patch.object(
                        interactive_menu, "discover_nearby_projects", return_value=discovered
                    ):
                        with patch.object(
                            interactive_menu,
                            "get_project_info",
                            return_value={"provider": "anthropic", "model": "claude-x"},
                        ):
                            result = interactive_menu.show_main_menu()

        assert result == "exit"
        # The discovered project became a select Choice.
        passed_choices = select.call_args.kwargs["choices"]
        select_values = [getattr(c, "value", None) for c in passed_choices]
        assert ("select_project", tmp_path / "alpha") in select_values


class TestCheckDirectoryHasActiveMounts:
    def test_no_runtime_returns_false(self, tmp_path):
        with patch.object(
            interactive_menu, "get_runtime_command", side_effect=RuntimeError("none")
        ):
            has, details = interactive_menu.check_directory_has_active_mounts(tmp_path)
        assert has is False
        assert details == []

    def test_matching_mount_is_reported(self, tmp_path):
        dir_str = str(tmp_path.resolve())
        ps = MagicMock(returncode=0, stdout="cont1\n")
        inspect = MagicMock(
            returncode=0,
            stdout=f'[{{"Source": "{dir_str}/data"}}]',
        )
        with patch.object(interactive_menu, "get_runtime_command", return_value=["docker"]):
            with patch("subprocess.run", side_effect=[ps, inspect]):
                has, details = interactive_menu.check_directory_has_active_mounts(tmp_path)
        assert has is True
        assert any("cont1" in d for d in details)

    def test_no_containers_returns_false(self, tmp_path):
        ps = MagicMock(returncode=0, stdout="\n")
        with patch.object(interactive_menu, "get_runtime_command", return_value=["docker"]):
            with patch("subprocess.run", return_value=ps):
                has, details = interactive_menu.check_directory_has_active_mounts(tmp_path)
        assert has is False


class TestNavigationLoop:
    def test_exit_action_breaks_loop(self):
        with patch.object(interactive_menu, "console"):
            with patch.object(interactive_menu, "show_banner"):
                with patch.object(interactive_menu, "show_main_menu", return_value="exit"):
                    interactive_menu.navigation_loop()  # returns cleanly

    def test_none_action_breaks_loop(self):
        with patch.object(interactive_menu, "console"):
            with patch.object(interactive_menu, "show_banner"):
                with patch.object(interactive_menu, "show_main_menu", return_value=None):
                    interactive_menu.navigation_loop()

    def test_deploy_then_exit_dispatches_handler(self):
        with patch.object(interactive_menu, "console"):
            with patch.object(interactive_menu, "show_banner"):
                with patch.object(
                    interactive_menu, "show_main_menu", side_effect=["deploy", "exit"]
                ):
                    with patch.object(interactive_menu, "handle_deploy_action") as deploy:
                        interactive_menu.navigation_loop()
        deploy.assert_called_once()

    def test_select_project_tuple_dispatches_selection(self, tmp_path):
        action = ("select_project", tmp_path)
        with patch.object(interactive_menu, "console"):
            with patch.object(interactive_menu, "show_banner"):
                with patch.object(interactive_menu, "show_main_menu", side_effect=[action, "exit"]):
                    with patch.object(interactive_menu, "handle_project_selection") as select:
                        interactive_menu.navigation_loop()
        select.assert_called_once_with(tmp_path)

    def test_help_dispatches_contextual_help(self):
        with patch.object(interactive_menu, "console"):
            with patch.object(interactive_menu, "show_banner"):
                with patch.object(interactive_menu, "show_main_menu", side_effect=["help", "exit"]):
                    with patch.object(
                        interactive_menu, "is_project_initialized", return_value=True
                    ):
                        with patch.object(interactive_menu, "handle_help_action") as help_fn:
                            interactive_menu.navigation_loop()
        help_fn.assert_called_once()


class TestLaunchTui:
    def test_missing_questionary_exits_1(self):
        with patch.object(interactive_menu, "questionary", None):
            with patch.object(interactive_menu, "console"):
                with pytest.raises(SystemExit) as exc:
                    interactive_menu.launch_tui()
        assert exc.value.code == 1

    def test_keyboard_interrupt_exits_0(self):
        with patch.object(interactive_menu, "console"):
            with patch.object(interactive_menu, "navigation_loop", side_effect=KeyboardInterrupt):
                with pytest.raises(SystemExit) as exc:
                    interactive_menu.launch_tui()
        assert exc.value.code == 0

    def test_unexpected_error_exits_1(self):
        with patch.object(interactive_menu, "console"):
            with patch.object(interactive_menu, "navigation_loop", side_effect=ValueError("boom")):
                with pytest.raises(SystemExit) as exc:
                    interactive_menu.launch_tui()
        assert exc.value.code == 1
