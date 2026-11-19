"""Tests for cross-layer workspace / config-path resolution helpers.

These functions resolve where a deployment's config and agent-data live. They
read ``OSPREY_CONFIG`` (config location), ``OSPREY_SESSION_ID`` (per-session
data isolation), and the config's ``agent_data.base_dir`` / ``project_root``.

The anchor is the *repo root*, not the directory holding ``config.yml``: the
rendered config lives in the disposable ``build/`` zone, one level below the
repo, so anchoring on its parent would put durable agent data inside the tree
that every ``osprey build`` re-creates. These tests pin that from both sides —
a config in ``build/`` resolves data one level up, while a config sitting at a
directory's root (the shape a container sees) still resolves data beside it.

A loaded config answers the repo root and the paths under it for itself
(``LoadedConfig.repo_root``, ``repo_path``, ``agent_data_root``,
``session_data_root``); ``locate_repo_root`` answers for a process with no
config, and the framework's ``config_paths`` helpers put the two together for
readers that degrade rather than fail.

The autouse ``reset_state_between_tests`` fixture clears ``OSPREY_CONFIG`` /
``CONFIG_FILE`` and the config-cache singleton around every test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.utils.config_paths import (
    loaded_or_empty,
    process_agent_data_root,
    process_repo_root,
    repo_root_of,
    session_agent_data_root,
)
from osprey.utils.workspace import repo_root_for_agent_data
from osprey_connectors.config import load_config, locate_config
from osprey_connectors.testing import clear_config_state, config_of
from osprey_connectors.workspace import (
    DEFAULT_AGENT_DATA_BASE_DIR,
    RENDERED_CONFIG_RELPATH,
    locate_repo_root,
    rendered_config_path,
)


def _write_config(dir_path: Path, body: str) -> Path:
    cfg = dir_path / "config.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def _write_repo(repo: Path, body: str) -> Path:
    """Write *body* as the rendered config of the three-zone repo at *repo*.

    The realistic shape: the config is an output under ``build/``, and the repo
    root above it is what relative paths anchor on.
    """
    build = repo / "build"
    build.mkdir(parents=True, exist_ok=True)
    return _write_config(build, body)


class TestResolveConfigPath:
    def test_uses_osprey_config_env_when_set(self, tmp_path, monkeypatch):
        cfg = tmp_path / "custom.yml"
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert locate_config() == cfg

    def test_prefers_the_render_when_standing_in_a_repo(self, tmp_path, monkeypatch):
        """A repo root with a build in it resolves to the rendered config."""
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        _write_repo(tmp_path, "project_root: /somewhere\n")
        monkeypatch.chdir(tmp_path)
        assert locate_config() == tmp_path / RENDERED_CONFIG_RELPATH

    def test_defaults_to_cwd_config_yml(self, tmp_path, monkeypatch):
        """No render below cwd: the container shape, and the not-found report."""
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert locate_config() == tmp_path / "config.yml"

    def test_expands_shell_variables(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("OSPREY_CONFIG", "$MY_PROJECT_DIR/config.yml")
        assert locate_config() == tmp_path / "config.yml"


class TestRenderedConfigPath:
    def test_lands_in_the_build_zone(self, tmp_path):
        assert rendered_config_path(tmp_path) == tmp_path / "build" / "config.yml"

    def test_accepts_a_string_root(self, tmp_path):
        assert rendered_config_path(str(tmp_path)) == tmp_path / "build" / "config.yml"


class TestRepoRoot:
    def test_a_configured_root_that_exists_wins(self, tmp_path):
        """The container case: built at one path, running at another."""
        deployed = tmp_path / "app" / "demo"
        deployed.mkdir(parents=True)
        cfg = _write_repo(tmp_path / "elsewhere", f"project_root: {deployed}\n")
        assert load_config(cfg).repo_root == deployed
        assert repo_root_of({"project_root": str(deployed)}) == deployed

    def test_ignores_a_configured_root_this_machine_lacks(self, tmp_path):
        """The deployed overlay: a config staged on the host, read in a container.

        The compose generator flattens the host render into
        ``build/services/<svc>/config.yml`` and the service bind-mounts it over
        its own — so ``project_root`` names a HOST directory that the container
        does not have. Anchoring on it would put the agent-data root outside the
        mounted volume; the mount target is filesystem truth.
        """
        cfg = _write_repo(tmp_path, "project_root: /build/host/only/proj\n")
        assert load_config(cfg).repo_root == tmp_path

    def test_expands_home_in_the_configured_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "deployments" / "demo").mkdir(parents=True)
        assert repo_root_of({"project_root": "~/deployments/demo"}) == (
            tmp_path / "deployments" / "demo"
        )

    def test_unwraps_the_build_zone_when_the_key_is_absent(self, tmp_path):
        assert load_config(_write_repo(tmp_path, "{}\n")).repo_root == tmp_path

    def test_config_at_a_directory_root_anchors_there(self, tmp_path):
        """The container layout, whose project directory *is* the render."""
        assert load_config(_write_config(tmp_path, "{}\n")).repo_root == tmp_path

    def test_the_process_config_answers_for_the_process(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "{}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert process_repo_root() == tmp_path


class TestLocateRepoRoot:
    """The repo a process with no loaded config stands in."""

    def test_a_config_file_that_does_not_load_still_names_its_repo(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, ": not yaml [\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert locate_repo_root() == tmp_path
        assert process_repo_root() == tmp_path

    def test_falls_back_to_cwd_without_a_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert locate_repo_root() == Path.cwd()

    def test_walks_up_to_the_repo_from_an_unbuilt_render(self, tmp_path, monkeypatch):
        """With no config yet, ``build/`` is the render zone and never the anchor."""
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        (tmp_path / "profile.yml").write_text("{}\n")
        (tmp_path / "build").mkdir()
        monkeypatch.chdir(tmp_path / "build")
        assert locate_repo_root() == tmp_path

    def test_walks_up_from_a_nested_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        (tmp_path / "profile.yml").write_text("{}\n")
        (tmp_path / "a" / "b").mkdir(parents=True)
        assert locate_repo_root(tmp_path / "a" / "b") == tmp_path

    def test_the_innermost_marker_wins(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        (tmp_path / "profile.yml").write_text("{}\n")
        (tmp_path / "inner" / "build").mkdir(parents=True)
        (tmp_path / "inner" / "profile.yml").write_text("{}\n")
        monkeypatch.chdir(tmp_path / "inner" / "build")
        assert locate_repo_root() == tmp_path / "inner"

    def test_a_root_recorded_for_another_machine_names_nothing_here(self, tmp_path, monkeypatch):
        """A ``project_root`` this machine lacks gives way to the marker, then the cwd.

        A config mapping held without a config file (a ``--runtime-root``
        render read elsewhere) no longer answers its recorded root: it is not
        a directory here, so the repo the process stands in answers.
        """
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "absent" / "config.yml"))
        monkeypatch.chdir(tmp_path)
        assert repo_root_of({"project_root": "/app/demo"}) == tmp_path
        (tmp_path / "profile.yml").write_text("{}\n")
        (tmp_path / "sub").mkdir()
        monkeypatch.chdir(tmp_path / "sub")
        assert repo_root_of({"project_root": "/app/demo"}) == tmp_path

    def test_the_harness_project_dir_is_not_an_input(self, tmp_path, monkeypatch):
        """``CLAUDE_PROJECT_DIR`` is a hook input; the framework anchors on its own cwd."""
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        (tmp_path / "harness").mkdir()
        (tmp_path / "process").mkdir()
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "harness"))
        monkeypatch.chdir(tmp_path / "process")
        assert locate_repo_root() == tmp_path / "process"


class TestLoadedOrEmpty:
    def test_loads_parsed_config(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "project_root: /somewhere\nfoo:\n  bar: 7\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        config = loaded_or_empty()
        assert config["project_root"] == "/somewhere"
        assert config["foo"]["bar"] == 7

    def test_missing_file_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does_not_exist.yml"))
        assert loaded_or_empty() == {}

    def test_resolves_env_var_placeholders(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "system:\n  timezone: ${WS_TZ:-UTC}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        monkeypatch.setenv("WS_TZ", "America/Los_Angeles")
        config = loaded_or_empty()
        assert config["system"]["timezone"] == "America/Los_Angeles"


class TestClearConfigState:
    def test_clears_the_default_and_every_cached_load(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "project_root: /x\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        loaded_or_empty()  # the lazy default loads and caches the located file

        from osprey_connectors import config as config_module

        # Read, not installed: a degrading reader makes nothing the process default.
        assert config_module._installed is None
        assert config_module._last_lazy is not None
        assert config_module._loads
        clear_config_state()
        assert config_module._installed is None
        assert config_module._last_lazy is None
        assert config_module._loads == {}


class TestSessionAgentDataRoot:
    def test_anchors_on_the_repo_root_not_the_render(self, tmp_path, monkeypatch):
        """The property the split turns on: data lands beside ``build/``, not in it."""
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert session_agent_data_root() == (tmp_path / "var" / "agent_data").resolve()

    def test_default_base_dir_is_the_state_zone(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, f"project_root: {tmp_path}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert session_agent_data_root() == (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR).resolve()

    def test_configured_project_root_wins_over_the_config_location(self, tmp_path, monkeypatch):
        """A container config names the runtime root; the build path it came from is stale."""
        runtime_root = tmp_path / "app" / "demo"
        runtime_root.mkdir(parents=True)
        cfg = _write_repo(
            tmp_path, f"project_root: {runtime_root}\nagent_data:\n  base_dir: var/agent_data\n"
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert session_agent_data_root() == (runtime_root / "var" / "agent_data").resolve()

    def test_custom_base_dir(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: custom/data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert session_agent_data_root() == (tmp_path / "custom" / "data").resolve()

    def test_absolute_base_dir_is_left_alone(self, tmp_path, monkeypatch):
        elsewhere = tmp_path / "elsewhere"
        cfg = _write_repo(tmp_path, f"agent_data:\n  base_dir: {elsewhere}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert session_agent_data_root() == elsewhere.resolve()

    def test_defaults_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert session_agent_data_root() == (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR).resolve()

    def test_session_id_appends_isolation_path(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        monkeypatch.setenv("OSPREY_SESSION_ID", "sess-123")
        expected = (tmp_path / "var" / "agent_data").resolve() / "sessions" / "sess-123"
        assert session_agent_data_root() == expected

    def test_the_loaded_config_names_a_session_explicitly(self, tmp_path, monkeypatch):
        """``LoadedConfig.session_data_root`` reads no environment; the session is an argument."""
        cfg = load_config(_write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n"))
        monkeypatch.setenv("OSPREY_SESSION_ID", "ignored")
        assert cfg.session_data_root("sess-9") == cfg.agent_data_root / "sessions" / "sess-9"
        with pytest.raises(ValueError, match="session id"):
            cfg.session_data_root(" ")

    def test_no_session_id_has_no_sessions_segment(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
        assert "sessions" not in session_agent_data_root().parts


class TestSharedAgentDataRoot:
    def test_ignores_session_isolation(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        monkeypatch.setenv("OSPREY_SESSION_ID", "sess-123")
        # Shared root must NOT include the per-session segment.
        assert process_agent_data_root() == (tmp_path / "var" / "agent_data").resolve()
        assert "sessions" not in process_agent_data_root().parts

    def test_anchors_on_the_repo_root_not_the_render(self, tmp_path):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        assert "build" not in load_config(cfg).agent_data_root.parts

    def test_defaults_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert process_agent_data_root() == (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR).resolve()

    def test_a_config_written_in_code_answers_its_own_root(self, tmp_path):
        config = {"project_root": str(tmp_path), "agent_data": {"base_dir": "var/data"}}
        assert config_of(config).agent_data_root == (tmp_path / "var" / "data").resolve()


class TestRepoRootForAgentData:
    """The anchor that turns an agent-data root back into the repo root."""

    @pytest.mark.parametrize(
        ("root", "base_dir"),
        [
            pytest.param("/srv/repo/var/agent_data", "var/agent_data", id="multi-segment-base-dir"),
            pytest.param("/srv/repo/agent_data", "agent_data", id="single-segment-base-dir"),
            # A root that was not produced by this base_dir: the pre-base_dir answer.
            pytest.param("/srv/repo/elsewhere", "var/agent_data", id="tail-does-not-match"),
        ],
    )
    def test_answers_the_repo_root(self, root, base_dir):
        assert repo_root_for_agent_data(Path(root), base_dir) == Path("/srv/repo")

    @pytest.mark.parametrize("base_dir", ["/data/agent", "/data", "/srv/deploy/data/agent"])
    def test_an_absolute_base_dir_answers_instead_of_raising(self, base_dir):
        """An absolute ``agent_data.base_dir`` must not blow up the caller.

        Regression guard: an absolute base_dir's parts begin with the filesystem
        root, so the tail matched the WHOLE of the root's parts and the
        ``parents[len(tail) - 1]`` index ran off the end — an IndexError, which
        is not what the one caller guards for. Every ``save_data`` evaluates
        this (via ``BaseStore.repo_root``) inside a ``try/except ValueError``,
        so the exception escaped as a crash on artifact save; the gallery's
        candidate list evaluates it too.

        The fallback is the parent, and precisely what that means: the parent is
        an ancestor of everything under the root, so the caller's
        ``relative_to`` SUCCEEDS and produces a wrong-but-plausible pointer
        rather than degrading to a bare filename. No repo-relative pointer
        exists for a root outside the repo; this is the committed baseline's
        answer, kept rather than replaced with an invented one.
        """
        root = Path(base_dir)
        assert repo_root_for_agent_data(root, base_dir) == root.parent


class TestRepoPath:
    def test_absolute_path_returned_unchanged(self, tmp_path):
        abs_path = tmp_path / "already" / "absolute"
        cfg = load_config(_write_config(tmp_path, f"project_root: {tmp_path}\n"))
        assert cfg.repo_path(str(abs_path)) == abs_path

    def test_relative_path_resolved_against_the_repo_root(self, tmp_path):
        cfg = load_config(_write_repo(tmp_path, "{}\n"))
        assert cfg.repo_path("data/limits.json") == tmp_path / "data" / "limits.json"

    def test_home_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = load_config(_write_repo(tmp_path / "repo", "{}\n"))
        assert cfg.repo_path("~/x.json") == tmp_path / "x.json"


class TestFrameworkCopiesOfConnectorPrivates:
    """The layout names the framework owns and the connector package spells privately.

    The connector package's resolvers need the repo-root marker and the state
    zone, but those names are the framework's vocabulary, so the package keeps
    its own spelling private and the framework's is the public one. Two
    spellings of one path drift silently; this holds them equal.
    """

    def test_the_repo_root_marker_is_spelled_once(self):
        from osprey.utils import workspace as framework_workspace
        from osprey_connectors import workspace as connector_workspace

        assert framework_workspace.PROFILE_FILENAME == connector_workspace._PROFILE_FILENAME

    def test_the_state_zone_is_spelled_once(self):
        from osprey.utils import workspace as framework_workspace
        from osprey_connectors import workspace as connector_workspace

        assert framework_workspace.STATE_DIR_NAME == connector_workspace._STATE_DIR_NAME
        assert DEFAULT_AGENT_DATA_BASE_DIR.startswith(f"{framework_workspace.STATE_DIR_NAME}/")

    def test_the_runtime_write_keys_are_the_ones_the_config_load_warns_about(self):
        from osprey.utils.workspace import RUNTIME_WRITE_PATH_KEYS
        from osprey_connectors import config as connector_config

        assert connector_config._RUNTIME_WRITE_PATH_KEYS == RUNTIME_WRITE_PATH_KEYS
