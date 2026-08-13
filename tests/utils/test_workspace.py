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

The autouse ``reset_state_between_tests`` fixture clears ``OSPREY_CONFIG`` /
``CONFIG_FILE`` and the config-cache singleton around every test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.utils.workspace import (
    DEFAULT_AGENT_DATA_BASE_DIR,
    RENDERED_CONFIG_RELPATH,
    load_osprey_config,
    rendered_config_path,
    repo_root_for_agent_data,
    reset_config_cache,
    resolve_agent_data_root,
    resolve_config_path,
    resolve_path,
    resolve_project_root,
    resolve_shared_data_root,
    resolve_workspace_root,
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
        assert resolve_config_path() == cfg

    def test_prefers_the_render_when_standing_in_a_repo(self, tmp_path, monkeypatch):
        """A repo root with a build in it resolves to the rendered config."""
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        _write_repo(tmp_path, "project_root: /somewhere\n")
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path() == tmp_path / RENDERED_CONFIG_RELPATH

    def test_defaults_to_cwd_config_yml(self, tmp_path, monkeypatch):
        """No render below cwd: the container shape, and the not-found report."""
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path() == tmp_path / "config.yml"

    def test_expands_shell_variables(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("OSPREY_CONFIG", "$MY_PROJECT_DIR/config.yml")
        assert resolve_config_path() == tmp_path / "config.yml"


class TestRenderedConfigPath:
    def test_lands_in_the_build_zone(self, tmp_path):
        assert rendered_config_path(tmp_path) == tmp_path / "build" / "config.yml"

    def test_accepts_a_string_root(self, tmp_path):
        assert rendered_config_path(str(tmp_path)) == tmp_path / "build" / "config.yml"


class TestResolveProjectRoot:
    def test_configured_key_wins(self, tmp_path, monkeypatch):
        """The container case: built at one path, running at another."""
        deployed = tmp_path / "app" / "demo"
        deployed.mkdir(parents=True)
        cfg = _write_repo(tmp_path / "elsewhere", f"project_root: {deployed}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_project_root({"project_root": str(deployed)}) == deployed

    def test_ignores_a_configured_root_this_machine_lacks(self, tmp_path, monkeypatch):
        """The deployed overlay: a config staged on the host, read in a container.

        The compose generator flattens the host render into
        ``build/services/<svc>/config.yml`` and the service bind-mounts it over
        its own — so ``project_root`` names a HOST directory that the container
        does not have. Anchoring on it would put the agent-data root outside the
        mounted volume; the mount target is filesystem truth.
        """
        cfg = _write_repo(tmp_path, "{}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_project_root({"project_root": "/build/host/only/proj"}) == tmp_path

    def test_keeps_the_configured_key_without_a_config_file(self, tmp_path, monkeypatch):
        """A ``--runtime-root`` render names the path it will run at, elsewhere."""
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "absent" / "config.yml"))
        assert resolve_project_root({"project_root": "/app/demo"}) == Path("/app/demo")

    def test_expands_home_in_the_configured_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "deployments" / "demo").mkdir(parents=True)
        assert resolve_project_root({"project_root": "~/deployments/demo"}) == (
            tmp_path / "deployments" / "demo"
        )

    def test_unwraps_the_build_zone_when_the_key_is_absent(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "{}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_project_root({}) == tmp_path

    def test_config_at_a_directory_root_anchors_there(self, tmp_path, monkeypatch):
        """The container layout, whose project directory *is* the render."""
        cfg = _write_config(tmp_path, "{}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_project_root({}) == tmp_path

    def test_falls_back_to_cwd_without_a_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_project_root({}) == Path.cwd()


class TestLoadOspreyConfig:
    def test_loads_parsed_config(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "project_root: /somewhere\nfoo:\n  bar: 7\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        config = load_osprey_config()
        assert config["project_root"] == "/somewhere"
        assert config["foo"]["bar"] == 7

    def test_missing_file_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does_not_exist.yml"))
        assert load_osprey_config() == {}

    def test_resolves_env_var_placeholders(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "system:\n  timezone: ${WS_TZ:-UTC}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        monkeypatch.setenv("WS_TZ", "America/Los_Angeles")
        config = load_osprey_config()
        assert config["system"]["timezone"] == "America/Los_Angeles"


class TestResetConfigCache:
    def test_clears_singleton_and_cache(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "project_root: /x\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
        load_osprey_config()  # populates default singleton

        from osprey.utils import config as config_module

        assert config_module._default_config is not None
        reset_config_cache()
        assert config_module._default_config is None
        assert config_module._default_configurable is None
        assert config_module._config_cache == {}


class TestResolveAgentDataRoot:
    def test_anchors_on_the_repo_root_not_the_render(self, tmp_path, monkeypatch):
        """The property the split turns on: data lands beside ``build/``, not in it."""
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_agent_data_root() == (tmp_path / "var" / "agent_data").resolve()

    def test_default_base_dir_is_the_state_zone(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, f"project_root: {tmp_path}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_agent_data_root() == (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR).resolve()

    def test_configured_project_root_wins_over_the_config_location(self, tmp_path, monkeypatch):
        """A container config names the runtime root; the build path it came from is stale."""
        runtime_root = tmp_path / "app" / "demo"
        runtime_root.mkdir(parents=True)
        cfg = _write_repo(
            tmp_path, f"project_root: {runtime_root}\nagent_data:\n  base_dir: var/agent_data\n"
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_agent_data_root() == (runtime_root / "var" / "agent_data").resolve()

    def test_custom_base_dir(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: custom/data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_agent_data_root() == (tmp_path / "custom" / "data").resolve()

    def test_absolute_base_dir_is_left_alone(self, tmp_path, monkeypatch):
        elsewhere = tmp_path / "elsewhere"
        cfg = _write_repo(tmp_path, f"agent_data:\n  base_dir: {elsewhere}\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_agent_data_root() == elsewhere.resolve()

    def test_defaults_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_agent_data_root() == (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR).resolve()

    def test_session_id_appends_isolation_path(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        monkeypatch.setenv("OSPREY_SESSION_ID", "sess-123")
        expected = (tmp_path / "var" / "agent_data").resolve() / "sessions" / "sess-123"
        assert resolve_agent_data_root() == expected

    def test_no_session_id_has_no_sessions_segment(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
        assert "sessions" not in resolve_agent_data_root().parts

    def test_workspace_root_alias(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert resolve_workspace_root() == resolve_agent_data_root()


class TestResolveSharedDataRoot:
    def test_ignores_session_isolation(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        monkeypatch.setenv("OSPREY_SESSION_ID", "sess-123")
        # Shared root must NOT include the per-session segment.
        assert resolve_shared_data_root() == (tmp_path / "var" / "agent_data").resolve()
        assert "sessions" not in resolve_shared_data_root().parts

    def test_anchors_on_the_repo_root_not_the_render(self, tmp_path, monkeypatch):
        cfg = _write_repo(tmp_path, "agent_data:\n  base_dir: var/agent_data\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(cfg))
        assert "build" not in resolve_shared_data_root().parts

    def test_defaults_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_shared_data_root() == (tmp_path / DEFAULT_AGENT_DATA_BASE_DIR).resolve()


class TestRepoRootForAgentData:
    """The anchor that turns an agent-data root back into the repo root."""

    def test_strips_a_multi_segment_base_dir(self):
        assert repo_root_for_agent_data(Path("/srv/repo/var/agent_data"), "var/agent_data") == Path(
            "/srv/repo"
        )

    def test_strips_a_single_segment_base_dir(self):
        assert repo_root_for_agent_data(Path("/srv/repo/agent_data"), "agent_data") == Path(
            "/srv/repo"
        )

    def test_falls_back_to_the_parent_when_the_tail_does_not_match(self):
        # A root that was not produced by this base_dir: the pre-base_dir answer.
        assert repo_root_for_agent_data(Path("/srv/repo/elsewhere"), "var/agent_data") == Path(
            "/srv/repo"
        )

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


class TestResolvePath:
    def test_absolute_path_returned_unchanged(self, tmp_path, monkeypatch):
        _write_config(tmp_path, f"project_root: {tmp_path}\n")
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
        abs_path = tmp_path / "already" / "absolute"
        assert resolve_path(str(abs_path)) == abs_path

    def test_relative_path_resolved_against_project_root(self, tmp_path, monkeypatch):
        _write_config(tmp_path, f"project_root: {tmp_path}\n")
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
        assert resolve_path("data/limits.json") == tmp_path / "data" / "limits.json"
