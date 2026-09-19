"""The environment an MCP server starts with, across the repo/render split.

Two halves of one contract, pinned together because they have to agree:

* what the registry renders into ``.mcp.json`` — ``OSPREY_CONFIG`` and
  ``CONFIG_FILE``, pointing at the rendered config in the ``build/`` zone;
* what :func:`osprey.mcp_env.load_dotenv_from_project` does with that value at
  startup — walk *up* out of the render to the repo root, because there is one
  ``.env`` per deployment and it is durable, while ``build/`` is re-created from
  scratch by every ``osprey build``.

The env vars themselves are deliberately untouched by the lifecycle redesign:
they are the runtime contract every server, service compose file, and the
executor wrapper read. Only the path they carry moved. These tests pin both
facts — that the injection is still there, for every framework server that had
it, and that its value now names the render.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from osprey.mcp_env import load_dotenv_from_project
from osprey.registry.mcp import (
    FRAMEWORK_SERVERS,
    RENDERED_CONFIG_ENV_VALUE,
    ServerDefinition,
    resolve_servers,
)
from osprey.utils.workspace import RENDERED_CONFIG_RELPATH

REPO = "/tmp/test-repo"

#: Servers whose config env is rendered from the repo root. Named rather than
#: derived from the catalog so that dropping the injection from one of them
#: fails here instead of silently shrinking the expectation.
CONFIG_ENV_SERVERS = (
    "controls",
    "python",
    "osprey_workspace",
    "ariel",
    "health",
    "channel-finder",
)


def _base_ctx(**overrides):
    ctx = {
        "project_root": REPO,
        "current_python_env": "/usr/bin/python3",
        "channel_finder_pipeline": "hierarchical",
    }
    ctx.update(overrides)
    return ctx


def _resolved() -> dict[str, dict]:
    return {s["name"]: s for s in resolve_servers({}, _base_ctx())}


# ---------------------------------------------------------------------------
# Rendered .mcp.json env — the FR-10 carve-out
# ---------------------------------------------------------------------------


class TestRenderedConfigEnv:
    def test_value_names_the_render_not_the_repo_root(self) -> None:
        assert RENDERED_CONFIG_ENV_VALUE == f"{{project_root}}/{RENDERED_CONFIG_RELPATH}"

    @pytest.mark.parametrize("name", CONFIG_ENV_SERVERS)
    def test_both_env_vars_survive_and_point_into_build(self, name: str) -> None:
        """The carve-out: the injection stays, only its value moved."""
        env = _resolved()[name]["env"]
        expected = f"{REPO}/build/config.yml"
        assert env["OSPREY_CONFIG"] == expected
        assert env["CONFIG_FILE"] == expected

    def test_no_framework_server_still_points_at_the_repo_root(self) -> None:
        """A config.yml at the repo root is the pre-split shape and is gone."""
        stale = {
            name: value
            for name, server in FRAMEWORK_SERVERS.items()
            for value in server.env.values()
            if value == "{project_root}/config.yml"
        }
        assert stale == {}

    def test_shell_placeholders_are_still_left_for_runtime(self) -> None:
        """Only ``{...}`` is substituted at render time; ``${...}`` is not."""
        assert _resolved()["controls"]["env"]["EPICS_CA_ADDR_LIST"] == "${EPICS_CA_ADDR_LIST:-}"


# ---------------------------------------------------------------------------
# Startup .env discovery — the other end of the same value
# ---------------------------------------------------------------------------


def _repo_with_render(root: Path) -> Path:
    """Build a three-zone repo at *root* and return its rendered config path."""
    build = root / "build"
    build.mkdir(parents=True)
    config = build / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    return config


@pytest.fixture(autouse=True)
def _clear_probe_var(monkeypatch):
    """The key each test writes into ``.env`` and reads back out of the env."""
    monkeypatch.delenv("OSPREY_TEST_DOTENV_KEY", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)


class TestDotenvDiscovery:
    def test_loads_the_repo_root_env_not_one_in_the_render(self, tmp_path, monkeypatch) -> None:
        """The property the split turns on: ``.env`` is durable, ``build/`` is not."""
        config = _repo_with_render(tmp_path)
        (tmp_path / ".env").write_text("OSPREY_TEST_DOTENV_KEY=repo-root\n", encoding="utf-8")
        (tmp_path / "build" / ".env").write_text(
            "OSPREY_TEST_DOTENV_KEY=stale-render\n", encoding="utf-8"
        )
        monkeypatch.setenv("OSPREY_CONFIG", str(config))

        load_dotenv_from_project()

        import os

        assert os.environ["OSPREY_TEST_DOTENV_KEY"] == "repo-root"

    def test_falls_back_to_the_config_directory(self, tmp_path, monkeypatch) -> None:
        """The container layout: the project directory *is* the render."""
        config = tmp_path / "config.yml"
        config.write_text("{}\n", encoding="utf-8")
        (tmp_path / ".env").write_text("OSPREY_TEST_DOTENV_KEY=beside-config\n", encoding="utf-8")
        monkeypatch.setenv("OSPREY_CONFIG", str(config))

        load_dotenv_from_project()

        import os

        assert os.environ["OSPREY_TEST_DOTENV_KEY"] == "beside-config"

    def test_missing_env_file_is_not_an_error(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("OSPREY_CONFIG", str(_repo_with_render(tmp_path)))

        load_dotenv_from_project()  # must not raise

        import os

        assert "OSPREY_TEST_DOTENV_KEY" not in os.environ

    def test_without_osprey_config_it_reads_the_working_directory(
        self, tmp_path, monkeypatch
    ) -> None:
        (tmp_path / ".env").write_text("OSPREY_TEST_DOTENV_KEY=from-cwd\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        load_dotenv_from_project()

        import os

        assert os.environ["OSPREY_TEST_DOTENV_KEY"] == "from-cwd"


# ---------------------------------------------------------------------------
# Rendered .mcp.json headers — the URL-server half of the same contract
# ---------------------------------------------------------------------------


def _render_mcp_json(claude_code_config: dict, **ctx_overrides) -> dict:
    """Render ``claude_code/mcp.json.j2`` and parse it back."""
    from osprey.cli.templates.manager import TemplateManager

    ctx = _base_ctx(**ctx_overrides)
    ctx["servers"] = resolve_servers(claude_code_config, ctx)
    template = TemplateManager().jinja_env.get_template("claude_code/mcp.json.j2")
    return json.loads(template.render(**ctx))


class TestUrlServerHeaders:
    """A URL server carries ``headers``; a stdio server carries ``env``.

    The two are exclusive in the rendered file, which is why they are pinned
    beside each other: an entry that grew both would be telling Claude Code to
    authenticate a remote server with a subprocess environment it never gets.
    """

    def test_framework_url_server_renders_headers_and_no_env(self, monkeypatch) -> None:
        """A framework URL entry renders ``{type, url, headers}`` and nothing else."""
        monkeypatch.setitem(
            FRAMEWORK_SERVERS,
            "test_url_framework",
            ServerDefinition(
                name="test_url_framework",
                module="",
                url="http://127.0.0.1:${OSPREY_WEB_PORT:-}/panel/events/mcp",
                transport="http",
                headers={"Authorization": "Bearer ${OSPREY_PANEL_TOKEN:-}"},
            ),
        )

        entry = _render_mcp_json({})["mcpServers"]["test_url_framework"]

        assert entry == {
            "type": "http",
            "url": "http://127.0.0.1:${OSPREY_WEB_PORT:-}/panel/events/mcp",
            "headers": {"Authorization": "Bearer ${OSPREY_PANEL_TOKEN:-}"},
        }
        assert "env" not in entry

    def test_owner_header_is_stripped_from_a_custom_spec(self, caplog) -> None:
        """The owner header names the acting human — a spec may never mint one."""
        config = {
            "servers": {
                "remote-api": {
                    "url": "http://remote:8001/mcp",
                    "headers": {"X-OSPREY-OWNER": "bob", "X-Trace": "keep-me"},
                }
            }
        }

        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            resolved = {s["name"]: s for s in resolve_servers(config, _base_ctx())}

        assert resolved["remote-api"]["headers"] == {"X-Trace": "keep-me"}
        owner_warnings = [r for r in caplog.records if "X-OSPREY-OWNER" in r.getMessage()]
        assert len(owner_warnings) == 1
        assert "remote-api" in owner_warnings[0].getMessage()
        # The stripped value is never echoed: a header value is a credential
        # far more often than an env value is.
        assert "bob" not in owner_warnings[0].getMessage()

    def test_custom_spec_keeps_its_own_authorization_header(self, caplog) -> None:
        """A custom server authenticates to a service the framework knows nothing of."""
        config = {
            "servers": {
                "remote-api": {
                    "url": "http://remote:8001/mcp",
                    "headers": {"Authorization": "Bearer facility-token"},
                }
            }
        }

        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            entry = _render_mcp_json(config)["mcpServers"]["remote-api"]

        assert entry["headers"] == {"Authorization": "Bearer facility-token"}
        assert [r for r in caplog.records if "remote-api" in r.getMessage()] == []

    def test_headers_resolve_render_time_placeholders_only(self) -> None:
        """``{key}`` is substituted here; ``${VAR:-}`` is left for the CLI."""
        config = {
            "servers": {
                "remote-api": {
                    "url": "http://remote:8001/mcp",
                    "headers": {
                        "X-Root": "{project_root}/build",
                        "Authorization": "Bearer ${OSPREY_PANEL_TOKEN:-}",
                    },
                }
            }
        }

        headers = _render_mcp_json(config)["mcpServers"]["remote-api"]["headers"]

        assert headers["X-Root"] == f"{REPO}/build"
        assert headers["Authorization"] == "Bearer ${OSPREY_PANEL_TOKEN:-}"

    def test_a_command_server_declaring_headers_is_warned_and_renders_none(self, caplog) -> None:
        """Headers are a URL-transport concept — a stdio entry has nowhere to put them."""
        config = {
            "servers": {
                "local-thing": {
                    "command": "/usr/bin/thing",
                    "headers": {"Authorization": "Bearer nope"},
                }
            }
        }

        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            entry = _render_mcp_json(config)["mcpServers"]["local-thing"]

        assert "headers" not in entry
        assert any("local-thing" in r.getMessage() for r in caplog.records)

    def test_malformed_headers_are_ignored_with_a_warning(self, caplog) -> None:
        """A list where a mapping belongs fails closed on the one spec, not the render."""
        config = {
            "servers": {
                "remote-api": {"url": "http://remote:8001/mcp", "headers": ["Authorization"]},
            }
        }

        with caplog.at_level(logging.WARNING, logger="osprey.registry.mcp"):
            entry = _render_mcp_json(config)["mcpServers"]["remote-api"]

        assert entry == {"type": "http", "url": "http://remote:8001/mcp"}
        assert any("remote-api" in r.getMessage() for r in caplog.records)
