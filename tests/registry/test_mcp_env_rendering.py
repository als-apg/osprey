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

from pathlib import Path

import pytest

from osprey.mcp_env import load_dotenv_from_project
from osprey.registry.mcp import FRAMEWORK_SERVERS, RENDERED_CONFIG_ENV_VALUE, resolve_servers
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
