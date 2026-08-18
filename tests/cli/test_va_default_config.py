"""SC7 acceptance: default-config-check for the virtual_accelerator type.

A freshly scaffolded Control Assistant project must:
  1. Use the Virtual Accelerator control system by default (the VA soft-IOC
     ships and is deployed unconditionally as part of the turn-key Bluesky
     stack, so plans drive it end to end out of the box).
  2. Engage the Mock connector when control_system.type is switched to mock
     (real ConnectorFactory resolution using the scaffolded connector.mock
     config block, not just a string check) — the documented fallback for
     environments with no containers to depend on.
  3. Leave the epics block's production values untouched by that switch.

Complements tests/templates/test_preset_va_block.py (which renders the raw
.j2 template in isolation) by exercising the real lifecycle end to end:
`osprey init` materializes the source zone, `osprey build` renders it, and
`osprey set connector=mock` performs the flip — which, because `osprey set`
writes the profile and never the render, means the flip is only visible in
`build/config.yml` after a second build. That sequencing IS the contract, so
the tests below run it rather than short-circuiting it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey.cli.set_cmd import set as set_cmd
from osprey.connectors.control_system.mock_connector import MockConnector
from osprey.connectors.factory import (
    ConnectorFactory,
    isolated_connector_registries,
    register_builtin_connectors,
)

# The epics block's values as committed prior to the VA feature — the
# untouched ALS production configuration (mirrors
# tests/templates/test_preset_va_block.py's ORIGINAL_EPICS_BLOCK).
ORIGINAL_EPICS_BLOCK = {
    "timeout": 5.0,
    "gateways": {
        "read_only": {
            "address": "cagw-alsdmz.als.lbl.gov",
            "port": 5064,
            "use_name_server": False,
        },
        "write_access": {
            "address": "cagw-alsdmz.als.lbl.gov",
            "port": 5084,
            "use_name_server": False,
        },
    },
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _build(runner: CliRunner, repo: Path) -> None:
    """Render *repo*'s build/ zone, failing the test if it does not."""
    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output


@pytest.fixture
def scaffolded_repo(runner: CliRunner, tmp_path: Path) -> Path:
    """A fresh Control Assistant deployment repo, built once."""
    repo = tmp_path / "smoke"
    result = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert result.exit_code == 0, result.output
    _build(runner, repo)
    assert (repo / "build" / "config.yml").exists()
    return repo


def _switch_to_mock(runner: CliRunner, repo: Path) -> None:
    """The documented flip: edit the profile, then re-render it.

    Both halves, deliberately. `osprey set` writes profile.yml and leaves the
    render alone, so a test that stopped after the set would assert against the
    build the flip has not reached yet.
    """
    result = runner.invoke(set_cmd, ["--repo", str(repo), "connector=mock"])
    assert result.exit_code == 0, result.output
    _build(runner, repo)


def _load_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def clean_connector_factory():
    """Isolate ConnectorFactory global state across tests.

    The registries start empty so ``register_builtin_connectors()`` is
    observed doing the registration; snapshot/restore brackets the clear so
    registrations made elsewhere in the process survive teardown.
    """
    with isolated_connector_registries(clear=True):
        yield


class TestFreshProjectDefaultsToVirtualAccelerator:
    """State 1: a freshly scaffolded project uses the Virtual Accelerator by
    default."""

    def test_default_control_system_type_is_virtual_accelerator(self, scaffolded_repo: Path):
        config = _load_config(scaffolded_repo)
        assert config["control_system"]["type"] == "virtual_accelerator"

    def test_mock_and_virtual_accelerator_and_epics_blocks_all_present(self, scaffolded_repo: Path):
        """The three-state switch is fully materialized even though only
        'mock' is active — the other two blocks are ready to flip to."""
        connector = _load_config(scaffolded_repo)["control_system"]["connector"]
        assert "mock" in connector
        assert "virtual_accelerator" in connector
        assert "epics" in connector


class TestSwitchingToMockEngagesTheConnector:
    """State 2: switching control_system.type engages MockConnector — the
    documented fallback flip for environments with no containers to depend
    on."""

    def test_cli_switch_updates_config_type(self, runner: CliRunner, scaffolded_repo: Path):
        _switch_to_mock(runner, scaffolded_repo)

        config = _load_config(scaffolded_repo)
        assert config["control_system"]["type"] == "mock"

    def test_the_set_alone_does_not_move_the_render(self, runner: CliRunner, scaffolded_repo: Path):
        """`osprey set` edits the source and nothing else: until a build runs,
        the deployment still answers as the Virtual Accelerator. This is the
        property the flip test above depends on, so it is asserted rather than
        assumed."""
        result = runner.invoke(set_cmd, ["--repo", str(scaffolded_repo), "connector=mock"])
        assert result.exit_code == 0, result.output

        assert "control_system.type: mock" in (scaffolded_repo / "profile.yml").read_text(
            encoding="utf-8"
        )
        assert _load_config(scaffolded_repo)["control_system"]["type"] == "virtual_accelerator"

    @pytest.mark.asyncio
    async def test_scaffolded_mock_config_block_resolves_to_mock_connector(
        self, runner: CliRunner, scaffolded_repo: Path, monkeypatch
    ):
        """The actual connector.mock block from the scaffolded project, fed
        through the real ConnectorFactory, produces a MockConnector instance —
        not just a config string. Unlike the VA/epics connectors, MockConnector
        has no real network I/O in connect(), so no stubbing is needed.

        chdir into the repo root first: the mock block's ``simulation_file`` is
        a path relative to the project root (resolved against the
        ``project_root`` config value in production, which for a three-zone
        deployment IS the repo root; this test reads build/config.yml directly
        rather than going through the app's config loader, so CWD is the
        fallback resolution root instead).
        """
        _switch_to_mock(runner, scaffolded_repo)

        register_builtin_connectors()
        cs_config = _load_config(scaffolded_repo)["control_system"]
        assert cs_config["type"] == "mock"

        monkeypatch.chdir(scaffolded_repo)
        connector = await ConnectorFactory.create_control_system_connector(cs_config)
        try:
            assert isinstance(connector, MockConnector)
            assert connector._connected is True
        finally:
            await connector.disconnect()


class TestEpicsBlockRemainsUntouched:
    """State 3: the epics block still holds untouched production values."""

    def test_epics_block_unchanged_before_switch(self, scaffolded_repo: Path):
        epics = _load_config(scaffolded_repo)["control_system"]["connector"]["epics"]
        assert epics == ORIGINAL_EPICS_BLOCK

    def test_epics_block_unchanged_after_switching_to_mock(
        self, runner: CliRunner, scaffolded_repo: Path
    ):
        """Switching the connector must not perturb the epics block — the write
        reaches exactly one key, and the re-render carries the rest through
        unchanged."""
        _switch_to_mock(runner, scaffolded_repo)

        epics = _load_config(scaffolded_repo)["control_system"]["connector"]["epics"]
        assert epics == ORIGINAL_EPICS_BLOCK
