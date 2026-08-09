"""SC7 acceptance: default-config-check for the virtual_accelerator type.

A freshly scaffolded Control Assistant project must:
  1. Use the Virtual Accelerator control system by default (the VA soft-IOC
     ships and is deployed unconditionally as part of the turn-key Bluesky
     stack, so scan plans drive it end to end out of the box).
  2. Engage the Mock connector when control_system.type is switched to mock
     (real ConnectorFactory resolution using the scaffolded connector.mock
     config block, not just a string check) — the documented fallback for
     environments with no containers to depend on.
  3. Leave the epics block's production values untouched by that switch.

Complements tests/templates/test_preset_va_block.py (which renders the raw
.j2 template in isolation) by exercising the actual `osprey build` scaffolder
and the `osprey config set-control-system` CLI path end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.config_cmd import set_control_system
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


@pytest.fixture
def scaffolded_project(runner: CliRunner, tmp_path: Path) -> Path:
    """Scaffold a fresh Control Assistant project into a tmp dir."""
    result = runner.invoke(
        build,
        [
            "smoke",
            "--preset",
            "control-assistant",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "smoke"
    assert (project_dir / "config.yml").exists()
    return project_dir


def _load_config(project_dir: Path) -> dict:
    return yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))


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

    def test_default_control_system_type_is_virtual_accelerator(self, scaffolded_project: Path):
        config = _load_config(scaffolded_project)
        assert config["control_system"]["type"] == "virtual_accelerator"

    def test_mock_and_virtual_accelerator_and_epics_blocks_all_present(
        self, scaffolded_project: Path
    ):
        """The three-state switch is fully materialized even though only
        'mock' is active — the other two blocks are ready to flip to."""
        connector = _load_config(scaffolded_project)["control_system"]["connector"]
        assert "mock" in connector
        assert "virtual_accelerator" in connector
        assert "epics" in connector


class TestSwitchingToMockEngagesTheConnector:
    """State 2: switching control_system.type engages MockConnector — the
    documented fallback flip for environments with no containers to depend
    on."""

    def test_cli_switch_updates_config_type(self, runner: CliRunner, scaffolded_project: Path):
        result = runner.invoke(set_control_system, ["mock", "--project", str(scaffolded_project)])
        assert result.exit_code == 0, result.output

        config = _load_config(scaffolded_project)
        assert config["control_system"]["type"] == "mock"

    @pytest.mark.asyncio
    async def test_scaffolded_mock_config_block_resolves_to_mock_connector(
        self, runner: CliRunner, scaffolded_project: Path, monkeypatch
    ):
        """The actual connector.mock block from the scaffolded project, fed
        through the real ConnectorFactory, produces a MockConnector instance —
        not just a config string. Unlike the VA/epics connectors, MockConnector
        has no real network I/O in connect(), so no stubbing is needed.

        chdir into the scaffolded project first: the mock block's
        ``simulation_file`` is a path relative to the project root (resolved
        against the ``project_root`` config value in production; this test
        reads config.yml directly rather than going through the app's config
        loader, so CWD is the fallback resolution root instead).
        """
        result = runner.invoke(set_control_system, ["mock", "--project", str(scaffolded_project)])
        assert result.exit_code == 0, result.output

        register_builtin_connectors()
        cs_config = _load_config(scaffolded_project)["control_system"]
        assert cs_config["type"] == "mock"

        monkeypatch.chdir(scaffolded_project)
        connector = await ConnectorFactory.create_control_system_connector(cs_config)
        try:
            assert isinstance(connector, MockConnector)
            assert connector._connected is True
        finally:
            await connector.disconnect()


class TestEpicsBlockRemainsUntouched:
    """State 3: the epics block still holds untouched production values."""

    def test_epics_block_unchanged_before_switch(self, scaffolded_project: Path):
        epics = _load_config(scaffolded_project)["control_system"]["connector"]["epics"]
        assert epics == ORIGINAL_EPICS_BLOCK

    def test_epics_block_unchanged_after_switching_to_mock(
        self, runner: CliRunner, scaffolded_project: Path
    ):
        """Switching control_system.type must not perturb the epics block —
        `set_control_system_type` only ever touches the `type` field."""
        result = runner.invoke(set_control_system, ["mock", "--project", str(scaffolded_project)])
        assert result.exit_code == 0, result.output

        epics = _load_config(scaffolded_project)["control_system"]["connector"]["epics"]
        assert epics == ORIGINAL_EPICS_BLOCK
