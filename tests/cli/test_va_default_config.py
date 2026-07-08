"""SC7 acceptance: default-config-check for the virtual_accelerator type.

A freshly scaffolded Control Assistant project must:
  1. Use the mock control system by default (safe out of the box).
  2. Engage the Virtual Accelerator connector when control_system.type is
     switched to virtual_accelerator (real ConnectorFactory resolution using
     the scaffolded connector.virtual_accelerator config block, not just a
     string check).
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
from osprey.connectors.control_system.va_connector import VirtualAcceleratorConnector
from osprey.connectors.factory import ConnectorFactory, register_builtin_connectors

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
    """Isolate ConnectorFactory global state across tests."""
    ConnectorFactory._control_system_connectors.clear()
    yield
    ConnectorFactory._control_system_connectors.clear()


class TestFreshProjectDefaultsToMock:
    """State 1: a freshly scaffolded project uses mock by default."""

    def test_default_control_system_type_is_mock(self, scaffolded_project: Path):
        config = _load_config(scaffolded_project)
        assert config["control_system"]["type"] == "mock"

    def test_mock_and_virtual_accelerator_and_epics_blocks_all_present(
        self, scaffolded_project: Path
    ):
        """The three-state switch is fully materialized even though only
        'mock' is active — the other two blocks are ready to flip to."""
        connector = _load_config(scaffolded_project)["control_system"]["connector"]
        assert "mock" in connector
        assert "virtual_accelerator" in connector
        assert "epics" in connector


class TestSwitchingToVirtualAcceleratorEngagesTheConnector:
    """State 2: switching control_system.type engages VirtualAcceleratorConnector."""

    def test_cli_switch_updates_config_type(self, runner: CliRunner, scaffolded_project: Path):
        result = runner.invoke(
            set_control_system, ["virtual_accelerator", "--project", str(scaffolded_project)]
        )
        assert result.exit_code == 0, result.output

        config = _load_config(scaffolded_project)
        assert config["control_system"]["type"] == "virtual_accelerator"

    @pytest.mark.asyncio
    async def test_scaffolded_va_config_block_resolves_to_va_connector(
        self, runner: CliRunner, scaffolded_project: Path, monkeypatch
    ):
        """The actual connector.virtual_accelerator block from the scaffolded
        project, fed through the real ConnectorFactory, produces a
        VirtualAcceleratorConnector instance — not just a config string.

        `connect()` is stubbed out here: the scaffolded gateway block points
        at localhost:5064 in CA name-server mode, and with no soft-IOC
        actually running in the test environment, letting the real
        EPICSConnector.connect() (inherited unmodified by
        VirtualAcceleratorConnector) run its CA context/repeater setup can
        block for a long time waiting on network I/O that will never
        resolve. That's an environment property of live Channel Access, not
        something this config-resolution test needs to exercise — the
        gateway shape itself is covered by
        tests/templates/test_preset_va_block.py.
        """
        result = runner.invoke(
            set_control_system, ["virtual_accelerator", "--project", str(scaffolded_project)]
        )
        assert result.exit_code == 0, result.output

        register_builtin_connectors()
        cs_config = _load_config(scaffolded_project)["control_system"]
        assert cs_config["type"] == "virtual_accelerator"

        captured: dict = {}

        async def fake_connect(self, config):
            captured["config"] = config
            self._connected = True

        monkeypatch.setattr(VirtualAcceleratorConnector, "connect", fake_connect)

        connector = await ConnectorFactory.create_control_system_connector(cs_config)
        try:
            assert isinstance(connector, VirtualAcceleratorConnector)
            assert connector._connected is True
            # The exact scaffolded connector.virtual_accelerator block reached connect().
            assert captured["config"] == cs_config["connector"]["virtual_accelerator"]
        finally:
            await connector.disconnect()


class TestEpicsBlockRemainsUntouched:
    """State 3: the epics block still holds untouched production values."""

    def test_epics_block_unchanged_before_switch(self, scaffolded_project: Path):
        epics = _load_config(scaffolded_project)["control_system"]["connector"]["epics"]
        assert epics == ORIGINAL_EPICS_BLOCK

    def test_epics_block_unchanged_after_switching_to_virtual_accelerator(
        self, runner: CliRunner, scaffolded_project: Path
    ):
        """Switching control_system.type must not perturb the epics block —
        `set_control_system_type` only ever touches the `type` field."""
        result = runner.invoke(
            set_control_system, ["virtual_accelerator", "--project", str(scaffolded_project)]
        )
        assert result.exit_code == 0, result.output

        epics = _load_config(scaffolded_project)["control_system"]["connector"]["epics"]
        assert epics == ORIGINAL_EPICS_BLOCK
