"""Tests for simulation-engine integration in the mock connectors.

Covers both directions of the backward-compatibility contract: without
``simulation_file`` behavior is unchanged, and with it PVs unknown to the
engine fall back to the legacy procedural paths.
"""

import json
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.mock_connector import MockConnector

TEST_MACHINE = {
    "name": "TestRig",
    "description": "Tiny inline test machine",
    "channels": {
        "T:Q1:CUR:SP": {
            "value": 42.0,
            "units": "A",
            "noise": 0.0,
            "description": "Test quad current setpoint (nominal 42.0 A)",
        },
        "T:Q1:CUR:RB": {
            "expr": "ch('T:Q1:CUR:SP')",
            "units": "A",
            "noise": 0.0,
            "description": "Test quad current readback",
        },
        "T:TRANS": {
            "expr": "max(0.0, 98.5 - 0.85 * abs(ch('T:Q1:CUR:SP') - 42.0))",
            "units": "%",
            "noise": 0.0,
            "description": "Beam transmission",
        },
        "T:MODE": {"value": "CW", "description": "Operating mode"},
    },
    "scenarios": {
        "nominal": {"description": "All systems nominal."},
        "quad-drift": {
            "description": "Q1 left at a stale setpoint.",
            "overrides": {"T:Q1:CUR:SP": 28.4},
            "archiver": [
                {
                    "channel": "T:Q1:CUR:SP",
                    "events": [{"shape": "step", "at": 0.35, "to": 28.4}],
                }
            ],
        },
    },
}

QUAD_DRIFT_TRANS = 98.5 - 0.85 * abs(28.4 - 42.0)  # 86.94


def _config_with_writes_enabled(key, default=None):
    """Mock get_config_value that enables writes but returns sane defaults otherwise."""
    if key == "control_system.writes_enabled":
        return True
    return default


@pytest.fixture
def machine_file(tmp_path):
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(TEST_MACHINE))
    return path


class TestMockConnectorSimulation:
    """MockConnector with a simulation_file configured."""

    @pytest.mark.asyncio
    async def test_no_simulation_file_means_no_engine(self):
        """Backward compat: without simulation_file the engine is never loaded."""
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0})

            assert connector._sim_engine is None
            result = await connector.read_channel("T:Q1:CUR:SP")
            # Legacy path: synthetic value, generic mock description
            assert "Mock channel" in result.metadata.description

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_engine_channel(self, machine_file):
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0, "simulation_file": str(machine_file)})

            result = await connector.read_channel("T:Q1:CUR:SP")
            assert result.value == 42.0
            assert result.metadata.units == "A"
            assert "nominal 42.0 A" in result.metadata.description

            derived = await connector.read_channel("T:TRANS")
            assert derived.value == pytest.approx(98.5)

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_string_channel_passes_through(self, machine_file):
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0, "simulation_file": str(machine_file)})

            result = await connector.read_channel("T:MODE")
            assert result.value == "CW"

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_unknown_pv_falls_back_to_legacy(self, machine_file):
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0, "simulation_file": str(machine_file)})

            result = await connector.read_channel("BEAM:CURRENT")
            assert isinstance(result.value, float)
            assert "Mock channel" in result.metadata.description

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_engine_channel_readback_via_expr(self, machine_file):
        """Engine readbacks come from machine-file exprs, not legacy mirroring."""
        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect({"response_delay_ms": 0, "simulation_file": str(machine_file)})

            result = await connector.write_channel(
                "T:Q1:CUR:SP", 30.0, verification_level="readback", tolerance=0.001
            )
            assert result.success is True
            assert result.verification.verified is True

            rb = await connector.read_channel("T:Q1:CUR:RB")
            assert rb.value == 30.0  # exact: expr readback, no legacy offset

            derived = await connector.read_channel("T:TRANS")
            assert derived.value == pytest.approx(98.5 - 0.85 * 12.0)

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_unknown_pv_uses_legacy_mirroring(self, machine_file):
        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect(
                {
                    "response_delay_ms": 0,
                    "noise_level": 0.0,
                    "simulation_file": str(machine_file),
                }
            )

            await connector.write_channel("MAGNET:CURRENT:SP", 100.0)
            rb = await connector.read_channel("MAGNET:CURRENT:RB")
            assert abs(rb.value - 100.0) < 1.0  # legacy :SP -> :RB mirroring

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_scenario_override_visible_through_connector(self, machine_file):
        (machine_file.parent / "active_scenario").write_text("quad-drift\n")
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0, "simulation_file": str(machine_file)})

            result = await connector.read_channel("T:Q1:CUR:SP")
            assert result.value == 28.4
            derived = await connector.read_channel("T:TRANS")
            assert derived.value == pytest.approx(QUAD_DRIFT_TRANS)

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_relative_path_resolved_against_project_root(self, tmp_path):
        sim_dir = tmp_path / "data" / "simulation"
        sim_dir.mkdir(parents=True)
        (sim_dir / "machine.json").write_text(json.dumps(TEST_MACHINE))

        def config_side_effect(key, default=None):
            if key == "project_root":
                return str(tmp_path)
            return default

        with patch("osprey.utils.config.get_config_value", side_effect=config_side_effect):
            connector = MockConnector()
            await connector.connect(
                {"response_delay_ms": 0, "simulation_file": "data/simulation/machine.json"}
            )

            assert connector._sim_engine is not None
            result = await connector.read_channel("T:Q1:CUR:SP")
            assert result.value == 42.0

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_metadata_from_engine(self, machine_file):
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0, "simulation_file": str(machine_file)})

            metadata = await connector.get_metadata("T:Q1:CUR:SP")
            assert metadata.units == "A"
            assert "nominal 42.0 A" in metadata.description

            await connector.disconnect()


class TestMockArchiverSimulation:
    """MockArchiverConnector with a simulation_file configured."""

    @pytest.mark.asyncio
    async def test_no_simulation_file_means_no_engine(self):
        """Backward compat: legacy procedural series without simulation_file."""
        connector = MockArchiverConnector()
        await connector.connect({"noise_level": 0.01})

        assert connector._sim_engine is None
        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 1, 0, 0),
        )
        assert df["BEAM:CURRENT"].mean() == pytest.approx(500.0, rel=0.1)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_engine_baseline_series(self, machine_file):
        connector = MockArchiverConnector()
        await connector.connect({"simulation_file": str(machine_file)})

        df = await connector.get_data(
            pv_list=["T:Q1:CUR:SP"],
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 1, 0, 0),
        )
        assert (df["T:Q1:CUR:SP"] == 42.0).all()

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_scenario_step_and_pointwise_expr(self, machine_file):
        (machine_file.parent / "active_scenario").write_text("quad-drift\n")
        connector = MockArchiverConnector()
        await connector.connect({"simulation_file": str(machine_file)})

        df = await connector.get_data(
            pv_list=["T:Q1:CUR:SP", "T:TRANS"],
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 1, 0, 0),
        )

        sp = df["T:Q1:CUR:SP"].to_numpy()
        trans = df["T:TRANS"].to_numpy()
        t = np.linspace(0, 1, len(sp))

        # Step at t=0.35 to the override-consistent value
        assert (sp[t < 0.35] == 42.0).all()
        assert (sp[t >= 0.35] == 28.4).all()
        # Derived channel shows correlated history (pointwise expr)
        assert trans[t < 0.35].max() == pytest.approx(98.5)
        assert trans[t >= 0.35].max() == pytest.approx(QUAD_DRIFT_TRANS)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_unknown_pv_falls_back_to_legacy(self, machine_file):
        connector = MockArchiverConnector()
        await connector.connect({"noise_level": 0.01, "simulation_file": str(machine_file)})

        df = await connector.get_data(
            pv_list=["T:Q1:CUR:SP", "BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 1, 0, 0),
        )
        assert (df["T:Q1:CUR:SP"] == 42.0).all()
        assert df["BEAM:CURRENT"].mean() == pytest.approx(500.0, rel=0.1)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_string_channel_series(self, machine_file):
        connector = MockArchiverConnector()
        await connector.connect({"simulation_file": str(machine_file)})

        df = await connector.get_data(
            pv_list=["T:MODE"],
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 0, 10, 0),
        )
        assert (df["T:MODE"] == "CW").all()

        await connector.disconnect()
