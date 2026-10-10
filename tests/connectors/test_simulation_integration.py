"""The mock connectors over a served tree's simulator view.

A setpoint reads its seeded nominal, a write is echoed into the readback its
pair names, unit and description come from the channel record, and a string
channel passes through; the mock archiver serves the same values as history and
refuses an address the view does not hold.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.va_in_process_connector import VAInProcessConnector
from tests.facility.served_tree import in_process_config, served_tree

#: The test rig's channels, in the facility schema's own spelling.
RIG = {
    "T:Q1:CUR:SP": {
        "unit": "A",
        "description": "Test quad current setpoint (nominal 42.0 A)",
        "simulation": {"nominal": 42.0},
    },
    "T:Q1:CUR:RB": {"unit": "A", "description": "Test quad current readback"},
    "T:TRANS": {"unit": "%", "description": "Beam transmission", "simulation": {"nominal": 98.5}},
    "T:MODE": {
        "value_type": "string",
        "description": "Operating mode",
        "simulation": {"nominal": "CW"},
    },
}


def _config_with_writes_enabled(key, default=None):
    """Mock get_config_value that enables writes but returns sane defaults otherwise."""
    if key == "control_system.writes_enabled":
        return True
    return default


@pytest.fixture
def view(tmp_path):
    """A served tree holding every address the cases read or write."""
    return served_tree(
        tmp_path / "served",
        {"MAGNET:CURRENT:SP": "MAGNET:CURRENT:RB", "T:Q1:CUR:SP": "T:Q1:CUR:RB"},
        ["BEAM:CURRENT", "T:MODE", "T:TRANS"],
        channels=RIG,
    )


class TestVAInProcessConnectorSimulation:
    """VAInProcessConnector over the rig's simulator view."""

    @pytest.mark.asyncio
    async def test_read_engine_channel(self, view):
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = VAInProcessConnector()
            await connector.connect(in_process_config(view, response_delay_ms=0))

            result = await connector.read_channel("T:Q1:CUR:SP")
            assert result.value == 42.0
            assert result.metadata.units == "A"
            assert "nominal 42.0 A" in result.metadata.description

            derived = await connector.read_channel("T:TRANS")
            assert derived.value == pytest.approx(98.5)

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_string_channel_passes_through(self, view):
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = VAInProcessConnector()
            await connector.connect(in_process_config(view, response_delay_ms=0))

            result = await connector.read_channel("T:MODE")
            assert result.value == "CW"

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_is_echoed_into_the_paired_readback(self, view):
        connector = VAInProcessConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect(in_process_config(view, response_delay_ms=0))

            await connector.write_channel("MAGNET:CURRENT:SP", 100.0)
            rb = await connector.read_channel("MAGNET:CURRENT:RB")
            assert abs(rb.value - 100.0) < 1.0

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_metadata_from_the_channel_record(self, view):
        with patch("osprey.utils.config.get_config_value", return_value=False):
            connector = VAInProcessConnector()
            await connector.connect(in_process_config(view, response_delay_ms=0))

            metadata = await connector.get_metadata("T:Q1:CUR:SP")
            assert metadata.units == "A"
            assert "nominal 42.0 A" in metadata.description

            await connector.disconnect()


class TestMockArchiverSimulation:
    """MockArchiverConnector over the rig's simulator view."""

    @pytest.mark.asyncio
    async def test_engine_baseline_series(self, view):
        connector = MockArchiverConnector()
        await connector.connect(in_process_config(view))

        df = await connector.get_data(
            channels=["T:Q1:CUR:SP"],
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 1, 0, 0),
        )
        sp = df.loc[df["channel"] == "T:Q1:CUR:SP", "value"]
        # Guard against an empty selection making .all() vacuously True.
        assert len(sp) > 0
        assert (sp == 42.0).all()

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_an_address_outside_the_view_is_refused(self, view):
        connector = MockArchiverConnector()
        await connector.connect(in_process_config(view))

        df = await connector.get_data(
            channels=["T:Q1:CUR:SP"],
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 1, 0, 0),
        )
        sp = df.loc[df["channel"] == "T:Q1:CUR:SP", "value"]
        # Guard against an empty selection making .all() vacuously True.
        assert len(sp) > 0
        assert (sp == 42.0).all()

        with pytest.raises(ValueError, match="UNSERVED:PV is not in build/facility.json"):
            await connector.get_data(
                channels=["T:Q1:CUR:SP", "UNSERVED:PV"],
                start_date=datetime(2024, 1, 1, 0, 0, 0),
                end_date=datetime(2024, 1, 1, 1, 0, 0),
            )

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_mixed_numeric_and_string_channels_both_present(self, view):
        """A single request mixing a numeric channel and a string (enum/status)
        channel returns rows for both — neither is dropped or coerced."""
        connector = MockArchiverConnector()
        await connector.connect(in_process_config(view))

        df = await connector.get_data(
            channels=["T:Q1:CUR:SP", "T:MODE"],
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 0, 10, 0),
        )
        sp = df.loc[df["channel"] == "T:Q1:CUR:SP", "value"]
        mode = df.loc[df["channel"] == "T:MODE", "value"]

        assert len(sp) > 0
        assert len(mode) > 0
        assert (sp == 42.0).all()
        assert (mode == "CW").all()

        await connector.disconnect()
