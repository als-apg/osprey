"""Tests for mock connector."""

import pytest
from datetime import datetime

from osprey.connectors.control_system.mock_connector import MockConnector
from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector


class TestMockConnector:
    """Test MockConnector functionality."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connector connection and disconnection."""
        connector = MockConnector()
        config = {
            'response_delay_ms': 0,
            'noise_level': 0.01,
            'enable_writes': True
        }

        await connector.connect(config)
        assert connector._connected is True

        await connector.disconnect()
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_read_pv_accepts_any_name(self):
        """Test that mock connector accepts any PV name."""
        connector = MockConnector()
        await connector.connect({'response_delay_ms': 0})

        # Test with arbitrary PV names
        result1 = await connector.read_pv('MADE:UP:CHANNEL')
        assert result1.value is not None
        assert isinstance(result1.value, float)

        result2 = await connector.read_pv('ANY:RANDOM:NAME')
        assert result2.value is not None

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_pv_infers_units(self):
        """Test that connector infers units from PV names."""
        connector = MockConnector()
        await connector.connect({'response_delay_ms': 0})

        # Test beam current units
        beam_result = await connector.read_pv('BEAM:CURRENT')
        assert 'mA' in beam_result.metadata.units or 'A' in beam_result.metadata.units

        # Test voltage units
        voltage_result = await connector.read_pv('MAGNET:VOLTAGE')
        assert 'V' in voltage_result.metadata.units

        # Test pressure units
        pressure_result = await connector.read_pv('VACUUM:PRESSURE')
        assert 'Torr' in pressure_result.metadata.units

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_and_read_maintains_state(self):
        """Test that mock connector maintains state between writes and reads."""
        connector = MockConnector()
        await connector.connect({
            'response_delay_ms': 0,
            'enable_writes': True,
            'noise_level': 0.0  # No noise for exact comparison
        })

        # Write a value
        pv_name = 'TEST:SETPOINT:SP'
        test_value = 123.45
        success = await connector.write_pv(pv_name, test_value)
        assert success is True

        # Read it back
        result = await connector.read_pv(pv_name)
        assert abs(result.value - test_value) < 0.1  # Allow tiny variance

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_creates_readback(self):
        """Test that writing to :SP creates corresponding :RB."""
        connector = MockConnector()
        await connector.connect({
            'response_delay_ms': 0,
            'enable_writes': True,
            'noise_level': 0.001
        })

        # Write to setpoint
        sp_name = 'MAGNET:CURRENT:SP'
        rb_name = 'MAGNET:CURRENT:RB'
        test_value = 100.0

        await connector.write_pv(sp_name, test_value)

        # Check that readback exists and is close
        rb_result = await connector.read_pv(rb_name)
        assert abs(rb_result.value - test_value) < 1.0

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_disabled(self):
        """Test that writes can be disabled."""
        connector = MockConnector()
        await connector.connect({
            'response_delay_ms': 0,
            'enable_writes': False
        })

        success = await connector.write_pv('TEST:PV', 100.0)
        assert success is False

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_multiple_pvs(self):
        """Test reading multiple PVs concurrently."""
        connector = MockConnector()
        await connector.connect({'response_delay_ms': 0})

        pv_names = ['PV:1', 'PV:2', 'PV:3', 'PV:4']
        results = await connector.read_multiple_pvs(pv_names)

        assert len(results) == len(pv_names)
        for pv_name in pv_names:
            assert pv_name in results
            assert results[pv_name].value is not None

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_validate_pv_always_true(self):
        """Test that all PV names are valid in mock mode."""
        connector = MockConnector()
        await connector.connect({'response_delay_ms': 0})

        assert await connector.validate_pv('ANY:PV:NAME') is True
        assert await connector.validate_pv('RANDOM:CHANNEL') is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_metadata(self):
        """Test getting PV metadata."""
        connector = MockConnector()
        await connector.connect({'response_delay_ms': 0})

        metadata = await connector.get_metadata('BEAM:CURRENT')
        assert metadata.units is not None
        assert metadata.description is not None
        assert 'Mock' in metadata.description

        await connector.disconnect()


class TestMockArchiverConnector:
    """Test MockArchiverConnector functionality."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test archiver connection and disconnection."""
        connector = MockArchiverConnector()
        config = {
            'sample_rate_hz': 1.0,
            'noise_level': 0.01
        }

        await connector.connect(config)
        assert connector._connected is True

        await connector.disconnect()
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_get_data_accepts_any_pvs(self):
        """Test that mock archiver accepts any PV names."""
        connector = MockArchiverConnector()
        await connector.connect({'noise_level': 0.01})

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0, 0)
        pv_list = ['FAKE:PV:1', 'RANDOM:PV:2', 'ANY:NAME:3']

        df = await connector.get_data(
            pv_list=pv_list,
            start_date=start_date,
            end_date=end_date
        )

        assert df is not None
        assert len(df) > 0
        assert len(df.columns) == len(pv_list)
        for pv in pv_list:
            assert pv in df.columns

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_returns_dataframe(self):
        """Test that get_data returns proper DataFrame format."""
        connector = MockArchiverConnector()
        await connector.connect({'noise_level': 0.01})

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 0, 10, 0)

        df = await connector.get_data(
            pv_list=['BEAM:CURRENT'],
            start_date=start_date,
            end_date=end_date,
            precision_ms=1000
        )

        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_metadata(self):
        """Test getting archiver metadata."""
        connector = MockArchiverConnector()
        await connector.connect({})

        metadata = await connector.get_metadata('BEAM:CURRENT')
        assert metadata.pv_name == 'BEAM:CURRENT'
        assert metadata.is_archived is True
        assert metadata.archival_start is not None

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_all_true(self):
        """Test that all PVs are available in mock archiver."""
        connector = MockArchiverConnector()
        await connector.connect({})

        pv_names = ['PV:1', 'PV:2', 'PV:3']
        availability = await connector.check_availability(pv_names)

        assert len(availability) == len(pv_names)
        for pv in pv_names:
            assert availability[pv] is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_generated_time_series_has_variation(self):
        """Test that generated time series have realistic variation."""
        connector = MockArchiverConnector()
        await connector.connect({'noise_level': 0.1})

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0, 0)

        df = await connector.get_data(
            pv_list=['BEAM:CURRENT'],
            start_date=start_date,
            end_date=end_date
        )

        # Check that values vary (not all the same)
        values = df['BEAM:CURRENT'].values
        assert len(set(values)) > 1
        assert values.std() > 0

        await connector.disconnect()

