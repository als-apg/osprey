"""Tests for QuestDB archiver connector."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from osprey.connectors.archiver.base import ArchiverMetadata
from osprey.connectors.archiver.questdb_archiver_connector import QuestDBArchiverConnector

asyncpg = pytest.importorskip("asyncpg", reason="asyncpg not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(ts: str, pv: str, value: float) -> MagicMock:
    """Simulate an asyncpg Record row."""
    record = MagicMock()
    record.__iter__ = MagicMock(return_value=iter([ts, pv, value]))
    record.__getitem__ = MagicMock(
        side_effect=lambda k: {"ts": ts, "pv_name": pv, "value": value}[k]
    )
    return record


def _make_pool(fetch_return=None, fetchrow_return=None, fetchval_return=1):
    """Build a mock asyncpg pool."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=fetch_return or [])
    mock_conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    mock_conn.fetchval = AsyncMock(return_value=fetchval_return)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_pool = AsyncMock()
    mock_pool.acquire = MagicMock(return_value=mock_ctx)
    mock_pool.close = AsyncMock()
    return mock_pool, mock_conn


def _make_connector_with_pool(pool):
    """Return a QuestDBArchiverConnector with pool injected directly."""
    connector = QuestDBArchiverConnector()
    connector._pool = pool
    connector._connected = True
    return connector


# ---------------------------------------------------------------------------
# Connect / Disconnect
# ---------------------------------------------------------------------------


class TestConnectDisconnectLifecycle:
    @pytest.mark.asyncio
    async def test_connect_missing_host_raises_value_error(self):
        connector = QuestDBArchiverConnector()
        with pytest.raises(ValueError, match="host"):
            await connector.connect({"username": "admin", "password_env": "PW"})

    @pytest.mark.asyncio
    async def test_connect_missing_username_raises_value_error(self):
        connector = QuestDBArchiverConnector()
        with pytest.raises(ValueError, match="username"):
            await connector.connect({"host": "localhost", "password_env": "PW"})

    @pytest.mark.asyncio
    async def test_connect_missing_password_env_raises_value_error(self):
        connector = QuestDBArchiverConnector()
        with pytest.raises(ValueError, match="password_env"):
            await connector.connect({"host": "localhost", "username": "admin"})

    @pytest.mark.asyncio
    async def test_connect_unset_password_env_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("QUESTDB_PW", raising=False)
        connector = QuestDBArchiverConnector()
        with pytest.raises(ValueError, match="not set"):
            await connector.connect(
                {
                    "host": "localhost",
                    "username": "admin",
                    "password_env": "QUESTDB_PW",
                }
            )

    @pytest.mark.asyncio
    async def test_connect_success_sets_connected(self, monkeypatch):
        monkeypatch.setenv("QUESTDB_PW", "secret")
        pool, _ = _make_pool()

        with patch("asyncpg.create_pool", AsyncMock(return_value=pool)):
            connector = QuestDBArchiverConnector()
            await connector.connect(
                {
                    "host": "localhost",
                    "username": "admin",
                    "password_env": "QUESTDB_PW",
                }
            )

        assert connector._connected is True
        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connect_schema_overrides_applied(self, monkeypatch):
        monkeypatch.setenv("QUESTDB_PW", "secret")
        pool, _ = _make_pool()

        with patch("asyncpg.create_pool", AsyncMock(return_value=pool)):
            connector = QuestDBArchiverConnector()
            await connector.connect(
                {
                    "host": "localhost",
                    "username": "admin",
                    "password_env": "QUESTDB_PW",
                    "table": "beam_data",
                    "pv_column": "channel",
                    "value_column": "reading",
                    "ts_column": "timestamp",
                }
            )

        assert connector._table == "beam_data"
        assert connector._pv_col == "channel"
        assert connector._val_col == "reading"
        assert connector._ts_col == "timestamp"
        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_closes_pool(self):
        pool, _ = _make_pool()
        connector = _make_connector_with_pool(pool)
        await connector.disconnect()
        pool.close.assert_awaited_once()
        assert connector._connected is False
        assert connector._pool is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_is_safe(self):
        connector = QuestDBArchiverConnector()
        await connector.disconnect()
        assert connector._connected is False


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------


class TestGetDataMethod:
    @pytest.mark.asyncio
    async def test_not_connected_raises_runtime_error(self):
        connector = QuestDBArchiverConnector()
        with pytest.raises(RuntimeError, match="not connected"):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )

    @pytest.mark.asyncio
    async def test_empty_pv_list_raises_value_error(self):
        pool, _ = _make_pool()
        connector = _make_connector_with_pool(pool)
        with pytest.raises(ValueError, match="pv_list"):
            await connector.get_data(
                pv_list=[],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )

    @pytest.mark.asyncio
    async def test_invalid_time_range_raises_value_error(self):
        pool, _ = _make_pool()
        connector = _make_connector_with_pool(pool)
        with pytest.raises(ValueError, match="start_date"):
            await connector.get_data(
                pv_list=["PV:X"],
                start_date=datetime(2024, 1, 2),
                end_date=datetime(2024, 1, 1),
            )

    @pytest.mark.asyncio
    async def test_returns_dataframe_with_datetime_index(self):
        rows = [
            ("2024-01-01T00:00:00Z", "BEAM:CURRENT", 499.8),
            ("2024-01-01T00:00:01Z", "BEAM:CURRENT", 499.7),
        ]
        pool, conn = _make_pool(fetch_return=rows)
        connector = _make_connector_with_pool(pool)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "BEAM:CURRENT" in df.columns

    @pytest.mark.asyncio
    async def test_correct_values_returned(self):
        rows = [
            ("2024-01-01T00:00:00Z", "PV:X", 1.0),
            ("2024-01-01T00:00:01Z", "PV:X", 2.0),
            ("2024-01-01T00:00:02Z", "PV:X", 3.0),
        ]
        pool, _ = _make_pool(fetch_return=rows)
        connector = _make_connector_with_pool(pool)

        df = await connector.get_data(
            pv_list=["PV:X"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 1),
        )

        assert list(df["PV:X"]) == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_multi_pv_one_column_each(self):
        rows = [
            ("2024-01-01T00:00:00Z", "PV:1", 1.0),
            ("2024-01-01T00:00:00Z", "PV:2", 2.0),
        ]
        pool, _ = _make_pool(fetch_return=rows)
        connector = _make_connector_with_pool(pool)

        df = await connector.get_data(
            pv_list=["PV:1", "PV:2"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 1),
        )

        assert "PV:1" in df.columns
        assert "PV:2" in df.columns

    @pytest.mark.asyncio
    async def test_absent_pv_filled_with_nan(self):
        rows = [("2024-01-01T00:00:00Z", "PV:A", 5.0)]
        pool, _ = _make_pool(fetch_return=rows)
        connector = _make_connector_with_pool(pool)

        df = await connector.get_data(
            pv_list=["PV:A", "PV:MISSING"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 1),
        )

        assert "PV:MISSING" in df.columns
        assert df["PV:MISSING"].isna().all()

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_dataframe(self):
        pool, _ = _make_pool(fetch_return=[])
        connector = _make_connector_with_pool(pool)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self):
        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(10)

        pool, conn = _make_pool()
        conn.fetch = slow_fetch
        connector = _make_connector_with_pool(pool)

        with pytest.raises(TimeoutError):
            await connector.get_data(
                pv_list=["BEAM:CURRENT"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
                timeout=1,
            )


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------


class TestGetMetadataMethod:
    @pytest.mark.asyncio
    async def test_not_connected_raises_runtime_error(self):
        connector = QuestDBArchiverConnector()
        with pytest.raises(RuntimeError, match="not connected"):
            await connector.get_metadata("BEAM:CURRENT")

    @pytest.mark.asyncio
    async def test_empty_pv_name_raises_value_error(self):
        pool, _ = _make_pool()
        connector = _make_connector_with_pool(pool)
        with pytest.raises(ValueError):
            await connector.get_metadata("")

    @pytest.mark.asyncio
    async def test_returns_archiver_metadata(self):
        row = MagicMock()
        row.__getitem__ = MagicMock(
            side_effect=lambda k: {
                "archival_start": "2024-01-01T00:00:00Z",
                "archival_end": "2024-01-02T00:00:00Z",
                "sample_count": 86400,
                "avg_period_ms": 1000,
            }[k]
        )

        pool, _ = _make_pool(fetchrow_return=row)
        connector = _make_connector_with_pool(pool)
        metadata = await connector.get_metadata("BEAM:CURRENT")

        assert isinstance(metadata, ArchiverMetadata)
        assert metadata.pv_name == "BEAM:CURRENT"
        assert metadata.is_archived is True
        assert metadata.sampling_period == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_not_archived_when_no_rows(self):
        row = MagicMock()
        row.__getitem__ = MagicMock(
            side_effect=lambda k: {
                "archival_start": None,
                "archival_end": None,
                "sample_count": 0,
                "avg_period_ms": None,
            }[k]
        )

        pool, _ = _make_pool(fetchrow_return=row)
        connector = _make_connector_with_pool(pool)
        metadata = await connector.get_metadata("NONEXISTENT:PV")

        assert metadata.is_archived is False


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------


class TestCheckAvailability:
    @pytest.mark.asyncio
    async def test_not_connected_raises_runtime_error(self):
        connector = QuestDBArchiverConnector()
        with pytest.raises(RuntimeError, match="not connected"):
            await connector.check_availability(["PV:X"])

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_dict(self):
        pool, _ = _make_pool()
        connector = _make_connector_with_pool(pool)
        result = await connector.check_availability([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_present_pvs_true_absent_false(self):
        rows = [("PV:1",), ("PV:2",)]
        pool, conn = _make_pool(fetch_return=rows)
        connector = _make_connector_with_pool(pool)
        result = await connector.check_availability(["PV:1", "PV:2", "PV:MISSING"])

        assert result["PV:1"] is True
        assert result["PV:2"] is True
        assert result["PV:MISSING"] is False

    @pytest.mark.asyncio
    async def test_single_batched_query(self):
        pool, conn = _make_pool(fetch_return=[])
        connector = _make_connector_with_pool(pool)
        await connector.check_availability(["PV:1", "PV:2", "PV:3"])
        assert conn.fetch.call_count == 1


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    @pytest.fixture(autouse=True)
    def setup_factory(self):
        from osprey.connectors.factory import ConnectorFactory

        ConnectorFactory.register_archiver("questdb_archiver", QuestDBArchiverConnector)
        yield
        ConnectorFactory._archiver_connectors.pop("questdb_archiver", None)

    @pytest.mark.asyncio
    async def test_factory_creates_questdb_archiver_connector(self, monkeypatch):
        monkeypatch.setenv("QUESTDB_PW", "secret")
        pool, _ = _make_pool()

        with patch("asyncpg.create_pool", AsyncMock(return_value=pool)):
            from osprey.connectors.factory import ConnectorFactory

            config = {
                "type": "questdb_archiver",
                "questdb_archiver": {
                    "host": "localhost",
                    "username": "admin",
                    "password_env": "QUESTDB_PW",
                },
            }
            connector = await ConnectorFactory.create_archiver_connector(config)

        assert isinstance(connector, QuestDBArchiverConnector)
        assert connector._connected is True
        await connector.disconnect()


# ---------------------------------------------------------------------------
# Integration test (requires live QuestDB)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestQuestDBIntegration:
    @pytest.mark.asyncio
    async def test_connect_and_query_live_instance(self, monkeypatch):
        """
        Integration test against a real QuestDB instance.

        Start one locally with:
            docker run -p 8812:8812 -p 9000:9000 questdb/questdb

        Set environment variable QUESTDB_PW=quest before running.
        """
        monkeypatch.setenv("QUESTDB_PW", "quest")
        connector = QuestDBArchiverConnector()
        await connector.connect(
            {
                "host": "localhost",
                "port": 8812,
                "username": "admin",
                "password_env": "QUESTDB_PW",
            }
        )
        assert connector._connected is True
        await connector.disconnect()
